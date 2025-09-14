# scripts/build_and_call_rank_and_suggest.py
from __future__ import annotations
import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Config ---
BASE_URL = os.getenv("RANK_API_URL", "http://127.0.0.1:8000")
BASE_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = BASE_DIR / "data" / "processed" / "dataset_triagem_fe.csv"
META_PATH = BASE_DIR / "models" / "prod" / "meta.json"

# Heurísticas de nomes de grupo
POSS_GROUP_VAGA = ["codigo_vaga"]
POSS_GROUP_CAND = ["codigo_applicant", "codigo_prospect", "id_candidato", "cand_id"]

def _pick_first_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def load_schema_from_meta() -> Dict[str, List[str]]:
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    feats = meta.get("features_used", {})
    return {
        "num": list(feats.get("num", [])),
        "cat": list(feats.get("cat", [])),
        "txt": list(feats.get("txt", [])),
    }

def split_fields(schema: Dict[str, List[str]]):
    """Separa campos de vaga vs candidato com base em prefixo 'perfil_vaga.' e listas do meta."""
    txt = schema["txt"]
    vaga_fields = [c for c in txt if c.startswith("perfil_vaga.")]
    candidate_fields = [c for c in (schema["num"] + schema["cat"] + schema["txt"]) if c not in vaga_fields]
    return vaga_fields, candidate_fields

def _norm_id(x: Any) -> str:
    """Normaliza IDs para comparação robusta (remove espaços e sufixo '.0')."""
    s = str(x).strip()
    if s.lower() in {"nan", "none"}:
        return ""
    return re.sub(r"\.0$", "", s)

def build_payload_for_job(df: pd.DataFrame, job_id_val, top_n: Optional[int] = None) -> Dict[str, Any]:
    """Monta payload {'vaga': {...}, 'candidatos': [...]} para uma vaga específica."""
    cols = set(df.columns)
    group_vaga_col = _pick_first_col(cols, POSS_GROUP_VAGA)
    group_cand_col = _pick_first_col(cols, POSS_GROUP_CAND)

    if not group_vaga_col:
        raise ValueError(f"Não encontrei coluna de grupo da vaga. Tente uma destas: {POSS_GROUP_VAGA}")
    if group_vaga_col not in df.columns:
        raise ValueError(f"Coluna {group_vaga_col} não existe no CSV.")

    # filtro com normalização
    job_norm = _norm_id(job_id_val)
    col_norm = df[group_vaga_col].map(_norm_id)
    sub = df[col_norm == job_norm]
    if sub.empty:
        sample = col_norm.dropna().unique().tolist()[:15]
        raise ValueError(
            f"Nenhuma linha para {group_vaga_col}={job_id_val} (normalizado='{job_norm}'). "
            f"Alguns valores encontrados: {sample}"
        )

    schema = load_schema_from_meta()
    vaga_fields, cand_fields = split_fields(schema)

    # dicionário de vaga (primeira linha válida)
    base_row = sub.iloc[0]
    vaga: Dict[str, Any] = {}
    for f in vaga_fields:
        if f in sub.columns:
            val = base_row[f]
            if pd.isna(val):
                continue
            vaga[f] = str(val)

    # candidatos: 1 linha por candidato
    if group_cand_col and group_cand_col in sub.columns:
        sub = sub.sort_values(by=group_cand_col).drop_duplicates(subset=[group_cand_col], keep="first")
    else:
        key = "cv_pt" if "cv_pt" in sub.columns else sub.columns[0]
        sub = sub.drop_duplicates(subset=[key], keep="first")

    if top_n:
        sub = sub.head(int(top_n))

    candidatos: List[Dict[str, Any]] = []
    for idx, row in sub.iterrows():
        cand: Dict[str, Any] = {}
        for f in cand_fields:
            if f in sub.columns:
                v = row[f]
                if pd.isna(v):
                    continue
                if f in schema["num"] and isinstance(v, (int, float)):
                    cand[f] = float(v)
                else:
                    cand[f] = str(v)

        meta: Dict[str, Any] = {}
        if group_cand_col and group_cand_col in row:
            meta["external_id"] = str(row[group_cand_col])  # sempre string
        else:
            meta["external_id"] = str(idx)  # fallback índice como string

        candidatos.append({"meta": meta, "candidato": cand})

    return {"vaga": vaga, "candidatos": candidatos}

def wait_for_api(base_url: str, timeout: int = 15) -> None:
    """Aguarda /health responder 200 OK por até 'timeout' segundos."""
    health = f"{base_url}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(health, timeout=2)
            if r.ok:
                return
        except requests.RequestException:
            time.sleep(1)
    raise SystemExit(f"[ERRO] API indisponível em {health}. Verifique host/porta e se o servidor está rodando.")

def make_session() -> requests.Session:
    s = requests.Session()
    s.mount(
        "http://",
        HTTPAdapter(
            max_retries=Retry(
                total=3,
                backoff_factor=0.5,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=frozenset(["GET", "POST"]),
            )
        ),
    )
    return s

def main(
    job_id_val,
    top_k: int = 5,
    include_questions: bool = True,
    threshold: Optional[float] = None,
    top_n_candidates: Optional[int] = None,
):
    # lê CSV
    df = pd.read_csv(CSV_PATH)
    payload = build_payload_for_job(df, job_id_val, top_n=top_n_candidates)

    # aguarda API pronta
    wait_for_api(BASE_URL)

    # query params
    params = {
        "top_k": top_k,
        "include_questions": str(include_questions).lower(),
    }
    if threshold is not None:
        params["threshold"] = threshold

    url = f"{BASE_URL}/rank-and-suggest"

    session = make_session()
    r = session.post(url, params=params, json=payload, timeout=60)

    print("Status:", r.status_code)
    try:
        print(json.dumps(r.json(), ensure_ascii=False, indent=2))
    except Exception:
        print(r.text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", type=str, default="1939", help="Valor de codigo_vaga a consultar (string).")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--include-questions", action="store_true", default=True)
    parser.add_argument("--no-questions", dest="include_questions", action="store_false")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--top-n-candidates", type=int, default=50)
    args = parser.parse_args()

    main(
        job_id_val=args.job_id,
        top_k=args.top_k,
        include_questions=args.include_questions,
        threshold=args.threshold,
        top_n_candidates=args.top_n_candidates,
    )




