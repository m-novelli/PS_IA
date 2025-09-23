# scripts/build_rank_payload.py
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = BASE_DIR / "data" / "processed" / "dataset_triagem_clean.csv"  # ajuste se necessário
META_PATH = BASE_DIR / "models" / "prod" / "meta.json"
SCRIPTS_DIR = BASE_DIR / "scripts"

POSS_GROUP_VAGA = ["codigo_vaga"]
POSS_GROUP_CAND = ["codigo_applicant", "codigo_prospect", "id_candidato", "cand_id"]

def _pick_first_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def _norm_id(x: Any) -> str:
    s = str(x).strip()
    if s.lower() in {"nan", "none"}:
        return ""
    return re.sub(r"\.0$", "", s)

def _load_schema():
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    schema_in = meta.get("schema_in")
    if schema_in:
        cat = list(schema_in.get("cat", []))
        txt = list(schema_in.get("txt", []))
        # vaga: tudo de txt que começa com perfil_vaga.
        vaga = [c for c in txt if c.startswith("perfil_vaga.")]
        # candidato: resto (cat + txt que não é de vaga)
        cand = [c for c in (cat + txt) if c not in vaga]
        return vaga, cand
    # fallback para meta antigo (features_used)
    feats = meta.get("features_used", {})
    num = list(feats.get("num", []))
    cat = list(feats.get("cat", []))
    txt = list(feats.get("txt", []))
    vaga = [c for c in txt if c.startswith("perfil_vaga.")]
    cand = [c for c in (num + cat + txt) if c not in vaga]
    return vaga, cand

def build_payload_for_job(df: pd.DataFrame, job_id_val, cand_col_hint: Optional[str], top_n: Optional[int]) -> Dict[str, Any]:
    cols = set(df.columns)
    group_vaga_col = _pick_first_col(cols, POSS_GROUP_VAGA)
    if not group_vaga_col:
        raise ValueError(f"Coluna de vaga não encontrada. Tente uma destas: {POSS_GROUP_VAGA}")

    # escolher coluna do candidato
    group_cand_col = cand_col_hint if cand_col_hint in cols else _pick_first_col(cols, POSS_GROUP_CAND)

    job_norm = _norm_id(job_id_val)
    col_norm = df[group_vaga_col].map(_norm_id)
    sub = df[col_norm == job_norm]
    if sub.empty:
        sample = sorted({_norm_id(v) for v in df[group_vaga_col].dropna().unique()})[:15]
        raise ValueError(f"Nenhuma linha para {group_vaga_col}={job_id_val}. Exemplos existentes: {sample}")

    vaga_fields, cand_fields = _load_schema()

    base_row = sub.iloc[0]
    vaga: Dict[str, Any] = {}
    for f in vaga_fields:
        if f in sub.columns:
            v = base_row[f]
            if pd.isna(v):
                continue
            vaga[f] = str(v)

    # dedupe por candidato se possível
    if group_cand_col:
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
                val = row[f]
                if pd.isna(val):
                    continue
                # entrada da API é string para cat/txt; números (se houver no schema antigo) ficam como float
                if isinstance(val, (int, float)) and not f.startswith("perfil_vaga."):
                    cand[f] = float(val)
                else:
                    cand[f] = str(val)
        meta: Dict[str, Any] = {}
        if group_cand_col and group_cand_col in row:
            meta["external_id"] = _norm_id(row[group_cand_col])
        else:
            meta["external_id"] = str(idx)
        candidatos.append({"meta": meta, "candidato": cand})

    return {"vaga": vaga, "candidatos": candidatos}

def main():
    ap = argparse.ArgumentParser(description="Gera payload (vaga+candidatos) e salva compacto em scripts/payload_<JOB>.txt")
    ap.add_argument("--job-id", required=True, help="Valor de codigo_vaga")
    ap.add_argument("--cand-col", default=None, help="Nome da coluna de ID do candidato (ex: codigo_applicant)")
    ap.add_argument("--top-n", type=int, default=None, help="Limita candidatos")
    args = ap.parse_args()

    if not CSV_PATH.exists():
        raise SystemExit(f"CSV não encontrado: {CSV_PATH}")
    if not META_PATH.exists():
        raise SystemExit(f"meta.json não encontrado: {META_PATH}")

    df = pd.read_csv(CSV_PATH)
    payload = build_payload_for_job(df, args.job_id, args.cand_col, args.top_n)

    out_path = SCRIPTS_DIR / f"payload_{args.job_id}.txt"
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    print(f"[OK] salvo: {out_path.resolve()}  (JSON compacto)")

if __name__ == "__main__":
    main()

