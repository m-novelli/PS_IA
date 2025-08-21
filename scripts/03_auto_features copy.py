# scripts/02_auto_features.py
from pathlib import Path
import re, json
import pandas as pd
import numpy as np

# ===================== Path resolving =====================
BASE_DIR = Path(__file__).resolve().parents[1]

# Localiza o CSV de entrada (interim > processed > busca recursiva)
CSV_CANDIDATES = [
    BASE_DIR / "data" / "interim" / "dataset_triagem_clean.csv",
    BASE_DIR / "data" / "processed" / "dataset_triagem_clean.csv",
]
CSV = next((p for p in CSV_CANDIDATES if p.exists()), None)
if CSV is None:
    matches = list((BASE_DIR / "data").rglob("dataset_triagem_clean.csv"))
    if matches:
        CSV = matches[0]
if CSV is None:
    raise FileNotFoundError("dataset_triagem_clean.csv não encontrado em data/interim ou data/processed.")

OUT_CFG = BASE_DIR / "configs" / "triagem_features.json"
OUT_CFG.parent.mkdir(parents=True, exist_ok=True)
OUT_SUM = CSV.parent / "columns_summary.csv"

# ===================== Parâmetros =====================
CATEGORICAL_MAX_UNIQUE = 50       # até 50 valores únicos -> categórica
MAX_MISSING_RATIO = 0.98          # > 98% ausente -> descarta
# Nunca entram como feature (alvo/IDs/status)
EXCLUDE = {
    "y", "codigo_vaga", "codigo_applicant",
    "status_simplificado", "status_candidato",
    "situacao_candidato", "situacao"
}
# Nomes que indicam resultado/etapa -> potencial leak
LEAKAGE_NAME_PATTERNS = re.compile(
    r"(status|situac|situacao|resultado|contratad|aprovad|reprovad|negad|proposta|documenta)",
    re.IGNORECASE
)
# Nomes que indicam TEXTO longo
TEXT_PATTERNS = re.compile(
    r"(cv|descricao|descrição|titulo|título|requisit|competenc|atividade|responsabil|resumo|perfil|skill|texto|observac|coment)",
    re.IGNORECASE
)

# Overrides opcionais (deixe vazio se não quiser forçar nada)
FORCE_TEXT: set[str] = set()
FORCE_CAT:  set[str] = set()
FORCE_NUM:  set[str] = set()

def is_constant_or_empty(s: pd.Series) -> bool:
    nunique = s.nunique(dropna=True)
    if nunique <= 1:
        return True
    miss = s.isna().mean()
    return miss > MAX_MISSING_RATIO

def main():
    df = pd.read_csv(CSV)
    print("CSV:", CSV)
    print("Shape:", df.shape)

    # Resumo para documentação
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append({
            "coluna": col,
            "dtype": str(s.dtype),
            "n_unique": int(s.nunique(dropna=True)),
            "missing_ratio": float(s.isna().mean()),
            "exemplos": ", ".join(map(str, s.dropna().unique()[:5]))
        })
    pd.DataFrame(rows).to_csv(OUT_SUM, index=False, encoding="utf-8")
    print(f"Resumo salvo em: {OUT_SUM}")

    # Pré-filtros
    candidates = [c for c in df.columns if c not in EXCLUDE]
    valid_cols = []
    for col in candidates:
        s = df[col]
        # corta colunas com nome suspeito de leak
        if LEAKAGE_NAME_PATTERNS.search(col):
            continue
        # corta constantes/muito ausentes/datetime
        if is_constant_or_empty(s):
            continue
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        valid_cols.append(col)

    print(f"Colunas candidatas após filtros: {len(valid_cols)}")

    num_cols, cat_cols, txt_cols = [], [], []

    for col in valid_cols:
        # Overrides
        if col in FORCE_TEXT: txt_cols.append(col); continue
        if col in FORCE_CAT:  cat_cols.append(col); continue
        if col in FORCE_NUM:  num_cols.append(col); continue

        s = df[col]

        # Pelo nome: texto longo?
        if TEXT_PATTERNS.search(col):
            txt_cols.append(col); continue

        # Pelo dtype
        if pd.api.types.is_numeric_dtype(s):
            num_cols.append(col); continue

        if pd.api.types.is_bool_dtype(s):
            cat_cols.append(col); continue

        # Strings: cardinalidade + comprimento médio
        if pd.api.types.is_string_dtype(s):
            nunique = s.nunique(dropna=True)
            mean_len = s.dropna().astype(str).map(len).mean() if nunique > 0 else 0
            if nunique <= CATEGORICAL_MAX_UNIQUE and mean_len < 25:
                cat_cols.append(col)
            else:
                txt_cols.append(col)
            continue
        # demais tipos: ignorar

    # Dedup/ordena
    num_cols = sorted(set(num_cols))
    cat_cols = sorted(set(cat_cols))
    txt_cols = sorted(set(txt_cols))

    # Fallback: se nada sobrou, usar todas strings (não-leak) como texto
    fallback_used = False
    if len(num_cols) + len(cat_cols) + len(txt_cols) == 0:
        print("⚠️  Nenhuma feature classificada. Fallback: todas strings (não-leak) => texto.")
        str_cols = [
            c for c in candidates
            if (pd.api.types.is_string_dtype(df[c]) and not LEAKAGE_NAME_PATTERNS.search(c))
        ]
        txt_cols = sorted(set(str_cols))
        fallback_used = True

    print("\n=== LISTAS GERADAS ===")
    print("NUM_CANDIDATES =", num_cols)
    print("CAT_CANDIDATES =", cat_cols)
    print("TXT_CANDIDATES =", txt_cols)

    with open(OUT_CFG, "w", encoding="utf-8") as f:
        json.dump(
            {
                "NUM_CANDIDATES": num_cols,
                "CAT_CANDIDATES": cat_cols,
                "TXT_CANDIDATES": txt_cols,
                "EXCLUDE": sorted(EXCLUDE),
                "fallback_used": fallback_used,
                "params": {
                    "CATEGORICAL_MAX_UNIQUE": CATEGORICAL_MAX_UNIQUE,
                    "MAX_MISSING_RATIO": MAX_MISSING_RATIO
                }
            },
            f, ensure_ascii=False, indent=2
        )
    print(f"\nConfig salva em: {OUT_CFG}")

if __name__ == "__main__":
    main()
