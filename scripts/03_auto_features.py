from pathlib import Path
import re
import json
import pandas as pd
import numpy as np

# ===================== Caminhos =====================
BASE_DIR = Path(__file__).resolve().parents[1]
CSV_IN = BASE_DIR / "data" / "interim" / "dataset_triagem.csv"
CFG_OUT = BASE_DIR / "configs" / "triagem_features.json"
SUM_OUT = BASE_DIR / "data" / "interim" / "columns_summary.csv"

CFG_OUT.parent.mkdir(parents=True, exist_ok=True)
SUM_OUT.parent.mkdir(parents=True, exist_ok=True)

# ===================== Parâmetros =====================
MAX_MISSING_RATIO = 0.98
CATEGORICAL_MAX_UNIQUE = 50

EXCLUDE = {
    "target_triagem", "codigo_applicant", "codigo_vaga",
    "situacao_candidato", "status_simplificado"
}

LEAKAGE_PATTERNS = re.compile(
    r"(status|situac|situacao|resultado|contratad|aprovad|reprovad|negad|proposta|documenta)",
    re.IGNORECASE
)

TEXT_PATTERNS = re.compile(
    r"(cv|descricao|descrição|titulo|título|requisit|competenc|atividade|responsabil|resumo|perfil|skill|texto|observac|coment)",
    re.IGNORECASE
)

FORCE_TEXT, FORCE_CAT, FORCE_NUM = set(), set(), set()

# ===================== Funções auxiliares =====================
def is_invalid_feature(s: pd.Series) -> bool:
    if s.nunique(dropna=True) <= 1:
        return True
    if s.isna().mean() > MAX_MISSING_RATIO:
        return True
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    return False

# ===================== Função principal =====================
def main():
    df = pd.read_csv(CSV_IN)
    print(f"CSV: {CSV_IN}  |  Shape: {df.shape}")

    summary = []
    for col in df.columns:
        s = df[col]
        summary.append({
            "coluna": col,
            "dtype": str(s.dtype),
            "n_unique": s.nunique(dropna=True),
            "missing_ratio": round(s.isna().mean(), 4),
            "exemplos": ", ".join(map(str, s.dropna().unique()[:5]))
        })
    pd.DataFrame(summary).to_csv(SUM_OUT, index=False)
    print(f"Resumo salvo em: {SUM_OUT}")

    candidates = [c for c in df.columns if c not in EXCLUDE]
    valid = [c for c in candidates if not LEAKAGE_PATTERNS.search(c) and not is_invalid_feature(df[c])]

    num, cat, txt = [], [], []
    for col in valid:
        s = df[col]
        if col in FORCE_TEXT:
            txt.append(col)
        elif col in FORCE_CAT:
            cat.append(col)
        elif col in FORCE_NUM:
            num.append(col)
        elif TEXT_PATTERNS.search(col):
            txt.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            num.append(col)
        elif pd.api.types.is_bool_dtype(s):
            cat.append(col)
        elif pd.api.types.is_string_dtype(s):
            if s.nunique(dropna=True) <= CATEGORICAL_MAX_UNIQUE and s.dropna().astype(str).map(len).mean() < 25:
                cat.append(col)
            else:
                txt.append(col)

    num, cat, txt = sorted(set(num)), sorted(set(cat)), sorted(set(txt))
    fallback = len(num + cat + txt) == 0

    config = {
        "NUM_CANDIDATES": num,
        "CAT_CANDIDATES": cat,
        "TXT_CANDIDATES": txt,
        "EXCLUDE": sorted(EXCLUDE),
        "fallback_used": fallback,
        "params": {
            "CATEGORICAL_MAX_UNIQUE": CATEGORICAL_MAX_UNIQUE,
            "MAX_MISSING_RATIO": MAX_MISSING_RATIO
        }
    }

    with open(CFG_OUT, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Config salva em: {CFG_OUT}")

if __name__ == "__main__":
    main()
