from pathlib import Path
import json
import pandas as pd

# ===================== Caminhos =====================
BASE_DIR = Path(__file__).resolve().parents[2]
CSV_IN = BASE_DIR / "data" / "processed" / "dataset_triagem_fe.csv"
FEATURE_MAP = BASE_DIR / "data" / "processed" / "feature_map.json"
CFG_OUT = BASE_DIR / "configs" / "triagem_features.json"
SUM_OUT = BASE_DIR / "data" / "interim" / "columns_summary.csv"

CFG_OUT.parent.mkdir(parents=True, exist_ok=True)
SUM_OUT.parent.mkdir(parents=True, exist_ok=True)

# ===================== Carrega base e feature_map =====================
df = pd.read_csv(CSV_IN)
with open(FEATURE_MAP, "r", encoding="utf-8") as f:
    fmap = json.load(f)

print(f"Base carregada: {df.shape}")
print(f"feature_map carregado: {FEATURE_MAP.name}")

# ===================== Validação do feature_map =====================
expected_keys = {
    "id", "target", "date", "personal_info", "leakage_risk",
    "vaga_info", "candidato_info", "text"
}
missing_keys = expected_keys - fmap.keys()
if missing_keys:
    print(f"\n Atenção: as seguintes chaves estão ausentes do feature_map: {missing_keys}")

# ===================== Gera resumo para auditoria =====================
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

# ===================== Classificação de features =====================
cols = set(df.columns)

for key in fmap:
    fmap[key] = [col for col in fmap[key] if col in cols]

exclude = set(
    fmap.get("id", [])
    + fmap.get("target", [])
    + fmap.get("date", [])
    + fmap.get("personal_info", [])
    + fmap.get("group", [])
    + fmap.get("leakage_risk", [])
)
exclude = exclude & cols  # Garante que estão na base

# Agrupamentos com base no feature_map
cat = sorted(set(fmap.get("vaga_info", []) + fmap.get("candidato_info", [])) & cols)
txt = sorted(set(fmap.get("text", [])) & cols)

# Colunas numéricas restantes e novas features contínuas
num = sorted([
    col for col in cols
    if pd.api.types.is_numeric_dtype(df[col])
    and col not in exclude
    and col not in cat
    and col not in txt
])

# Checa colunas não mapeadas em nenhum grupo
all_mapped = exclude | set(cat) | set(txt) | set(num)
unmapped = sorted(cols - all_mapped)

if unmapped:
    print("\n Colunas não mapeadas no feature_map e não classificadas automaticamente:")
    print(unmapped)

# ===================== Salva config final =====================
config = {
    "NUM_CANDIDATES": sorted(num),
    "CAT_CANDIDATES": sorted(cat),
    "TXT_CANDIDATES": sorted(txt),
    "EXCLUDE": sorted(exclude),
    "fallback_used": False,
    "params": {
        "source": "feature_map.json",
        "base_input": str(CSV_IN.name)
    }
}

with open(CFG_OUT, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print(f"\nConfig salva em: {CFG_OUT}")
