# scripts/06_check_leakage.py

import pandas as pd
from pathlib import Path

# Caminhos
BASE_DIR = Path(__file__).resolve().parents[1]
CSV = BASE_DIR / "data" / "interim" / "dataset_triagem.csv"
TARGET = "target_triagem"

# Lê o dataset
print(f"Lendo: {CSV}")
df = pd.read_csv(CSV, low_memory=False)
print("Shape:", df.shape)

# Confere se o target está presente
if TARGET not in df.columns:
    raise ValueError(f"Coluna alvo '{TARGET}' não encontrada no dataset.")

# Converte o target para numérico
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

# Seleciona colunas numéricas (exceto o target)
numeric_cols = df.select_dtypes(include=["number"]).columns.drop(TARGET, errors="ignore")

# Calcula a correlação de cada variável numérica com o target
correlations = df[numeric_cols].corrwith(df[TARGET]).sort_values(ascending=False)

# Mostra as correlações mais altas
print("\nTop 20 correlações com o target:")
print(correlations.head(20))

# Exporta resultado para CSV (opcional)
OUT = BASE_DIR / "reports" / "correlation_with_target.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)
correlations.to_csv(OUT, header=["correlation"])
print(f"\nArquivo salvo em: {OUT}")
