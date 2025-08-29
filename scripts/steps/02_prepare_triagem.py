from pathlib import Path
import pandas as pd
import json

# ================================
# Função utilitária para o feature_map
# ================================
def load_feature_map(path: Path) -> dict:
    assert path.exists(), f"Arquivo não encontrado: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ================================
# Caminhos
# ================================
CSV_IN = Path("data/processed/df_total.csv")
BASE_DIR = CSV_IN.parent  # data/processed

FEATURE_MAP_PATH = BASE_DIR.parent / "processed" / "feature_map.json"
OUT_TRIAGEM = BASE_DIR.parent / "interim" / "dataset_triagem.csv"
OUT_TRIAGEM_CLEAN = BASE_DIR / "dataset_triagem_clean.csv"

OUT_TRIAGEM.parent.mkdir(parents=True, exist_ok=True)

# ================================
# Leitura da base consolidada
# ================================
assert CSV_IN.exists(), f"Arquivo não encontrado: {CSV_IN}"
df = pd.read_csv(CSV_IN)
print("Arquivo carregado:", CSV_IN)
print("Shape inicial:", df.shape)

# ================================
# Verificação da existência da coluna crítica
# ================================
assert "situacao_candidato" in df.columns, "'situacao_candidato' não encontrada no dataframe!"

# ================================
# Mapeamento para status_simplificado
# ================================
MAP_STATUS_FINAL = {
    "contratado pela decision": "contratado",
    "contratado como hunting": "contratado",
    "aprovado": "contratado",
    "proposta aceita": "contratado",
    "encaminhar proposta": "contratado",
    "documentação pj": "contratado",
    "documentação clt": "contratado",
    "documentação cooperado": "contratado",
    "não aprovado pelo cliente": "negado",
    "não aprovado pelo rh": "negado",
    "não aprovado pelo requisitante": "negado",
    "recusado": "negado",
    "desistiu": "negado",
    "desistiu da contratação": "negado",
    "sem interesse nesta vaga": "negado",
    "prospect": "em_processo",
    "inscrito": "em_processo",
    "encaminhado ao requisitante": "em_processo",
    "entrevista técnica": "em_processo",
    "entrevista com cliente": "em_processo",
    "em avaliação pelo rh": "em_processo",
}

df["situacao_candidato"] = df["situacao_candidato"].fillna("").str.lower()
df["status_simplificado"] = df["situacao_candidato"].map(MAP_STATUS_FINAL).fillna("vazio")

print("\nDistribuição de status_simplificado:")
print(df["status_simplificado"].value_counts())

# ================================
# Remoção de registros iniciais
# ================================
iniciais = ["prospect", "inscrito"]
antes = len(df)
df = df[~df["situacao_candidato"].isin(iniciais)].copy()
removidos = antes - len(df)
print(f"\nRegistros removidos por estarem em estágio inicial (Prospect/Inscrito): {removidos}")

# ================================
# Criação do dataset de Triagem
# ================================
df_triagem = df[df["status_simplificado"] != "vazio"].copy()
df_triagem["target_triagem"] = df_triagem["status_simplificado"].apply(
    lambda x: 1 if x in ["contratado", "em_processo"] else 0
)
print("\nDataset Triagem:", df_triagem.shape)
print("Proporção de target_triagem == 1:", round(df_triagem["target_triagem"].mean(), 3))

# ================================
# Filtro de colunas com base no feature_map.json
# ================================
fmap = load_feature_map(FEATURE_MAP_PATH)

# ⚠️ Se desejar, inclua no JSON:
# "target": ["status_simplificado", "target_triagem", "target_contratacao"]
cols_to_remove = (
    fmap.get("id", [])
    + fmap.get("date", [])
    + fmap.get("personal_info", [])
    + fmap.get("leakage_risk", [])
)

cols_to_remove_existing = [col for col in cols_to_remove if col in df_triagem.columns]
cols_not_found = [col for col in cols_to_remove if col not in df_triagem.columns]

df_triagem_clean = df_triagem.drop(columns=cols_to_remove_existing)

print(f"\nColunas removidas com base no feature_map ({len(cols_to_remove_existing)}):")
print(cols_to_remove_existing)

if cols_not_found:
    print(f"Aviso: as seguintes colunas do feature_map não estavam no dataframe e foram ignoradas:\n{cols_not_found}")

print("Shape final do dataset limpo:", df_triagem_clean.shape)

# ================================
# Salvamento dos datasets
# ================================
df_triagem.to_csv(OUT_TRIAGEM, index=False)
print("Arquivo salvo:", OUT_TRIAGEM)

df_triagem_clean.to_csv(OUT_TRIAGEM_CLEAN, index=False)
print("Arquivo salvo:", OUT_TRIAGEM_CLEAN)
