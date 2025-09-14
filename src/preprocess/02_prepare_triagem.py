from pathlib import Path
import pandas as pd
import json

# ================================
# Caminhos 
# ================================
BASE_DIR = Path(__file__).resolve().parents[2]

CSV_IN            = BASE_DIR / "data" / "interim" / "df_total.csv"
FEATURE_MAP_PATH  = BASE_DIR / "data" / "processed" / "feature_map.json"
OUTPUT            = BASE_DIR / "data" / "processed" / "dataset_triagem_clean.csv"

# ================================
# Função utilitária para o feature_map
# ================================
def load_feature_map(path: Path) -> dict:
    assert path.exists(), f"Arquivo não encontrado: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

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
iniciais = {"prospect", "inscrito"}
antes = len(df)
df = df[~df["situacao_candidato"].isin(iniciais)].copy()
removidos = antes - len(df)
print(f"\nRemovidos por estágio inicial (Prospect/Inscrito): {removidos}")

# ================================
# Dataset de Triagem + target
# ================================
df_triagem = df[df["status_simplificado"] != "vazio"].copy()
df_triagem["target_triagem"] = df_triagem["status_simplificado"].apply(
    lambda x: 1 if x in {"contratado", "em_processo"} else 0
)
print("\nDataset Triagem:", df_triagem.shape)
print("Proporção target_triagem==1:", round(df_triagem["target_triagem"].mean(), 3))

# ================================
# Filtro de colunas com base no feature_map.json
# ================================
fmap = load_feature_map(FEATURE_MAP_PATH)

cols_to_remove = (
    fmap.get("id", [])
    + fmap.get("date", [])
    + fmap.get("personal_info", [])
    + fmap.get("leakage_risk", [])
)

cols_to_remove_existing = [c for c in cols_to_remove if c in df_triagem.columns]
cols_not_found = [c for c in cols_to_remove if c not in df_triagem.columns]

df_triagem_clean = df_triagem.drop(columns=cols_to_remove_existing)

print(f"\nColunas removidas ({len(cols_to_remove_existing)}):")
print(cols_to_remove_existing)
if cols_not_found:
    print(f"Aviso: colunas do feature_map não encontradas (ignoradas): {cols_not_found}")

print("Shape final do dataset limpo:", df_triagem_clean.shape)

# ================================
# (Opcional) Checagens que ajudam o treino/serving
# ================================
# Garante que colunas-chave que o pipeline usa tendem a existir
chaves_recomendadas = [
    "codigo_vaga",
    "cv_pt",
    "perfil_vaga.principais_atividades",
]
faltantes = [c for c in chaves_recomendadas if c not in df_triagem_clean.columns]
if faltantes:
    print("[AVISO] Colunas recomendadas ausentes (o pipeline lida com faltas, mas pode degradar qualidade):", faltantes)

# ================================
# Salvamento
# ================================
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df_triagem_clean.to_csv(OUTPUT, index=False)
print("Arquivo salvo:", OUTPUT)
