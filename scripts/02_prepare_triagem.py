from pathlib import Path
import pandas as pd

# ================================
# Caminhos
# ================================
CSV_IN = Path("data/processed/df_total.csv")
BASE_DIR = CSV_IN.parent

# ================================
# Leitura da base consolidada
# ================================
assert CSV_IN.exists(), f"Arquivo não encontrado: {CSV_IN}"
df = pd.read_csv(CSV_IN)
print("Arquivo carregado:", CSV_IN)
print("Shape inicial:", df.shape)

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

df["status_simplificado"] = (
    df["situacao_candidato"].str.lower().map(MAP_STATUS_FINAL).fillna("vazio")
)

print("\nDistribuição de status_simplificado:")
print(df["status_simplificado"].value_counts())

# ================================
# Remoção de registros iniciais
# ================================
iniciais = ["prospect", "inscrito"]
antes = len(df)
df = df[~df["situacao_candidato"].str.lower().isin(iniciais)].copy()
removidos = antes - len(df)
print(f"\nRegistros removidos por estarem em estágio inicial (Prospect/Inscrito): {removidos}")

# ================================
# Criação dos datasets derivados
# ================================

# Dataset de Triagem
df_triagem = df[df["status_simplificado"] != "vazio"].copy()
df_triagem["target_triagem"] = df_triagem["status_simplificado"].apply(
    lambda x: 1 if x in ["contratado", "em_processo"] else 0
)
print("\nDataset Triagem:", df_triagem.shape)
print("Proporção de target_triagem == 1:", round(df_triagem["target_triagem"].mean(), 3))

# Dataset de Contratação
df_contratacao = df[df["status_simplificado"].isin(["contratado", "negado"])].copy()
df_contratacao["target_contratacao"] = (
    df_contratacao["status_simplificado"] == "contratado"
).astype(int)
print("\nDataset Contratação:", df_contratacao.shape)
print("Proporção de target_contratacao == 1:", round(df_contratacao["target_contratacao"].mean(), 3))

# ================================
# Salvamento dos datasets em data/interim/
# ================================
OUT_TRIAGEM = BASE_DIR.parent / "interim" / "dataset_triagem.csv"
OUT_CONTRATACAO = BASE_DIR.parent / "interim" / "dataset_contratacao.csv"
OUT_TRIAGEM.parent.mkdir(parents=True, exist_ok=True)

df_triagem.to_csv(OUT_TRIAGEM, index=False)
print("Arquivo salvo:", OUT_TRIAGEM)

df_contratacao.to_csv(OUT_CONTRATACAO, index=False)
print("Arquivo salvo:", OUT_CONTRATACAO)
