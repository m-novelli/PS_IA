import json
import pandas as pd
from pathlib import Path

# Diretórios
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# =====================
# FUNÇÕES AUXILIARES
# =====================
def load_dict_json_flat(path: Path, id_col: str) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return pd.json_normalize([{id_col: k, **v} for k, v in data.items()])

def load_prospects(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    flattened = []
    for codigo_vaga, registro in data.items():
        titulo = registro.get("titulo")
        modalidade = registro.get("modalidade")
        for candidato in registro.get("prospects", []):
            flattened.append({
                "codigo_vaga": codigo_vaga,
                "titulo_vaga": titulo,
                "modalidade": modalidade,
                "codigo_applicant": candidato.get("codigo"),
                "nome_candidato": candidato.get("nome"),
                "situacao_candidato": candidato.get("situacao_candidado"),
                "data_candidatura": candidato.get("data_candidatura"),
                "ultima_atualizacao": candidato.get("ultima_atualizacao"),
                "comentario": candidato.get("comentario"),
                "recrutador": candidato.get("recrutador"),
            })

    return pd.DataFrame(flattened)

# =====================
# CARREGAMENTO E FLATTEN
# =====================
print("🔄 Carregando bases...")
vagas = load_dict_json_flat(RAW_DIR / "vagas.json", "codigo_vaga")
applicants = load_dict_json_flat(RAW_DIR / "applicants.json", "codigo_applicant")
prospects = load_prospects(RAW_DIR / "prospects.json")

# =====================
# MERGE FINAL
# =====================
print("🔗 Realizando merges...")
vagas_com_prospects = vagas.merge(prospects, on="codigo_vaga", how="left")
df_total = vagas_com_prospects.merge(applicants, on="codigo_applicant", how="left", suffixes=("", "_applicant"))

print(f"✅ df_total gerado com shape: {df_total.shape}")

# =====================
# SALVAMENTO
# =====================
output_path = PROCESSED_DIR / "df_total.parquet"
df_total.to_parquet(output_path, index=False)
print(f"💾 Arquivo salvo em: {output_path}")

output_csv = PROCESSED_DIR / "df_total.csv"
df_total.to_csv(output_csv, index=False)
print(f"📄 CSV salvo em: {output_csv}")
