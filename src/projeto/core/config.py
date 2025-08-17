# src/projeto/core/config.py

from pathlib import Path

# Diretórios principais
BASE_DIR = Path(__file__).resolve().parents[3]  # ajusta se necessário
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models" / "artifacts"

# Arquivos de entrada
FILE_JOBS = DATA_RAW / "vagas.json"
FILE_PROSPECTS = DATA_RAW / "prospects.json"
FILE_APPLICANTS = DATA_RAW / "applicants.json"

# Arquivo de saída
FILE_DATASET_FINAL = DATA_PROCESSED / "dataset_final.csv"

# Parâmetros do modelo (exemplo)
MODEL_PATH = MODELS_DIR / "modelo_match.pkl"

# Modelo de matching
MATCHING_MODEL_PATH = MODELS_DIR / "matching_model.pkl"