
# scripts/00_check_config.py
from pathlib import Path
import sys

def hint_env():
    print("\nDica: se aparecer 'No module named src', rode na mesma janela do PowerShell:")
    print("$env:PYTHONPATH = (Get-Location).Path\n")

# === garantir que a raiz do projeto entre no sys.path ===
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# ========================================================


try:
    from src.projeto.core.config import (
        FILE_JOBS, FILE_PROSPECTS, FILE_APPLICANTS,
        DATA_RAW, DATA_PROCESSED, BASE_DIR
    )
except Exception as e:
    print("FALHOU importar src.projeto.core.config ->", e)
    hint_env()
    sys.exit(1)

paths = {
    "BASE_DIR": BASE_DIR,
    "DATA_RAW": DATA_RAW,
    "DATA_PROCESSED": DATA_PROCESSED,
    "FILE_JOBS": FILE_JOBS,
    "FILE_PROSPECTS": FILE_PROSPECTS,
    "FILE_APPLICANTS": FILE_APPLICANTS,
}

for name, p in paths.items():
    print(f"{name}: {p} | exists={Path(p).exists()}")

print("\n OK: import e caminhos verificados.")
