# PS_IA

## Setup

### Option 1: Using uv (Recommended)

Install uv if not already installed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create virtual environment and install dependencies:
```bash
uv sync
```

Activate the virtual environment:
```bash
source .venv/bin/activate
```

### Option 2: Using traditional venv

Criação de um virtual enviroment

```bash
python -m venv .venv
source .venv/bin/activate
```

No Mac, para rodar o XGBoost, instalar também:

```bash
brew install libomp
```


Após acessar o virtual environment, instale as dependências

```bash
uv sync
#pip install -r requirements.txt
```


Download dos dados brutos

```bash
curl -L "https://drive.google.com/uc?export=download&id=1h8Lk5LM8VE5TF80mngCcbsQ14qA2rbw_" -o data/raw/vagas.zip
curl -L "https://drive.google.com/uc?export=download&id=1Z0dOk8FMjazQo03PuUeNGZOW-rxtpzmO" -o data/raw/applicants.zip
curl -L "https://drive.google.com/uc?export=download&id=1-hNfS7Z01fMM_JnT2K-zQrOoGm_C-jUT" -o data/raw/prospects.zip
```

## Geração do modelo

```bash
uvicorn src.ml.train_pipeline:main
```

## Run Server

```bash
uvicorn app.main:app --reload
```





