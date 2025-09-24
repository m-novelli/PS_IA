# PS_IA

## Setup

Criação de um virtual enviroment

```bash
python -m venv .venv
source .venv/bin/activate
```

Após acessar o virtual environment, instale as dependências

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload
```





