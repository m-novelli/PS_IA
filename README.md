# Projeto PS_IA: API de Triagem de Candidatos e Geração de Perguntas com Machine Learning e IA

## Visão Geral do Projeto

O projeto **PS_IA** é uma solução abrangente que integra Machine Learning e Inteligência Artificial Generativa para otimizar o processo de recrutamento e seleção. Ele consiste em uma API, desenvolvida com FastAPI, capaz de realizar a triagem automática de candidatos para vagas e, opcionalmente, gerar perguntas personalizadas para entrevistas utilizando um Large Language Model (LLM). Além da API, o projeto inclui um pipeline completo para treinamento e monitoramento do modelo de Machine Learning, garantindo sua performance e adaptabilidade ao longo do tempo.

## 🎯 Objetivos do Projeto

Os principais objetivos deste projeto são:

1.  **Automatizar a Triagem de Candidatos**: Desenvolver um modelo de Machine Learning capaz de classificar candidatos com base em sua adequação a uma vaga, agilizando o processo de seleção.
2.  **Otimizar o Processo de Entrevista**: Integrar um LLM para gerar perguntas de entrevista relevantes e personalizadas, tanto comuns quanto específicas para cada candidato, a partir de descrições de vagas e currículos.
3.  **Garantir a Qualidade do Modelo em Produção**: Implementar mecanismos de monitoramento de drift para detectar desvios nos dados ou no desempenho do modelo, assegurando sua eficácia contínua.
4.  **Fornecer uma API Escalável e Robusta**: Construir uma API com FastAPI que seja fácil de integrar, performática e que ofereça endpoints claros para predição e ranking de candidatos.

##  Arquitetura e Estrutura

O projeto segue uma arquitetura modular, dividida em componentes principais:

*   **API (FastAPI)**: Serve como interface para interagir com o modelo de ML e o LLM.
*   **Módulo de Machine Learning**: Contém o pipeline de treinamento, avaliação e persistência do modelo.
*   **Módulo de Pré-processamento**: Responsável pela preparação dos dados para o treinamento e inferência.
*   **Módulo de Geração de Perguntas (LLM)**: Integração com OpenAI para IA Generativa.
*   **Módulo de Monitoramento**: Ferramentas para verificar a saúde e performance do modelo em produção.

### Estrutura de Diretórios Simplificada

```
PS_IA-main/
├── app/                         # Aplicação FastAPI (endpoints, lógica de negócio, LLM)
├── artifacts/                   # Artefatos gerados (relatórios de drift)
├── configs/                     # Arquivos de configuração
├── data/                        # Dados brutos e processados
├── models/prod/                 # Modelos treinados e metadados para a API
├── src/                         # Código fonte principal (ML, pré-processamento, monitoramento)
│   ├── ml/                      # Pipeline de ML, treinamento, transformadores
│   ├── monitoring/              # Lógica de detecção de drift
│   └── preprocess/              # Scripts de pré-processamento de dados
├── scripts/                     # Scripts utilitários (check_drift, testadores)
├── tests/                       # Testes unitários e de integração
├── requirements.txt             # Dependências do projeto
└── README_FINAL_GITHUB.md       # Este arquivo
```

## 🛠️ Tecnologias Utilizadas

*   **Python 3.8+**: Linguagem de programação principal.
*   **FastAPI**: Framework web para construção da API.
*   **Scikit-learn**: Biblioteca para Machine Learning (pipelines, transformadores).
*   **XGBoost**: Algoritmo de Machine Learning para classificação.
*   **Pandas**: Manipulação e análise de dados.
*   **Joblib**: Serialização de modelos Python.
*   **MLflow**: Plataforma para gerenciar o ciclo de vida de Machine Learning (rastreamento de experimentos, registro de modelos).
*   **OpenAI SDK**: Integração com modelos de linguagem (LLMs) para geração de texto.
*   **Pytest**: Framework para testes.
*   **uv / pip**: Gerenciadores de pacotes Python.

## ⚙️ Como Rodar o Projeto

Para que o professor possa analisar e executar o projeto, siga os passos abaixo:

### Pré-requisitos

*   Python 3.8+
*   `uv` (recomendado) ou `pip`
*   Chave de API do OpenAI (se for testar a funcionalidade de geração de perguntas).

### 1. Configuração do Ambiente

1.  **Clone o repositório:**
    ```bash
    git clone <URL_DO_REPOSITORIO>
    cd PS_IA-main
    ```

2.  **Instale `uv` (se não tiver):**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

3.  **Crie e ative o ambiente virtual e instale as dependências:**
    ```bash
uv sync
source .venv/bin/activate
    ```
    *(Alternativamente, use `python -m venv .venv` e `pip install -r requirements.txt`)*

### 2. Download dos Dados Brutos

Para replicar o ambiente de treinamento, você precisará dos dados. O projeto original usava `curl` para baixar arquivos zip do Google Drive. Adapte conforme a disponibilidade dos dados:

```bash

curl -L "https://drive.google.com/uc?export=download&id=1h8Lk5LM8VE5TF80mngCcbsQ14qA2rbw_" -o data/raw/vagas.zip
curl -L "https://drive.google.com/uc?export=download&id=1Z0dOk8FMjazQo03PuUeNGZOW-rxtpzmO" -o data/raw/applicants.zip
curl -L "https://drive.google.com/uc?export=download&id=1-hNfS7Z01fMM_JnT2K-zQrOoGm_C-jUT" -o data/raw/prospects.zip
# Descompacte os arquivos e execute os scripts de pré-processamento em src/preprocess/
```

### 3. Treinamento do Modelo

O modelo de Machine Learning é treinado e salvo localmente para ser consumido pela API. Execute o script de treinamento:

```bash
python src/ml/train_pipeline.py
```

Este script irá:
*   Carregar os dados processados. O script `src/ml/train_pipeline.py` espera o dataset consolidado `dataset_triagem_clean.csv` no diretório `data/processed/`. Para fins de teste e avaliação, se este arquivo já estiver disponível (por exemplo, como `df_total` ou similar), a etapa de pré-processamento pode ser ignorada. Caso contrário, ele deve ser gerado a partir dos dados brutos (`vagas.zip`, `applicants.zip`, `prospects.zip`) utilizando os scripts de pré-processamento em `src/preprocess/` (por exemplo, `01_json_to_df.py` e `02_prepare_triagem.py`).
*   Construir e treinar um pipeline de ML (XGBoost).
*   Avaliar o modelo e logar métricas no MLflow (se configurado).
*   Salvar o modelo treinado (`model.joblib`) e seus metadados (`meta.json`) no diretório `models/prod/`.


### 4. Iniciar a API

Defina as variáveis de ambiente necessárias (especialmente `OPENAI_API_KEY` se for usar a funcionalidade de LLM) e inicie a aplicação FastAPI:

```bash
export OPENAI_API_KEY="sua_chave_openai"


uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Após iniciar, a API estará disponível em `http://localhost:8000`. A documentação interativa (Swagger UI) pode ser acessada em `http://localhost:8000/docs`.

## 📊 Principais Funcionalidades e Resultados

### API RESTful (FastAPI)

A API expõe os seguintes endpoints principais:

*   **`/health` (GET)**: Retorna o status da API e metadados do modelo carregado (versão, métricas, schema de entrada).
*   **`/schema` (GET)**: Apresenta o esquema de entrada esperado pelo modelo (colunas categóricas e de texto).
*   **`/predict` (POST)**: Realiza a predição para um único item (candidato/vaga), retornando a probabilidade de avanço e a classificação.
*   **`/rank-and-suggest` (POST)**: Recebe uma vaga e uma lista de candidatos, retorna um ranking dos candidatos mais adequados e, opcionalmente, gera perguntas de entrevista via LLM.

### Modelo de Machine Learning

O modelo é um pipeline `scikit-learn` com um classificador `XGBoost`, treinado para identificar a probabilidade de um candidato avançar para a próxima fase do processo seletivo. As métricas de avaliação (ROC AUC, PR AUC, F1-Score) são calculadas durante o treinamento e logadas no MLflow, demonstrando a performance do modelo em dados de holdout.

### Geração de Perguntas com LLM

A funcionalidade de `/rank-and-suggest` pode invocar um LLM (OpenAI) para gerar:

*   **Perguntas Comuns**: 5 perguntas gerais aplicáveis a todos os candidatos para a vaga.
*   **Perguntas Personalizadas**: 3 perguntas específicas para cada candidato, baseadas em seu currículo e na descrição da vaga.

Esta funcionalidade visa enriquecer o processo de entrevista, fornecendo insights mais profundos sobre cada candidato.

### Monitoramento de Drift

O projeto inclui um script (`scripts/check_drift.py`) e um módulo (`src/monitoring/drift.py`) para monitorar o drift de dados e de modelo. Isso é crucial para garantir que o modelo continue performando bem em produção, alertando sobre mudanças nas características dos dados de entrada que possam impactar a qualidade das predições.

## ✅ Testes

O projeto conta com uma suíte de testes abrangente (`tests/`) utilizando `pytest`, cobrindo:

*   **Testes de API**: Validação dos endpoints, respostas e tratamento de erros.
*   **Testes de Lógica de Negócio**: Verificação das funções auxiliares e do pré-processamento.
*   **Testes de Pipeline de ML**: Garantia da correta construção e funcionamento do pipeline de treinamento.
*   **Testes de LLM**: Validação da integração e saída da geração de perguntas.


