# Projeto PS_IA: API de Triagem de Candidatos e Geração de Perguntas com Machine Learning

## 🚀 Visão Geral do Projeto

O projeto **PS_IA** é uma solução abrangente que integra Machine Learning e Inteligência Artificial Generativa para otimizar o processo de recrutamento e seleção. Ele consiste em uma API robusta, desenvolvida com FastAPI, capaz de realizar a triagem automática de candidatos para vagas e, opcionalmente, gerar perguntas personalizadas para entrevistas utilizando um Large Language Model (LLM). Além da API, o projeto inclui um pipeline completo para treinamento e monitoramento do modelo de Machine Learning, garantindo sua performance e adaptabilidade ao longo do tempo.

## 🎯 Objetivos do Projeto

Os principais objetivos deste projeto são:

1.  **Automatizar a Triagem de Candidatos**: Desenvolver um modelo de Machine Learning capaz de classificar candidatos com base em sua adequação a uma vaga, agilizando o processo de seleção.
2.  **Otimizar o Processo de Entrevista**: Integrar um LLM para gerar perguntas de entrevista relevantes e personalizadas, tanto comuns quanto específicas para cada candidato, a partir de descrições de vagas e currículos.
3.  **Garantir a Qualidade do Modelo em Produção**: Implementar mecanismos de monitoramento de drift para detectar desvios nos dados ou no desempenho do modelo, assegurando sua eficácia contínua.
4.  **Fornecer uma API Escalável e Robusta**: Construir uma API com FastAPI que seja fácil de integrar, performática e que ofereça endpoints claros para predição e ranking de candidatos.

## 🏗️ Arquitetura e Estrutura

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

### 2. Sequência de Execução

Para rodar o projeto do zero, siga a sequência de passos abaixo:

#### 2.1. Download dos Dados Brutos (se necessário)

Caso os dados brutos não estejam disponíveis, você pode baixá-los:

```bash
# Exemplo de download 
curl -L "https://drive.google.com/uc?export=download&id=1h8Lk5LM8VE5TF80mngCcbsQ14qA2rbw_" -o data/raw/vagas.zip
unzip data/raw/vagas.zip -d data/raw/
curl -L "https://drive.google.com/uc?export=download&id=1Z0dOk8FMjazQo03PuUeNGZOW-rxtpzmO" -o data/raw/applicants.zip
unzip data/raw/applicants.zip -d data/raw/
curl -L "https://drive.google.com/uc?export=download&id=1-hNfS7Z01fMM_JnT2K-zQrOoGm_C-jUT" -o data/raw/prospects.zip
unzip data/raw/prospects.zip -d data/raw/
```

Os arquivos estão disponiveis no Google Drive, o link está em data/raw/links.txt.

Após o download, descompacte os arquivos zip para o diretório `data/raw/`.

#### 2.2. Pré-processamento dos Dados

Esta etapa é crucial para transformar os dados brutos em um formato utilizável pelo modelo. O script `src/ml/train_pipeline.py` espera o dataset consolidado `dataset_triagem_clean.csv` no diretório `data/processed/`.

Para gerar este arquivo, execute os scripts de pré-processamento na ordem correta. Por exemplo:

```bash
python src/preprocess/01_json_to_df.py
python src/preprocess/02_prepare_triagem.py
# Certifique-se de que o arquivo dataset_triagem_clean.csv seja gerado em data/processed/
```

**Observação:** Para fins de teste e avaliação, se o arquivo `data/processed/dataset_triagem_clean.csv` já estiver disponível (por exemplo, como `df_total` ou similar), esta etapa pode ser ignorada.

#### 2.3. Treinamento do Modelo

Com os dados processados disponíveis, o modelo de Machine Learning pode ser treinado e salvo localmente para ser consumido pela API. Execute o script de treinamento:

```bash
python src/ml/train_pipeline.py
```

Este script irá:
*   Carregar o `dataset_triagem_clean.csv` de `data/processed/`.
*   Construir e treinar um pipeline de ML (XGBoost).
*   Avaliar o modelo e logar métricas no MLflow (se configurado).
*   Salvar o modelo treinado (`model.joblib`) e seus metadados (`meta.json`) no diretório `models/prod/`.


### 3. Iniciar a API

```bash
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

#### Exemplo de Uso da API

**Endpoint: `/predict` (POST)**

**Entrada (JSON):**
```json
{
  "meta": {"external_id": "cand-11010"},
  "features": {
    "cv_pt": "Consultor SAP Basis com HANA e Fiori.",
    "informacoes_profissionais.conhecimentos_tecnicos": "SAP Basis, HANA, Fiori",
    "perfil_vaga.principais_atividades": "Gestão de incidentes e SLAs.",
    "perfil_vaga.competencia_tecnicas_e_comportamentais": "SAP Basis, liderança"
  }
}
```

**Saída (JSON):**
```json
{
  "meta": {"external_id": "cand-11010"},
  "prediction": {
    "prob_next_phase": 0.85,
    "label": 1,
    "threshold": 0.6
  },
  "model": {
    "name": "sklearn-pipeline-xgboost",
    "artifact": "model.joblib",
    "version": "2.0.0"
  }
}
```

**Endpoint: `/rank-and-suggest` (POST)**

**Entrada (JSON):**
```json
{
  "vaga": {
    "perfil_vaga.principais_atividades": "Operações e gestão de incidentes, SLAs.",
    "perfil_vaga.competencia_tecnicas_e_comportamentais": "SAP Basis, liderança"
  },
  "candidatos": [
    {
      "meta": {"external_id": "11010"},
      "candidato": {
        "cv_pt": "Experiência forte em SAP Basis e HANA.",
        "informacoes_profissionais.conhecimentos_tecnicos": "SAP Basis, HANA, Fiori"
      }
    },
    {
      "meta": {"external_id": "11011"},
      "candidato": {
        "cv_pt": "Gestão de operações e incidentes, ITIL.",
        "informacoes_profissionais.conhecimentos_tecnicos": "ITIL, SLAs, Linux"
      }
    }
  ]
}
```

**Saída (JSON - com `include_questions=True`):**
```json
{
  "job_meta": {
    "perfil_vaga.principais_atividades": "Operações e gestão de incidentes, SLAs.",
    "perfil_vaga.competencia_tecnicas_e_comportamentais": "SAP Basis, liderança"
  },
  "top_k": 5,
  "threshold_used": 0.6,
  "results": [
    {
      "external_id": "11010",
      "prob_next_phase": 0.92,
      "label": 1
    },
    {
      "external_id": "11011",
      "prob_next_phase": 0.78,
      "label": 1
    }
  ],
  "questions": {
    "common_questions": [
      "Quais são suas principais experiências com gestão de incidentes e SLAs?",
      "Como você lida com a pressão em ambientes de alta demanda?",
      "Descreva uma situação em que você demonstrou liderança técnica.",
      "Quais são seus conhecimentos em SAP Basis e como os aplicaria nesta função?",
      "Como você se mantém atualizado sobre as novas tecnologias em sua área?"
    ],
    "per_candidate": {
      "11010": [
        "Pode detalhar sua experiência com HANA e Fiori no contexto de SAP Basis?",
        "Como sua experiência em SAP Basis se alinha com a gestão de incidentes?",
        "Descreva um projeto desafiador em que você utilizou SAP Basis e HANA."
      ],
      "11011": [
        "Qual sua experiência com ITIL e como ela contribui para a gestão de operações?",
        "Como você aplicaria seus conhecimentos em Linux para otimizar SLAs?",
        "Pode descrever um cenário onde sua experiência em gestão de incidentes foi crucial?"
      ]
    }
  }
}
```

### 4. Executar com Docker

Se preferir rodar a aplicação via Docker, sem precisar configurar o ambiente Python localmente, siga os passos abaixo:

#### 4.1. Construir a imagem
No diretório raiz do projeto (onde está o `Dockerfile`), execute:

```bash
docker build -t ps_ia:with-model .
```
Isso criará uma imagem chamada ps_ia:with-model

#### 4.2. Rodar o container
Para iniciar a aplicação, expondo a porta 8000
```bash
docker run -it --rm -p 8000:8000 \
  -e OPENAI_API_KEY="sua-chave-aqui" \
  ps_ia:with-model
```
OPENAI_API_KEY= necessária se quiser utilizar a feature de gerar perguntas para os candidatos

#### 4.3. Usando .env
Para não expor a chave diretamente na linha de comando, você pode criar um arquivo .env na raiz do projeto (não commitado no Git) com o seguinte conteúdo:
```
OPENAI_API_KEY=sua-chave-aqui
```
E rodar:
```bash
docker run -it --rm -p 8000:8000 --env-file .env ps_ia:with-model
```

Após rodar o container, a API estará disponível em:

Swagger UI: http://localhost:8000/docs

OpenAPI JSON: http://localhost:8000/openapi.json

## Modelo de Machine Learning

O modelo central para a triagem de candidatos é um pipeline `scikit-learn` que incorpora um classificador **XGBoost**. A escolha do XGBoost se deu pela sua comprovada eficácia em problemas de classificação tabular, combinando alta performance, robustez a dados ruidosos e capacidade de lidar com desbalanceamento de classes (através do parâmetro `scale_pos_weight`). Ele é treinado para identificar a probabilidade de um candidato avançar para a próxima fase do processo seletivo.

#### Avaliação e Métricas (Classification Report)

Durante o treinamento, o modelo é avaliado em um conjunto de dados de *holdout* para garantir sua generalização. As métricas de avaliação, incluindo ROC AUC, PR AUC e F1-Score, são calculadas e logadas no MLflow. Além disso, um `classification_report` detalhado é gerado, fornecendo insights sobre o desempenho do modelo para cada classe (candidato avança vs. candidato não avança).

**Exemplo de `classification_report` (valores ilustrativos):**

```
               precision    recall  f1-score   support

           0      0.417     0.696     0.521      1899
           1      0.802     0.558     0.659      4194

    accuracy                          0.601      6093
   macro avg      0.609     0.627     0.590      6093
weighted avg      0.682     0.601     0.616      6093

```

*   **Precision (Precisão)**: A proporção de identificações positivas que estavam corretas. Para a classe `1` (candidato avança), uma precisão de 0.80 significa que 808% dos candidatos classificados como "avançar" realmente avançaram.
*   **Recall (Revocação/Sensibilidade)**: A proporção de positivos reais que foram identificados corretamente. Para a classe `1`, um recall de 0.65 indica que o modelo identificou 65% de todos os candidatos que deveriam avançar.
*   **F1-Score**: A média harmônica da precisão e do recall. É uma métrica útil quando há um desequilíbrio de classes, fornecendo um equilíbrio entre precisão e recall.
*   **Support**: O número de ocorrências reais de cada classe no conjunto de teste.

Essas métricas são cruciais para entender o trade-off entre identificar corretamente os candidatos que avançam (recall) e evitar falsos positivos (precision), sendo o F1-Score um bom indicador geral do desempenho do modelo para a classe minoritária (candidatos que avançam).

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

- **Cobertura total:** 85%  

Comando utilizado:
```bash
  pytest --cov=app --cov=src
```

## Limitações & Próximos Passos 

    * Texto: TF-IDF é bag-of-words; próximo passo razoável: embeddings (SBERT) e re-rankers.

    * Calibração: avaliar Brier e calibradores (Platt/Isotonic) para thresholds estáveis.

    * Fairness/PII: auditoria sistemática de vieses e reforço de controles.

    * MLOps: model registry, shadow deploy, alertas automáticos de drift.
