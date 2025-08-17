# Guia dos Datasets Otimizados

## 📋 Visão Geral

A partir do dataset original (`dataset_final.csv`), foram criadas **3 versões otimizadas** para diferentes propósitos, cada uma com características específicas:

| Dataset | Registros | Colunas | Tamanho (CSV) | Tamanho (Pickle) | Propósito |
|---------|-----------|---------|---------------|------------------|-----------|
| **full_optimized** | 53.759 | 111 | 326 MB | 121 MB | Dados completos otimizados |
| **exploratory** | 53.759 | 23 | 16 MB | 5 MB | Análise exploratória |
| **matching** | 53.759 | 23 | 8 MB | 2 MB | Modelos de matching |

## 🎯 1. Dataset FULL_OPTIMIZED

### **Propósito**: Versão completa com todos os dados + otimização de tipos
### **Quando usar**: Análises completas, backup otimizado, processamento geral

**Otimizações aplicadas:**
- ✅ **Redução de 70.6% no uso de memória** (776 MB → 228 MB)
- ✅ Conversão de strings repetitivas para `category`
- ✅ Downcast de inteiros e floats
- ✅ Otimização de tipos de dados

**Colunas incluídas:**
- **Todas as 111 colunas originais**
- 7 colunas de prospects
- 44 colunas de vagas  
- 57 colunas de candidatos
- 3 colunas de controle (vaga_id, candidato_id, status)

## 📊 2. Dataset EXPLORATORY

### **Propósito**: Análise exploratória, dashboards, relatórios
### **Quando usar**: EDA, visualizações, análises de negócio

**Colunas selecionadas (23 total):**

### **Identificação & Controle**
- `vaga_id`, `candidato_id`, `status`

### **Processo de Candidatura**
- `prospect__nome` - Nome do candidato
- `prospect__situacao_candidado` - Status da candidatura
- `prospect__data_candidatura` - Data da aplicação
- `prospect__recrutador` - Responsável pelo processo

### **Informações da Vaga**
- `vaga__informacoes_basicas_titulo_vaga` - Título da posição
- `vaga__informacoes_basicas_cliente` - Cliente/empresa
- `vaga__informacoes_basicas_tipo_contratacao` - CLT, PJ, etc.
- `vaga__perfil_vaga_pais` - País da vaga
- `vaga__perfil_vaga_estado` - Estado
- `vaga__perfil_vaga_cidade` - Cidade
- `vaga__perfil_vaga_nivel profissional` - Júnior, Pleno, Sênior
- `vaga__perfil_vaga_nivel_academico` - Escolaridade exigida
- `vaga__perfil_vaga_areas_atuacao` - Área técnica

### **Informações do Candidato**
- `cand__infos_basicas_nome` - Nome completo
- `cand__infos_basicas_email` - Email de contato
- `cand__infos_basicas_telefone` - Telefone
- `cand__infos_basicas_data_criacao` - Quando foi cadastrado

### **Campos Derivados** (criados automaticamente)
- `data_candidatura_parsed` - Data em formato datetime
- `ano_candidatura` - Ano da candidatura
- `mes_candidatura` - Mês da candidatura

## 🤖 3. Dataset MATCHING

### **Propósito**: Modelos de Machine Learning, algoritmos de matching
### **Quando usar**: Treinar modelos, prever compatibilidade vaga-candidato

**Features principais (23 colunas):**

### **Target Variable**
- `match_success` - **Taxa de sucesso: 17.14%**
  - 1 = Match bem-sucedido (contratado, aprovado, selecionado)
  - 0 = Match não bem-sucedido

### **Features de Vaga (Requisitos)**
- `vaga__perfil_vaga_nivel profissional` - Senioridade exigida
- `vaga__perfil_vaga_nivel_academico` - Escolaridade
- `vaga__perfil_vaga_nivel_ingles` - Nível de inglês
- `vaga__perfil_vaga_nivel_espanhol` - Nível de espanhol
- `vaga__perfil_vaga_areas_atuacao` - Área técnica
- `vaga__perfil_vaga_pais` - Localização
- `vaga__perfil_vaga_estado` - Estado
- `vaga__perfil_vaga_cidade` - Cidade

### **Features de Candidato (Perfil)**
- Campos relacionados a: formação, experiência, habilidades, competências, idiomas, certificações

## 📁 Formatos Disponíveis

### **CSV** 
- ✅ **Vantagens**: Universal, legível, compatível com Excel
- ❌ **Desvantagens**: Grande, lento para carregar
- 🎯 **Melhor para**: Intercâmbio de dados, análises em Excel

### **Pickle**
- ✅ **Vantagens**: Muito rápido, preserva tipos de dados Python, compacto
- ❌ **Desvantagens**: Específico do Python, não legível
- 🎯 **Melhor para**: Processamento em Python, modelos ML

### **Parquet** (requer instalação: `pip install pyarrow`)
- ✅ **Vantagens**: Colunar, muito compacto, rápido, multi-linguagem
- ❌ **Desvantagens**: Requer biblioteca adicional
- 🎯 **Melhor para**: Análises grandes, performance

### **Feather** (requer instalação: `pip install pyarrow`)
- ✅ **Vantagens**: Rápido, cross-language, boa compressão
- ❌ **Desvantagens**: Requer biblioteca adicional
- 🎯 **Melhor para**: Intercâmbio entre R/Python

## 🚀 Recomendações de Uso

| Caso de Uso | Dataset Recomendado | Formato |
|-------------|-------------------|---------|
| **📈 Análise Exploratória** | `exploratory` | `.pickle` ou `.csv` |
| **🤖 Modelos de ML** | `matching` | `.pickle` |
| **📊 Dashboards/BI** | `exploratory` | `.csv` |
| **🔄 Compartilhamento** | `full_optimized` | `.csv` |
| **⚡ Performance máxima** | Qualquer | `.parquet` |
| **🐍 Apenas Python** | Qualquer | `.pickle` |

## 💡 Exemplos de Uso

### **Análise Exploratória**
```python
import pandas as pd

# Carrega dataset otimizado para EDA
df = pd.read_pickle('data/processed/exploratory.pickle')

# Análises rápidas
print(f"Total de candidaturas: {len(df)}")
print(f"Taxa de sucesso por área: {df.groupby('vaga__perfil_vaga_areas_atuacao')['status'].value_counts()}")
print(f"Candidaturas por mês: {df['mes_candidatura'].value_counts()}")
```

### **Modelo de Matching**
```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Carrega dataset para ML
df = pd.read_pickle('data/processed/matching.pickle')

# Features e target
X = df.drop(['match_success', 'vaga_id', 'candidato_id'], axis=1)
y = df['match_success']

# Split para treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## 📦 Próximos Passos

Para melhor performance, considere instalar:
```bash
pip install pyarrow fastparquet
```

Isso habilitará formatos **Parquet** e **Feather**, que são ainda mais eficientes.
