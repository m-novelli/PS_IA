# Mapeamento de Chaves para Merge - Dados de Recrutamento

## 📋 Resumo da Análise

Com base na inspeção dos arquivos JSON, foi identificada a estrutura e as chaves necessárias para realizar o merge entre os dados de vagas, prospects e candidatos.

## 📁 Estrutura dos Arquivos

### 1. `vagas.json`
- **Estrutura**: `dict[vaga_id → dados_da_vaga]`
- **Total de registros**: 14.081 vagas
- **Chave principal**: ID da vaga (string numérica, ex: "5185", "5184")
- **Campos principais**:
  - `informacoes_basicas`: dados básicos da vaga
  - `perfil_vaga`: requisitos e localização
  - `beneficios`: informações de valores

### 2. `prospects.json` 
- **Estrutura**: `dict[vaga_id → {"titulo": str, "modalidade": str, "prospects": [lista_candidatos]}]`
- **Total de registros**: 14.222 vagas
- **Chave principal**: ID da vaga (string numérica, ex: "4530", "4531")
- **Estrutura interna dos prospects**:
  ```json
  {
    "4530": {
      "titulo": "CONSULTOR CONTROL M",
      "modalidade": "",
      "prospects": [
        {
          "nome": "José Vieira",
          "codigo": "25632",  // ← CHAVE PARA CANDIDATOS
          "situacao_candidado": "Encaminhado ao Requisitante",
          "data_candidatura": "25-03-2021",
          "ultima_atualizacao": "25-03-2021",
          "comentario": "...",
          "recrutador": "Ana Lívia Moreira"
        }
      ]
    }
  }
  ```

### 3. `applicants.json`
- **Estrutura**: `dict[candidato_id → dados_do_candidato]`
- **Tamanho**: >200MB (arquivo muito grande)
- **Chave principal**: ID do candidato (string numérica, ex: "31000", "31001")
- **Campos principais**: 
  - `infos_basicas`: dados pessoais e contato
  - `codigo_profissional`: ID do candidato

## 🔗 Estratégia de Merge

### Chaves Identificadas
1. **Chave Primária**: `vaga_id`
   - Conecta `vagas.json` ↔ `prospects.json`
   - Formato: string numérica

2. **Chave Secundária**: `candidato_id` / `codigo`
   - Conecta `prospects.json` ↔ `applicants.json`
   - No prospects: campo `codigo` dentro de cada item da lista
   - No applicants: chave principal do dicionário

### Fluxo de Merge
```
prospects (base) → vagas (LEFT JOIN por vaga_id) → applicants (LEFT JOIN por candidato_id)
```

### Processo Detalhado
1. **Explosão dos Prospects**: 
   - Transformar a estrutura aninhada em linhas individuais
   - Cada candidato de cada vaga vira uma linha

2. **Merge com Vagas**:
   - `LEFT JOIN` usando `vaga_id`
   - Preserva todos os prospects, mesmo se vaga não existir

3. **Merge com Candidatos**:
   - `LEFT JOIN` usando `candidato_id` (código do prospect)
   - Preserva todos os prospects, mesmo se candidato não existir

## ⚠️ Observações Importantes

### Diferenças de Range de IDs
- **Vagas**: IDs na faixa de 5000+ (ex: 5185, 5184)
- **Prospects**: IDs na faixa de 4000+ (ex: 4530, 4531)
- **Candidatos**: IDs na faixa de 25000+ e 31000+ (ex: 25632, 31000)

### Possíveis Problemas
1. **IDs de vagas diferentes**: vagas.json e prospects.json têm ranges diferentes
2. **Vagas órfãs**: vagas sem prospects correspondentes
3. **Prospects órfãos**: prospects sem vagas correspondentes
4. **Candidatos órfãos**: prospects referenciando candidatos inexistentes

## 💾 Implementação no Código

O script `merge_recrutamento.py` já implementa esta estratégia:

```python
# 1. Explosão dos prospects
prospects_records = coerce_prospects_anyshape(prospects_raw, applicant_keyset)

# 2. Merge com vagas
df = df_prospects.merge(df_jobs, on="vaga_id", how="left", validate="m:1")

# 3. Merge com candidatos  
df = df.merge(df_applicants, on="candidato_id", how="left", validate="m:1")
```

## 🎯 Resultado Final

O dataset final terá:
- **Uma linha por prospect** (candidato aplicado para uma vaga)
- **Todas as informações da vaga** (prefixo `vaga__`)
- **Todas as informações do candidato** (prefixo `cand__`)
- **Informações do processo de candidatura** (prefixo `prospect__`)

## 📊 Próximos Passos

1. ✅ Executar o merge com o script existente
2. 🔍 Verificar qualidade dos dados resultantes
3. 📈 Analisar cobertura (% de matches entre as tabelas)
4. 🧹 Limpar dados inconsistentes se necessário
