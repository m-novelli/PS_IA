#!/usr/bin/env python3
"""
Script para inspecionar as chaves e estrutura dos arquivos JSON
sem carregar tudo na memória.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

# Importa configurações
try:
    from src.projeto.core.config import (
        FILE_JOBS,
        FILE_PROSPECTS, 
        FILE_APPLICANTS
    )
except ImportError as e:
    print(f"Erro ao importar config: {e}")
    print("Certifique-se que está executando do diretório correto (raiz do projeto).")
    sys.exit(1)


def analyze_json_structure(filepath: Path, max_samples: int = 5):
    """Analisa a estrutura de um arquivo JSON sem carregar tudo na memória."""
    print(f"\n{'='*60}")
    print(f"ANALISANDO: {filepath.name}")
    print(f"{'='*60}")
    
    if not filepath.exists():
        print(f"❌ Arquivo não encontrado: {filepath}")
        return
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📊 Tipo principal: {type(data).__name__}")
        
        if isinstance(data, dict):
            print(f"📋 Total de chaves: {len(data)}")
            print(f"🔑 Primeiras chaves: {list(data.keys())[:10]}")
            
            # Analisa algumas amostras
            sample_keys = list(data.keys())[:max_samples]
            print(f"\n🔍 Analisando {len(sample_keys)} amostras:")
            
            for i, key in enumerate(sample_keys, 1):
                value = data[key]
                print(f"\n  📍 Amostra {i} - Chave: '{key}'")
                print(f"     Tipo do valor: {type(value).__name__}")
                
                if isinstance(value, dict):
                    print(f"     Sub-chaves ({len(value)}): {list(value.keys())}")
                    
                    # Se tem sub-estruturas, analisa uma amostra
                    for sub_key, sub_value in list(value.items())[:3]:
                        print(f"       • {sub_key}: {type(sub_value).__name__}")
                        if isinstance(sub_value, dict) and len(sub_value) <= 10:
                            print(f"         └─ {list(sub_value.keys())}")
                        elif isinstance(sub_value, list) and len(sub_value) > 0:
                            print(f"         └─ Lista com {len(sub_value)} items, primeiro item: {type(sub_value[0]).__name__}")
                            if isinstance(sub_value[0], dict):
                                print(f"            └─ Chaves do primeiro item: {list(sub_value[0].keys())}")
                
                elif isinstance(value, list):
                    print(f"     Lista com {len(value)} items")
                    if len(value) > 0:
                        first_item = value[0]
                        print(f"     Tipo do primeiro item: {type(first_item).__name__}")
                        if isinstance(first_item, dict):
                            print(f"     Chaves do primeiro item: {list(first_item.keys())}")
                
                else:
                    print(f"     Valor: {str(value)[:100]}...")
        
        elif isinstance(data, list):
            print(f"📋 Total de items: {len(data)}")
            if len(data) > 0:
                first_item = data[0]
                print(f"🔑 Tipo do primeiro item: {type(first_item).__name__}")
                if isinstance(first_item, dict):
                    print(f"🔑 Chaves do primeiro item: {list(first_item.keys())}")
        
        else:
            print(f"⚠️  Tipo não esperado: {type(data)}")
            
    except json.JSONDecodeError as e:
        print(f"❌ Erro ao decodificar JSON: {e}")
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")


def analyze_applicants_sample(filepath: Path, sample_size: int = 1000):
    """Analisa uma amostra do arquivo de candidatos (muito grande)."""
    print(f"\n{'='*60}")
    print(f"ANALISANDO AMOSTRA: {filepath.name}")
    print(f"{'='*60}")
    
    if not filepath.exists():
        print(f"❌ Arquivo não encontrado: {filepath}")
        return
    
    try:
        # Lê apenas o início do arquivo para entender a estrutura
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read(10000)  # Primeiros 10KB
        
        print("📄 Primeiros caracteres do arquivo:")
        print(content[:500] + "..." if len(content) > 500 else content)
        
        # Tenta identificar o padrão das chaves
        if content.strip().startswith('{'):
            print("\n🔍 Detectado formato de dicionário")
            # Procura por padrões de chaves
            import re
            key_pattern = r'"(\d+)"\s*:'
            keys_found = re.findall(key_pattern, content)
            if keys_found:
                print(f"🔑 Padrão de chaves detectado: {keys_found[:10]}")
                print(f"📊 Chaves são numéricas: {all(k.isdigit() for k in keys_found)}")
        
        # Tenta carregar apenas uma pequena parte
        try:
            # Carrega limitado
            with open(filepath, 'r', encoding='utf-8') as f:
                chunk = f.read(50000)  # 50KB
                # Tenta encontrar uma entrada completa
                if '}: {' in chunk:
                    # Pega até a primeira entrada completa
                    end_pos = chunk.find('}: {') + 1
                    partial_json = chunk[:end_pos] + '}'
                    
                    try:
                        partial_data = json.loads(partial_json)
                        if isinstance(partial_data, dict):
                            first_key = list(partial_data.keys())[0]
                            first_value = partial_data[first_key]
                            print(f"\n📋 Primeira entrada encontrada:")
                            print(f"   Chave: '{first_key}'")
                            print(f"   Tipo do valor: {type(first_value).__name__}")
                            if isinstance(first_value, dict):
                                print(f"   Sub-chaves: {list(first_value.keys())}")
                    except:
                        pass
        except:
            pass
            
    except Exception as e:
        print(f"❌ Erro ao analisar amostra: {e}")


def identify_merge_keys():
    """Identifica as possíveis chaves para merge baseado na análise."""
    print(f"\n{'='*60}")
    print("🔗 IDENTIFICAÇÃO DE CHAVES PARA MERGE")
    print(f"{'='*60}")
    
    print("""
Com base na análise dos arquivos, as chaves identificadas são:

📁 VAGAS.JSON:
   • Estrutura: dict[vaga_id -> dados_da_vaga]
   • Chave principal: ID da vaga (ex: "5185", "5184")
   • Campo interno 'codigo': não identificado diretamente
   
📁 PROSPECTS.JSON:
   • Estrutura: dict[vaga_id -> {"prospects": [lista_de_candidatos]}]
   • Chave principal: ID da vaga (ex: "4530", "4531") 
   • Dentro de cada prospect:
     - "codigo": ID do candidato (ex: "25632", "25529")
     - Outros campos: nome, situacao_candidado, data_candidatura, etc.

📁 APPLICANTS.JSON:
   • Estrutura: dict[candidato_id -> dados_do_candidato]
   • Chave principal: ID do candidato (numérico como string)

🔗 ESTRATÉGIA DE MERGE:
   1. Chave primária: vaga_id (conecta vagas ↔ prospects)
   2. Chave secundária: candidato_id/codigo (conecta prospects ↔ applicants)
   
   FLUXO: prospects → vagas (por vaga_id) → applicants (por candidato_id)
""")


def main():
    """Função principal."""
    print("🔍 INSPEÇÃO DE ESTRUTURA DOS ARQUIVOS JSON")
    print("=" * 60)
    
    # Analisa cada arquivo
    analyze_json_structure(FILE_JOBS, max_samples=3)
    analyze_json_structure(FILE_PROSPECTS, max_samples=3)
    
    # Para applicants, usa análise especial devido ao tamanho
    analyze_applicants_sample(FILE_APPLICANTS)
    
    # Sumário das chaves para merge
    identify_merge_keys()
    
    print(f"\n✅ Análise concluída!")
    print("\n💡 PRÓXIMOS PASSOS:")
    print("   1. Usar vaga_id como chave principal entre vagas e prospects")
    print("   2. Usar candidato_id/codigo como chave entre prospects e applicants")
    print("   3. Verificar se os IDs são consistentes entre os arquivos")


if __name__ == "__main__":
    main()
