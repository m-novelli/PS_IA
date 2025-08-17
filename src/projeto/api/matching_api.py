#!/usr/bin/env python3
"""
API para Modelo de Matching Candidato-Vaga

Fornece endpoints para predizer compatibilidade entre candidatos e vagas.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import json

# Adicionar caminho do projeto
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from src.projeto.models.matching_model import MatchingModel, MatchingConfig
    from src.projeto.core.config import MODELS_DIR
except ImportError as e:
    print(f"Erro ao importar módulos: {e}")
    sys.exit(1)


class MatchingAPI:
    """API para modelo de matching."""
    
    def __init__(self, model_path: Path = None):
        self.model = None
        self.model_path = model_path or (MODELS_DIR / "matching_model.pkl")
        self.load_model()
    
    def load_model(self):
        """Carrega modelo treinado."""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Modelo não encontrado: {self.model_path}")
            
            self.model = MatchingModel()
            self.model.load_model(self.model_path)
            print(f"✅ Modelo carregado com sucesso!")
            
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            self.model = None
    
    def predict_single_match(self, vaga_data: Dict[str, Any], candidato_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prediz match para uma combinação vaga-candidato.
        
        Args:
            vaga_data: Dados da vaga
            candidato_data: Dados do candidato
            
        Returns:
            Predição com probabilidade e recomendação
        """
        if self.model is None:
            return {"error": "Modelo não está carregado"}
        
        try:
            result = self.model.predict_match(vaga_data, candidato_data)
            
            # Adicionar informações extras
            result.update({
                "timestamp": pd.Timestamp.now().isoformat(),
                "model_version": "1.0",
                "status": "success"
            })
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "status": "error"
            }
    
    def predict_multiple_matches(self, vaga_data: Dict[str, Any], candidatos_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prediz matches para uma vaga com múltiplos candidatos.
        
        Args:
            vaga_data: Dados da vaga
            candidatos_list: Lista de dados de candidatos
            
        Returns:
            Lista de predições ordenada por probabilidade
        """
        if self.model is None:
            return [{"error": "Modelo não está carregado"}]
        
        results = []
        
        for i, candidato_data in enumerate(candidatos_list):
            try:
                prediction = self.model.predict_match(vaga_data, candidato_data)
                prediction.update({
                    "candidato_index": i,
                    "candidato_id": candidato_data.get("candidato_id", f"candidato_{i}")
                })
                results.append(prediction)
                
            except Exception as e:
                results.append({
                    "candidato_index": i,
                    "candidato_id": candidato_data.get("candidato_id", f"candidato_{i}"),
                    "error": str(e),
                    "status": "error"
                })
        
        # Ordenar por probabilidade (decrescente)
        results_sorted = sorted(
            [r for r in results if "match_probability" in r],
            key=lambda x: x["match_probability"],
            reverse=True
        )
        
        # Adicionar ranking
        for rank, result in enumerate(results_sorted, 1):
            result["ranking"] = rank
        
        return results_sorted
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o modelo."""
        if self.model is None:
            return {"error": "Modelo não está carregado"}
        
        info = {
            "model_type": type(self.model.best_model).__name__,
            "model_version": "1.0",
            "features_count": len(self.model.preprocessor.categorical_features + 
                                 self.model.preprocessor.numerical_features + 
                                 self.model.preprocessor.text_features),
            "categorical_features": len(self.model.preprocessor.categorical_features),
            "numerical_features": len(self.model.preprocessor.numerical_features),
            "text_features": len(self.model.preprocessor.text_features),
            "threshold": self.model.config.match_threshold,
            "status": "loaded"
        }
        
        # Adicionar métricas se disponíveis
        if self.model.metrics:
            info.update({
                "performance": {
                    "roc_auc": self.model.metrics.get("roc_auc"),
                    "avg_precision": self.model.metrics.get("avg_precision")
                }
            })
        
        return info
    
    def get_top_features(self, n: int = 10) -> Dict[str, Any]:
        """Retorna top features mais importantes."""
        if self.model is None or self.model.feature_importance is None:
            return {"error": "Feature importance não disponível"}
        
        top_features = self.model.feature_importance.head(n)
        
        return {
            "top_features": [
                {
                    "feature": row["feature"],
                    "importance": float(row["importance"]),
                    "rank": idx + 1
                }
                for idx, (_, row) in enumerate(top_features.iterrows())
            ],
            "total_features": len(self.model.feature_importance)
        }


def create_sample_data():
    """Cria dados de exemplo para teste."""
    
    sample_vaga = {
        "vaga__perfil_vaga_nivel profissional": "Sênior",
        "vaga__perfil_vaga_nivel_academico": "Ensino Superior Completo",
        "vaga__perfil_vaga_nivel_ingles": "Avançado",
        "vaga__perfil_vaga_nivel_espanhol": "Básico",
        "vaga__perfil_vaga_areas_atuacao": "TI - Desenvolvimento",
        "vaga__perfil_vaga_pais": "Brasil",
        "vaga__perfil_vaga_estado": "São Paulo",
        "vaga__perfil_vaga_cidade": "São Paulo"
    }
    
    sample_candidatos = [
        {
            "candidato_id": "cand_001",
            "cand__formacao_e_idiomas_nivel_academico": "Ensino Superior Completo",
            "cand__formacao_e_idiomas_nivel_ingles": "Avançado",
            "cand__informacoes_profissionais_nivel_profissional": "Sênior",
            "cand__informacoes_profissionais_area_atuacao": "TI - Desenvolvimento"
        },
        {
            "candidato_id": "cand_002", 
            "cand__formacao_e_idiomas_nivel_academico": "Ensino Médio Completo",
            "cand__formacao_e_idiomas_nivel_ingles": "Básico",
            "cand__informacoes_profissionais_nivel_profissional": "Júnior",
            "cand__informacoes_profissionais_area_atuacao": "TI - Suporte"
        }
    ]
    
    return sample_vaga, sample_candidatos


def test_api():
    """Testa a API com dados de exemplo."""
    
    print("🧪 TESTANDO API DO MODELO DE MATCHING")
    print("=" * 50)
    
    # Inicializar API
    api = MatchingAPI()
    
    if api.model is None:
        print("❌ Não foi possível carregar o modelo para teste")
        return
    
    # Testar informações do modelo
    print(f"\n📋 Informações do Modelo:")
    model_info = api.get_model_info()
    print(json.dumps(model_info, indent=2, ensure_ascii=False))
    
    # Testar top features
    print(f"\n⭐ Top 5 Features:")
    top_features = api.get_top_features(5)
    if "top_features" in top_features:
        for feature in top_features["top_features"]:
            print(f"   {feature['rank']}. {feature['feature']}: {feature['importance']:.3f}")
    
    # Testar predição simples
    print(f"\n🔍 Teste de Predição Simples:")
    sample_vaga, sample_candidatos = create_sample_data()
    
    result = api.predict_single_match(sample_vaga, sample_candidatos[0])
    print(f"Resultado: {json.dumps(result, indent=2, ensure_ascii=False)}")
    
    # Testar predições múltiplas
    print(f"\n🎯 Teste de Predições Múltiplas:")
    results = api.predict_multiple_matches(sample_vaga, sample_candidatos)
    
    for result in results:
        if "error" not in result:
            print(f"   Candidato {result['candidato_id']}: "
                  f"Probabilidade {result['match_probability']:.3f} "
                  f"(Ranking #{result['ranking']})")


def main():
    """Função principal para testar a API."""
    test_api()


if __name__ == "__main__":
    import pandas as pd
    main()

