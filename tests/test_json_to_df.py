import pandas as pd
import pytest
import json
from src.preprocess import json_to_df  # importa do wrapper novo


def test_load_dict_json_flat(tmp_path):
    data = {"123": {"nome": "Carlos"}, "456": {"nome": "Ana"}}
    file = tmp_path / "data.json"
    file.write_text(json.dumps(data), encoding="utf-8")

    df = json_to_df.load_dict_json_flat(file, "codigo")
    assert isinstance(df, pd.DataFrame)
    assert "codigo" in df.columns
    assert len(df) == 2


def test_load_prospects(tmp_path):
    data = {
        "vaga01": {
            "titulo": "Cientista de Dados",
            "modalidade": "Remoto",
            "prospects": [
                {"codigo": "cand1", "nome": "Carlos", "situacao_candidado": "ativo"},
                {"codigo": "cand2", "nome": "Ana", "situacao_candidado": "inativo"}
            ]
        }
    }
    file = tmp_path / "prospects.json"
    file.write_text(json.dumps(data), encoding="utf-8")

    df = json_to_df.load_prospects(file)
    assert len(df) == 2
    assert "codigo_applicant" in df.columns
    assert df["titulo_vaga"].iloc[0] == "Cientista de Dados"