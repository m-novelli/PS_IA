import pandas as pd
import pytest
import json
from src.preprocess import json_to_df  # importa do wrapper novo


def test_load_dict_json_flat_empty(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("{}", encoding="utf-8")
    df = json_to_df.load_dict_json_flat(path, "id")
    assert df.empty


def test_load_dict_json_flat_with_none(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"123": {"campo": None}}), encoding="utf-8")
    df = json_to_df.load_dict_json_flat(path, "codigo")
    assert "codigo" in df.columns
    assert pd.isna(df.loc[0, "campo"])


def test_load_prospects_no_prospects(tmp_path):
    data = {"vaga1": {"titulo": "Dev", "modalidade": "remoto"}}
    path = tmp_path / "prospects.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    df = json_to_df.load_prospects(path)
    assert df.empty


def test_load_prospects_with_empty_list(tmp_path):
    data = {"vaga1": {"titulo": "Dev", "modalidade": "remoto", "prospects": []}}
    path = tmp_path / "prospects.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    df = json_to_df.load_prospects(path)
    assert df.empty


def test_load_prospects_with_partial_data(tmp_path):
    data = {
        "vaga1": {
            "titulo": "Dev",
            "modalidade": "remoto",
            "prospects": [{"codigo": "c1", "nome": "Ana"}],  # campos faltando
        }
    }
    path = tmp_path / "prospects.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    df = json_to_df.load_prospects(path)

    assert df.loc[0, "codigo_vaga"] == "vaga1"
    assert df.loc[0, "nome_candidato"] == "Ana"
    # checa se os campos ausentes viram None
    assert pd.isna(df.loc[0, "comentario"])