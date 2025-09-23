def test_rank_and_suggest(client):
    payload = {
        "vaga": {
            "titulo": "Analista de Dados",
            "descricao": "Responsável por análises e relatórios."
        },
        "candidatos": [
            {
                "meta": {"external_id": "cand1"},
                "candidato": {"nome": "Carlos", "idade": "30", "sexo": "M"}  # idade como string
            },
            {
                "meta": {"external_id": "cand2"},
                "candidato": {"nome": "Ana", "idade": "25", "sexo": "F"}  # idade como string
            }
        ]
    }

    r = client.post("/rank-and-suggest", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "results" in body
    assert len(body["results"]) == 2


def test_rank_and_suggest_no_candidates(client):
    payload = {
        "vaga": {
            "titulo": "Analista de Dados",
            "descricao": "Responsável por análises e relatórios."
        },
        "candidatos": []  # vazio de propósito
    }

    r = client.post("/rank-and-suggest", json=payload)
    assert r.status_code == 400
    body = r.json()
    assert "detail" in body
    assert "nenhum candidato" in body["detail"].lower()