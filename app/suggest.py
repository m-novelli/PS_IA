# app/suggest.py
from __future__ import annotations
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import os, json, re

# OpenAI SDK
from openai import OpenAI

# Limites simples para evitar textos gigantes
MAX_JOB_CHARS = int(os.getenv("SUGGEST_MAX_JOB_CHARS", "2000"))
MAX_CV_CHARS  = int(os.getenv("SUGGEST_MAX_CV_CHARS",  "2000"))
OPENAI_MODEL  = os.getenv("SUGGEST_MODEL", "gpt-4o-mini")


class SuggestQuestionsRequest(BaseModel):
    vaga: Dict[str, str] = Field(..., description="Campos livres: descricao, requisitos, atividades, etc.")
    candidatos: List[Dict[str, str]] = Field(..., description="Lista de dicts com 'external_id' e 'cv' (texto).")


class SuggestQuestionsResponse(BaseModel):
    common_questions: List[str]
    per_candidate: Dict[str, List[str]]


def _cut(s: str | None, n: int) -> str:
    if not s:
        return ""
    s = str(s)
    if len(s) <= n:
        return s
    return s[:n] + "..."


def _build_prompt(vaga: Dict[str, str], candidatos: List[Dict[str, str]]) -> str:
    desc = _cut(vaga.get("descricao") or vaga.get("perfil_vaga.principais_atividades") or "", MAX_JOB_CHARS)
    reqs = _cut(vaga.get("requisitos") or vaga.get("perfil_vaga.competencia_tecnicas_e_comportamentais") or "", MAX_JOB_CHARS)
    ativ = _cut(vaga.get("atividades") or "", MAX_JOB_CHARS)

    lines = []
    lines.append("Você é um recrutador técnico. Gere EXATAMENTE um JSON válido conforme o schema abaixo.")
    lines.append("Crie 5 perguntas comuns (iguais para todos) e 3 perguntas personalizadas por candidato.")
    lines.append("Perguntas devem ser objetivas, avaliando fit técnico e comportamental.")
    lines.append("")
    lines.append("SCHEMA:")
    lines.append('{"common_questions": ["...", "...", "...", "...", "..."], "per_candidate": {"<external_id>": ["...", "...", "..."]}}')
    lines.append("")
    lines.append("VAGA:")
    lines.append(f"Descricao: {desc}")
    lines.append(f"Requisitos: {reqs}")
    if ativ:
        lines.append(f"Atividades: {ativ}")
    lines.append("")
    lines.append("CANDIDATOS:")
    for c in candidatos:
        eid = str(c.get("external_id", "sem_id"))
        cv  = _cut(c.get("cv", ""), MAX_CV_CHARS)
        lines.append(f"- id={eid}: {cv}")
    lines.append("")
    lines.append("Responda apenas com o JSON solicitado. Não inclua explicações adicionais.")
    return "\n".join(lines)


def _extract_json(s: str) -> str:
    """
    Se o modelo devolver texto extra, tenta extrair o maior bloco JSON.
    """
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    # pega o primeiro { ... } balanceado (na prática, heurístico)
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        return m.group(0)
    return s  # deixa falhar no json.loads para o caller tratar


def suggest_questions(req: SuggestQuestionsRequest) -> SuggestQuestionsResponse:
    testing = os.environ.get("TESTING") == "1"
    api_key = os.environ.get("OPENAI_API_KEY")

    # modo offline para testes
    if testing and not api_key:
        return SuggestQuestionsResponse(
            common_questions=[
                "Pergunta simulada comum 1",
                "Pergunta simulada comum 2",
                "Pergunta simulada comum 3",
                "Pergunta simulada comum 4",
                "Pergunta simulada comum 5",
            ],
            per_candidate={
                str(c.get("external_id", "sem_id")): [
                    f"Pergunta simulada personalizada 1 para {c.get('external_id', 'sem_id')}",
                    f"Pergunta simulada personalizada 2 para {c.get('external_id', 'sem_id')}",
                    f"Pergunta simulada personalizada 3 para {c.get('external_id', 'sem_id')}",
                ]
                for c in req.candidatos
            }
        )

    # caminho normal (produção)
    client = OpenAI(api_key=api_key)
    prompt = _build_prompt(req.vaga, req.candidatos)

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
    )
    content = resp.choices[0].message.content or "{}"
    raw_json = _extract_json(content)

    # >>> Fallback robusto para JSON inválido <<<
    try:
        data = json.loads(raw_json)
    except Exception:
        # resposta mínima válida para não quebrar testes
        empty = {
            "common_questions": [],
            "per_candidate": {str(c.get("external_id", "sem_id")): [] for c in req.candidatos},
        }
        return SuggestQuestionsResponse(**empty)

    # valida e normaliza
    out = SuggestQuestionsResponse(**data)

    # enforce contagem: 5 comuns, 3 por candidato
    out.common_questions = list(out.common_questions)[:5]
    norm = {}
    for c in req.candidatos:
        eid = str(c.get("external_id", "sem_id"))
        qs = out.per_candidate.get(eid) or out.per_candidate.get(str(eid)) or []
        norm[eid] = list(qs)[:3]
    out.per_candidate = norm

    return out
