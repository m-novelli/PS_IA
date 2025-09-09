#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import logging
import re
import unicodedata

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# ===== Stopwords PT com fallback =====
import nltk
from nltk.corpus import stopwords

# utilitário de normalização antes das stopwords
_norm_re = re.compile(r"[^a-z\s]")

def normalize_text(s: str) -> str:
    """Minimiza e remove acentuação e pontuação; mantém apenas [a-z espaços]."""
    s = str(s).lower().strip()
    s = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("utf-8")
    s = _norm_re.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

try:
    _raw_stop = stopwords.words("portuguese")
except LookupError:
    nltk.download("stopwords")
    _raw_stop = stopwords.words("portuguese")

# normaliza stopwords (sem acento) e converte em lista
STOPWORDS_PT = list(sorted({normalize_text(w) for w in _raw_stop if normalize_text(w)}))
STOPWORDS_PT_SET = set(STOPWORDS_PT)

# ================================
# Caminhos default
# ================================
BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = BASE_DIR / "data" / "interim" / "dataset_triagem_clean.csv"
DEFAULT_OUTPUT = BASE_DIR / "data" / "processed" / "dataset_triagem_fe.csv"
FEATURE_MAP_PATH = BASE_DIR / "data" / "processed" / "feature_map.json"

# ================================
# Logging
# ================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

# ================================
# Utilitários de texto
# ================================
def tokenize_no_stopwords(s: str) -> list:
    """Normaliza e remove stopwords, retorna lista de tokens."""
    toks = normalize_text(s).split()
    return [t for t in toks if t and t not in STOPWORDS_PT_SET]

def jaccard_overlap(text1: str, text2: str) -> float:
    """Sobreposição Jaccard entre dois textos, ignorando stopwords."""
    set1 = set(tokenize_no_stopwords(text1))
    set2 = set(tokenize_no_stopwords(text2))
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

# ================================
# Núcleo de FE
# ================================
REQUIRED_FE_COLS = [
    "perfil_vaga.nivel_academico",
    "formacao_e_idiomas.nivel_academico",
    "perfil_vaga.competencia_tecnicas_e_comportamentais",
    "informacoes_profissionais.conhecimentos_tecnicos",
    "cv_pt",
    "perfil_vaga.principais_atividades",
]

TARGET_PRIORITY = ["target_triagem", "status_simplificado", "target_contratacao"]
GROUP_FALLBACK = ["codigo_vaga", "codigo_applicant"]

def load_feature_map(path: Path) -> dict:
    if not path.exists():
        log.warning("feature_map.json não encontrado em %s — seguindo com fallback.", path)
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Erro ao ler feature_map.json: {e}")

def resolve_groups_and_target(df: pd.DataFrame, fmap: dict) -> tuple[list, str]:
    group_cols = [c for c in fmap.get("group", GROUP_FALLBACK) if c in df.columns]
    target_candidates = [c for c in fmap.get("target", TARGET_PRIORITY) if c in df.columns]

    if not target_candidates:
        raise ValueError(
            f"Nenhuma coluna de target encontrada no INPUT. "
            f"Esperado uma dessas: {TARGET_PRIORITY} "
        )
    target_main = next((c for c in TARGET_PRIORITY if c in target_candidates), target_candidates[0])

    if not group_cols:
        raise ValueError(
            "Nenhuma coluna de grupo encontrada (ex.: identificadores de vaga/candidato). "
            f"Esperado uma dessas: {GROUP_FALLBACK}"
        )
    return group_cols, target_main

def compute_features(
    df: pd.DataFrame,
    *,
    max_features: int = 2000,
    min_df: int | float = 3,
    max_df: float = 0.9,
    ngram_min: int = 1,
    ngram_max: int = 1
) -> pd.DataFrame:
    # 1) Match direto de nível acadêmico
    df["match_nivel_academico"] = (
        df["perfil_vaga.nivel_academico"].fillna("").str.lower().str.strip()
        == df["formacao_e_idiomas.nivel_academico"].fillna("").str.lower().str.strip()
    ).astype(int)

    # 2) Overlap de competências exigidas x oferecidas
    df["overlap_skills"] = df.apply(
        lambda row: jaccard_overlap(
            row.get("perfil_vaga.competencia_tecnicas_e_comportamentais", ""),
            row.get("informacoes_profissionais.conhecimentos_tecnicos", "")
        ),
        axis=1
    )

    # 3) Similaridade textual CV vs atividades da vaga
    cv_series   = df["cv_pt"].fillna("").astype(str)
    vaga_series = df["perfil_vaga.principais_atividades"].fillna("").astype(str)

    vectorizer = TfidfVectorizer(
        stop_words=STOPWORDS_PT,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=(ngram_min, ngram_max),
        lowercase=True,
        tokenizer=tokenize_no_stopwords,
        token_pattern=None
    )

    vectorizer.fit(pd.concat([cv_series, vaga_series], axis=0, ignore_index=True))
    X_cv   = vectorizer.transform(cv_series)
    X_vaga = vectorizer.transform(vaga_series)

    X_cv_n   = normalize(X_cv, norm="l2", copy=False)
    X_vaga_n = normalize(X_vaga, norm="l2", copy=False)

    df["sim_cv_vaga"] = (X_cv_n.multiply(X_vaga_n)).sum(axis=1).A1

    log.info(
        "Features criadas: match_nivel_academico, overlap_skills, sim_cv_vaga | Vocabulário TF-IDF: %d termos",
        len(vectorizer.vocabulary_)
    )
    return df

def run(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    *,
    max_features: int = 2000,
    min_df: int | float = 3,
    max_df: float = 0.9,
    ngram_min: int = 1,
    ngram_max: int = 1
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {input_path}")

    df = pd.read_csv(input_path)
    log.info("Base carregada: shape=%s de %s", df.shape, input_path)

    # Checagem de colunas obrigatórias
    missing_cols = [c for c in REQUIRED_FE_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing_cols}")

    fmap = load_feature_map(FEATURE_MAP_PATH)
    group_cols, target_main = resolve_groups_and_target(df, fmap)

    df = compute_features(
        df,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_min=ngram_min,
        ngram_max=ngram_max
    )

    # Ordem: grupos -> target -> resto -> novas features
    new_features = ["match_nivel_academico", "overlap_skills", "sim_cv_vaga"]
    ordered = [*group_cols, target_main] + [
        c for c in df.columns if c not in group_cols + [target_main] + new_features
    ] + new_features

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[ordered].to_csv(output_path, index=False)

    # --- feedback final ---
    print(f"[OK] Arquivo salvo: {output_path} | Shape final: {df[ordered].shape}")
    print(f"[OK] Grupo(s) no output: {group_cols} | Target principal: {target_main}")

# ================================
# Execução direta
# ================================
if __name__ == "__main__":
    run()
