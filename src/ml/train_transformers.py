from __future__ import annotations
from typing import Iterable, List, Optional
import re, unicodedata

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# ========= Normalização e stopwords (mesma lógica do seu FE) =========
_norm_re = re.compile(r"[^a-z\s]")

def normalize_text(s: str) -> str:
    s = str(s).lower().strip()
    s = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("utf-8")
    s = _norm_re.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _build_stopwords_pt() -> List[str]:
    try:
        import nltk
        from nltk.corpus import stopwords
        try:
            raw = stopwords.words("portuguese")
        except LookupError:
            nltk.download("stopwords")
            raw = stopwords.words("portuguese")
    except Exception:
        # fallback básico
        raw = ["de","a","o","que","e","do","da","em","um","para","é","com","não","uma","os","no","se","na",
               "por","mais","as","dos","como","mas","foi","ao","ele","das","tem","à","seu","sua","ou","ser",
               "quando","muito","há","nos","já","está","eu","também","só","pelo","pela","até","isso","ela",
               "entre","era","depois","sem","mesmo","aos","ter","seus","quem","nas","me","esse","eles","você",
               "essa","num","nem","suas","meu","às","minha","têm","numa","pelos","elas","havia","seja","qual",
               "será","nós","tenho","lhe","deles","essas","esses","pelas","este","dele","tu","te","vocês","vos",
               "lhes","meus","minhas","teu","tua","teus","tuas","nosso","nossa","nossos","nossas","dela","delas",
               "esta","estes","estas","aquele","aquela","aqueles","aquelas","isto","aquilo"]
    sw = {normalize_text(w) for w in raw if normalize_text(w)}
    return sorted(sw)

STOPWORDS_PT = _build_stopwords_pt()
STOPWORDS_PT_SET = set(STOPWORDS_PT)

def tokenize_no_stopwords(s: str) -> List[str]:
    toks = normalize_text(s).split()
    return [t for t in toks if t and t not in STOPWORDS_PT_SET]

def jaccard_overlap(a: str, b: str) -> float:
    sa = set(tokenize_no_stopwords(a))
    sb = set(tokenize_no_stopwords(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

# ==================== Transformers ====================

class TypeCoercionTransformer(BaseEstimator, TransformerMixin):
    """Converte cat/txt para string e preenche NaN com ''. Evita quebras downstream."""
    def __init__(self, cat_cols: Iterable[str], txt_cols: Iterable[str]):
        self.cat_cols = list(cat_cols)
        self.txt_cols = list(txt_cols)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in self.cat_cols:
            if c in X:
                X[c] = X[c].astype("string").fillna("")
        for c in self.txt_cols:
            if c in X:
                X[c] = X[c].astype("string").fillna("")
        return X

class MatchNivelAcademicoEqTransformer(BaseEstimator, TransformerMixin):
    """Mesma regra do seu FE: igualdade (lower/strip) entre nível da vaga e do candidato."""
    def __init__(self,
                 vaga_col: str = "perfil_vaga.nivel_academico",
                 cand_col: str = "formacao_e_idiomas.nivel_academico"):
        self.vaga_col = vaga_col
        self.cand_col = cand_col

    def fit(self, X, y=None): return self

    def transform(self, X):
        X = pd.DataFrame(X)
        v = X.get(self.vaga_col, pd.Series([""] * len(X))).astype(str).str.lower().str.strip()
        c = X.get(self.cand_col, pd.Series([""] * len(X))).astype(str).str.lower().str.strip()
        match = (v == c).astype(float)
        return pd.DataFrame({"match_nivel_academico": match})

class OverlapSkillsJaccardTransformer(BaseEstimator, TransformerMixin):
    """Jaccard entre requisitos da vaga e conhecimentos do candidato (com normalização/stopwords)."""
    def __init__(self,
                 vaga_col: str = "perfil_vaga.competencia_tecnicas_e_comportamentais",
                 cand_col: str = "informacoes_profissionais.conhecimentos_tecnicos"):
        self.vaga_col = vaga_col
        self.cand_col = cand_col

    def fit(self, X, y=None): return self

    def transform(self, X):
        X = pd.DataFrame(X)
        v = X.get(self.vaga_col, pd.Series([""] * len(X))).astype(str)
        c = X.get(self.cand_col, pd.Series([""] * len(X))).astype(str)
        vals = [jaccard_overlap(a, b) for a, b in zip(v, c)]
        return pd.DataFrame({"overlap_skills": vals})

class SimilarityTFIDFTransformer(BaseEstimator, TransformerMixin):
    """
    TF-IDF (com stopwords/tokenizer iguais ao seu FE) em CV vs. atividades da vaga.
    Similaridade = dot(normalize(X_cv), normalize(X_vaga)).
    """
    def __init__(self,
                 cand_text_col: str = "cv_pt",
                 vaga_text_col: str = "perfil_vaga.principais_atividades",
                 max_features: int = 2000,
                 min_df: int | float = 3,
                 max_df: float = 0.9,
                 ngram_min: int = 1,
                 ngram_max: int = 1):
        self.cand_text_col = cand_text_col
        self.vaga_text_col = vaga_text_col
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        cv   = X.get(self.cand_text_col, pd.Series([""] * len(X))).astype(str)
        vaga = X.get(self.vaga_text_col, pd.Series([""] * len(X))).astype(str)

        vec = TfidfVectorizer(
            stop_words=STOPWORDS_PT,
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=(self.ngram_min, self.ngram_max),
            lowercase=True,
            tokenizer=tokenize_no_stopwords,
            token_pattern=None
        )
        vec.fit(pd.concat([cv, vaga], ignore_index=True))
        self._vec_ = vec
        return self

    def transform(self, X):
        check_is_fitted(self, "_vec_")
        X = pd.DataFrame(X)
        cv   = X.get(self.cand_text_col, pd.Series([""] * len(X))).astype(str)
        vaga = X.get(self.vaga_text_col, pd.Series([""] * len(X))).astype(str)
        X_cv   = self._vec_.transform(cv)
        X_vaga = self._vec_.transform(vaga)
        X_cv_n   = normalize(X_cv,   norm="l2", copy=False)
        X_vaga_n = normalize(X_vaga, norm="l2", copy=False)
        sim = (X_cv_n.multiply(X_vaga_n)).sum(axis=1).A1
        return pd.DataFrame({"sim_cv_vaga": sim})
