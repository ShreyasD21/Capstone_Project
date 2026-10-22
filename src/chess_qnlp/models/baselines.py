"""Classical baselines (scikit-learn). Each returns motif probabilities for every evaluation set.

B0  prior          every example gets the training label frequencies
B1  keyword        1 if the motif word appears in the comment (no training)
B2  board          MLP on the board + move vector
B3  bow            TF-IDF bag of words + logistic regression (word order ignored)
B5  board_bow      MLP on board vector + TF-IDF
"""

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier

from chess_qnlp.constants import MOTIFS
from chess_qnlp.data.masking import KEYWORDS


def _mlp(seed: int) -> MLPClassifier:
    return MLPClassifier(hidden_layer_sizes=(128,), early_stopping=True, max_iter=300, random_state=seed)


def _proba(model, X) -> np.ndarray:
    p = model.predict_proba(X)
    return np.column_stack([q[:, 1] for q in p]) if isinstance(p, list) else p


def prior(data: dict, seed: int) -> dict:
    freq = data["y"]["train"].mean(axis=0)
    return {name: np.tile(freq, (len(data["y"][name]), 1)) for name in data["eval_sets"]}


def keyword(data: dict, seed: int) -> dict:
    def score(texts):
        return np.array([[float(bool(re.search(KEYWORDS[m], t))) for m in MOTIFS] for t in texts])
    return {name: score(data["text"][name]) for name in data["eval_sets"]}


def board(data: dict, seed: int) -> dict:
    model = _mlp(seed).fit(data["board"]["train"], data["y"]["train"])
    return {name: _proba(model, data["board"][name]) for name in data["eval_sets"]}


def bow(data: dict, seed: int) -> dict:
    vec = TfidfVectorizer(token_pattern=r"[a-z0-9]+")
    X = vec.fit_transform(data["text"]["train"])
    model = OneVsRestClassifier(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))
    model.fit(X, data["y"]["train"])
    return {name: model.predict_proba(vec.transform(data["text"][name])) for name in data["eval_sets"]}


def board_bow(data: dict, seed: int) -> dict:
    vec = TfidfVectorizer(token_pattern=r"[a-z0-9]+")
    vec.fit(data["text"]["train"])

    def features(name):
        return np.hstack([data["board"][name], vec.transform(data["text"][name]).toarray()])

    model = _mlp(seed).fit(features("train"), data["y"]["train"])
    return {name: _proba(model, features(name)) for name in data["eval_sets"]}


BASELINES = {
    "B0_prior": prior,
    "B1_keyword": keyword,
    "B2_board": board,
    "B3_bow": bow,
    "B5_board_bow": board_bow,
}
