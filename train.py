"""AfriNews multilingual topic classification: language-balanced weighted F1.

The prepared features tx_000..tx_383 are L2-normalized, noise-perturbed hashed
character n-gram vectors of each article. On this unit-norm geometry three
neighbor-style views of the data are combined by a logistic meta-learner:

  - an RBF SVM (its kernel on unit vectors is a function of cosine similarity),
  - a cosine-distance kNN, and
  - transductive label spreading over a kNN graph built from the labeled rows
    together with the unlabeled pool and the test rows. Because the features are
    unit-norm, a Euclidean kNN graph is equivalent to a cosine graph, so the
    graph follows the same geometry as the kNN model but lets topic labels flow
    through the unlabeled/test manifold. This is the largest single contributor
    to the worst-language F1, which the metric weights heavily.

The meta-learner also receives a one-hot of the language, which lets it shift
toward each language's class prior. Each language only ever uses a fixed subset
of the seven topics (e.g. Amharic articles are never label 1/4/6), so predicting
a topic absent from a language is always wrong: per-language probabilities are
masked to the topics observed for that language in training before the argmax.

Base-model probabilities for stacking are produced out-of-fold (5-fold,
stratified by language and topic) so the meta-learner never trains on a base
model's in-sample predictions; the label-spreading out-of-fold pass treats the
held-out fold as unlabeled alongside the unlabeled pool, mirroring how the test
rows are scored. Self-training (hard pseudo-labels) was tested and discarded:
label noise from a ~65%-accurate labeler lowered CV, whereas soft label
spreading used only as a meta feature helps.

Run from the task workspace root; reads public/ and writes
submission/submission.csv. CPU only, a few minutes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import LabelSpreading
from sklearn.svm import SVC

SEED = 42
LABELS = list(range(7))
N_FOLDS = 5
SVM_C = 10.0
SVM_GAMMA = 4.0
KNN_K = 11
LS_K = 11
LS_ALPHA = 0.4

PUBLIC = Path("public")
OUT_DIR = Path("submission")


def load_features(frame: pd.DataFrame) -> np.ndarray:
    feature_cols = [c for c in frame.columns if c.startswith("tx_")]
    return frame[feature_cols].to_numpy(dtype=np.float32)


def make_svm() -> SVC:
    return SVC(C=SVM_C, gamma=SVM_GAMMA, probability=True, random_state=SEED)


def make_knn() -> KNeighborsClassifier:
    return KNeighborsClassifier(n_neighbors=KNN_K, metric="cosine", weights="distance")


def language_onehot(languages: np.ndarray, categories: list[str]) -> np.ndarray:
    codes = pd.Categorical(languages, categories=categories)
    return pd.get_dummies(codes).to_numpy(dtype=np.float32)


def label_spreading_proba(
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    X_target: np.ndarray,
    X_extra_unlabeled: np.ndarray,
) -> np.ndarray:
    """Transductive class probabilities for X_target from a kNN graph over
    labeled rows plus target and extra unlabeled rows. Columns are aligned 0..6."""
    X_graph = np.vstack([X_labeled, X_extra_unlabeled, X_target])
    y_graph = np.concatenate(
        [y_labeled, np.full(len(X_extra_unlabeled), -1), np.full(len(X_target), -1)]
    )
    model = LabelSpreading(
        kernel="knn", n_neighbors=LS_K, alpha=LS_ALPHA, max_iter=60, tol=1e-3
    )
    model.fit(X_graph, y_graph)

    target_dist = model.label_distributions_[-len(X_target):]
    proba = np.zeros((len(X_target), len(LABELS)), dtype=np.float64)
    for column, label in enumerate(model.classes_):
        proba[:, int(label)] = target_dist[:, column]
    return proba


def oof_proba(make_model, X: np.ndarray, y: np.ndarray, folds) -> np.ndarray:
    """Out-of-fold class probabilities for an inductive estimator."""
    oof = np.zeros((len(X), len(LABELS)), dtype=np.float64)
    for train_idx, val_idx in folds:
        model = make_model()
        model.fit(X[train_idx], y[train_idx])
        oof[val_idx] = model.predict_proba(X[val_idx])
    return oof


def oof_label_spreading(
    X: np.ndarray, y: np.ndarray, X_unlabeled: np.ndarray, folds
) -> np.ndarray:
    """Out-of-fold label-spreading probabilities; each held-out fold is treated
    as unlabeled alongside the unlabeled pool, matching the test-time graph."""
    oof = np.zeros((len(X), len(LABELS)), dtype=np.float64)
    for train_idx, val_idx in folds:
        oof[val_idx] = label_spreading_proba(
            X[train_idx], y[train_idx], X[val_idx], X_unlabeled
        )
    return oof


def mask_to_language_topics(
    proba: np.ndarray, languages: np.ndarray, allowed: dict[str, set[int]]
) -> np.ndarray:
    """Zero out topics a language never uses so they cannot be predicted."""
    masked = proba.copy()
    for language in np.unique(languages):
        forbidden = [c for c in LABELS if c not in allowed[language]]
        if forbidden:
            masked[np.ix_(languages == language, forbidden)] = -1.0
    return masked


def main() -> None:
    np.random.seed(SEED)

    train = pd.read_csv(PUBLIC / "train.csv")
    test = pd.read_csv(PUBLIC / "test.csv")
    unlabeled = pd.read_csv(PUBLIC / "train_unlabeled.csv")

    X_train = load_features(train)
    y_train = train["label"].to_numpy()
    train_languages = train["language"].to_numpy()
    X_test = load_features(test)
    test_languages = test["language"].to_numpy()
    X_unlabeled = load_features(unlabeled)

    languages = sorted(np.unique(train_languages).tolist())
    # Topics observed per language in training; the stratified split keeps the
    # same per-language topic support in the test set.
    allowed = {
        language: set(train.loc[train["language"] == language, "label"].unique())
        for language in languages
    }

    # Stratify folds by language and topic so out-of-fold probabilities match
    # the metric's per-language structure.
    strata = np.array([f"{lang}_{lab}" for lang, lab in zip(train_languages, y_train)])
    folds = list(
        StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(
            X_train, strata
        )
    )

    svm_oof = oof_proba(make_svm, X_train, y_train, folds)
    knn_oof = oof_proba(make_knn, X_train, y_train, folds)
    ls_oof = oof_label_spreading(X_train, y_train, X_unlabeled, folds)
    lang_oof = language_onehot(train_languages, languages)

    meta = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
    meta.fit(np.hstack([svm_oof, knn_oof, ls_oof, lang_oof]), y_train)

    # Test predictions: inductive models refit on all training data; label
    # spreading run transductively over train + unlabeled + test.
    svm_test = make_svm().fit(X_train, y_train).predict_proba(X_test)
    knn_test = make_knn().fit(X_train, y_train).predict_proba(X_test)
    ls_test = label_spreading_proba(X_train, y_train, X_test, X_unlabeled)
    lang_test = language_onehot(test_languages, languages)

    proba = meta.predict_proba(np.hstack([svm_test, knn_test, ls_test, lang_test]))
    proba = mask_to_language_topics(proba, test_languages, allowed)

    predictions = proba.argmax(axis=1)
    confidence = proba[np.arange(len(proba)), predictions].clip(0.0, 1.0)

    submission = pd.DataFrame(
        {
            "sample_id": test["sample_id"].to_numpy(),
            "label": predictions.astype(int),
            "confidence": np.round(confidence, 6),
        }
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUT_DIR / "submission.csv", index=False)
    print(f"Wrote {OUT_DIR / 'submission.csv'} with {len(submission)} rows.")


if __name__ == "__main__":
    main()
