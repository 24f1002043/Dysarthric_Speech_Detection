"""Leave-one-speaker-out evaluation shared by every model in this project.

With only 15 speakers, any split that lets a speaker appear in both train and
test turns the task into speaker identification. Every fold here holds out one
entire speaker; nothing about that speaker (not one utterance, not the scaler
statistics) is visible during training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

META_COLS = ["utt_id", "speaker", "group", "label", "sex", "severity",
             "session", "mic", "duration", "prompt"]


def split_meta_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta_cols = [c for c in META_COLS if c in df.columns]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    return df[meta_cols].reset_index(drop=True), df[feat_cols].reset_index(drop=True)


def sample_weights(meta: pd.DataFrame) -> np.ndarray:
    """Equalise the influence of each speaker, then of each class.

    Utterance counts per speaker range from a few hundred to >1500. Without
    this, a handful of talkative speakers would define the decision boundary.
    """
    w = np.ones(len(meta), dtype=float)
    for spk, idx in meta.groupby("speaker").groups.items():
        w[list(idx)] = 1.0 / len(idx)
    for lab, idx in meta.groupby("label").groups.items():
        i = list(idx)
        w[i] = w[i] / w[i].sum()
    return w * len(meta) / w.sum()


def make_model(kind: str, seed: int = 0):
    if kind == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.6,
            reg_lambda=2.0, min_child_weight=5,
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", random_state=seed, n_jobs=-1,
        )
    if kind == "logreg":
        return Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, max_iter=3000, random_state=seed)),
        ])
    if kind == "mlp":
        return Pipeline([
            ("sc", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-2,
                                  max_iter=400, early_stopping=True,
                                  n_iter_no_change=15, random_state=seed)),
        ])
    raise ValueError(f"unknown model kind: {kind}")


def _fit(model, X, y, w, kind, seed: int = 0):
    if kind == "xgb":
        model.fit(X, y, sample_weight=w)
    elif kind == "logreg":
        model.fit(X, y, clf__sample_weight=w)
    else:
        # MLPClassifier takes no sample_weight, so draw a training set whose
        # composition matches the weights instead of ignoring them.
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), size=len(y), replace=True, p=w / w.sum())
        model.fit(X[idx], y[idx])
    return model


def run_loso(df: pd.DataFrame, kind: str = "xgb", seed: int = 0,
             verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-utterance predictions, per-fold metrics)."""
    meta, feats = split_meta_features(df)
    X, y = feats.to_numpy(dtype=np.float32), meta["label"].to_numpy()
    speakers = sorted(meta["speaker"].unique())

    preds, fold_rows = [], []
    for spk in speakers:
        te = (meta["speaker"] == spk).to_numpy()
        tr = ~te
        w = sample_weights(meta[tr].reset_index(drop=True))

        model = _fit(make_model(kind, seed), X[tr], y[tr], w, kind, seed)
        p = model.predict_proba(X[te])[:, 1]

        block = meta[te].copy()
        block["prob"] = p
        block["pred"] = (p >= 0.5).astype(int)
        block["correct"] = (block["pred"] == block["label"]).astype(int)
        preds.append(block)

        # A held-out fold contains exactly one class, so utterance AUC is
        # undefined here; it is computed once over the pooled predictions.
        fold_rows.append({
            "speaker": spk,
            "group": block["group"].iloc[0],
            "severity": block["severity"].iloc[0],
            "label": int(block["label"].iloc[0]),
            "n_utts": int(te.sum()),
            "utt_acc": float(block["correct"].mean()),
            "mean_prob": float(p.mean()),
            "spk_correct": int((p.mean() >= 0.5) == bool(block["label"].iloc[0])),
        })
        if verbose:
            r = fold_rows[-1]
            print(f"  {spk:5s} [{r['severity']:>8s}] n={r['n_utts']:5d} "
                  f"utt_acc={r['utt_acc']:.3f} mean_p={r['mean_prob']:.3f} "
                  f"speaker={'OK ' if r['spk_correct'] else 'MISS'}")

    return pd.concat(preds, ignore_index=True), pd.DataFrame(fold_rows)


def summarise(preds: pd.DataFrame, folds: pd.DataFrame) -> dict:
    y, p, yh = preds["label"], preds["prob"], preds["pred"]
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return {
        "utt_accuracy": float(accuracy_score(y, yh)),
        "utt_balanced_accuracy": float(balanced_accuracy_score(y, yh)),
        "utt_f1": float(f1_score(y, yh)),
        "utt_auc": float(roc_auc_score(y, p)),
        "utt_sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "utt_specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "speaker_accuracy": float(folds["spk_correct"].mean()),
        "speaker_correct": f"{int(folds['spk_correct'].sum())}/{len(folds)}",
        "macro_speaker_utt_acc": float(folds["utt_acc"].mean()),
        "n_utts": int(len(preds)),
        "n_speakers": int(len(folds)),
    }
