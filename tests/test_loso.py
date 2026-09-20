"""Tests for the claims the whole project rests on.

The headline numbers are only meaningful if the evaluation really is
speaker-disjoint and really is balanced. These check that directly rather than
trusting the prose.

    python -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loso import run_loso, sample_weights, split_meta_features, summarise  # noqa: E402


def synthetic(n_speakers: int = 6, per_speaker: int = 40, seed: int = 0) -> pd.DataFrame:
    """Speakers whose class is separable, with a strong per-speaker offset.

    The offset is the point: a model that latches onto speaker identity will
    do well in-sample and badly under LOSO.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_speakers):
        label = int(s % 2 == 0)
        spk_offset = rng.normal(0, 3.0, size=4)
        for j in range(per_speaker):
            x = rng.normal(0, 1, size=4) + spk_offset + (2.0 if label else -2.0)
            rows.append({
                "utt_id": f"S{s}_u{j}", "speaker": f"S{s}",
                "group": "dysarthric" if label else "control",
                "label": label, "sex": "F", "severity": "mild" if label else "control",
                "session": "Session1", "mic": "head", "duration": 2.0,
                "prompt": f"word{j % 7}",
                **{f"f{k}": x[k] for k in range(4)},
            })
    return pd.DataFrame(rows)


def test_features_exclude_metadata():
    df = synthetic()
    meta, feats = split_meta_features(df)
    assert list(feats.columns) == ["f0", "f1", "f2", "f3"]
    for leaky in ("speaker", "label", "severity", "group", "duration", "mic"):
        assert leaky not in feats.columns


def test_no_speaker_appears_in_its_own_training_set():
    df = synthetic()
    meta, feats = split_meta_features(df)
    for spk in sorted(meta.speaker.unique()):
        train = meta[meta.speaker != spk]
        assert spk not in set(train.speaker), f"{spk} leaked into training"
        assert len(train) + (meta.speaker == spk).sum() == len(meta)


def test_every_speaker_is_held_out_exactly_once():
    df = synthetic()
    preds, folds = run_loso(df, kind="logreg", verbose=False)
    assert len(folds) == df.speaker.nunique()
    assert folds.speaker.is_unique
    # every utterance is predicted exactly once, by a model that never saw it
    assert len(preds) == len(df)
    assert preds.utt_id.is_unique


def test_sample_weights_balance_speakers_and_classes():
    df = synthetic(n_speakers=4, per_speaker=10)
    # make one speaker dominate the utterance count
    df = pd.concat([df, df[df.speaker == "S0"].iloc[:200]], ignore_index=True)
    meta, _ = split_meta_features(df)
    w = sample_weights(meta)

    by_class = pd.Series(w).groupby(meta.label).sum()
    assert np.allclose(by_class.iloc[0], by_class.iloc[1]), "classes not balanced"

    within = pd.Series(w).groupby(meta.speaker).sum()
    for lab in (0, 1):
        spk = meta[meta.label == lab].speaker.unique()
        vals = within[list(spk)].to_numpy()
        assert np.allclose(vals, vals[0]), "speakers not equally weighted"


def test_loso_is_harder_than_a_random_split():
    """Sanity check that LOSO is measuring something stricter."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    df = synthetic(n_speakers=6, per_speaker=60, seed=3)
    meta, feats = split_meta_features(df)
    X, y = feats.to_numpy(), meta.label.to_numpy()

    random_split = cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=5).mean()
    preds, folds = run_loso(df, kind="logreg", verbose=False)
    loso = summarise(preds, folds)["utt_accuracy"]

    assert random_split >= loso, (
        f"random split {random_split:.3f} should be optimistic vs LOSO {loso:.3f}")
