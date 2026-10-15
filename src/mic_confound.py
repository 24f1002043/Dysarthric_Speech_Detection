"""Quantify the TORGO microphone trap instead of just asserting it.

TORGO records every session with a head-mounted mic and an array mic. The two
channels are trivially distinguishable, so if the microphone is not held
constant across groups a detector can score near-perfectly while knowing
nothing about dysarthria. Four experiments:

  A  matched     train head  -> test head    (the protocol used in this project)
  B  mismatched  train head  -> test array   (how much is channel, not disorder?)
  C  channel probe            predict WHICH MIC from the same features
  D  confounded  dysarthric=head vs control=array, then the swap

D is the cautionary result: it looks like the best model in the repo and is
entirely worthless.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from loso import (_fit, make_model, run_loso, sample_weights,
                  split_meta_features, summarise)


def cross_mic(df: pd.DataFrame, train_mic: str, test_mic: str) -> dict:
    """LOSO, but the held-out speaker is scored on a different microphone."""
    meta, feats = split_meta_features(df)
    X, y = feats.to_numpy(np.float32), meta["label"].to_numpy()
    rows = []
    for spk in sorted(meta["speaker"].unique()):
        tr = ((meta["speaker"] != spk) & (meta["mic"] == train_mic)).to_numpy()
        te = ((meta["speaker"] == spk) & (meta["mic"] == test_mic)).to_numpy()
        if te.sum() == 0 or tr.sum() == 0:
            continue
        w = sample_weights(meta[tr].reset_index(drop=True))
        model = _fit(make_model("xgb"), X[tr], y[tr], w, "xgb")
        p = model.predict_proba(X[te])[:, 1]
        b = meta[te].copy()
        b["prob"], b["pred"] = p, (p >= 0.5).astype(int)
        rows.append(b)
    out = pd.concat(rows, ignore_index=True)
    return {
        "train_mic": train_mic, "test_mic": test_mic,
        "n_utts": int(len(out)),
        "utt_accuracy": float(accuracy_score(out.label, out.pred)),
        "utt_auc": float(roc_auc_score(out.label, out.prob)),
        "speaker_accuracy": float(
            (out.groupby("speaker")["prob"].mean().ge(0.5)
             == out.groupby("speaker")["label"].first().astype(bool)).mean()),
    }


def channel_probe(df: pd.DataFrame) -> dict:
    """Can the SAME features identify the microphone? (Held-out speakers.)"""
    d = df.copy()
    d["label"] = (d["mic"] == "array").astype(int)
    preds, folds = run_loso(d, kind="xgb", verbose=False)
    s = summarise(preds, folds)
    return {"task": "predict microphone", "utt_accuracy": s["utt_accuracy"],
            "utt_auc": s["utt_auc"], "n_utts": s["n_utts"]}


def confounded(df: pd.DataFrame, dys_mic: str, ctl_mic: str) -> dict:
    """Deliberately align microphone with group, the mistake to avoid."""
    sub = df[((df.label == 1) & (df.mic == dys_mic)) |
             ((df.label == 0) & (df.mic == ctl_mic))].reset_index(drop=True)
    preds, folds = run_loso(sub, kind="xgb", verbose=False)
    s = summarise(preds, folds)
    return {"dysarthric_mic": dys_mic, "control_mic": ctl_mic,
            "utt_accuracy": s["utt_accuracy"], "utt_auc": s["utt_auc"],
            "speaker_accuracy": s["speaker_accuracy"], "n_utts": s["n_utts"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features/egemaps.parquet")
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()

    df = pd.read_parquet(a.features)
    both = sorted(df.mic.unique())
    print(f"microphones present: {both}")
    if len(both) < 2:
        raise SystemExit("Need both microphones; re-extract with --mic both")

    res = {}
    print("\n[A] matched: head -> head  (primary protocol)")
    res["A_matched_head"] = cross_mic(df, "head", "head")
    print("[A'] matched: array -> array")
    res["A_matched_array"] = cross_mic(df, "array", "array")
    print("[B] mismatched: head -> array")
    res["B_mismatch_head_to_array"] = cross_mic(df, "head", "array")
    print("[B'] mismatched: array -> head")
    res["B_mismatch_array_to_head"] = cross_mic(df, "array", "head")
    print("[C] channel probe")
    res["C_channel_probe"] = channel_probe(df)
    print("[D] confounded: dysarthric=head, control=array")
    res["D_confounded_dys_head"] = confounded(df, "head", "array")
    print("[D'] confounded, swapped")
    res["D_confounded_dys_array"] = confounded(df, "array", "head")

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "mic_confound.json").write_text(json.dumps(res, indent=2))

    print("\n================ MICROPHONE CONFOUND ================")
    for k, v in res.items():
        acc = v.get("utt_accuracy")
        auc = v.get("utt_auc")
        print(f"{k:32s} acc={acc:.3f} auc={auc:.3f}")
    print(f"\nWrote {out}/mic_confound.json")


if __name__ == "__main__":
    main()
