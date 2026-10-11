"""Additional experiments for the paper, all on cached features.

  wavlm audit   the microphone audit of mic_confound.py repeated for frozen WavLM
  pooled        train on both microphones, test on each (multi-condition training)
  nolevel       cross-microphone transfer once absolute level features are removed
  leakage       utterance-level K-fold against leave-one-speaker-out
  seeds         seed sensitivity of the XGBoost headline run

Everything is written to results/paper/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from loso import _fit, make_model, sample_weights, split_meta_features
from run_experiment import drop_level_features


def _one_fold(X, y, meta, tr, te, kind, seed):
    w = sample_weights(meta[tr].reset_index(drop=True))
    model = make_model(kind, seed)
    if kind == "xgb":
        model.set_params(n_jobs=4)
    model = _fit(model, X[tr], y[tr], w, kind, seed)
    b = meta[te].copy()
    b["prob"] = model.predict_proba(X[te])[:, 1]
    return b


def masked_loso(df, kind, train_mask=None, test_mask=None, label=None,
                seed=0, n_jobs=4) -> pd.DataFrame:
    """LOSO where train and test rows can be restricted independently."""
    meta, feats = split_meta_features(df)
    if label is not None:
        meta = meta.assign(label=label)
    X, y = feats.to_numpy(np.float32), meta["label"].to_numpy()
    trm = np.ones(len(meta), bool) if train_mask is None else np.asarray(train_mask)
    tem = np.ones(len(meta), bool) if test_mask is None else np.asarray(test_mask)
    jobs = []
    for spk in sorted(meta["speaker"].unique()):
        held = (meta["speaker"] == spk).to_numpy()
        tr, te = ~held & trm, held & tem
        if tr.sum() and te.sum():
            jobs.append(delayed(_one_fold)(X, y, meta, tr, te, kind, seed))
    out = pd.concat(Parallel(n_jobs=n_jobs)(jobs), ignore_index=True)
    out["pred"] = (out["prob"] >= 0.5).astype(int)
    return out


def score(p: pd.DataFrame) -> dict:
    spk = p.groupby("speaker").agg(prob=("prob", "mean"), label=("label", "first"))
    res = {
        "n_utts": int(len(p)),
        "utt_accuracy": float(accuracy_score(p.label, p.pred)),
        "utt_balanced_accuracy": float(balanced_accuracy_score(p.label, p.pred)),
        "utt_auc": float(roc_auc_score(p.label, p.prob)),
    }
    if spk.label.nunique() == 2 and len(spk) == p.speaker.nunique():
        res["speaker_correct"] = int(((spk.prob >= 0.5) == spk.label.astype(bool)).sum())
        res["n_speakers"] = int(len(spk))
        res["speaker_auc"] = float(roc_auc_score(spk.label, spk.prob))
    return res


def audit(df, kind, tag, out, n_jobs):
    head, array = (df.mic == "head").to_numpy(), (df.mic == "array").to_numpy()
    dys = (df.label == 1).to_numpy()
    runs = {
        "A_head_to_head": dict(train_mask=head, test_mask=head),
        "A_array_to_array": dict(train_mask=array, test_mask=array),
        "B_head_to_array": dict(train_mask=head, test_mask=array),
        "B_array_to_head": dict(train_mask=array, test_mask=head),
        "P_pooled_to_head": dict(test_mask=head),
        "P_pooled_to_array": dict(test_mask=array),
        "D_dys_head_ctl_array": dict(train_mask=(dys & head) | (~dys & array),
                                     test_mask=(dys & head) | (~dys & array)),
        "D_dys_array_ctl_head": dict(train_mask=(dys & array) | (~dys & head),
                                     test_mask=(dys & array) | (~dys & head)),
    }
    res = {}
    for name, kw in runs.items():
        p = masked_loso(df, kind, n_jobs=n_jobs, **kw)
        p.drop(columns=["prompt"], errors="ignore").to_csv(
            out / f"preds_{tag}_{name}.csv", index=False)
        res[name] = score(p)
        print(f"  {tag:14s} {name:22s} acc={res[name]['utt_accuracy']:.3f} "
              f"auc={res[name]['utt_auc']:.3f}", flush=True)
    # channel probe: the label is the microphone, speakers still held out
    p = masked_loso(df, kind, label=array.astype(int), n_jobs=n_jobs)
    res["C_channel_probe"] = {k: v for k, v in score(p).items() if k.startswith(("utt", "n_utts"))}
    print(f"  {tag:14s} C_channel_probe        acc={res['C_channel_probe']['utt_accuracy']:.3f} "
          f"auc={res['C_channel_probe']['utt_auc']:.3f}", flush=True)
    return res


def leakage(df, kind, n_splits=15, seed=0, n_jobs=4) -> dict:
    """Same data, same model, but folds are drawn over utterances."""
    meta, feats = split_meta_features(df)
    X, y = feats.to_numpy(np.float32), meta["label"].to_numpy()
    strat = meta["speaker"].astype(str)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    jobs = []
    for tr_i, te_i in skf.split(X, strat):
        tr = np.zeros(len(y), bool); tr[tr_i] = True
        jobs.append(delayed(_one_fold)(X, y, meta, tr, ~tr, kind, seed))
    p = pd.concat(Parallel(n_jobs=n_jobs)(jobs), ignore_index=True)
    p["pred"] = (p["prob"] >= 0.5).astype(int)
    return score(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/paper")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    want = lambda k: not a.only or k in a.only.split(",")  # noqa: E731

    ege = pd.read_parquet("features/egemaps.parquet")

    if want("egemaps"):
        res = {"egemaps_xgb": audit(ege, "xgb", "egemaps_xgb", out, a.jobs),
               "egemaps_xgb_nolevel": audit(drop_level_features(ege), "xgb",
                                            "egemaps_nolevel", out, a.jobs)}
        (out / "audit_egemaps.json").write_text(json.dumps(res, indent=2))

    if want("seeds"):
        h = ege[ege.mic == "head"].reset_index(drop=True)
        rows = []
        for s in range(5):
            r = score(masked_loso(h, "xgb", seed=s, n_jobs=a.jobs))
            rows.append({"seed": s, **r})
            print("  seed", s, round(r["utt_auc"], 4), r["speaker_correct"], flush=True)
        (out / "seeds_egemaps_xgb.json").write_text(json.dumps(rows, indent=2))

    if want("leakage"):
        res = {}
        h = ege[ege.mic == "head"].reset_index(drop=True)
        res["egemaps_xgb"] = {"loso": score(masked_loso(h, "xgb", n_jobs=a.jobs)),
                              "utterance_kfold": leakage(h, "xgb", n_jobs=a.jobs)}
        print("  leakage egemaps", res["egemaps_xgb"], flush=True)
        wh = pd.read_parquet("features/wavlm_head.parquet")
        res["wavlm_logreg"] = {"loso": score(masked_loso(wh, "logreg", n_jobs=a.jobs)),
                               "utterance_kfold": leakage(wh, "logreg", n_jobs=a.jobs)}
        print("  leakage wavlm", res["wavlm_logreg"], flush=True)
        del wh
        (out / "leakage.json").write_text(json.dumps(res, indent=2))

    if want("wavlm"):
        w = pd.read_parquet("features/wavlm.parquet")
        res = {"wavlm_logreg": audit(w, "logreg", "wavlm_logreg", out, a.jobs)}
        (out / "audit_wavlm.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
