"""Uncertainty and subgroup statistics computed from stored LOSO predictions.

  bootstrap   speaker-level (cluster) bootstrap CIs, and the paired WavLM - eGeMAPS gap
  listening   how speaker-level decisions improve with the number of utterances heard
  subgroup    sex and severity breakdown
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

RES = Path("results")
OUT = RES / "paper"
RUNS = ["egemaps_xgb", "egemaps_xgb_sharedprompt", "egemaps_xgb_nolevel",
        "egemaps_xgb_array", "wavlm_logreg", "wavlm_logreg_sharedprompt",
        "wavlm_mlp", "wavlm_logreg_array"]


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(RES / f"preds_{name}.csv")


def clopper_pearson(k: int, n: int, a: float = 0.05) -> tuple[float, float]:
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)


def boot_indices(spk_label: pd.Series, n_boot: int, rng) -> list[list[str]]:
    """Resample speakers with replacement inside each group."""
    dys = spk_label.index[spk_label == 1].to_numpy()
    ctl = spk_label.index[spk_label == 0].to_numpy()
    return [list(rng.choice(dys, len(dys))) + list(rng.choice(ctl, len(ctl)))
            for _ in range(n_boot)]


def metrics_on(by_spk: dict, sample: list[str]) -> tuple[float, float]:
    y = np.concatenate([by_spk[s][0] for s in sample])
    p = np.concatenate([by_spk[s][1] for s in sample])
    return roc_auc_score(y, p), balanced_accuracy_score(y, p >= 0.5)


def bootstrap(n_boot: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    preds = {r: load(r) for r in RUNS}
    spk_label = preds["egemaps_xgb"].groupby("speaker")["label"].first()
    samples = boot_indices(spk_label, n_boot, rng)
    out, draws = {}, {}
    for r, p in preds.items():
        by = {s: (g.label.to_numpy(), g.prob.to_numpy()) for s, g in p.groupby("speaker")}
        d = np.array([metrics_on(by, s) for s in samples])
        draws[r] = d
        k = int(((p.groupby("speaker").prob.mean() >= 0.5)
                 == spk_label.astype(bool)).sum())
        out[r] = {
            "auc": float(roc_auc_score(p.label, p.prob)),
            "auc_ci": [float(x) for x in np.percentile(d[:, 0], [2.5, 97.5])],
            "bacc": float(balanced_accuracy_score(p.label, p.pred)),
            "bacc_ci": [float(x) for x in np.percentile(d[:, 1], [2.5, 97.5])],
            "speakers_correct": k,
            "speakers_ci": list(clopper_pearson(k, len(spk_label))),
        }
    for a, b in [("wavlm_logreg", "egemaps_xgb"), ("egemaps_xgb", "egemaps_xgb_nolevel"),
                 ("wavlm_logreg", "wavlm_logreg_array"), ("egemaps_xgb", "egemaps_xgb_array")]:
        diff = draws[a][:, 0] - draws[b][:, 0]
        out[f"delta_auc__{a}__minus__{b}"] = {
            "delta": out[a]["auc"] - out[b]["auc"],
            "ci": [float(x) for x in np.percentile(diff, [2.5, 97.5])],
            "frac_boot_le_0": float((diff <= 0).mean()),
        }
    return out


def listening(ks=(1, 2, 3, 5, 10, 20, 50, 100), n_rep: int = 2000, seed: int = 0) -> dict:
    """Speaker-level decision from k utterances drawn at random."""
    rng = np.random.default_rng(seed)
    out = {}
    for r in ["egemaps_xgb", "wavlm_logreg"]:
        p = load(r)
        groups = {s: (g.prob.to_numpy(), int(g.label.iloc[0]), g.severity.iloc[0])
                  for s, g in p.groupby("speaker")}
        rows = []
        for k in ks:
            correct = {s: 0.0 for s in groups}
            aucs = []
            for _ in range(n_rep):
                m, y = [], []
                for s, (pr, lab, _) in groups.items():
                    v = rng.choice(pr, size=min(k, len(pr)), replace=False).mean()
                    correct[s] += float((v >= 0.5) == bool(lab))
                    m.append(v); y.append(lab)
                aucs.append(roc_auc_score(y, m))
            row = {"k": k, "speaker_auc": float(np.mean(aucs)),
                   "speaker_acc": float(np.mean([c / n_rep for c in correct.values()]))}
            for sev in ["control", "mild", "moderate", "severe"]:
                row[f"acc_{sev}"] = float(np.mean(
                    [correct[s] / n_rep for s, g in groups.items() if g[2] == sev]))
            rows.append(row)
        out[r] = rows
    return out


def subgroup() -> dict:
    out = {}
    for r in ["egemaps_xgb", "wavlm_logreg"]:
        p = load(r)
        rows = {}
        for sex, g in p.groupby("sex"):
            rows[f"sex_{sex}"] = {
                "n_speakers": int(g.speaker.nunique()), "n_utts": int(len(g)),
                "auc": float(roc_auc_score(g.label, g.prob)),
                "sensitivity": float(g[g.label == 1].pred.mean()),
                "specificity": float(1 - g[g.label == 0].pred.mean()),
            }
        out[r] = rows
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    b = bootstrap()
    (OUT / "bootstrap.json").write_text(json.dumps(b, indent=2))
    for k, v in b.items():
        print(k, json.dumps(v))
    l = listening()
    (OUT / "listening.json").write_text(json.dumps(l, indent=2))
    for r, rows in l.items():
        print(r); print(pd.DataFrame(rows).round(3).to_string(index=False))
    s = subgroup()
    (OUT / "subgroup.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
