"""Aggregate every LOSO run into the tables and figures used in the report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

SEV_ORDER = ["control", "mild", "moderate", "severe"]


def load_runs(outdir: Path) -> tuple[pd.DataFrame, dict, dict]:
    summaries, preds, folds = [], {}, {}
    for f in sorted(outdir.glob("summary_*.json")):
        s = json.loads(f.read_text())
        name = s["name"]
        summaries.append(s)
        p = outdir / f"preds_{name}.csv"
        k = outdir / f"folds_{name}.csv"
        if p.exists():
            preds[name] = pd.read_csv(p)
        if k.exists():
            folds[name] = pd.read_csv(k)
    return pd.DataFrame(summaries), preds, folds


def severity_table(folds: dict) -> pd.DataFrame:
    rows = []
    for name, f in folds.items():
        for sev, g in f.groupby("severity"):
            rows.append({
                "run": name, "severity": sev,
                "speakers": g.speaker.nunique(),
                "utts": int(g.n_utts.sum()),
                "utt_acc": float(np.average(g.utt_acc, weights=g.n_utts)),
                "speakers_correct": f"{int(g.spk_correct.sum())}/{len(g)}",
            })
    df = pd.DataFrame(rows)
    df["severity"] = pd.Categorical(df.severity, SEV_ORDER, ordered=True)
    return df.sort_values(["run", "severity"]).reset_index(drop=True)


def plot_severity(sev: pd.DataFrame, out: Path) -> None:
    piv = sev.pivot(index="severity", columns="run", values="utt_acc").reindex(SEV_ORDER)
    ax = piv.plot(kind="bar", figsize=(8, 4.5), width=0.8, edgecolor="black", lw=.5)
    ax.set_ylabel("Utterance accuracy (held-out speaker)")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.02)
    ax.axhline(0.5, ls="--", c="grey", lw=1, label="chance")
    ax.set_title("Detection accuracy by dysarthria severity\n"
                 "(each band holds only 2-4 speakers - see the per-speaker table)",
                 fontsize=11)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out / "accuracy_by_severity.png", dpi=180)
    plt.close()


def plot_per_speaker(folds: dict, out: Path) -> None:
    n = len(folds)
    ncol = min(3, n)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.6 * nrow),
                             squeeze=False, sharex=True)
    flat = [ax for row in axes for ax in row]
    for ax, (name, f) in zip(flat, folds.items()):
        f = f.sort_values(["label", "severity", "speaker"])
        colors = ["#c44e52" if l else "#4c72b0" for l in f.label]
        ax.barh(f.speaker, f.utt_acc, color=colors, edgecolor="black", lw=.5)
        ax.axvline(0.5, ls="--", c="grey", lw=1)
        ax.set_xlim(0, 1)
        ax.set_title(name, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("utterance accuracy when held out", fontsize=8)
    for ax in flat[n:]:
        ax.axis("off")
    fig.suptitle("Per-speaker accuracy (red = dysarthric, blue = control)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(out / "per_speaker_accuracy.png", dpi=180)
    plt.close()


def plot_roc(preds: dict, out: Path) -> None:
    plt.figure(figsize=(5.5, 5.5))
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(p.label, p.prob)
        plt.plot(fpr, tpr, lw=2, label=name)
    plt.plot([0, 1], [0, 1], "--", c="grey", lw=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Pooled leave-one-speaker-out ROC")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "roc.png", dpi=180)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()
    out = Path(a.outdir)

    summ, preds, folds = load_runs(out)
    if summ.empty:
        raise SystemExit(f"No summary_*.json in {out}")

    cols = ["name", "model", "mic", "n_features", "n_utts",
            "utt_accuracy", "utt_balanced_accuracy", "utt_f1", "utt_auc",
            "utt_sensitivity", "utt_specificity", "speaker_correct"]
    tbl = summ[[c for c in cols if c in summ.columns]].round(4)
    tbl.to_csv(out / "model_comparison.csv", index=False)
    print("\n================ MODEL COMPARISON ================")
    print(tbl.to_string(index=False))

    sev = severity_table(folds)
    sev.to_csv(out / "severity_breakdown.csv", index=False)
    print("\n================ BY SEVERITY ================")
    print(sev.round(4).to_string(index=False))

    print("\n================ HARDEST SPEAKERS ================")
    for name, f in folds.items():
        worst = f.sort_values("utt_acc").head(4)
        print(f"\n{name}:")
        print(worst[["speaker", "group", "severity", "n_utts",
                     "utt_acc", "spk_correct"]].round(3).to_string(index=False))

    if not sev.empty:
        plot_severity(sev, out)
    plot_per_speaker(folds, out)
    plot_roc(preds, out)
    print(f"\nFigures written to {out}/")


if __name__ == "__main__":
    main()
