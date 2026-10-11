"""Figures for the paper, drawn from results/ and results/paper/."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES, P, FIG = Path("results"), Path("results/paper"), Path("paper/figs")
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
C_E, C_W = "#b5651d", "#1f4e79"
SEV_ORDER = ["control", "mild", "moderate", "severe"]


def fig_audit():
    e = json.loads((P / "audit_egemaps.json").read_text())["egemaps_xgb"]
    w = json.loads((P / "audit_wavlm.json").read_text())["wavlm_logreg"]
    lk = json.loads((P / "leakage.json").read_text())
    stored = json.loads((RES / "mic_confound.json").read_text())
    e_auc = {
        "A_head_to_head": stored["A_matched_head"]["utt_auc"],
        "A_array_to_array": stored["A_matched_array"]["utt_auc"],
        "B_head_to_array": stored["B_mismatch_head_to_array"]["utt_auc"],
        "B_array_to_head": stored["B_mismatch_array_to_head"]["utt_auc"],
        "P_pooled_to_head": e["P_pooled_to_head"]["utt_auc"],
        "P_pooled_to_array": e["P_pooled_to_array"]["utt_auc"],
        "D_dys_head_ctl_array": stored["D_confounded_dys_head"]["utt_auc"],
        "K": lk["egemaps_xgb"]["utterance_kfold"]["utt_auc"],
    }
    w_auc = {k: w[k]["utt_auc"] for k in e_auc if k != "K"}
    w_auc["A_head_to_head"] = json.loads((RES / "summary_wavlm_logreg.json").read_text())["utt_auc"]
    w_auc["A_array_to_array"] = json.loads((RES / "summary_wavlm_logreg_array.json").read_text())["utt_auc"]
    w_auc["K"] = lk["wavlm_logreg"]["utterance_kfold"]["utt_auc"]
    labels = ["head\u2192head", "array\u2192array", "head\u2192array", "array\u2192head",
              "both\u2192head", "both\u2192array", "mic aligned\nwith group", "utterance\nk-fold"]
    keys = list(e_auc)
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(7.1, 2.1))
    ax.bar(x - 0.19, [e_auc[k] for k in keys], 0.38, color=C_E, label="eGeMAPS + XGBoost")
    ax.bar(x + 0.19, [w_auc[k] for k in keys], 0.38, color=C_W, label="WavLM + logistic regression")
    for i, k in enumerate(keys):
        ax.text(i - 0.19, e_auc[k] + 0.015, f"{e_auc[k]:.2f}", ha="center", fontsize=6)
        ax.text(i + 0.19, w_auc[k] + 0.015, f"{w_auc[k]:.2f}", ha="center", fontsize=6)
    ax.axhline(0.5, color="k", lw=0.6, ls="--")
    ax.text(len(keys) - 0.45, 0.515, "chance", fontsize=6, ha="right")
    for xv in (1.5, 3.5, 5.5):
        ax.axvline(xv, color="0.8", lw=0.5)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Utterance-level AUC")
    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0.0, 1.18))
    fig.savefig(FIG / "audit.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_speakers():
    fe = pd.read_csv(RES / "folds_egemaps_xgb.csv").set_index("speaker")
    fw = pd.read_csv(RES / "folds_wavlm_logreg.csv").set_index("speaker")
    order = (fe.assign(s=fe.severity.map({s: i for i, s in enumerate(SEV_ORDER)}))
               .sort_values(["s"]).index.tolist())
    y = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(3.45, 2.9))
    ax.axvline(0.5, color="k", lw=0.6, ls="--")
    ax.hlines(y, fe.loc[order, "mean_prob"], fw.loc[order, "mean_prob"], color="0.75", lw=0.8)
    ax.scatter(fe.loc[order, "mean_prob"], y, s=18, marker="s", color=C_E, label="eGeMAPS + XGBoost", zorder=3)
    ax.scatter(fw.loc[order, "mean_prob"], y, s=18, marker="o", color=C_W, label="WavLM + log. reg.", zorder=3)
    short = {"control": "ctrl", "mild": "mild", "moderate": "mod.", "severe": "sev."}
    ax.set_yticks(y, [f"{s} ({short[fe.loc[s, 'severity']]})" for s in order])
    ax.axhline(y[6] - 0.5, color="0.6", lw=0.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Mean predicted probability of dysarthria (speaker held out)")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2,
              handletextpad=0.2, columnspacing=1.0)
    fig.savefig(FIG / "speakers.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_listening():
    L = json.loads((P / "listening.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.75), sharey=True)
    style = {"control": ("0.35", "-", "o"), "mild": ("#2a7f62", "-", "^"),
             "moderate": ("#c0392b", "--", "s"), "severe": ("#1f4e79", ":", "D")}
    for ax, (run, title) in zip(axes, [("egemaps_xgb", "eGeMAPS + XGBoost"),
                                       ("wavlm_logreg", "WavLM + log. reg.")]):
        d = pd.DataFrame(L[run])
        for sev, (c, ls, mk) in style.items():
            ax.plot(d.k, d[f"acc_{sev}"], color=c, ls=ls, marker=mk, ms=2.5, lw=0.9, label=sev)
        ax.set_xscale("log")
        ax.set_xticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
        ax.set_title(title, fontsize=7.5, pad=2)
        ax.set_xlabel("Utterances heard")
        ax.set_ylim(0.38, 1.03)
    axes[0].set_ylabel("Speakers classified correctly")
    fig.subplots_adjust(wspace=0.08)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.30),
               fontsize=7, handlelength=2.2, columnspacing=1.2)
    fig.savefig(FIG / "listening.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_shap():
    s = pd.read_csv(RES / "shap_feature_importance.csv").head(12).iloc[::-1]
    level = s.feature.str.match(r"^(loudness_sma3_(amean|percentile|pctlrange)|equivalentSoundLevel)")
    names = (s.readable.str.replace(r" \(rate of spectral change\)", "", regex=True)
                       .str.replace("glottal source / ", "", regex=False))
    fig, ax = plt.subplots(figsize=(3.45, 2.2))
    ax.barh(names, s.mean_abs_shap, color=np.where(level, "#c0392b", "0.55"), height=0.7)
    ax.set_xlabel("Mean |SHAP| on held-out speakers (log-odds)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#c0392b", label="absolute level feature"),
                       Patch(color="0.55", label="other")],
              frameon=False, loc="lower right")
    fig.savefig(FIG / "shap.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    fig_audit(); fig_speakers(); fig_listening(); fig_shap()
    print("figures written to", FIG)
