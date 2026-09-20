"""SHAP explanation of the eGeMAPS + XGBoost detector.

The model is refit inside a leave-one-speaker-out loop and SHAP values are
computed only for held-out utterances, so the explanation describes genuine
out-of-sample behaviour rather than what the model memorised.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from loso import _fit, make_model, sample_weights, split_meta_features

# Readable names for the eGeMAPS parameters that dominate the explanation.
FRIENDLY = {
    "F0semitoneFrom27.5Hz_sma3nz_amean": "Pitch (mean)",
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm": "Pitch variation",
    "loudness_sma3_amean": "Loudness (mean)",
    "loudness_sma3_stddevNorm": "Loudness variation",
    "VoicedSegmentsPerSec": "Speaking rate (voiced segments/s)",
    "MeanVoicedSegmentLengthSec": "Mean voiced segment length",
    "MeanUnvoicedSegmentLength": "Mean pause length",
    "StddevVoicedSegmentLengthSec": "Voiced segment length variation",
    "loudnessPeaksPerSec": "Syllable rate (loudness peaks/s)",
    "jitterLocal_sma3nz_amean": "Jitter (pitch perturbation)",
    "shimmerLocaldB_sma3nz_amean": "Shimmer (amplitude perturbation)",
    "HNRdBACF_sma3nz_amean": "Harmonics-to-noise ratio",
    "F1frequency_sma3nz_amean": "F1 frequency (mean)",
    "F2frequency_sma3nz_amean": "F2 frequency (mean)",
    "F3frequency_sma3nz_amean": "F3 frequency (mean)",
    "F1bandwidth_sma3nz_amean": "F1 bandwidth",
    "alphaRatioV_sma3nz_amean": "Alpha ratio (spectral tilt, voiced)",
    "hammarbergIndexV_sma3nz_amean": "Hammarberg index (voiced)",
    "spectralFluxV_sma3nz_amean": "Spectral flux (voiced)",
    "slopeV0-500_sma3nz_amean": "Spectral slope 0-500 Hz",
    "loudness_sma3_percentile20.0": "Loudness in quiet passages (20th pct)",
    "loudness_sma3_percentile50.0": "Loudness (median)",
    "loudness_sma3_percentile80.0": "Loudness in loud passages (80th pct)",
    "equivalentSoundLevel_dBp": "Equivalent sound level (dB)",
    "logRelF0-H1-H2_sma3nz_amean": "H1-H2 (glottal source / breathiness)",
    "logRelF0-H1-A3_sma3nz_amean": "H1-A3 (spectral decay of the voice source)",
    "F0semitoneFrom27.5Hz_sma3nz_percentile20.0": "Pitch floor (20th pct)",
    "F0semitoneFrom27.5Hz_sma3nz_percentile50.0": "Pitch (median)",
    "F0semitoneFrom27.5Hz_sma3nz_percentile80.0": "Pitch ceiling (80th pct)",
    "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2": "Pitch range",
    "spectralFlux_sma3_stddevNorm": "Spectral flux variation (rate of spectral change)",
    "spectralFlux_sma3_amean": "Spectral flux (rate of spectral change)",
    "mfcc1V_sma3nz_stddevNorm": "MFCC1 variation (voiced)",
    "mfcc2V_sma3nz_stddevNorm": "MFCC2 variation (voiced)",
    "mfcc3_sma3_stddevNorm": "MFCC3 variation",
    "mfcc4_sma3_stddevNorm": "MFCC4 variation",
    "StddevUnvoicedSegmentLength": "Pause length variation",
    "slopeV500-1500_sma3nz_stddevNorm": "Spectral slope 500-1500 Hz, variation",
    "slopeV0-500_sma3nz_stddevNorm": "Spectral slope 0-500 Hz, variation",
    "F1frequency_sma3nz_stddevNorm": "F1 frequency variation",
    "F2frequency_sma3nz_stddevNorm": "F2 frequency variation",
    "F3frequency_sma3nz_stddevNorm": "F3 frequency variation",
    "F1amplitudeLogRelF0_sma3nz_amean": "F1 amplitude relative to F0",
    "alphaRatioUV_sma3nz_amean": "Alpha ratio (spectral tilt, unvoiced)",
    "hammarbergIndexUV_sma3nz_amean": "Hammarberg index (unvoiced)",
}


def pretty(name: str) -> str:
    return FRIENDLY.get(name, name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features/egemaps.parquet")
    ap.add_argument("--mic", default="head")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()

    df = pd.read_parquet(a.features)
    if a.mic != "both" and "mic" in df.columns:
        df = df[df.mic == a.mic].reset_index(drop=True)
    meta, feats = split_meta_features(df)
    X, y = feats.to_numpy(np.float32), meta["label"].to_numpy()
    names = list(feats.columns)

    sv_all, meta_all = [], []
    for spk in sorted(meta["speaker"].unique()):
        te = (meta["speaker"] == spk).to_numpy()
        tr = ~te
        w = sample_weights(meta[tr].reset_index(drop=True))
        model = _fit(make_model("xgb"), X[tr], y[tr], w, "xgb")
        sv_all.append(shap.TreeExplainer(model).shap_values(X[te]))
        meta_all.append(meta[te])
        print(f"  shap fold {spk} done")

    SV = np.vstack(sv_all)
    M = pd.concat(meta_all, ignore_index=True)
    XT = np.vstack([X[(meta["speaker"] == s).to_numpy()]
                    for s in sorted(meta["speaker"].unique())])

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    imp = (pd.DataFrame({"feature": names,
                         "readable": [pretty(n) for n in names],
                         "mean_abs_shap": np.abs(SV).mean(0)})
           .sort_values("mean_abs_shap", ascending=False)
           .reset_index(drop=True))
    imp.to_csv(out / "shap_feature_importance.csv", index=False)
    print("\n--- top features driving dysarthria detection ---")
    print(imp.head(a.top).to_string(index=False))

    # Keep the raw values so the figures can be restyled without refitting.
    np.savez_compressed(out / "shap_values.npz", shap_values=SV, features=XT,
                        feature_names=np.array(names),
                        speaker=M["speaker"].to_numpy(),
                        severity=M["severity"].to_numpy())

    disp = [pretty(n) for n in names]
    plt.figure(figsize=(11, 7))
    shap.summary_plot(SV, XT, feature_names=disp, max_display=a.top, show=False)
    plt.title("What drives the dysarthria decision (held-out SHAP)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out / "shap_summary.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 7))
    shap.summary_plot(SV, XT, feature_names=disp, plot_type="bar",
                      max_display=a.top, show=False)
    plt.title("Mean |SHAP| — eGeMAPS + XGBoost", fontsize=11)
    plt.tight_layout()
    plt.savefig(out / "shap_bar.png", dpi=180)
    plt.close()

    # How the explanation shifts as the disorder gets milder.
    sev = (pd.DataFrame(np.abs(SV), columns=names)
           .assign(severity=M["severity"].values)
           .groupby("severity").mean().T)
    sev["readable"] = [pretty(n) for n in sev.index]
    sev.to_csv(out / "shap_by_severity.csv")
    print(f"\nWrote SHAP outputs to {out}/")


if __name__ == "__main__":
    main()
