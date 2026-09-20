"""Score a single recording for signs of dysarthria.

This is the screening-shaped end of the project: one short wav in, a
probability and the acoustic reasons for it out.

    python src/predict.py --train features/egemaps.parquet --wav sample.wav

The model is fit on every TORGO speaker, so the probability is only meaningful
for a speaker the model has never heard. It is a research prototype: it is not
a diagnosis, and it is not validated for clinical use.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from explain_shap import pretty
from loso import _fit, make_model, sample_weights, split_meta_features


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="features/egemaps.parquet")
    ap.add_argument("--wav", required=True, nargs="+")
    ap.add_argument("--mic", default="head")
    ap.add_argument("--top", type=int, default=6)
    a = ap.parse_args()

    df = pd.read_parquet(a.train)
    if a.mic != "both" and "mic" in df.columns:
        df = df[df.mic == a.mic].reset_index(drop=True)
    meta, feats = split_meta_features(df)
    X, y = feats.to_numpy(np.float32), meta["label"].to_numpy()

    model = _fit(make_model("xgb"), X, y, sample_weights(meta), "xgb")
    print(f"fit on {len(df)} utterances from {meta.speaker.nunique()} speakers "
          f"({a.mic} mic)\n")

    import shap
    from extract_egemaps import build_smile
    smile = build_smile()
    explainer = shap.TreeExplainer(model)
    names = list(feats.columns)

    for w in a.wav:
        v = smile.process_file(w).to_numpy().ravel()
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        p = float(model.predict_proba(v[None, :])[0, 1])
        sv = explainer.shap_values(v[None, :])[0]
        order = np.argsort(-np.abs(sv))[:a.top]

        parts = Path(w).resolve().parts
        print("/".join(parts[-4:]) if len(parts) >= 4 else Path(w).name)
        print(f"  P(dysarthric) = {p:.3f}  ->  "
              f"{'signs of dysarthria' if p >= .5 else 'typical speech'}")
        print("  strongest evidence:")
        for i in order:
            direction = "toward dysarthric" if sv[i] > 0 else "toward control"
            print(f"    {pretty(names[i]):<44s} {direction:>18s}  "
                  f"(shap {sv[i]:+.3f})")
        print()

    print("Research prototype - not a medical device, not a diagnosis.")


if __name__ == "__main__":
    main()
