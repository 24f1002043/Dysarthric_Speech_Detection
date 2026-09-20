"""Extract eGeMAPSv02 functionals (88 features/utterance) with openSMILE.

eGeMAPS is the minimalistic parameter set designed for clinical and affective
voice research: pitch, loudness, jitter, shimmer, harmonicity, formants,
spectral shape and timing. Every feature has a name a speech-language
pathologist can argue with, which is what makes the SHAP step meaningful.
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import opensmile
import pandas as pd
from tqdm import tqdm

_SMILE: opensmile.Smile | None = None


def build_smile() -> opensmile.Smile:
    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
        num_workers=1,
    )


def _init_worker() -> None:
    global _SMILE
    _SMILE = build_smile()


def _process(path: str) -> np.ndarray | None:
    """Return the 88-dim functional vector, or None if the file is unusable."""
    global _SMILE
    if _SMILE is None:
        _SMILE = build_smile()
    try:
        arr = _SMILE.process_file(path).to_numpy().ravel()
    except Exception:
        return None
    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out", default="features/egemaps.parquet")
    ap.add_argument("--mic", default="head", choices=["head", "array", "both"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()

    df = pd.read_csv(a.manifest)
    df = df[df.keep]
    if a.mic != "both":
        df = df[df.mic == a.mic]
    df = df.reset_index(drop=True)
    print(f"Extracting eGeMAPSv02 for {len(df)} utterances "
          f"(mic={a.mic}, workers={a.workers})")

    paths = df["path"].tolist()
    if a.workers > 1:
        with ProcessPoolExecutor(a.workers, initializer=_init_worker) as ex:
            results = list(tqdm(ex.map(_process, paths, chunksize=32),
                                total=len(paths), desc="egemaps"))
    else:
        _init_worker()
        results = [_process(p) for p in tqdm(paths, desc="egemaps")]

    ok = [i for i, r in enumerate(results) if r is not None]
    bad = len(results) - len(ok)
    if bad:
        print(f"  ! skipped {bad} unreadable files")

    names = list(build_smile().feature_names)
    X = pd.DataFrame(np.vstack([results[i] for i in ok]), columns=names)
    meta = df.loc[ok, ["utt_id", "speaker", "group", "label", "sex",
                       "severity", "session", "mic", "duration", "prompt"]
                  ].reset_index(drop=True)
    out = pd.concat([meta, X], axis=1)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    print(f"Wrote {a.out}  shape={out.shape}  ({len(names)} features)")


if __name__ == "__main__":
    main()
