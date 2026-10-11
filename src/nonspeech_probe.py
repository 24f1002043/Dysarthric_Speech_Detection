"""Can dysarthria be 'detected' from the quietest frames of a recording?

If a classifier that only sees the low-energy frames of each utterance (pauses,
leading and trailing silence, the noise floor) separates the groups, then some
of the group difference lives in the recording rather than in the speech.

Three descriptors per utterance, each 40 log-mel bands x (mean, std):
  quiet        the quietest 10% of frames
  quiet_shape  the same, with each frame's mean log-energy removed (colour only)
  loud         the loudest 50% of frames, for reference
"""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed

SR, N_FFT, HOP, N_MELS = 16000, 400, 160, 40
MEL = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS)


def describe(path: str) -> np.ndarray | None:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    if len(wav) < N_FFT * 4:
        return None
    S = np.abs(librosa.stft(wav, n_fft=N_FFT, hop_length=HOP, center=False)) ** 2
    logmel = np.log(MEL @ S + 1e-10).T                    # (frames, mels)
    energy = logmel.mean(axis=1)
    order = np.argsort(energy)
    nq = max(3, int(round(0.10 * len(order))))
    quiet, loud = logmel[order[:nq]], logmel[order[len(order) // 2:]]
    shape = quiet - quiet.mean(axis=1, keepdims=True)
    parts = [quiet.mean(0), quiet.std(0), shape.mean(0), shape.std(0),
             loud.mean(0), loud.std(0)]
    return np.concatenate(parts).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out", default="features/nonspeech.parquet")
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()

    man = pd.read_csv(a.manifest)
    man = man[man.keep].reset_index(drop=True)
    feats = Parallel(n_jobs=a.jobs, batch_size=64, verbose=2)(
        delayed(describe)(p) for p in man.path)
    ok = [f is not None for f in feats]
    names = [f"{blk}_{stat}_{i}" for blk in ("quiet", "quietshape", "loud")
             for stat in ("mean", "std") for i in range(N_MELS)]
    F = pd.DataFrame(np.stack([f for f in feats if f is not None]), columns=names)
    meta = man.loc[ok, ["utt_id", "speaker", "group", "label", "sex", "severity",
                        "session", "mic", "duration"]].reset_index(drop=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pd.concat([meta, F], axis=1).to_parquet(a.out, index=False)
    print(f"wrote {a.out}: {len(F)} utterances, {F.shape[1]} features")


if __name__ == "__main__":
    main()
