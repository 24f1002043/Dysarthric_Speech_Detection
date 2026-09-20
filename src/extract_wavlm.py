"""Extract frozen WavLM embeddings (mean+std pooled) for each utterance.

WavLM is never fine-tuned: it is a fixed feature extractor, so the only thing
learned from TORGO is a small classifier on top. With 15 speakers, fine-tuning
a 94M-parameter encoder would memorise speakers rather than the disorder.

Runs on GPU (Colab) or CPU. Masked pooling is used so that padding frames
never contribute to an utterance's embedding.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoModel

TARGET_SR = 16000


def load_audio(path: str) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
    m = np.abs(wav).max()
    if m > 0:
        wav = wav / m                      # peak-normalise: removes gain cues
    return wav.astype(np.float32)


@torch.no_grad()
def embed_batch(model, wavs: list[np.ndarray], layer: int, device) -> np.ndarray:
    lens = [len(w) for w in wavs]
    mx = max(lens)
    batch = np.zeros((len(wavs), mx), dtype=np.float32)
    for i, w in enumerate(wavs):
        batch[i, :len(w)] = w
    x = torch.from_numpy(batch).to(device)
    mask = torch.zeros((len(wavs), mx), dtype=torch.long, device=device)
    for i, L in enumerate(lens):
        mask[i, :L] = 1

    out = model(x, attention_mask=mask, output_hidden_states=True)
    h = out.hidden_states[layer]                          # (B, T, D)

    fl = model._get_feat_extract_output_lengths(
        torch.tensor(lens, device=device)).long()
    T = h.shape[1]
    fmask = (torch.arange(T, device=device)[None, :] < fl[:, None]).float()[..., None]

    n = fmask.sum(1).clamp(min=1)
    mean = (h * fmask).sum(1) / n
    var = ((h - mean[:, None, :]) ** 2 * fmask).sum(1) / n
    std = var.clamp(min=1e-8).sqrt()
    return torch.cat([mean, std], dim=-1).float().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out", default="features/wavlm.parquet")
    ap.add_argument("--model", default="microsoft/wavlm-base-plus")
    ap.add_argument("--mic", default="head", choices=["head", "array", "both"])
    ap.add_argument("--layer", type=int, default=-1,
                    help="hidden_states index; -1 = final layer")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="8 is a good CPU default; raise it on a GPU")
    ap.add_argument("--threads", type=int, default=0,
                    help="torch CPU threads; 0 = leave the default alone")
    ap.add_argument("--max-seconds", type=float, default=12.0)
    a = ap.parse_args()

    df = pd.read_csv(a.manifest)
    df = df[df.keep]
    if a.mic != "both":
        df = df[df.mic == a.mic]
    # Longest-first keeps padding waste low and surfaces OOM immediately.
    df = df.sort_values("duration", ascending=False).reset_index(drop=True)

    if a.threads > 0:
        torch.set_num_threads(a.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  model={a.model}  layer={a.layer}  n={len(df)}  "
          f"batch={a.batch_size}  threads={torch.get_num_threads()}")
    model = AutoModel.from_pretrained(a.model).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    layer = a.layer if a.layer >= 0 else model.config.num_hidden_layers

    max_len = int(a.max_seconds * TARGET_SR)
    vecs, ok = [], []
    nb = math.ceil(len(df) / a.batch_size)
    for b in tqdm(range(nb), desc="wavlm"):
        chunk = df.iloc[b * a.batch_size:(b + 1) * a.batch_size]
        wavs, idx = [], []
        for i, row in chunk.iterrows():
            try:
                w = load_audio(row.path)
            except Exception as e:
                tqdm.write(f"  ! {row.utt_id}: {e}")
                continue
            if len(w) < 400:                   # shorter than one conv frame
                continue
            wavs.append(w[:max_len])
            idx.append(i)
        if not wavs:
            continue
        try:
            vecs.append(embed_batch(model, wavs, layer, device))
            ok.extend(idx)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            for w, i in zip(wavs, idx):        # fall back to one at a time
                vecs.append(embed_batch(model, [w], layer, device))
                ok.append(i)

    X = np.vstack(vecs)
    cols = [f"wavlm_mean_{i}" for i in range(X.shape[1] // 2)] + \
           [f"wavlm_std_{i}" for i in range(X.shape[1] // 2)]
    meta = df.loc[ok, ["utt_id", "speaker", "group", "label", "sex",
                       "severity", "session", "mic", "duration"]].reset_index(drop=True)
    out = pd.concat([meta, pd.DataFrame(X, columns=cols)], axis=1)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    print(f"Wrote {a.out}  shape={out.shape}")


if __name__ == "__main__":
    main()
