"""Run a leave-one-speaker-out experiment and write predictions + metrics."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from loso import run_loso, split_meta_features, summarise


LEVEL_FEATURE = re.compile(
    r"^(loudness_sma3_(amean|percentile|pctlrange)|equivalentSoundLevel)")


def drop_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """Remove absolute loudness / sound-level features.

    SHAP puts an absolute loudness statistic at the top of the ranking, and
    absolute level is partly a property of the recording (gain, distance,
    session) rather than of the speaker's speech -- though reduced vocal
    intensity is also a genuine clinical sign. Only genuinely absolute
    statistics are removed; scale-invariant ones such as the coefficient of
    variation are kept, since those survive any change of gain.
    """
    drop = [c for c in df.columns if LEVEL_FEATURE.match(c)]
    print(f"  dropping {len(drop)} absolute-level features: "
          f"{', '.join(drop[:4])}{' ...' if len(drop) > 4 else ''}")
    return df.drop(columns=drop)


def restrict_to_shared_prompts(df: pd.DataFrame, manifest: str) -> pd.DataFrame:
    """Keep only prompts that BOTH groups actually spoke.

    TORGO speakers do not all read the same material, so a detector could in
    principle key on *what* was said rather than *how*. Restricting to the
    shared prompt set removes that escape route; if accuracy survives, the
    decision was about the speech, not the script.
    """
    if "prompt" not in df.columns:
        man = pd.read_csv(manifest, usecols=["utt_id", "prompt"])
        df = df.merge(man, on="utt_id", how="left")
    p = df["prompt"].fillna("").str.strip().str.lower()
    df = df.assign(_p=p)
    spoken_by = df.groupby("_p")["label"].nunique()
    shared = set(spoken_by[spoken_by == 2].index) - {""}
    out = df[df["_p"].isin(shared)].drop(columns=["_p"]).reset_index(drop=True)
    print(f"  shared-prompt filter: {len(out)}/{len(df)} utterances, "
          f"{len(shared)} prompt types spoken by both groups")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--model", default="xgb", choices=["xgb", "logreg", "mlp"])
    ap.add_argument("--mic", default="head", choices=["head", "array", "both"])
    ap.add_argument("--name", required=True, help="tag used for output files")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shared-prompts", action="store_true",
                    help="restrict to prompts spoken by both groups")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--no-level-features", action="store_true",
                    help="drop absolute loudness / sound-level features")
    a = ap.parse_args()

    df = pd.read_parquet(a.features)
    if a.mic != "both" and "mic" in df.columns:
        df = df[df.mic == a.mic].reset_index(drop=True)
    if a.shared_prompts:
        df = restrict_to_shared_prompts(df, a.manifest)
    if a.no_level_features:
        df = drop_level_features(df)

    print(f"\n=== {a.name} | {a.model} | mic={a.mic} | "
          f"{len(df)} utts, {df.speaker.nunique()} speakers ===")
    preds, folds = run_loso(df, kind=a.model, seed=a.seed)
    summary = summarise(preds, folds)
    summary.update(name=a.name, model=a.model, mic=a.mic,
                   shared_prompts=a.shared_prompts,
                   no_level_features=a.no_level_features,
                   features=Path(a.features).name,
                   n_features=split_meta_features(df)[1].shape[1])

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    # the prompt text is recoverable from the manifest via utt_id; dropping
    # it keeps the committed prediction files small
    preds.drop(columns=["prompt"], errors="ignore").to_csv(
        out / f"preds_{a.name}.csv", index=False)
    folds.to_csv(out / f"folds_{a.name}.csv", index=False)
    (out / f"summary_{a.name}.json").write_text(json.dumps(summary, indent=2))

    print("\n--- summary ---")
    for k, v in summary.items():
        print(f"  {k:26s} {v}")

    print("\n--- utterance accuracy by severity ---")
    print(folds.groupby("severity")
               .agg(speakers=("speaker", "nunique"),
                    utts=("n_utts", "sum"),
                    mean_utt_acc=("utt_acc", "mean"),
                    speakers_correct=("spk_correct", "sum"))
               .round(3).to_string())


if __name__ == "__main__":
    main()
