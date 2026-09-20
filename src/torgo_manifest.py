"""Build an utterance-level manifest of the TORGO corpus.

Layout produced by scripts/fetch_torgo.sh:
    data/TORGO/<SPK>/Session<N>[_3]/wav_headMic/<uid>.wav
    data/TORGO/<SPK>/Session<N>[_3]/wav_arrayMic/<uid>.wav
    data/TORGO/<SPK>/Session<N>[_3]/prompts/<uid>.txt

The manifest records BOTH microphones so that the choice of microphone is an
explicit experimental variable rather than an accident of what happened to be
on disk (see README, "The microphone trap").
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf
from tqdm import tqdm

# --- Speaker metadata -------------------------------------------------------
# 8 dysarthric + 7 control speakers.
DYSARTHRIC = ["F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05"]
CONTROL = ["FC01", "FC02", "FC03", "MC01", "MC02", "MC03", "MC04"]

# Severity of the dysarthric speakers, grouped from the Frenchay Dysarthria
# Assessment intelligibility scores reported in Rudzicz et al. (2012).
# These groupings follow the convention used throughout the TORGO literature.
# Controls are not assessed (the corpus assumes normal function) -> "control".
SEVERITY = {
    "F01": "severe",
    "M01": "severe",
    "M02": "severe",
    "M04": "severe",
    "F03": "moderate",
    "M05": "moderate",
    "F04": "mild",
    "M03": "mild",
}

# Prompts that are instructions to the participant rather than speech to model.
NON_SPEECH_PROMPT = re.compile(
    r"(?:xxx|^\s*\[.*\]\s*$|describe .*(?:image|picture)|input/output|"
    r"relax|smile|open (?:your )?mouth|stick out|breathe|swallow)",
    re.IGNORECASE,
)


def speaker_meta(spk: str) -> dict:
    return {
        "speaker": spk,
        "group": "dysarthric" if spk in DYSARTHRIC else "control",
        "label": 1 if spk in DYSARTHRIC else 0,
        "sex": "F" if spk.startswith("F") else "M",
        "severity": SEVERITY.get(spk, "control"),
    }


def read_prompt(prompt_dir: Path, uid: str) -> str | None:
    p = prompt_dir / f"{uid}.txt"
    if not p.exists():
        return None
    try:
        return p.read_text(errors="ignore").strip()
    except OSError:
        return None


def build(root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    speakers = sorted(DYSARTHRIC + CONTROL)
    for spk in tqdm(speakers, desc="speakers"):
        sdir = root / spk
        if not sdir.is_dir():
            print(f"  ! missing speaker dir: {sdir}", file=sys.stderr)
            continue
        meta = speaker_meta(spk)
        for sess in sorted(p for p in sdir.iterdir() if p.is_dir()):
            prompts = sess / "prompts"
            for mic, mdir in (("head", sess / "wav_headMic"),
                              ("array", sess / "wav_arrayMic")):
                if not mdir.is_dir():
                    continue
                for wav in sorted(mdir.glob("*.wav")):
                    uid = wav.stem
                    try:
                        info = sf.info(str(wav))
                        dur, sr = info.duration, info.samplerate
                    except Exception:            # unreadable / truncated file
                        continue
                    text = read_prompt(prompts, uid)
                    rows.append({
                        **meta,
                        "session": sess.name,
                        "mic": mic,
                        "uid": uid,
                        "utt_id": f"{spk}_{sess.name}_{mic}_{uid}",
                        "path": str(wav.resolve()),
                        "duration": round(dur, 4),
                        "samplerate": sr,
                        "prompt": text,
                        "has_prompt": text is not None,
                    })
    return pd.DataFrame(rows)


def apply_quality_filter(df: pd.DataFrame, min_dur: float, max_dur: float,
                         require_prompt: bool = False) -> pd.DataFrame:
    """Drop what is demonstrably not speech, and nothing else by default.

    A missing prompt file is a gap in TORGO's metadata, not evidence that the
    recording is unusable -- and the gap is not evenly spread across speakers.
    Dropping on it would quietly remove more of one group than the other, which
    is the kind of silent filtering this project is trying to avoid. So by
    default an utterance is kept unless its prompt is explicitly an instruction
    ("[relax your mouth]") or its duration is implausible. Pass
    --require-prompt to use only utterances with a known transcript.
    """
    df = df.copy()
    prompt = df["prompt"].fillna("")
    df["is_speech_prompt"] = ~prompt.str.contains(NON_SPEECH_PROMPT, na=False)
    df["keep"] = df["duration"].between(min_dur, max_dur) & df["is_speech_prompt"]
    if require_prompt:
        df["keep"] &= df["has_prompt"]
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/TORGO")
    ap.add_argument("--out", default="data/manifest.csv")
    ap.add_argument("--min-dur", type=float, default=0.30)
    ap.add_argument("--max-dur", type=float, default=30.0)
    ap.add_argument("--require-prompt", action="store_true",
                    help="keep only utterances with a prompt transcript")
    a = ap.parse_args()

    df = build(Path(a.root))
    if df.empty:
        sys.exit(f"No audio found under {a.root}")
    df = apply_quality_filter(df, a.min_dur, a.max_dur, a.require_prompt)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)

    print(f"\nWrote {a.out}: {len(df)} utterances, {df.speaker.nunique()} speakers")
    print("\n--- utterances kept, by speaker x mic ---")
    k = df[df.keep]
    print(pd.crosstab(
        [k.group, k.severity, k.speaker], k.mic, margins=True
    ).to_string())
    print("\n--- microphone coverage (speakers per mic) ---")
    print(k.groupby("mic").agg(
        speakers=("speaker", "nunique"),
        dysarthric=("speaker", lambda s: s[k.loc[s.index, "label"] == 1].nunique()),
        control=("speaker", lambda s: s[k.loc[s.index, "label"] == 0].nunique()),
        utts=("utt_id", "count"),
        hours=("duration", lambda d: round(d.sum() / 3600, 2)),
    ).to_string())


if __name__ == "__main__":
    main()
