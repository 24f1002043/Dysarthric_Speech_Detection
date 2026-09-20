"""Render RESULTS.md from whatever is in results/ — no numbers typed by hand."""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

SEV_ORDER = ["control", "mild", "moderate", "severe"]


def md_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out", default="RESULTS.md")
    a = ap.parse_args()
    R = Path(a.outdir)

    L: list[str] = [
        "# Results", "",
        f"_Generated {date.today().isoformat()} by `src/make_report.py`._", "",
        "All numbers are **leave-one-speaker-out** over the 15 TORGO speakers: "
        "every fold trains on 14 speakers and tests on the held-out 15th.", "",
    ]

    # --- corpus ---
    mf = Path(a.manifest)
    if mf.exists():
        m = pd.read_csv(mf)
        k = m[m.keep]
        L += ["## Corpus after filtering", ""]
        cov = (k.groupby("mic")
                 .agg(speakers=("speaker", "nunique"),
                      utterances=("utt_id", "count"),
                      hours=("duration", lambda d: round(d.sum() / 3600, 2)))
                 .reset_index())
        L += [md_table(cov), "",
              f"Total utterances kept: **{len(k)}** of {len(m)} "
              f"({k.speaker.nunique()} speakers).", ""]

        drop = (m.assign(
                    no_prompt=~m.has_prompt,
                    instruction_prompt=m.has_prompt & ~m.is_speech_prompt,
                    too_short_or_long=~m.duration.between(0.30, 30.0))
                 .groupby("group")[["no_prompt", "instruction_prompt",
                                    "too_short_or_long"]].mean().round(3)
                 .reset_index())
        L += ["An utterance is dropped only when its prompt is explicitly an "
              "instruction rather than speech (e.g. *[relax your mouth]*) or its "
              "duration is implausible. A missing prompt transcript is a gap in "
              "TORGO's metadata, not evidence that a recording is unusable, so it is "
              "**not** a reason to drop by default. Any filter that removed much more "
              "of one group than the other would itself be a confound, so the rates "
              "are reported here (only the middle column affects `keep`):", "",
              md_table(drop), ""]

    # --- model comparison ---
    cmp_path = R / "model_comparison.csv"
    if cmp_path.exists():
        L += ["## Model comparison", "",
              md_table(pd.read_csv(cmp_path)), "",
              "`speaker_correct` counts speakers classified correctly by the mean "
              "probability over all their utterances — the number that matters for a "
              "screening tool, which sees a person, not an utterance.", ""]

    # --- severity ---
    sev_path = R / "severity_breakdown.csv"
    if sev_path.exists():
        s = pd.read_csv(sev_path)
        s["severity"] = pd.Categorical(s.severity, SEV_ORDER, ordered=True)
        L += ["## Accuracy by severity", "",
              md_table(s.sort_values(["run", "severity"]).round(4)), "",
              "![accuracy by severity](results/accuracy_by_severity.png)", "",
              "Mild dysarthria is the hard case, and it is the case that matters most "
              "for early screening: by the time a speaker is severe, detection is not "
              "the clinical question. A single pooled accuracy figure hides this.", ""]

    # --- mic confound ---
    mic_path = R / "mic_confound.json"
    if mic_path.exists():
        mc = json.loads(mic_path.read_text())
        rows = [{"experiment": k,
                 "utt_accuracy": round(v.get("utt_accuracy", float("nan")), 4),
                 "utt_auc": round(v.get("utt_auc", float("nan")), 4),
                 "speaker_accuracy": (round(v["speaker_accuracy"], 4)
                                      if "speaker_accuracy" in v else "")}
                for k, v in mc.items()]
        L += ["## The microphone confound", "",
              md_table(pd.DataFrame(rows)), "",
              "**How to read this.** `A_matched` is the honest protocol. "
              "`C_channel_probe` shows the microphone is almost perfectly recoverable "
              "from the same eGeMAPS features the detector uses. `D_confounded_*` "
              "aligns microphone with group on purpose and produces the highest "
              "accuracy in this repository — a microphone detector, not a dysarthria "
              "detector. `B_mismatch_*` shows how little of the matched model "
              "transfers when only the channel changes.", "",
              "This is why every headline result here fixes the microphone to the "
              "head-mounted channel for **both** groups.", ""]

    # --- shap ---
    shap_path = R / "shap_feature_importance.csv"
    if shap_path.exists():
        sh = pd.read_csv(shap_path).head(15)[["readable", "mean_abs_shap"]]
        sh.columns = ["acoustic parameter", "mean |SHAP|"]
        L += ["## What the model listens to (SHAP, held-out utterances)", "",
              md_table(sh.round(4)), "",
              "![shap](results/shap_bar.png)", "",
              "These are computed only on held-out speakers, so they describe "
              "out-of-sample behaviour rather than what the model memorised.", ""]

    # --- per-speaker ---
    for f in sorted(R.glob("folds_*.csv")):
        name = f.stem.replace("folds_", "")
        d = pd.read_csv(f).sort_values(["label", "severity", "speaker"])
        L += [f"## Per-speaker detail — `{name}`", "",
              md_table(d[["speaker", "group", "severity", "n_utts",
                          "utt_acc", "mean_prob", "spk_correct"]].round(3)), ""]
    if (R / "per_speaker_accuracy.png").exists():
        L += ["![per speaker](results/per_speaker_accuracy.png)", ""]

    L += ["---", "",
          "TORGO is used under its academic licence; see README for the required "
          "citation. This is a research prototype, not a medical device.", ""]

    Path(a.out).write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {a.out} ({len(L)} lines)")


if __name__ == "__main__":
    main()
