"""LOSO on the quiet-frame descriptors written by nonspeech_probe.py."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from paper_experiments import masked_loso, score

BLOCKS = {"quiet": "quiet_", "quiet_shape": "quietshape_", "loud": "loud_"}
META = ["utt_id", "speaker", "group", "label", "sex", "severity", "session",
        "mic", "duration"]


def main() -> None:
    df = pd.read_parquet("features/nonspeech.parquet")
    out = Path("results/paper")
    res = {}
    for mic in ("head", "array"):
        d = df[df.mic == mic].reset_index(drop=True)
        for name, prefix in BLOCKS.items():
            cols = META + [c for c in d.columns if c.startswith(prefix)]
            for kind in ("logreg", "xgb"):
                p = masked_loso(d[cols], kind, n_jobs=4)
                key = f"{mic}__{name}__{kind}"
                res[key] = score(p)
                p.to_csv(out / f"preds_nonspeech_{key}.csv", index=False)
                print(f"{key:32s} acc={res[key]['utt_accuracy']:.3f} "
                      f"auc={res[key]['utt_auc']:.3f} "
                      f"spk={res[key]['speaker_correct']}/15 "
                      f"spk_auc={res[key]['speaker_auc']:.3f}", flush=True)
    # microphone from the quiet frames alone
    cols = META + [c for c in df.columns if c.startswith("quiet_")]
    p = masked_loso(df[cols], "logreg", label=(df.mic == "array").astype(int).to_numpy(), n_jobs=4)
    res["channel_probe__quiet__logreg"] = {k: v for k, v in score(p).items() if "utt" in k}
    print("channel probe", res["channel_probe__quiet__logreg"])
    (out / "nonspeech.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
