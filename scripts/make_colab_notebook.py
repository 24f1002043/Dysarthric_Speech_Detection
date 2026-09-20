"""Generate a self-contained Colab notebook from the scripts in src/.

The notebook embeds the source files verbatim, so the GPU run on Colab is
exactly the pipeline in this repository rather than a copy that drifts.
"""
from __future__ import annotations

import base64
import json
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
FILES = ["torgo_manifest.py", "extract_egemaps.py", "extract_wavlm.py",
         "loso.py", "run_experiment.py", "explain_shap.py",
         "mic_confound.py", "analyze.py", "make_report.py", "predict.py"]

NL = chr(10)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip().splitlines(keepends=True)}


def payload() -> str:
    """zlib+base64 of all src files, so the unpack cell stays a single line."""
    blob = {f: (SRC / f).read_text(encoding="utf-8") for f in FILES}
    blob["fetch_torgo.sh"] = (ROOT / "scripts" / "fetch_torgo.sh").read_text(encoding="utf-8")
    raw = json.dumps(blob).encode("utf-8")
    return base64.b64encode(zlib.compress(raw, 9)).decode("ascii")


def payload_cell() -> str:
    return NL.join([
        "import base64, json, zlib, pathlib",
        '_P = "' + payload() + '"',
        "_files = json.loads(zlib.decompress(base64.b64decode(_P)).decode())",
        'pathlib.Path("src").mkdir(exist_ok=True); pathlib.Path("scripts").mkdir(exist_ok=True)',
        "for _n, _t in _files.items():",
        '    _d = "scripts" if _n.endswith(".sh") else "src"',
        '    pathlib.Path(_d, _n).write_text(_t, encoding="utf-8")',
        'print("wrote:", sorted(_files))',
    ])


def build() -> dict:
    cells = [
        md("""
# Dysarthric Speech Detection on TORGO - GPU run

Detecting dysarthria (slurred speech caused by cerebral palsy or ALS) from a
short recording, for early screening.

This notebook runs the whole pipeline on a Colab GPU:

1. Download TORGO (academic-use licence - cite Rudzicz et al., 2012)
2. Build an utterance manifest, recording the microphone explicitly
3. eGeMAPS + XGBoost, and frozen WavLM + a small classifier
4. **Leave-one-speaker-out** over all 15 speakers, one microphone for both groups
5. SHAP explanation, severity breakdown, and the microphone-confound audit

**Runtime -> Change runtime type -> GPU** before running.
"""),
        code("""
import torch, subprocess
print(subprocess.run(["nvidia-smi","--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "Set Runtime -> Change runtime type -> GPU"
"""),
        code("!pip -q install opensmile==2.6.0 xgboost shap soundfile librosa pyarrow transformers --upgrade"),
        md("## Unpack the pipeline" + NL + "The repository's `src/` files are embedded below."),
        code(payload_cell()),
        md("""
## 1. Download TORGO

Licence: free for **academic (non-profit)** use, and any publication using it
must cite Rudzicz et al. Only the audio and prompts are extracted; the
articulatory (.pos) and video data are discarded.
"""),
        code("""
!mkdir -p data/TORGO
!for S in F FC M MC; do bash scripts/fetch_torgo.sh data/TORGO $S; done
!ls data/TORGO
"""),
        md("## 2. Manifest (microphone recorded explicitly)"),
        code("!python src/torgo_manifest.py --root data/TORGO --out data/manifest.csv"),
        md("""
## 3. Features

eGeMAPS = 88 interpretable acoustic functionals (pitch, loudness, jitter,
shimmer, rate, spectral shape). WavLM is **frozen** - only a small classifier is
trained on top, so 15 speakers cannot be memorised by a 94M-parameter encoder.
"""),
        code("!python src/extract_egemaps.py --mic both --out features/egemaps.parquet"),
        code("!python src/extract_wavlm.py --mic both --out features/wavlm.parquet --batch-size 16"),
        md("## 4. Leave-one-speaker-out" + NL
           + "Primary protocol: head-mounted microphone for **both** groups."),
        code("""
!python src/run_experiment.py --features features/egemaps.parquet --model xgb    --mic head --name egemaps_xgb
!python src/run_experiment.py --features features/wavlm.parquet   --model logreg --mic head --name wavlm_logreg
!python src/run_experiment.py --features features/wavlm.parquet   --model mlp    --mic head --name wavlm_mlp
"""),
        md("## 5. Controls: array microphone, shared prompts, and no absolute loudness"),
        code("""
!python src/run_experiment.py --features features/egemaps.parquet --model xgb    --mic array --name egemaps_xgb_array
!python src/run_experiment.py --features features/wavlm.parquet   --model logreg --mic array --name wavlm_logreg_array
!python src/run_experiment.py --features features/egemaps.parquet --model xgb    --mic head --shared-prompts --name egemaps_xgb_sharedprompt
!python src/run_experiment.py --features features/wavlm.parquet   --model logreg --mic head --shared-prompts --name wavlm_logreg_sharedprompt
!python src/run_experiment.py --features features/egemaps.parquet --model xgb    --mic head --no-level-features --name egemaps_xgb_nolevel
"""),
        md("## 6. Explanation, severity and the microphone audit"),
        code("!python src/explain_shap.py --features features/egemaps.parquet --mic head"),
        code("!python src/mic_confound.py --features features/egemaps.parquet"),
        code("""
!python src/analyze.py --outdir results
!python src/make_report.py --outdir results --out RESULTS.md
"""),
        md("## 7. Download the results"),
        code("""
!zip -qr results.zip results RESULTS.md data/manifest.csv features/egemaps.parquet
from google.colab import files; files.download("results.zip")
"""),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 0,
    }


if __name__ == "__main__":
    out = ROOT / "notebooks" / "colab_train.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(build(), indent=1), encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size/1024:.0f} KB)")
