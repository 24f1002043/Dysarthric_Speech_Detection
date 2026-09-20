#!/usr/bin/env bash
# Full pipeline. Usage: bash scripts/run_all.sh [python_executable]
#
# Every experiment fixes ONE microphone for BOTH groups.
#   primary     : head-mounted mic  (close-talking, available for all 15 speakers)
#   replication : array mic         (same protocol, different channel)
# Mixing the two is exactly what src/mic_confound.py shows you must not do.
set -euo pipefail
PY="${1:-.venv/Scripts/python.exe}"

echo "== 1. manifest =="
$PY src/torgo_manifest.py --root data/TORGO --out data/manifest.csv

echo "== 2. eGeMAPS features (both mics, so the mic audit is possible) =="
$PY src/extract_egemaps.py --manifest data/manifest.csv --mic both --out features/egemaps.parquet

echo "== 3. frozen WavLM features =="
$PY src/extract_wavlm.py --manifest data/manifest.csv --mic both --out features/wavlm.parquet --threads $(nproc 2>/dev/null || echo 8)

echo "== 4. PRIMARY: leave-one-speaker-out, head-mounted mic for both groups =="
$PY src/run_experiment.py --features features/egemaps.parquet --model xgb    --mic head --name egemaps_xgb
$PY src/run_experiment.py --features features/wavlm.parquet   --model logreg --mic head --name wavlm_logreg
$PY src/run_experiment.py --features features/wavlm.parquet   --model mlp    --mic head --name wavlm_mlp

echo "== 5. REPLICATION: same protocol on the array microphone =="
$PY src/run_experiment.py --features features/egemaps.parquet --model xgb    --mic array --name egemaps_xgb_array
$PY src/run_experiment.py --features features/wavlm.parquet   --model logreg --mic array --name wavlm_logreg_array

echo "== 6. CONTROL: restrict to prompts BOTH groups actually spoke =="
$PY src/run_experiment.py --features features/egemaps.parquet --model xgb    --mic head --shared-prompts --name egemaps_xgb_sharedprompt
$PY src/run_experiment.py --features features/wavlm.parquet   --model logreg --mic head --shared-prompts --name wavlm_logreg_sharedprompt

echo "== 6b. CONTROL: does it still work without absolute loudness? =="
$PY src/run_experiment.py --features features/egemaps.parquet --model xgb    --mic head --no-level-features --name egemaps_xgb_nolevel

echo "== 7. SHAP explanation =="
$PY src/explain_shap.py --features features/egemaps.parquet --mic head

echo "== 8. microphone confound audit =="
$PY src/mic_confound.py --features features/egemaps.parquet

echo "== 9. aggregate report =="
$PY src/analyze.py --outdir results
$PY src/make_report.py --outdir results --out RESULTS.md
