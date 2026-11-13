# Dysarthric Speech Detection on TORGO

Binary classification of dysarthric vs control speech on the TORGO corpus, evaluated with leave-one-speaker-out (LOSO) across all 15 speakers. Two model families (eGeMAPS + XGBoost, frozen WavLM + linear/MLP), SHAP explanations, and a check of the microphone confound in TORGO.

Paper: [What Is a Dysarthria Detector Listening To? Microphone, Speaker and Noise-Floor Shortcuts on TORGO](paper/main.pdf)
Abhiram Radhakrishnan, Gengaraj P, Praveen V. Submitted to the 2nd Workshop on AI in Big Data Analytics for Healthcare, IEEE BigData 2026 (under review).

Main result: with speaker and microphone controlled, frozen WavLM gets 0.883 AUC (14/15 speakers). If the microphone is allowed to line up with the diagnosis the AUC goes to 0.999, and training on one microphone and testing on the other drops it to 0.135.

## Results

LOSO over 15 speakers, head-mounted mic for both groups. `Speakers` is the number of speakers classified correctly using the mean probability over their utterances.

| Model | Features | Utt. accuracy | AUC | Speakers |
|---|---:|---:|---:|---:|
| WavLM (frozen) + logistic regression | 1536 | 0.822 | 0.883 | 14/15 |
| WavLM (frozen) + small MLP | 1536 | 0.803 | 0.870 | 13/15 |
| eGeMAPS + XGBoost | 88 | 0.720 | 0.765 | 13/15 |
| control: shared prompts only (WavLM) | 1536 | 0.805 | 0.875 | 14/15 |
| control: shared prompts only (eGeMAPS) | 88 | 0.729 | 0.771 | 13/15 |
| control: eGeMAPS without absolute loudness | 82 | 0.575 | 0.589 | 10/15 |
| replication: WavLM, array mic | 1536 | 0.695 | 0.729 | 10/15 |
| replication: eGeMAPS, array mic | 88 | 0.556 | 0.590 | 10/15 |

Notes:
- Frozen WavLM does much better than eGeMAPS (0.883 vs 0.765 AUC) with just a linear model on top.
- Using only prompts that both groups read barely changes the results, so the models are not keying on the text.
- Changing the microphone hurts more than changing the model. WavLM holds up better than eGeMAPS on the array mic (0.729 vs 0.590).
- These numbers are lower than the high-90s often reported on TORGO. The microphone section below shows where those come from.

Full tables, per-speaker results and figures are in [RESULTS.md](RESULTS.md).

## Setup choices

- LOSO: with only 15 speakers, any split that puts the same speaker in train and test ends up measuring speaker identity. Each fold holds out one whole speaker, and the scaler is fit on training speakers only.
- Same microphone for both groups: TORGO records with a head-mounted mic and an array mic, and the two are easy to tell apart. If the mic is not fixed, a model can score near perfect without learning anything about speech.
- Frozen WavLM: only a small classifier is trained. Fine-tuning a 94M parameter encoder on 15 speakers would just memorise them.
- Matched prompts: speakers did not all read the same prompts, so every main result is repeated on prompts both groups spoke (`--shared-prompts`).

## Microphone confound

`src/mic_confound.py` runs these experiments:

| Experiment | Question |
|---|---|
| A: matched | train head mic, test head mic (the protocol used everywhere else) |
| B: mismatched | train head mic, test array mic. How much of the decision was the channel? |
| C: channel probe | can the same eGeMAPS features predict which mic was used? |
| D: confounded | dysarthric speakers on one mic, controls on the other |

Results (utterance-level, eGeMAPS + XGBoost, LOSO):

| Experiment | Accuracy | AUC |
|---|---:|---:|
| A: matched, head mic | 0.720 | 0.765 |
| A': matched, array mic | 0.556 | 0.590 |
| B: train head, test array | 0.233 | 0.135 |
| B': train array, test head | 0.253 | 0.283 |
| C: predict the microphone | 0.995 | 1.000 |
| D: confounded, dysarthric=head, control=array | 0.968 | 0.999 |
| D': confounded, swapped | 0.993 | 1.000 |

D gets 0.999 AUC while A gets 0.765, and D has learned nothing about speech. C shows the mic can be recovered perfectly from the same features. B is below chance, so the model's decisions flip when only the channel changes.

Every speaker has both mics but the amounts are uneven. Some speakers have far more array-mic utterances than head-mic, and one has the reverse. The manifest stores the mic per utterance, the main protocol uses the head mic for both groups, and the array mic is run as a replication.

## Models

| Model | Features | Classifier |
|---|---|---|
| eGeMAPS + XGBoost | 88 openSMILE functionals (pitch, loudness, jitter, shimmer, rate, spectral shape) | gradient boosted trees |
| WavLM (frozen) + linear / MLP | mean+std pooled hidden states from `microsoft/wavlm-base-plus` | logistic regression / small MLP |

Both use the same LOSO loop. Utterances are weighted so each speaker, and then each class, contributes equally.

## Explainability

`src/explain_shap.py` computes SHAP values for the XGBoost model on held-out utterances only. Outputs in `results/`:

- `shap_summary.png`, `shap_bar.png`: which features drive the prediction
- `shap_feature_importance.csv`: ranked, with readable feature names
- `shap_by_severity.csv`: how the explanation changes from severe to mild

Top features: loudness in quiet passages (about 3x the next one), H1-H2 (breathiness), pitch variation, pause length variation and syllable rate. The directions match the clinical picture: low loudness (hypophonia), low pitch variation (monotone) and long irregular pauses all push toward dysarthric.

Loudness is also the feature most affected by recording gain and mic distance, so `--no-level-features` drops all absolute loudness/level features and keeps scale-invariant ones like the coefficient of variation:

| Model | AUC | Speakers correct |
|---|---:|---:|
| eGeMAPS + XGBoost, all 88 features | 0.765 | 13/15 |
| same, 6 absolute level features removed | 0.589 | 10/15 |

So most of the signal here is vocal intensity. Hypophonia is a real clinical sign, but in this corpus it cannot be separated from recording level. A real screening tool would need calibrated recording conditions or level-normalised features, with lower expected accuracy.

## Severity

Dysarthric speakers have Frenchay Dysarthria Assessment intelligibility ratings, grouped as mild / moderate / severe. The grouping is in the `SEVERITY` dict in `src/torgo_manifest.py`.

`src/analyze.py` reports accuracy per band. For WavLM + logistic regression:

| Severity | Speakers | Utt. accuracy | Speakers correct |
|---|---:|---:|---:|
| severe | 4 | 0.987 | 4/4 |
| moderate | 2 | 0.690 | 1/2 |
| mild | 2 | 0.572 | 2/2 |
| control | 7 | 0.858 | 7/7 |

Severe cases are detected almost perfectly. Mild is barely above chance per utterance and only works at speaker level by averaging many utterances. Mild is the worst band in all four WavLM runs (head, array, shared prompts, MLP). eGeMAPS finds moderate hardest because its biggest error is one moderate speaker (F03).

Each band has only 2 to 4 speakers, so band numbers are really about those individuals. That is why per-speaker accuracy is reported next to every aggregate.

## Speakers

The released corpus has 8 dysarthric speakers (F01, F03, F04, M01-M05) and 7 controls (FC01-FC03, MC01-MC04), so 15 LOSO folds. F02 is not in the release.

## FAQ

Why ~72% when papers report high 90s?
The folds are speaker-disjoint. `tests/test_loso.py` checks that no test speaker is in its training set, and that a random utterance-level split scores higher than LOSO. Much of the gap in published numbers comes from speaker leakage or an uncontrolled mic.

Was the microphone controlled?
Yes. Every main result fixes one mic for both groups and is replicated on the other.

Do both groups read the same material?
Not fully, so main results are repeated on shared prompts.

Where does it fail?
On non-severe cases, and on two speakers: F03 (moderate, predicted control) and MC04 (control, predicted dysarthric).

Is 15 speakers enough?
No. This is a pilot. One speaker is 6.7% of speaker-level accuracy, so confidence intervals are wide.

## Layout

```
src/torgo_manifest.py    utterance manifest (mic, severity, prompt, filters)
src/extract_egemaps.py   openSMILE eGeMAPSv02 functionals
src/extract_wavlm.py     frozen WavLM embeddings, masked mean+std pooling
src/loso.py              LOSO loop, weighting and metrics
src/run_experiment.py    one LOSO run; --shared-prompts and --no-level-features
src/explain_shap.py      SHAP on held-out data
src/mic_confound.py      microphone experiments
src/analyze.py           tables and figures
src/make_report.py       builds RESULTS.md from results/
src/predict.py           score one recording with reasons
scripts/fetch_torgo.sh   resumable download (audio + prompts only)
scripts/run_all.sh       full pipeline
scripts/make_colab_notebook.py  rebuilds the Colab notebook from src/
notebooks/colab_train.ipynb     GPU run
tests/test_loso.py       checks folds are speaker-disjoint
```

## Running

```bash
python -m venv .venv && .venv/Scripts/activate   # or source .venv/bin/activate
pip install -r requirements.txt
pip install torch transformers

for S in F FC M MC; do bash scripts/fetch_torgo.sh data/TORGO $S; done
bash scripts/run_all.sh
```

On Linux/macOS use `bash scripts/run_all.sh .venv/bin/python`.

Tests:

```bash
python -m pytest tests -q
```

Everything runs on CPU. On 16 cores: eGeMAPS about 20 min, WavLM about 2.5 h (`--threads 16 --batch-size 8`), LOSO fits in minutes. For GPU, open `notebooks/colab_train.ipynb` in Colab and set the runtime to GPU. The notebook contains the same `src/` code.

## Single recording

```bash
python src/predict.py --train features/egemaps.parquet --mic head \
       --wav data/TORGO/M01/Session1/wav_headMic/0001.wav
```

```
M01/Session1/wav_headMic/0001.wav
  P(dysarthric) = 0.997  ->  signs of dysarthria
  strongest evidence:
    Equivalent sound level (dB)              toward dysarthric  (shap +1.495)
    Pitch variation                          toward dysarthric  (shap +1.243)
    Loudness in quiet passages (20th pct)    toward dysarthric  (shap +0.875)
```

The probability only means something for a speaker the model was not trained on.

## Data

TORGO is from the University of Toronto and is free for academic (non-profit) use if you cite it. The audio is not included here; `scripts/fetch_torgo.sh` downloads it and `data/` is git-ignored.

> Rudzicz, F., Namasivayam, A. K., & Wolff, T. (2012). The TORGO database of acoustic and articulatory speech from speakers with dysarthria. Language Resources and Evaluation, 46(4), 523-541.

The corpus page lists two companion papers; work using the data must cite at least one of the three.

This is a research prototype, not a medical device. It is trained on 15 speakers reading prompts in a lab and is not validated for clinical use.

## License

Code is MIT (see [LICENSE](LICENSE)). The datasets keep their own terms and the paper PDF is not covered.
