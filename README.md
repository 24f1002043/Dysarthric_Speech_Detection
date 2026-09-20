# Dysarthric Speech Detection on TORGO


Binary detection (dysarthric vs. control) on the University of Toronto **TORGO**
corpus, evaluated **leave-one-speaker-out** across all 15 speakers, with two
independent model families, a SHAP explanation of *what the model listens to*,
and an explicit audit of the recording-channel confound that makes many
published TORGO numbers look better than they are.

---

## Results at a glance

Leave-one-speaker-out over all 15 speakers, head-mounted microphone for both
groups. `speaker_correct` is how many of the 15 people were classified
correctly from the mean probability across their utterances — the number a
screening tool actually cares about.

| Model | Features | Utt. accuracy | AUC | Speakers |
|---|---:|---:|---:|---:|
| **WavLM (frozen) + logistic regression** | 1536 | **0.822** | **0.883** | **14/15** |
| WavLM (frozen) + small MLP | 1536 | 0.803 | 0.870 | 13/15 |
| eGeMAPS + XGBoost | 88 | 0.720 | 0.765 | 13/15 |
| *control:* only prompts both groups spoke (WavLM) | 1536 | 0.805 | 0.875 | 14/15 |
| *control:* only prompts both groups spoke (eGeMAPS) | 88 | 0.729 | 0.771 | 13/15 |
| *control:* eGeMAPS with absolute loudness removed | 82 | 0.575 | 0.589 | 10/15 |
| *replication:* WavLM on the array microphone | 1536 | 0.695 | 0.729 | 10/15 |
| *replication:* eGeMAPS on the array microphone | 88 | 0.556 | 0.590 | 10/15 |

Four things are worth noticing:

1. **Frozen WavLM beats hand-designed acoustics by a wide margin** (0.883 vs
   0.765 AUC) without any fine-tuning — only a linear model on top.
2. **The controls hold.** Restricting to the prompts both groups spoke barely
   moves either model, so neither is reading the script instead of the speech.
3. **The microphone matters more than the model.** Moving from the
   head-mounted mic to the far-field array costs more accuracy than switching
   between the two model families. WavLM degrades more gracefully than eGeMAPS
   (0.729 vs 0.590 AUC), but neither survives the channel change well.
4. **These numbers are far below the high-90s often quoted on TORGO**, and that
   is the point. The microphone audit below shows how a 0.999 AUC is available
   to anyone who does not control the recording channel.

Full tables, per-speaker breakdowns and figures: **[RESULTS.md](RESULTS.md)**.

---

## Why this problem, done this way

Dysarthria is acoustically distinct, so a detector can reach high accuracy
easily. That is exactly why the *protocol* matters more than the headline
number. Three decisions carry most of the scientific weight here:

1. **Leave-one-speaker-out (LOSO).** 15 speakers is a small pool. Any split
   that puts the same speaker in train and test measures speaker recognition,
   not disorder detection. Every fold here holds out one entire speaker — not
   one utterance of theirs is seen, and even the feature scaler is fit on the
   training speakers only.
2. **One microphone for both groups.** TORGO records every session with a
   head-mounted mic *and* an array mic. The two channels are trivially
   separable. If the microphone is not held constant across groups, a model can
   score near-perfectly while knowing nothing about speech. This repo
   **measures** that effect rather than just asserting it (see below).
3. **Frozen self-supervised features.** WavLM is used as a fixed feature
   extractor with only a small classifier trained on top. Fine-tuning a
   94M-parameter encoder on 15 speakers would memorise the speakers.
4. **Matched material.** TORGO speakers do not all read the same prompts, so a
   detector could in principle key on *what* was said. Every headline result is
   repeated on the subset of prompts that both groups actually spoke
   (`--shared-prompts`); if accuracy survives, the decision was about the
   speech and not the script.

---

## The microphone trap, quantified

`src/mic_confound.py` runs the audit. It is the most important result in the
repo, because it shows what the careful protocol is protecting against:

| Experiment | What it asks |
|---|---|
| **A — matched** | train head-mic → test head-mic (the protocol used everywhere else here) |
| **B — mismatched** | train head-mic → test array-mic: how much of the decision was channel, not disorder? |
| **C — channel probe** | can the *same eGeMAPS features* identify which microphone was used? |
| **D — confounded** | deliberately give dysarthric speakers one mic and controls the other |

What this run actually found (utterance-level AUC, eGeMAPS + XGBoost, LOSO):

| Experiment | Accuracy | AUC |
|---|---:|---:|
| A — matched, head mic (the honest protocol) | 0.720 | **0.765** |
| A' — matched, array mic | 0.556 | 0.590 |
| B — train head, test array | 0.233 | **0.135** |
| B' — train array, test head | 0.253 | 0.283 |
| C — predict the *microphone* from the same features | 0.995 | **1.000** |
| D — confounded: dysarthric=head, control=array | 0.968 | **0.999** |
| D' — confounded, swapped | 0.993 | 1.000 |

Read row **D** against row **A**. The confounded setup scores an AUC of 0.999
and the honest one scores 0.765 — a "better" model that has learned nothing
about speech. Row **C** shows why it is so easy: the microphone is perfectly
recoverable from the very features the detector uses. Row **B** is the sharpest
part: a model trained on one microphone and tested on the other lands at AUC
0.135, *below chance*, meaning its decisions invert when only the channel
changes. Whatever it learned, most of it was not dysarthria.

A practical wrinkle worth knowing: every speaker has both microphones, but the
**amounts are very uneven** — some speakers have far more array-mic than
head-mic utterances, and one has the reverse. So "use the same microphone" is
not free: it changes how much data each speaker contributes. The manifest
records the microphone for every utterance, the headline protocol fixes the
head-mounted channel for both groups, and the array channel is run as a
replication so the choice is visible rather than silent.

---

## Models

| Model | Features | Classifier | Why |
|---|---|---|---|
| **eGeMAPS + XGBoost** | 88 interpretable functionals (pitch, loudness, jitter, shimmer, rate, spectral shape) from openSMILE | gradient-boosted trees | interpretable, strong on small data, and SHAP-friendly |
| **WavLM (frozen) + linear / MLP** | mean+std pooled hidden states from `microsoft/wavlm-base-plus` | logistic regression / small MLP | learned representations, no speaker memorisation |

Both are scored with the same LOSO loop, and utterances are weighted so that
each speaker — and then each class — contributes equally. Without that, a few
talkative speakers would define the decision boundary.

---

## Explainability

`src/explain_shap.py` computes SHAP values for the XGBoost model **on held-out
utterances only**, so the explanation describes genuine out-of-sample
behaviour. Outputs land in `results/`:

- `shap_summary.png`, `shap_bar.png` — which acoustic parameters drive detection
- `shap_feature_importance.csv` — ranked, with plain-English feature names
- `shap_by_severity.csv` — how the explanation shifts from severe to mild cases

This is the speech analogue of a Grad-CAM saliency map: it turns a yes/no
classifier into a claim a clinician can argue with.

### What SHAP found, and the awkward question it raised

The ranking is dominated by **vocal loudness** — loudness in quiet passages is
roughly three times more influential than the next feature — followed by
**H1–H2** (a glottal-source measure related to breathiness), **pitch
variation**, **pause-length variation** and **syllable rate**. Those are
recognisable dysarthria correlates: reduced vocal intensity, disordered
phonation, flattened prosody and slowed, irregular timing.

The *directions* are clinically coherent too, which is the part worth checking.
In `results/shap_summary.png`, **low** loudness pushes toward "dysarthric"
(hypophonia), **low** pitch variation pushes toward "dysarthric" (monotone
prosody), and **long, irregular pauses** push the same way. A model that had
latched onto an artefact would have no reason to line up with the clinical
picture like that.

But "loudness" is also the feature most easily contaminated by recording gain
and microphone distance, so it is worth asking whether the detector is really
just a loudness meter. The repo answers that directly rather than hoping nobody
asks — `--no-level-features` drops every absolute loudness and sound-level
statistic (keeping scale-invariant ones such as the coefficient of variation):

| Model | AUC | Speakers correct |
|---|---:|---:|
| eGeMAPS + XGBoost, all 88 features | **0.765** | 13/15 |
| the same, with 6 absolute level features removed | 0.589 | 10/15 |

So most of the detectable signal here *is* vocal intensity. That is a real
finding, not a bug: reduced loudness (hypophonia) is a genuine clinical sign.
But it cannot be cleanly separated from recording level in this corpus, and it
would be dishonest to report 0.765 without saying so. A deployed screening tool
would need calibrated recording conditions, or level-normalised features and a
correspondingly lower expected accuracy.

---

## Severity breakdown

TORGO's dysarthric speakers carry Frenchay Dysarthria Assessment intelligibility
ratings, grouped here as **mild / moderate / severe**. The grouping is the one
conventional in the TORGO literature and lives in one place — the `SEVERITY`
dict in `src/torgo_manifest.py` — so it can be checked against the corpus
documentation or changed without touching the pipeline.

`src/analyze.py` reports accuracy per severity band. For the better model
(frozen WavLM + logistic regression):

| Severity | Speakers | Utt. accuracy | Speakers correct |
|---|---:|---:|---:|
| severe | 4 | 0.987 | 4/4 |
| moderate | 2 | 0.690 | 1/2 |
| **mild** | 2 | **0.572** | 2/2 |
| control | 7 | 0.858 | 7/7 |

Severe dysarthria is detected essentially perfectly. **Mild dysarthria is the
hard case** — barely above chance at the utterance level, and only recoverable
at the speaker level by averaging many utterances together. That is the
clinically awkward shape: the closer speech is to typical, the worse the
detector does, and mild cases are exactly what early screening exists to catch.
A screening tool built on this would need to listen to a whole session, not a
single word, before saying anything about a mild case.

Mild is the worst band in **all four** WavLM runs (head, array, shared-prompt,
MLP), so this is not an artefact of one configuration. eGeMAPS disagrees — it
finds *moderate* hardest, because its single largest error is one moderate
speaker (F03). That disagreement is itself informative: with two speakers per
band, band-level numbers track individuals, and the per-speaker table in
[RESULTS.md](RESULTS.md) is the honest unit of analysis.

The honest caveat is that **each severity band contains only two to four
speakers**, so a band's number is really a statement about those individuals.
In this run the largest single error is one moderate speaker (F03) whom the
model reads as a control; that one person moves the whole "moderate" band. This
is why `results/` reports per-speaker accuracy next to every aggregate, and why
the severity figure should be read as a description of 15 people rather than an
estimate of a population.

---

### A note on speaker counts

The released corpus contains **8 dysarthric speakers** (F01, F03, F04, M01–M05)
and **7 controls** (FC01–FC03, MC01–MC04) — 15 in total, hence 15 LOSO folds.
Note that F02 is absent from the release.

---

## Questions this project is built to survive

**"Why is the accuracy only ~72% when papers report the high 90s?"**
Because the folds are speaker-disjoint. `tests/test_loso.py` asserts that no
held-out speaker appears in its own training set, and that a random
utterance-level split scores *higher* than LOSO — which is the direction it must
go. Much of the headroom in published TORGO numbers comes from splits that let
a speaker appear on both sides, or from an uncontrolled recording channel; the
microphone table above shows what that is worth (AUC 0.999, and meaningless).

**"TORGO has two microphones. Did you control for that?"**
Yes, and it is measured rather than asserted. The same features can identify
the microphone almost perfectly, and deliberately aligning microphone with
group yields the highest accuracy in the repo — a result included precisely
because it is worthless. Every real result fixes one microphone for both groups
and is replicated on the other channel.

**"Do the groups read the same material?"**
Not entirely, so every headline result is repeated on the prompts both groups
actually spoke.

**"What is the model actually listening to?"**
SHAP over held-out utterances, with readable parameter names: vocal loudness
first, then H1–H2 (breathiness), pitch variation, pause-length variation and
syllable rate. And because loudness dominates, the repo also reports what
happens when every absolute level feature is removed — AUC falls from 0.765 to
0.589, which is stated rather than hidden.

**"Where does it fail?"**
On everything that is not severe, and on two specific people: F03, a moderate
dysarthric speaker read as a control, and MC04, a control read as dysarthric.
Both are reported per speaker rather than averaged away. With two to four
speakers per severity band, one person moves a whole band, which is itself the
limitation worth leading with.

**"Is 15 speakers enough?"**
No. Fifteen speakers is a pilot, not evidence of clinical generalisation. With
this many speakers, one speaker is 6.7% of speaker-level accuracy, so
confidence intervals are wide and per-speaker results matter more than the
pooled mean. That is why per-speaker tables are reported alongside every
headline number.

---

## Repository layout

```
src/torgo_manifest.py    build the utterance manifest (microphone, severity, prompt, filters)
src/extract_egemaps.py   openSMILE eGeMAPSv02 functionals, parallelised
src/extract_wavlm.py     frozen WavLM embeddings, masked mean+std pooling (GPU or CPU)
src/loso.py              leave-one-speaker-out engine, weighting and metrics
src/run_experiment.py    one LOSO run; --shared-prompts and --no-level-features controls
src/explain_shap.py      held-out SHAP explanation with readable feature names
src/mic_confound.py      the microphone audit (matched / mismatched / probe / confounded)
src/analyze.py           aggregate tables and figures
src/make_report.py       render RESULTS.md from results/ -- no numbers typed by hand
src/predict.py           score a single recording, with the reasons for the score
scripts/fetch_torgo.sh   resumable download; extracts audio + prompts only
scripts/run_all.sh       the whole pipeline end to end
scripts/make_colab_notebook.py  regenerate the Colab notebook from src/
notebooks/colab_train.ipynb     self-contained GPU run
tests/test_loso.py       asserts the folds really are speaker-disjoint
```

---

## Reproducing

```bash
python -m venv .venv && .venv/Scripts/activate   # or source .venv/bin/activate
pip install -r requirements.txt
pip install torch transformers                   # for the WavLM model

for S in F FC M MC; do bash scripts/fetch_torgo.sh data/TORGO $S; done
bash scripts/run_all.sh
```

Run the tests that check the protocol itself (speaker-disjoint folds, balanced
weighting):

```bash
python -m pytest tests -q
```

### Compute

Everything here runs on a CPU. On 16 cores the whole pipeline is roughly:
eGeMAPS ~20 min, frozen WavLM ~2.5 h (about 6x real time, `--threads 16
--batch-size 8`), and every LOSO fit in minutes. For the GPU path, open
`notebooks/colab_train.ipynb` in Google Colab and set **Runtime → Change runtime
type → GPU**; it downloads TORGO, extracts features and runs the same LOSO code,
with WavLM extraction dropping to a few minutes. The notebook embeds `src/`
verbatim, so the Colab run and the local run execute identical code.

---

## Try it on a single recording

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

The probability is only meaningful for a speaker the model has not been trained
on, and the tool reports *why* it decided rather than just deciding.

---

## Data and licence

TORGO is distributed by the University of Toronto and is **free for academic
(non-profit) use**, on the condition that work using it cites the corpus. The
audio is *not* redistributed in this repository — `scripts/fetch_torgo.sh`
downloads it from the original source, and `data/` is git-ignored.

> Rudzicz, F., Namasivayam, A. K., & Wolff, T. (2012). The TORGO database of
> acoustic and articulatory speech from speakers with dysarthria.
> *Language Resources and Evaluation*, 46(4), 523–541.

The corpus page also lists two companion papers; any publication using these
data must cite at least one of the three.

**This is a research prototype, not a medical device.** It is trained on 15
speakers reading prompted material in a lab, and nothing here is validated for
clinical use or for any decision about a real person's care.
