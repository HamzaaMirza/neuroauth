# DECISIONS — NeuroAuth

An ADR log, written for a reader who was not in the room. Newest at the bottom within
each phase. Every entry states the alternatives considered and why they lost.

---

## Phase 1

### D-001 — MNE is confined to `dsp/io.py`

**Decision.** MNE is imported in exactly one module, the EDF loader. Preprocessing,
windowing, and feature extraction take `(n_channels, n_samples)` numpy arrays and
return numpy arrays.

**Alternatives.** Pass MNE `Raw` objects through the pipeline and use `raw.filter()`,
`mne.make_fixed_length_epochs`, and `raw.compute_psd()`. That is the idiomatic MNE
way and is less code in Phase 1.

**Why rejected.** In Phase 2 the WebSocket stream delivers raw arrays. There is no
`Raw` object anywhere in the serving path, so an MNE-shaped pipeline would have to be
reimplemented for streaming — and the reimplementation would be a second, subtly
different transform sitting between enrollment and verification. That is train/serve
skew, and it is the kind that produces a mysterious FRR regression rather than a
crash. Confining MNE now costs a few extra functions and buys an identical transform
on both paths.

**Secondary benefit.** The whole pipeline unit-tests against synthetic sinusoids with
no dataset on disk. Only `tests/test_io.py` needs the download.

---

### D-002 — Time-blocked splits with a guard band, never random

**Decision.** Train/test splits cut each recording into a contiguous leading block
and a contiguous trailing block, separated by a guard of at least one window length
(2.0 s for a 2 s window). `assert_no_window_overlap` runs on every split, in
production code, not only in tests.

**Alternatives.** `train_test_split(shuffle=True)`, or stratified k-fold over windows.

**Why rejected.** Windows overlap by 50%. A shuffled split places windows that share
half their samples on both sides of the boundary, so the classifier is scored partly
on its ability to recognize samples it was trained on. The resulting macro-F1
measures memorization, not identity.

**Consequence, stated up front.** The published number will sit below most figures
reported on eegmmidb. That is the correct outcome and the reason is a feature of the
write-up, not an apology.

---

### D-003 — Cross-condition (R01 → R02) is the headline split

**Decision.** The headline number trains on eyes-open and tests on eyes-closed. The
within-condition temporal split is reported alongside it as a second, easier number.

**Alternatives.** Pool R01 and R02 and report a single within-condition temporal
split.

**Why.** Not because pooling offers a shortcut — it does not. Every subject has both
runs, so condition is balanced across labels and carries no information about which
subject a window belongs to. The reason is that cross-condition measures whether
identity features survive a change in brain state, which is the actual deployment
question: a user enrolls calm and alert, then authenticates tired at the end of the
day. Eyes-open to eyes-closed is the closest approximation this dataset permits. A
model that only works within a single brain state is not an authenticator.

*(An earlier draft of this entry justified the split as removing a condition
shortcut. That reasoning was wrong and is recorded here so it does not reappear.)*

---

### D-004 — Relative band power is the headline; absolute is the comparison

**Decision.** `FeatureConfig.normalization` defaults to `"relative"` — each band
divided by that channel's total 1–50 Hz power. `"absolute_log"` is run as a second
configuration and both numbers are published, along with the gap between them.

**Alternatives.** Absolute log power as the primary feature, which is the more common
choice in the EEG-biometrics literature and generally scores higher.

**Why rejected as the headline.** eegmmidb is single-session. Every recording-specific
artifact — electrode impedance, cap placement, amplifier gain that day — is perfectly
confounded with subject identity, because each subject appears in exactly one
recording session. Absolute power carries those offsets straight into the feature
vector, so a model can score well by recognizing that a recording has slightly higher
broadband amplitude while learning nothing about the person. That is a leak wearing a
feature's clothes, and it would not survive contact with a second session.

**How the gap is reported.** If absolute scores materially higher, the delta is
published with the interpretation attached: the excess is attributed to
single-session amplitude confounds rather than to identity information. Quantifying
the confound is a better result than the higher number would have been.

**Known property.** The default bands tile 1–50 Hz without gaps, so relative values
sum to 1.0 per channel and one band per channel is linearly dependent on the others.
A tree ensemble is indifferent; a linear model would not be. Revisit if the model
family changes.

---

### D-004b — Frontal channels carry a second single-session confound: ocular artifact

*(Numbered out of sequence and placed here on purpose: this is the same underlying
problem as D-004, reached by a different path.)*

**The problem.** The frontal electrodes Fp1, Fp2, AF7 and AF8 sit just above the
eyes. They pick up electro-oculographic (EOG) activity — blinks and eye movements —
at amplitudes well above cortical EEG. On subjects 1–10 the median eyes-open 2 s
window on Fp1 and AF7 is ~300 µV peak-to-peak, against ~135 µV on C3 and Cz (D-015).
Blink rate, blink amplitude and eye-movement habits are person-specific and stable
within one sitting. In a single-session dataset they are therefore confounded with
identity exactly as amplitude offsets are: a model can learn how someone blinked
during that one recording and get credit for recognizing their brain.

**Why D-004 does not fix it.** Relative band power removes a per-channel gain. EOG
is not a gain. It changes spectral *shape*, piling power into delta at the frontal
channels, and relative power preserves shape. The confound passes through D-004
untouched.

**Why the headline split should partly resist it, and the temporal split won't.**
The cross-condition model trains on eyes-open, where blinks are frequent, and tests
on eyes-closed, where they mostly are not (Fp1 median ~146 µV). Blink features
learned in training are largely absent at test, so leaning on them should *lower*
the cross-condition score. The within-condition temporal split has blinks on both
sides and gets no such protection. This is an expectation to check, not a result.

**Decision.** Once the baseline trains, report the share of impurity importance on
Fp1/Fp2/AF7/AF8 for the eyes-open-trained model, next to the share those four
channels would carry if importance were uniform (4 of 64 = 6.25%). See
`channel_importance` and `importance_share`. Per-recording frontal flag rates go to
`artifacts/quality_flag_rates.csv`.

**If importance concentrates frontally,** that is a README finding in the same
section as the absolute/relative gap. A frontal-excluded feature variant then gets
run and reported next to the headline. It is not built until the check says it is
needed.

**Alternatives.** (a) Drop the frontal channels up front. That discards the channels
most likely to carry the confound *before measuring it*, and removes the finding.
(b) Remove EOG with regression or ICA, the standard clinical fix. eegmmidb has no
dedicated EOG channels (all 64 are scalp EEG), so this would mean ICA with frontal
channels as a proxy, where choosing which components to remove is itself a judgment
call. Disproportionate until the check shows a problem.

---

### D-005 — Welch `nperseg=160` inside a 2 s window

**Decision.** 160 samples per segment with 80 overlap: 1 Hz resolution, roughly three
averaged segments per window.

**Alternatives.** `nperseg=320` (the whole window) gives 0.5 Hz resolution but a
single unaveraged periodogram, whose variance does not shrink with averaging.
Lengthening the window to 4 s would allow both.

**Why rejected.** Lengthening the window is the expensive one: window length is the
floor on time-to-detect for an impostor swap, which is a headline Phase 2 metric. It
stays at 2 s. Between the two Welch settings, three averaged segments at 1 Hz beats
one unaveraged estimate at 0.5 Hz for band-power stability.

**Known cost.** Delta (1–4 Hz) gets roughly three bins. That is thin. Check delta's
aggregate feature importance once the baseline is fitted (`band_importance`); if it
contributes nothing, drop the band and record the result here.

**Outcome (full run from `b3450bf`, 2026-09-14).** Delta carries 13.4% of impurity
importance on the relative cross-condition model and 14.5% on the relative temporal
model — level with theta (13.2%, 14.2%) and alpha (14.1%, 14.3%). On the absolute
models it carries 6.8% and 9.6%. It is not contributing nothing, so it stays. Caveat:
relative band powers sum to one per channel, so importance is shared across linearly
dependent columns. This is a screen, not proof that delta adds information.

---

### D-006 — Common average reference off in Phase 1

**Decision.** `PreprocessConfig.common_average_reference` defaults to `False`.

**Why.** It is not in the Phase 1 scope, and leaving it off keeps each channel
independent of the others.

**The part that matters later.** CAR over 64 channels and CAR over an 8-channel
reduced montage are *different operations* — the reference signal is an average over
a different set of electrodes. If the channel-reduction experiment happens (the DEAP
work suggested only ~1.85pp loss going from 32 to 8 channels), the CAR flag has to be
a dimension of that experiment's design, not a fixed setting carried over from the
64-channel run. Comparing a CAR-64 model against a CAR-8 model conflates two changes.

---

### D-007 — Shuffled-label control on every committed run

**Decision.** Every evaluation that writes artifacts also fits the same model on
permuted training labels and reports the resulting macro-F1 alongside the real one
and alongside the chance level (1 / n_classes). If the control does not collapse to
roughly chance, the run aborts instead of writing artifacts.

**Why.** It costs one refit. It catches any path by which the true test labels reach
the predictions other than through training: test labels leaking into features or
into the fit, row/label misalignment, metric bugs. A macro-F1 with no chance level
next to it is also uninterpretable to a reader — for 89 enrollable subjects chance is
about 0.011, and that context belongs in the same table as the result.

**Pass criterion.** Shuffled-label macro-F1 at most 3× chance
(`check_shuffled_label_control`), fixed a priori (D-016).

The control is noisier than an independent-rows estimate suggests. A forest trained on
permuted labels still keeps each subject's test windows together, and likely votes much
of a subject's cluster onto a single random label, so each subject tends to land on its
own label all-or-nothing. The 5-subject smoke run on 2026-09-14 showed the effect: one
control read 0.41 against a chance level of 0.20 — inside the 3× ceiling, but twice
chance. With 89 classes the effect averages over many subjects and should shrink, but a
clean pipeline can occasionally breach 3×.

If a control fails, the run writes nothing, and the response is to investigate and
document — including the failing value — not to re-run permutation seeds until one
passes, and not to raise the ceiling.

*(An earlier version of this paragraph put the spread at "a few thousandths" by
treating rows as independent. Corrected after the smoke run and before the first full
run.)*

**What it cannot catch: the split leak.** A test window that shares samples with a
train window inherits that train window's label, and after permutation that label is
random. A model memorizing overlapping windows therefore reads as chance on this
control while its real score is inflated.
`test_shuffled_control_cannot_detect_window_overlap_leakage` demonstrates this.
Window-overlap leakage is ruled out structurally instead, by `assert_no_window_overlap`
on every split (D-002).

*(An earlier draft of this entry called the control "the only cheap check that catches
a leaking split." That was wrong for the leak this project worries about most, and is
recorded here so the claim does not reappear.)*

---

### D-008 — Impostor holdout fixed before any result: seeded, nested, committed

**Decision.** `neuroauth.cohorts.select_impostor_holdout` picks the impostor subjects
from a seeded permutation of the sorted subject ids, truncated to `n_impostors`.
`scripts/select_holdout.py` runs it once and writes `config/impostor_holdout.json`:
seed, candidate list, both cohorts, a UTC timestamp, and the numpy version. That file
is committed on its own, with a message marking it pre-results, before any
identification or verification result exists. Phase 1 trains and evaluates on the
enrollable cohort only.

Values: `seed=20260821`, `n_impostors=20`, candidates = all 109 subjects (D-014).

**The file, not the seed, is the source of truth.** Every consumer loads the list with
`load_holdout_record`, which checks structure but never re-derives from the seed.
`scripts/ingest_subjects.py` asserts that database cohorts match the file and aborts on
any mismatch. numpy does not promise that a seeded permutation stays identical across
versions, so re-deriving could silently yield a different holdout after an upgrade;
storing the list rules that out. `write_holdout_record` opens the file in
exclusive-create mode, so a second run cannot overwrite it.

**Evidence of timing.** A local commit date can be edited, so a commit alone proves
little. What does: the seed and count were committed in `9cdc613`, before any data
statistics were computed, and the selection is a deterministic function of them plus
the numpy version recorded in the file. Pushing the holdout commit to a remote before
any results commit adds an independent timestamp.

**Alternatives.** Choose the holdout in Phase 2, when the FAR-stability argument that
determines the right count can actually be made.

**Why rejected.** Choosing after seeing Phase 1 results is selection bias, and it
would compromise the Phase 2 EER before Phase 2 started. The count genuinely is a
Phase 2 question, which is why the selection is a *truncated permutation* rather than
a random sample: the chosen set is nested in `n_impostors`, so raising the count later
keeps every subject already committed to the holdout and only adds more. The Phase 2
decision about how many impostors a stable FAR estimate needs can therefore be made
on its merits without re-rolling anything.

**Enforcement.** Four layers: the committed file, written in exclusive-create mode;
the `holdout_never_enrolled` CHECK constraint in `migrations/001_phase1_core.sql`;
`assert_holdout_excluded`, raised explicitly on every training-set construction; and
`tests/test_cohorts.py`, which includes a check of the committed file itself.

**Selected impostor subjects** (from `config/impostor_holdout.json`, selected
2026-09-14T14:06:19Z; the file is authoritative if this list ever disagrees with it):
3, 11, 18, 20, 21, 35, 36, 38, 49, 65, 75, 79, 82, 86, 87, 96, 97, 105, 106, 107.

**Disclosure.** Subject 3 is in the holdout, and its baseline recordings were among
subjects 1–10 used for aggregate quality-mask statistics (D-015) before selection ran.
No model was fit and nothing identity-bearing was computed. The selection itself is
fixed by the seed committed in `9cdc613`, before those statistics existed, so it could
not have been steered by them.

---

### D-009 — Plain SQL migrations, no Alembic

**Decision.** Numbered `.sql` files in `migrations/`, applied in order and tracked in
a `schema_migrations` table by a ~40-line runner.

**Alternatives.** Alembic.

**Why rejected for now.** There are no ORM models in Phase 1, so Alembic's
autogeneration — the thing you actually pay it for — has nothing to work with. Hard
rule 7. Revisit in Phase 2 when SQLAlchemy models arrive; that is the point where it
starts earning its place.

---

### D-010 — `text` + CHECK instead of Postgres ENUM for `event_type`

**Decision.** Enumerated columns are `text` with a CHECK constraint.

**Why.** The event-type set grows in every remaining phase. Widening a CHECK is a
one-line constraint swap; `ALTER TYPE ... ADD VALUE` has transactional restrictions on
older Postgres and values cannot be removed from an enum at all.

---

### D-011 — `cohort` column exists in Phase 1

**Decision.** The `subjects.cohort` column and the `holdout_never_enrolled` CHECK ship
in the Phase 1 migration, even though enrollment is a Phase 2 feature.

**Alternatives.** Add it in Phase 2 alongside the templates table.

**Why rejected.** The alternative is a migration that introduces a security boundary
after data already exists on both sides of it. Shipping the constraint first makes
hard rule 1 a database invariant rather than a convention that application code is
trusted to respect.

---

### D-012 — `IdentificationReport` has no `accuracy` field

**Decision.** The report dataclass exposes macro-F1, per-class F1, a confusion matrix,
and the chance level. There is no accuracy field and none will be added.

**Why.** Hard rule 2. Making it structurally impossible to report is stronger than a
convention, and Phase 1's closed-set framing is exactly where the temptation would
otherwise arise.

---

### D-013 — Band power integrates exactly [low, high], with interpolated edges

**Decision.** `band_powers` inserts linearly interpolated PSD values at each band's
two edge frequencies, then applies the trapezoid rule over exactly `[low_hz,
high_hz]`.

**Alternatives.** (a) Sum the PSD bins inside the band. (b) Trapezoid over only the
bins falling in `[low_hz, high_hz)` — which is what the original contract described.
(c) Mean PSD over the band's bins, as many EEG pipelines do.

**Why rejected.** (b) silently drops one bin-width from every band. At the default
1 Hz resolution delta integrates 1–3 Hz instead of 1–4 Hz. For a flat spectrum the
shortfall is 33% delta, 25% theta, 20% alpha, 6% beta, 5% gamma — biased against
exactly the low bands, which then distorts relative power toward high bands. The
shortfall also changes with `welch_nperseg`, so retuning Welch would quietly change
every feature. (a) has the same edge problem plus a dependence on bin width. (c) is
resolution-stable but either double-counts the bin shared by adjacent bands or leaves
a gap, so bands don't tile.

**What the chosen method guarantees, and the tests that hold it:** the five default
bands sum to the 1–50 Hz integral to 1e-12 (`test_bands_tile_without_gaps`), and
broadband band power agrees within 5% between `nperseg=160` and `nperseg=320`
(`test_band_power_is_resolution_invariant`).

*(The original contract claimed resolution invariance while specifying method (b),
which cannot deliver it. Caught during implementation.)*

---

### D-014 — Dataset structural pass: all 218 baseline recordings are usable

**Finding.** Every subject's R01 and R02 load at 160 Hz with 64 channels. The
loader's skip path and `verify_dataset.py` both anticipated irregular recordings;
none exist in the baseline runs. No subject is dropped.

Five recordings are 9,600 samples (60.0 s) rather than the usual 9,760 (61.0 s):
S014R01, S051R01, S069R01, S097R02, S109R01. Each yields 59 windows instead of 60.
The resulting class imbalance is under 2% and is not corrected.

---

### D-015 — Quality mask: 500 µV threshold, reported but not excluded in Phase 1

**Finding.** The textbook 250 µV peak-to-peak threshold flagged 30% of windows on
subjects 1–10, and it was not detecting clipping:

- eyes-open: the median window on Fp1/AF7 was ~300 µV — ocular activity (D-004b)
- eyes-closed: O1 reached ~450 µV at the 95th percentile — occipital alpha, i.e. real
  signal
- uneven: S009 lost all 60 windows in both runs; S001R02, S003 and S010R02 lost about
  two-thirds

**Decision.**

1. `QualityConfig.max_peak_to_peak_v` raised to 500 µV. On subjects 1–10 that keeps 90%
   of eyes-open and 100% of eyes-closed windows, so the mask flags gross artifacts
   rather than physiology.
2. In Phase 1, flagged windows are **scored, not excluded**. Every report carries
   `n_test_not_ok` next to its score.
3. Per-recording flag rates, including frontal-channel flags, go to
   `artifacts/quality_flag_rates.csv`.

**Why not exclude.** Excluding flagged windows would have removed S009 from the class
set entirely and biased evaluation toward clean recordings — inflating exactly the
number D-002 and D-003 exist to keep honest. Deciding what to do with a low-quality
window is session logic, and session logic arrives in Phase 2.

**Alternatives.** (a) Keep 250 µV and exclude: rejected above. (b) Per-channel
thresholds calibrated from the data, e.g. median + k·MAD: the right Phase 2 answer,
premature while the mask gates nothing. (c) 350 µV: still marks 18% of eyes-open
windows not-ok.

**Provenance caveat.** The 500 µV figure came from subjects 1–10 before the holdout was
selected, and subject 3 is in the holdout (D-008). That is acceptable for Phase 1 only
because the threshold changes no score — nothing is excluded. Phase 2 recalibrates on
the enrollable cohort only.

---

### D-016 — Evaluation thresholds fixed a priori

**Decision.** Two thresholds were set on 2026-09-14, before any model had been fit on
real EEG, and are **not to be adjusted after seeing results**:

| Threshold | Value | Constant | What it decides |
|---|---|---|---|
| Shuffled-label control ceiling | 3× chance | `CONTROL_MAX_CHANCE_RATIO` | A run whose control exceeds it writes no artifacts (D-007) |
| Materiality of the absolute–relative gap | 0.05 macro-F1 | `MATERIAL_DELTA` | Above it, the gap is reported as a single-session amplitude confound (D-004) |

**Why a priori.** A threshold chosen after seeing the numbers can be placed to make a
control pass or a gap look immaterial, and nobody reading the result afterwards could
tell. Fixing both before the first real-data run is what makes the pass/fail and the
"material" label mean something.

**Enforcement.** `test_a_priori_thresholds_are_unchanged` pins both values. If it
fails, the fix is not editing the test. It is a new entry here explaining why the value
changed, with results reported under both the old and the new value.

The split parameters and seeds in `scripts/train_baseline.py` (train fraction 0.7,
guard equal to the 2 s window, control seed, leaky-split seed) were likewise fixed
before the first real-data run, and every run records them in `run_summary.json`.

---

### D-017 — A deliberately leaky random split, run as a labelled demonstration

**Decision.** `train_baseline.py` runs one extra evaluation on a shuffled row split
(`leaky_random_split`) over the same 50%-overlapping windows, with relative band
power. It is written to its own artifact, `artifacts/leakage_demonstration.json`,
labelled "DELIBERATELY LEAKY". It never produces a headline result, never goes through
`write_report`, and its output fails `assert_no_window_overlap` by design.

**Why.** D-002 argues that a random split over overlapping windows measures
memorization, and D-007 showed the shuffled-label control cannot see that leak. This
run turns the argument into a number on real data: the gap between the leaky score and
the guarded score. That gap is the README's "my number is lower because…", in figures.

**What it is compared against.** The guarded temporal split with the same
normalization. Both split windows within the same recordings, so the difference is
whether windows that share samples can straddle the boundary. Not the cross-condition
split, which also changes brain state and would mix two effects into one gap.

**What the gap includes.** Shared samples, and also plain temporal adjacency —
neighbouring windows resemble each other even where they share nothing, and the
guarded split discards those too. The artifact reports the fraction of leaky test
windows that actually share samples with a training window (expected about 91% at a
70/30 split, since each window overlaps its two neighbours: 1 − 0.3²), so a reader can
see how pervasive the overlap is.

**Also recorded.** The shuffled-label control on the leaky split — the real-data
counterpart of `test_shuffled_control_cannot_detect_window_overlap_leakage`. If D-007
holds, it reads near chance while the leaky score is inflated.
