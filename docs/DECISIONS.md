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
vector, so a model can score well partly by recognizing a recording's broadband
amplitude — a leak wearing a feature's clothes, which would not survive contact with a
second session.

Absolute amplitude also carries anatomy: skull thickness, tissue conductivity, head
geometry. That part is person-specific and *would* survive a second session. With one
session per subject, the two cannot be told apart.

**How the gap is reported.** If absolute scores materially higher (D-016), the gap is
published as what it is: amplitude information that cannot be separated from session
artifacts, so the gap is an upper bound on their contribution — not a measure of it,
and not evidence that none of it is identity. Bounding the confound honestly is a
better result than the higher number would have been.

*(The first version of this entry attributed the gap to session artifacts "rather than
identity information". That claims a decomposition single-session data cannot
support. Corrected on 2026-09-14, after the first full run and before the result was
written up.)*

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

**Outcome (full run from `b3450bf`, 2026-09-14): does not concentrate.** On the
eyes-open-trained (cross-condition) models, Fp1/Fp2/AF7/AF8 carry 7.7% of impurity
importance with relative power and 8.4% with absolute — 1.23× and 1.35× the 6.25% they
would carry under uniform importance. Individually they rank 9th–20th of 64 channels,
and 11–13% of random 4-channel groups carry at least as much. The models trained on
both conditions (temporal split) put 6.6% and 5.5% there — 1.05× and 0.88×.

The contrast runs in the direction the hypothesis predicts: more frontal weight when the
training data is eyes-open, where blinks are frequent. The effect is too small to act
on, so the frontal-excluded variant is not run. No a priori criterion for
"concentrates" was set before the run; this judgment was made on these numbers and is
recorded as such.

The same run pointed elsewhere: in all four models the top channels are lateral-temporal
(T9, T10, T7, T8) and gamma is the top band. That is followed up as a possible
muscle-artifact confound in D-018.

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

**Decision.** Three thresholds are fixed and **not to be adjusted after seeing
results**. The first two were set on 2026-09-14, before any model had been fit on real
EEG. The third was set after the first full run and before either ablation was run.

| Threshold | Value | Constant | What it decides |
|---|---|---|---|
| Shuffled-label control ceiling | 3× chance | `CONTROL_MAX_CHANCE_RATIO` | A run whose control exceeds it writes no artifacts (D-007) |
| Materiality of the absolute–relative gap | 0.05 macro-F1 | `MATERIAL_DELTA` | Above it, the gap is reported as an upper bound on the session-artifact contribution (D-004) |
| Materiality of an ablation drop | 0.05 macro-F1 | `ABLATION_MATERIAL_DROP` | Above it, the headline is reported as depending on the removed features (D-018) |

**Why a priori.** A threshold chosen after seeing the numbers can be placed to make a
control pass, a gap look immaterial, or an ablation look harmless, and nobody reading
the result afterwards could tell. Fixing each one before the run it judges is what
makes the pass/fail and the "material" label mean something. The ablation threshold was
set knowing the headline (0.378) but not what either ablation would score.

**Enforcement.** `test_a_priori_thresholds_are_unchanged` pins all three values. If it
fails, the fix is not editing the test. It is a new entry here explaining why the value
changed, with results reported under both the old and the new value.

The split parameters and seeds in `scripts/train_baseline.py` (train fraction 0.7,
guard equal to the 2 s window, control seed, leaky-split seed) were likewise fixed
before the first real-data run, and every run records them in `run_summary.json`.

**Phase 2 additions.** The evaluation thresholds are in D-022. The session parameters were
fixed on 2026-09-16, from cohort scores and genuine-only cohort replays, before any holdout
session result:

| Threshold | Value | Constant | What it decides |
|---|---|---|---|
| Session unit | protected score, not the LLR | `session/logic.py` | The units every level below is in (D-025) |
| EMA half-life | 4 s | `PRE_REGISTERED_THRESHOLDS.ema_half_life_s` | How fast confidence follows the windows |
| Revoke below | 0.56 | `.revoke_below` | Terminal revocation |
| Revoke dwell | 3 decisions | `.revoke_dwell_decisions` | Consecutive sub-threshold decisions revocation needs |
| Challenge below | 0.58 | `.challenge_below` | Step-up prompt, no dwell |
| Recover above | 0.62 | `.recover_above` | A challenged session returns to active |

The same rule applies to them: not adjustable after holdout results, and a change needs an
entry here reporting results under both values. `tests/test_session_parameters.py` pins them,
and the reasoning is in D-025.

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

---

### D-018 — EMG ablations: bound the muscle-artifact contribution, don't decompose it

**Why.** In the first full run, all four models put their top importance on
lateral-temporal channels (T9, T10, T7, T8) and on the gamma band (38–48% of
importance). The temporal electrodes sit over the temporalis muscle, and 30–50 Hz is
where scalp muscle activity (EMG) shows up. Muscle tone and jaw habits are
person-specific and stable within a sitting, so EMG would be the same class of
single-session confound as amplitude (D-004) and eye movement (D-004b). The importance
pattern fits that; it does not prove it.

**Decision.** Two ablations of the headline model (relative power, eyes-open →
eyes-closed), each with its own shuffled-label control, judged against a materiality
threshold of 0.05 macro-F1 fixed before either was run (D-016):

1. **Without gamma.** Gamma is removed from `FeatureConfig.bands` and features are
   re-extracted, so relative power is renormalized over 1–30 Hz. Dropping the gamma
   column alone would not remove it: relative bands sum to one per channel, so gamma
   stays recoverable as one minus the other four
   (`test_a_dropped_relative_gamma_column_is_still_recoverable`).
2. **Without the temporal sites.** Every band at FT7, FT8, T7, T8, TP7, TP8, T9 and T10
   is removed (`config.TEMPORAL_EMG_CHANNELS`). The set is anatomical — every lateral
   temporal electrode in the montage — not the top of the importance ranking. Removing
   columns is exact here because relative power is per channel and CAR is off (D-006).

Results go to `artifacts/emg_ablation.json`.

**How the outcome is framed: a bound, not a decomposition.** Gamma at temporal sites
contains both muscle and neural activity, and scalp EEG cannot separate them. So:

- A **material drop** means the headline depends on information unique to the removed
  features, and the drop bounds that combined contribution from above. It does not say
  how much of it is EMG.
- A **drop within the threshold** means information unique to those features, muscle
  and neural together, is not a material part of the headline.
- **Either way the bound is scoped.** Muscle activity left in the retained features
  (beta at temporal sites in the gamma ablation, gamma at other sites in the channel
  ablation), and anything the forest recovers from correlated features, is not bounded
  by this check.

**Alternatives.** (a) Remove gamma and the temporal channels together: a bigger single
cut, but it cannot show which of the two carries the dependence. Not run. (b) EMG
removal by ICA or regression: component selection is itself a judgment call, and
eegmmidb has no reference EMG channels. (c) Adding Iz, near the neck muscles, which
also ranks high: it is not a temporal site, and adding it would shape the set around the
ranking. Not tested.

**Outcome (full run from `fa98250`, 2026-09-14): both drops material.**

| Headline model | Macro-F1 | Drop | Shuffled-label control |
|---|---|---|---|
| all features | 0.378 | — | 0.009 |
| without gamma | 0.263 | 0.116 | 0.009 |
| without the temporal sites | 0.300 | 0.078 | 0.007 |

Read as bounds: at most 0.116 of the headline comes from information unique to the
gamma band, and at most 0.078 from information unique to the eight temporal sites —
muscle and neural together in each case. Neither number is an estimate of EMG.

The importance shift makes the scope limit concrete. Without gamma, beta becomes the
top band (40%, up from 19%) and the temporal sites still lead the channel ranking.
Without the temporal sites, gamma stays the top band (38%), and Iz and the frontal
AF7/AF8 move to the top of the channel ranking. (Importance is recorded per band and per
channel, not per channel-band pair, so which channels carry the remaining gamma is not
known.) Each ablation leaves the other route open, so neither drop bounds the
model's dependence on temporal-site or high-frequency features as a whole. Removing
both together (alternative (a)) is the check that would. It has not been run, and if
it is, its materiality threshold has to be fixed first.

The same run reproduced all four baseline models' artifacts byte for byte against the
`b3450bf` run.

---

### D-019 — Docker Compose and the Postgres setup move from Phase 1 to Phase 2

**Decision.** Phase 1 closed on 2026-09-14 with the Postgres schema and Docker Compose
file written but never exercised. Running Compose, applying the schema, and implementing
`db/connection.py`, `db/migrate.py`, and `scripts/ingest_subjects.py` move to Phase 2.

**Why.** None of it was on the Phase 1 exit path, and nothing in Phase 1 reads or writes
the database. Phase 2 is the first phase with real consumers — enrollment, sessions,
events — so the database gets exercised against actual use rather than in isolation.
Docker was also not installed on the development machine.

**What carries over unchanged.** `migrations/001_phase1_core.sql` as written, including
the `holdout_never_enrolled` constraint. `ingest_subjects.py` must assert that database
cohorts match the committed `config/impostor_holdout.json` and never re-derive the
selection (D-008).

---

## Phase 2

### D-020 — Bounded-context filtering: offline and live features are identical by construction

**Decision.** Every Phase 2 window's features are a pure function of one raw slice: the
window plus 2 s before it and 2 s after it, filtered with the unchanged Phase 1 `preprocess`
and cropped back to the window (`neuroauth.dsp.streaming`). Offline, the slice is cut from
the recording. Live, the window is emitted once its right margin has arrived. Embedding
training, enrollment, evaluation, and live sessions all use this one path. Whole-recording
filtering (`pipeline.process_recording`) is Phase 1 only.

**The problem.** D-001 put one `preprocess` function on both paths. That removes code skew,
not buffer skew. `sosfiltfilt` gets zero phase by filtering forward and then backward. An
interior window of a 61 s recording sits seconds from either end, but the newest window of a
live stream sits exactly where the backward pass starts. On S001R01, filtering each 2 s
buffer on its own moved relative delta power by 0.013 (median) and 0.058 (95th percentile)
compared with whole-recording filtering. That is the same function giving different
features, and it would have surfaced as an unexplained FRR gap between evaluation and the
live demo.

**Alternatives.**

- **A warm-up buffer.** Past context removes the left-edge transient. The transient that
  matters is on the right, the side with no samples yet, and nothing in the past reaches it.
- **A causal filter with carried state** (`sosfilt` with `zi`). Chunked output equals
  one-shot output bit for bit, and it adds no latency. It lost because it changes the
  features: relative delta power moves by 0.033 (median) and 0.124 (p95) against Phase 1's,
  about a quarter of delta's window-to-window spread (0.145) at the median. Applying the
  cascade twice, so its magnitude response matches filtfilt's, made it worse (0.041 / 0.156).
  The cause is therefore group delay, not attenuation, and there is no cheap correction. It
  would also put a different pipeline under Phase 1's findings and confound the Phase 1 to
  Phase 2 tail comparison.
- **Filtering each buffer independently.** This is the trap described above.

**Margins.** The margins follow from a data-free criterion: the notch-plus-bandpass impulse
response stays below 1e-3 of its peak after 1.58 s (`SETTLING_TOLERANCE`), which rounds up to
the 1 s hop as 2 s. `check_margins` refuses anything shorter. The criterion was picked after
the measurements below had been seen, so the whole grid is published and a reader can judge
the choice. On S001R01, a 1 s margin on either side allows differences up to 1.3e-2 against
whole-recording filtering, 2 s + 2 s allows 1.4e-3, and 3 s + 3 s allows 2.8e-4.

**Cost: latency, and a floor on time-to-detect.** A session's first decision arrives 6 s
after stream start, and each later decision arrives 2 s after its window ends. After a
simulated swap at 30 s:

| Decision after the swap | What its input holds |
|---|---|
| +1 s (31 s) | first post-swap samples, in the right margin only; they reach the window through the filter tail |
| +3 s (33 s) | first post-swap samples inside the 2 s window (D-005) |
| +5 s (35 s) | a window entirely after the 0.5 s crossfade; its left margin still holds 1.5 s of pre-swap and crossfade signal |
| +7 s (37 s) | the first raw context that is entirely impostor signal |

*(This entry first gave 3 s and 5 s as the floors for "any" and "entirely" impostor signal.
Those figures counted only the window and ignored the 6 s of raw context each window is
filtered over. Corrected on 2026-09-15, while building the swap measurements.)*

These are floors set by the signal path, not expected detection
times. Session confidence accumulates over several windows, so measured time-to-detect is
longer. It is reported as a distribution beside the floor, never as the floor alone.

**Evidence.** `scripts/measurements/edge_effects.py` writes
`artifacts/measurements/edge_effects.json`, generated from committed code (`0a4abb7`, clean
tree). It measures signal-processing properties of one enrollable recording. It is not a
result, and no threshold judges it.

**Enforcement.** `tests/test_streaming.py` checks four things:

- a stream fed in random chunk sizes equals offline features bit for bit;
- bounded-context features stay within 5e-3 of whole-recording filtering;
- per-buffer filtering fails that same check, a negative control showing the check has teeth;
- margins shorter than the settling time are refused.

The 5e-3 tolerance judges a unit test, not a result, so D-016 does not govern it. It was
chosen after the measurement and is disclosed as such.

**Fingerprints.** `PipelineConfig` is unchanged, so Phase 1 fingerprints still reproduce.
Phase 2 features carry `StreamingConfig.fingerprint()`, which is namespaced so the two can
never collide.

---

### D-021 — The embedding is frozen; retraining adjusts a decision layer on stable templates

**Decision.** The *representation* is fixed when a template is enrolled, and the correction
loop never retrains it. The representation is `StreamingConfig`, the LDA embedding, and the
cancelable transform, versioned together as `representation_version`. Phase 3 instead
retrains a *decision layer*: a small logistic model from score-level inputs to a calibrated
log-likelihood ratio, versioned as `decision_version`. Templates never go stale. The
correction loop (review, training set, retrain, gate, promote) and the gate criteria are
unchanged; only the component they act on changes.

**The problem.** A template is `sign(R_k · E(x))`, bound to the embedding E that produced
it. Raw features are never persisted (hard rule 3), so a template cannot be recomputed under
a retrained E, only re-enrolled from new signal. A loop that retrains E would invalidate every
template on every promotion. No system forces its users to re-enroll weekly.

**Why retraining the embedding does not pay here.** A new representation pays only if it is
better. For EEG, "better" means robust across sessions, and single-session eegmmidb cannot
measure that. Every mechanism that would make a retrained embedding shippable (per-user
version state, serving several models at once, stateful rollback) would be built to deliver
an improvement this project cannot demonstrate.

**An architectural tension, not a limitation.** Hard rule 3 and representation learning pull
in opposite directions. Improving a representation from production evidence requires that
evidence at feature level, and the rule exists precisely so that feature-level evidence never
accumulates anywhere. A reviewed false rejection arrives long after its window's features
were discarded. `record_false_rejection` can hold only scores, quality flags, versions,
thresholds, and a verdict. This is not a gap to engineer around later: any design that closes
it does so by storing features in some form, which is what the rule forbids. The project
resolves the tension in favour of the rule and pays for it with a fixed representation.

**Alternatives.**

- **Retrain the embedding and re-enroll everyone on each promotion.** Correct and simple, but
  unshippable, because every promotion forces every user to re-enroll.
- **Keep several template versions and re-enroll lazily on the next successful
  authentication.**
  - It still cannot train on production rejections, because of the tension above. "Learns
    from false rejections" would be true only in the demo, where EDF files on disk let
    features be re-derived: a loop that could not exist in deployment.
  - Template poisoning: re-enrollment trusts whoever the session accepted, so a false accept
    becomes a permanent enrollment under the new version. Preventing that needs a
    re-enrollment bar far above the accept bar, which slows migration.
  - Version fan-out: users who authenticate rarely keep old versions alive, so retiring a
    version needs deadlines and forced re-enrollment for stragglers. Rollback becomes
    stateful.
  - The gate would compare populations mid-migration.
  - Its benefit cannot be measured on single-session data, and it carries a large Phase 2
    cost.
- **Backward-compatible representation learning** (Shen et al., "Towards
  Backward-Compatible Representation Learning", CVPR 2020). A new embedding is trained with a
  loss that keeps its outputs comparable with the old space, so old templates stay valid
  without re-enrollment. It is the standard industrial answer to the cost of re-indexing.
  It is rejected here specifically because **it requires stored features.** The
  compatibility loss is computed on training inputs, so improving the model from production
  data means keeping production features. The conflict with hard rule 3 is direct, not
  incidental. Training only on dataset recordings would sidestep the rule but not the
  tension. It would also tie every future model to the old space and add a cross-version
  check to the gate.
- **Escrow encrypted features for later re-derivation.** Encryption is invertible by design,
  and hard rule 3 forbids anything invertible from reaching storage.

**What the decision layer can and cannot learn.**

- It *cannot* fix a false rejection caused by the representation itself, such as the
  brain-state failures behind the Phase 1 per-subject tail. The README says so.
- It *can* learn calibration, quality handling, and per-template score offsets.
- Calibration alone is invisible to the gate. A monotone transform of one global score
  leaves the pooled DET curve, and so the EER, unchanged. A calibration-only retrain could
  never pass "EER improvement above threshold", so the layer takes inputs beyond the score.
- Key invariance limits the inputs. Each subject's projection is an independent random
  rotation, so bit *i* means a different direction for every user. Per-bit weights cannot be
  shared. Only summaries of bit agreement that ignore bit order can be, which in practice
  means the Hamming similarity.

**Decision layer v0** (Phase 2, fixed before any result):

- **Inputs, per window:** the Hamming similarity *s*; its z-score against the claimed
  template's enrollment statistics, `(s − μ) / max(σ, 1/n_bits)`; and `window_ok`.
- **The statistics:** μ and σ are the mean and standard deviation of each enrollment
  window's similarity to a leave-one-out template, built without that window. They are two
  scalars per template, not invertible, and are recorded on `ProtectedTemplate` at
  enrollment. They cannot be computed later, because the enrollment windows are gone.
- **Model:** logistic regression (lbfgs, C = 1.0). The training set's log prior odds are
  subtracted, so the output is a log-likelihood ratio.
- **Training data:** only genuine and cohort-impostor score rows, never holdout rows, which
  is asserted. In evaluation the layer is cross-fitted: fold k's scores come from a layer
  trained on the other folds' rows.

**Headline unchanged.** The Phase 2 headline was registered before this decision and stays:
FRR at FAR = 0.01 on protected-domain similarity, cross-condition split, holdout impostors.
The v0 decision layer is reported beside it and becomes the incumbent the Phase 3 gate
compares candidates against.

**Representation upgrades.** A better representation, such as the Phase 6 CNN compared
against the LDA embedding, is a deliberate migration that requires re-enrollment, outside the
correction loop. It is documented, not built. Templates carry `representation_version`, so
the path stays open without re-architecture.

**Phase 2 changes.**

- `representation_version` and `decision_version` replace the single model version.
- `ProtectedTemplate` gains the enrollment statistics.
- `WindowScore` and `WindowObservation` carry the LLR beside the raw similarity. Which of
  the two `update_session` uses, and so the units of its thresholds, is the author's call.
- Migration 002 records both versions on sessions.

---

### D-022 — Open-set verification: shared embedding, subject-disjoint folds, thresholds that never see the holdout

**Decision.** A probe window is scored against the claimed identity's template, using a
shared embedding fitted on other subjects. The embedding is log relative power,
standardized, shrinkage LDA, and centered (`neuroauth.verification`). The evaluation
protocol has three parts:

- The 89 enrollable subjects are split into five subject-disjoint folds.
- The 20 committed holdout subjects serve only as impostor probes.
- Operating thresholds and the decision layer are fitted only on cohort-impostor scores.

**Why a shared embedding and not per-subject classifiers.** A score has to be a comparison
inside the keyed space of the cancelable transform (D-023). A one-vs-rest model's fitted
parameters would amount to an unprotected template, and every new user would need a
retrain. The LDA-plus-cosine design is the classic from speaker verification. Being
linear, it reopens D-004's known property: relative bands sum to one per channel. The log
transform plus shrinkage handles that. `LinearDiscriminantAnalysis.means_` holds every
training subject's mean feature vector, so the estimator is never kept. `EmbeddingModel`
copies out the projection and pooled statistics only, and a test pins its fields.

**Who is unseen by what.**

| Cohort | Fitted on | Enrolled | Role |
|---|---|---|---|
| Impostor holdout (20) | never | never | probes only; headline FAR, FRR at FAR, time-to-detect |
| Fold k of the enrollable cohort | not by fold k's model | yes | genuine users, and cohort impostors for each other |
| The other folds | yes | not in fold k | embedding training |

Genuine users are unseen by the model as well as impostors, which is what deployment looks
like. `score_fold` asserts disjointness of training, fold, and holdout on every fold.

**Protocol choices.**

- **Headline split: cross-condition.** Enrollment is on eyes-open, probes on eyes-closed,
  and impostor probes come from eyes-closed too. A genuine and an impostor comparison
  therefore differ only in identity.
- **Temporal split: secondary.** Enrollment on the leading 70% of eyes-open, probes after a
  one-window guard.
- **Keys.** Every protected comparison uses the *claimed* identity's key. Scoring an
  impostor under their own key is the "unstolen token" error that drives BioHash EERs
  toward zero in parts of the literature. A test checks that scores match the claimed key
  and differ under the impostor's key.
- **Threshold selection.** Thresholds come from cohort scores and are reported on the
  holdout as the deployable number, next to the oracle EER. Choosing a threshold on the
  holdout and then reporting on it would be selection on the test set, so it is asserted
  against.
- **Served model versus evaluation.** The served demo model is fitted on all 89 subjects.
  Evaluation numbers come only from the fold models, and the README says so.

**Metric definitions.**

- **EER** is the smallest achievable max(FAR, FRR), with its threshold. A real threshold
  achieves it, unlike an interpolated crossing, and it differs from that crossing by at most
  one step.
- **FRR at FAR** uses the smallest threshold meeting the target. Tied scores are accepted or
  rejected together, which matters because Hamming similarity moves in 1/64 steps.
- **Resolution** is counted in (claimed, impostor-subject) pairs, not windows. The headline
  is FRR at FAR = 0.01, resolved at 17.8 expected errors. FAR = 0.001 (1.8 expected errors)
  is reported and flagged under-resolved. If failures to enroll ever push the headline
  itself below resolution, the report flags it rather than switching operating points.
- **Per-subject EER** uses a per-subject threshold, so it is optimistic, and it is always
  shown beside pooled EER.
- **Confidence intervals** come from a two-way bootstrap over claimed and impostor
  *subjects*. Windows are never resampled individually.
- **Time-to-detect** is measured at decision time and always reported beside the false
  challenge and revoke rate on genuine-only replays.
- **Failures to enroll** are counted, never dropped.

**Impostor count stays at 20.** Resolving FAR = 0.001 needs at least 3000 pairs. Over 109
subjects, (109 − n)·n peaks at 2970, so no holdout size resolves it. Raising the count would
cost enrollable subjects without buying the resolution. D-008 stands.

**Controls.**

- **Random pairing.** Each claimed subject's genuine scores are replaced by its scores
  against another enrolled subject (a seeded derangement). EER must be at least 0.40, or
  nothing is written. This catches pairing and metric bugs. It cannot catch leakage into
  the embedding fit, which the disjointness assertions rule out structurally, the same
  split D-007 draws.
- **Self-splice** (sessions). A subject is spliced to a later part of their own probe
  recording. Its revocation rate may exceed the genuine-only rate by at most 0.10;
  otherwise the splice artifact is doing the detecting.

**A priori thresholds.** Fixed in `0a4abb7`, before any Phase 2 result, and pinned by
`test_a_priori_thresholds_are_unchanged` and `test_replay_constants_are_pinned`.

| Constant | Value | Judges |
|---|---|---|
| `HEADLINE_FAR`, `FAR_TARGETS` | 0.01; (0.01, 0.001) | Headline and reported operating points |
| `MIN_EXPECTED_ERRORS` | 3 pairs | Under-resolved flag |
| `THRESHOLD_SELECTION_FAR` | 0.01, on cohort scores | Deployed threshold |
| `RANDOM_PAIRING_CONTROL_MIN_EER` | 0.40 | Run writes nothing below it |
| `PROTECTION_MATERIAL_EER_COST` | 0.02 | "Material cost of protection" |
| `TAIL_HEAVY_MIN_RATIO`, `_GAP` | 2× and +0.10 over the median | P1a |
| `CONCORDANCE_*` | ρ ≤ −0.30 with p < 0.05 confirms; ρ > −0.10 refutes | P1b |
| `REVOCATION_AGREEMENT_BAND`, `_MAX_ACCEPTED_FRACTION` | [0.45, 0.55]; 3% | Revocation works |
| `SPLICE_CONTROL_MAX_EXCESS` | 0.10 | Whether time-to-detect is reportable |
| Protocol | 5 folds (seed 20260915); swap 30 s; crossfade 0.5 s; horizon 25 s | — |
| Model | 64 components and bits; Ledoit-Wolf shrinkage; floor 1e-6; min 20 enrollment windows | Not tuned |

**Predictions carried in from Phase 1.**

- **P1a, heavy per-subject tail.** Judged by the rule above.
- **P1b, the same subjects in both tails.** Spearman ρ against the committed Phase 1
  per-subject F1. Both are judged once, on the headline table. The recorded caveat: both
  phases score the same recordings, so a poor eyes-closed recording can put a subject in
  both tails for reasons other than identity.
- **P2, EMG and cross-session (D-018).** Not testable on single-session data, and nothing
  in Phase 2 is presented as bearing on it.

**Alternatives.**

- **Fitting the embedding on every enrollable subject and evaluating on the same ones.**
  Genuine scores would be optimistic, because the model learned exactly those subjects'
  discriminative directions.
- **Choosing thresholds on the holdout.** That is selection on the test set.
- **A window-level bootstrap.** It treats 56 windows from one person as 56 people.
- **Interpolated or ROC-convex-hull EER.** The first reports a rate no threshold attains;
  the second adds nothing at these sample sizes.

---

### D-023 — Cancelable templates: keyed BioHash, derived keys, no stored seed, no rekey

**Decision.** A template is `sign(R_k · e)`: the mean enrollment embedding, projected by a
per-subject orthogonal matrix R_k and reduced to its signs. It has as many bits as the
embedding has dimensions, 64 at the defaults.

- **The key** is `HMAC-SHA256(master_secret, TRANSFORM_VERSION, subject_ref, key_version)`.
- **R_k** is built from SHAKE-256 output, turned into normals with `scipy.special.ndtri`,
  then orthonormalized by QR with its diagonal signs fixed.
- **Stored:** the bits, the versions (`key_version`, `transform_version`,
  `representation_version`, `embedding_version`, streaming fingerprint), the window counts,
  and the two enrollment statistics D-021 needs.
- **Never stored:** a key, a seed, an embedding, or a feature vector.

**What each property rests on.**

- **Non-invertible with respect to features.** The embedding maps 320 dimensions to 64, and
  `sign` discards magnitude. Even with the key, the feature vector cannot be recovered.
- **Revocable and unlinkable.** A new `key_version` gives an independent R, so bit
  agreement between the revoked and reissued templates of one subject sits near chance.
  This is tested on synthetic data and judged on real data by the D-022 revocation
  criteria.
- **Not spoof-resistant when the key and the template both leak.** With R and the bits, an
  attacker can build an embedding that reproduces them. That is a pre-image, not an
  inversion, and it survives re-keying because it approximates the person rather than the
  template. Re-keying defends against a template leak alone. This is a stated limit for
  THREAT_MODEL.md.

**Why no stored seed.** A seed in the templates table would make a database leak a key leak.
Deriving keys from a master secret held in `.env` or Secrets Manager means the database
alone yields bits without R.

**Why not `np.random.Generator`.** NEP 19 does not promise stable distribution streams
across numpy versions. A silent change to R after an upgrade would make every stored
template fail to verify, a mass false rejection with no error. This is the same reasoning as
D-008. `test_projection_is_stable_across_numpy_versions` pins R to 1e-12, and its docstring
says the fix for a failure is a new transform version plus re-enrollment, never editing the
golden values.

**Why no `rekey(template)`.** Re-keying stored bits would require inverting the transform;
if that function could be written, the template would not be cancelable. Reissue
re-enrolls from fresh features. eegmmidb has one session per subject, so the demo reissues
from the same recording, and the unlinkability shown comes entirely from the key. That is
exactly the property under test.

**Evaluation secret.** Evaluation uses a fixed, public value
(`scripts/evaluate_verification.py`). The evaluation measures the protocol, not key
secrecy. Open question 5 of the contracts review is resolved this way.

**Alternatives.**

- **Continuous random projection with no quantization.** It preserves distances, but with
  the key it is exactly invertible back to the embedding.
- **Fewer bits than dimensions.** It discards more of the embedding for protection that the
  320 → 64 reduction and `sign` already provide.
- **More bits than dimensions.** The rows cannot be orthonormal, and it leaks more about the
  embedding's direction.
- **A majority vote over per-window bits.** The sign of the mean is steadier than a vote
  over noisy per-window bits.
- **Fuzzy commitment, fuzzy vault, or homomorphic matching.** Each needs error-correcting
  code design for noisy EEG, or cryptographic machinery the scale does not justify (hard
  rule 7).

---

### D-024 — Phase 2 reporting: EER headline chosen after the results; the tail as a primary result

**Decision.**

1. The reported headline is EER, stated with the FAR and FRR at its threshold. The
   registered headline, FRR at FAR 1%, stays in the same table.
2. The per-subject EER tail gets its own results section rather than a caveat.
3. Every analysis requested after the results is labelled post hoc and descriptive.

**This change came after the results, and is recorded as such.** D-022 registered FRR at
FAR 1% as the headline, and `HEADLINE_FAR` is still pinned by its test. The run (`72a1b1e`,
clean tree) produced:

- EER 16.5% [12.3–18.4] at threshold 0.59375, where FAR is 11.8% and FRR is 16.5%;
- FRR 55.2% at threshold 0.6875 for the FAR 1% target.

The author then moved the headline to EER. The stated reason is that the FAR 1% point sits
far past the EER crossover, where FRR climbs quickly, for a window-level security target the
system was not asked to meet. That reason post-dates the number. D-016's rule for a changed
value applies: explain the change, and report results under both the old and the new choice.
Accordingly:

- nothing that judges a result has changed;
- both numbers appear side by side in the headline table;
- the README states the timing next to the headline;
- `run_summary.json` still records the headline as registered at run time, which is the
  correct provenance.

**What a reader should weigh.**

- **At the EER point**, about one impostor window in 8.5 is accepted. That is the security
  cost the EER headline leaves unspoken.
- **At FAR 1%**, more than half of genuine windows are rejected.
- **Both are window-level.** Sessions integrate windows and have not been evaluated, so
  neither number describes what a user experiences.

**Alternatives.**

- **Keep FRR at FAR 1% as the headline.** It was registered, and it is the security-relevant
  window-level number. Not chosen, by the author's call.
- **Report EER alone.** Rejected. Hard rule 2 requires FRR at a fixed FAR in every
  evaluation output, and EER alone would hide the security side.

**Post-hoc analyses, reported as descriptive.** All come from
`scripts/report_verification_tail.py`, which reads committed artifacts only.

- **Overlap with Phase 1.** 6 of the 9 worst-decile subjects scored F1 = 0 in Phase 1 (1.4
  expected; hypergeometric p = 3×10⁻⁴), and 8 of 9 scored below 0.1. This sits alongside the
  registered P1b result (ρ = −0.63, confirmed; 10 of 14 zero-F1 subjects in the worst
  quartile against 3.5 expected).
- **Cross-split check.** 7 of the 9 worst-decile subjects verify within eyes-open at a
  per-subject EER below 10%. Only S043 and S068 fall in the worst decile of both splits (0.9
  expected; p = 0.22). The per-subject rank correlation across splits is ρ = 0.48, and the
  within-state estimates rest on 13–14 genuine windows each.
- **Oracle bound.** The mean per-subject EER is 13.2%. With equal counts per subject, the
  pooled FAR and FRR under each subject's own best threshold are both at most that.

**How the tail may be interpreted.** The phases agree because two models and two task
framings fail on the same people, which rules out a classifier artifact or a
closed-set-framing artifact. It does not make the failure a fixed property of the person.
Both phases share the recordings, the features, and the eyes-open → eyes-closed structure, and
the within-state split shows most of these subjects verify well without the state change. The
supported reading is a property of the subject together with the state change. The unsupported
reading is "these people cannot be verified." Neither separates a person from their single
recording session.

**The 2.8× degradation** (EER 5.9% within eyes-open against 16.5% across) goes the same way
as Phase 1 (0.846 against 0.378). It is the same effect seen under a second model and task
framing on shared data, not an independent replication. The ratios are on different metrics
and are not compared numerically.

---

### D-025 — Session parameters, pre-registered from cohort data

**Decision.** Fixed on 2026-09-16, from cohort scores and genuine-only cohort replays only.
The holdout was never read. Under the D-016 rule these are not adjustable after holdout
results, and `tests/test_session_parameters.py` pins them.

| Parameter | Value |
|---|---|
| Unit | protected Hamming similarity, not the LLR |
| EMA half-life | 4 s |
| Revoke below | 0.56, after a dwell of 3 consecutive sub-threshold decisions |
| Challenge below | 0.58, on the first decision (no dwell) |
| Recover above (challenged to active) | 0.62 |

**Why the score and not the LLR.** Impostor sessions against worst-decile templates peak at a
median LLR of −0.53, against −1.24 for other accounts (p90 1.25 against 0.56). In score units
the same sessions peak at 0.547 against 0.537. A global LLR threshold therefore gives the
weakest accounts a higher FAR than everyone else: security traded for usability on exactly
the accounts least able to afford it, which hard rule 4 forbids. The decision layer's
per-template z-score input is what tilts it. The LLR still travels beside the score for
logging and for Phase 3, and the session logic does not read it.

**Why a 4 s half-life.** Genuine within-session SD falls 2.6× from raw windows to 4 s, and
only 1.8× more from 4 s to 12 s. On swaps, 2 s catches sooner but lets 1.6% of impostor
sessions through at 0.56; 8 s has a median detection of 16 s and 4.7% escaping; 12 s catches
only 59% of armed swaps within 25 s.

**Why revoke at 0.56,** raised from 0.54. At 0.54, 7.1% of impostor sessions never cross and
the 90th percentile of detection is censored beyond 25 s. At 0.56: genuine false-cross 6.2%,
median detection 10 s, p90 23 s, never-caught 2.9%. The extra false revocations buy a 2.4×
reduction in impostors who escape entirely.

**Why challenge at 0.58.** Genuine false-cross 16.2%, 96.1% of armed swaps caught within
25 s, median detection 8 s. A challenge is a step-up prompt, not a lockout, so a higher false
rate is affordable there.

**Why recover at 0.62.** Impostor session maxima at a 4 s half-life have a p90 of 0.601, so
0.62 sits above nearly all of them: an impostor session should almost never climb back into
an active state.

**Known limitation: challenge and revoke are 0.02 apart.** That is about one within-session
SD at this half-life (0.021 for other accounts), so the two will often fire nearly together
and a challenge will frequently be a formality before revocation. This is not a tuning
failure. The genuine and impostor distributions are close (settled medians 0.679 and 0.501,
both with a within-session SD near 0.02), so any usable gap between the levels is small. The
gap is a property of the representation, and D-021 fixed that representation.

**Self-splice control: passes.** Registered in D-022 at a maximum excess of 0.10 over
genuine-only replays at the revoke level. A self-splice is the subject's own recording
spliced onto itself, which keeps the discontinuity and removes the identity change.
Cohort-only, at the parameters above, crossing within (30, 55] s:

| Level | Group | Genuine | Self-splice | Excess | Real swaps |
|---|---|---|---|---|---|
| revoke 0.56 | all 89 | 14.6% | 13.5% | −1.1 | 91.9% |
| revoke 0.56 | others (80) | 6.2% | 3.8% | −2.5 | 91.2% |
| revoke 0.56 | worst decile (9) | 88.9% | 100.0% | +11.1 | 98.0% |
| challenge 0.58 | all 89 | 19.1% | 16.9% | −2.2 | 96.7% |

The splice artifact is not what detection is picking up, so swap timings may be reported as
time-to-detect. The worst-decile row exceeds the limit, but the criterion was registered over
genuine-only replays as a whole, that row is one session out of nine, and its genuine rate of
88.9% leaves almost no headroom to measure an excess in. It is reported, not treated as
evidence about the splice.

**Longer sessions are projected, not measured.** Every rate above comes from a 55 s replay:
52 settled decisions, 51 s. Two Poisson projections bracket a longer session, and for revoke
at 0.56 on the 80 non-tail subjects they are far apart:

| Horizon | Per-subject projection | Pooled projection |
|---|---|---|
| 5 min | 6.2% | 55.5% |
| 10 min | 6.2% | 80.2% |
| 30 min | 6.2% | 99.2% |

The per-subject projection is flat because every crossing comes from the same 6.2% of
subjects (median rate 0, p90 0), and for those subjects a crossing is near-certain within a
minute. The pooled projection spreads the group's average rate over every user, which the
data contradict. The honest reading is the lower one: a small identifiable minority is
revoked almost immediately, and the rest were never observed to cross. What the data cannot
do is bound the rest: 51 s per subject leaves a 95% upper bound near 3.5 crossings per minute
for a subject with none observed, and stationarity over half an hour (drowsiness, movement,
electrode drift) is untested on single-session data.

**Dwell: revoke needs three consecutive sub-threshold decisions.** Fixed 2026-09-16, before
the holdout session run, because it changes both reported numbers. Challenge keeps no dwell:
a spurious step-up prompt is cheap, and a dwell would only delay it.

The reason is the unmeasured risk rather than the measured one. Crossings cluster; 51 s per
subject cannot bound the 94% of subjects with no crossing observed; and stationarity is
untested, with drowsiness, movement and electrode drift all pushing the rate up over a longer
session. A dwell is cheap insurance against a long-session false-revoke rate that can only be
worse than what was measured.

Measured on cohort data at the pre-registered parameters, k = 1 against k = 3:

| Group | Genuine revoked | Self-splice | Swaps caught | Median | p90 | Impostors escaping |
|---|---|---|---|---|---|---|
| others (80), k = 1 | 6.2% | 3.8% | 91.2% | 10 s | 23 s | 2.9% |
| others (80), k = 3 | 5.0% | 0.0% | 87.8% | 12 s | > 25 s | 3.9% |
| all 89, k = 1 | 14.6% | 13.5% | 91.4% | 9 s | 22 s | 2.8% |
| all 89, k = 3 | 13.5% | 7.9% | 87.8% | 11 s | > 25 s | 3.9% |
| worst decile (9), k = 3 | 88.9% | 77.8% | 88.1% | 7 s | > 25 s | 3.3% |

So the dwell costs about 2 s of median detection, as expected from the swap trajectory being
continuously below 0.56 from roughly +10 s. The tail cost is larger than the median cost: the
90th percentile crosses the 25 s horizon, and the share of swaps caught within it falls by
3.4 points while impostors escaping entirely rises by 1.0. On the measured 51 s window the
dwell buys only 1.2 points of genuine false-revoke, and it removes self-splice revocations
for the non-tail subjects outright (3.8% to 0.0%). Its purpose is the long-session rate that
this dataset cannot measure.

**The stationarity check does not reassure, and the direction is the wrong one.** Between the
first and second halves of the settled 51 s, revoke crossings per minute rise in both groups
(worst decile 2.353 to 3.922; others 0.147 to 0.176), while challenge crossings fall in both
(4.183 to 3.660; 0.588 to 0.529). None of the four differences is distinguishable from
Poisson noise: the counts behind them are 9 against 15 crossings for the worst-decile revoke
rows and 5 against 6 for the others. The check is therefore underpowered, not evidence of
stationarity, and the flat per-subject projections above should be read as a known optimism:
the revoke rate trends upward inside the only window we can see, and everything untested
about longer sessions pushes the same way.

---

### D-026 — The worst decile is unusable under these parameters, and that is reported, not tuned away

**Finding.** With the D-025 parameters, for the nine subjects in the worst decile of
cohort per-subject EER:

- 88.9% of their genuine sessions cross the 0.56 revoke level within the horizon window, and
  100% cross the 0.58 challenge level;
- their confidence barely moves when someone else takes over: 0.553 at the swap and 0.521
  twelve seconds later, because their genuine and impostor levels nearly coincide (settled
  medians 0.554 and 0.512);
- so the system both rejects them constantly and cannot tell when the person changes.

**This is a limitation to report, not a tuning target.** Softening the levels for nine
subjects would raise FAR for all 89, which hard rule 4 forbids. Per-subject thresholds do not
help either: their per-subject EER of 39.5% is already measured at each subject's own best
threshold, so the failure is separability, not calibration
(`docs/PHASE2_PER_SUBJECT_THRESHOLDS.md`).

**What would help** is enrolling across brain states, since the within-state split shows most
of these subjects verify well without the state change (D-024). That is a Phase 3 protocol
question. Until then the product answer is a fallback factor for these users, and the rate is
reported rather than hidden: about one user in ten on this dataset, under a change of brain
state that enrollment did not cover.
