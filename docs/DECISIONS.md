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

**Why.** It costs one refit and a few minutes, and it is the only cheap check that
catches a leaking split before the number is published. A macro-F1 with no chance
level next to it is also uninterpretable to a reader — for 89 enrollable subjects
chance is about 0.011, and that context belongs in the same table as the result.

---

### D-008 — Impostor holdout fixed at ingest, seeded and nested

**Decision.** `neuroauth.cohorts.select_impostor_holdout` picks the impostor subjects
from a seeded permutation of the sorted subject ids, truncated to `n_impostors`. It
runs during `scripts/ingest_subjects.py`, before any Phase 1 result exists. Phase 1
trains and evaluates on the enrollable cohort only.

Starting values: `seed=20260821`, `n_impostors=20`.

**Alternatives.** Choose the holdout in Phase 2, when the FAR-stability argument that
determines the right count can actually be made.

**Why rejected.** Choosing after seeing Phase 1 results is selection bias, and it
would compromise the Phase 2 EER before Phase 2 started. The count genuinely is a
Phase 2 question, which is why the selection is a *truncated permutation* rather than
a random sample: the chosen set is nested in `n_impostors`, so raising the count later
keeps every subject already committed to the holdout and only adds more. The Phase 2
decision about how many impostors a stable FAR estimate needs can therefore be made
on its merits without re-rolling anything.

**Enforcement.** Three layers: the `holdout_never_enrolled` CHECK constraint in
`migrations/001_phase1_core.sql`, `assert_holdout_excluded` called on every training
set construction, and `tests/test_cohorts.py`.

**Selected impostor subjects:** *(recorded here by `scripts/ingest_subjects.py` on
first run — do not edit by hand)*

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
