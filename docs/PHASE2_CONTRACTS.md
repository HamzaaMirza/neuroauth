# Phase 2 contracts: proposal for review

**Status:** proposal, 2026-09-15. Contracts only: every body is `raise NotImplementedError`.
After review, the decisions here move into `docs/DECISIONS.md` as D-020 onward and this
file is deleted.

**Review, 2026-09-15:**
- §1 approved: bounded context, 2 s margins, and the latency cost.
- FAR = 0.001 is reported but flagged under-resolved. The headline is FRR at FAR = 0.01 (§5).
- Cut order confirmed (§10).
- **§11, templates across retrains, must be decided before implementation.**

**Committing the contracts commits the thresholds.** Every constant in section 6 lives in
code (`verification/metrics.py`, `protocol.py`, `templates/*`, `session/replay.py`), so the
commit that lands these files is the timestamp showing each threshold predates its result.

Work order, as reordered: (1) open-set verification, (2) cancelable transform + enrollment +
revoke-and-reissue, (3) streaming and session runtime, (4) Docker, `db/`, migrations,
`ingest_subjects.py`, (5) wire enrollment to the database. Session evaluation
(time-to-detect) waits on `update_session`, which is the author's.

## Files

| File | Contents |
|---|---|
| `config.py` (appended) | `ContextConfig`, `StreamingConfig` with its own fingerprint. `PipelineConfig` is untouched, so Phase 1 fingerprints still reproduce |
| `dsp/streaming.py` | Bounded-context featurization, `push_samples`, settling-time check |
| `verification/embedding.py` | Shared embedding: log-relative power, standardize, shrinkage LDA, center |
| `verification/scoring.py` | `Verifier`, `score_windows` (never raises) |
| `verification/protocol.py` | Subject-disjoint folds, enrollment/probe rows, `score_fold`, threshold selection, controls |
| `verification/metrics.py` | EER, FAR/FRR, FRR@FAR, DET, per-subject tail, P1 verdicts, time-to-detect, report |
| `templates/cancelable.py` | Keyed BioHash, HMAC key derivation, SHAKE-256 projection |
| `templates/enrollment.py` | `ProtectedTemplate`, `enroll_subject`, `revoke_and_reissue` |
| `session/logic.py` | Types plus `update_session` and `initial_session_state` stubs **(author's)** |
| `session/runtime.py` | Wire protocol, transport-free `handle_frame` |
| `session/replay.py` | Replay framing, crossfaded splice for impostor injection |

---

## 1. Filter edge effects: bounded context, not a warm-up buffer *(approved)*

**A warm-up buffer does not fix this.** `sosfiltfilt` runs forward, then backward. On a live
stream, the backward pass starts at the newest sample, which is where the window being
scored sits. Past context removes the left-edge transient. The right-edge transient needs
future samples, and only waiting for them (delay) or a filter that never looks ahead
(causal) avoids it.

**Measured** on S001R01 (enrollable cohort, a signal property only, nothing identity-bearing).
Values are the |difference| in relative band power against whole-recording `preprocess`
(the Phase 1 path), over interior windows. For scale, delta relative power is about 0.42,
with a window-to-window std of 0.145 within the recording.

| Approach | Offline vs live | Latency | Delta vs Phase 1 features, median / p95 |
|---|---|---|---|
| `preprocess` on each 2 s buffer (the trap) | **skewed** | 0 | 0.013 / 0.058 |
| **Zero-phase over a bounded context, 2 s + 2 s (proposed)** | **bit-identical by construction** | **+2 s** | all bands pooled: p95 2.9e-4, max 1.4e-3 |
| Causal `sosfilt` with carried state | bit-identical (checked: chunked == one-shot) | 0 | 0.033 / 0.124 |
| Causal, cascade applied twice (magnitude matches filtfilt) | bit-identical | 0 | 0.041 / 0.156 |

**Proposal:** a window's features are a pure function of the raw slice `[start − 2 s, end +
2 s)`, filtered with the unchanged Phase 1 `preprocess` and cropped back to the window.
Offline, the slice is cut from the recording. Live, the window is emitted once its right
margin arrives. Embedding training, enrollment, evaluation, and live sessions all go through
`dsp/streaming.py`, so offline and live agree bit for bit, not within a tolerance.

**Why causal lost.** Its features move by about a quarter of delta's natural spread at the
median. Matching filtfilt's magnitude response made that worse, not better, so the cause is
group delay, not attenuation, and there is no cheap equalizer. Switching would also change
the pipeline under Phase 1's findings, which would confound the P1b comparison (section 7).
Its only advantage is the 2 s of latency the proposal pays.

**Margins come from a data-free criterion,** not from the table. The notch-plus-bandpass
impulse response stays below 1e-3 of its peak after 1.58 s, which rounds up to the 1 s hop
as 2 s. `check_margins` refuses anything shorter. The table agrees: 1 s on either side
allows a max of 1.3e-2, 2 s + 2 s allows 1.4e-3, and 3 s + 3 s allows 2.8e-4.

**Cost, stated for the README: floors, not expected times.** The first decision comes 6 s
after stream start. After a swap that starts at `SWAP_AT_S = 30`:

- No decision reflects *any* post-swap signal before **3 s**. The first window that
  contains the swap (29–31 s) is decided at 33 s.
- No decision rests *entirely* on impostor signal before **5 s**. The first window clear of
  the 0.5 s crossfade starts at 31 s, ends at 33 s, and is decided at 35 s.

Of those 5 s, 2 s is the filter's right margin, 2 s is the window length (D-005), and 1 s
is the crossfade plus alignment to the 1 s hop grid. **These are floors set by the signal
path, not expected detection times.** `update_session` accumulates confidence over several
windows before it challenges or revokes, so measured time-to-detect will be longer. The
README reports the measured distribution next to the floor, never the floor alone.

**Reproduce:** `python -m scripts.measurements.edge_effects` writes
`artifacts/measurements/edge_effects.json`.

**Tests that catch it** (`tests/test_streaming.py`):

1. `test_stream_matches_offline_bit_for_bit`: a synthetic 60 s, 64-channel 1/f signal with
   alpha, fed to `push_samples` in seeded random chunk sizes (including 1-sample chunks and
   chunks longer than a window). The concatenated output must `assert_array_equal`
   `process_recording_bounded` on values, onsets, and the quality mask. This fails the
   moment the live path filters differently from training. Once `update_session` exists,
   an end-to-end variant runs through `encode_frame` → `handle_frame`.
2. `test_bounded_context_matches_whole_recording_filtering`: max |Δ relative band power| ≤
   **5e-3** on interior windows. This catches margins that are too short, the case where
   both paths agree with each other but both carry edge transients.
3. `test_naive_per_buffer_filtering_fails_the_same_check`: the negative control. Per-buffer
   `preprocess` must exceed the same 5e-3, which proves test 2 has teeth (same pattern as
   `test_shuffled_control_cannot_detect_window_overlap_leakage`).
4. `test_margins_below_settling_time_are_refused`.
5. `slow`: tests 2 and 3 on S001R01.

*This threshold judges a unit test, not a result.* The 5e-3 tolerance decides whether a test
passes. It judges no reported number, so D-016's a priori rule, which governs thresholds
that judge results, does not apply to it. It was chosen *after* the measurement above and is
disclosed as such, not presented as a priori. If the synthetic signal cannot separate a
correct configuration from the naive one at 5e-3, I will raise it with you rather than
retune.

---

## 2. Open-set verification

**Shared embedding plus template matching, not per-subject classifiers.** A score has to be
a comparison inside the keyed, protected space. A one-vs-rest model's parameters are an
unprotected template, and every new user would need a retrain. The embedding is the
speaker-verification classic: log relative power, standardize, shrinkage LDA, center.
Because the model is linear, D-004's simplex property (relative bands sum to 1 per
channel) now matters. The log transform plus shrinkage handles it.

**The sklearn estimator is never kept.** `LinearDiscriminantAnalysis.means_` holds every
training subject's mean feature vector, which would put 70-odd raw-feature templates inside
the model object. `EmbeddingModel` copies out the projection and pooled statistics only.

**Who is unseen by what:**

| Cohort | Fitted on? | Enrolled? | Role |
|---|---|---|---|
| Impostor holdout H (20) | never | never | Probes only. **Headline FAR, FRR@FAR, time-to-detect** |
| Fold k of E (about 18) | not by fold k's model | yes | Genuine users, and **cohort impostors** for each other |
| E minus fold k (about 71) | yes | not in fold k | Embedding training |

- Five subject-disjoint folds, so genuine users are unseen too, matching deployment.
- **Operating thresholds are chosen on cohort-impostor scores only.** H is used only for
  reporting. The report carries both the oracle EER on H and the FAR/FRR on H at the
  cohort-chosen threshold, which is the deployable number.
- Headline enrollment and probes: enroll on eyes-open, probe on eyes-closed. **Impostor
  probes come from eyes-closed too**, so genuine and impostor comparisons differ only in
  identity.
- Headline table, fixed now: **cross-condition × protected domain × holdout impostors.** The
  embedding domain is reported to cost out protection. The temporal split is secondary.
- The served demo model is fitted on all 89. Evaluation numbers come from the fold models
  only. The demo enrolls subjects the served model was trained on, and the README says so.

---

## 3. Cancelable transform, enrollment, revocation

- **Keyed BioHash:** `bits = sign(R_k · e)`, 64 bits, R orthogonal. Non-invertibility with
  respect to features comes from the 320→64 reduction plus `sign`. Revocability and
  unlinkability come from the key.
- **No seed or key is stored anywhere.** `key = HMAC-SHA256(master_secret, subject_ref,
  key_version)`. The templates table will hold `key_version`, not a seed, so a database leak
  alone yields bits without R.
- **R is built from SHAKE-256, not `np.random.Generator`.** NEP 19 does not promise stable
  distribution streams across numpy versions, and a changed R would silently fail every
  stored template. A golden-value test pins R. Same reasoning as D-008.
- **The key is always the claimed identity's.** Scoring impostors under their own key is the
  "unstolen token" error that inflates BioHash results.
- **There is no `rekey(template)`.** If one could be written, the template would be
  invertible. Reissue re-enrolls from fresh features. On this dataset that means the same
  recording, so the unlinkability shown comes entirely from the key, which is the property
  under test.
- **Stated limit:** if the key *and* the template leak, an attacker can construct a
  pre-image embedding, and it survives re-keying. Re-keying defends against a template leak
  alone. This goes in THREAT_MODEL.md. (Optional: a labelled demonstration, D-017 style.)
- Failure to enroll (fewer than 20 windows) is counted and reported, never silently dropped.
  Not-ok windows are included in the template: the mask gates nothing until it is
  recalibrated (D-015).

## 4. Streaming and session

- `handle_frame(runtime, bytes) -> RuntimeOutput` is pure and never raises. The FastAPI
  endpoint only moves bytes. FastAPI, uvicorn, and httpx are not installed yet; the
  endpoint module and those dependencies arrive with the implementation.
- Frames carry **float64**. float32 would break the bit-exact offline/live test, and the
  cost is 82 kB/s.
- A sequence gap resets the buffer, flags `gap_before`, and restarts warm-up. Nothing is
  filtered across a hole.
- `update_session` and `initial_session_state` are stubs. The runtime relies only on these
  guarantees: pure, never raises, returns a transition iff the state changes, terminal
  states absorb, observations arrive in time order. **Threshold values may come only from
  cohort scores and genuine-only replays of E**, never from holdout replays.
- **Time-to-detect** is measured at decision time (window end + right margin) and is always
  reported next to the false challenge/revoke rate on genuine-only sessions. Revoking
  everything instantly would score zero. The 3 s and 5 s figures in §1 are floors; the
  reported number is the measured distribution.
- **Splice honesty:** swaps are crossfaded over 0.5 s in the raw domain. A **self-splice
  control** (a subject spliced to a later part of their own probe recording) keeps the
  discontinuity and removes the identity change. If self-splices are revoked more often than
  genuine-only sessions by more than 0.10, the splice is doing the detecting, and
  time-to-detect is not reported as a result.

## 5. Metric definitions worth defending

- **EER** is the smallest achievable max(FAR, FRR), with its threshold. This is achievable
  by a real threshold, unlike an interpolated crossing, and differs from it by at most one
  step.
- **FRR@FAR** uses the smallest threshold with FAR ≤ target. Ties are atomic (Hamming
  similarity moves in steps of 1/64).
- **Resolution** is counted in (claimed, impostor-subject) pairs, not windows. With
  89 × 20 = 1780 pairs:
  - FAR = 0.01 expects 17.8 errors, so it is resolved. **FRR at FAR = 0.01 is the headline**
    (`HEADLINE_FAR`).
  - FAR = 0.001 expects 1.8 errors. It is **reported, flagged under-resolved, and never the
    headline.**
  - If enrollment failures ever pushed pairs below 300, the report would flag the headline
    too. It would not switch to another operating point.
- **Per-subject EER** uses per-subject thresholds, which is optimistic, and is always shown
  beside pooled EER.
- **Bootstrap CIs** use a two-way resample of claimed and impostor *subjects*, never
  individual windows.

## 6. A priori thresholds (fixed on commit, before any Phase 2 result)

| Constant | Value | Judges |
|---|---|---|
| `FAR_TARGETS` | 0.01, 0.001 | Operating points reported |
| `HEADLINE_FAR` | 0.01 | The headline operating point |
| `MIN_EXPECTED_ERRORS` | 3 pairs | Under-resolved flag |
| `THRESHOLD_SELECTION_FAR` | 0.01 on cohort scores | Deployed threshold |
| `RANDOM_PAIRING_CONTROL_MIN_EER` | ≥ 0.40, else nothing written | Pairing/metric bugs |
| `PROTECTION_MATERIAL_EER_COST` | 0.02 | "Material cost of protection" wording |
| `TAIL_HEAVY_MIN_RATIO` / `_GAP` | 2× and +0.10 over median | P1a verdict |
| `CONCORDANCE_*` | ρ ≤ −0.30 and p < 0.05 confirms; ρ > −0.10 refutes | P1b verdict |
| `REVOCATION_AGREEMENT_BAND` | mean in [0.45, 0.55] | Unlinkability |
| `REVOCATION_MAX_ACCEPTED_FRACTION` | ≤ 3% at the FAR = 0.01 threshold | Revocation works |
| `SPLICE_CONTROL_MAX_EXCESS` | 0.10 | Whether time-to-detect is reportable |
| `SETTLING_TOLERANCE` | 1e-3, giving 2 s margins | Context length |
| Protocol | 5 folds, seed 20260915; enroll fraction 0.7 (temporal) | — |
| Model | 64 components, 64 bits, shrinkage auto, floor 1e-6, min 20 enrollment windows | Not tuned |
| Replay | swap 30 s, crossfade 0.5 s, horizon 25 s, 0.1 s frames | — |
| Seeds | bootstrap 20260916 (2000), permutation 20260917 (10000), pairing 20260918 | — |

Deliberately left out of this table: the 5e-3 edge-effect test tolerance (§1). It judges a
unit test, not a result, and it was chosen after the measurement, so listing it among a
priori thresholds would misstate both what it judges and when it was set.

## 7. Predictions carried in

- **P1a, heavy tail:** confirmed iff the worst-decile mean per-subject EER is ≥ 2× the
  median and ≥ median + 0.10, on the headline table. Otherwise refuted.
- **P1b, same subjects in both tails:** Spearman ρ between Phase 1 per-subject F1 (committed
  `relative_cross_condition_per_class_f1.csv`) and Phase 2 per-subject EER. Confirmed iff
  ρ ≤ −0.30 with one-sided permutation p < 0.05. Refuted iff ρ > −0.10. Otherwise
  inconclusive. The overlap of the 14 zero-F1 subjects with the worst EER quartile
  (3.5 expected by chance) is descriptive only. Caveat recorded with the verdict: both
  phases score the same recordings, so a bad eyes-closed recording can put a subject in both
  tails for reasons other than identity features.
- **P2, EMG and cross-session degradation (D-018):** **not testable in Phase 2.** Every
  Phase 2 number is still single-session, and nothing will be presented as bearing on P2.
  (A verification-domain gamma ablation is possible, but it measures dependence, not
  cross-session survival, and is not proposed.)

## 8. Impostor count: keep 20

FAR = 0.001 at pair resolution needs (claimed × impostor) ≥ 3000. With 109 subjects,
(109 − n)·n peaks at 2970 when n = 54 or 55, so **no holdout size makes FAR = 0.001
resolvable on this dataset.** Raising the count would cost enrollable subjects and Phase 1
comparability without buying the resolution. FAR = 0.01 needs only 300 pairs and is
resolved at 20, which is why it is the headline. D-008 stands.

## 9. Open questions for you

1. ~~2 s of added latency~~. **Resolved:** approved.
2. **Templates across retrains:** moved to §11. It must be decided before implementation.
3. **Model hash across machines:** the served model is refitted at startup, and LDA float
   results can differ across BLAS builds, so a template enrolled on the laptop may not
   verify in the container. Options: enroll in the same environment, or hash quantized
   arrays.
4. **Quality mask:** nothing in items 1–2 gates on it. If `update_session` uses
   `quality_ok`, the D-015 recalibration on E must come first.
5. **Evaluation master secret:** a fixed, labelled evaluation-only value derived in code,
   refused by the server settings. Or read from `.env`?
6. ~~Edge-effect evidence~~. **Resolved:** `scripts/measurements/edge_effects.py`, output in
   `artifacts/measurements/edge_effects.json`.

## 10. Scope

This is more than one weekend. **Cut first to cut last:**

1. bootstrap CIs
2. the temporal split
3. embedding-domain reporting beyond the single protection-cost number
4. React view polish

The temporal split outlasts the CIs because a second split that shows whether the result
holds is worth more than an interval on one split. Build priority is the reverse order.

Not cuttable: fold and holdout assertions, the pairing control, the self-splice control,
the edge-effect tests.

---

## 11. Templates across retrains *(decide before implementation)*

**The problem.** A template is `sign(R_k · E(x))`. Retrain the embedding E, and every
stored template sits in the old E's space. Raw features are never kept (hard rule 3), so
templates cannot be recomputed, only re-enrolled from new signal. As designed, every
promoted retrain would invalidate every template.

**A second problem, which I think settles it.** Retraining E needs the *feature vectors* of
the reviewed false rejections. Hard rule 3 forbids persisting them, and review is
asynchronous, so by the time a reviewer confirms a rejection, its features are gone. What
`record_false_rejection` can keep is score-level: window scores, quality flags, versions,
thresholds, and the verdict. An embedding retrain works in the demo only because the EDF
files are on disk and features can be re-derived from (subject, run, onset). That would be a
loop that cannot exist in deployment, presented as though it could.

### Option A: freeze the representation, retrain a decision layer

- **Representation** = `StreamingConfig` + embedding + cancelable transform, versioned
  together as `representation_version`. It is frozen for the retraining loop. Templates bind
  to it and never go stale.
- **Decision layer** = a small logistic model mapping what is persistable about a window to
  a calibrated log-likelihood ratio. Its inputs:
  - Hamming similarity to the claimed template.
  - Quality: `window_ok`, bad-channel fraction, and flag categories.
  - Per-template score statistics recorded **at enrollment**: the mean and std of each
    enrollment window's similarity to a leave-one-out template (built without that window).
    That is two scalars per template, not invertible. They must be captured at enrollment,
    because the enrollment windows are gone afterwards.
- It trains on score-level rows only: genuine scores, cohort-impostor scores, and reviewed
  false rejections. The hard-rule-5 assertions carry over to score rows, so no holdout
  subject's rows can reach training.
- It is versioned as `decision_version`. Sessions and events record both versions.

Costs:

1. **The ceiling is frozen.** The loop cannot fix a false rejection caused by the embedding
   itself, such as the brain-state failures behind the P1 tail. It can fix calibration,
   quality handling, and per-user score offsets. The README must say what the loop learns
   and what it cannot.
2. **Calibration alone is invisible to the gate.** One monotone transform of one global
   score leaves the pooled DET curve, and therefore EER, unchanged. For a retrain ever to
   pass "EER improvement above threshold", the layer must use information beyond the score:
   quality and per-user normalization. That is a Phase 3 design constraint, stated now.
3. **Key invariance limits the inputs.** Each subject's R is an independent random rotation,
   so bit *i* means a different direction for every user. A shared weight per bit is
   meaningless; only permutation-invariant summaries of bit agreement can be shared, which in
   practice means the Hamming count. Nothing is learned inside the protected representation.
4. **Representation upgrades become rare, deliberate migrations.** A better embedding
   (Phase 6's CNN comparison, now against the LDA embedding rather than the RF) means
   re-enrollment, outside the regular loop. That is documented, not built. Templates already
   carry the version, so the path stays open.
5. **Phase 2 cost: small, but it has to happen now.**
   - (a) `model_version` splits into `representation_version` and `decision_version`.
   - (b) `ProtectedTemplate` gains the two enrollment score statistics. They cannot be added
     later.
   - (c) A v0 decision layer ships in Phase 2: logistic calibration fitted on cohort scores.
     Phase 3 then retrains an existing component instead of inserting a new one.
   - (d) `WindowScore` and `WindowObservation` carry the calibrated LLR next to the raw
     similarity, so `update_session` thresholds would be in LLR units. That touches your
     code, so it is your call.
   - (e) Migration 002 records both versions on sessions.

### Option B: concurrent template versions, lazy re-enrollment

Templates are keyed by (subject, `representation_version`, `key_version`). A promotion loads
the new representation next to the old one. A session verifies under the subject's existing
template. If the session clears a re-enrollment bar, its windows (still in memory) are
embedded under the new representation and a new template is written. Old versions retire at
a deadline.

Costs:

1. **The retrain itself still conflicts with hard rule 3** (see above). The new
   representation can be trained only on dataset recordings, never on reviewed production
   rejections, so "learns from false rejections" is true only in the demo.
2. **Template poisoning.** Re-enrollment trusts whoever the session accepted, so a false
   accept under the old representation becomes a permanent enrollment under the new one. The
   re-enrollment bar must be far stricter than the accept bar (a long session, never
   challenged, confidence well above threshold), which slows migration.
3. **Version fan-out.** Weekly promotions, combined with users who authenticate less than
   weekly, means many representations loaded at once. Retiring one needs a deadline and
   forced re-enrollment for stragglers: the very thing this option set out to avoid, now as
   a tail.
4. **Stateful rollback.** Rolling back a promotion must keep migrated users' old templates,
   and templates enrolled under the rolled-back version are orphaned.
5. **The gate compares populations mid-migration.** Production FAR and FRR depend on who has
   migrated, so offline evaluation has to simulate re-enrollment.
6. **Unmeasurable on eegmmidb.** With one session per subject, re-enrollment windows and later
   verification windows come from the same recording. An honest evaluation would split the
   probe recording into re-enroll and verify halves. That is thin, and it edges toward the
   cross-session claim this dataset cannot support.
7. **Phase 2 cost: large.** A multi-version verifier, dual inference in the runtime, a
   re-enrollment buffer and bar, version-keyed templates, re-enrollment provenance events,
   and a poisoning test.

What B buys: it is the only option under which the representation itself improves.

**Also considered: backward-compatible training** (Shen et al., CVPR 2020). The new embedding
is trained with a loss that keeps its outputs comparable with the old space, so old templates
stay valid and nothing needs re-enrolling. It does not avoid the hard-rule-3 problem, because
it still needs features to train, so it is dataset-only. It also constrains every future
model to the old space, and the gate would need a cross-version check (new probes against old
templates). It is the right technique if representation upgrades ever become frequent. They
should not be.

### Recommendation: A

A is the only option compatible with hard rule 3 end to end. It keeps templates stable, and
its Phase 3 loop is real rather than simulated. Its real cost, a frozen ceiling, is honest and
easy to state. Representation upgrades become a documented migration, using B's mechanism
with its poisoning control, triggered deliberately and not built in Phases 2–5.

If you choose A, before implementing I will:

- apply 5(a)–(e) to the contracts;
- update ROADMAP Phase 3: the retrain script retrains the decision layer, `build_training_set`
  builds score-level rows, and `evaluate_candidate` compares decision layers on score tables
  from the frozen representation;
- update ROADMAP Phase 6: CNN versus the LDA embedding, framed as a migration decision.
