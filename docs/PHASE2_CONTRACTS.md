# Phase 2 contracts: proposal for review

**Status:** proposal, 2026-09-15. Contracts only: every body is `raise NotImplementedError`.
After review, the decisions here move into `docs/DECISIONS.md` as D-020 onward and this
file is deleted.

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

## 1. Filter edge effects: bounded context, not a warm-up buffer

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

**Cost, stated for the README.** The first decision comes 6 s after stream start. With
`SWAP_AT_S = 30`, the first decision based entirely on impostor signal is at 35 s, so the
floor on time-to-detect from fully impostor evidence is **5 s** (window 2 s + margin 2 s +
crossfade and grid alignment 1 s). Mixed windows can trigger earlier.

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

*Disclosure:* 5e-3 was chosen after the scratch measurement above. It judges a unit test,
not a result. If the synthetic signal cannot separate a correct configuration from the
naive one at 5e-3, I will raise it with you rather than retune.

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
  everything instantly would score zero.
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
- **Resolvability** is counted in (claimed, impostor-subject) pairs, not windows.
  89 × 20 = 1780 pairs, so FAR = 0.001 expects 1.8 errors and is **reported but flagged
  unresolvable**.
- **Per-subject EER** uses per-subject thresholds, which is optimistic, and is always shown
  beside pooled EER.
- **Bootstrap CIs** use a two-way resample of claimed and impostor *subjects*, never
  individual windows.

## 6. A priori thresholds (fixed on commit, before any Phase 2 result)

| Constant | Value | Judges |
|---|---|---|
| `FAR_TARGETS` | 0.01, 0.001 | Operating points reported |
| `MIN_EXPECTED_ERRORS` | 3 pairs | Resolvability flag |
| `THRESHOLD_SELECTION_FAR` | 0.01 on cohort scores | Deployed threshold |
| `RANDOM_PAIRING_CONTROL_MIN_EER` | ≥ 0.40, else nothing written | Pairing/metric bugs |
| `PROTECTION_MATERIAL_EER_COST` | 0.02 | "Material cost of protection" wording |
| `TAIL_HEAVY_MIN_RATIO` / `_GAP` | 2× and +0.10 over median | P1a verdict |
| `CONCORDANCE_*` | ρ ≤ −0.30 and p < 0.05 confirms; ρ > −0.10 refutes | P1b verdict |
| `REVOCATION_AGREEMENT_BAND` | mean in [0.45, 0.55] | Unlinkability |
| `REVOCATION_MAX_ACCEPTED_FRACTION` | ≤ 3% at the FAR = 0.01 threshold | Revocation works |
| `SPLICE_CONTROL_MAX_EXCESS` | 0.10 | Whether time-to-detect is reportable |
| `SETTLING_TOLERANCE` | 1e-3, giving 2 s margins | Context length |
| Edge test tolerance | 5e-3 (see disclosure) | Unit test only |
| Protocol | 5 folds, seed 20260915; enroll fraction 0.7 (temporal) | — |
| Model | 64 components, 64 bits, shrinkage auto, floor 1e-6, min 20 enrollment windows | Not tuned |
| Replay | swap 30 s, crossfade 0.5 s, horizon 25 s, 0.1 s frames | — |
| Seeds | bootstrap 20260916 (2000), permutation 20260917 (10000), pairing 20260918 | — |

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
resolvable at 20. D-008 stands.

## 9. Open questions for you

1. **2 s of added latency** for bit-exact train/serve identity. Is that acceptable, or do
   you want the causal filter and a re-baselined Phase 1?
2. **Phase 3 consequence:** retraining the embedding invalidates every stored template, and
   raw features are never kept, so promotion forces re-enrollment. Not solved here, but it
   shapes Phase 3.
3. **Model hash across machines:** the served model is refitted at startup, and LDA float
   results can differ across BLAS builds, so a template enrolled on the laptop may not
   verify in the container. Options: enroll in the same environment, or hash quantized
   arrays.
4. **Quality mask:** nothing in items 1–2 gates on it. If `update_session` uses
   `quality_ok`, the D-015 recalibration on E must come first.
5. **Evaluation master secret:** a fixed, labelled evaluation-only value derived in code,
   refused by the server settings. Or read from `.env`?
6. **Edge-effect evidence:** commit the scratch measurement as
   `scripts/measure_edge_effects.py` so the table above is reproducible, or rely on the
   slow tests?

## 10. Scope

This is more than one weekend. If a cut is needed, in order: (1) bootstrap CIs, (2) the
temporal split, (3) embedding-domain reporting beyond the single protection-cost number,
(4) React view polish. Not cuttable: fold/holdout assertions, the pairing control, the
self-splice control, the edge-effect tests.
