# PROGRESS — NeuroAuth

**Current phase:** 2 — Verification + continuous session + template protection
**Status:** Items 1–3 implemented; verification results written up (README, D-024). Session
parameters are pre-registered from cohort data (D-025) and the self-splice control passes, so
swap timings may be reported as time-to-detect. `update_session` and `initial_session_state`
are the author's next piece; the types are settled. Items 4–5 (Docker, database) remain.

### Phase 2 verification results (`72a1b1e`)

| | Eyes open → closed (headline) | Within eyes open |
|---|---|---|
| EER, protected [95% CI] | **16.5%** [12.3–18.4] (FAR 11.8% / FRR 16.5% at threshold) | 5.9% [4.6–8.2] |
| FRR at FAR 1% (registered headline) | 55.2% | 19.5% |
| EER, unprotected embedding / decision v0 | 9.0% / 14.2% | 3.0% / 5.4% |

- **Per-subject tail (P1a confirmed).** Median 10.7%, p90 32.1%, worst-decile mean 39.5%.
- **Same subjects as Phase 1 (P1b confirmed, ρ = −0.63).** 6 of the 9 worst-decile subjects
  had Phase 1 F1 = 0 (post hoc, p = 3×10⁻⁴).
- **State change drives the tail (post hoc).** 7 of those 9 verify within eyes-open below
  10% EER.
- **Protection cost is material.** 7.4 points across the state change, 2.9 within it.
- **Revocation works.** 49.2% bit agreement, 0 of 89 revoked templates accepted.
- **Controls pass.** Pairing-control EER between 45.7% and 54.9%; no failures to enroll.
**Last updated:** 2026-09-15

---

## Where things stand

**Phase 1 is complete** (results in the README; D-001 to D-019).

**Phase 2 decisions recorded:**

- **D-020:** bounded-context filtering, identical offline and live.
- **D-021:** the embedding is frozen, and retraining adjusts a decision layer.
- **D-022:** the verification protocol, metric definitions, and a priori thresholds.
- **D-023:** the cancelable transform.

`docs/PHASE2_CONTRACTS.md` stays in the repo as the review record.

### What exists

| Area | Modules | Notes |
|---|---|---|
| Streaming features | `dsp/streaming.py` | Chunk-invariant bit for bit; margins refused below the settling time |
| Verification | `verification/` (embedding, decision, scoring, protocol, metrics) | Folds, cohort assertions, cross-fitted decision layer v0, pairing control |
| Templates | `templates/` (cancelable, enrollment) | Golden-pinned projection; no stored key or seed; revoke and reissue |
| Sessions | `session/` (runtime, replay, logic stubs), `api/stream.py` | Frame protocol, gaps, warm-up, splice; the endpoint only moves bytes |
| Evaluation | `scripts/evaluate_verification.py` | Refuses dirty-tree artifacts; gates before any write |
| Evidence | `scripts/measurements/edge_effects.py` | Artifact generated from `0a4abb7` |

326 tests pass, including the slow S001R01 checks. ruff and `mypy --strict` are clean.
FastAPI, uvicorn, and httpx were added to the dependencies.

A partial smoke run of the evaluation driver (15 enrollable subjects, 4 impostors, no
bootstrap) wrote to a scratch directory and exited cleanly. Its metrics were deliberately
not looked at.

---

## Next up (Phase 2)

1. **Commit the verification artifacts from `72a1b1e`, then the write-up.** Run
   `python -m scripts.report_verification_tail` from the clean tree and commit
   `per_subject_tail.{json,png}`.
2. ~~Per-subject thresholds~~ **resolved:** one global threshold, at the pre-registered levels
   in D-025. `docs/PHASE2_PER_SUBJECT_THRESHOLDS.md` stays as the record of what they would
   have cost.
3. **Author writes `initial_session_state` and `update_session`** (`session/logic.py`), to the
   settled types and the pre-registered parameters (D-025), including the revoke dwell of
   three consecutive sub-threshold decisions; challenge has none. If `quality_ok` gates
   anything, the D-015 recalibration on the enrollable cohort comes first.
4. **Session evaluation driver**, once `update_session` exists: time-to-detect over holdout
   swaps, false challenge and revoke rates on genuine-only replays, and the self-splice
   control. Report distributions beside the signal-path floors (D-020: +1, +3, +5, +7 s).
5. **Docker, `db/`, migrations, `ingest_subjects.py`** (D-019, D-008). Migration 002: a
   templates table (bits, key_version, transform, representation and embedding versions,
   enrollment statistics; no seed column), and sessions recording
   `representation_version` and `decision_version` (D-021).
6. **Wire enrollment, the session resolver, and the transition sink to the database.**
   Verify revocation end to end: a revoked key_version is refused at session start.
7. Minimal React live-session view.

### Phase 2 expectation: the per-subject tail

**14 of 89 subjects score F1 = 0 on the Phase 1 headline split** (21 below 0.1), while
the median is 0.36. The prediction is a heavy per-subject EER tail, with the same subjects
in both tails. Both are judged by the a priori criteria in D-022 on the headline table only.

---

## Open questions

- **Model hash across machines.** The served embedding is refitted at startup, and LDA float
  results can differ across BLAS builds, so a template enrolled on the laptop may not verify
  in the container (refused as a representation mismatch). Options: enroll in the serving
  environment, or hash quantized arrays. Decide at item 5.
- **Quality-mask recalibration** on the enrollable cohort, before the mask gates session
  decisions (D-015).
- **Combined EMG ablation (D-018).** Not run; if it is, fix its materiality threshold first.
- **Window size.** 2 s. With 2 s margins, a decision's whole raw context is impostor signal
  only 7 s after a swap (D-005, D-020).
- **Channel subset.** CAR must be part of that experiment's design (D-006).
- **Test warning:** starlette's TestClient warns that httpx is deprecated in favour of
  httpx2. Harmless today; revisit when pinning versions.
- **Lint:** `scripts/verify_dataset.py` has an en dash in a print string (RUF001).

---

## Decisions deferred

- A CNN embedding against the frozen LDA embedding: Phase 6. Adoption would be a
  representation migration requiring re-enrollment (D-021).
- Container service choice (ECS Fargate vs App Runner): Phase 4, gated on WebSocket support.
- Challenge and revoke threshold values: the author, from cohort scores only.
- Alembic: when SQLAlchemy models exist (D-009).

---

## Blockers

Docker is not installed and may be blocked on this work-managed machine. It is needed at
item 4, deliberately after the NumPy work.

---

## Session log

<!-- Append one entry per session. Newest at top. Keep entries short. -->

### 2026-09-15
**Phase:** 2 (contracts, then items 1–3 implemented)

**Shipped:**
- Contracts and their review (`docs/PHASE2_CONTRACTS.md`).
- Edge-effect measurement script and its artifact from committed code.
- Items 1–3, with tests: streaming features, embedding, decision layer v0, scoring, the
  protocol, metrics, the cancelable transform, enrollment and revocation, the session
  runtime, replay and splice, the WebSocket endpoint, and the evaluation driver.
- D-020 to D-023. Roadmap Phase 3 and Phase 6 reframed around the frozen embedding.

**Review decisions:**
- Bounded context approved, 2 s latency included.
- Headline is FRR at FAR = 0.01; FAR = 0.001 is flagged under-resolved.
- Cut order: CIs are cut before the temporal split.
- Embedding frozen (Option A). Backward-compatible training rejected because it requires
  stored features, a direct conflict with hard rule 3.
- `update_session` and `initial_session_state` stay with the author.

**Next:** Commit; full evaluation from the clean tree; author writes the session logic.

**Notes / decisions:**
- The runtime reads only `state` and `confidence` from `SessionState`, and records a client
  stop as a transition to "closed".
- A session's time counts received samples; the length of a gap is unknown to the server.
- The evaluation master secret is a fixed public value (D-023).
- The v0 decision layer is reported beside the protected-domain headline, not instead of it.

### 2026-09-14
**Phase:** 1 → complete
**Shipped:** `dsp/` (io, preprocess, windowing, features, pipeline); `cohorts.py` and
the committed impostor holdout; `models/` (splits, baseline, evaluation, ablation);
`config.fingerprint()`; `scripts/select_holdout.py`, `scripts/train_baseline.py` with
holdout-committed and clean-tree preconditions. Baseline run, leakage demonstration,
EMG ablations; README results section. D-004 (corrected), D-004b, D-005, D-007
(corrected), D-008 (updated), D-013–D-019. Docker/Postgres moved to Phase 2.
**Next:** Phase 2, starting with Docker and the database.
**Notes / decisions:** Headline 0.378 (34× chance). Brain-state change dominates. The
shuffled-label control cannot detect overlap leakage — shown on synthetic and real data.
The absolute/relative gap and the EMG drops are reported as bounds, never as
decompositions. All evaluation thresholds were fixed before the runs they judge.
Artifacts reproduced byte for byte. User commits; never auto-commit.

### 2026-08-21
**Phase:** 1
**Shipped:** Repo structure; signal-pipeline contracts (io, preprocess, windowing,
features, pipeline); cohort/holdout contracts; splits, baseline, evaluation contracts;
`migrations/001_phase1_core.sql`; pyproject, Dockerfile, docker-compose; test skeleton;
`docs/DECISIONS.md` D-001 through D-012; README with ODC-By attribution.
**Next:** Install dev deps, then implement the `dsp` modules bottom-up.
**Notes / decisions:** Relative band power is the headline, absolute is a published
comparison — single-session amplitude confounds are perfectly correlated with subject
(D-004). Cross-condition R01→R02 is the headline split, justified as brain-state
robustness rather than as removing a shortcut; the shortcut reasoning was wrong and is
recorded as wrong in D-003. Shuffled-label control runs on every committed evaluation
(D-007). Impostor holdout is fixed at ingest, seeded and nested (D-008). `signal/`
renamed `dsp/`. `requires-python = ">=3.12"`. Root `test.py` deleted,
`verify_dataset.py` moved to `scripts/`.
