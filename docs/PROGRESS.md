# PROGRESS — NeuroAuth

**Current phase:** 2 — Verification + continuous session + template protection
**Status:** Not started. Phase 1 complete and committed (`a1e6596`, pushed).
**Last updated:** 2026-09-14

---

## Where things stand

**Phase 1 is complete.** Exit criteria met: a window goes in, a subject prediction
comes out, and macro-F1 with a confusion matrix is committed. Docker Compose and the
Postgres setup were moved to Phase 2 (D-019).

### Phase 1 results (89 enrollable subjects, chance 0.011)

| Normalization | Split | Macro-F1 | Shuffled control (ceiling 0.034) |
|---|---|---|---|
| **relative** | **cross-condition (headline)** | **0.378** | 0.009 |
| relative | temporal | 0.846 | 0.010 |
| absolute_log | cross-condition | 0.597 | 0.005 |
| absolute_log | temporal | 0.956 | 0.015 |

- **Brain-state change is the dominant effect:** 0.846 within condition vs 0.378 across.
- **Absolute − relative:** +0.219 / +0.110. Upper bound on the session-artifact
  contribution; anatomy cannot be separated out (D-004).
- **Leakage demonstration (D-017):** guarded 0.846 vs deliberately leaky 0.906; the
  shuffled-label control on the leaky split reads chance.
- **Frontal EOG (D-004b):** does not concentrate; too small to act on.
- **EMG ablations (D-018):** without gamma 0.263 (drop 0.116), without temporal sites
  0.300 (drop 0.078). Both material; bounds, not EMG estimates; importance re-routes.
- **Reproducibility:** baseline artifacts byte-identical across runs from `b3450bf` and
  `fa98250`.

### What Phase 2 inherits

- **Signal pipeline:** `dsp/` (io, preprocess, windowing, features, pipeline), MNE
  confined to `io.py` (D-001). `window_stream_chunk` exists for the streaming path.
- **Impostor holdout:** 20 subjects in `config/impostor_holdout.json`, committed alone
  before any result (`1f25cd2`). The file is the source of truth (D-008).
- **Evaluation discipline:** leakage-safe splits with an overlap assertion, the
  shuffled-label gate, thresholds fixed before the runs they judge (D-016), and a driver
  that refuses to write artifacts from uncommitted code.
- **Written but unexercised:** `migrations/001_phase1_core.sql`, `docker-compose.yml`,
  `Dockerfile`, stubs in `db/` and `scripts/ingest_subjects.py`.

---

## Next up (Phase 2)

Read `docs/ROADMAP.md` Phase 2 first. The carried-over items are listed there.

1. Install Docker; run Compose; implement `db/connection.py` and `db/migrate.py`;
   apply `001_phase1_core.sql` (D-019).
2. `ingest_subjects.py` — assert DB cohorts match the committed holdout file; never
   re-derive (D-008).
3. Confirm the impostor count against a FAR-stability argument (nested selection means
   raising it keeps the 20 already committed).
4. Reframe to open-set verification; EER, FAR, FRR, FRR@FAR=0.001, DET curve.
5. Fix Phase 2 evaluation thresholds before the runs they judge (D-016 rule).

### Phase 2 expectation: the per-subject tail

**14 of 89 subjects score F1 = 0 on the Phase 1 headline split** (21 below 0.1), while
the median is 0.36. Identity features survive the brain-state change for most subjects
and not at all for some. Expect the per-subject EER distribution to carry a heavy tail.
Report per-subject EER (distribution and worst decile), not only the pooled number, and
check whether the same subjects sit in both tails.

---

## Open questions

- **Filter edge effects on short buffers.** Same preprocessing function on both paths,
  but `sosfiltfilt` edge transients differ between a 61 s recording and a short live
  buffer. D-001 removes code skew, not this. Needs a buffering strategy for the
  WebSocket path.
- **Combined EMG ablation (D-018).** Each single ablation leaves the other route open.
  Not run; if it is, fix its materiality threshold first.
- **Window size.** 2 s; also the time-to-detect floor for an impostor swap (D-005).
- **How many impostor subjects.** 20; nested, so the count can rise (D-008).
- **Quality-mask recalibration** on the enrollable cohort, once the mask gates sessions
  (D-015).
- **Channel subset.** CAR must be part of that experiment's design (D-006).
- **Lint:** `scripts/verify_dataset.py` has an en dash in a print string (RUF001).

---

## Decisions deferred

- EEGNet vs Random Forest — Phase 6, only if Phases 1–5 are done
- Container service choice (ECS Fargate vs App Runner) — Phase 4, gated on WebSocket
  support
- Threshold values for challenge/revoke — Phase 2, tuned against the DET curve
- Alembic — Phase 2, when SQLAlchemy models exist (D-009)

---

## Blockers

Docker is not installed; needed at the start of Phase 2.

---

## Session log

<!-- Append one entry per session. Newest at top. Keep entries short. -->

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
