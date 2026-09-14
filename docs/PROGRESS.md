# PROGRESS — NeuroAuth

**Current phase:** 1 — Signal pipeline + identification baseline
**Status:** Final Phase 1 run complete from clean, pushed commit `fa98250`, with the EMG
ablations. README results section written. **Exit criteria are met once the regenerated
artifacts, README, and docs are committed.** Two checklist items remain off the exit
path: Docker Compose and applying the Postgres schema.
**Last updated:** 2026-09-14

---

## Phase 1 results (89 enrollable subjects, chance 0.011)

| Normalization | Split | Macro-F1 | Shuffled control (ceiling 0.034) |
|---|---|---|---|
| **relative** | **cross-condition (headline)** | **0.378** | 0.009 |
| relative | temporal | 0.846 | 0.010 |
| absolute_log | cross-condition | 0.597 | 0.005 |
| absolute_log | temporal | 0.956 | 0.015 |

- **Brain-state change is the dominant effect:** 0.846 within condition vs 0.378 across.
- **Absolute − relative:** +0.219 / +0.110. Upper bound on the session-artifact
  contribution; anatomy cannot be separated out (D-004).
- **Leakage demonstration (D-017):** guarded 0.846 vs deliberately leaky 0.906; 90% of
  leaky test windows share samples; control on the leaky split 0.011 (chance).
- **Frontal EOG (D-004b):** does not concentrate; too small to act on.
- **EMG ablations (D-018):** without gamma 0.263 (drop 0.116), without the temporal
  sites 0.300 (drop 0.078). Both material; bounds, not EMG estimates. Importance
  re-routes in each: without gamma, beta leads and temporal sites still top the channel
  ranking; without the temporal sites, gamma still leads and Iz/frontal channels rise.
- **Delta (D-005):** kept. **Quality (D-015):** flagged windows scored, not excluded.
- **Reproducibility:** runs from `b3450bf` and `fa98250` produced byte-identical
  baseline artifacts.

### Phase 2 expectation: the per-subject tail

**14 of 89 subjects score F1 = 0 on the headline split** (21 below 0.1), while the
median is 0.36. Identity features survive the brain-state change for most subjects and
not at all for some. Expect the Phase 2 per-subject EER distribution to carry a heavy
tail. Report per-subject EER (distribution and worst decile), not only the pooled
number, and check whether the same subjects sit in both tails.

---

## Next up

1. **Commit** regenerated artifacts, README, and docs; push.
2. **Phase 1 close-out:** install Docker, run Compose, implement `db/connection.py` and
   `db/migrate.py`, apply `001_phase1_core.sql`, implement `ingest_subjects.py`
   (asserting DB cohorts match the committed holdout file).
3. Tick the Phase 1 checklist in `docs/ROADMAP.md`, then Phase 2.

---

## Open questions

- **Combined EMG ablation (D-018).** Each single ablation leaves the other route open.
  Removing gamma and the temporal sites together would bound the dependence on both at
  once. Not run; if it is, fix its materiality threshold first.
- **Phase 2: filter edge effects on short buffers.** Same preprocessing function on
  both paths, but `sosfiltfilt` edge transients differ between a 61 s recording and a
  short live buffer. D-001 removes code skew, not this.
- **Window size.** 2 s. Revisit in Phase 2 (D-005).
- **How many impostor subjects.** 20; nested, so the count can rise in Phase 2 (D-008).
- **Channel subset.** Not Phase 1. CAR must be part of that experiment's design (D-006).
- **Lint:** `scripts/verify_dataset.py` has an en dash in a print string (RUF001).

---

## Decisions deferred

- EEGNet vs Random Forest — Phase 6, only if Phases 1–5 are done
- Container service choice (ECS Fargate vs App Runner) — Phase 4, gated on WebSocket
  support
- Threshold values for challenge/revoke — Phase 2, tuned against the DET curve
- Quality-mask recalibration on the enrollable cohort — Phase 2 (D-015)
- Alembic — Phase 2, when SQLAlchemy models exist (D-009)

---

## Blockers

Docker is not installed; needed for the remaining Phase 1 checklist items.

---

## Session log

<!-- Append one entry per session. Newest at top. Keep entries short. -->

### 2026-09-14
**Phase:** 1
**Shipped:** `dsp/` (io, preprocess, windowing, features, pipeline) with tests;
egg-info untracked. `drop_partial` removed; quality threshold 250 → 500 uV.
`cohorts.py`, `scripts/select_holdout.py`, holdout committed alone and pushed.
`models/splits.py`, `models/baseline.py`, `models/evaluation.py`, `models/ablation.py`.
`config.fingerprint()`. Deliberately leaky split + overlap counting.
`scripts/train_baseline.py` with holdout-committed and clean-tree preconditions.
Smoke run; full baseline run from `b3450bf` (committed `6fa508d`); final run with EMG
ablations from `fa98250`. README results section. D-004 (corrected), D-004b, D-005,
D-018 (outcomes), D-007 (corrected), D-008 (updated), D-013–D-018.
**Next:** Commit; Docker/Postgres close-out.
**Notes / decisions:** Band-power contract was wrong, fixed (D-013). Quality mask at
250 uV flagged EOG and eyes-closed alpha; raised and made report-only (D-015). The
shuffled-label control cannot detect overlap leakage (D-007), confirmed on real data
(D-017). All evaluation thresholds fixed before the runs they judge (D-016). Headline
0.378. Frontal EOG does not concentrate. Absolute/relative wording corrected to an upper
bound. Both EMG ablations material, reported as bounds with re-routing noted. Baseline
artifacts reproduced byte for byte. User commits; never auto-commit.

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
