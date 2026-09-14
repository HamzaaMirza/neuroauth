# PROGRESS — NeuroAuth

**Current phase:** 1 — Signal pipeline + identification baseline
**Status:** Baseline run committed (`6fa508d`). EMG ablations and the corrected
absolute/relative wording are implemented and tested, **not yet committed or run**. The
next full run regenerates every artifact from one commit and adds `emg_ablation.json`.
**Last updated:** 2026-09-14

---

## Phase 1 results (full run from `b3450bf`, 89 enrollable subjects, chance 0.011)

| Normalization | Split | Macro-F1 | Shuffled control (ceiling 0.034) |
|---|---|---|---|
| **relative** | **cross-condition (headline)** | **0.378** | 0.009 |
| relative | temporal | 0.846 | 0.010 |
| absolute_log | cross-condition | 0.597 | 0.005 |
| absolute_log | temporal | 0.956 | 0.015 |

- **Brain-state change is the dominant effect.** Relative: 0.846 within condition vs
  0.378 across. Headline per-subject F1: median 0.359, IQR 0.110–0.611.
- **Absolute − relative:** +0.219 cross-condition, +0.110 temporal. Both material
  (D-016). Reported as an upper bound on the session-artifact contribution — amplitude
  also reflects anatomy, which one session cannot separate (D-004, corrected).
- **Leakage demonstration (D-017):** guarded temporal 0.846 vs deliberately leaky random
  0.906 (+0.060; apparent error 0.154 → 0.094). 90% of leaky test windows share samples
  with training. Control on the leaky split: 0.011, i.e. chance (D-007 confirmed).
- **Frontal EOG (D-004b): does not concentrate.** 1.23× / 1.35× uniform on eyes-open
  models vs 1.05× / 0.88× on both-condition models. Right direction, too small to act
  on. Frontal-excluded variant not run.
- **Lateral-temporal gamma leads importance** in all four models → EMG ablations
  (D-018), pending.
- **Delta (D-005):** 13–15% on relative models, level with theta and alpha. Kept.
- **Quality (D-015):** not-ok windows 5.7% eyes-open, 1.0% eyes-closed. S009 scores F1
  0.81 on the headline — excluding flagged windows would have dropped it.

### Phase 2 expectation: the per-subject tail

**14 of 89 subjects score F1 = 0 on the headline split** (21 below 0.1), while the
median is 0.36. Identity features survive the brain-state change for most subjects and
not at all for some. Expect the Phase 2 per-subject EER distribution to carry a heavy
tail. Report per-subject EER (distribution and worst decile), not only the pooled
number, and check whether the same subjects sit in both tails.

---

## Next up

1. **Commit** the ablation work and the wording correction; push.
2. **Full run** from that commit: regenerates all artifacts. The four baseline models
   use fixed seeds, so their numbers should reproduce exactly — check that, then read
   `emg_ablation.json`.
3. **README results section**, framing the EMG outcome as a bound, not a decomposition.
4. Commit artifacts → **Phase 1 exit criteria met**.
5. Remaining Phase 1 checklist items off the exit path: Docker Compose, applying the
   Postgres schema.

---

## Open questions

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

None for the exit-criteria path. Docker install needed before Phase 1 closes.

---

## Session log

<!-- Append one entry per session. Newest at top. Keep entries short. -->

### 2026-09-14
**Phase:** 1
**Shipped:** `dsp/` (io, preprocess, windowing, features, pipeline) with tests;
egg-info untracked. `drop_partial` removed; quality threshold 250 → 500 uV.
`cohorts.py`, `scripts/select_holdout.py`, holdout committed alone and pushed.
`models/splits.py`, `models/baseline.py`, `models/evaluation.py`.
`config.fingerprint()`. Deliberately leaky split + overlap counting.
`scripts/train_baseline.py` with holdout-committed and clean-tree preconditions.
Smoke run, then the full baseline run (661 s) from `b3450bf`, committed as `6fa508d`.
Then: `models/ablation.py`, EMG ablations in the driver, corrected absolute/relative
wording. D-004 (corrected), D-004b (outcome), D-005 (outcome), D-007 (corrected),
D-008 (updated), D-013–D-018.
**Next:** Commit, full run with ablations, README results.
**Notes / decisions:** Band-power contract was wrong, fixed (D-013). Quality mask at
250 uV flagged EOG and eyes-closed alpha; raised and made report-only (D-015). The
shuffled-label control cannot detect overlap leakage (D-007), confirmed on real data
(D-017). All evaluation thresholds fixed before the runs they judge (D-016). Headline
0.378. Frontal EOG does not concentrate. The absolute/relative wording claimed a
decomposition single-session data cannot support — corrected to an upper bound.
Lateral-temporal gamma leads importance → two ablations, framed as bounds (D-018).
User commits; never auto-commit.

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
