# PROGRESS — NeuroAuth

**Current phase:** 1 — Signal pipeline + identification baseline
**Status:** Full run complete from clean, pushed commit `b3450bf`. Every shuffled-label
control passed. Artifacts written to `artifacts/`, **not yet committed**. The README
results section waits on two decisions (below).
**Last updated:** 2026-09-14

---

## Phase 1 results (full run, 89 enrollable subjects, chance 0.011)

| Normalization | Split | Macro-F1 | Shuffled control (ceiling 0.034) |
|---|---|---|---|
| **relative** | **cross-condition (headline)** | **0.378** | 0.009 |
| relative | temporal | 0.846 | 0.010 |
| absolute_log | cross-condition | 0.597 | 0.005 |
| absolute_log | temporal | 0.956 | 0.015 |

- **Brain-state change is the dominant effect.** Relative: 0.846 within condition vs
  0.378 across. Headline per-subject F1: median 0.359, IQR 0.110–0.611, 14 of 89
  subjects at 0.
- **Absolute − relative:** +0.219 cross-condition, +0.110 temporal. Both over the 0.05
  materiality threshold (D-016). The temporal gap is compressed by the ceiling.
- **Leakage demonstration (D-017):** guarded temporal 0.846 vs deliberately leaky
  random 0.906 (+0.060; apparent error 0.154 → 0.094). 90% of leaky test windows share
  samples with training. The control on the leaky split reads 0.011 — chance —
  confirming on real data that it cannot see this leak (D-007).
- **Frontal EOG (D-004b):** eyes-open-trained models put 1.23× (relative) and 1.35×
  (absolute) the uniform share on Fp1/Fp2/AF7/AF8. Those channels rank 9th–20th of
  64, and 11–13% of random 4-channel groups carry as much. Models trained on both
  conditions: 1.05× and 0.88×. The direction fits the hypothesis; the size is modest.
  No a priori criterion for "concentrates" was set.
- **Top importance is lateral-temporal and gamma** in all four models: T9, T10, T7,
  T8, TP7/TP8, FT8 and Iz lead; gamma is the top band (38–48%). Open question below.
- **Delta (D-005):** 13–15% on relative models, level with theta and alpha. Kept.
- **Quality (D-015):** not-ok windows 5.7% eyes-open, 1.0% eyes-closed; frontal flags
  20.2% vs 1.4%. S009 — which lost every window at 250 uV — scores F1 0.81 on the
  headline. Excluding flagged windows would have removed a well-identified subject.
- **Provenance:** `git_head` b3450bf, `git_dirty` false, fingerprints
  relative 3d3c06755c838642 / absolute_log be030ad69d2fd7a3.

---

## Next up

1. **Commit `artifacts/` on its own** (produced by `b3450bf`), then push.
2. **Decide the absolute/relative wording** before it reaches the README (open question).
3. **Decide on an EMG check** (open question). If yes: fix the comparison criterion
   before running, as D-016 did.
4. **Decide on the frontal-excluded variant** (D-004b). Current read: not warranted.
5. README results section → **Phase 1 exit criteria met**.
6. Remaining Phase 1 checklist items off the exit path: Docker Compose, applying the
   Postgres schema.

---

## Open questions

- **Absolute/relative interpretation wording.** The generated text says the gap is
  "consistent with recording-level amplitude confounds … rather than identity
  information". The first half holds. The second overclaims: absolute amplitude also
  reflects anatomy — skull thickness, tissue conductivity, head geometry — which is
  person-specific and would survive a second session. Single-session data cannot
  separate the two, so the gap is an upper bound on the session-artifact
  contribution, not a measure of it.
- **Muscle artifact (EMG) as a third confound.** Gamma (30–50 Hz) at lateral-temporal
  electrodes (T9/T10/T7/T8 sit over the temporalis muscle) and at Iz (near neck
  muscles) is where scalp EMG shows up. Muscle tone and jaw habits are person-specific
  and stable within a sitting — the same class of confound as D-004 and D-004b. The
  importance pattern fits that; it does not prove it. A check could drop the gamma
  band, drop the temporal-edge channels, or both, and report each against the
  headline.
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
Smoke run, then the full run (661 s) from `b3450bf`. D-004b, D-008 (updated),
D-013–D-017; D-005 outcome; D-007 corrected twice.
**Next:** Commit artifacts; decide wording, EMG check, frontal variant; README results.
**Notes / decisions:** Band-power contract was wrong, fixed (D-013). Quality mask at
250 uV flagged EOG and eyes-closed alpha; raised and made report-only (D-015). The
shuffled-label control cannot detect overlap leakage (D-007), confirmed on real data by
the leakage demonstration (D-017). Both evaluation thresholds fixed a priori (D-016).
Full run: headline 0.378 (34× chance); the brain-state change is the dominant effect;
absolute power is materially higher; frontal EOG signal modest; lateral-temporal gamma
leads importance — possible EMG confound, open. User commits; never auto-commit.

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
