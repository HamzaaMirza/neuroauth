# PROGRESS — NeuroAuth

**Current phase:** 1 — Signal pipeline + identification baseline
**Status:** Phase 1 code complete. A 5-subject smoke run passed end to end on real EEG.
The full run is next, and needs a clean working tree first: the driver refuses to write
`artifacts/` from uncommitted code.
**Last updated:** 2026-09-14

---

## Where things stand

- **Impostor holdout** — `config/impostor_holdout.json`, committed alone in `1f25cd2`
  and pushed to `origin/main` before any model touched real EEG (D-008).
- **`dsp/`** — io, preprocess, windowing, features, and now `pipeline.py`.
- **`config.fingerprint()`** — canonical SHA-256, band order included, stable across
  processes.
- **`models/`** — splits (temporal with guard, cross-condition, the deliberately leaky
  random split, overlap counting), baseline, evaluation. Committed in `a396033` except
  this session's additions.
- **`scripts/train_baseline.py`** — four evaluations plus the leakage demonstration.
  Checked before any EEG loads: the holdout file is in HEAD and unmodified, and writing
  `artifacts/` requires a clean tree. Every shuffled-label control is gated before any
  artifact is written. Records git HEAD, a dirty flag, config fingerprints, split
  parameters, the a priori thresholds, and package versions.
- **Tests:** 185. mypy strict clean across `src/neuroauth`.
- **Smoke run** (5 subjects, scratchpad only, 17 s): every artifact produced and
  rendered. Too few subjects for the numbers to mean anything. One control read 0.41
  against chance 0.20 — inside the ceiling, but it showed the control is noisier than
  D-007 first claimed. D-007 was corrected before the full run.

---

## Next up

1. **Commit the current work** and push.
2. **Full run:** `python -m scripts.train_baseline` → `artifacts/`.
3. **Review before writing anything up:**
   - every shuffled-label control against its 3× ceiling (D-007, D-016)
   - absolute vs relative gap against 0.05 (D-004, D-016)
   - frontal importance share vs 6.25% uniform on the eyes-open models (D-004b)
   - delta band importance (D-005)
   - leaky-minus-honest inflation (D-017)
4. README results section; commit artifacts → **exit criteria met**.

**Deferred this weekend:** Docker Compose, `db/`, migrations, `ingest_subjects.py`.

---

## Open questions

- **Frontal-excluded variant (D-004b).** Built only if frontal importance concentrates
  on the eyes-open-trained models.
- **Control false alarms (D-007).** A clean pipeline can occasionally breach 3× chance.
  If the full run fails a control: investigate and document, don't re-roll seeds or
  move the ceiling.
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
5-subject smoke run. D-004b, D-008 (updated), D-013–D-017; D-007 corrected twice.
**Next:** Commit, full run, review, README results.
**Notes / decisions:** Band-power contract was wrong, fixed (D-013). Quality mask at
250 uV flagged EOG and eyes-closed alpha; raised and made report-only (D-015). Frontal
EOG is a second single-session confound that relative power does not remove (D-004b).
The shuffled-label control cannot detect overlap leakage (D-007 correction, with a
synthetic demonstration); a labelled leaky split measures that on real data (D-017).
Both evaluation thresholds fixed a priori and pinned by a test (D-016). Smoke run
showed the control is noisier than first estimated; D-007 corrected before the full
run. Holdout subject 3 was in pre-selection quality statistics; disclosed. User
commits; never auto-commit.

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
