# PROGRESS — NeuroAuth

**Current phase:** 1 — Signal pipeline + identification baseline
**Status:** Contracts approved and scaffolded. No function bodies written yet.
**Last updated:** 2026-08-21

---

## Where things stand

Dataset verified: 109 subjects, 64 channels at 160 Hz confirmed on subject 1.
`scripts/verify_dataset.py` works.

Phase 1 contracts are reviewed, amended, and scaffolded to disk. Every function is
signature + type hints + docstring with a `raise NotImplementedError` body. Postgres
schema, Docker Compose, pyproject, and the test skeleton are in place.
`docs/DECISIONS.md` has twelve entries covering every non-obvious Phase 1 choice.

Structural decision that shapes everything downstream: **MNE is confined to
`src/neuroauth/dsp/io.py`.** Preprocessing, windowing, and features are numpy in /
numpy out, so the identical transform runs over the Phase 2 WebSocket stream where no
MNE `Raw` object exists. See D-001.

---

## Next up

1. `pip install -e ".[dev]"` — dev deps are not yet in the venv
2. Implement `dsp/io.py`, `dsp/preprocess.py`, `dsp/windowing.py`, `dsp/features.py`
3. Implement `cohorts.py` and run `scripts/ingest_subjects.py` — **fixes the impostor
   holdout before any result is looked at**; paste the subject list into D-008
4. Implement `models/splits.py` with the overlap assertion live
5. Implement `models/baseline.py` and `models/evaluation.py`
6. Fill in the test suite (currently function names + docstrings, all skipped)
7. Run `scripts/train_baseline.py` — four runs, artifacts committed

---

## Open questions

- **Window size.** 2 s. Shorter reduces time-to-detect for impostor swaps but adds
  noise. Revisit with data in Phase 2. Do not lengthen it to buy spectral resolution
  (D-005).
- **Delta band.** Roughly three Welch bins at 1 Hz resolution — thin. Check
  `band_importance` after the first fit; drop the band and record why if it
  contributes nothing (D-005).
- **How many impostor subjects.** Starting at 20 of 109. The selection is a truncated
  seeded permutation, so it is *nested*: raising the count in Phase 2 keeps every
  subject already committed and only adds more. The count can be decided on a FAR
  stability argument without re-rolling anything (D-008).
- **Channel subset.** 64 available. DEAP work showed ~1.85pp loss from 32 to 8. Not a
  Phase 1 concern — and if it happens, the CAR flag must be part of that experiment's
  design, not a fixed setting (D-006).

---

## Decisions deferred

- EEGNet vs Random Forest — Phase 6, only if Phases 1–5 are done
- Container service choice (ECS Fargate vs App Runner) — Phase 4, gated on WebSocket
  support
- Threshold values for challenge/revoke — Phase 2, tuned against the DET curve
- Alembic — Phase 2, when SQLAlchemy models exist for it to autogenerate against
  (D-009)

---

## Blockers

None.

---

## Session log

<!-- Append one entry per session. Newest at top. Keep entries short. -->

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
