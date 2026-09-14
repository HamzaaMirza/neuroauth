# PROGRESS — NeuroAuth

**Current phase:** 1 — Signal pipeline + identification baseline
**Status:** `dsp/` implemented and tested. Paused for review before `models/`.
**Last updated:** 2026-09-14

---

## Where things stand

Dataset verified at the header level for all 218 baseline recordings (D-014).

`src/neuroauth/dsp/` — `io`, `preprocess`, `windowing`, `features` — is implemented
to contract. 82 tests pass (including the 4 slow real-data loader tests); the 12
remaining skips are the `cohorts` and `splits` stubs. mypy strict and ruff are clean.
On real data the chain runs at roughly 180 ms per 61 s recording and every feature
value is finite.

MNE is confined to `dsp/io.py` (D-001). One contract error was caught and fixed during
implementation: band-power integration (D-013).

Not yet implemented: `dsp/pipeline.py`, `config.fingerprint()`, `cohorts.py`,
everything in `models/` and `db/`, all scripts except `verify_dataset.py`.

---

## Next up

1. **Review checkpoint** — test results and the open questions below.
2. **Resolve the quality-mask threshold question** before any evaluation is designed
   around window exclusion.
3. **Fix the impostor holdout.** Approved design:
   - `cohorts.py` writes `config/impostor_holdout.json` with seed, subject list, and
     a UTC timestamp.
   - That file is committed **on its own**, with a message marking it pre-results.
   - `ingest_subjects.py` (later, with Docker) **asserts the DB rows match the file**
     and never re-derives the selection, so seed or library drift cannot silently
     produce a different holdout.
   - Paste the list into D-008.
4. `dsp/pipeline.py` + `config.fingerprint()`
5. `models/splits.py`, overlap assertion live
6. `models/baseline.py` + `models/evaluation.py`, shuffled-label control
7. `scripts/train_baseline.py` — four runs, artifacts committed → exit criteria

**Deferred this weekend:** Docker Compose, `db/`, migrations, `ingest_subjects.py`.
Docker is not installed and these are off the exit-criteria path.

---

## Open questions

- **Quality-mask peak-to-peak threshold (blocking step 6).** The textbook 250 uV
  default rejects 30% of windows on subjects 1–10, and it is not detecting clipping:
  - eyes-open: median window peak-to-peak on Fp1/AF7 is ~300 uV — ocular artifact
  - eyes-closed: O1 p95 is ~450 uV — occipital alpha, i.e. real signal
  - uneven: S009 loses all 60 windows in both runs; S001R02, S003, S010R02 lose ~2/3
  - windows ok at 350 uV: 82% open / 99% closed; at 500 uV: 90% / 100%

  If evaluation drops flagged windows, S009 leaves the class set entirely and the
  cross-condition split loses alpha-heavy eyes-closed windows. Needs a decision on the
  threshold and on whether Phase 1 evaluation excludes flagged windows at all. If the
  threshold is recalibrated from data, do it on the enrollable cohort only, after the
  holdout is committed.
- **`drop_partial` is a dead flag.** As contracted, both values produce identical
  output. Remove it from `WindowConfig` and the windowing signatures, or give it a
  meaning.
- **Phase 2: filter edge effects on short buffers.** Preprocessing is the same
  function on both paths, but Phase 1 filters a 61 s recording while a live stream
  filters a short buffer, and `sosfiltfilt` edge transients differ. D-001 removes
  code skew, not this. Phase 2 needs a buffering strategy (filter a rolling buffer
  longer than the window and keep its interior).
- **Window size.** 2 s. Revisit in Phase 2. Do not lengthen it for spectral
  resolution (D-005).
- **Delta band.** ~3 Welch bins. Check `band_importance` after the first fit (D-005).
- **How many impostor subjects.** Starting at 20; selection is nested, so the count
  can rise in Phase 2 without re-rolling (D-008).
- **Channel subset.** Not Phase 1. CAR must be part of that experiment's design (D-006).

---

## Decisions deferred

- EEGNet vs Random Forest — Phase 6, only if Phases 1–5 are done
- Container service choice (ECS Fargate vs App Runner) — Phase 4, gated on WebSocket
  support
- Threshold values for challenge/revoke — Phase 2, tuned against the DET curve
- Alembic — Phase 2, when SQLAlchemy models exist (D-009)

---

## Blockers

None for the exit-criteria path. Docker install needed before Phase 1 closes.

---

## Session log

<!-- Append one entry per session. Newest at top. Keep entries short. -->

### 2026-09-14
**Phase:** 1
**Shipped:** `dsp/io.py`, `dsp/preprocess.py`, `dsp/windowing.py`, `dsp/features.py`
with tests (82 passing). `src/neuroauth.egg-info/` untracked and gitignored.
D-013, D-014.
**Next:** Review, resolve the quality threshold, then commit the holdout file.
**Notes / decisions:** Band-power integration contract was wrong — fixed with
interpolated edges (D-013). All 218 baseline recordings load; none dropped (D-014).
Quality mask at 250 uV flags ocular artifacts and eyes-closed alpha as clipping —
open question. Holdout design approved: JSON file committed alone pre-results, DB
ingest asserts against it. Docker deferred.

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
