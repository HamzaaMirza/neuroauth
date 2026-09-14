# PROGRESS — NeuroAuth

**Current phase:** 1 — Signal pipeline + identification baseline
**Status:** `dsp/`, `cohorts`, and `models/` implemented and tested. Impostor holdout
selected; its file awaits its own pre-results commit. No model has touched real data.
**Last updated:** 2026-09-14

---

## Where things stand

- **`dsp/`** (io, preprocess, windowing, features) — implemented, committed in
  `99230d8`. `drop_partial` since removed.
- **Quality mask** — peak-to-peak threshold raised to 500 uV; flagged windows are
  scored, not excluded, in Phase 1 (D-015).
- **Impostor holdout** — `config/impostor_holdout.json` written 2026-09-14T14:06:19Z:
  subjects 3, 11, 18, 20, 21, 35, 36, 38, 49, 65, 75, 79, 82, 86, 87, 96, 97, 105, 106,
  107. The file is the source of truth; nothing re-derives it (D-008). **Not committed
  yet.**
- **`models/`** — `splits` (temporal with guard, cross-condition, overlap assertion),
  `baseline` (RF, shuffled-label control, band/channel importance, importance share),
  `evaluation` (report, control gate, normalization comparison, artifacts, per-recording
  quality flag rates). Tested on synthetic data only.
- **Tests:** 156 (155 pass + the committed-holdout check, which runs once the file
  exists). mypy strict clean on `dsp`, `config`, `cohorts`, `models`.

Not yet implemented: `dsp/pipeline.py`, `config.fingerprint()`,
`scripts/train_baseline.py`, everything in `db/`.

---

## Next up

1. **Commit the holdout file on its own** (pre-results), then push if a remote exists.
   No real-data model run happens before this.
2. Commit the remaining work.
3. `dsp/pipeline.py` + `config.fingerprint()`.
4. `scripts/train_baseline.py`: four runs, shuffled-label gate, holdout loaded from the
   file and asserted excluded, `quality_flag_rates.csv`, frontal importance share on
   eyes-open-trained models, band importance.
5. Review results before writing them up: absolute/relative gap (D-004), frontal share
   vs 6.25% uniform (D-004b), delta importance (D-005).
6. README results section; commit artifacts → **exit criteria met**.

**Deferred this weekend:** Docker Compose, `db/`, migrations, `ingest_subjects.py`.

---

## Open questions

- **Random-split reference run (D-007).** The shuffled-label control cannot detect
  window-overlap leakage. A deliberately leaky random split, run once and labelled as
  such, would quantify how much overlap inflates the score — the empirical half of
  the "my number is lower because" argument. Not built; needs a yes/no.
- **Frontal-excluded variant (D-004b).** Built only if frontal importance concentrates
  on the eyes-open-trained model.
- **Phase 2: filter edge effects on short buffers.** Same preprocessing function on
  both paths, but `sosfiltfilt` edge transients differ between a 61 s recording and a
  short live buffer. D-001 removes code skew, not this. Needs a buffering strategy.
- **Window size.** 2 s. Revisit in Phase 2 (D-005).
- **Delta band.** ~3 Welch bins. Check `band_importance` after the first fit (D-005).
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
**Shipped:** `dsp/` (io, preprocess, windowing, features) with tests; egg-info untracked.
Then: `drop_partial` removed; quality threshold 250 → 500 uV; `cohorts.py` +
`scripts/select_holdout.py`; holdout file generated; `models/splits.py`,
`models/baseline.py`, `models/evaluation.py` with tests. D-004b, D-008 (updated),
D-013, D-014, D-015; D-007 corrected.
**Next:** Holdout commit, then `pipeline.py`, `fingerprint()`, `train_baseline.py`.
**Notes / decisions:** Band-power contract was wrong, fixed (D-013). Quality mask at
250 uV flagged EOG and eyes-closed alpha as clipping; raised and made report-only
(D-015). Frontal EOG is a second single-session identity confound that relative power
does not remove (D-004b). The shuffled-label control cannot detect overlap leakage —
D-007 claimed it could; corrected, and a test demonstrates the limit.
`temporal_split` gained a `window_s` argument (onsets can't imply window length).
`IdentificationReport.n_low_quality_excluded` replaced by `n_test_not_ok`. Holdout
subject 3 was in the pre-selection quality statistics; disclosed in D-015. User
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
