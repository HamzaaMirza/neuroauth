# PROGRESS — NeuroAuth

**Current phase:** 1 — Signal pipeline + identification baseline
**Status:** Not started
**Last updated:** (set on first session)

---

## Where things stand

Nothing built yet. Repo is empty. Specs and roadmap are written.

---

## Next up

Phase 1, in order:

1. Environment: Python 3.11+, MNE-Python, scikit-learn, FastAPI, pytest
2. Download `eegmmidb` from PhysioNet — verify 109 subject directories, EDF+ readable
3. Loader for baseline runs (R01 eyes-open, R02 eyes-closed)
4. Preprocessing: bandpass 1–50 Hz, notch 60 Hz
5. Windowing: 2s, 50% overlap
6. `extract_features` — Welch PSD band powers per channel + quality mask
7. Random Forest identification baseline
8. Evaluation: macro-F1, per-class F1, confusion matrix committed to repo
9. Postgres schema + Docker Compose
10. pytest for the feature pipeline

---

## Open questions

- **Window size.** 2s is the starting point. Shorter reduces time-to-detect for impostor swaps
  but adds noise. Revisit with data in Phase 2 — this is a defensible trade-off to write up.
- **Channel subset.** 64 channels available. The DEAP work showed only ~1.85pp loss going from
  32 to 8 channels. Worth testing here too — a reduced montage is more realistic for consumer
  hardware and makes a good README section. Not a Phase 1 concern.
- **How many impostor subjects to hold out.** Needs to be enough for a stable FAR estimate.
  Decide in Phase 2, record the reasoning.

---

## Decisions deferred

- EEGNet vs Random Forest — Phase 6, only if Phases 1–5 are done
- Container service choice (ECS Fargate vs App Runner) — Phase 4, gated on WebSocket support
- Threshold values for challenge/revoke — Phase 2, tuned against the DET curve

---

## Blockers

None.

---

## Session log

<!-- Append one entry per session. Newest at top. Keep entries short. -->

### (template)
**Date:**
**Phase:**
**Shipped:**
**Next:**
**Notes / decisions:**
