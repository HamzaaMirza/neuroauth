# ROADMAP — NeuroAuth

Six phases. Phases 1–5 are the complete, resume-bearing project. Phase 6 is optional polish.

**Rule: do not start a phase until the previous phase's exit criteria are met.** Scope creep is
the main failure mode for this project.

---

## Phase 1 — Signal pipeline + identification baseline
**Budget: 1 weekend**

Get from raw EDF files to a working classifier. Deliberately closed-set at this stage — Phase 2
reframes it. This phase exists to build the signal pipeline and establish a baseline number.

- [x] Download and verify `eegmmidb` (109 subjects, EDF+)
- [x] MNE loader: read EDF, extract baseline runs R01/R02
- [x] Preprocessing: bandpass 1–50 Hz, notch 60 Hz
- [x] Windowing: 2s windows, 50% overlap
- [x] Feature extraction: Welch PSD band powers (δ θ α β γ) per channel, with quality mask
- [x] Random Forest identification baseline across subjects
- [x] Evaluation: macro-F1, per-class F1, confusion matrix
- [x] pytest suite covering the feature pipeline
- [x] `docs/DECISIONS.md` started
- Postgres schema and Docker Compose → moved to Phase 2 (D-019)

**Exit criteria:** a window goes in, a subject prediction comes out, and there is a macro-F1
number with a confusion matrix committed to the repo.

**Status: complete (2026-09-14).** Results in the README; artifacts in `artifacts/`.

---

## Phase 2 — Verification + continuous session + template protection
**Budget: 1 weekend**

This is where it becomes an authentication system rather than a classifier.

- [ ] Docker Compose: app + postgres *(moved from Phase 1, D-019)*
- [ ] Apply the Postgres schema — `migrations/001_phase1_core.sql` is written; implement
      `db/connection.py`, `db/migrate.py`, and `ingest_subjects.py` *(moved from Phase 1, D-019)*
- [ ] Hold out N subjects entirely as impostors — never enrolled, never trained on
      *(20 already fixed and committed in Phase 1, D-008; confirm the count for a stable FAR)*
- [x] Reframe to open-set verification: score against a claimed identity *(code; full run
      pending, D-022)*
- [x] Bounded-context filtering shared by offline and live paths (D-020)
- [x] Metrics: EER, FAR, FRR, FRR@FAR=0.001, DET curve *(headline FRR@FAR=0.01; 0.001 is
      reported but flagged under-resolved)*
- [x] Decision layer v0 on score-level inputs, cross-fitted in evaluation (D-021)
- [x] Cancelable transform: keyed BioHash, per-user key derived from a master secret, no
      stored seed (D-023)
- [x] `enroll_subject` — protected template only, raw features never persisted
- [ ] `revoke_and_reissue` — verify revocation actually works end to end
- [ ] WebSocket `/stream/{session_id}` — windowed inference over a replayed stream
- [ ] `update_session` — EMA confidence decay, challenge and revoke thresholds *(author writes)*
- [ ] Minimal React live-session view: confidence over time, lock state
- [ ] Impostor injection: swap subjects mid-stream

**Exit criteria:** replay a genuine stream, swap in an impostor mid-session, watch the session
revoke. DET curve and EER committed.

---

## Phase 3 — The human-in-the-loop retraining loop
**Budget: 1 weekend**

The centerpiece. This is what separates the project from portfolio ML.

The embedding is fixed at enrollment; retraining adjusts the decision layer on stable
templates (D-021). The correction loop and the gate are unaffected.

- [ ] False-rejection review UI (admin view)
- [ ] `record_false_rejection` — full provenance: window scores, representation_version,
      decision_version, thresholds, reviewer
- [ ] `build_training_set` — score-level rows for the decision layer, with holdout-overlap and
      impostor-leakage assertions
- [ ] MLflow: tracking server, experiment logging, model registry
- [ ] Backfill existing DEAP experiment runs into MLflow *(metadata only — no DEAP data in repo)*
- [ ] `evaluate_candidate` — EER/FAR/FRR/latency for candidate vs incumbent decision layer, on
      score tables from the frozen representation
- [ ] `promotion_gate` — promote/reject with rationale *(author writes)*
- [ ] Retrain script, runnable manually — retrains the decision layer; the embedding stays fixed
- [ ] Rollback path: pinned model version in config

**Exit criteria:** review a rejection, run a retrain, and watch the gate **reject** a model that
regressed FAR. The rejection must be logged with a readable rationale.

---

## Phase 4 — Cloud + CI/CD
**Budget: 1 weekend**

- [ ] Billing alarm at $40 — **before provisioning anything**
- [ ] Terraform: VPC, RDS, S3, ECR, ECS Fargate (or App Runner), Secrets Manager, IAM roles
- [ ] Verify WebSocket support on the chosen container service before committing to it
- [ ] Scoped IAM task roles — no wildcard policies
- [ ] GitHub Actions: app CI (lint, test, build, push, deploy on merge to main)
- [ ] GitHub Actions: scheduled weekly retrain running the full gate
- [ ] `terraform destroy` verified to work cleanly

**Exit criteria:** merge to main deploys automatically; the weekly retrain runs unattended and
logs its gate decision.

---

## Phase 5 — Monitoring + load testing
**Budget: 1 weekend**

- [ ] `/api/metrics` aggregates endpoint
- [ ] Admin dashboard: score distributions, FRR trend, revocation rate, latency
- [ ] `compute_drift_metrics` — PSI on band powers, score shift, FRR trend *(author writes)*
- [ ] Within-session temporal drift detection across runs
- [ ] `tests/test_drift.py` — inject synthetic shift, assert the alert fires
- [ ] CloudWatch alarms on drift and error rate
- [ ] Load test (k6 or Locust): per-window p50/p95/p99, max concurrent streams
- [ ] Publish latency numbers in README

**Exit criteria:** a dashboard worth screenshotting and a latency number worth quoting.

---

## Phase 6 — Security hardening + model comparison *(optional)*
**Budget: 1 weekend**

- [ ] Replay-attack detection: window hashing, stationarity checks, timing envelope
- [ ] `tests/test_replay.py` — mount an actual replay attack, assert it's caught
- [ ] `docs/THREAT_MODEL.md` — what an attacker gets, what they can't reconstruct, known limits
- [ ] Rate limiting on enroll and challenge endpoints
- [ ] EEGNet / 1D CNN vs the frozen LDA embedding — publish the delta on both EER and latency
- [ ] If the CNN doesn't clearly win, keep the LDA embedding and say so in the README. Adopting
      it would be a representation migration requiring re-enrollment, not a retrain (D-021)

---

## Cut list

In order, when time runs short:

1. EEGNet comparison (Phase 6)
2. Replay detection (Phase 6)
3. Admin dashboard polish (Phase 5) — keep the metrics, drop the visual refinement
4. Task-run robustness holdout (Phase 2 stretch)

**Phases 1–5 alone are a complete, defensible project.** Do not start Phase 6 until they are
genuinely finished.

---

## Final deliverables

- [ ] Live demo URL, seeded, always working
- [ ] README: architecture diagram, DET curve, EER table, latency numbers, single-session
      limitation, design decisions section, ODC-By attribution
- [ ] 3-minute demo video: enroll → session → impostor swap → revoke → review → retrain → gate
      rejection → dashboard
- [ ] `docs/DECISIONS.md` — the ADR log, written as interview prep
- [ ] `docs/THREAT_MODEL.md` *(Phase 6)*

---

## Parked

**CivicTriage** — complaint routing with the same skeleton (intake → model → human correction →
gated retrain → monitoring). Second portfolio piece. Revisit after Phase 5; the shared
architecture makes it substantially faster to build second. Spec in `civictriage-spec.md`.
