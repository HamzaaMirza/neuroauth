# CLAUDE.md — NeuroAuth

Read this at the start of every session. Then read `docs/PROGRESS.md` for current state and
`docs/ROADMAP.md` for where this phase sits.

---

## What this is

A continuous biometric authentication system. It verifies identity from a live EEG stream,
revokes the session when the signal stops matching, learns from reviewed false rejections, and
retrains itself behind an evaluation gate.

This is a portfolio project targeting ML Platform / AI Product Engineer roles. Optimize for
**defensibility over impressiveness**. Every component must be something the author can
whiteboard and justify under interview questioning — including things deliberately not built.

---

## Hard rules — do not violate these

These are architectural invariants, not preferences. If a change would break one, stop and
raise it rather than working around it.

1. **Verification, not identification.** This is an open-set problem: "is this stream subject
   X, or an impostor?" Never collapse it to closed-set multi-class classification over known
   subjects. A held-out set of impostor subjects — never enrolled, never trained on — must
   exist at all times.

2. **Report EER, FAR, FRR. Never accuracy.** Accuracy on a biometric system is meaningless.
   Any evaluation output must include EER and FRR at a fixed FAR.

3. **Raw EEG and raw feature vectors are never persisted.** Enrollment applies a cancelable
   (non-invertible, revocable) transform before storage. Nothing invertible reaches Postgres,
   S3, or a log line. If a function would return or write raw features outside the in-memory
   inference path, that's a bug.

4. **The promotion gate must never trade security for usability.** A candidate model that
   improves FRR while regressing FAR is strictly worse and must be rejected. Gate criteria:
   EER improvement above threshold AND FAR regression within tolerance.

5. **Holdout integrity is asserted in code, not assumed.** `build_training_set` must assert
   zero overlap with the frozen holdout and that no held-out impostor subject appears in
   training data. These assertions run on every retrain.

6. **DEAP data never enters this repo.** DEAP is research-licensed and belongs to a separate
   track. This project uses PhysioNet eegmmidb (ODC-By) exclusively.

7. **Don't add infrastructure the scale doesn't justify.** No Kubernetes, no GPU, no SageMaker,
   no dedicated vector DB, no Airflow. If a simpler tool works, use it and record why in
   `docs/DECISIONS.md`.

---

## Author-writes-by-hand list

Do not generate these. Scaffold the signature and types, leave the body as `raise
NotImplementedError` with a TODO, and tell the author it's theirs:

- `update_session` — session confidence decay and threshold logic
- `promotion_gate` — the promote/reject decision
- `compute_drift_metrics` — drift detection logic

These are the components interviewers probe hardest. The author must own them fully.

---

## Stack

| Layer | Choice |
|---|---|
| Signal processing | MNE-Python (EDF loading, filtering), SciPy, NumPy |
| ML | scikit-learn (baseline RF), PyTorch (EEGNet, later phase) |
| Backend | FastAPI, WebSocket for streaming, REST for control |
| DB | PostgreSQL |
| Tracking | MLflow (tracking + model registry) |
| Frontend | React + Vite |
| Container | Docker, Docker Compose locally |
| Cloud | AWS — ECS Fargate/App Runner, RDS, S3, ECR, Secrets Manager, CloudWatch |
| IaC | Terraform |
| CI/CD | GitHub Actions |
| Tests | pytest |

---

## Data

**PhysioNet EEG Motor Movement/Imagery Database (`eegmmidb` v1.0.0)**
109 subjects, 64 channels (10-10 montage), 160 Hz, EDF+.
Per subject: R01 (baseline, eyes open), R02 (baseline, eyes closed), R03–R14 (task runs).

- **Primary signal: the baseline runs.** Resting-state is the realistic auth input.
- **Task runs are a robustness holdout** — does verification survive a change in mental task?
- **License: ODC-By.** Attribution required in README and app footer. Non-negotiable.
- **Single-session limitation.** Cross-session drift cannot be demonstrated with this dataset.
  Monitoring detects *within-session temporal drift* across runs. Never imply otherwise.

---

## Conventions

- **Contracts before implementation.** The author specifies function signatures and docstrings
  first. Implement to the contract; if the contract seems wrong, say so rather than silently
  changing it.
- Type hints everywhere. `mypy`-clean where practical.
- Pure functions for signal processing — no I/O inside `extract_features` and friends.
- Inference path never raises on malformed input. Return a low-quality/low-confidence result
  and let session logic decide.
- Every model artifact is versioned in MLflow. No pickle files in folders.
- Every gate decision, session transition, and rejection review is logged with provenance:
  timestamp, model_version, threshold config in effect, actor.
- Secrets in Secrets Manager or `.env` (gitignored). Never committed, never in a log line.
- Commit messages: `phase-N: short imperative summary`.

---

## Working style

- One phase at a time. Do not start work belonging to a later phase, even if it seems trivial —
  scope creep is the main risk to this project.
- At the end of each session, update `docs/PROGRESS.md`: what shipped, what's next, any open
  questions or decisions deferred.
- When a non-obvious choice is made (library, architecture, threshold), append it to
  `docs/DECISIONS.md` with the alternatives considered and why they were rejected. This file is
  interview prep, so write it for a reader who wasn't there.
- Flag honestly when something is simulated (replayed streams, injected impostors, injected
  drift). These get documented in the README, not hidden.
- If something is taking longer than the phase budget, say so and propose what to cut.

---

## Cost discipline

AWS budget: under $30/month. Billing alarm at $40 before any provisioning. Free-tier RDS
`db.t4g.micro`. Single small container task. `terraform destroy` between work sessions once the
demo video exists. No GPU instances, ever.
