# NeuroAuth

Continuous biometric authentication from a live EEG stream. Verifies identity against
a claimed subject, revokes the session when the signal stops matching, learns from
reviewed false rejections, and retrains behind an evaluation gate.

**Status: Phase 1 in progress** — signal pipeline and identification baseline. See
[`docs/ROADMAP.md`](docs/ROADMAP.md) for the plan and
[`docs/PROGRESS.md`](docs/PROGRESS.md) for current state.

---

## Data and attribution

This project uses the **PhysioNet EEG Motor Movement/Imagery Database (eegmmidb)
v1.0.0** — 109 subjects, 64 channels, 160 Hz, EDF+.

> Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R.,
> Mietus, J. E., Moody, G. B., Peng, C. K., & Stanley, H. E. (2000). PhysioBank,
> PhysioToolkit, and PhysioNet: Components of a new research resource for complex
> physiologic signals. *Circulation*, 101(23), e215–e220.
>
> Schalk, G., McFarland, D. J., Hinterberger, T., Birbaumer, N., & Wolpaw, J. R.
> (2004). BCI2000: A general-purpose brain-computer interface (BCI) system. *IEEE
> Transactions on Biomedical Engineering*, 51(6), 1034–1043.

Licensed under **Open Data Commons Attribution License (ODC-By) v1.0**. Attribution
is required and is reproduced here and in the application footer.

### Stated limitation

eegmmidb is **single-session**. Cross-session drift cannot be demonstrated with this
dataset, and this project never claims otherwise. Monitoring detects *within-session
temporal drift* across runs.

---

## Getting started

```bash
python -m venv .venv && .venv/Scripts/activate    # Windows
pip install -e ".[dev]"

python -m scripts.verify_dataset --subjects 5      # fetch + validate
docker compose up -d postgres
python -m scripts.migrate_db
python -m scripts.ingest_subjects                  # fixes the impostor holdout
python -m scripts.train_baseline
```

---

## Results

*Populated at the end of Phase 1. Every reported number carries its chance level and
its shuffled-label control alongside it.*
