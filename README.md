# NeuroAuth

Continuous biometric authentication from a live EEG stream. Verifies identity against
a claimed subject, revokes the session when the signal stops matching, learns from
reviewed false rejections, and retrains behind an evaluation gate.

**Status: Phase 1 results in** — signal pipeline and identification baseline. See
[`docs/ROADMAP.md`](docs/ROADMAP.md) for the plan, [`docs/PROGRESS.md`](docs/PROGRESS.md)
for current state, and [`docs/DECISIONS.md`](docs/DECISIONS.md) for why things are the
way they are.

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
temporal drift* across runs. The same limitation bounds what the Phase 1 results below
can say about confounds.

---

## Phase 1 results: identification baseline

Closed-set identification over the 89 enrollable subjects. This is a scaffold that
validates the signal pipeline; Phase 2 replaces it with open-set verification reported
as EER, FAR, and FRR. Twenty impostor subjects were chosen and committed before any
result existed ([`config/impostor_holdout.json`](config/impostor_holdout.json)) and are
untouched here.

**Setup.** Resting-state runs only (R01 eyes open, R02 eyes closed). 2 s windows with
50% overlap, Welch band power in five bands across 64 channels, Random Forest. Chance
macro-F1 is 1/89 = **0.011**. Every model also gets a shuffled-label control, and every
threshold used to judge a result was fixed before the run it judges
([D-016](docs/DECISIONS.md)).

| Band power | Split | Macro-F1 | Shuffled-label control |
|---|---|---|---|
| **relative** | **eyes open → eyes closed (headline)** | **0.378** | 0.009 |
| relative | within recording, time-blocked | 0.846 | 0.010 |
| absolute (log) | eyes open → eyes closed | 0.597 | 0.005 |
| absolute (log) | within recording, time-blocked | 0.956 | 0.015 |

![Confusion matrix for the headline model](artifacts/relative_cross_condition_confusion_matrix.png)

### 1. A change of brain state is the dominant effect

Training on eyes-open and testing on eyes-closed is the closest this dataset gets to
the deployment question: a user enrolls alert and authenticates tired. Macro-F1 falls
from 0.846 within a recording to 0.378 across the state change. Per subject, the median
F1 is 0.36 and **14 of 89 subjects score zero** — identity features survive the change
for most people and not at all for some.

### 2. A random split would score higher, and the difference is leakage

Windows overlap by 50%, so a random split puts windows that share samples on both sides
of the train/test boundary. Run deliberately as a labelled demonstration on the
within-recording data, it scores **0.906 against 0.846** for the guarded time-blocked
split. 90% of its test windows share samples with a training window.

The shuffled-label control on that leaky split reads **0.011 — chance**. The standard
leakage control cannot see this leak: each memorized window carries a scrambled label.
Overlap is ruled out structurally instead, by an assertion that runs on every split
([D-002, D-007, D-017](docs/DECISIONS.md)).

### 3. Absolute power scores higher, and that gap is an upper bound

Absolute band power beats relative by **0.219** across the state change and **0.110**
within a recording. With one session per subject, the amplitude information behind that
gap **cannot be separated from session artifacts** — electrode impedance, cap placement,
amplifier gain — **so the gap is an upper bound on their contribution**. Part of it may
be anatomy, such as skull thickness, which is person-specific and would survive a second
session. Relative power is the headline because it discards per-channel amplitude
scaling, whatever its source ([D-004](docs/DECISIONS.md)).

### 4. Eye movement: a small effect, not acted on

Electrodes above the eyes pick up blinks, which are person-specific within a sitting.
Models trained on eyes-open data put **1.23× (relative) and 1.35× (absolute)** the
share of importance on Fp1/Fp2/AF7/AF8 that even spreading would give them. Models
trained on both conditions put **1.05× and 0.88×** there. The contrast runs the way the
hypothesis predicts, but the effect is too small to act on ([D-004b](docs/DECISIONS.md)).

### 5. Muscle activity: a material dependence, bounded rather than decomposed

In every model the most important channels sat over the temporal muscles (T9, T10, T7,
T8) and gamma (30–50 Hz) was the top band — where scalp muscle activity (EMG) shows up.
Two ablations of the headline test how much it depends on those features, judged
against a 0.05 macro-F1 materiality threshold fixed before either ran
([D-018](docs/DECISIONS.md)):

| Headline model | Macro-F1 | Drop | Shuffled-label control |
|---|---|---|---|
| all features | 0.378 | — | 0.009 |
| without the gamma band (relative power renormalized over 1–30 Hz) | 0.263 | **0.116** | 0.009 |
| without the 8 lateral temporal sites | 0.300 | **0.078** | 0.007 |

Both drops are material. **Neither is a measure of EMG.** Gamma at temporal sites
contains both muscle and neural activity, and scalp EEG cannot separate them. So each
drop is an upper bound on what information unique to the removed features — muscle and
neural together — contributes to the headline.

The bounds are also narrower than they look. With gamma removed, beta becomes the top
band and the temporal sites still lead the channel ranking. With the temporal sites
removed, gamma stays the top band, and Iz and frontal channels move to the top of the
channel ranking. Each ablation leaves the other route open, so neither bounds the
model's dependence on temporal-site or high-frequency features as a whole.

### Reproducibility

- Artifacts come from a clean, pushed commit. `scripts/train_baseline.py` refuses to
  write `artifacts/` from uncommitted code, and refuses to run unless the impostor
  holdout file is committed and unmodified.
- Seeds are fixed. Re-running from a later commit reproduced all four baseline models'
  artifacts byte for byte.
- Every artifact records its config fingerprint, split parameters, thresholds, and
  package versions (`artifacts/run_summary.json`). None contains a feature vector.

---

## Getting started

```bash
python -m venv .venv && .venv/Scripts/activate    # Windows
pip install -e ".[dev]"

python -m scripts.verify_dataset --subjects 5      # fetch + validate
pytest                                             # add -m "not slow" without data
python -m scripts.train_baseline                   # needs a clean, committed tree
```

The impostor holdout is already fixed in `config/impostor_holdout.json`;
`scripts/select_holdout.py` refuses to overwrite it. The Postgres schema and Docker
Compose setup are written but not yet exercised.
