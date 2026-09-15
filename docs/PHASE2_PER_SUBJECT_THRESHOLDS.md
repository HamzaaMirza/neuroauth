# Per-subject thresholds: proposal, not implemented

**Status:** proposal for review, 2026-09-15. Nothing is implemented, and no session
parameter depends on it. The numbers come from the committed run `72a1b1e`
(`artifacts/verification/`) and from `per_subject_tail.json`. Everything derived here is
post hoc (D-024).

**The idea.** Calibrate each subject's accept threshold at enrollment instead of using one
global threshold, so that the per-subject tail is handled user by user. For reference, the
median per-subject EER is 10.7% and the worst decile averages 39.5%.

---

## 1. The form it has to take

**Equalize FAR per user, never FRR.** A per-user threshold can be calibrated to one of two
targets.

- **A per-user FRR target.** Every user is rejected equally often. A user whose genuine and
  impostor scores overlap gets a low threshold, so anyone claiming to be them gets in more
  easily. That trades security for usability on exactly the weakest accounts, and hard
  rule 4 forbids it.
- **A per-user FAR target.** Every account is equally hard to impersonate, and a poorly
  separable user pays in rejections instead.

Only the second is admissible. In the terms of Doddington et al. (1998, "Sheep, goats, lambs
and wolves"), it protects the *lambs* (accounts that are easy to imitate) at the expense of
the *goats* (users who are hard to verify).

**Calibrate from impostor scores at enrollment.** For user u:

1. Project a background set of other people's embeddings under u's key.
2. Compare each with u's template.
3. Keep the mean and spread, μ_imp,u and σ_imp,u.

The threshold is then t_u = μ_imp,u + k·σ_imp,u.

**That is one global threshold on a normalized score.**

    s ≥ t_u   ⇔   (s − μ_imp,u) / σ_imp,u ≥ k

This is Z-norm from speaker verification. "Per-subject thresholds" therefore need not add any
per-user session parameter. The session keeps a single global k, and the per-user part lives
inside the score. The natural home for it is an input to the decision layer (D-021). That
makes it a Phase 3 candidate for the promotion gate to judge, not a session parameter to fix
now.

---

## 2. What it could change, bounded by the results in hand

1. **Pooled error: at most 16.5% → 13.2%.** Every subject has the same genuine and impostor
   counts (56 and 1,119). So when each subject sits at their own EER threshold, pooled FAR and
   FRR are both at most the mean per-subject EER, 13.2%.
   - Those thresholds are chosen on the holdout's own scores; enrollment-calibrated
     thresholds will do worse.
   - **The ceiling is 3.3 EER points,** and the realistic gain is smaller.
2. **The worst decile: nothing.** Their 39.5% is already measured at each subject's best
   threshold. Calibration changes how they fail, not whether: under FAR equalization, they
   are rejected most of the time.
3. **The worst of the tail is invisible at enrollment.** 7 of the 9 worst-decile subjects
   verify within eyes-open at a per-subject EER below 10% (post hoc, 13–14 genuine windows
   each). The tail is mostly a failure to survive the brain-state change. A threshold
   calibrated from eyes-open enrollment windows cannot anticipate where a subject's
   eyes-closed scores will fall.
4. **What it genuinely improves: security that is the same for every account.** A global
   threshold gives every account the same threshold, not the same FAR. Per-user FAR at the
   global threshold is not measured today, because score tables are not persisted.
   Equalizing it is a security property, and arguably a stronger reason to do this than the
   EER ceiling.

---

## 3. What it costs

- **Resolution.** Per-user FAR at 1% needs roughly 300 independent impostor units per user
  (rule of three, counted in subjects). A fold's background of about 71 training subjects
  gives 0.7 expected errors.
  - So per-user thresholds at FAR 1% cannot be set empirically, only parametrically, by
    assuming a normal impostor tail (k ≈ 2.33).
  - Hamming similarity on 64 bits is discrete, so that assumption has to be checked, and the
    realized per-user FAR on the holdout reported against the target.
- **The background set.** Calibration needs other people's embeddings at enrollment.
  - The embedding's own training data is consistent with the frozen representation (D-021)
    and stores nothing new. But those subjects were fitted on, so their impostor scores are
    likely better separated than a stranger's. Thresholds would come out too low, and the
    realized FAR would exceed the target.
  - Enrolled users' features as background would mean persisting them, which hard rule 3
    forbids.
- **Retrofit.** Impostor-side statistics need only the template bits, the key (derivable
  from the master secret), and the background set. They need no enrollment windows, so
  existing templates can be calibrated without re-enrolling anyone. Genuine-side statistics
  cannot be retrofitted; the decision layer already has those.
- **Storage and serving.** Two scalars per template plus a background-set version, one extra
  projection of the background set per enrollment (milliseconds), and two more columns in
  migration 002.
- **Evaluation.** A new run, with its judging criteria fixed before it. At minimum:
  - realized per-user FAR on the holdout against the target (for example, the fraction of
    users above twice the target);
  - pooled EER, and FRR at FAR 1%;
  - FRR of the worst decile at the target.

  The values are yours to set; this proposal does not set them.
- **Reporting.** "The system has FAR x" becomes "each account has FAR ≈ x." The pooled DET no
  longer describes a single deployable threshold, so the report needs both views.

---

## 4. What would act on the tail itself

- **Enrollment across brain states.** The tail comes from the state change, so enroll across
  states. On eegmmidb this uses up the eyes-closed run, so probes would move to the task
  runs R03–R14 (the robustness holdout on the cut list). That is a new protocol with newly
  registered criteria, not a tweak.
- **A fallback factor for unsuitable users.** Measure how many users need a second factor and
  report that as a rate. It cannot be triggered from single-state enrollment data alone,
  because it needs the scores from a user's first sessions.

---

## 5. Recommendation

- **Do not fix the session parameters around per-subject thresholds.** Keep one global
  threshold.
- **If calibration is pursued,** implement it as impostor-side Z-norm feeding the decision
  layer and judged by the Phase 3 gate. Expect at most about 3 EER points, and pursue it for
  the equal-security-per-account property rather than for the EER.
- **The tail itself** needs enrollment across brain states, which is a protocol decision to
  take separately.
