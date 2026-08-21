-- Phase 1 core schema: subjects, sessions, events.
-- Templates arrive in Phase 2, after the cancelable transform exists.
--
-- Hard rule 3: no raw EEG and no raw feature vector is ever written here. There is
-- deliberately no column in this schema capable of holding one.

CREATE EXTENSION IF NOT EXISTS pgcrypto;   -- gen_random_uuid()

CREATE TABLE schema_migrations (
    version     text        PRIMARY KEY,
    applied_at  timestamptz NOT NULL DEFAULT now()
);

-- ------------------------------------------------------------------- subjects

CREATE TABLE subjects (
    id            uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    external_ref  text        NOT NULL UNIQUE,   -- 'eegmmidb-S001'
    display_name  text,

    -- Cohort is set once, at ingest, by a seeded deterministic selection made
    -- before any Phase 1 result has been looked at (see neuroauth.cohorts).
    -- Choosing the holdout after seeing results would be selection bias and would
    -- compromise the Phase 2 EER before Phase 2 begins.
    cohort        text        NOT NULL
                  CHECK (cohort IN ('enrollable', 'impostor_holdout')),

    status        text        NOT NULL DEFAULT 'pending'
                  CHECK (status IN ('pending', 'enrolled', 'revoked')),

    created_at    timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now(),

    -- Hard rule 1, as a database invariant rather than a convention: an
    -- impostor-holdout subject can never reach 'enrolled'.
    CONSTRAINT holdout_never_enrolled
        CHECK (cohort <> 'impostor_holdout' OR status <> 'enrolled')
);

CREATE INDEX subjects_cohort_status_idx ON subjects (cohort, status);

-- ------------------------------------------------------------------- sessions

CREATE TABLE sessions (
    id                 uuid        PRIMARY KEY DEFAULT gen_random_uuid(),

    -- The CLAIMED identity. Verification decides whether the claim holds; this
    -- column is never evidence of who is actually at the electrodes.
    subject_id         uuid        NOT NULL REFERENCES subjects (id)
                                   ON DELETE RESTRICT,

    state              text        NOT NULL DEFAULT 'active'
                       CHECK (state IN ('active', 'challenged', 'revoked',
                                        'expired', 'closed')),

    -- Provenance, captured at session start and immutable thereafter.
    model_version      text        NOT NULL,
    config_fingerprint text        NOT NULL,   -- PipelineConfig.fingerprint()
    threshold_config   jsonb       NOT NULL,   -- thresholds in effect

    -- Honesty about simulation. A replayed stream is labelled as one in the
    -- database, not only in the README.
    stream_source      text        NOT NULL DEFAULT 'replay'
                       CHECK (stream_source IN ('replay', 'live')),

    started_at         timestamptz NOT NULL DEFAULT now(),
    ended_at           timestamptz,

    CONSTRAINT ended_after_started
        CHECK (ended_at IS NULL OR ended_at >= started_at),
    CONSTRAINT terminal_state_has_end
        CHECK (state IN ('active', 'challenged') OR ended_at IS NOT NULL)
);

CREATE INDEX sessions_subject_started_idx ON sessions (subject_id, started_at DESC);
CREATE INDEX sessions_state_idx ON sessions (state)
    WHERE state IN ('active', 'challenged');

-- --------------------------------------------------------------------- events

-- Append-only audit log. Every session transition, gate decision, and rejection
-- review lands here with full provenance. No UPDATE, no DELETE.
CREATE TABLE events (
    id                 bigint      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    -- Both nullable: system-level events belong to no session, and some events
    -- (dataset ingest, model promotion) belong to no subject.
    session_id         uuid        REFERENCES sessions (id) ON DELETE RESTRICT,
    subject_id         uuid        REFERENCES subjects (id) ON DELETE RESTRICT,

    -- text + CHECK rather than a Postgres ENUM: widening the set is a one-line
    -- constraint swap instead of an ALTER TYPE migration, and the set grows in
    -- every remaining phase.
    event_type         text        NOT NULL
                       CHECK (event_type IN (
                           'subject_ingested',
                           'cohort_assigned',
                           'session_started',
                           'session_state_changed',
                           'session_ended',
                           'pipeline_run',
                           'evaluation_completed'
                       )),

    occurred_at        timestamptz NOT NULL DEFAULT now(),

    -- Who or what caused this: 'system', 'admin:<uuid>', 'script:<name>'.
    actor              text        NOT NULL,

    model_version      text,
    config_fingerprint text,

    -- Scores, state transitions, gate rationale. NEVER a feature vector and never a
    -- raw sample -- enforced by a code-level denylist and a test, and reviewed on
    -- every schema change.
    payload            jsonb       NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX events_session_time_idx ON events (session_id, occurred_at DESC);
CREATE INDEX events_type_time_idx    ON events (event_type, occurred_at DESC);
CREATE INDEX events_subject_time_idx ON events (subject_id, occurred_at DESC);

REVOKE UPDATE, DELETE ON events FROM PUBLIC;
