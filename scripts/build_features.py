"""Extract features for the enrollable cohort and cache them to disk.

Optional convenience: train_baseline.py can extract on the fly, but a cached .npz
turns a five-minute rerun into a five-second one while iterating on the model.

The cache holds feature values, not raw EEG and not raw windows, and it lives in a
gitignored directory. It is a local scratch artifact, not persistence in the sense
of hard rule 3 -- nothing here reaches Postgres, S3, or a log line.

Usage:
    python -m scripts.build_features [--out .cache/features.npz]
"""


def main() -> int:
    """Entry point. Returns a process exit code."""
    raise NotImplementedError("TODO(phase-1): feature cache builder")


if __name__ == "__main__":
    raise SystemExit(main())
