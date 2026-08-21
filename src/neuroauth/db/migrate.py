"""Migration runner.

Plain numbered .sql files applied in order, tracked in schema_migrations. No Alembic
in Phase 1: there are no ORM models to autogenerate against, and hard rule 7 says
not to add infrastructure the scale does not justify. Revisit when SQLAlchemy models
arrive in Phase 2 -- that is the point where Alembic starts paying for itself.
"""

from pathlib import Path

from neuroauth.db.connection import DatabaseSettings


def applied_versions(settings: DatabaseSettings) -> frozenset[str]:
    """Return the migration versions already recorded in schema_migrations.

    Args:
        settings: Validated database settings.

    Returns:
        Applied version strings. Empty when the table does not yet exist.
    """
    raise NotImplementedError("TODO(phase-1): read applied versions")


def migrate(settings: DatabaseSettings, migrations_dir: Path) -> list[str]:
    """Apply every pending migration, each in its own transaction.

    Args:
        settings: Validated database settings.
        migrations_dir: Directory holding NNN_name.sql files.

    Returns:
        Versions applied by this call, in order. Empty when already current.

    Raises:
        RuntimeError: If a migration fails. The failing migration is rolled back;
            earlier ones in this call stay applied and recorded.
    """
    raise NotImplementedError("TODO(phase-1): apply pending migrations")
