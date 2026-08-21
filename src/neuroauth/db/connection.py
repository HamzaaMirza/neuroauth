"""Postgres connection handling.

psycopg 3 rather than psycopg2: v2 is maintenance-only, and v3 ships real type hints
plus native row factories. The [binary] extra avoids needing libpq on Windows.
"""

from collections.abc import Iterator
from contextlib import AbstractContextManager

import psycopg
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database configuration, read from the environment or a local .env.

    Validated at startup so a misconfiguration surfaces as a startup error rather
    than as a None three call frames later. The URL is a secret: it is never logged,
    never echoed, and never written into an event payload.

    Attributes:
        database_url: Full libpq connection URI.
        connect_timeout_s: Seconds before a connection attempt gives up.
    """

    model_config = SettingsConfigDict(env_file=".env", env_prefix="NEUROAUTH_")

    database_url: str
    connect_timeout_s: int = 10


def connect(settings: DatabaseSettings) -> AbstractContextManager[psycopg.Connection]:
    """Open a connection as a context manager.

    Phase 1 opens a connection per script run. A pool would be infrastructure the
    scale does not justify (hard rule 7); it arrives with FastAPI in Phase 2, where
    concurrent requests actually exist.

    Args:
        settings: Validated database settings.

    Returns:
        A context manager yielding an open connection, committed on clean exit and
        rolled back on exception.
    """
    raise NotImplementedError("TODO(phase-1): connection context manager")


def healthcheck(settings: DatabaseSettings) -> bool:
    """Return whether the database answers a trivial query.

    Args:
        settings: Validated database settings.

    Returns:
        True on success. False on any connection or query failure -- this is used by
        Docker Compose and by scripts deciding whether to wait, so it reports rather
        than raises.
    """
    raise NotImplementedError("TODO(phase-1): healthcheck")


def iter_migrations(migrations_dir: str) -> Iterator[tuple[str, str]]:
    """Yield (version, sql) for each migration file, in version order.

    Args:
        migrations_dir: Directory holding NNN_name.sql files.

    Yields:
        (version, sql_text), sorted ascending by the numeric prefix.
    """
    raise NotImplementedError("TODO(phase-1): migration discovery")
