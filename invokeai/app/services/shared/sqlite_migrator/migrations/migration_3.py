import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration

from .util.migrate_yaml_config_1 import MigrateModelYamlToDb1


class Migration3Callback:
    def __init__(self) -> None:
        pass

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._drop_model_manager_metadata(cursor)
        self._recreate_model_config(cursor)
        self._migrate_model_config_records(cursor)

    def _drop_model_manager_metadata(self, cursor: sqlite3.Cursor) -> None:
        """Drops the `model_manager_metadata` table."""
        cursor.execute("DROP TABLE IF EXISTS model_manager_metadata;")

    def _recreate_model_config(self, cursor: sqlite3.Cursor) -> None:
        """
        Drops the `model_config` table, recreating it.

        In 3.4.0, this table used explicit columns but was changed to use json_extract 3.5.0.

        Because this table is not used in production, we are able to simply drop it and recreate it.
        """

        cursor.execute("DROP TABLE IF EXISTS model_config;")

        cursor.execute(
            """--sql
            CREATE TABLE IF NOT EXISTS model_config (
                id TEXT NOT NULL PRIMARY KEY,
                -- The next 3 fields are enums in python, unrestricted string here
                base TEXT GENERATED ALWAYS as (json_extract(config, '$.base')) VIRTUAL NOT NULL,
                type TEXT GENERATED ALWAYS as (json_extract(config, '$.type')) VIRTUAL NOT NULL,
                name TEXT GENERATED ALWAYS as (json_extract(config, '$.name')) VIRTUAL NOT NULL,
                path TEXT GENERATED ALWAYS as (json_extract(config, '$.path')) VIRTUAL NOT NULL,
                format TEXT GENERATED ALWAYS as (json_extract(config, '$.format')) VIRTUAL NOT NULL,
                original_hash TEXT, -- could be null
                -- Serialized JSON representation of the whole config object,
                -- which will contain additional fields from subclasses
                config TEXT NOT NULL,
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                -- Updated via trigger
                updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                -- unique constraint on combo of name, base and type
                UNIQUE(name, base, type)
            );
            """
        )

    def _migrate_model_config_records(self, cursor: sqlite3.Cursor) -> None:
        """After updating the model config table, we repopulate it."""
        model_record_migrator = MigrateModelYamlToDb1(cursor)
        model_record_migrator.migrate()


def build_migration_3() -> Migration:
    """
    Build the migration from database version 2 to 3.

    This migration does the following:
    - Drops the `model_config` table, recreating it
    - Migrates data from `models.yaml` into the `model_config` table
    """
    migration_3 = Migration(
        from_version=2,
        to_version=3,
        callback=Migration3Callback(),
    )

    return migration_3
