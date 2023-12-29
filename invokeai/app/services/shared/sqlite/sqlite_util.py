from logging import Logger

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.image_files.image_files_base import ImageFileStorageBase
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_1 import build_migration_1
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2 import build_migration_2
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_3 import build_migration_3
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_impl import SqliteMigrator


def init_db(config: InvokeAIAppConfig, logger: Logger, image_files: ImageFileStorageBase) -> SqliteDatabase:
    """
    Initializes the SQLite database.

    :param config: The app config
    :param logger: The logger
    :param image_files: The image files service (used by migration 2)

    This function:
    - Instantiates a :class:`SqliteDatabase`
    - Instantiates a :class:`SqliteMigrator` and registers all migrations
    - Runs all migrations
    """
    db_path = None if config.use_memory_db else config.db_path
    db = SqliteDatabase(db_path=db_path, logger=logger, verbose=config.log_sql)

    migrator = SqliteMigrator(db=db)
    migrator.register_migration(build_migration_1())
    migrator.register_migration(build_migration_2(image_files=image_files, logger=logger))
    migrator.register_migration(build_migration_3())
    migrator.run_migrations()

    return db
