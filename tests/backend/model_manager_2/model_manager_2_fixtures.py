# Fixtures to support testing of the model_manager v2 installer, metadata and record store

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

import pytest
from pydantic import BaseModel
from requests.sessions import Session
from requests_testadapter import TestAdapter, TestSession

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.download import DownloadQueueService
from invokeai.app.services.events.events_base import EventServiceBase
from invokeai.app.services.model_install import ModelInstallService, ModelInstallServiceBase
from invokeai.app.services.model_records import ModelRecordServiceSQL
from invokeai.backend.model_manager.config import (
    BaseModelType,
    ModelFormat,
    ModelType,
)
from invokeai.backend.model_manager.metadata import ModelMetadataStore
from invokeai.backend.util.logging import InvokeAILogger
from tests.backend.model_manager_2.model_metadata.metadata_examples import (
    RepoCivitaiModelMetadata1,
    RepoCivitaiVersionMetadata1,
    RepoHFMetadata1,
    RepoHFMetadata1_nofp16,
    RepoHFModelJson1,
)
from tests.fixtures.sqlite_database import create_mock_sqlite_database


class DummyEvent(BaseModel):
    """Dummy Event to use with Dummy Event service."""

    event_name: str
    payload: Dict[str, Any]


class DummyEventService(EventServiceBase):
    """Dummy event service for testing."""

    events: List[DummyEvent]

    def __init__(self) -> None:
        super().__init__()
        self.events = []

    def dispatch(self, event_name: str, payload: Any) -> None:
        """Dispatch an event by appending it to self.events."""
        self.events.append(DummyEvent(event_name=payload["event"], payload=payload["data"]))


# Create a temporary directory using the contents of `./data/invokeai_root` as the template
@pytest.fixture
def mm2_root_dir(tmp_path_factory) -> Path:
    root_template = Path(__file__).resolve().parent / "data" / "invokeai_root"
    temp_dir: Path = tmp_path_factory.mktemp("data") / "invokeai_root"
    shutil.copytree(root_template, temp_dir)
    return temp_dir


@pytest.fixture
def mm2_model_files(tmp_path_factory) -> Path:
    root_template = Path(__file__).resolve().parent / "data" / "test_files"
    temp_dir: Path = tmp_path_factory.mktemp("data") / "test_files"
    shutil.copytree(root_template, temp_dir)
    return temp_dir


@pytest.fixture
def embedding_file(mm2_model_files: Path) -> Path:
    return mm2_model_files / "test_embedding.safetensors"


@pytest.fixture
def diffusers_dir(mm2_model_files: Path) -> Path:
    return mm2_model_files / "test-diffusers-main"


@pytest.fixture
def mm2_app_config(mm2_root_dir: Path) -> InvokeAIAppConfig:
    app_config = InvokeAIAppConfig(
        root=mm2_root_dir,
        models_dir=mm2_root_dir / "models",
    )
    return app_config


@pytest.fixture
def mm2_record_store(mm2_app_config: InvokeAIAppConfig) -> ModelRecordServiceSQL:
    logger = InvokeAILogger.get_logger(config=mm2_app_config)
    db = create_mock_sqlite_database(mm2_app_config, logger)
    store = ModelRecordServiceSQL(db)
    # add five simple config records to the database
    raw1 = {
        "path": "/tmp/foo1",
        "format": ModelFormat("diffusers"),
        "name": "test2",
        "base": BaseModelType("sd-2"),
        "type": ModelType("vae"),
        "original_hash": "111222333444",
        "source": "stabilityai/sdxl-vae",
    }
    raw2 = {
        "path": "/tmp/foo2.ckpt",
        "name": "model1",
        "format": ModelFormat("checkpoint"),
        "base": BaseModelType("sd-1"),
        "type": "main",
        "config": "/tmp/foo.yaml",
        "variant": "normal",
        "original_hash": "111222333444",
        "source": "https://civitai.com/models/206883/split",
    }
    raw3 = {
        "path": "/tmp/foo3",
        "format": ModelFormat("diffusers"),
        "name": "test3",
        "base": BaseModelType("sdxl"),
        "type": ModelType("main"),
        "original_hash": "111222333444",
        "source": "author3/model3",
        "description": "This is test 3",
    }
    raw4 = {
        "path": "/tmp/foo4",
        "format": ModelFormat("diffusers"),
        "name": "test4",
        "base": BaseModelType("sdxl"),
        "type": ModelType("lora"),
        "original_hash": "111222333444",
        "source": "author4/model4",
    }
    raw5 = {
        "path": "/tmp/foo5",
        "format": ModelFormat("diffusers"),
        "name": "test5",
        "base": BaseModelType("sd-1"),
        "type": ModelType("lora"),
        "original_hash": "111222333444",
        "source": "author4/model5",
    }
    store.add_model("test_config_1", raw1)
    store.add_model("test_config_2", raw2)
    store.add_model("test_config_3", raw3)
    store.add_model("test_config_4", raw4)
    store.add_model("test_config_5", raw5)
    return store


@pytest.fixture
def mm2_metadata_store(mm2_record_store: ModelRecordServiceSQL) -> ModelMetadataStore:
    db = mm2_record_store._db  # to ensure we are sharing the same database
    return ModelMetadataStore(db)


@pytest.fixture
def mm2_session(embedding_file: Path, diffusers_dir: Path) -> Session:
    """This fixtures defines a series of mock URLs for testing download and installation."""
    sess = TestSession()
    sess.mount(
        "https://test.com/missing_model.safetensors",
        TestAdapter(
            b"missing",
            status=404,
        ),
    )
    sess.mount(
        "https://huggingface.co/api/models/stabilityai/sdxl-turbo",
        TestAdapter(
            RepoHFMetadata1,
            headers={"Content-Type": "application/json; charset=utf-8", "Content-Length": len(RepoHFMetadata1)},
        ),
    )
    sess.mount(
        "https://huggingface.co/api/models/stabilityai/sdxl-turbo-nofp16",
        TestAdapter(
            RepoHFMetadata1_nofp16,
            headers={"Content-Type": "application/json; charset=utf-8", "Content-Length": len(RepoHFMetadata1_nofp16)},
        ),
    )
    sess.mount(
        "https://civitai.com/api/v1/model-versions/242807",
        TestAdapter(
            RepoCivitaiVersionMetadata1,
            headers={
                "Content-Length": len(RepoCivitaiVersionMetadata1),
            },
        ),
    )
    sess.mount(
        "https://civitai.com/api/v1/models/215485",
        TestAdapter(
            RepoCivitaiModelMetadata1,
            headers={
                "Content-Length": len(RepoCivitaiModelMetadata1),
            },
        ),
    )
    sess.mount(
        "https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/model_index.json",
        TestAdapter(
            RepoHFModelJson1,
            headers={
                "Content-Length": len(RepoHFModelJson1),
            },
        ),
    )
    with open(embedding_file, "rb") as f:
        data = f.read()  # file is small - just 15K
    sess.mount(
        "https://www.test.foo/download/test_embedding.safetensors",
        TestAdapter(data, headers={"Content-Type": "application/octet-stream", "Content-Length": len(data)}),
    )
    sess.mount(
        "https://huggingface.co/api/models/stabilityai/sdxl-turbo",
        TestAdapter(
            RepoHFMetadata1,
            headers={"Content-Type": "application/json; charset=utf-8", "Content-Length": len(RepoHFMetadata1)},
        ),
    )
    for root, _, files in os.walk(diffusers_dir):
        for name in files:
            path = Path(root, name)
            url_base = path.relative_to(diffusers_dir).as_posix()
            url = f"https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/{url_base}"
            with open(path, "rb") as f:
                data = f.read()
            sess.mount(
                url,
                TestAdapter(
                    data,
                    headers={
                        "Content-Type": "application/json; charset=utf-8",
                        "Content-Length": len(data),
                    },
                ),
            )
    return sess


@pytest.fixture
def mm2_installer(mm2_app_config: InvokeAIAppConfig, mm2_session: Session) -> ModelInstallServiceBase:
    logger = InvokeAILogger.get_logger()
    db = create_mock_sqlite_database(mm2_app_config, logger)
    events = DummyEventService()
    store = ModelRecordServiceSQL(db)
    metadata_store = ModelMetadataStore(db)

    download_queue = DownloadQueueService(requests_session=mm2_session)
    download_queue.start()

    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=store,
        download_queue=download_queue,
        metadata_store=metadata_store,
        event_bus=events,
        session=mm2_session,
    )
    installer.start()
    return installer
