"""Model installation class."""

import os
import re
import threading
import time
from hashlib import sha256
from pathlib import Path
from queue import Empty, Queue
from random import randbytes
from shutil import copyfile, copytree, move, rmtree
from tempfile import mkdtemp
from typing import Any, Dict, List, Optional, Set, Union

from huggingface_hub import HfFolder
from pydantic.networks import AnyHttpUrl
from requests import Session

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.download import DownloadJob, DownloadQueueServiceBase
from invokeai.app.services.events.events_base import EventServiceBase
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.model_records import DuplicateModelException, ModelRecordServiceBase, ModelRecordServiceSQL
from invokeai.backend.model_manager.config import (
    AnyModelConfig,
    BaseModelType,
    InvalidModelConfigException,
    ModelRepoVariant,
    ModelType,
)
from invokeai.backend.model_manager.hash import FastModelHash
from invokeai.backend.model_manager.metadata import (
    AnyModelRepoMetadata,
    CivitaiMetadataFetch,
    HuggingFaceMetadataFetch,
    ModelMetadataStore,
    ModelMetadataWithFiles,
    RemoteModelFile,
)
from invokeai.backend.model_manager.probe import ModelProbe
from invokeai.backend.model_manager.search import ModelSearch
from invokeai.backend.util import Chdir, InvokeAILogger
from invokeai.backend.util.devices import choose_precision, choose_torch_device

from .model_install_base import (
    CivitaiModelSource,
    HFModelSource,
    InstallStatus,
    LocalModelSource,
    ModelInstallJob,
    ModelInstallServiceBase,
    ModelSource,
    URLModelSource,
)

TMPDIR_PREFIX = "tmpinstall_"


class ModelInstallService(ModelInstallServiceBase):
    """class for InvokeAI model installation."""

    def __init__(
        self,
        app_config: InvokeAIAppConfig,
        record_store: ModelRecordServiceBase,
        download_queue: DownloadQueueServiceBase,
        metadata_store: Optional[ModelMetadataStore] = None,
        event_bus: Optional[EventServiceBase] = None,
        session: Optional[Session] = None,
    ):
        """
        Initialize the installer object.

        :param app_config: InvokeAIAppConfig object
        :param record_store: Previously-opened ModelRecordService database
        :param event_bus: Optional EventService object
        """
        self._app_config = app_config
        self._record_store = record_store
        self._event_bus = event_bus
        self._logger = InvokeAILogger.get_logger(name=self.__class__.__name__)
        self._install_jobs: List[ModelInstallJob] = []
        self._install_queue: Queue[ModelInstallJob] = Queue()
        self._cached_model_paths: Set[Path] = set()
        self._models_installed: Set[str] = set()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._downloads_changed_event = threading.Event()
        self._download_queue = download_queue
        self._download_cache: Dict[AnyHttpUrl, ModelInstallJob] = {}
        self._running = False
        self._session = session
        self._next_job_id = 0
        # There may not necessarily be a metadata store initialized
        # so we create one and initialize it with the same sql database
        # used by the record store service.
        if metadata_store:
            self._metadata_store = metadata_store
        else:
            assert isinstance(record_store, ModelRecordServiceSQL)
            self._metadata_store = ModelMetadataStore(record_store.db)

    @property
    def app_config(self) -> InvokeAIAppConfig:  # noqa D102
        return self._app_config

    @property
    def record_store(self) -> ModelRecordServiceBase:  # noqa D102
        return self._record_store

    @property
    def event_bus(self) -> Optional[EventServiceBase]:  # noqa D102
        return self._event_bus

    # make the invoker optional here because we don't need it and it
    # makes the installer harder to use outside the web app
    def start(self, invoker: Optional[Invoker] = None) -> None:
        """Start the installer thread."""
        with self._lock:
            if self._running:
                raise Exception("Attempt to start the installer service twice")
            self._start_installer_thread()
            self._remove_dangling_install_dirs()
            self.sync_to_config()

    def stop(self, invoker: Optional[Invoker] = None) -> None:
        """Stop the installer thread; after this the object can be deleted and garbage collected."""
        with self._lock:
            if not self._running:
                raise Exception("Attempt to stop the install service before it was started")
            self._stop_event.set()
            with self._install_queue.mutex:
                self._install_queue.queue.clear()  # get rid of pending jobs
            active_jobs = [x for x in self.list_jobs() if x.running]
            if active_jobs:
                self._logger.warning("Waiting for active install job to complete")
            self.wait_for_installs()
            self._download_cache.clear()
            self._running = False

    def register_path(
        self,
        model_path: Union[Path, str],
        config: Optional[Dict[str, Any]] = None,
    ) -> str:  # noqa D102
        model_path = Path(model_path)
        config = config or {}
        if config.get("source") is None:
            config["source"] = model_path.resolve().as_posix()
        return self._register(model_path, config)

    def install_path(
        self,
        model_path: Union[Path, str],
        config: Optional[Dict[str, Any]] = None,
    ) -> str:  # noqa D102
        model_path = Path(model_path)
        config = config or {}
        if config.get("source") is None:
            config["source"] = model_path.resolve().as_posix()

        info: AnyModelConfig = self._probe_model(Path(model_path), config)
        old_hash = info.original_hash
        dest_path = self.app_config.models_path / info.base.value / info.type.value / model_path.name
        try:
            new_path = self._copy_model(model_path, dest_path)
        except FileExistsError as excp:
            raise DuplicateModelException(
                f"A model named {model_path.name} is already installed at {dest_path.as_posix()}"
            ) from excp
        new_hash = FastModelHash.hash(new_path)
        assert new_hash == old_hash, f"{model_path}: Model hash changed during installation, possibly corrupted."

        return self._register(
            new_path,
            config,
            info,
        )

    def import_model(self, source: ModelSource, config: Optional[Dict[str, Any]] = None) -> ModelInstallJob:  # noqa D102
        if isinstance(source, LocalModelSource):
            install_job = self._import_local_model(source, config)
            self._install_queue.put(install_job)  # synchronously install
        elif isinstance(source, CivitaiModelSource):
            install_job = self._import_from_civitai(source, config)
        elif isinstance(source, HFModelSource):
            install_job = self._import_from_hf(source, config)
        elif isinstance(source, URLModelSource):
            install_job = self._import_from_url(source, config)
        else:
            raise ValueError(f"Unsupported model source: '{type(source)}'")

        self._install_jobs.append(install_job)
        return install_job

    def list_jobs(self) -> List[ModelInstallJob]:  # noqa D102
        return self._install_jobs

    def get_job_by_source(self, source: ModelSource) -> List[ModelInstallJob]:  # noqa D102
        return [x for x in self._install_jobs if x.source == source]

    def get_job_by_id(self, id: int) -> ModelInstallJob:  # noqa D102
        jobs = [x for x in self._install_jobs if x.id == id]
        if not jobs:
            raise ValueError(f"No job with id {id} known")
        assert len(jobs) == 1
        assert isinstance(jobs[0], ModelInstallJob)
        return jobs[0]

    def wait_for_installs(self, timeout: int = 0) -> List[ModelInstallJob]:  # noqa D102
        """Block until all installation jobs are done."""
        start = time.time()
        while len(self._download_cache) > 0:
            if self._downloads_changed_event.wait(timeout=5):  # in case we miss an event
                self._downloads_changed_event.clear()
            if timeout > 0 and time.time() - start > timeout:
                raise Exception("Timeout exceeded")
        self._install_queue.join()
        return self._install_jobs

    def cancel_job(self, job: ModelInstallJob) -> None:
        """Cancel the indicated job."""
        job.cancel()
        with self._lock:
            self._cancel_download_parts(job)

    def prune_jobs(self) -> None:
        """Prune all completed and errored jobs."""
        unfinished_jobs = [x for x in self._install_jobs if not x.in_terminal_state]
        self._install_jobs = unfinished_jobs

    def sync_to_config(self) -> None:
        """Synchronize models on disk to those in the config record store database."""
        self._scan_models_directory()
        if autoimport := self._app_config.autoimport_dir:
            self._logger.info("Scanning autoimport directory for new models")
            installed = self.scan_directory(self._app_config.root_path / autoimport)
            self._logger.info(f"{len(installed)} new models registered")
        self._logger.info("Model installer (re)initialized")

    def scan_directory(self, scan_dir: Path, install: bool = False) -> List[str]:  # noqa D102
        self._cached_model_paths = {Path(x.path) for x in self.record_store.all_models()}
        callback = self._scan_install if install else self._scan_register
        search = ModelSearch(on_model_found=callback)
        self._models_installed.clear()
        search.search(scan_dir)
        return list(self._models_installed)

    def unregister(self, key: str) -> None:  # noqa D102
        self.record_store.del_model(key)

    def delete(self, key: str) -> None:  # noqa D102
        """Unregister the model. Delete its files only if they are within our models directory."""
        model = self.record_store.get_model(key)
        models_dir = self.app_config.models_path
        model_path = models_dir / model.path
        if model_path.is_relative_to(models_dir):
            self.unconditionally_delete(key)
        else:
            self.unregister(key)

    def unconditionally_delete(self, key: str) -> None:  # noqa D102
        model = self.record_store.get_model(key)
        path = self.app_config.models_path / model.path
        if path.is_dir():
            rmtree(path)
        else:
            path.unlink()
        self.unregister(key)

    # --------------------------------------------------------------------------------------------
    # Internal functions that manage the installer threads
    # --------------------------------------------------------------------------------------------
    def _start_installer_thread(self) -> None:
        threading.Thread(target=self._install_next_item, daemon=True).start()
        self._running = True

    def _install_next_item(self) -> None:
        done = False
        while not done:
            if self._stop_event.is_set():
                done = True
                continue
            try:
                job = self._install_queue.get(timeout=1)
            except Empty:
                continue

            assert job.local_path is not None
            try:
                if job.cancelled:
                    self._signal_job_cancelled(job)

                elif job.errored:
                    self._signal_job_errored(job)

                elif (
                    job.waiting or job.downloading
                ):  # local jobs will be in waiting state, remote jobs will be downloading state
                    job.total_bytes = self._stat_size(job.local_path)
                    job.bytes = job.total_bytes
                    self._signal_job_running(job)
                    if job.inplace:
                        key = self.register_path(job.local_path, job.config_in)
                    else:
                        key = self.install_path(job.local_path, job.config_in)
                    job.config_out = self.record_store.get_model(key)

                    # enter the metadata, if there is any
                    if job.source_metadata:
                        self._metadata_store.add_metadata(key, job.source_metadata)
                    self._signal_job_completed(job)

            except InvalidModelConfigException as excp:
                if any(x.content_type is not None and "text/html" in x.content_type for x in job.download_parts):
                    job.set_error(
                        InvalidModelConfigException(
                            f"At least one file in {job.local_path} is an HTML page, not a model. This can happen when an access token is required to download."
                        )
                    )
                else:
                    job.set_error(excp)
                self._signal_job_errored(job)

            except (OSError, DuplicateModelException) as excp:
                job.set_error(excp)
                self._signal_job_errored(job)

            finally:
                # if this is an install of a remote file, then clean up the temporary directory
                if job._install_tmpdir is not None:
                    rmtree(job._install_tmpdir)
                self._install_queue.task_done()

        self._logger.info("Install thread exiting")

    # --------------------------------------------------------------------------------------------
    # Internal functions that manage the models directory
    # --------------------------------------------------------------------------------------------
    def _remove_dangling_install_dirs(self) -> None:
        """Remove leftover tmpdirs from aborted installs."""
        path = self._app_config.models_path
        for tmpdir in path.glob(f"{TMPDIR_PREFIX}*"):
            self._logger.info(f"Removing dangling temporary directory {tmpdir}")
            rmtree(tmpdir)

    def _scan_models_directory(self) -> None:
        """
        Scan the models directory for new and missing models.

        New models will be added to the storage backend. Missing models
        will be deleted.
        """
        defunct_models = set()
        installed = set()

        with Chdir(self._app_config.models_path):
            self._logger.info("Checking for models that have been moved or deleted from disk")
            for model_config in self.record_store.all_models():
                path = Path(model_config.path)
                if not path.exists():
                    self._logger.info(f"{model_config.name}: path {path.as_posix()} no longer exists. Unregistering")
                    defunct_models.add(model_config.key)
            for key in defunct_models:
                self.unregister(key)

            self._logger.info(f"Scanning {self._app_config.models_path} for new and orphaned models")
            for cur_base_model in BaseModelType:
                for cur_model_type in ModelType:
                    models_dir = Path(cur_base_model.value, cur_model_type.value)
                    installed.update(self.scan_directory(models_dir))
            self._logger.info(f"{len(installed)} new models registered; {len(defunct_models)} unregistered")

    def _sync_model_path(self, key: str, ignore_hash_change: bool = False) -> AnyModelConfig:
        """
        Move model into the location indicated by its basetype, type and name.

        Call this after updating a model's attributes in order to move
        the model's path into the location indicated by its basetype, type and
        name. Applies only to models whose paths are within the root `models_dir`
        directory.

        May raise an UnknownModelException.
        """
        model = self.record_store.get_model(key)
        old_path = Path(model.path)
        models_dir = self.app_config.models_path

        if not old_path.is_relative_to(models_dir):
            return model

        new_path = models_dir / model.base.value / model.type.value / model.name
        self._logger.info(f"Moving {model.name} to {new_path}.")
        new_path = self._move_model(old_path, new_path)
        new_hash = FastModelHash.hash(new_path)
        model.path = new_path.relative_to(models_dir).as_posix()
        if model.current_hash != new_hash:
            assert (
                ignore_hash_change
            ), f"{model.name}: Model hash changed during installation, model is possibly corrupted"
            model.current_hash = new_hash
            self._logger.info(f"Model has new hash {model.current_hash}, but will continue to be identified by {key}")
        self.record_store.update_model(key, model)
        return model

    def _scan_register(self, model: Path) -> bool:
        if model in self._cached_model_paths:
            return True
        try:
            id = self.register_path(model)
            self._sync_model_path(id)  # possibly move it to right place in `models`
            self._logger.info(f"Registered {model.name} with id {id}")
            self._models_installed.add(id)
        except DuplicateModelException:
            pass
        return True

    def _scan_install(self, model: Path) -> bool:
        if model in self._cached_model_paths:
            return True
        try:
            id = self.install_path(model)
            self._logger.info(f"Installed {model} with id {id}")
            self._models_installed.add(id)
        except DuplicateModelException:
            pass
        return True

    def _copy_model(self, old_path: Path, new_path: Path) -> Path:
        if old_path == new_path:
            return old_path
        new_path.parent.mkdir(parents=True, exist_ok=True)
        if old_path.is_dir():
            copytree(old_path, new_path)
        else:
            copyfile(old_path, new_path)
        return new_path

    def _move_model(self, old_path: Path, new_path: Path) -> Path:
        if old_path == new_path:
            return old_path

        new_path.parent.mkdir(parents=True, exist_ok=True)

        # if path already exists then we jigger the name to make it unique
        counter: int = 1
        while new_path.exists():
            path = new_path.with_stem(new_path.stem + f"_{counter:02d}")
            if not path.exists():
                new_path = path
            counter += 1
        move(old_path, new_path)
        return new_path

    def _probe_model(self, model_path: Path, config: Optional[Dict[str, Any]] = None) -> AnyModelConfig:
        info: AnyModelConfig = ModelProbe.probe(Path(model_path))
        if config:  # used to override probe fields
            for key, value in config.items():
                setattr(info, key, value)
        return info

    def _create_key(self) -> str:
        return sha256(randbytes(100)).hexdigest()[0:32]

    def _register(
        self, model_path: Path, config: Optional[Dict[str, Any]] = None, info: Optional[AnyModelConfig] = None
    ) -> str:
        info = info or ModelProbe.probe(model_path, config)
        key = self._create_key()

        model_path = model_path.absolute()
        if model_path.is_relative_to(self.app_config.models_path):
            model_path = model_path.relative_to(self.app_config.models_path)

        info.path = model_path.as_posix()

        # add 'main' specific fields
        if hasattr(info, "config"):
            # make config relative to our root
            legacy_conf = (self.app_config.root_dir / self.app_config.legacy_conf_dir / info.config).resolve()
            info.config = legacy_conf.relative_to(self.app_config.root_dir).as_posix()
        self.record_store.add_model(key, info)
        return key

    def _next_id(self) -> int:
        with self._lock:
            id = self._next_job_id
            self._next_job_id += 1
        return id

    @staticmethod
    def _guess_variant() -> ModelRepoVariant:
        """Guess the best HuggingFace variant type to download."""
        precision = choose_precision(choose_torch_device())
        return ModelRepoVariant.FP16 if precision == "float16" else ModelRepoVariant.DEFAULT

    def _import_local_model(self, source: LocalModelSource, config: Optional[Dict[str, Any]]) -> ModelInstallJob:
        return ModelInstallJob(
            id=self._next_id(),
            source=source,
            config_in=config or {},
            local_path=Path(source.path),
            inplace=source.inplace,
        )

    def _import_from_civitai(self, source: CivitaiModelSource, config: Optional[Dict[str, Any]]) -> ModelInstallJob:
        if not source.access_token:
            self._logger.info("No Civitai access token provided; some models may not be downloadable.")
        metadata = CivitaiMetadataFetch(self._session).from_id(str(source.version_id))
        assert isinstance(metadata, ModelMetadataWithFiles)
        remote_files = metadata.download_urls(session=self._session)
        return self._import_remote_model(source=source, config=config, metadata=metadata, remote_files=remote_files)

    def _import_from_hf(self, source: HFModelSource, config: Optional[Dict[str, Any]]) -> ModelInstallJob:
        # Add user's cached access token to HuggingFace requests
        source.access_token = source.access_token or HfFolder.get_token()
        if not source.access_token:
            self._logger.info("No HuggingFace access token present; some models may not be downloadable.")

        metadata = HuggingFaceMetadataFetch(self._session).from_id(source.repo_id)
        assert isinstance(metadata, ModelMetadataWithFiles)
        remote_files = metadata.download_urls(
            variant=source.variant or self._guess_variant(),
            subfolder=source.subfolder,
            session=self._session,
        )

        return self._import_remote_model(
            source=source,
            config=config,
            remote_files=remote_files,
            metadata=metadata,
        )

    def _import_from_url(self, source: URLModelSource, config: Optional[Dict[str, Any]]) -> ModelInstallJob:
        # URLs from Civitai or HuggingFace will be handled specially
        url_patterns = {
            r"^https?://civitai.com/": CivitaiMetadataFetch,
            r"^https?://huggingface.co/[^/]+/[^/]+$": HuggingFaceMetadataFetch,
        }
        metadata = None
        for pattern, fetcher in url_patterns.items():
            if re.match(pattern, str(source.url), re.IGNORECASE):
                metadata = fetcher(self._session).from_url(source.url)
                break
        self._logger.debug(f"metadata={metadata}")
        if metadata and isinstance(metadata, ModelMetadataWithFiles):
            remote_files = metadata.download_urls(session=self._session)
        else:
            remote_files = [RemoteModelFile(url=source.url, path=Path("."), size=0)]
        return self._import_remote_model(
            source=source,
            config=config,
            metadata=metadata,
            remote_files=remote_files,
        )

    def _import_remote_model(
        self,
        source: ModelSource,
        remote_files: List[RemoteModelFile],
        metadata: Optional[AnyModelRepoMetadata],
        config: Optional[Dict[str, Any]],
    ) -> ModelInstallJob:
        # TODO: Replace with tempfile.tmpdir() when multithreading is cleaned up.
        # Currently the tmpdir isn't automatically removed at exit because it is
        # being held in a daemon thread.
        tmpdir = Path(
            mkdtemp(
                dir=self._app_config.models_path,
                prefix=TMPDIR_PREFIX,
            )
        )
        install_job = ModelInstallJob(
            id=self._next_id(),
            source=source,
            config_in=config or {},
            source_metadata=metadata,
            local_path=tmpdir,  # local path may change once the download has started due to content-disposition handling
            bytes=0,
            total_bytes=0,
        )
        # we remember the path up to the top of the tmpdir so that it may be
        # removed safely at the end of the install process.
        install_job._install_tmpdir = tmpdir
        assert install_job.total_bytes is not None  # to avoid type checking complaints in the loop below

        self._logger.info(f"Queuing {source} for downloading")
        self._logger.debug(f"remote_files={remote_files}")
        for model_file in remote_files:
            url = model_file.url
            path = model_file.path
            self._logger.info(f"Downloading {url} => {path}")
            install_job.total_bytes += model_file.size
            assert hasattr(source, "access_token")
            dest = tmpdir / path.parent
            dest.mkdir(parents=True, exist_ok=True)
            download_job = DownloadJob(
                source=url,
                dest=dest,
                access_token=source.access_token,
            )
            self._download_cache[download_job.source] = install_job  # matches a download job to an install job
            install_job.download_parts.add(download_job)

            self._download_queue.submit_download_job(
                download_job,
                on_start=self._download_started_callback,
                on_progress=self._download_progress_callback,
                on_complete=self._download_complete_callback,
                on_error=self._download_error_callback,
                on_cancelled=self._download_cancelled_callback,
            )
        return install_job

    def _stat_size(self, path: Path) -> int:
        size = 0
        if path.is_file():
            size = path.stat().st_size
        elif path.is_dir():
            for root, _, files in os.walk(path):
                size += sum(self._stat_size(Path(root, x)) for x in files)
        return size

    # ------------------------------------------------------------------
    # Callbacks are executed by the download queue in a separate thread
    # ------------------------------------------------------------------
    def _download_started_callback(self, download_job: DownloadJob) -> None:
        self._logger.info(f"{download_job.source}: model download started")
        with self._lock:
            install_job = self._download_cache[download_job.source]
            install_job.status = InstallStatus.DOWNLOADING

            assert download_job.download_path
            if install_job.local_path == install_job._install_tmpdir:
                partial_path = download_job.download_path.relative_to(install_job._install_tmpdir)
                dest_name = partial_path.parts[0]
                install_job.local_path = install_job._install_tmpdir / dest_name

            # Update the total bytes count for remote sources.
            if not install_job.total_bytes:
                install_job.total_bytes = sum(x.total_bytes for x in install_job.download_parts)

    def _download_progress_callback(self, download_job: DownloadJob) -> None:
        with self._lock:
            install_job = self._download_cache[download_job.source]
            if install_job.cancelled:  # This catches the case in which the caller directly calls job.cancel()
                self._cancel_download_parts(install_job)
            else:
                # update sizes
                install_job.bytes = sum(x.bytes for x in install_job.download_parts)
                self._signal_job_downloading(install_job)

    def _download_complete_callback(self, download_job: DownloadJob) -> None:
        with self._lock:
            install_job = self._download_cache[download_job.source]
            self._download_cache.pop(download_job.source, None)

            # are there any more active jobs left in this task?
            if all(x.complete for x in install_job.download_parts):
                #  now enqueue job for actual installation into the models directory
                self._install_queue.put(install_job)

            # Let other threads know that the number of downloads has changed
            self._downloads_changed_event.set()

    def _download_error_callback(self, download_job: DownloadJob, excp: Optional[Exception] = None) -> None:
        with self._lock:
            install_job = self._download_cache.pop(download_job.source, None)
            assert install_job is not None
            assert excp is not None
            install_job.set_error(excp)
            self._logger.error(
                f"Cancelling {install_job.source} due to an error while downloading {download_job.source}: {str(excp)}"
            )
            self._cancel_download_parts(install_job)

            # Let other threads know that the number of downloads has changed
            self._downloads_changed_event.set()

    def _download_cancelled_callback(self, download_job: DownloadJob) -> None:
        with self._lock:
            install_job = self._download_cache.pop(download_job.source, None)
            if not install_job:
                return
            self._downloads_changed_event.set()
            self._logger.warning(f"Download {download_job.source} cancelled.")
            # if install job has already registered an error, then do not replace its status with cancelled
            if not install_job.errored:
                install_job.cancel()
            self._cancel_download_parts(install_job)

            # Let other threads know that the number of downloads has changed
            self._downloads_changed_event.set()

    def _cancel_download_parts(self, install_job: ModelInstallJob) -> None:
        # on multipart downloads, _cancel_components() will get called repeatedly from the download callbacks
        # do not lock here because it gets called within a locked context
        for s in install_job.download_parts:
            self._download_queue.cancel_job(s)

        if all(x.in_terminal_state for x in install_job.download_parts):
            # When all parts have reached their terminal state, we finalize the job to clean up the temporary directory and other resources
            self._install_queue.put(install_job)

    # ------------------------------------------------------------------------------------------------
    # Internal methods that put events on the event bus
    # ------------------------------------------------------------------------------------------------
    def _signal_job_running(self, job: ModelInstallJob) -> None:
        job.status = InstallStatus.RUNNING
        self._logger.info(f"{job.source}: model installation started")
        if self._event_bus:
            self._event_bus.emit_model_install_running(str(job.source))

    def _signal_job_downloading(self, job: ModelInstallJob) -> None:
        if self._event_bus:
            parts: List[Dict[str, str | int]] = [
                {
                    "url": str(x.source),
                    "local_path": str(x.download_path),
                    "bytes": x.bytes,
                    "total_bytes": x.total_bytes,
                }
                for x in job.download_parts
            ]
            assert job.bytes is not None
            assert job.total_bytes is not None
            self._event_bus.emit_model_install_downloading(
                str(job.source),
                local_path=job.local_path.as_posix(),
                parts=parts,
                bytes=job.bytes,
                total_bytes=job.total_bytes,
            )

    def _signal_job_completed(self, job: ModelInstallJob) -> None:
        job.status = InstallStatus.COMPLETED
        assert job.config_out
        self._logger.info(
            f"{job.source}: model installation completed. {job.local_path} registered key {job.config_out.key}"
        )
        if self._event_bus:
            assert job.local_path is not None
            assert job.config_out is not None
            key = job.config_out.key
            self._event_bus.emit_model_install_completed(str(job.source), key)

    def _signal_job_errored(self, job: ModelInstallJob) -> None:
        self._logger.info(f"{job.source}: model installation encountered an exception: {job.error_type}\n{job.error}")
        if self._event_bus:
            error_type = job.error_type
            error = job.error
            assert error_type is not None
            assert error is not None
            self._event_bus.emit_model_install_error(str(job.source), error_type, error)

    def _signal_job_cancelled(self, job: ModelInstallJob) -> None:
        self._logger.info(f"{job.source}: model installation was cancelled")
        if self._event_bus:
            self._event_bus.emit_model_install_cancelled(str(job.source))
