# Copyright 2023, Lincoln D. Stein and the InvokeAI Team
"""
Abstract base class and implementation for recursive directory search for models.

Example usage:
```
  from invokeai.backend.model_manager import ModelSearch, ModelProbe

  def find_main_models(model: Path) -> bool:
    info = ModelProbe.probe(model)
    if info.model_type == 'main' and info.base_type == 'sd-1':
        return True
    else:
        return False

  search = ModelSearch(on_model_found=report_it)
  found = search.search('/tmp/models')
  print(found)   #  list of matching model paths
  print(search.stats)  #  search stats
```
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional, Set, Union

from pydantic import BaseModel, Field

from invokeai.backend.util.logging import InvokeAILogger

default_logger = InvokeAILogger.get_logger()


class SearchStats(BaseModel):
    items_scanned: int = 0
    models_found: int = 0
    models_filtered: int = 0


class ModelSearchBase(ABC, BaseModel):
    """
    Abstract directory traversal model search class

    Usage:
       search = ModelSearchBase(
            on_search_started = search_started_callback,
            on_search_completed = search_completed_callback,
            on_model_found = model_found_callback,
       )
       models_found = search.search('/path/to/directory')
    """

    # fmt: off
    on_search_started   : Optional[Callable[[Path], None]]      = Field(default=None, description="Called just before the search starts.")  # noqa E221
    on_model_found      : Optional[Callable[[Path], bool]]      = Field(default=None, description="Called when a model is found.")          # noqa E221
    on_search_completed : Optional[Callable[[Set[Path]], None]] = Field(default=None, description="Called when search is complete.")        # noqa E221
    stats               : SearchStats                           = Field(default_factory=SearchStats, description="Summary statistics after search")  # noqa E221
    logger              : InvokeAILogger                        = Field(default=default_logger, description="Logger instance.")     # noqa E221
    # fmt: on

    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def search_started(self) -> None:
        """
        Called before the scan starts.

        Passes the root search directory to the Callable `on_search_started`.
        """
        pass

    @abstractmethod
    def model_found(self, model: Path) -> None:
        """
        Called when a model is found during search.

        :param model: Model to process - could be a directory or checkpoint.

        Passes the model's Path to the Callable `on_model_found`.
        This Callable receives the path to the model and returns a boolean
        to indicate whether the model should be returned in the search
        results.
        """
        pass

    @abstractmethod
    def search_completed(self) -> None:
        """
        Called before the scan starts.

        Passes the Set of found model Paths to the Callable `on_search_completed`.
        """
        pass

    @abstractmethod
    def search(self, directory: Union[Path, str]) -> Set[Path]:
        """
        Recursively search for models in `directory` and return a set of model paths.

        If provided, the `on_search_started`, `on_model_found` and `on_search_completed`
        Callables will be invoked during the search.
        """
        pass


class ModelSearch(ModelSearchBase):
    """
    Implementation of ModelSearch with callbacks.
    Usage:
       search = ModelSearch()
       search.model_found = lambda path : 'anime' in path.as_posix()
       found = search.list_models(['/tmp/models1','/tmp/models2'])
       # returns all models that have 'anime' in the path
    """

    models_found: Set[Path] = Field(default=None)
    scanned_dirs: Set[Path] = Field(default=None)
    pruned_paths: Set[Path] = Field(default=None)

    def search_started(self) -> None:
        self.models_found = set()
        self.scanned_dirs = set()
        self.pruned_paths = set()
        if self.on_search_started:
            self.on_search_started(self._directory)

    def model_found(self, model: Path) -> None:
        self.stats.models_found += 1
        if not self.on_model_found or self.on_model_found(model):
            self.stats.models_filtered += 1
            self.models_found.add(model)

    def search_completed(self) -> None:
        if self.on_search_completed:
            self.on_search_completed(self._models_found)

    def search(self, directory: Union[Path, str]) -> Set[Path]:
        self._directory = Path(directory)
        self.stats = SearchStats()  # zero out
        self.search_started()  # This will initialize _models_found to empty
        self._walk_directory(directory)
        self.search_completed()
        return self.models_found

    def _walk_directory(self, path: Union[Path, str]) -> None:
        for root, dirs, files in os.walk(path, followlinks=True):
            # don't descend into directories that start with a "."
            # to avoid the Mac .DS_STORE issue.
            if str(Path(root).name).startswith("."):
                self.pruned_paths.add(Path(root))
            if any(Path(root).is_relative_to(x) for x in self.pruned_paths):
                continue

            self.stats.items_scanned += len(dirs) + len(files)
            for d in dirs:
                path = Path(root) / d
                if path.parent in self.scanned_dirs:
                    self.scanned_dirs.add(path)
                    continue
                if any(
                    (path / x).exists()
                    for x in [
                        "config.json",
                        "model_index.json",
                        "learned_embeds.bin",
                        "pytorch_lora_weights.bin",
                        "image_encoder.txt",
                    ]
                ):
                    self.scanned_dirs.add(path)
                    try:
                        self.model_found(path)
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        self.logger.warning(str(e))

            for f in files:
                path = Path(root) / f
                if path.parent in self.scanned_dirs:
                    continue
                if path.suffix in {".ckpt", ".bin", ".pth", ".safetensors", ".pt"}:
                    try:
                        self.model_found(path)
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        self.logger.warning(str(e))
