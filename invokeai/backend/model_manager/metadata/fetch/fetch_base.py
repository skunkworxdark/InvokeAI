# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Development Team

"""
This module is the base class for subclasses that fetch metadata from model repositories

Usage:

from invokeai.backend.model_manager.metadata.fetch import CivitAIMetadataFetch

fetcher = CivitaiMetadataFetch()
metadata = fetcher.from_url("https://civitai.com/models/206883/split")
print(metadata.trained_words)
"""

from abc import ABC, abstractmethod
from typing import Optional

from pydantic.networks import AnyHttpUrl
from requests.sessions import Session

from ..metadata_base import AnyModelRepoMetadata, AnyModelRepoMetadataValidator


class ModelMetadataFetchBase(ABC):
    """Fetch metadata from remote generative model repositories."""

    @abstractmethod
    def __init__(self, session: Optional[Session] = None):
        """
        Initialize the fetcher with an optional requests.sessions.Session object.

        By providing a configurable Session object, we can support unit tests on
        this module without an internet connection.
        """
        pass

    @abstractmethod
    def from_url(self, url: AnyHttpUrl) -> AnyModelRepoMetadata:
        """
        Given a URL to a model repository, return a ModelMetadata object.

        This method will raise a `UnknownMetadataException`
        in the event that the requested model metadata is not found at the provided location.
        """
        pass

    @abstractmethod
    def from_id(self, id: str) -> AnyModelRepoMetadata:
        """
        Given an ID for a model, return a ModelMetadata object.

        This method will raise a `UnknownMetadataException`
        in the event that the requested model's metadata is not found at the provided id.
        """
        pass

    @classmethod
    def from_json(cls, json: str) -> AnyModelRepoMetadata:
        """Given the JSON representation of the metadata, return the corresponding Pydantic object."""
        metadata = AnyModelRepoMetadataValidator.validate_json(json)
        return metadata
