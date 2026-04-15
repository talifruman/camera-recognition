from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Protocol, TypedDict, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

# FaceEmbedding is an opaque 1D float32 vector.  This module validates only
# its structural contract (non-null, expected dimension, expected dtype).
FaceEmbedding = np.ndarray


# ---------------------------------------------------------------------------
# Public data structures — spec §3.1, §10
# ---------------------------------------------------------------------------


class GalleryEntry(TypedDict):
    """Loaded gallery record (spec §3.1).  person_id and embedding only."""
    person_id: str
    embedding: FaceEmbedding


# ---------------------------------------------------------------------------
# Internal data structures — spec §10 (never exposed through public API)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PersonScanRecord:
    """Person directory scan result produced by GalleryDirectoryScanner."""
    person_id: str
    file_paths: list[str]


# ---------------------------------------------------------------------------
# Configuration — spec §9.1
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FaceGalleryLoaderConfig:
    """Immutable configuration loaded once at module initialization."""
    embedding_file_extension: str = ".npy"
    expected_embedding_dim: int = 512
    expected_dtype: str = "float32"


# ---------------------------------------------------------------------------
# Structured errors — spec §11
# ---------------------------------------------------------------------------


class GalleryPathValidationError(ValueError):
    """Raised by GalleryPathValidator when gallery_root_path is invalid."""


class GalleryLoadError(RuntimeError):
    """Raised by FaceGalleryLoaderModule when no valid embeddings are found."""


# ---------------------------------------------------------------------------
# EmbeddingFileReader interface — spec §6.1
# ---------------------------------------------------------------------------


@runtime_checkable
class EmbeddingFileReader(Protocol):
    """
    File-reading engine abstraction (spec §6.1, §8.4).

    Returns the FaceEmbedding on success, or None (structured failure) on any
    format, dimension, dtype, or I/O error.  This is the only component with
    knowledge of the embedding file format on disk.
    """

    def read_embedding(self, file_path: str) -> FaceEmbedding | None: ...


# ---------------------------------------------------------------------------
# StubEmbeddingFileReader — Phase 1 default (replaces NpyEmbeddingFileReader)
# ---------------------------------------------------------------------------


class StubEmbeddingFileReader:
    """
    Stub implementation of EmbeddingFileReader for Phase 1 testing.

    Validates that the file exists and the extension matches configuration,
    then returns a constant, deterministic, normalized float32 vector of the
    configured dimension without reading the file contents.  The same vector
    is returned for every valid file.

    Implements EmbeddingFileReader.
    """

    def __init__(
        self,
        embedding_file_extension: str,
        expected_embedding_dim: int,
        expected_dtype: str,
    ) -> None:
        self._extension = embedding_file_extension
        self._dim = expected_embedding_dim
        self._dtype = expected_dtype
        # Pre-compute the constant embedding once.  Normalized vector where
        # every element equals 1 / sqrt(dim), so ||v||_2 == 1.
        value = 1.0 / np.sqrt(expected_embedding_dim)
        self._constant_embedding: FaceEmbedding = np.full(
            expected_embedding_dim, value, dtype=np.float32
        )

    def read_embedding(self, file_path: str) -> FaceEmbedding | None:
        """
        Validate minimal conditions and return a constant embedding.

        Returns None (structured failure) if:
          - the file does not exist
          - the file extension does not match the configured extension
        """
        if not os.path.isfile(file_path):
            return None
        _, ext = os.path.splitext(file_path)
        if ext.lower() != self._extension.lower():
            return None
        # Return the pre-computed constant; callers must not mutate it.
        return self._constant_embedding


# ---------------------------------------------------------------------------
# GalleryPathValidator — spec §8.2
# ---------------------------------------------------------------------------


class GalleryPathValidator:
    """
    Validates gallery_root_path before any scanning or loading begins
    (spec §8.2, §2.4).

    Responsibilities:
      - non-empty string check
      - existence and directory check
      - read permission check
      - at least one direct child entry check

    Does NOT scan directory contents beyond confirming at least one entry
    exists, read any embedding file, or make decisions about person
    directories or file types.
    """

    def validate(self, gallery_root_path: str) -> None:
        """Raise GalleryPathValidationError on any contract violation."""
        if not gallery_root_path or not gallery_root_path.strip():
            raise GalleryPathValidationError(
                "gallery_root_path must be a non-empty string."
            )

        if not os.path.exists(gallery_root_path):
            raise GalleryPathValidationError(
                f"gallery_root_path does not exist: {gallery_root_path!r}"
            )

        if not os.path.isdir(gallery_root_path):
            raise GalleryPathValidationError(
                f"gallery_root_path is not a directory: {gallery_root_path!r}"
            )

        if not os.access(gallery_root_path, os.R_OK):
            raise GalleryPathValidationError(
                f"gallery_root_path is not readable by this process: "
                f"{gallery_root_path!r}"
            )

        try:
            entries = os.listdir(gallery_root_path)
        except OSError as exc:
            raise GalleryPathValidationError(
                f"Cannot list gallery_root_path: {gallery_root_path!r}"
            ) from exc

        if not entries:
            raise GalleryPathValidationError(
                f"gallery_root_path contains no entries: {gallery_root_path!r}"
            )


# ---------------------------------------------------------------------------
# GalleryDirectoryScanner — spec §8.3
# ---------------------------------------------------------------------------


class GalleryDirectoryScanner:
    """
    Discovers person directories and enumerates embedding files within them
    (spec §8.3).

    Responsibilities:
      - list all direct subdirectories under gallery_root_path (one per person)
      - derive person_id from subdirectory name
      - enumerate files within each person directory that match
        embedding_file_extension
      - skip non-directory entries under gallery_root_path without error

    Does NOT read file contents, validate embedding values, scan nested
    subdirectories inside a person directory, or decide whether any file is a
    valid embedding.
    """

    def __init__(self, embedding_file_extension: str) -> None:
        self._extension = embedding_file_extension

    def scan(self, gallery_root_path: str) -> list[PersonScanRecord]:
        """
        Scan gallery_root_path and return a PersonScanRecord for each direct
        subdirectory containing at least one file with the configured extension.

        Non-directory entries under gallery_root_path are silently skipped.
        The returned list is sorted by person_id for deterministic ordering.
        Files within each record are sorted by filename for determinism.
        """
        records: list[PersonScanRecord] = []

        try:
            child_names = sorted(os.listdir(gallery_root_path))
        except OSError:
            return records

        for name in child_names:
            child_path = os.path.join(gallery_root_path, name)
            if not os.path.isdir(child_path):
                # Non-directory entries are silently skipped (spec §8.3).
                continue

            person_id = name
            matched_files: list[str] = []

            try:
                file_names = sorted(os.listdir(child_path))
            except OSError:
                continue

            for file_name in file_names:
                _, ext = os.path.splitext(file_name)
                if ext.lower() == self._extension.lower():
                    matched_files.append(os.path.join(child_path, file_name))

            records.append(PersonScanRecord(person_id=person_id, file_paths=matched_files))

        return records


# ---------------------------------------------------------------------------
# FaceGalleryCache — spec §8.5
# ---------------------------------------------------------------------------


class FaceGalleryCache:
    """
    Stores the loaded gallery in memory and supports read access (spec §8.5).

    This is the only component that holds gallery state in memory.

    Internal storage: dict[person_id, list[GalleryEntry]] where the list
    preserves file enumeration order within each person.
    """

    def __init__(self) -> None:
        # _index maps person_id → entries in file enumeration order.
        self._index: dict[str, list[GalleryEntry]] = {}

    def replace_all(self, entries: list[GalleryEntry]) -> None:
        """
        Atomically replace all stored entries (spec §8.5).

        Rebuilds the internal index from the provided list.  The list is
        expected to be sorted by person_id then by file order within each
        person (enforced by the orchestrator).
        """
        new_index: dict[str, list[GalleryEntry]] = {}
        for entry in entries:
            pid = entry["person_id"]
            if pid not in new_index:
                new_index[pid] = []
            new_index[pid].append(entry)
        # Atomic replacement.
        self._index = new_index

    def get_all_entries(self) -> list[GalleryEntry]:
        """
        Return all entries sorted by person_id lexicographically, then by
        file enumeration order within each person (spec §5, §8.5).
        """
        result: list[GalleryEntry] = []
        for pid in sorted(self._index.keys()):
            result.extend(self._index[pid])
        return result

    def get_entries_by_person(self, person_id: str) -> list[GalleryEntry]:
        """
        Return entries for the given person_id in file enumeration order.
        Returns an empty list if person_id is not present (not an error).
        """
        return list(self._index.get(person_id, []))

    def get_person_ids(self) -> list[str]:
        """Return all loaded person identifiers in lexicographically sorted order."""
        return sorted(self._index.keys())


# ---------------------------------------------------------------------------
# FaceGalleryLoaderModule — spec §8.1 (orchestration layer only)
# ---------------------------------------------------------------------------


class FaceGalleryLoaderModule:
    """
    Orchestration layer for the FaceGalleryLoader module (spec §4, §8.1).

    Coordinates internal components in the correct order and exposes the
    public API.  Does not embed path validation, directory scanning, file
    reading, embedding content validation, or cache construction logic.

    Public API (spec §4):
        load_gallery(gallery_root_path: str) -> None
        get_all_embeddings()               -> list[GalleryEntry]
        get_embeddings(person_id: str)     -> list[GalleryEntry]
        get_person_ids()                   -> list[str]
    """

    def __init__(
        self,
        config: FaceGalleryLoaderConfig | None = None,
        reader: EmbeddingFileReader | None = None,
    ) -> None:
        """
        Initialize the module and wire internal components.

        config  — if None, a default FaceGalleryLoaderConfig is used.
        reader  — if None, StubEmbeddingFileReader is used (Phase 1 default).
                  In Phase 2, pass NpyEmbeddingFileReader(config) here to
                  switch to real deserialization without any other change.
        """
        if config is None:
            config = FaceGalleryLoaderConfig()

        self._config = config

        # Wire the reader: Phase 1 default is the stub.
        if reader is None:
            reader = StubEmbeddingFileReader(
                embedding_file_extension=config.embedding_file_extension,
                expected_embedding_dim=config.expected_embedding_dim,
                expected_dtype=config.expected_dtype,
            )
        self._reader: EmbeddingFileReader = reader

        self._validator = GalleryPathValidator()
        self._scanner = GalleryDirectoryScanner(config.embedding_file_extension)
        self._cache = FaceGalleryCache()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_gallery(self, gallery_root_path: str) -> None:
        """
        Load embeddings from gallery_root_path into the in-memory cache.

        Pipeline (spec §8.6):
          1. GalleryPathValidator.validate
          2. GalleryDirectoryScanner.scan  →  PersonScanRecord[]
          3. EmbeddingFileReader.read_embedding per file  →  skip failures
          4. Verify at least one GalleryEntry collected
          5. FaceGalleryCache.replace_all

        Raises GalleryPathValidationError if the path is invalid.
        Raises GalleryLoadError if no valid embeddings are found.
        Cache is NOT modified on any error.
        """
        # Step 1: path validation.
        self._validator.validate(gallery_root_path)

        # Step 2: directory scan.
        person_records = self._scanner.scan(gallery_root_path)

        # Step 3 & 4: load embeddings; apply skip-or-fail policy.
        collected: list[GalleryEntry] = []
        skipped_count = 0

        for record in person_records:
            for file_path in record.file_paths:
                embedding = self._reader.read_embedding(file_path)
                if embedding is None:
                    # Structured failure — skip this file, record warning.
                    skipped_count += 1
                    warnings.warn(
                        f"Skipping embedding file (structured failure): "
                        f"{os.path.basename(file_path)} "
                        f"[person_id={record.person_id}]",
                        stacklevel=2,
                    )
                    continue
                collected.append(
                    GalleryEntry(person_id=record.person_id, embedding=embedding)
                )

        # Step 5: verify non-empty.
        if not collected:
            raise GalleryLoadError(
                "No valid embeddings were loaded from the gallery. "
                f"gallery_root_path={gallery_root_path!r}, "
                f"files_skipped={skipped_count}"
            )

        # Step 6: atomic cache replacement.
        self._cache.replace_all(collected)

    def get_all_embeddings(self) -> list[GalleryEntry]:
        """Return all cached GalleryEntry values (spec §4)."""
        return self._cache.get_all_entries()

    def get_embeddings(self, person_id: str) -> list[GalleryEntry]:
        """
        Return GalleryEntry values for person_id.

        Returns an empty list if person_id is not found — not an error
        (spec §2.4, §4, §11).
        """
        return self._cache.get_entries_by_person(person_id)

    def get_person_ids(self) -> list[str]:
        """Return all loaded person identifiers in lexicographically sorted order."""
        return self._cache.get_person_ids()
