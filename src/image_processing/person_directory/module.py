"""
PersonDirectory — Internal Component
=====================================
All internal components for PersonDirectory as specified in
doc/modules/person_directory.md.

Components (spec §8):
    PersonDirectoryJsonLoader   — spec §8.2
    PersonDirectoryValidator    — spec §8.3
    PersonDirectoryStore        — spec §8.4
    PersonDirectory             — spec §8.1  (orchestrator only)

Public API (spec §8.1):
    PersonDirectory(config: PersonDirectoryConfig)
    PersonDirectory.load()                               -> None
    PersonDirectory.get_person(person_id: str)           -> PersonDirectoryOutput
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TypedDict


# ---------------------------------------------------------------------------
# Public data structures — spec §4.1, §5.1
# ---------------------------------------------------------------------------


class PersonDirectoryInput(TypedDict):
    """Input contract for PersonDirectory.get_person() (spec §4.1)."""
    person_id: str


class PersonRecord(TypedDict):
    """Resolved person record (spec §5.1)."""
    person_id: str
    person_name: str


class PersonDirectoryOutput(TypedDict):
    """Output produced by PersonDirectoryStore.get() (spec §5.1)."""
    person_id: str
    person_name: str
    found: bool


# ---------------------------------------------------------------------------
# Internal raw record (loader output, pre-validation)
# ---------------------------------------------------------------------------


class RawPersonRecord(TypedDict):
    """Raw record as parsed from JSON before validation (spec §8.2)."""
    person_name: str


# ---------------------------------------------------------------------------
# Configuration — spec §6.1
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PersonDirectoryConfig:
    """Immutable configuration injected at construction time (spec §6.1)."""
    json_file_path: str
    is_optional: bool = False


# ---------------------------------------------------------------------------
# Canonical UNKNOWN output — spec §5.3
# ---------------------------------------------------------------------------

_UNKNOWN_OUTPUT: PersonDirectoryOutput = PersonDirectoryOutput(
    person_id="UNKNOWN",
    person_name="UNKNOWN",
    found=False,
)


# ---------------------------------------------------------------------------
# Errors — spec §13
# ---------------------------------------------------------------------------


class PersonDirectoryLoadError(RuntimeError):
    """Raised when the JSON file is missing (and not optional) or unparseable."""


class PersonDirectoryValidationError(ValueError):
    """Raised when any validation rule is violated during load."""


# ---------------------------------------------------------------------------
# PersonDirectoryJsonLoader — spec §8.2
# ---------------------------------------------------------------------------


class PersonDirectoryJsonLoader:
    """
    Reads and parses the JSON file.  Returns a raw record map.

    Responsibilities (spec §8.2):
    - Read the file at the given path.
    - Parse JSON content.
    - Detect and reject duplicate person_id keys.
    - Return map<person_id, RawPersonRecord>.

    Does NOT validate business rules or handle missing files.
    """

    def load(self, json_file_path: str) -> dict[str, RawPersonRecord]:
        """
        Read and parse *json_file_path*.

        Raises:
            PersonDirectoryLoadError: if the file cannot be read or parsed,
                or if duplicate person_id keys are detected.
        """
        try:
            with open(json_file_path, encoding="utf-8") as fh:
                raw_text = fh.read()
        except OSError as exc:
            raise PersonDirectoryLoadError(
                f"Cannot read person directory file '{json_file_path}': {exc}"
            ) from exc

        # Detect duplicate keys before the standard parser silently collapses them.
        # Use a per-object local set so nested objects don't cross-contaminate.
        duplicate_keys: list[str] = []

        def _object_pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
            local_seen: set[str] = set()
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in local_seen:
                    duplicate_keys.append(key)
                local_seen.add(key)
                result[key] = value
            return result

        try:
            data = json.loads(raw_text, object_pairs_hook=_object_pairs_hook)
        except json.JSONDecodeError as exc:
            raise PersonDirectoryLoadError(
                f"JSON parse error in '{json_file_path}': {exc}"
            ) from exc

        if duplicate_keys:
            raise PersonDirectoryValidationError(
                f"Duplicate person_id keys in '{json_file_path}': "
                + ", ".join(f'"{k}"' for k in sorted(set(duplicate_keys)))
            )

        if not isinstance(data, dict):
            raise PersonDirectoryLoadError(
                f"Person directory file '{json_file_path}' must contain a "
                "top-level JSON object."
            )

        # Convert each value to RawPersonRecord; additional fields are ignored.
        records: dict[str, RawPersonRecord] = {}
        for person_id, record_obj in data.items():
            if not isinstance(record_obj, dict):
                raise PersonDirectoryLoadError(
                    f"Record for person_id '{person_id}' in '{json_file_path}' "
                    "must be a JSON object."
                )
            person_name = record_obj.get("person_name", "")
            records[person_id] = RawPersonRecord(person_name=person_name)

        return records


# ---------------------------------------------------------------------------
# PersonDirectoryValidator — spec §8.3
# ---------------------------------------------------------------------------


class PersonDirectoryValidator:
    """
    Enforces all validation rules against the loaded raw records (spec §8.3).

    Raises PersonDirectoryValidationError on the first violation (fail-fast).

    Does NOT load files, build the store, or perform lookups.
    """

    _RESERVED_ID = "UNKNOWN"

    def validate(self, records: dict[str, RawPersonRecord]) -> None:
        """
        Validate *records* against all business rules.

        Raises:
            PersonDirectoryValidationError: on the first rule violation.
        """
        seen_ids: set[str] = set()

        for person_id, raw in records.items():
            # Rule: person_id must be non-empty.
            if not person_id:
                raise PersonDirectoryValidationError(
                    "A record with an empty person_id key was found in the "
                    "person directory."
                )

            # Rule: reserved key "UNKNOWN" must not appear.
            if person_id == self._RESERVED_ID:
                raise PersonDirectoryValidationError(
                    'The reserved person_id "UNKNOWN" must not appear in the '
                    "person directory JSON file."
                )

            # Rule: person_name must be non-empty.
            if not raw.get("person_name", ""):
                raise PersonDirectoryValidationError(
                    f"Record for person_id '{person_id}' has an empty person_name."
                )

            # Secondary structural duplicate check (spec §8.3).
            if person_id in seen_ids:
                raise PersonDirectoryValidationError(
                    f"Duplicate person_id '{person_id}' detected during validation."
                )
            seen_ids.add(person_id)


# ---------------------------------------------------------------------------
# PersonDirectoryStore — spec §8.4
# ---------------------------------------------------------------------------


class PersonDirectoryStore:
    """
    In-memory person_id → PersonRecord map with atomic replace semantics
    (spec §8.4).

    Does NOT load files, parse JSON, or apply validation rules.
    """

    def __init__(self) -> None:
        self._records: dict[str, PersonRecord] = {}

    def replace_all(self, records: dict[str, PersonRecord]) -> None:
        """
        Atomically replace the current store with *records* (spec §8.4).

        The old map is discarded only after the new one is fully assigned.
        """
        self._records = dict(records)  # take a defensive copy

    def get(self, person_id: str) -> PersonDirectoryOutput:
        """
        Look up *person_id* and return a PersonDirectoryOutput.

        Returns the canonical UNKNOWN output for any unresolvable input:
        - unknown person_id
        - empty person_id
        - person_id == "UNKNOWN"
        """
        if not person_id or person_id == "UNKNOWN":
            return _UNKNOWN_OUTPUT.copy()

        record = self._records.get(person_id)
        if record is None:
            return _UNKNOWN_OUTPUT.copy()

        return PersonDirectoryOutput(
            person_id=record["person_id"],
            person_name=record["person_name"],
            found=True,
        )


# ---------------------------------------------------------------------------
# PersonDirectory — spec §8.1  (orchestrator)
# ---------------------------------------------------------------------------


class PersonDirectory:
    """
    Public-facing orchestration component for PersonDirectory (spec §8.1).

    Owns no loading, validation, or storage logic directly.

    Public API:
        PersonDirectory(config: PersonDirectoryConfig)
        load()                             -> None
        get_person(person_id: str)         -> PersonDirectoryOutput
    """

    def __init__(self, config: PersonDirectoryConfig) -> None:
        self._config = config
        self._loader = PersonDirectoryJsonLoader()
        self._validator = PersonDirectoryValidator()
        self._store = PersonDirectoryStore()
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load, validate, and store the person directory (spec §9.1, §10.1).

        Must be called once before any get_person() call.

        Raises:
            PersonDirectoryLoadError: if the file is missing (and not
                optional) or unparseable.
            PersonDirectoryValidationError: if any record violates a
                validation rule.
        """
        try:
            raw_records = self._loader.load(self._config.json_file_path)
        except PersonDirectoryLoadError:
            if self._config.is_optional:
                # Silently initialize an empty store (spec §9.1).
                self._store.replace_all({})
                self._loaded = True
                return
            raise

        # An empty object {} is valid — skip validation, store empty map.
        if not raw_records:
            self._store.replace_all({})
            self._loaded = True
            return

        self._validator.validate(raw_records)

        validated: dict[str, PersonRecord] = {
            pid: PersonRecord(person_id=pid, person_name=raw["person_name"])
            for pid, raw in raw_records.items()
        }
        self._store.replace_all(validated)
        self._loaded = True

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------

    def get_person(self, person_id: str) -> PersonDirectoryOutput:
        """
        Resolve *person_id* to a PersonDirectoryOutput (spec §10.2).

        Raises:
            RuntimeError: if ``load()`` has not been called successfully.
        """
        if not self._loaded:
            raise RuntimeError("PersonDirectory.load() must be called before get_person()")
        return self._store.get(person_id)
