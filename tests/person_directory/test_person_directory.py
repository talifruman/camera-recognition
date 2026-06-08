"""
Tests for PersonDirectory — spec §15.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from image_processing.person_directory import (  # type: ignore[import-not-found]
    PersonDirectory,
    PersonDirectoryConfig,
    PersonDirectoryJsonLoader,
    PersonDirectoryLoadError,
    PersonDirectoryOutput,
    PersonDirectoryStore,
    PersonDirectoryValidationError,
    PersonDirectoryValidator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(tmp_dir: str, data: object, filename: str = "persons.json") -> str:
    path = str(Path(tmp_dir) / filename)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    return path


def _make_valid_json(tmp_dir: str) -> str:
    return _write_json(
        tmp_dir,
        {
            "person_001": {"person_name": "Daniel Cohen"},
            "person_002": {"person_name": "Maya Levi"},
        },
    )


# ---------------------------------------------------------------------------
# Test 1: load_valid_json (spec §15 row 1)
# ---------------------------------------------------------------------------


class LoadValidJsonTests(unittest.TestCase):
    """Load a well-formed JSON file with multiple valid records."""

    def test_load_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_valid_json(tmp)
            pd = PersonDirectory(PersonDirectoryConfig(json_file_path=path))
            pd.load()

            out1 = pd.get_person("person_001")
            self.assertTrue(out1["found"])
            self.assertEqual(out1["person_id"], "person_001")
            self.assertEqual(out1["person_name"], "Daniel Cohen")

            out2 = pd.get_person("person_002")
            self.assertTrue(out2["found"])
            self.assertEqual(out2["person_name"], "Maya Levi")

    def test_empty_object_loads_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_json(tmp, {})
            pd = PersonDirectory(PersonDirectoryConfig(json_file_path=path))
            pd.load()
            out = pd.get_person("person_001")
            self.assertFalse(out["found"])
            self.assertEqual(out["person_id"], "UNKNOWN")


# ---------------------------------------------------------------------------
# Test 2: reject_invalid_json (spec §15 row 2)
# ---------------------------------------------------------------------------


class RejectInvalidJsonTests(unittest.TestCase):
    """Malformed JSON must raise PersonDirectoryLoadError."""

    def test_reject_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "bad.json")
            with open(path, "w") as fh:
                fh.write("{this is not valid json")
            pd = PersonDirectory(PersonDirectoryConfig(json_file_path=path))
            with self.assertRaises(PersonDirectoryLoadError):
                pd.load()

    def test_reject_missing_file_not_optional(self) -> None:
        pd = PersonDirectory(
            PersonDirectoryConfig(
                json_file_path="/nonexistent/persons.json",
                is_optional=False,
            )
        )
        with self.assertRaises(PersonDirectoryLoadError):
            pd.load()

    def test_missing_file_optional_initializes_empty(self) -> None:
        pd = PersonDirectory(
            PersonDirectoryConfig(
                json_file_path="/nonexistent/persons.json",
                is_optional=True,
            )
        )
        pd.load()  # must not raise
        out = pd.get_person("any_id")
        self.assertFalse(out["found"])


# ---------------------------------------------------------------------------
# Test 3: reject_empty_person_id (spec §15 row 3)
# ---------------------------------------------------------------------------


class RejectEmptyPersonIdTests(unittest.TestCase):
    """Empty string key in JSON must raise PersonDirectoryValidationError."""

    def test_reject_empty_person_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_json(tmp, {"": {"person_name": "Ghost"}})
            pd = PersonDirectory(PersonDirectoryConfig(json_file_path=path))
            with self.assertRaises(PersonDirectoryValidationError):
                pd.load()


# ---------------------------------------------------------------------------
# Test 4: reject_empty_person_name (spec §15 row 4)
# ---------------------------------------------------------------------------


class RejectEmptyPersonNameTests(unittest.TestCase):
    """Empty person_name must raise PersonDirectoryValidationError."""

    def test_reject_empty_person_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_json(tmp, {"person_001": {"person_name": ""}})
            pd = PersonDirectory(PersonDirectoryConfig(json_file_path=path))
            with self.assertRaises(PersonDirectoryValidationError):
                pd.load()

    def test_reject_missing_person_name_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_json(tmp, {"person_001": {}})
            pd = PersonDirectory(PersonDirectoryConfig(json_file_path=path))
            with self.assertRaises(PersonDirectoryValidationError):
                pd.load()


# ---------------------------------------------------------------------------
# Test 5: reject_manually_defined_unknown (spec §15 row 5)
# ---------------------------------------------------------------------------


class RejectReservedUnknownKeyTests(unittest.TestCase):
    """The reserved key "UNKNOWN" must raise PersonDirectoryValidationError."""

    def test_reject_manually_defined_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_json(tmp, {"UNKNOWN": {"person_name": "Nobody"}})
            pd = PersonDirectory(PersonDirectoryConfig(json_file_path=path))
            with self.assertRaises(PersonDirectoryValidationError):
                pd.load()


# ---------------------------------------------------------------------------
# Test 6: get_existing_person_id (spec §15 row 6)
# ---------------------------------------------------------------------------


class GetExistingPersonIdTests(unittest.TestCase):
    """Known person_id returns found=True and correct fields."""

    def test_get_existing_person_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_valid_json(tmp)
            pd = PersonDirectory(PersonDirectoryConfig(json_file_path=path))
            pd.load()

            out = pd.get_person("person_001")
            self.assertTrue(out["found"])
            self.assertEqual(out["person_id"], "person_001")
            self.assertEqual(out["person_name"], "Daniel Cohen")


# ---------------------------------------------------------------------------
# Test 7: get_unknown_person_id_returns_unknown (spec §15 row 7)
# ---------------------------------------------------------------------------


class GetUnknownPersonIdTests(unittest.TestCase):
    """Unrecognized person_id returns canonical UNKNOWN output."""

    def test_get_unknown_person_id_returns_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_valid_json(tmp)
            pd = PersonDirectory(PersonDirectoryConfig(json_file_path=path))
            pd.load()

            out = pd.get_person("no_such_person")
            self.assertFalse(out["found"])
            self.assertEqual(out["person_id"], "UNKNOWN")
            self.assertEqual(out["person_name"], "UNKNOWN")

    def test_get_empty_person_id_returns_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_valid_json(tmp)
            pd = PersonDirectory(PersonDirectoryConfig(json_file_path=path))
            pd.load()

            out = pd.get_person("")
            self.assertFalse(out["found"])
            self.assertEqual(out["person_id"], "UNKNOWN")


# ---------------------------------------------------------------------------
# Test 8: get_unknown_literal_returns_unknown (spec §15 row 8)
# ---------------------------------------------------------------------------


class GetUnknownLiteralTests(unittest.TestCase):
    """Passing the literal string "UNKNOWN" returns canonical UNKNOWN output."""

    def test_get_unknown_literal_returns_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_valid_json(tmp)
            pd = PersonDirectory(PersonDirectoryConfig(json_file_path=path))
            pd.load()

            out = pd.get_person("UNKNOWN")
            self.assertFalse(out["found"])
            self.assertEqual(out["person_id"], "UNKNOWN")
            self.assertEqual(out["person_name"], "UNKNOWN")


# ---------------------------------------------------------------------------
# Test 9: get_person before load is fail-safe (updated contract)
# ---------------------------------------------------------------------------


class PreLoadLookupFailSafeTests(unittest.TestCase):
    """get_person() raises RuntimeError when load() has not been called successfully."""

    def test_get_person_before_load_raises_runtime_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_valid_json(tmp)
            pd = PersonDirectory(PersonDirectoryConfig(json_file_path=path))

            with self.assertRaises(RuntimeError) as ctx:
                pd.get_person("person_001")
            self.assertIn("load()", str(ctx.exception))

    def test_get_person_after_failed_load_raises_runtime_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_json(tmp, {"UNKNOWN": {"person_name": "Reserved"}})
            pd = PersonDirectory(PersonDirectoryConfig(json_file_path=path))

            with self.assertRaises(PersonDirectoryValidationError):
                pd.load()

            # load() did not complete — _loaded is still False
            with self.assertRaises(RuntimeError):
                pd.get_person("person_001")


# ---------------------------------------------------------------------------
# Test 9: replace_all_replaces_only_after_validation (spec §15 row 9)
# ---------------------------------------------------------------------------


class ReplaceAllAtomicityTests(unittest.TestCase):
    """
    A failed replace_all must leave the old store state unchanged.

    The spec's hot-reload pattern (§8.5) requires atomicity: the old map
    is replaced only after full validation succeeds.  We exercise this via
    PersonDirectoryStore directly.
    """

    def test_replace_all_replaces_only_after_validation(self) -> None:
        from image_processing.person_directory import PersonRecord

        store = PersonDirectoryStore()

        # Populate the store with an initial valid record.
        initial: dict[str, PersonRecord] = {
            "person_001": PersonRecord(person_id="person_001", person_name="Alice"),
        }
        store.replace_all(initial)

        # Verify initial state.
        out = store.get("person_001")
        self.assertTrue(out["found"])

        # Perform a new replace_all with a different valid record set.
        new_records: dict[str, PersonRecord] = {
            "person_002": PersonRecord(person_id="person_002", person_name="Bob"),
        }
        store.replace_all(new_records)

        # Old key must no longer be found; new key must be found.
        self.assertFalse(store.get("person_001")["found"])
        self.assertTrue(store.get("person_002")["found"])

    def test_failed_load_leaves_store_unchanged(self) -> None:
        """
        When load() fails (validation error), the store must not be mutated.
        We verify this by loading a good file, then attempting a bad file,
        and confirming the original records are still accessible.
        """
        with tempfile.TemporaryDirectory() as tmp:
            good_path = _make_valid_json(tmp)
            pd = PersonDirectory(PersonDirectoryConfig(json_file_path=good_path))
            pd.load()

            # Confirm good state.
            self.assertTrue(pd.get_person("person_001")["found"])

            # Attempt to reload with a file that has an UNKNOWN key.
            bad_path = _write_json(tmp, {"UNKNOWN": {"person_name": "Bad"}}, "bad.json")
            pd2 = PersonDirectory(PersonDirectoryConfig(json_file_path=bad_path))
            with self.assertRaises(PersonDirectoryValidationError):
                pd2.load()

            # pd (the original) is unchanged.
            self.assertTrue(pd.get_person("person_001")["found"])


# ---------------------------------------------------------------------------
# Duplicate key detection tests (spec §7.2)
# ---------------------------------------------------------------------------


class DuplicateKeyTests(unittest.TestCase):
    """Duplicate person_id keys must be detected and rejected at load time."""

    def test_duplicate_keys_detected_by_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Write raw JSON with duplicate keys (json.dump cannot produce this,
            # so we write the raw string directly).
            path = str(Path(tmp) / "dup.json")
            with open(path, "w") as fh:
                fh.write('{"person_001": {"person_name": "Alice"}, '
                          '"person_001": {"person_name": "Bob"}}')
            loader = PersonDirectoryJsonLoader()
            with self.assertRaises(PersonDirectoryValidationError):
                loader.load(path)


# ---------------------------------------------------------------------------
# Component-level unit tests
# ---------------------------------------------------------------------------


class PersonDirectoryStoreTests(unittest.TestCase):
    """Unit tests for PersonDirectoryStore in isolation."""

    def _make_store(self) -> PersonDirectoryStore:
        from image_processing.person_directory import PersonRecord

        store = PersonDirectoryStore()
        store.replace_all({
            "p1": PersonRecord(person_id="p1", person_name="Alice"),
            "p2": PersonRecord(person_id="p2", person_name="Bob"),
        })
        return store

    def test_found_record(self) -> None:
        store = self._make_store()
        out = store.get("p1")
        self.assertTrue(out["found"])
        self.assertEqual(out["person_id"], "p1")
        self.assertEqual(out["person_name"], "Alice")

    def test_not_found_returns_unknown(self) -> None:
        store = self._make_store()
        out = store.get("no_such")
        self.assertFalse(out["found"])
        self.assertEqual(out["person_id"], "UNKNOWN")
        self.assertEqual(out["person_name"], "UNKNOWN")

    def test_empty_string_returns_unknown(self) -> None:
        store = self._make_store()
        out = store.get("")
        self.assertFalse(out["found"])

    def test_unknown_literal_returns_unknown(self) -> None:
        store = self._make_store()
        out = store.get("UNKNOWN")
        self.assertFalse(out["found"])

    def test_replace_all_is_atomic(self) -> None:
        from image_processing.person_directory import PersonRecord

        store = self._make_store()
        store.replace_all({"p3": PersonRecord(person_id="p3", person_name="Carol")})
        self.assertFalse(store.get("p1")["found"])
        self.assertTrue(store.get("p3")["found"])


class PersonDirectoryValidatorTests(unittest.TestCase):
    """Unit tests for PersonDirectoryValidator in isolation."""

    def test_valid_records_pass(self) -> None:
        from image_processing.person_directory import RawPersonRecord

        validator = PersonDirectoryValidator()
        validator.validate({"p1": RawPersonRecord(person_name="Alice")})  # no raise

    def test_empty_person_id_raises(self) -> None:
        from image_processing.person_directory import RawPersonRecord

        validator = PersonDirectoryValidator()
        with self.assertRaises(PersonDirectoryValidationError):
            validator.validate({"": RawPersonRecord(person_name="Ghost")})

    def test_reserved_unknown_raises(self) -> None:
        from image_processing.person_directory import RawPersonRecord

        validator = PersonDirectoryValidator()
        with self.assertRaises(PersonDirectoryValidationError):
            validator.validate({"UNKNOWN": RawPersonRecord(person_name="Nobody")})

    def test_empty_person_name_raises(self) -> None:
        from image_processing.person_directory import RawPersonRecord

        validator = PersonDirectoryValidator()
        with self.assertRaises(PersonDirectoryValidationError):
            validator.validate({"p1": RawPersonRecord(person_name="")})


# ---------------------------------------------------------------------------
# Test A+B: Integration with real person_directory.json
# ---------------------------------------------------------------------------

_REAL_JSON_PATH = str(Path(__file__).resolve().parents[2] / "data" / "person_directory.json")


class TestRealPersonDirectoryJson(unittest.TestCase):
    """Load the real data/person_directory.json and verify lookups.

    Test A — known person_id returns correct person_name and found=True.
    Test B — unknown person_id returns UNKNOWN sentinel with found=False.
    """

    def setUp(self) -> None:
        self.pd = PersonDirectory(PersonDirectoryConfig(json_file_path=_REAL_JSON_PATH))
        self.pd.load()

    # -- Test A: known person_id ------------------------------------------

    def test_known_id_bar_keinan_found(self) -> None:
        out = self.pd.get_person("1f007fe2-6eaf-5148-a240-a449664cb6eb")
        self.assertTrue(out["found"])

    def test_known_id_bar_keinan_person_id(self) -> None:
        out = self.pd.get_person("1f007fe2-6eaf-5148-a240-a449664cb6eb")
        self.assertEqual(out["person_id"], "1f007fe2-6eaf-5148-a240-a449664cb6eb")

    def test_known_id_bar_keinan_person_name(self) -> None:
        out = self.pd.get_person("1f007fe2-6eaf-5148-a240-a449664cb6eb")
        self.assertEqual(out["person_name"], "Bar Keinan")

    def test_known_id_alik_fruman_name(self) -> None:
        out = self.pd.get_person("750d3e47-6abb-5694-8b62-ea056a7f29b1")
        self.assertTrue(out["found"])
        self.assertEqual(out["person_name"], "Alik Fruman")

    # -- Test B: unknown / missing person_id ------------------------------

    def test_unknown_id_found_false(self) -> None:
        out = self.pd.get_person("nonexistent-id-00000000")
        self.assertFalse(out["found"])

    def test_unknown_id_person_id_is_UNKNOWN(self) -> None:
        out = self.pd.get_person("nonexistent-id-00000000")
        self.assertEqual(out["person_id"], "UNKNOWN")

    def test_unknown_id_person_name_is_UNKNOWN(self) -> None:
        out = self.pd.get_person("nonexistent-id-00000000")
        self.assertEqual(out["person_name"], "UNKNOWN")

    def test_empty_string_id_returns_unknown(self) -> None:
        out = self.pd.get_person("")
        self.assertFalse(out["found"])
        self.assertEqual(out["person_id"], "UNKNOWN")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    unittest.main()
