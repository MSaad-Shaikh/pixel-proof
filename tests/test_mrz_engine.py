"""
tests/test_mrz_engine.py
========================
Unit tests for the MRZ validation engine (mrz_engine.py).

Test coverage:
  1. Fully valid TD3 MRZ -- all checks must pass.
  2. Broken date-of-birth check digit -- DOB check must fail.
  3. Altered document number -- document number check must fail.

TD3 reference passport (synthetic, non-real data):
  Line 1: P<GBRSMITH<<JOHN<JAMES<<<<<<<<<<<<<<<<<<<<<<
  Line 2: 1234567897GBR8001014M2501016<<<<<<<<<<<<<<<6

Check digit computation (all synthetic):
  Document number: 123456789 -> CD=7
  DOB:             800101    -> CD=4
  Expiry:          250101    -> CD=6
  Personal number: <<<<<<<<<<<<<<  (all fillers) -> CD=0
  Composite:       1234567897800101425010163 + 14x< + 0 -> CD=6

We verify computations in test setup comments.
"""

import sys
import os
import pytest

# Ensure the parent directory is on the path so we can import mrz_engine
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mrz_engine import (
    compute_check_digit,
    validate_mrz,
    parse_and_validate,
    MRZResult,
    FieldResult,
)


# ---------------------------------------------------------------------------
# Synthetic TD3 reference data
# ---------------------------------------------------------------------------
# We construct a valid TD3 MRZ pair and use compute_check_digit to derive the
# correct check digits programmatically (avoids hardcoding errors).

DOCUMENT_NUMBER = "123456789"
DOB = "800101"
EXPIRY = "250101"
PERSONAL_BODY = "<<<<<<<<<<<<<<"  # 14 filler chars
NATIONALITY = "GBR"
SEX = "M"

CD_DOC = compute_check_digit(DOCUMENT_NUMBER)        # expected: 7
CD_DOB = compute_check_digit(DOB)                    # expected: 4
CD_EXPIRY = compute_check_digit(EXPIRY)              # depends on EXPIRY
CD_PERSONAL = compute_check_digit(PERSONAL_BODY)    # expected: 0

# Composite body per ICAO 9303 Part 4 (excludes nationality and sex):
# doc_number(9) + cd_doc(1) + dob(6) + cd_dob(1) + expiry(6)
# + cd_expiry(1) + personal(14) + cd_personal(1) = 39 chars
COMPOSITE_BODY = (
    DOCUMENT_NUMBER
    + str(CD_DOC)
    + DOB
    + str(CD_DOB)
    + EXPIRY
    + str(CD_EXPIRY)
    + PERSONAL_BODY
    + str(CD_PERSONAL)
)
CD_COMPOSITE = compute_check_digit(COMPOSITE_BODY)   # independent checksum

# Build line 2 with CORRECT ICAO 9303 Part 4 Table 11 offsets (0-indexed):
# [0-8]  doc_number [9] cd_doc [10-12] nationality [13-18] dob [19] cd_dob
# [20]   sex [21-26] expiry [27] cd_expiry [28-41] personal(14) [42] cd_personal
# [43]   composite CD
LINE2 = (
    DOCUMENT_NUMBER          # [0-8]   (9 chars)
    + str(CD_DOC)            # [9]     (1 char)
    + NATIONALITY            # [10-12] (3 chars)
    + DOB                    # [13-18] (6 chars)
    + str(CD_DOB)            # [19]    (1 char)
    + SEX                    # [20]    (1 char)
    + EXPIRY                 # [21-26] (6 chars)
    + str(CD_EXPIRY)         # [27]    (1 char)
    + PERSONAL_BODY          # [28-41] (14 chars)
    + str(CD_PERSONAL)       # [42]    (1 char)
    + str(CD_COMPOSITE)      # [43]    (1 char)
)
assert len(LINE2) == 44, f"LINE2 length error: {len(LINE2)}"

LINE1 = "P<GBRSMITH<<JOHN<JAMES<<<<<<<<<<<<<<<<<<<<<<" 
assert len(LINE1) == 44, f"LINE1 length error: {len(LINE1)}"

VALID_MRZ_TEXT = LINE1 + "\n" + LINE2


# ---------------------------------------------------------------------------
# Test 1: Fully valid TD3 MRZ
# ---------------------------------------------------------------------------

class TestValidMRZ:
    """All check digits are correct; status should be OK."""

    def test_status_is_ok(self):
        result = validate_mrz(LINE1, LINE2)
        assert result.status == "OK", (
            f"Expected OK, got {result.status}. "
            f"Failures: {result.failed_explanations}"
        )

    def test_all_checks_passed(self):
        result = validate_mrz(LINE1, LINE2)
        assert result.all_checks_passed is True

    def test_no_failed_explanations(self):
        result = validate_mrz(LINE1, LINE2)
        assert result.failed_explanations == []

    def test_document_number_check_passes(self):
        result = validate_mrz(LINE1, LINE2)
        assert result.document_number_check is not None
        assert result.document_number_check.passed is True

    def test_dob_check_passes(self):
        result = validate_mrz(LINE1, LINE2)
        assert result.dob_check is not None
        assert result.dob_check.passed is True

    def test_expiry_check_passes(self):
        result = validate_mrz(LINE1, LINE2)
        assert result.expiry_check is not None
        assert result.expiry_check.passed is True

    def test_personal_number_check_passes(self):
        result = validate_mrz(LINE1, LINE2)
        assert result.personal_number_check is not None
        assert result.personal_number_check.passed is True

    def test_composite_check_passes(self):
        result = validate_mrz(LINE1, LINE2)
        assert result.composite_check is not None
        assert result.composite_check.passed is True

    def test_parsed_fields(self):
        result = validate_mrz(LINE1, LINE2)
        assert result.surname == "SMITH"
        assert result.given_names == "JOHN JAMES"
        assert result.document_number == "123456789"
        assert result.nationality == "GBR"
        assert result.date_of_birth == DOB
        assert result.expiration_date == EXPIRY

    def test_parse_and_validate_helper(self):
        """End-to-end via parse_and_validate with raw text input."""
        result = parse_and_validate(VALID_MRZ_TEXT)
        assert result.status == "OK"

    def test_compute_check_digit_known_values(self):
        """Spot-check compute_check_digit against manually verified values."""
        # 'SMITH' -> S=28, M=22, I=18, T=29, H=17
        # weights: 7,3,1,7,3
        # products: 196,66,18,203,51 -> sum=534 -> 534%10=4
        assert compute_check_digit("SMITH") == 4


# ---------------------------------------------------------------------------
# Test 2: Broken date-of-birth check digit
# ---------------------------------------------------------------------------

class TestBrokenDOBCheck:
    """Flip one digit in the DOB check digit; only DOB check should fail."""

    @staticmethod
    def _make_bad_dob_line2() -> str:
        """Replace the correct DOB check digit with a wrong one.

        In the correct ICAO TD3 layout, cd_dob is at position [19].
        """
        wrong_cd = str((CD_DOB + 1) % 10)
        bad = (
            LINE2[:19]      # everything up to (not including) cd_dob
            + wrong_cd      # corrupted DOB check digit at position [19]
            + LINE2[20:]    # rest unchanged (sex, expiry, etc.)
        )
        # Composite check will also fail because composite includes cd_dob
        return bad

    def test_dob_check_fails(self):
        bad_line2 = self._make_bad_dob_line2()
        result = validate_mrz(LINE1, bad_line2)
        assert result.dob_check is not None
        assert result.dob_check.passed is False

    def test_status_is_checksum_fail(self):
        bad_line2 = self._make_bad_dob_line2()
        result = validate_mrz(LINE1, bad_line2)
        assert result.status == "CHECKSUM_FAIL"

    def test_failure_explanation_mentions_dob(self):
        bad_line2 = self._make_bad_dob_line2()
        result = validate_mrz(LINE1, bad_line2)
        combined = " ".join(result.failed_explanations).lower()
        assert "date of birth" in combined, (
            f"Expected 'date of birth' in failure explanation; got: {result.failed_explanations}"
        )

    def test_document_number_check_unaffected(self):
        """Altering DOB check digit must not affect the document number check."""
        bad_line2 = self._make_bad_dob_line2()
        result = validate_mrz(LINE1, bad_line2)
        assert result.document_number_check is not None
        assert result.document_number_check.passed is True


# ---------------------------------------------------------------------------
# Test 3: Altered document number
# ---------------------------------------------------------------------------

class TestAlteredDocumentNumber:
    """Change one character in the document number field; doc check should fail."""

    @staticmethod
    def _make_bad_docnum_line2() -> str:
        """Replace first char of document number to create a checksum mismatch."""
        # Original LINE2[0] = '1'; replace with '2' to break the field checksum.
        bad = "2" + LINE2[1:]
        return bad

    def test_document_number_check_fails(self):
        bad_line2 = self._make_bad_docnum_line2()
        result = validate_mrz(LINE1, bad_line2)
        assert result.document_number_check is not None
        assert result.document_number_check.passed is False

    def test_status_is_checksum_fail(self):
        bad_line2 = self._make_bad_docnum_line2()
        result = validate_mrz(LINE1, bad_line2)
        assert result.status == "CHECKSUM_FAIL"

    def test_failure_explanation_mentions_document_number(self):
        bad_line2 = self._make_bad_docnum_line2()
        result = validate_mrz(LINE1, bad_line2)
        combined = " ".join(result.failed_explanations).lower()
        assert "document number" in combined, (
            f"Expected 'document number' in explanations; got: {result.failed_explanations}"
        )

    def test_dob_check_unaffected(self):
        """Altering the document number must not affect DOB check digit."""
        bad_line2 = self._make_bad_docnum_line2()
        result = validate_mrz(LINE1, bad_line2)
        assert result.dob_check is not None
        assert result.dob_check.passed is True


# ---------------------------------------------------------------------------
# Test 4: MRZ_UNREADABLE handling
# ---------------------------------------------------------------------------

class TestMRZUnreadable:
    """Ensure malformed or empty input yields MRZ_UNREADABLE, not an exception."""

    def test_empty_string_returns_unreadable(self):
        result = parse_and_validate("")
        assert result.status == "MRZ_UNREADABLE"

    def test_short_lines_return_unreadable(self):
        result = validate_mrz("TOOSHORT", "ALSOSHORT")
        assert result.status == "MRZ_UNREADABLE"

    def test_invalid_chars_return_unreadable(self):
        bad1 = "P<GBR" + "?" * 39   # contains '?' which is invalid
        bad2 = LINE2
        result = validate_mrz(bad1, bad2)
        assert result.status == "MRZ_UNREADABLE"

    def test_parse_and_validate_garbage(self):
        result = parse_and_validate("this is not an mrz at all")
        assert result.status == "MRZ_UNREADABLE"
        assert len(result.failed_explanations) > 0
