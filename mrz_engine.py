"""
mrz_engine.py  ICAO 9303 TD3 MRZ Validation Engine
=====================================================
Supports TD3 format (2 lines x 44 characters) -- standard machine-readable passport.

NOTE: TD1 (3-line, 30-char, identity-card MRZ) requires a separate parser
with different field offsets and a different composite checksum formula.
That format is OUT OF SCOPE for this prototype.

References:
  . ICAO Doc 9303 Part 3 (MRTDs -- Specifications Common to all MRTDs)
  . ICAO Doc 9303 Part 4 (Specifications for Machine Readable Passports)

Character-value table:
  0 - 9  -> 0-9
  A - Z  -> 10-35
  <      -> 0
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

TD3_LINE_LENGTH: int = 44
TD3_LINES: int = 2

TD1_LINE_LENGTH: int = 30   # National ID cards, residence permits
TD1_LINES: int = 3

TD2_LINE_LENGTH: int = 36   # Some visas, older EU travel docs
TD2_LINES: int = 2

_WEIGHTS: tuple = (7, 3, 1)

# Common ISO 3166-1 alpha-3 and numeric country code mappings for MRTD display
ISO_COUNTRY_CODES: dict[str, str] = {
    "AUS": "Australia",
    "036": "Australia",
    "405": "Australia (Specimen/Numeric)",
    "IND": "India",
    "356": "India",
    "USA": "United States",
    "840": "United States",
    "GBR": "United Kingdom",
    "826": "United Kingdom",
    "CAN": "Canada",
    "124": "Canada",
    "FRA": "France",
    "250": "France",
    "DEU": "Germany",
    "276": "Germany",
    "JPN": "Japan",
    "392": "Japan",
    "CHN": "China",
    "156": "China",
    "SGP": "Singapore",
    "702": "Singapore",
    "NZL": "New Zealand",
    "554": "New Zealand",
    "ZAF": "South Africa",
    "710": "South Africa",
    "ARE": "United Arab Emirates",
    "784": "United Arab Emirates",
    "UTO": "Utopia (Test Document)",
}


def lookup_country(code: str) -> str:
    """Format and lookup an ISO 3166-1 country code (alpha-3 or numeric).

    Args:
        code: 3-character string (e.g. 'AUS', '405', 'IND').

    Returns:
        Formatted string (e.g. 'AUS (Australia)' or '405 (Australia)').
    """
    clean = code.strip().upper()
    if clean in ISO_COUNTRY_CODES:
        return f"{clean} ({ISO_COUNTRY_CODES[clean]})"
    return clean


@dataclass
class FieldResult:
    """Validation result for a single MRZ field."""
    value: str
    expected_check_digit: str
    computed_check_digit: int
    passed: bool
    explanation: str


@dataclass
class MRZResult:
    """Full structured result from MRZ parsing and validation (TD1/TD2/TD3)."""
    status: str
    raw_line1: str = ""
    raw_line2: str = ""
    raw_line3: str = ""           # TD1 only
    mrz_format: str = "TD3"       # "TD1", "TD2", or "TD3"
    document_type: str = ""
    issuing_state: str = ""
    surname: str = ""
    given_names: str = ""
    document_number: str = ""
    nationality: str = ""
    date_of_birth: str = ""
    sex: str = ""
    expiration_date: str = ""
    personal_number: str = ""
    document_number_check: Optional[FieldResult] = None
    dob_check: Optional[FieldResult] = None
    expiry_check: Optional[FieldResult] = None
    personal_number_check: Optional[FieldResult] = None
    composite_check: Optional[FieldResult] = None
    all_checks_passed: bool = False
    failed_explanations: list = field(default_factory=list)



def _char_value(ch: str) -> int:
    """Return the ICAO 9303 numeric value for a single MRZ character.

    Mapping: 0-9 -> 0-9, A-Z -> 10-35, < -> 0

    Args:
        ch: A single uppercase character or '<'.

    Returns:
        Integer value 0-35.

    Raises:
        ValueError: If ch is not a valid MRZ character.
    """
    if ch == "<":
        return 0
    if "0" <= ch <= "9":
        return int(ch)
    if "A" <= ch <= "Z":
        return ord(ch) - ord("A") + 10
    raise ValueError(f"Invalid MRZ character: {ch!r}")


def compute_check_digit(field_str: str) -> int:
    """Compute the ICAO 9303 weighted mod-10 check digit for a field string.

    Algorithm (ICAO 9303 Part 3 section 4.9):
      1. Map each character to its numeric value via _char_value().
      2. Multiply by the repeating weight sequence [7, 3, 1, 7, 3, 1, ...].
      3. Sum all products.
      4. Result = sum mod 10.

    Args:
        field_str: The MRZ field string to check (uppercase, may contain '<').

    Returns:
        Integer check digit in range 0-9.
    """
    total = 0
    for i, ch in enumerate(field_str):
        total += _char_value(ch) * _WEIGHTS[i % 3]
    return total % 10


def _validate_field(value: str, check_char: str, label: str) -> FieldResult:
    """Validate one named MRZ field against its check digit character.

    Args:
        value:      The field body (without the check digit character).
        check_char: The single check-digit character from the MRZ string.
        label:      Human-readable field name for error messages.

    Returns:
        A FieldResult with pass/fail and explanation.
    """
    computed = compute_check_digit(value)
    try:
        expected = int(check_char)
    except ValueError:
        return FieldResult(
            value=value,
            expected_check_digit=check_char,
            computed_check_digit=computed,
            passed=False,
            explanation=f"{label}: check digit character {check_char!r} is not numeric.",
        )

    passed = computed == expected
    explanation = (
        ""
        if passed
        else f"{label}: check digit expected {expected}, computed {computed}."
    )
    return FieldResult(
        value=value,
        expected_check_digit=check_char,
        computed_check_digit=computed,
        passed=passed,
        explanation=explanation,
    )


def _extract_td3_fields(line1: str, line2: str) -> dict:
    """Extract raw field substrings from the two TD3 MRZ lines.

    Field offsets per ICAO 9303 Part 4 Table 11 (0-indexed):
      Line 1:
        [0-1]  Document type (2)
        [2-4]  Issuing state (3)
        [5-43] Primary identifier: surname<<given names (39)

      Line 2:
        [0-8]   Document number (9)
        [9]     CD document number (1)
        [10-12] Nationality (3)
        [13-18] Date of birth YYMMDD (6)
        [19]    CD date of birth (1)
        [20]    Sex M/F/< (1)
        [21-26] Date of expiry YYMMDD (6)
        [27]    CD date of expiry (1)
        [28-41] Personal number / optional data (14)
        [42]    CD personal number (1)
        [43]    Composite check digit (1)

    Composite body (ICAO 9303 Part 4 section 4.2.3) = 39 chars:
      doc_number(9) + cd_doc(1) + dob(6) + cd_dob(1) + expiry(6)
      + cd_expiry(1) + personal(14) + cd_personal(1)
      NOTE: Nationality and sex are intentionally excluded from composite.

    Args:
        line1: First MRZ line, exactly 44 characters.
        line2: Second MRZ line, exactly 44 characters.

    Returns:
        Dict of named raw field strings.
    """
    return {
        "doc_type":        line1[0:2],
        "issuing_state":   line1[2:5],
        "primary_id":      line1[5:44],
        "document_number": line2[0:9],
        "cd_doc":          line2[9],
        "nationality":     line2[10:13],
        "dob":             line2[13:19],
        "cd_dob":          line2[19],
        "sex":             line2[20],
        "expiry":          line2[21:27],
        "cd_expiry":       line2[27],
        "personal_body":   line2[28:42],
        "cd_personal":     line2[42],
        "composite_cd":    line2[43],
        # Composite body = 39 chars (excludes nationality and sex)
        "composite_body":  (
            line2[0:10]    # doc_number(9) + cd_doc(1)
            + line2[13:20]  # dob(6) + cd_dob(1)
            + line2[21:28]  # expiry(6) + cd_expiry(1)
            + line2[28:43]  # personal_body(14) + cd_personal(1)
        ),
    }


def _parse_names(primary_id: str) -> tuple:
    """Parse the primary identifier field into surname and given names.

    The primary identifier uses '<<' as delimiter between surname and
    given names, and '<' as word separator within each part.

    Args:
        primary_id: The 39-character primary identifier string from line 1.

    Returns:
        Tuple (surname, given_names) with '<' replaced by spaces.
    """
    parts = primary_id.split("<<", 1)
    surname = parts[0].replace("<", " ").strip()
    given_names = parts[1].replace("<", " ").strip() if len(parts) > 1 else ""
    return surname, given_names


def clean_mrz_ocr_line(line: str) -> str:
    """Clean common OCR errors in MRZ lines (spaces, misrecognized angle brackets)."""
    cleaned = line.strip().upper()
    # Replace common OCR misreads for '<'
    substitutions = {
        "«": "<",
        "»": "<",
        "(": "<",
        ")": "<",
        "{": "<",
        "}": "<",
        "[": "<",
        "]": "<",
        "|": "<",
    }
    for old, new in substitutions.items():
        cleaned = cleaned.replace(old, new)
    # Remove spaces around and between '<' characters
    cleaned = re.sub(r"\s*<\s*", "<", cleaned)
    # Remove remaining whitespace
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned


def _extract_td1_fields(line1: str, line2: str, line3: str) -> dict:
    """Extract raw field substrings from a three-line TD1 MRZ (30 chars each).

    Field offsets per ICAO 9303 Part 5 (national ID cards):
      Line 1: [0-1] doc type, [2-4] issuing state, [5-13] doc number,
              [14] CD doc, [15-29] optional data 1
      Line 2: [0-5] DOB, [6] CD dob, [7] sex, [8-13] expiry, [14] CD expiry,
              [15-17] nationality, [18-28] optional data 2, [29] composite CD
      Line 3: [0-29] primary identifier (surname<<given names)
    """
    # Composite body for TD1: doc(5..14) + optional1 + dob(0..14) + optional2(15..29)
    composite_body = line1[5:30] + line2[0:7] + line2[8:15] + line2[18:29]
    return {
        "doc_type":        line1[0:2],
        "issuing_state":   line1[2:5],
        "document_number": line1[5:14],
        "cd_doc":          line1[14],
        "optional1":       line1[15:30],
        "dob":             line2[0:6],
        "cd_dob":          line2[6],
        "sex":             line2[7],
        "expiry":          line2[8:14],
        "cd_expiry":       line2[14],
        "nationality":     line2[15:18],
        "optional2":       line2[18:29],
        "composite_cd":    line2[29],
        "primary_id":      line3[0:30],
        "composite_body":  composite_body,
    }


def _extract_td2_fields(line1: str, line2: str) -> dict:
    """Extract raw field substrings from a two-line TD2 MRZ (36 chars each).

    Field offsets per ICAO 9303 Part 6:
      Line 1: [0-1] doc type, [2-4] issuing state, [5-35] primary identifier
      Line 2: [0-8] doc number, [9] CD doc, [10-12] nationality, [13-18] DOB,
              [19] CD dob, [20] sex, [21-26] expiry, [27] CD expiry,
              [28-34] optional, [35] composite CD
    """
    composite_body = (
        line2[0:10]   # doc_number + CD
        + line2[13:20]  # dob + CD
        + line2[21:28]  # expiry + CD
        + line2[28:35]  # optional + composite_cd
    )
    return {
        "doc_type":        line1[0:2],
        "issuing_state":   line1[2:5],
        "primary_id":      line1[5:36],
        "document_number": line2[0:9],
        "cd_doc":          line2[9],
        "nationality":     line2[10:13],
        "dob":             line2[13:19],
        "cd_dob":          line2[19],
        "sex":             line2[20],
        "expiry":          line2[21:27],
        "cd_expiry":       line2[27],
        "optional":        line2[28:35],
        "composite_cd":    line2[35],
        "composite_body":  composite_body,
    }


def validate_mrz_td1(line1: str, line2: str, line3: str) -> MRZResult:
    """Parse and validate a TD1 MRZ (three 30-character lines).

    Used for national identity cards (e.g. Aadhaar-linked travel docs,
    EU national ID cards, residence permits).

    Args:
        line1: First MRZ line (30 chars).
        line2: Second MRZ line (30 chars).
        line3: Third MRZ line (30 chars).

    Returns:
        MRZResult with format="TD1".
    """
    result = MRZResult(
        status="MRZ_UNREADABLE", raw_line1=line1, raw_line2=line2, raw_line3=line3,
        mrz_format="TD1",
    )
    valid_chars = re.compile(r"^[A-Z0-9<]+$")
    for i, (ln, expected_len) in enumerate([(line1, TD1_LINE_LENGTH),
                                             (line2, TD1_LINE_LENGTH),
                                             (line3, TD1_LINE_LENGTH)], 1):
        if len(ln) != expected_len or not valid_chars.match(ln):
            result.failed_explanations = [
                f"TD1 MRZ line {i} must be exactly {TD1_LINE_LENGTH} valid characters. "
                f"Got length {len(ln)}."
            ]
            return result

    f = _extract_td1_fields(line1, line2, line3)
    surname, given_names = _parse_names(f["primary_id"])

    result.document_type   = f["doc_type"].replace("<", "")
    result.issuing_state   = f["issuing_state"].replace("<", "")
    result.surname         = surname
    result.given_names     = given_names
    result.document_number = f["document_number"].replace("<", "")
    result.nationality     = f["nationality"].replace("<", "")
    result.date_of_birth   = f["dob"]
    result.expiration_date = f["expiry"]
    result.personal_number = f.get("optional2", "").replace("<", "")
    result.sex             = f["sex"]

    result.document_number_check = _validate_field(f["document_number"], f["cd_doc"], "Document number")
    result.dob_check             = _validate_field(f["dob"],             f["cd_dob"],    "Date of birth")
    result.expiry_check          = _validate_field(f["expiry"],          f["cd_expiry"], "Expiration date")
    result.composite_check       = _validate_field(f["composite_body"],  f["composite_cd"], "Composite")

    checks   = [result.document_number_check, result.dob_check,
                result.expiry_check, result.composite_check]
    failures = [c.explanation for c in checks if not c.passed]
    result.failed_explanations = failures
    result.all_checks_passed   = len(failures) == 0
    result.status              = "OK" if result.all_checks_passed else "CHECKSUM_FAIL"
    logger.info("TD1 MRZ validation complete — status=%s", result.status)
    return result


def validate_mrz_td2(line1: str, line2: str) -> MRZResult:
    """Parse and validate a TD2 MRZ (two 36-character lines).

    Used for some visas and older EU travel documents.

    Args:
        line1: First MRZ line (36 chars).
        line2: Second MRZ line (36 chars).

    Returns:
        MRZResult with format="TD2".
    """
    result = MRZResult(
        status="MRZ_UNREADABLE", raw_line1=line1, raw_line2=line2,
        mrz_format="TD2",
    )
    valid_chars = re.compile(r"^[A-Z0-9<]+$")
    for i, ln in enumerate([line1, line2], 1):
        if len(ln) != TD2_LINE_LENGTH or not valid_chars.match(ln):
            result.failed_explanations = [
                f"TD2 MRZ line {i} must be exactly {TD2_LINE_LENGTH} valid characters. "
                f"Got length {len(ln)}."
            ]
            return result

    f = _extract_td2_fields(line1, line2)
    surname, given_names = _parse_names(f["primary_id"])

    result.document_type   = f["doc_type"].replace("<", "")
    result.issuing_state   = f["issuing_state"].replace("<", "")
    result.surname         = surname
    result.given_names     = given_names
    result.document_number = f["document_number"].replace("<", "")
    result.nationality     = f["nationality"].replace("<", "")
    result.date_of_birth   = f["dob"]
    result.expiration_date = f["expiry"]
    result.personal_number = f.get("optional", "").replace("<", "")
    result.sex             = f["sex"]

    result.document_number_check = _validate_field(f["document_number"], f["cd_doc"],      "Document number")
    result.dob_check             = _validate_field(f["dob"],             f["cd_dob"],      "Date of birth")
    result.expiry_check          = _validate_field(f["expiry"],          f["cd_expiry"],   "Expiration date")
    result.composite_check       = _validate_field(f["composite_body"],  f["composite_cd"], "Composite")

    checks   = [result.document_number_check, result.dob_check,
                result.expiry_check, result.composite_check]
    failures = [c.explanation for c in checks if not c.passed]
    result.failed_explanations = failures
    result.all_checks_passed   = len(failures) == 0
    result.status              = "OK" if result.all_checks_passed else "CHECKSUM_FAIL"
    logger.info("TD2 MRZ validation complete — status=%s", result.status)
    return result


def extract_mrz_from_text(raw_text: str) -> Optional[tuple]:
    """Locate an MRZ block inside arbitrary OCR text.

    Auto-detects TD1 (3×30), TD2 (2×36), or TD3 (2×44) format by
    examining candidate line lengths.

    Returns:
        For TD3/TD2: tuple (line1, line2, format_str)
        For TD1: tuple (line1, line2, line3, format_str)
        Or None if no MRZ block is found.
    """
    raw_lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    cleaned_lines = [clean_mrz_ocr_line(ln) for ln in raw_lines]
    valid_chars = re.compile(r"^[A-Z0-9<]+$")

    # Stage 1: Exact TD3 (44 char) lines
    td3_exact = [ln for ln in cleaned_lines
                 if len(ln) == TD3_LINE_LENGTH and valid_chars.match(ln)]
    if len(td3_exact) >= 2:
        return td3_exact[-2], td3_exact[-1], "TD3"

    # Stage 2: Exact TD1 (30 char) lines — must have at least 3
    td1_exact = [ln for ln in cleaned_lines
                 if len(ln) == TD1_LINE_LENGTH and valid_chars.match(ln)]
    if len(td1_exact) >= 3:
        return td1_exact[-3], td1_exact[-2], td1_exact[-1], "TD1"

    # Stage 3: Exact TD2 (36 char) lines
    td2_exact = [ln for ln in cleaned_lines
                 if len(ln) == TD2_LINE_LENGTH and valid_chars.match(ln)]
    if len(td2_exact) >= 2:
        return td2_exact[-2], td2_exact[-1], "TD2"

    # Stage 4: Fuzzy fallback — near-44-char candidates (AI-generated passports etc.)
    mrz_like = [ln for ln in cleaned_lines
                if (ln.startswith("P<") or ln.startswith("I<") or ln.count("<") >= 2)
                and len(ln) >= 28]
    if len(mrz_like) >= 2:
        return mrz_like[-2], mrz_like[-1], "TD3"  # assume TD3 for fuzzy

    return None


def validate_mrz(line1: str, line2: str) -> MRZResult:
    """Parse and validate a TD3 MRZ (two 44-character lines).

    Performs:
      1. Length and character-set sanity checks.
      2. Per-field check-digit validation (document number, DOB, expiry,
         personal number).
      3. Composite check-digit validation per ICAO 9303 Part 4 section 4.2.3.

    Args:
        line1: First MRZ line (44 chars).
        line2: Second MRZ line (44 chars).

    Returns:
        MRZResult with full per-field and composite results.
    """
    result = MRZResult(status="MRZ_UNREADABLE", raw_line1=line1, raw_line2=line2)

    line1 = line1.strip().upper()
    line2 = line2.strip().upper()

    valid_chars = re.compile(r"^[A-Z0-9<]+$")
    if (
        len(line1) != TD3_LINE_LENGTH
        or len(line2) != TD3_LINE_LENGTH
        or not valid_chars.match(line1)
        or not valid_chars.match(line2)
    ):
        result.status = "MRZ_UNREADABLE"
        result.failed_explanations = [
            f"MRZ lines must each be exactly {TD3_LINE_LENGTH} valid characters "
            f"(A-Z, 0-9, '<'). Got lengths {len(line1)} / {len(line2)}."
        ]
        logger.warning("MRZ sanity check failed: length or charset error.")
        return result

    f = _extract_td3_fields(line1, line2)

    surname, given_names = _parse_names(f["primary_id"])

    result.document_type   = f["doc_type"].replace("<", "")
    result.issuing_state   = f["issuing_state"].replace("<", "")
    result.surname         = surname
    result.given_names     = given_names
    result.document_number = f["document_number"].replace("<", "")
    result.date_of_birth   = f["dob"]
    result.expiration_date = f["expiry"]
    result.personal_number = f["personal_body"].replace("<", "")
    result.nationality     = f["nationality"].replace("<", "")
    result.sex             = f["sex"]

    result.document_number_check = _validate_field(
        f["document_number"], f["cd_doc"], "Document number"
    )
    result.dob_check = _validate_field(
        f["dob"], f["cd_dob"], "Date of birth"
    )
    result.expiry_check = _validate_field(
        f["expiry"], f["cd_expiry"], "Expiration date"
    )
    result.personal_number_check = _validate_field(
        f["personal_body"], f["cd_personal"], "Personal number"
    )
    result.composite_check = _validate_field(
        f["composite_body"], f["composite_cd"], "Composite"
    )

    checks = [
        result.document_number_check,
        result.dob_check,
        result.expiry_check,
        result.personal_number_check,
        result.composite_check,
    ]
    failures = [c.explanation for c in checks if not c.passed]
    result.failed_explanations = failures
    result.all_checks_passed = len(failures) == 0
    result.status = "OK" if result.all_checks_passed else "CHECKSUM_FAIL"

    logger.info(
        "MRZ validation complete -- status=%s, failures=%d",
        result.status,
        len(failures),
    )
    return result


def parse_and_validate(ocr_text: str) -> MRZResult:
    """End-to-end helper: extract MRZ lines from OCR text then validate.

    Auto-detects TD1 (3×30 chars), TD2 (2×36 chars), or TD3 (2×44 chars)
    format and dispatches to the appropriate validator.

    Args:
        ocr_text: Raw string from an OCR engine or direct MRZ reader.

    Returns:
        A MRZResult. Status is 'MRZ_UNREADABLE' if extraction fails;
        'CHECKSUM_FAIL' or 'OK' otherwise. mrz_format is set to the
        detected format ("TD1", "TD2", or "TD3").
    """
    try:
        result_tuple = extract_mrz_from_text(ocr_text)
        if result_tuple is None:
            logger.warning("No MRZ lines found in OCR text.")
            return MRZResult(
                status="MRZ_UNREADABLE",
                failed_explanations=[
                    "Could not locate a valid MRZ block in the extracted text. "
                    "Tried TD3 (44×2), TD1 (30×3), and TD2 (36×2) formats. "
                    "Check image quality and orientation."
                ],
            )

        mrz_format = result_tuple[-1]  # last element is always the format string

        if mrz_format == "TD1" and len(result_tuple) == 4:
            line1, line2, line3, _ = result_tuple
            return validate_mrz_td1(line1, line2, line3)
        elif mrz_format == "TD2" and len(result_tuple) == 3:
            line1, line2, _ = result_tuple
            return validate_mrz_td2(line1, line2)
        else:
            # TD3 (or fuzzy fallback treated as TD3)
            line1, line2 = result_tuple[0], result_tuple[1]
            return validate_mrz(line1, line2)

    except Exception as exc:
        logger.error("Unexpected error in parse_and_validate: %s", exc, exc_info=True)
        return MRZResult(
            status="MRZ_UNREADABLE",
            failed_explanations=[f"Internal MRZ parsing error: {exc}"],
        )


def format_date(yymmdd: str) -> str:
    """Convert a YYMMDD date string to a human-readable format.

    Applies the ICAO convention: years 00-30 -> 2000-2030;
    years 31-99 -> 1931-1999.

    Args:
        yymmdd: Six-character date string in YYMMDD format.

    Returns:
        Human-readable date string, or the raw value if parsing fails.
    """
    try:
        yy = int(yymmdd[0:2])
        mm = int(yymmdd[2:4])
        dd = int(yymmdd[4:6])
        year = 2000 + yy if yy <= 30 else 1900 + yy
        return datetime(year, mm, dd).strftime("%d %b %Y")
    except (ValueError, IndexError):
        return yymmdd
