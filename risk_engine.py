"""
risk_engine.py  Composite Risk Score Aggregator
================================================
Combines four sub-checks into a single 0-100 risk score with
category labels (LOW RISK / SUSPICIOUS / CRITICAL RISK).

Sub-check weights:
  MRZ checksum failure     35%
  ELA mean intensity       25%
  Biometric cosine dist    25%
  Text consistency         15%

Normalisation (each raw value -> 0-100 risk contribution):
  MRZ check (bool):
    0   if all checks passed  ->  risk = 0
    100 if any check failed   ->  risk = 100

  ELA mean_error (float, 0-255 scale post-scale):
    Clamped to [0, ELA_MAX_MEANINGFUL] then scaled to 0-100.
    ELA_MAX_MEANINGFUL = 50.0 (empirical; values above this almost
    certainly indicate either forgery or extreme re-compression).

  Biometric cosine distance (float, 0-1+):
    At distance 0   -> risk = 0   (identical faces)
    At distance >= ARCFACE_COSINE_THRESHOLD -> risk = 100
    Linear interpolation between 0 and threshold.
    Distance above threshold is clamped at 100.

  Text consistency (fuzzy match ratio, 0-100 from RapidFuzz):
    ratio 100 (perfect match) -> risk = 0
    ratio 0   (no match)      -> risk = 100
    Inversion: risk = 100 - ratio

Missing sub-checks:
  If a sub-check could not run (e.g. MRZ unreadable, no face detected,
  PaddleOCR failed), its raw score is marked INCOMPLETE.
  The final score is computed over available checks only, with weights
  renormalised proportionally. The UI also displays an INCOMPLETE badge.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Weight definitions (must sum to 1.0)
# ---------------------------------------------------------------------------
WEIGHT_MRZ:        float = 0.35
WEIGHT_ELA:        float = 0.25
WEIGHT_BIOMETRIC:  float = 0.25
WEIGHT_TEXT:       float = 0.15

# ---------------------------------------------------------------------------
# Normalisation constants
# ---------------------------------------------------------------------------
# ELA: mean errors above this value are treated as maximum-risk (100).
ELA_MAX_MEANINGFUL: float = 50.0

# Biometric: cosine distance at which we assign maximum risk (100).
# Import from biometrics to stay consistent with that module's threshold.
try:
    from biometrics import ARCFACE_COSINE_THRESHOLD as _BIO_THRESHOLD
    BIO_MAX_DISTANCE: float = _BIO_THRESHOLD
except ImportError:
    BIO_MAX_DISTANCE: float = 0.68

# ---------------------------------------------------------------------------
# Risk category thresholds
# ---------------------------------------------------------------------------
RISK_LOW_MAX:      int = 25   # 0-25  -> LOW RISK   (green)
RISK_SUSPICIOUS_MAX: int = 60  # 26-60 -> SUSPICIOUS  (yellow)
# 61-100 -> CRITICAL RISK (red)


def _normalize_mrz(passed: bool) -> float:
    """Normalise MRZ checksum result to a 0-100 risk contribution.

    Mapping:
      True  (all checks passed) -> 0.0
      False (any check failed)  -> 100.0

    Args:
        passed: True if all MRZ check digits are correct.

    Returns:
        Float risk score in range [0, 100].
    """
    return 0.0 if passed else 100.0


def _normalize_ela(mean_error: float) -> float:
    """Normalise ELA mean error to a 0-100 risk contribution.

    Linear map: 0.0 -> 0, ELA_MAX_MEANINGFUL -> 100.
    Values above ELA_MAX_MEANINGFUL are clamped at 100.

    Args:
        mean_error: Mean absolute pixel error from perform_ela().

    Returns:
        Float risk score in range [0, 100].
    """
    clamped = max(0.0, min(mean_error, ELA_MAX_MEANINGFUL))
    return (clamped / ELA_MAX_MEANINGFUL) * 100.0


def _normalize_biometric(cosine_distance: float) -> float:
    """Normalise ArcFace cosine distance to a 0-100 risk contribution.

    Two-zone mapping to reflect pass/fail semantics properly:
      - PASSED (distance <= threshold): mapped linearly to 0-40 risk.
        A near-perfect match (0.0) gives 0 risk; at the threshold it gives 40.
      - FAILED (distance > threshold): mapped linearly from 40-100 risk.
        Ensures a failed match is always above the SUSPICIOUS threshold.

    Args:
        cosine_distance: Cosine distance from verify_faces().

    Returns:
        Float risk score in range [0, 100].
    """
    clamped = max(0.0, min(cosine_distance, 1.0))
    if clamped <= BIO_MAX_DISTANCE:
        # Passed: scale 0..threshold → 0..40
        return (clamped / BIO_MAX_DISTANCE) * 40.0
    else:
        # Failed: scale threshold..1.0 → 40..100
        excess = clamped - BIO_MAX_DISTANCE
        max_excess = 1.0 - BIO_MAX_DISTANCE
        return 40.0 + (excess / max_excess) * 60.0 if max_excess > 0 else 100.0


def _normalize_text(fuzzy_ratio: float) -> float:
    """Normalise fuzzy-match ratio to a 0-100 risk contribution.

    RapidFuzz ratios are 0-100 where 100 = perfect match.
    We invert so that perfect match -> 0 risk, no match -> 100 risk.

    Args:
        fuzzy_ratio: Match ratio from RapidFuzz (0.0-100.0).

    Returns:
        Float risk score in range [0, 100].
    """
    return max(0.0, min(100.0 - fuzzy_ratio, 100.0))


def _normalize_dob_to_yymmdd(raw: str) -> str:
    """Normalize various date formats to YYMMDD for comparison with MRZ.

    Handles DD/MM/YYYY, DD.MM.YYYY, DD-MM-YYYY, DD MMM YYYY, and YYMMDD.

    Args:
        raw: Raw date string extracted by OCR.

    Returns:
        YYMMDD string if parseable, else the original cleaned string.
    """
    import re as _re
    raw = raw.strip().upper()

    # Already YYMMDD (6 digits)
    if _re.fullmatch(r"\d{6}", raw):
        return raw

    # DD MMM YYYY → YYMMDD
    month_map = {
        "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
        "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
        "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
    }
    m = _re.match(r"(\d{2})\s+([A-Z]{3})\s+(\d{4})", raw)
    if m:
        dd, mon, yyyy = m.group(1), m.group(2), m.group(3)
        mm = month_map.get(mon, "00")
        return yyyy[2:] + mm + dd

    # DD/MM/YYYY or DD.MM.YYYY or DD-MM-YYYY
    m = _re.match(r"(\d{2})[.\-/](\d{2})[.\-/](\d{2,4})", raw)
    if m:
        dd, mm, yy = m.group(1), m.group(2), m.group(3)
        if len(yy) == 4:
            yy = yy[2:]
        return yy + mm + dd

    return raw


def extract_viz_fields_with_ocr(image_path: str) -> dict:
    """OCR the visible printed (VIZ) fields on a document image.

    Attempts to extract name, date-of-birth, and document number from
    the human-readable zone above the MRZ using EasyOCR (primary) or
    PaddleOCR (fallback). Uses None to mark unavailable fields so
    compute_text_consistency can distinguish "not found" from "empty match".

    Args:
        image_path: Path to the document image.

    Returns:
        Dict with keys 'name', 'dob', 'doc_number'.
        Values are str if found, or None if not found by OCR.
    """
    viz: dict = {"name": None, "dob": None, "doc_number": None}
    all_text: str = ""

    # Primary: EasyOCR (already installed, cached per session in app.py)
    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        lines = reader.readtext(image_path, detail=0)
        all_text = " ".join(lines).upper()
        logger.info("VIZ OCR: EasyOCR read %d text regions.", len(lines))
    except Exception as exc:
        logger.warning("EasyOCR unavailable for VIZ extraction: %s", exc)

    # Fallback: PaddleOCR
    if not all_text:
        try:
            from paddleocr import PaddleOCR
            ocr = PaddleOCR(use_angle_cls=True, lang="en")
            result = ocr.ocr(image_path, cls=True)
            if result and result[0]:
                all_text = " ".join(
                    line[1][0] for line in result[0] if line and line[1]
                ).upper()
                logger.info("VIZ OCR: PaddleOCR fallback extracted text.")
        except Exception as exc:
            logger.warning("PaddleOCR unavailable for VIZ extraction: %s", exc)

    if not all_text:
        logger.warning("VIZ OCR: no engine produced text; all fields unavailable.")
        return viz

    # Date of birth: DD/MM/YYYY, DD.MM.YYYY, DD-MM-YYYY, DD MMM YYYY, YYMMDD
    dob_match = re.search(
        r"\b(\d{2}[.\-/]\d{2}[.\-/]\d{2,4}|\d{2}\s+[A-Z]{3}\s+\d{4}|\d{6})\b",
        all_text,
    )
    if dob_match:
        viz["dob"] = dob_match.group(0).strip()

    # Document number: 1-2 letter prefix + 6-9 digits, or plain 7-9 digits
    doc_match = re.search(r"\b([A-Z]{1,2}\d{6,9}|\d{7,9})\b", all_text)
    if doc_match:
        viz["doc_number"] = re.sub(r"[\s\-]", "", doc_match.group(0)).strip()

    # Name: longest run of alphabetic tokens excluding document label words
    tokens = re.findall(r"[A-Z]{2,}", all_text)
    stop = {
        "PASSPORT", "REPUBLIC", "INDIA", "NATIONALITY", "SURNAME",
        "GIVEN", "NAME", "DATE", "BIRTH", "EXPIRY", "PERSONAL",
        "DOCUMENT", "NUMBER", "SEX", "MALE", "FEMALE", "PLACE",
        "ISSUE", "OFFICER", "AUTHORITY", "TYPE", "CODE",
    }
    name_tokens = [t for t in tokens if t not in stop and len(t) > 2]
    if name_tokens:
        viz["name"] = " ".join(name_tokens[:4])

    # PII hygiene: log only availability, never actual field values
    logger.info(
        "VIZ field availability — name: %s, dob: %s, doc_number: %s",
        viz["name"] is not None,
        viz["dob"] is not None,
        viz["doc_number"] is not None,
    )
    return viz


def compute_text_consistency(
    viz_fields: dict,
    mrz_surname: str,
    mrz_given: str,
    mrz_dob: str,
    mrz_doc_number: str,
) -> dict:
    """Fuzzy-compare VIZ-OCR fields against MRZ-parsed fields.

    Field-aware: fields that are None (not found by OCR) are tracked as
    UNAVAILABLE and excluded from the overall_ratio calculation.
    Only fields that were actually read contribute to the score.

    Uses RapidFuzz token_sort_ratio for name and partial_ratio for
    DOB and document number. DOBs are normalized to YYMMDD before compare.

    Args:
        viz_fields:     Dict from extract_viz_fields_with_ocr().
                        None values mean the field was not found by OCR.
        mrz_surname:    Surname parsed from MRZ line 1.
        mrz_given:      Given names parsed from MRZ line 1.
        mrz_dob:        YYMMDD date-of-birth string from MRZ line 2.
        mrz_doc_number: Document number from MRZ line 2.

    Returns:
        Dict with keys:
          'name_ratio'       (float|None) - 0-100 score, or None if unavailable.
          'dob_ratio'        (float|None) - 0-100 score, or None if unavailable.
          'doc_ratio'        (float|None) - 0-100 score, or None if unavailable.
          'overall_ratio'    (float)      - Average over available fields only.
          'available'        (bool)       - True if at least one field compared.
          'fields_available' (dict)       - Per-field availability flags.
    """
    unavailable_result = {
        "name_ratio": None,
        "dob_ratio": None,
        "doc_ratio": None,
        "overall_ratio": 0.0,
        "available": False,
        "fields_available": {"name": False, "dob": False, "doc_number": False},
    }

    try:
        from rapidfuzz import fuzz

        viz_name = viz_fields.get("name")       # None if not found
        viz_dob  = viz_fields.get("dob")        # None if not found
        viz_doc  = viz_fields.get("doc_number") # None if not found

        fields_available = {
            "name":       viz_name is not None,
            "dob":        viz_dob is not None,
            "doc_number": viz_doc is not None,
        }

        if not any(fields_available.values()):
            logger.warning("VIZ OCR returned no usable fields; text consistency unavailable.")
            return unavailable_result

        mrz_full_name = f"{mrz_surname} {mrz_given}".strip().upper()

        # Normalize DOB to YYMMDD before comparing
        mrz_dob_norm = _normalize_dob_to_yymmdd(mrz_dob)
        viz_dob_norm  = _normalize_dob_to_yymmdd(viz_dob) if viz_dob else None

        # Normalize doc numbers: strip spaces/hyphens, uppercase
        mrz_doc_norm = re.sub(r"[\s\-]", "", mrz_doc_number).upper()
        viz_doc_norm  = re.sub(r"[\s\-]", "", viz_doc).upper() if viz_doc else None

        name_ratio = fuzz.token_sort_ratio(mrz_full_name, viz_name.upper()) if viz_name else None
        dob_ratio  = fuzz.partial_ratio(mrz_dob_norm, viz_dob_norm) if viz_dob_norm else None
        doc_ratio  = fuzz.partial_ratio(mrz_doc_norm, viz_doc_norm) if viz_doc_norm else None

        # Overall = average over available fields only
        available_ratios = [r for r in [name_ratio, dob_ratio, doc_ratio] if r is not None]
        overall = sum(available_ratios) / len(available_ratios) if available_ratios else 0.0

        result = {
            "name_ratio":       name_ratio,
            "dob_ratio":        dob_ratio,
            "doc_ratio":        doc_ratio,
            "overall_ratio":    overall,
            "available":        True,
            "fields_available": fields_available,
        }
        # PII hygiene: log ratios only (no actual field values)
        logger.info(
            "Text consistency — overall: %.1f, name: %s, dob: %s, doc: %s",
            overall,
            f"{name_ratio:.0f}" if name_ratio is not None else "N/A",
            f"{dob_ratio:.0f}" if dob_ratio is not None else "N/A",
            f"{doc_ratio:.0f}" if doc_ratio is not None else "N/A",
        )
        return result

    except ImportError:
        logger.warning("RapidFuzz not available; text consistency check unavailable.")
        return unavailable_result
    except Exception as exc:
        logger.error("Text consistency error: %s", exc, exc_info=True)
        return unavailable_result


def compute_risk_score(
    mrz_passed: Optional[bool],
    ela_mean_error: Optional[float],
    biometric_distance: Optional[float],
    text_ratio: Optional[float],
) -> dict:
    """Compute the composite 0-100 risk score from four sub-checks.

    Any sub-check that received None is treated as INCOMPLETE and its
    weight is redistributed proportionally across the available checks.

    Args:
        mrz_passed:          True/False if MRZ checks ran; None if unreadable.
        ela_mean_error:      Mean ELA error (0-255); None if ELA failed.
        biometric_distance:  ArcFace cosine distance; None if biometrics unavailable.
        text_ratio:          Overall fuzzy-match ratio (0-100); None if OCR failed.

    Returns:
        Dict with keys:
          'score'            (float)     - Final 0-100 risk score.
          'score_int'        (int)       - Rounded integer score.
          'category'         (str)       - 'LOW RISK' / 'SUSPICIOUS' / 'CRITICAL RISK'.
          'color'            (str)       - 'green' / 'yellow' / 'red'.
          'sub_scores'       (dict)      - Per-check normalised 0-100 contributions.
          'sub_weights_used' (dict)      - Actual weights after renormalisation.
          'incomplete'       (list[str]) - Names of unavailable sub-checks.
    """
    # Map sub-check names to (raw_value, weight, normaliser_function)
    checks = {
        "MRZ Checksum":      (mrz_passed,          WEIGHT_MRZ,       _normalize_mrz),
        "ELA Intensity":     (ela_mean_error,       WEIGHT_ELA,       _normalize_ela),
        "Biometric Distance": (biometric_distance,   WEIGHT_BIOMETRIC, _normalize_biometric),
        "Text Consistency":  (text_ratio,           WEIGHT_TEXT,      _normalize_text),
    }

    available = {}
    incomplete = []

    for name, (raw, weight, norm_fn) in checks.items():
        if raw is None:
            incomplete.append(name)
        else:
            try:
                score = norm_fn(raw)
            except Exception as exc:
                logger.error("Normalisation error for %s: %s", name, exc)
                incomplete.append(name)
                continue
            available[name] = {"raw": raw, "weight": weight, "score": score}

    # Renormalise weights if any check is incomplete
    if not available:
        logger.warning("All sub-checks are INCOMPLETE; returning score=0 with INCOMPLETE status.")
        return {
            "score": 0.0,
            "score_int": 0,
            "category": "INCOMPLETE",
            "color": "gray",
            "sub_scores": {},
            "sub_weights_used": {},
            "incomplete": list(checks.keys()),
        }

    total_raw_weight = sum(v["weight"] for v in available.values())
    sub_weights_used = {}
    weighted_sum = 0.0
    sub_scores = {}

    for name, data in available.items():
        renorm_weight = data["weight"] / total_raw_weight
        sub_weights_used[name] = round(renorm_weight, 4)
        sub_scores[name] = round(data["score"], 2)
        weighted_sum += data["score"] * renorm_weight

    final_score = max(0.0, min(100.0, weighted_sum))
    score_int = int(round(final_score))

    # F2: MANUAL REVIEW — if both identity-confirmation checks are unavailable,
    # do not issue a definitive LOW RISK verdict regardless of MRZ/ELA scores.
    # A valid MRZ + quiet ELA alone is insufficient to confirm identity.
    bio_incomplete = "Biometric Distance" in incomplete
    text_incomplete = "Text Consistency" in incomplete
    manual_review = bio_incomplete and text_incomplete

    if manual_review:
        category, color = "MANUAL REVIEW", "orange"
    elif score_int <= RISK_LOW_MAX:
        category, color = "LOW RISK", "green"
    elif score_int <= RISK_SUSPICIOUS_MAX:
        category, color = "SUSPICIOUS", "yellow"
    else:
        category, color = "CRITICAL RISK", "red"

    result = {
        "score": final_score,
        "score_int": score_int,
        "category": category,
        "color": color,
        "sub_scores": sub_scores,
        "sub_weights_used": sub_weights_used,
        "incomplete": incomplete,
        "manual_review": manual_review,
    }

    logger.info(
        "Risk score -- %.1f (%s), incomplete=%s, manual_review=%s",
        final_score, category, incomplete, manual_review,
    )
    return result
