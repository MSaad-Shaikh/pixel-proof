import logging, re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

VISA_TYPE_KEYWORDS = {
    "tourist":   ["tourist", "tourism", "type b"],
    "business":  ["business", "commercial", "type c"],
    "transit":   ["transit", "airport transit", "type a"],
    "student":   ["student", "study", "education"],
    "work":      ["work", "employment", "labour"],
    "official":  ["official", "diplomatic", "government"],
    "medical":   ["medical", "treatment"],
}

@dataclass
class VisaResult:
    status: str = "UNPROCESSED"
    visa_number: Optional[str] = None
    visa_type: Optional[str] = None
    country_of_issue: Optional[str] = None
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    entries: Optional[str] = None
    stay_duration: Optional[str] = None
    applicant_name: Optional[str] = None
    nationality: Optional[str] = None
    raw_lines: list = field(default_factory=list)
    fields_found: int = 0
    confidence_note: str = ""


_VISA_SHARED_READER = None


def _get_visa_reader():
    global _VISA_SHARED_READER
    if _VISA_SHARED_READER is None:
        try:
            import easyocr
            _VISA_SHARED_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
        except Exception as exc:
            logger.error("EasyOCR initialization error in visa_engine: %s", exc)
    return _VISA_SHARED_READER


def _ocr_image(image_path: str, reader=None) -> list:
    try:
        r = reader if reader is not None else _get_visa_reader()
        if r is None:
            return []
        results = r.readtext(image_path, detail=0, paragraph=True)
        return results
    except Exception as exc:
        logger.error("EasyOCR error in visa_engine: %s", exc)
        return []


_DATE_PATTERNS = [
    r"\b(\d{2}[/-]\d{2}[/-]\d{4})\b",
    r"\b(\d{4}[/-]\d{2}[/-]\d{2})\b",
    r"\b(\d{2}\s(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s\d{4})\b",
    r"\b(\d{2}\.\d{2}\.\d{4})\b",
]

_VISA_NUMBER_PATTERNS = [
    r"\b([A-Z]{1,3}\s?\d{6,12})\b",
    r"\b(\d{8,12})\b",
    r"\b([A-Z0-9]{8,14})\b",
]

_DURATION_PATTERNS = [
    r"(\d{1,3})\s*(?:days?|day)",
    r"duration[:\s]+(\d{1,3})",
    r"stay[:\s]+(\d{1,3})",
]

_ENTRIES_MAP = {
    "multiple": ["multiple", "mult"],
    "double":   ["double", "dbl"],
    "single":   ["single"],
}


def _extract_date(text: str) -> Optional[str]:
    for pattern in _DATE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def _extract_visa_number(lines: list) -> Optional[str]:
    triggers = ["visa no", "visa number", "foil", "vignette", "no.", "number"]
    for line in lines:
        ll = line.lower()
        for trigger in triggers:
            if trigger in ll:
                for pattern in _VISA_NUMBER_PATTERNS:
                    m = re.search(pattern, line, re.IGNORECASE)
                    if m:
                        return m.group(1).strip()
    for line in lines:
        m = re.search(_VISA_NUMBER_PATTERNS[1], line)
        if m and len(m.group(1)) >= 7:
            return m.group(1).strip()
    return None


def _extract_visa_type(lines: list) -> Optional[str]:
    full_text = " ".join(lines).lower()
    for vtype, keywords in VISA_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in full_text:
                return vtype.capitalize()
    m = re.search(r"(?:type|category)[:\s]+([A-Za-z0-9\-]+)", full_text)
    if m:
        return m.group(1).upper()
    return None


def _extract_entries(lines: list) -> Optional[str]:
    full_text = " ".join(lines).lower()
    for label, keywords in _ENTRIES_MAP.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", full_text):
                return label.capitalize()
    return None


def _extract_duration(lines: list) -> Optional[str]:
    full_text = " ".join(lines).lower()
    for pattern in _DURATION_PATTERNS:
        m = re.search(pattern, full_text, re.IGNORECASE)
        if m:
            return f"{m.group(1)} days"
    return None


def _extract_name(lines: list) -> Optional[str]:
    name_triggers = ["name", "holder", "surname", "applicant", "family name"]
    for i, line in enumerate(lines):
        ll = line.lower()
        for trigger in name_triggers:
            if trigger in ll:
                m = re.search(r"[:\s]+([A-Z][A-Z\s,]{3,})", line)
                if m:
                    return m.group(1).strip().title()
                if i + 1 < len(lines) and re.match(r"^[A-Z\s,]{4,}", lines[i + 1]):
                    return lines[i + 1].strip().title()
    return None


def _extract_nationality(lines: list) -> Optional[str]:
    triggers = ["nationality", "national", "citizen", "country of applicant"]
    for line in lines:
        ll = line.lower()
        for trigger in triggers:
            if trigger in ll:
                m = re.search(r"[:\s]+([A-Za-z\s]{3,})", line)
                if m:
                    return m.group(1).strip().title()
    return None


def _extract_country_of_issue(lines: list) -> Optional[str]:
    triggers = ["issued by", "issuing", "embassy", "consulate", "country of issue"]
    for line in lines:
        ll = line.lower()
        for trigger in triggers:
            if trigger in ll:
                m = re.search(r"[:\s]+([A-Za-z\s]{3,})", line)
                if m:
                    return m.group(1).strip().title()
    return None


def _extract_dates(lines: list) -> tuple:
    from_triggers = ["from", "issued", "issue date", "valid from", "date of issue"]
    until_triggers = ["until", "expiry", "expir", "valid to", "valid until"]
    valid_from = None
    valid_until = None
    spare = []
    for line in lines:
        ll = line.lower()
        date = _extract_date(line)
        if not date:
            continue
        if any(t in ll for t in until_triggers) and not valid_until:
            valid_until = date
        elif any(t in ll for t in from_triggers) and not valid_from:
            valid_from = date
        else:
            spare.append(date)
    if not valid_from and spare:
        valid_from = spare.pop(0)
    if not valid_until and spare:
        valid_until = spare.pop(0)
    return valid_from, valid_until


def extract_visa_fields(image_path: str, reader=None) -> VisaResult:
    """Run EasyOCR on the visa image and extract all structured fields."""
    lines = _ocr_image(image_path, reader=reader)
    if not lines:
        return VisaResult(status="UNREADABLE", confidence_note="EasyOCR returned no text from this image.")
    lines = [l.strip() for l in lines if l.strip()]
    result = VisaResult(raw_lines=lines)
    result.visa_number       = _extract_visa_number(lines)
    result.visa_type         = _extract_visa_type(lines)
    result.country_of_issue  = _extract_country_of_issue(lines)
    result.valid_from, result.valid_until = _extract_dates(lines)
    result.entries           = _extract_entries(lines)
    result.stay_duration     = _extract_duration(lines)
    result.applicant_name    = _extract_name(lines)
    result.nationality       = _extract_nationality(lines)
    found = sum(1 for f in [
        result.visa_number, result.visa_type, result.country_of_issue,
        result.valid_from, result.valid_until, result.entries,
        result.stay_duration, result.applicant_name, result.nationality,
    ] if f is not None)
    result.fields_found = found
    if found >= 4:
        result.status = "OK"
        result.confidence_note = f"Extracted {found}/9 fields successfully."
    elif found >= 1:
        result.status = "PARTIAL"
        result.confidence_note = f"Only {found}/9 fields extracted. Image quality may be limiting OCR accuracy."
    else:
        result.status = "UNREADABLE"
        result.confidence_note = "Could not extract any structured visa fields."
    return result
