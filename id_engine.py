from __future__ import annotations
import logging, re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class IDResult:
    status: str = "UNPROCESSED"
    id_number: Optional[str] = None
    full_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    expiry_date: Optional[str] = None
    issuing_authority: Optional[str] = None
    address: Optional[str] = None
    nationality: Optional[str] = None
    doc_subtype: str = "National ID"   # "National ID" | "Driving Licence" | "Permit"
    raw_lines: list = field(default_factory=list)
    fields_found: int = 0
    confidence_note: str = ""


def _ocr_image(image_path: str) -> list:
    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        results = reader.readtext(image_path, detail=0, paragraph=True)
        return results
    except Exception as exc:
        logger.error("EasyOCR error in id_engine: %s", exc)
        return []


_DATE_PATTERNS = [
    r"\b(\d{2}[/-]\d{2}[/-]\d{4})\b",
    r"\b(\d{4}[/-]\d{2}[/-]\d{2})\b",
    r"\b(\d{2}\s(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s\d{4})\b",
    r"\b(\d{2}\.\d{2}\.\d{4})\b",
    r"\b(\d{2}/\d{2}/\d{2})\b",
]

_ID_NUMBER_PATTERNS = [
    r"\b([A-Z]{1,3}[- ]?\d{6,12})\b",
    r"\b(\d{10,16})\b",
    r"\b([A-Z0-9]{8,16})\b",
]

_GENDER_PATTERNS = [
    (r"\b(M|MALE|MAN)\b", "Male"),
    (r"\b(F|FEMALE|WOMAN)\b", "Female"),
    (r"\bSEX[:\s]+([MF])\b", None),
    (r"\bGENDER[:\s]+([MF])\b", None),
    (r"\bGENDER[:\s]+(MALE|FEMALE)\b", None),
]


def _extract_date(text: str) -> Optional[str]:
    for pattern in _DATE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def _extract_id_number(lines: list) -> Optional[str]:
    triggers = ["id no", "id number", "identity", "card no", "licence no",
                "license no", "dl no", "document no", "number", "no."]
    for line in lines:
        ll = line.lower()
        for trigger in triggers:
            if trigger in ll:
                for pattern in _ID_NUMBER_PATTERNS:
                    m = re.search(pattern, line, re.IGNORECASE)
                    if m:
                        return m.group(1).strip()
    for line in lines:
        for pattern in _ID_NUMBER_PATTERNS[:2]:
            m = re.search(pattern, line)
            if m and len(m.group(1)) >= 8:
                return m.group(1).strip()
    return None


def _extract_name(lines: list) -> Optional[str]:
    name_triggers = ["name", "holder", "surname", "full name", "given name", "family"]
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


def _extract_dob(lines: list) -> Optional[str]:
    dob_triggers = ["date of birth", "dob", "born", "birth", "d.o.b"]
    for line in lines:
        ll = line.lower()
        for trigger in dob_triggers:
            if trigger in ll:
                date = _extract_date(line)
                if date:
                    return date
    return None


def _extract_expiry(lines: list) -> Optional[str]:
    expiry_triggers = ["expiry", "expiration", "valid until", "valid to", "expires", "exp"]
    for line in lines:
        ll = line.lower()
        for trigger in expiry_triggers:
            if trigger in ll:
                date = _extract_date(line)
                if date:
                    return date
    return None


def _extract_gender(lines: list) -> Optional[str]:
    full_text = " ".join(lines).upper()
    for pattern, label in _GENDER_PATTERNS:
        m = re.search(pattern, full_text)
        if m:
            if label:
                return label
            val = m.group(1).upper()
            return "Male" if val in ("M", "MALE") else "Female"
    return None


def _extract_issuing_authority(lines: list) -> Optional[str]:
    triggers = ["issued by", "issuing authority", "authority", "issuer", "issued at", "place of issue"]
    for line in lines:
        ll = line.lower()
        for trigger in triggers:
            if trigger in ll:
                m = re.search(r"[:\s]+([A-Za-z\s]{3,})", line)
                if m:
                    return m.group(1).strip().title()
    return None


def _extract_address(lines: list) -> Optional[str]:
    addr_triggers = ["address", "residence", "domicile", "add.", "street"]
    for i, line in enumerate(lines):
        ll = line.lower()
        for trigger in addr_triggers:
            if trigger in ll:
                rest = line.split(":", 1)[-1].strip()
                if rest and len(rest) > 5:
                    return rest.title()
                if i + 1 < len(lines):
                    return lines[i + 1].strip().title()
    return None


def _detect_subtype(lines: list) -> str:
    full_text = " ".join(lines).lower()
    if any(kw in full_text for kw in ["driving licence", "driver", "dl", "license"]):
        return "Driving Licence"
    if any(kw in full_text for kw in ["permit", "residence permit", "work permit"]):
        return "Permit"
    return "National ID"


def _extract_nationality(lines: list) -> Optional[str]:
    triggers = ["nationality", "national", "citizen"]
    for line in lines:
        ll = line.lower()
        for trigger in triggers:
            if trigger in ll:
                m = re.search(r"[:\s]+([A-Za-z\s]{3,})", line)
                if m:
                    return m.group(1).strip().title()
    return None


def extract_id_fields(image_path: str) -> IDResult:
    """Run EasyOCR on the ID/licence image and extract all structured fields."""
    lines = _ocr_image(image_path)
    if not lines:
        return IDResult(status="UNREADABLE", confidence_note="EasyOCR returned no text from this image.")
    lines = [l.strip() for l in lines if l.strip()]
    result = IDResult(raw_lines=lines)
    result.doc_subtype      = _detect_subtype(lines)
    result.id_number        = _extract_id_number(lines)
    result.full_name        = _extract_name(lines)
    result.date_of_birth    = _extract_dob(lines)
    result.gender           = _extract_gender(lines)
    result.expiry_date      = _extract_expiry(lines)
    result.issuing_authority = _extract_issuing_authority(lines)
    result.address          = _extract_address(lines)
    result.nationality      = _extract_nationality(lines)
    found = sum(1 for f in [
        result.id_number, result.full_name, result.date_of_birth, result.gender,
        result.expiry_date, result.issuing_authority, result.address, result.nationality,
    ] if f is not None)
    result.fields_found = found
    if found >= 4:
        result.status = "OK"
        result.confidence_note = f"Extracted {found}/8 fields successfully."
    elif found >= 1:
        result.status = "PARTIAL"
        result.confidence_note = f"Only {found}/8 fields extracted. Ensure the document is well-lit and fully visible."
    else:
        result.status = "UNREADABLE"
        result.confidence_note = "Could not extract any structured ID fields."
    return result
