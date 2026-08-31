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
    issue_date: Optional[str] = None
    issuing_authority: Optional[str] = None
    address: Optional[str] = None
    nationality: Optional[str] = None
    blood_group: Optional[str] = None
    relation_name: Optional[str] = None   # Son/Daughter/Wife of
    doc_subtype: str = "National ID"
    raw_lines: list = field(default_factory=list)
    fields_found: int = 0
    confidence_note: str = ""


def _ocr_image(image_path: str) -> list:
    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        # Use paragraph=False to preserve individual text regions
        results = reader.readtext(image_path, detail=0, paragraph=False)
        return results
    except Exception as exc:
        logger.error("EasyOCR error in id_engine: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Date patterns (inclusive: DD-MM-YYYY, DD/MM/YYYY, DD.MM.YYYY, YYYY-MM-DD)
# ---------------------------------------------------------------------------
_DATE_PATTERNS = [
    r"\b(\d{2}[-/.]\d{2}[-/.]\d{4})\b",   # DD-MM-YYYY or DD/MM/YYYY or DD.MM.YYYY
    r"\b(\d{4}[-/.]\d{2}[-/.]\d{2})\b",   # YYYY-MM-DD
    r"\b(\d{2}\s(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s\d{4})\b",
    r"\b(\d{2}/\d{2}/\d{2})\b",            # DD/MM/YY
]

# ---------------------------------------------------------------------------
# ID number patterns (expanded to cover Indian state-code formats like WB23...)
# ---------------------------------------------------------------------------
_ID_NUMBER_PATTERNS = [
    # Indian DL: 2-letter state + 2-digit year + 7-11 digits, with optional space/dash
    r"\b([A-Z]{2}\d{2}[-\s]?\d{7,11})\b",
    # Indian Aadhaar: 12-digit number
    r"\b(\d{4}\s\d{4}\s\d{4})\b",
    # Generic: letters + digits, 8-16 chars
    r"\b([A-Z]{1,4}[-\s]?\d{6,14})\b",
    # Pure numeric long ID
    r"\b(\d{10,16})\b",
]

_BLOOD_GROUPS = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-",
                  "A", "B", "AB", "O"]

def _extract_date(text: str) -> Optional[str]:
    for pattern in _DATE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def _clean_value(text: str, after_colon: bool = True) -> str:
    """Extract the value from 'Label: VALUE' or just clean a raw line."""
    if after_colon and ":" in text:
        text = text.split(":", 1)[-1]
    return text.strip()


def _extract_id_number(lines: list) -> Optional[str]:
    """Find licence/ID number — often on its own prominent line near the top."""
    triggers = ["dl no", "dl.", "licence no", "license no", "id no", "id number",
                "card no", "document no", "reg no", "driving licence", "dl-"]
    # First: look for a line explicitly labelled as licence number
    for line in lines:
        ll = line.lower()
        for trigger in triggers:
            if trigger in ll:
                for pattern in _ID_NUMBER_PATTERNS:
                    m = re.search(pattern, line, re.IGNORECASE)
                    if m:
                        return m.group(1).replace(" ", "").strip()
    # Second: scan all lines for Indian DL-format numbers (most prominent)
    for line in lines:
        # Indian DL pattern specifically: e.g. WB23 20150223071
        m = re.search(r"\b([A-Z]{2}\d{2}[\s-]?\d{7,11})\b", line, re.IGNORECASE)
        if m:
            return m.group(1).replace(" ", "").strip()
    # Third: any other long alphanumeric
    for line in lines:
        m = re.search(_ID_NUMBER_PATTERNS[3], line)
        if m and len(m.group(1)) >= 10:
            return m.group(1).strip()
    return None


def _extract_name(lines: list) -> Optional[str]:
    """Extract full name — handles 'Name: JOHN DOE' on same line or next line."""
    name_triggers = ["name:", "full name:", "holder:", "cardholder:"]
    for i, line in enumerate(lines):
        ll = line.lower().strip()
        for trigger in name_triggers:
            if ll.startswith(trigger) or ll == trigger.rstrip(":"):
                # Value on same line after colon
                after = _clean_value(line, after_colon=True)
                if after and len(after) >= 2:
                    # Filter out non-name garbage (dates, numbers)
                    if not re.search(r"\d{2}[-/]\d{2}", after):
                        return after.strip().title()
                # Value on next line
                if i + 1 < len(lines):
                    nxt = lines[i + 1].strip()
                    if nxt and re.match(r"^[A-Za-z\s]{3,}$", nxt):
                        return nxt.title()
        # Also match inline: line contains "Name" and has uppercase name text after
        if "name" in ll and ":" in line:
            val = line.split(":", 1)[-1].strip()
            if val and re.match(r"^[A-Za-z\s]{3,}$", val):
                return val.title()
    return None


def _extract_dob(lines: list) -> Optional[str]:
    """Extract date of birth — looks for 'Date of Birth', 'DOB', 'D.O.B'."""
    dob_triggers = ["date of birth", "dob", "d.o.b", "birth date", "born"]
    for line in lines:
        ll = line.lower()
        for trigger in dob_triggers:
            if trigger in ll:
                date = _extract_date(line)
                if date:
                    return date
                # Sometimes date is on the next segment — scan current line aggressively
                m = re.search(r"(\d{2}[-/.]\d{2}[-/.]\d{4})", line)
                if m:
                    return m.group(1)
    return None


def _extract_issue_date(lines: list) -> Optional[str]:
    issue_triggers = ["issue date", "issued on", "issue", "date of issue", "doi"]
    for line in lines:
        ll = line.lower()
        for trigger in issue_triggers:
            if trigger in ll:
                date = _extract_date(line)
                if date:
                    return date
    return None


def _extract_expiry(lines: list) -> Optional[str]:
    """Extract expiry/validity date — handles 'Validity', 'Validity(NT)', 'Expiry'."""
    expiry_triggers = ["validity", "valid upto", "valid until", "expiry", "expiration",
                       "expires", "exp date", "exp.", "valid to", "nt)", "transport"]
    for line in lines:
        ll = line.lower()
        for trigger in expiry_triggers:
            if trigger in ll:
                # Find ALL dates in the line and pick the last (validity is usually last)
                dates = re.findall(r"\d{2}[-/.]\d{2}[-/.]\d{4}", line)
                if dates:
                    return dates[-1]  # Last date in line = validity/expiry
    return None


def _extract_blood_group(lines: list) -> Optional[str]:
    full_text = " ".join(lines)
    # Look for "Blood Group: X" or "Blood: X" or standalone "A+" etc.
    m = re.search(r"blood\s*(?:group)?[:\s]+([ABO]{1,2}[+-]?)", full_text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Fallback: scan for standalone blood group tokens
    for bg in _BLOOD_GROUPS:
        pattern = r"\b" + re.escape(bg) + r"\b"
        if re.search(pattern, full_text, re.IGNORECASE):
            return bg
    return None


def _extract_gender(lines: list) -> Optional[str]:
    full_text = " ".join(lines).upper()
    patterns = [
        (r"\bSEX[:\s]+([MF])\b", None),
        (r"\bGENDER[:\s]+(MALE|FEMALE|M|F)\b", None),
        (r"\b(MALE|FEMALE)\b", None),
    ]
    for pattern, _ in patterns:
        m = re.search(pattern, full_text)
        if m:
            v = m.group(1).upper()
            return "Male" if v in ("M", "MALE") else "Female"
    return None


def _extract_issuing_authority(lines: list) -> Optional[str]:
    triggers = ["issued by", "issuing authority", "authority", "government of",
                "state of", "regional transport", "rto", "transport office"]
    for line in lines:
        ll = line.lower()
        for trigger in triggers:
            if trigger in ll:
                val = line.strip()
                # Remove leading "Issued by" etc.
                val = re.sub(r"(?i)^issued\s*by\s*", "", val).strip()
                if val and len(val) >= 4:
                    return val.title()
    return None


def _extract_address(lines: list) -> Optional[str]:
    """Grab address block after 'Address:' label."""
    addr_triggers = ["address:", "add:", "residence:", "domicile:"]
    for i, line in enumerate(lines):
        ll = line.lower().strip()
        for trigger in addr_triggers:
            if ll.startswith(trigger):
                addr_parts = [_clean_value(line, after_colon=True)]
                # Grab the next 1-2 continuation lines
                for j in range(i + 1, min(i + 3, len(lines))):
                    nxt = lines[j].strip()
                    # Stop if we hit another labelled field
                    if re.search(r"^(name|date|validity|issue|blood|organ|son|daughter)[\s:]",
                                  nxt, re.IGNORECASE):
                        break
                    addr_parts.append(nxt)
                return " ".join(p for p in addr_parts if p).strip()
    return None


def _extract_relation(lines: list) -> Optional[str]:
    """Extract Son/Daughter/Wife of field."""
    triggers = ["son of", "daughter of", "wife of", "s/o", "d/o", "w/o",
                "son/daughter", "father", "guardian"]
    for line in lines:
        ll = line.lower()
        for trigger in triggers:
            if trigger in ll:
                # Extract what comes after the trigger
                pattern = re.escape(trigger) + r"[:\s]+([A-Za-z\s]{3,})"
                m = re.search(pattern, ll)
                if m:
                    return m.group(1).strip().title()
                # Or just grab the remaining text
                after = re.split(re.escape(trigger), line, flags=re.IGNORECASE)[-1]
                after = after.strip(" :\t")
                if after and len(after) >= 3:
                    return after.strip().title()
    return None


def _detect_subtype(lines: list) -> str:
    full_text = " ".join(lines).lower()
    if any(kw in full_text for kw in ["driving licence", "driving license", "driver", "dl"]):
        return "Driving Licence"
    if any(kw in full_text for kw in ["permit", "residence permit", "work permit"]):
        return "Permit"
    if any(kw in full_text for kw in ["voter", "election"]):
        return "Voter ID"
    if any(kw in full_text for kw in ["aadhaar", "aadhar", "uidai"]):
        return "Aadhaar Card"
    if any(kw in full_text for kw in ["pan card", "income tax", "permanent account"]):
        return "PAN Card"
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
    # Fallback: Indian DLs are implicitly Indian
    full_text = " ".join(lines).lower()
    if any(kw in full_text for kw in ["india", "government of", "union"]):
        return "Indian"
    return None


def extract_id_fields(image_path: str) -> IDResult:
    """Run EasyOCR on the ID/licence image and extract all structured fields."""
    lines = _ocr_image(image_path)
    if not lines:
        return IDResult(status="UNREADABLE", confidence_note="EasyOCR returned no text from this image.")

    lines = [l.strip() for l in lines if l.strip()]
    result = IDResult(raw_lines=lines)

    result.doc_subtype       = _detect_subtype(lines)
    result.id_number         = _extract_id_number(lines)
    result.full_name         = _extract_name(lines)
    result.date_of_birth     = _extract_dob(lines)
    result.gender            = _extract_gender(lines)
    result.issue_date        = _extract_issue_date(lines)
    result.expiry_date       = _extract_expiry(lines)
    result.issuing_authority = _extract_issuing_authority(lines)
    result.address           = _extract_address(lines)
    result.nationality       = _extract_nationality(lines)
    result.blood_group       = _extract_blood_group(lines)
    result.relation_name     = _extract_relation(lines)

    found = sum(1 for f in [
        result.id_number, result.full_name, result.date_of_birth, result.gender,
        result.expiry_date, result.issuing_authority, result.address,
        result.nationality, result.blood_group,
    ] if f is not None)

    result.fields_found = found

    if found >= 4:
        result.status = "OK"
        result.confidence_note = f"Extracted {found}/9 fields successfully."
    elif found >= 1:
        result.status = "PARTIAL"
        result.confidence_note = f"Only {found}/9 fields extracted. Ensure the document is well-lit and fully visible."
    else:
        result.status = "UNREADABLE"
        result.confidence_note = "Could not extract any structured ID fields."

    return result
