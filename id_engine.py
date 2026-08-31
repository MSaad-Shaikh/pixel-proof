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
    relation_name: Optional[str] = None
    doc_subtype: str = "National ID"
    raw_lines: list = field(default_factory=list)
    fields_found: int = 0
    confidence_note: str = ""


def _ocr_image(image_path: str) -> list:
    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        results = reader.readtext(image_path, detail=0, paragraph=False)
        return results
    except Exception as exc:
        logger.error("EasyOCR error in id_engine: %s", exc)
        return []


_DATE_RE = re.compile(r"\b(\d{2}[-/.]\d{2}[-/.]\d{4})\b")


def _find_date_in_or_next(lines: list, index: int) -> Optional[str]:
    # Check current line
    m = _DATE_RE.search(lines[index])
    if m:
        return m.group(1)
    # Check next 1-2 lines
    for offset in (1, 2):
        if index + offset < len(lines):
            m = _DATE_RE.search(lines[index + offset])
            if m:
                return m.group(1)
    return None


def _extract_all_fields(lines: list) -> IDResult:
    result = IDResult(raw_lines=lines)
    full_text = " \n ".join(lines)
    lower_lines = [l.lower().strip() for l in lines]

    # 1. Document Subtype
    if any("driving" in l or "motor" in l or "dl no" in l for l in lower_lines):
        result.doc_subtype = "Driving Licence"
    elif any("aadhaar" in l or "uidai" in l for l in lower_lines):
        result.doc_subtype = "Aadhaar Card"
    elif any("pan card" in l or "income tax" in l for l in lower_lines):
        result.doc_subtype = "PAN Card"
    elif any("passport" in l for l in lower_lines):
        result.doc_subtype = "Passport"
    elif any("permit" in l for l in lower_lines):
        result.doc_subtype = "Residence Permit"
    else:
        result.doc_subtype = "National ID"

    # 2. DL / ID Number
    # Look for "DL No MH12 20010149313" or similar
    dl_pattern = re.compile(r"\b([A-Z]{2}[-\s]?\d{2}[-\s]?\d{7,11})\b", re.IGNORECASE)
    for i, line in enumerate(lines):
        # Look for labelled DL number
        if any(kw in lower_lines[i] for kw in ["dl no", "licence no", "license no", "id no"]):
            # Try current line after removing label
            clean_line = re.sub(r"(?i)\b(dl|licence|license|id|no|card)[\s.:#]*", " ", line).strip()
            m = dl_pattern.search(clean_line)
            if m:
                result.id_number = m.group(1).upper()
                break
            # Standalone long alphanumeric on same or next line
            m_generic = re.search(r"\b([A-Z0-9\s]{8,20})\b", clean_line)
            if m_generic and any(c.isdigit() for c in m_generic.group(1)):
                result.id_number = m_generic.group(1).strip().upper()
                break
            if i + 1 < len(lines):
                m_next = dl_pattern.search(lines[i+1])
                if m_next:
                    result.id_number = m_next.group(1).upper()
                    break

    # Fallback search anywhere in document
    if not result.id_number:
        for line in lines:
            m = dl_pattern.search(line)
            if m:
                result.id_number = m.group(1).upper()
                break

    # 3. Issuing Authority
    # Look for State Header or "Issuing Authority: MH12 ..."
    state_match = None
    # Check for specific State or Department name first
    for line in lines:
        ll = line.lower()
        if ("state" in ll or "government of" in ll or "transport" in ll) and not any(k in ll for k in ["authorisation", "rule", "class", "vehicle"]):
            clean = line.strip(" :-\t")
            if len(clean) > 8:
                state_match = clean.title()
                break
    if not state_match:
        for line in lines:
            if "union of india" in line.lower():
                state_match = line.strip(" :-\t").title()
                break
    
    # Check for specific "Issuing Authority:" label
    auth_code = None
    for i, line in enumerate(lines):
        if "issuing authority" in lower_lines[i]:
            # Next line usually has officer code e.g. "MH12 2016502"
            if i + 1 < len(lines):
                nxt = lines[i+1].strip()
                if nxt and not any(k in nxt.lower() for k in ["impression", "holder", "signature"]):
                    auth_code = nxt.strip()
            elif ":" in line:
                val = line.split(":", 1)[-1].strip()
                if val and len(val) > 2:
                    auth_code = val

    if state_match and auth_code:
        result.issuing_authority = f"{state_match} ({auth_code})"
    elif state_match:
        result.issuing_authority = state_match
    elif auth_code:
        result.issuing_authority = auth_code

    # 4. Name
    for i, line in enumerate(lines):
        ll = lower_lines[i]
        if ll.startswith("name") or ll == "name" or "name :" in ll or "name:" in ll:
            # Case A: Same line "Name :NIVRUTTI BODAKE"
            after = line.split(":", 1)[-1].strip() if ":" in line else ""
            if after and re.search(r"[A-Za-z]{3,}", after):
                result.full_name = after.title()
                break
            # Case B: Name on subsequent 1-2 lines e.g. ["NivruTtI", "BODAKE"]
            name_parts = []
            for j in range(i + 1, min(i + 4, len(lines))):
                candidate = lines[j].strip()
                cand_lower = candidate.lower()
                # Stop if we hit another key
                if any(cand_lower.startswith(k) or k in cand_lower for k in [
                    "s/d/w", "sidn", "son", "daughter", "wife", "add", "dob", "bg", "blood", "pin", "signature"
                ]):
                    break
                if re.match(r"^[A-Za-z\s,.'-]+$", candidate) and len(candidate) >= 2:
                    name_parts.append(candidate)
                else:
                    break
            if name_parts:
                result.full_name = " ".join(name_parts).title()
                break

    # 5. Date of Birth (DOB)
    for i, line in enumerate(lines):
        ll = lower_lines[i]
        if "dob" in ll or "date of birth" in ll or "birth" in ll:
            dob_val = _find_date_in_or_next(lines, i)
            if dob_val:
                result.date_of_birth = dob_val
                break

    # 6. Issue Date (DOI)
    for i, line in enumerate(lines):
        ll = lower_lines[i]
        if "doi" in ll or "issue date" in ll or "issued on" in ll or "dld" in ll:
            doi_val = _find_date_in_or_next(lines, i)
            if doi_val and doi_val != result.date_of_birth:
                result.issue_date = doi_val
                break

    # 7. Expiry Date (Valid Till / Validity)
    for i, line in enumerate(lines):
        ll = lower_lines[i]
        if "valid" in ll or "expiry" in ll or "exp" in ll or "validity" in ll:
            exp_val = _find_date_in_or_next(lines, i)
            if exp_val and exp_val != result.date_of_birth and exp_val != result.issue_date:
                result.expiry_date = exp_val
                break

    # 8. Blood Group (BG)
    for i, line in enumerate(lines):
        ll = lower_lines[i]
        if "bg" in ll or "blood" in ll:
            # Check same line
            m = re.search(r"\b([ABO]{1,2}[+-])\b", line, re.IGNORECASE)
            if m:
                result.blood_group = m.group(1).upper()
                break
            # Check next line
            if i + 1 < len(lines):
                m_next = re.search(r"\b([ABO]{1,2}[+-])\b", lines[i+1], re.IGNORECASE)
                if m_next:
                    result.blood_group = m_next.group(1).upper()
                    break

    # 9. Guardian / Relation (S/D/W of)
    for i, line in enumerate(lines):
        ll = lower_lines[i]
        if any(kw in ll for kw in ["s/d/w", "sidn", "s/o", "d/o", "w/o", "son", "daughter", "wife"]):
            # Text after colon or keyword
            clean = re.split(r"(?i)(?:s/d/w|sidn|s/o|d/o|w/o|of)\s*[:.]?", line)[-1].strip()
            if clean and len(clean) >= 3 and re.search(r"[A-Za-z]", clean):
                result.relation_name = clean.title()
                break
            elif i + 1 < len(lines):
                nxt = lines[i+1].strip()
                if nxt and not any(k in nxt.lower() for k in ["add", "pin", "tal", "dist", "dob"]):
                    result.relation_name = nxt.title()
                    break

    # 10. Address
    for i, line in enumerate(lines):
        ll = lower_lines[i]
        if ll.startswith("add") or "add :" in ll or "address" in ll:
            addr_parts = []
            clean_first = re.split(r"(?i)add(?:ress)?\s*[:.]?", line)[-1].strip(" :-\t")
            if clean_first:
                addr_parts.append(clean_first)
            for j in range(i + 1, min(i + 4, len(lines))):
                nxt = lines[j].strip()
                nxt_lower = nxt.lower()
                if any(nxt_lower.startswith(k) for k in ["pin", "signature", "issuing", "holder"]):
                    if "pin" in nxt_lower:
                        addr_parts.append(nxt)
                    break
                addr_parts.append(nxt)
            if addr_parts:
                result.address = " ".join(addr_parts).title()
                break

    # 11. Nationality
    if any(k in full_text.lower() for k in ["india", "maharashtra", "west bengal", "delhi", "union of india"]):
        result.nationality = "Indian"

    # Count extracted fields
    found = sum(1 for f in [
        result.id_number, result.full_name, result.date_of_birth,
        result.issue_date, result.expiry_date, result.issuing_authority,
        result.address, result.nationality, result.relation_name,
    ] if f is not None)
    result.fields_found = found

    if found >= 4:
        result.status = "OK"
        result.confidence_note = f"Extracted {found}/9 fields successfully."
    elif found >= 1:
        result.status = "PARTIAL"
        result.confidence_note = f"Extracted {found}/9 fields. Some fields may have low contrast."
    else:
        result.status = "UNREADABLE"
        result.confidence_note = "Could not extract structured fields."

    return result


def extract_id_fields(image_path: str) -> IDResult:
    """Run EasyOCR on the ID/licence image and extract all structured fields."""
    lines = _ocr_image(image_path)
    if not lines:
        return IDResult(status="UNREADABLE", confidence_note="EasyOCR returned no text from this image.")
    lines = [l.strip() for l in lines if l.strip()]
    return _extract_all_fields(lines)
