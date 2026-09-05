"""
report_builder.py  Evidence Report Generator
=============================================
Generates a structured Markdown evidence report from a completed
PixelProof verification session. Suitable for export as a .md file.

The report includes:
  - Source document hash (SHA-256) and original filename
  - Inspection timestamp
  - MRZ parsed fields and per-check results
  - ELA forensics summary
  - Biometric matching result
  - Text/VIZ consistency result
  - Pre-flight document quality findings
  - Composite risk score and verdict
  - Reviewer decision fields
"""

from datetime import datetime, timezone
from typing import Optional


def build_evidence_report(
    original_filename: str,
    sha256_hash: str,
    mrz_result,
    ela_result: dict,
    bio_result: dict,
    text_consistency: dict,
    quality_report,
    risk: dict,
    mrz_method: str = "",
    doc_proc_note: str = "",
    doc_type: str = "Passport",
    id_result=None,
    visa_result=None,
) -> str:
    """Build a Markdown evidence report string from all verification outputs."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    verdict_emoji = {
        "LOW RISK": "🟢",
        "SUSPICIOUS": "🟡",
        "CRITICAL RISK": "🔴",
        "MANUAL REVIEW": "🟠",
        "INCOMPLETE": "⚪",
    }.get(risk.get("category", ""), "⚪")

    lines: list[str] = [
        "# PixelProof — Inspection Evidence Report",
        "",
        f"**Generated:** {ts}  ",
        f"**System:** PixelProof Educational Prototype  ",
        "> ⚠️ This report is produced by an educational prototype. It is NOT certified for operational border or immigration use.",
        "",
        "---",
        "",
        "## 1. Source Document",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Original Filename | `{original_filename}` |",
        f"| SHA-256 Hash | `{sha256_hash}` |",
        f"| Document Type | `{doc_type}` |",
        f"| Inspection Timestamp | `{ts}` |",
        f"| Extraction Engine | {mrz_method or 'EasyOCR'} |",
    ]
    if doc_proc_note:
        lines.append(f"| Processing Note | {doc_proc_note} |")
    lines += [
        "",
        "> **Evidence Note:** SHA-256 hash computed from original uploaded bytes. Any rotation or cropping was applied to a separate derivative image used for OCR and biometrics. ELA was run on the original file.",
        "",
        "---",
        "",
    ]

    # Section 2: Document specific field extraction
    if doc_type == "Passport":
        lines += [
            "## 2. MRZ Extraction & Validation",
            "",
        ]
        if mrz_result.status == "MRZ_UNREADABLE":
            lines += [
                "**Status:** ❌ MRZ UNREADABLE",
                "",
                "**Failure reasons:**",
            ]
            for exp in (mrz_result.failed_explanations or []):
                lines.append(f"- {exp}")
        else:
            status_icon = "✅" if mrz_result.all_checks_passed else "❌"
            lines += [
                f"**Status:** {status_icon} {'ALL CHECKS PASSED' if mrz_result.all_checks_passed else 'CHECKSUM FAILURE'}",
                "",
                "| Field | Value | Check |",
                "|-------|-------|-------|",
                f"| Document Type | `{mrz_result.document_type or '—'}` | — |",
                f"| Issuing State | `{mrz_result.issuing_state or '—'}` | — |",
                f"| Nationality | `{mrz_result.nationality or '—'}` | — |",
                f"| Surname | `{mrz_result.surname or '—'}` | — |",
                f"| Given Names | `{mrz_result.given_names or '—'}` | — |",
                f"| Date of Birth | `{mrz_result.date_of_birth or '—'}` | {'✅' if mrz_result.dob_check and mrz_result.dob_check.passed else '❌'} |",
                f"| Expiry Date | `{mrz_result.expiration_date or '—'}` | {'✅' if mrz_result.expiry_check and mrz_result.expiry_check.passed else '❌'} |",
                f"| Document Number | `{mrz_result.document_number or '—'}` | {'✅' if mrz_result.document_number_check and mrz_result.document_number_check.passed else '❌'} |",
                f"| Composite Check Digit | — | {'✅' if mrz_result.composite_check and mrz_result.composite_check.passed else '❌'} |",
                "",
                "**Raw TD3 String:**",
                "```",
                mrz_result.raw_line1 or "",
                mrz_result.raw_line2 or "",
                "```",
            ]
    elif doc_type == "Visa":
        lines += [
            "## 2. Visa Field Extraction",
            "",
        ]
        if visa_result and getattr(visa_result, "status", "") != "UNREADABLE":
            lines += [
                "**Status:** ✅ EXTRACTED",
                "",
                "| Field | Value |",
                "|-------|-------|",
                f"| Document Subtype | `{getattr(visa_result, 'doc_subtype', 'Visa')}` |",
                f"| Visa Number | `{getattr(visa_result, 'visa_number', '—')}` |",
                f"| Full Name | `{getattr(visa_result, 'full_name', '—')}` |",
                f"| Passport Number | `{getattr(visa_result, 'passport_number', '—')}` |",
                f"| Nationality | `{getattr(visa_result, 'nationality', '—')}` |",
                f"| Valid From | `{getattr(visa_result, 'valid_from', '—')}` |",
                f"| Valid Until | `{getattr(visa_result, 'valid_until', '—')}` |",
                f"| Issuing Post | `{getattr(visa_result, 'issuing_post', '—')}` |",
            ]
        else:
            lines += ["**Status:** ⚠️ UNREADABLE or Partially Extracted"]
    else:  # National ID / Driving Licence / Aadhaar
        lines += [
            f"## 2. {doc_type} Field Extraction",
            "",
        ]
        if id_result and getattr(id_result, "status", "") != "UNREADABLE":
            lines += [
                "**Status:** ✅ EXTRACTED",
                "",
                "| Field | Value |",
                "|-------|-------|",
                f"| Document Subtype | `{getattr(id_result, 'doc_subtype', doc_type)}` |",
                f"| ID / Licence Number | `{getattr(id_result, 'id_number', '—')}` |",
                f"| Full Name | `{getattr(id_result, 'full_name', '—')}` |",
                f"| Date of Birth | `{getattr(id_result, 'date_of_birth', getattr(id_result, 'dob', '—'))}` |",
                f"| Gender | `{getattr(id_result, 'gender', '—')}` |",
                f"| Issuing Authority | `{getattr(id_result, 'issuing_authority', '—')}` |",
                f"| Blood Group | `{getattr(id_result, 'blood_group', '—')}` |",
                f"| Expiry Date | `{getattr(id_result, 'expiry_date', '—')}` |",
            ]
        else:
            lines += ["**Status:** ⚠️ UNREADABLE or Partially Extracted"]
    lines += [
        "",
        "---",
        "",
        "## 3. Error Level Analysis (ELA)",
        "",
    ]

    ela_err = ela_result.get("mean_error")
    ela_alert = ela_result.get("alert", False)
    ela_caveat = ela_result.get("caveat", "")
    ela_error_msg = ela_result.get("error", "")

    if ela_error_msg:
        lines.append(f"**Status:** ⚠️ ELA could not run — {ela_error_msg}")
    else:
        ela_icon = "⚠️ ELEVATED" if ela_alert else "✅ NORMAL"
        lines += [
            f"**Status:** {ela_icon}  ",
            f"**Mean Artifact Intensity:** {ela_err:.2f} (threshold: 12.0)  ",
        ]
        if ela_caveat:
            lines.append(f"**Caveat:** {ela_caveat}")

    lines += [
        "",
        "> ELA is a heuristic indicator only. Elevated scores warrant further review but do not confirm tampering.",
        "",
        "---",
        "",
        "## 4. Biometric Face Verification",
        "",
    ]

    bio_status = bio_result.get("status", "UNKNOWN")
    bio_passed = bio_result.get("passed", False)
    bio_dist = bio_result.get("distance")
    bio_sim = bio_result.get("similarity_pct", 0.0)

    if bio_status in ("MODULE_UNAVAILABLE", "ERROR"):
        lines.append(f"**Status:** ⚪ UNAVAILABLE — {bio_result.get('message', '')}")
    else:
        bio_icon = "✅ MATCH" if bio_passed else "❌ MISMATCH"
        lines += [
            f"**Status:** {bio_icon}  ",
            f"**Similarity Score:** {bio_sim:.1f}%  ",
            f"**Cosine Distance:** {bio_dist:.4f}" if bio_dist is not None else "**Cosine Distance:** —",
        ]

    lines += [
        "",
        "---",
        "",
        "## 5. VIZ Text Consistency (Printed Fields vs. MRZ)",
        "",
    ]

    if doc_type != "Passport":
        lines.append("**Status:** ➖ NOT APPLICABLE (VIZ cross-check against MRZ is specific to Passports).")
    elif not text_consistency.get("available"):
        lines.append("**Status:** ⚪ UNAVAILABLE — OCR could not extract printed fields for comparison.")
    else:
        lines += [
            "**Status:** ✅ Ran",
            "",
            "| Field | Match Ratio | Available |",
            "|-------|-------------|-----------|",
        ]
        for field_key, label in [("name", "Name"), ("dob", "Date of Birth"), ("doc_number", "Document Number")]:
            ratio = text_consistency.get(f"{field_key}_ratio")
            avail = text_consistency.get(f"fields_available", {}).get(field_key, True)
            ratio_str = f"{ratio:.0f}%" if ratio is not None else "—"
            avail_str = "✅" if avail else "⚪ Not Found"
            lines.append(f"| {label} | {ratio_str} | {avail_str} |")
        lines.append(f"| **Overall** | **{text_consistency.get('overall_ratio', 0):.0f}%** | — |")

    lines += [
        "",
        "---",
        "",
        "## 6. Document Quality Assessment",
        "",
        f"**Quality Grade:** {quality_report.quality_grade}  ",
        f"**Blurry:** {'Yes' if quality_report.is_blurry else 'No'}  ",
        f"**Glare / Overexposure:** {'Yes' if quality_report.has_glare else 'No'}  ",
        f"**Underexposed:** {'Yes' if quality_report.is_underexposed else 'No'}  ",
    ]
    if quality_report.issues:
        lines.append("")
        lines.append("**Issues identified:**")
        for issue in quality_report.issues:
            lines.append(f"- {issue}")

    lines += [
        "",
        "---",
        "",
        "## 7. Composite Risk Assessment",
        "",
        f"**Score:** {risk.get('score_int', '—')} / 100  ",
        f"**Verdict:** {verdict_emoji} **{risk.get('category', 'UNKNOWN')}**  ",
        "",
        "| Check | Raw Risk (0-100) | Effective Weight | Status |",
        "|-------|-----------------|------------------|--------|",
    ]
    for name, score in risk.get("sub_scores", {}).items():
        w_pct = risk.get("sub_weights_used", {}).get(name, 0) * 100
        lines.append(f"| {name} | {score:.1f} | {w_pct:.0f}% | ✅ |")
    for name in risk.get("incomplete", []):
        lines.append(f"| {name} | — | 0% | ⚪ Incomplete |")

    if risk.get("category") == "MANUAL REVIEW":
        lines += [
            "",
            "> ⚠️ **MANUAL REVIEW REQUIRED:** One or more critical checks (biometrics, text consistency) could not run. A definitive automated verdict cannot be issued. Human review is mandatory before making any decision.",
        ]

    lines += [
        "",
        "---",
        "",
        "## 8. Reviewer Decision",
        "",
        "*(To be completed by the reviewing officer)*",
        "",
        "- [ ] **APPROVE** — Document and identity verified; proceed.",
        "- [ ] **REJECT** — Document failed verification; do not proceed.",
        "- [ ] **ESCALATE** — Refer to supervisor / secondary inspection.",
        "",
        "**Reviewer Name:** ___________________________",
        "",
        "**Badge / ID:** ___________________________",
        "",
        "**Notes:**",
        "",
        "_______________________________________________________________________",
        "",
        "_______________________________________________________________________",
        "",
        "---",
        "",
        "*Report generated by PixelProof Educational Prototype. Not validated for operational use.*",
    ]

    return "\n".join(lines)
