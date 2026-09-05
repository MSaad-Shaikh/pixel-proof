"""
app.py  PixelProof - Educational Document Verification Demo
================================================================
Streamlit dashboard for the PixelProof hackathon prototype.

DISCLAIMER:
  This is an EDUCATIONAL PROTOTYPE built for a hackathon proof-of-concept.
  It has NOT been validated for real border or immigration decisions.
  Do NOT use it in any operational, law-enforcement, or immigration context.

Pipeline:
  1. Upload a document image (passport/ID).
  2. Upload or capture a traveller photo.
  3. MRZ Engine validates ICAO 9303 check digits.
  4. ELA Forensics checks for elevated compression-artifact scores.
  5. Biometrics compares passport photo vs. traveller photo.
  6. Risk Engine aggregates a 0-100 composite risk score.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

# Prevent TensorFlow/CUDA from attempting GPU lookups or crashing on CPU cloud containers
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import streamlit as st
from PIL import Image

from quality_engine import assess_document_quality, rotate_image, crop_image_normalized, QualityReport

# ---------------------------------------------------------------------------
# Logging configuration (do before any module imports that log)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PixelProof â€” Verification Console",
    page_icon="ðŸ›¡ï¸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Modern Smooth Styling (Subtle slate cards, clean badges, crisp typography)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Clean base styling */
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Subtle disclaimer bar */
    .subtle-disclaimer {
        background: #1e293b;
        border-left: 3px solid #64748b;
        color: #94a3b8;
        padding: 8px 14px;
        border-radius: 4px;
        font-size: 0.82rem;
        margin-bottom: 1.2rem;
    }
    
    /* Inspection Header */
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #1e293b;
        margin-bottom: 1.25rem;
    }
    .header-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .header-tag {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 3px 8px;
        background: #1e293b;
        color: #94a3b8;
        border-radius: 4px;
        font-weight: 600;
    }
    
    /* Modern Slate Card Container */
    .slate-card {
        background: #131b2e;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 1.15rem;
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    /* Refined Risk Summary Card */
    .risk-card {
        background: #131b2e;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
    }
    .risk-header-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
    }
    .risk-score-num {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1;
    }
    .risk-bar-bg {
        height: 6px;
        background: #1e293b;
        border-radius: 3px;
        overflow: hidden;
        margin-top: 0.5rem;
    }
    
    /* Subtle Badges */
    .badge-pass {
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.25);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .badge-fail {
        background: rgba(239, 68, 68, 0.12);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.25);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .badge-suspicious {
        background: rgba(245, 158, 11, 0.12);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.25);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .badge-neutral {
        background: rgba(100, 116, 139, 0.15);
        color: #94a3b8;
        border: 1px solid rgba(100, 116, 139, 0.25);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    /* Clean Inspection Field Rows */
    .field-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.35rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.03);
        font-size: 0.84rem;
    }
    .field-label {
        color: #94a3b8;
    }
    .field-value {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        color: #f1f5f9;
        font-weight: 500;
    }
    
    /* Clean Stepper / Landing Box */
    .guide-box {
        background: #131b2e;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    .guide-step {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        margin-bottom: 1rem;
    }
    .step-num {
        background: #1e293b;
        color: #38bdf8;
        width: 26px;
        height: 26px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        font-weight: 600;
        flex-shrink: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading MRZ / OCR engineâ€¦")
def get_passport_eye_reader():
    """Attempt to load PassportEye MRZReader (cached across sessions).

    Returns:
        MRZReader instance, or None if PassportEye is unavailable.
    """
    try:
        from passporteye import read_mrz
        return read_mrz
    except Exception as exc:
        logger.warning("PassportEye unavailable: %s", exc)
        return None


@st.cache_resource(show_spinner="Loading EasyOCR engineâ€¦")
def get_easy_ocr_reader():
    """Load EasyOCR (cached across sessions).

    Returns:
        easyocr.Reader instance, or None if unavailable.
    """
    try:
        import easyocr
        return easyocr.Reader(["en"], gpu=False, verbose=False)
    except Exception as exc:
        logger.warning("EasyOCR unavailable: %s", exc)
        return None


@st.cache_resource(show_spinner="Loading PaddleOCR engineâ€¦")
def get_paddle_ocr():
    """Load PaddleOCR (cached â€” initialisation is expensive).

    Returns:
        PaddleOCR instance, or None if unavailable.
    """
    try:
        from paddleocr import PaddleOCR
        return PaddleOCR(lang="en")
    except Exception as exc:
        logger.warning("PaddleOCR unavailable: %s", exc)
        return None


@st.cache_resource(show_spinner="Warming up DeepFace / ArcFace modelâ€¦")
def warmup_deepface() -> bool:
    """Trigger DeepFace model download/cache on startup (ArcFace).

    Returns:
        True if DeepFace loaded successfully, False otherwise.
    """
    try:
        import deepface.DeepFace  # noqa: F401
        return True
    except Exception as exc:
        logger.warning("DeepFace unavailable: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Helper: save uploaded file to a temporary path
# ---------------------------------------------------------------------------

def uploaded_to_temp(uploaded_file, suffix: str = ".jpg") -> Optional[str]:
    """Write a Streamlit UploadedFile to a named temp file.

    Args:
        uploaded_file: The st.UploadedFile object.
        suffix:        File extension for the temp file.

    Returns:
        Absolute path string, or None on failure.
    """
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(uploaded_file.getbuffer())
        tmp.close()
        return tmp.name
    except Exception as exc:
        logger.error("Could not write uploaded file to temp: %s", exc)
        return None


def cleanup_temp(path: Optional[str]) -> None:
    """Remove a temporary file silently.

    Args:
        path: File path to delete, or None (no-op).
    """
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# MRZ extraction helper (EasyOCR -> PaddleOCR -> PassportEye -> fallback)
# ---------------------------------------------------------------------------

def extract_mrz_text(image_path: str) -> tuple:
    """Extract raw MRZ and VIZ text from a document image using best-available method.

    Tries in order:
      1. EasyOCR full-page / MRZ text reader.
      2. PaddleOCR full-page OCR.
      3. PassportEye (purpose-built MRZ reader).

    Args:
        image_path: Path to the document image.

    Returns:
        Tuple (mrz_text: str, method_used: str, warning: str).
        mrz_text is the raw extracted string (may be empty on failure).
    """
    # Method 1: EasyOCR (fast, reliable, pure PyTorch, works natively on Windows)
    reader = get_easy_ocr_reader()
    if reader is not None:
        try:
            results = reader.readtext(image_path, detail=0)
            if results:
                combined = "\n".join(results)
                logger.info("Text extracted via EasyOCR (%d lines).", len(results))
                return combined, "EasyOCR", ""
        except Exception as exc:
            logger.warning("EasyOCR text extraction failed: %s", exc)

    # Method 2: PaddleOCR
    paddle = get_paddle_ocr()
    if paddle is not None:
        try:
            result = paddle.ocr(image_path)
            if result and result[0]:
                all_lines = []
                for item in result[0]:
                    if item and len(item) > 1 and item[1]:
                        text = item[1][0].strip()
                        if text:
                            all_lines.append(text)
                combined = "\n".join(all_lines)
                if combined:
                    logger.info("OCR text extracted via PaddleOCR.")
                    return combined, "PaddleOCR", ""
        except Exception as exc:
            logger.warning("PaddleOCR MRZ extraction failed: %s", exc)

    # Method 3: PassportEye
    read_mrz_fn = get_passport_eye_reader()
    if read_mrz_fn is not None:
        try:
            mrz = read_mrz_fn(image_path)
            if mrz is not None:
                raw = mrz.to_dict()
                lines = getattr(mrz, "raw_text", None) or raw.get("raw_text", "")
                if lines and len(lines.strip()) >= 30:
                    logger.info("MRZ extracted via PassportEye.")
                    return lines, "PassportEye", ""
        except Exception as exc:
            logger.warning("PassportEye extraction failed: %s", exc)

    # Fallback warning
    warning = (
        "Document OCR extraction was unable to resolve text from the image. "
        "Ensure the image has good lighting and text/MRZ regions are sharp and legible."
    )
    logger.warning(warning)
    return "", "fallback", warning


# ---------------------------------------------------------------------------
# UI Badge & Component Renderers
# ---------------------------------------------------------------------------

def render_badge(label: str, status: str = "pass") -> str:
    """Render a clean, subtle badge."""
    if status == "pass":
        return f'<span class="badge-pass">âœ“ {label}</span>'
    elif status == "fail":
        return f'<span class="badge-fail">âœ— {label}</span>'
    elif status == "suspicious":
        return f'<span class="badge-suspicious">âš  {label}</span>'
    return f'<span class="badge-neutral">{label}</span>'


def render_field_row(label: str, value: str, badge_html: Optional[str] = None) -> str:
    """Render a clean key-value row for document inspection."""
    display_val = value if value else "â€”"
    badge_part = f"&nbsp;&nbsp;{badge_html}" if badge_html else ""
    return f"""
    <div class="field-row">
        <span class="field-label">{label}</span>
        <span><span class="field-value">{display_val}</span>{badge_part}</span>
    </div>
    """


# ---------------------------------------------------------------------------
# Authentication Gate (Demo Login Interface)
# ---------------------------------------------------------------------------
DEMO_CREDENTIALS = {
    "officer_demo": "pixelproof2026",
    "admin": "admin123",
    "demo": "demo",
}

if not st.session_state.get("authenticated", False):
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    _, col_center, _ = st.columns([1, 2, 1])
    
    with col_center:
        st.markdown(
            """
            <div style="background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 1.8rem; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4); text-align: center; margin-bottom: 1.2rem;">
                <div style="font-size: 2.2rem; margin-bottom: 0.3rem;">ðŸ›¡ï¸</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.02em;">PixelProof Portal</div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 0.2rem;">Border Control & Forensic Verification Console</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        with st.form("login_form"):
            username = st.text_input("Officer Username / ID", placeholder="e.g. officer_demo")
            password = st.text_input("Passcode", type="password", placeholder="Enter passcode")
            submit_btn = st.form_submit_button("Log In to Workstation", type="primary", width='stretch')
            
            if submit_btn:
                if username in DEMO_CREDENTIALS and DEMO_CREDENTIALS[username] == password:
                    st.session_state["authenticated"] = True
                    st.session_state["user"] = username
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please verify your officer username and passcode.")
        
        st.markdown(
            """
            <div style="background: #0f172a; border: 1px dashed #3b82f6; border-radius: 8px; padding: 0.9rem; margin-top: 1rem; margin-bottom: 1rem;">
                <div style="font-size: 0.82rem; font-weight: 600; color: #60a5fa; margin-bottom: 0.2rem;">
                    ðŸ’¡ Hackathon Presentation Demo Account
                </div>
                <div style="font-size: 0.8rem; color: #cbd5e1;">
                    <strong>User:</strong> <code>officer_demo</code> &nbsp;|&nbsp; <strong>Password:</strong> <code>pixelproof2026</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        if st.button("âš¡ 1-Click Demo Login (Instant Access)", width='stretch'):
            st.session_state["authenticated"] = True
            st.session_state["user"] = "officer_demo"
            st.rerun()
            
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar: Input Panel
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ðŸ›¡ï¸ PixelProof")
    st.caption("Automated Document Verification System")
    
    # Active Officer indicator & logout button
    c_user, c_out = st.columns([3, 2])
    with c_user:
        st.caption(f"ðŸ‘¤ `{st.session_state.get('user', 'officer_demo')}`")
    with c_out:
        if st.button("Log Out", key="logout_btn", width='stretch'):
            st.session_state["authenticated"] = False
            st.session_state.pop("user", None)
            st.rerun()
    st.markdown("---")

    st.markdown("**1. Document Type**")
    doc_type = st.selectbox(
        "Select document type",
        ["Passport", "Visa", "National ID / Driving Licence"],
        key="doc_type",
        help="Choose the type of document you are uploading.",
    )

    _upload_labels = {
        "Passport": "Upload passport image",
        "Visa": "Upload visa sticker / stamp image",
        "National ID / Driving Licence": "Upload ID card / driving licence image",
    }

    st.markdown("**2. Document Upload**")
    doc_upload = st.file_uploader(
        _upload_labels[doc_type],
        type=["jpg", "jpeg", "png"],
        key="doc_upload",
        help="Supported formats: JPEG, PNG",
    )

    # Orientation Selection Menu
    doc_rotation = 0
    if doc_upload is not None:
        doc_rotation = st.selectbox(
            "ðŸ”„ Orientation / Rotation",
            [0, 90, 180, 270],
            format_func=lambda x: f"{x}Â° (Normal)" if x == 0 else (f"{x}Â° (Clockwise)" if x == 90 else (f"{x}Â° (Inverted)" if x == 180 else f"{x}Â° (Counter-CW)")),
            key="doc_rotation",
        )

    st.markdown("**3. Traveller Photo**")
    photo_method = st.radio(
        "Input source",
        ["Webcam capture", "Upload file"],
        horizontal=True,
        key="photo_method",
    )

    live_image_data = None
    if photo_method == "Webcam capture":
        live_image_data = st.camera_input(
            "Live camera capture",
            key="webcam",
            help="Position face within frame.",
        )
    else:
        live_image_data = st.file_uploader(
            "Upload live photo",
            type=["jpg", "jpeg", "png"],
            key="live_upload",
        )

    st.markdown("---")
    run_btn = st.button("Run Document Verification", type="primary", width='stretch')

# Pre-warm deepface in background
deepface_ok = warmup_deepface()

# ---------------------------------------------------------------------------
# Main Workstation Canvas
# ---------------------------------------------------------------------------

# Top Header
st.markdown(
    """
    <div class="header-container">
        <div class="header-title">
            <span>ðŸ›¡ï¸ PixelProof</span>
            <span class="header-tag">Inspection Workstation</span>
        </div>
        <div style="font-size: 0.8rem; color: #64748b;">ICAO Doc 9303 TD3 Compliant</div>
    </div>
    <div class="subtle-disclaimer">
        â„¹ï¸ <strong>Educational Prototype:</strong> Evaluates document integrity heuristics (MRZ checksums, compression artifacts, facial biometrics). Not certified for operational border decisions.
    </div>
    """,
    unsafe_allow_html=True,
)

# Idle / Welcome Landing View
if not run_btn:
    col_w1, col_w2 = st.columns([3, 2])
    with col_w1:
        st.markdown(
            """
            <div class="guide-box">
                <div style="font-size: 1rem; font-weight: 600; color: #f1f5f9; margin-bottom: 1rem;">
                    Verification Pipeline Overview
                </div>
                <div class="guide-step">
                    <div class="step-num">1</div>
                    <div>
                        <strong style="color: #e2e8f0;">Document & MRZ Verification</strong>
                        <div style="font-size: 0.82rem; color: #94a3b8;">
                            Parses ICAO 9303 TD3 2-line machine-readable zone and validates modular-10 check digits.
                        </div>
                    </div>
                </div>
                <div class="guide-step">
                    <div class="step-num">2</div>
                    <div>
                        <strong style="color: #e2e8f0;">Error Level Forensics (ELA)</strong>
                        <div style="font-size: 0.82rem; color: #94a3b8;">
                            Analyzes localized JPEG compression artifacts to identify elevated digital edit discrepancies.
                        </div>
                    </div>
                </div>
                <div class="guide-step">
                    <div class="step-num">3</div>
                    <div>
                        <strong style="color: #e2e8f0;">ArcFace Biometric Matching</strong>
                        <div style="font-size: 0.82rem; color: #94a3b8;">
                            Automatically isolates passport portrait and compares against traveller live frame.
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_w2:
        # --- Live dependency status checks ---
        def _dep_check(import_path: str) -> bool:
            try:
                __import__(import_path)
                return True
            except Exception:
                return False

        dep_status = {
            "EasyOCR (Primary OCR Engine)": (_dep_check("easyocr"), False),
            "DeepFace / ArcFace":           (deepface_ok, False),
            "TensorFlow":                   (_dep_check("tensorflow"), False),
            "OpenCV Vision Engine":         (_dep_check("cv2"), False),
            "ELA Forensics Module":         (True, False),
            "MRZ Multi-Format Engine":      (True, False),
            "Risk Scoring Aggregator":      (True, False),
            "PaddleOCR (Optional Fallback)": (_dep_check("paddleocr"), True),
            "PassportEye (Optional Fallback)": (_dep_check("passporteye"), True),
        }

        rows_html = ""
        for dep_name, (ok, optional) in dep_status.items():
            if ok:
                badge = '<span class="badge-pass">âœ“ Active</span>'
            elif optional:
                badge = '<span style="color:#94a3b8;font-size:0.75rem;padding:2px 6px;border-radius:4px;background:#334155;">Optional</span>'
            else:
                badge = '<span class="badge-fail">âœ— Missing</span>'
            rows_html += f"""
            <div class="field-row">
                <span class="field-label">{dep_name}</span>
                {badge}
            </div>"""

        st.markdown(
            f"""
            <div class="guide-box">
                <div style="font-size: 0.95rem; font-weight: 600; color: #f1f5f9; margin-bottom: 0.8rem;">
                    ðŸ”§ Engine & Dependency Status
                </div>
                {rows_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        critical_missing = [k for k, (ok, optional) in dep_status.items() if not ok and not optional]
        if critical_missing:
            st.caption(f"âš ï¸ Missing critical components: {', '.join(critical_missing)}.")
        else:
            st.caption("ðŸŸ¢ All primary verification engines are online and ready.")
    st.stop()

# ---------------------------------------------------------------------------
# Validate inputs
# ---------------------------------------------------------------------------

if doc_upload is None:
    st.error("Please upload a document image before running.")
    st.stop()

if live_image_data is None:
    st.error("Please provide a traveller photo (webcam or upload) before running.")
    st.stop()

# ---------------------------------------------------------------------------
# Save uploads to temp files
# ---------------------------------------------------------------------------

doc_ext = Path(doc_upload.name).suffix.lower() if hasattr(doc_upload, "name") else ".jpg"
if doc_ext not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}:
    doc_ext = ".jpg"

# F1: Save original bytes untouched â€” this is the evidence-anchor file
doc_orig_temp = uploaded_to_temp(doc_upload, suffix=doc_ext)
# Also save for OCR/ELA derivative (will be overwritten with rotated version)
doc_upload.seek(0)
doc_proc_temp = uploaded_to_temp(doc_upload, suffix=".jpg")

live_ext = ".jpg"
if hasattr(live_image_data, "name"):
    live_ext = Path(live_image_data.name).suffix.lower() or ".jpg"
live_temp = uploaded_to_temp(live_image_data, suffix=live_ext)

if doc_orig_temp is None or doc_proc_temp is None or live_temp is None:
    st.error("Could not save uploaded files. Please try again.")
    cleanup_temp(doc_orig_temp)
    cleanup_temp(doc_proc_temp)
    cleanup_temp(live_temp)
    st.stop()

# F1: Compute SHA-256 of original bytes â€” immutable evidence hash
import hashlib as _hashlib
try:
    with open(doc_orig_temp, "rb") as _f:
        doc_sha256 = _hashlib.sha256(_f.read()).hexdigest()
except Exception:
    doc_sha256 = "unavailable"

try:
    # -------------------------------------------------------------------
    # Load images as numpy arrays for display / biometrics
    # -------------------------------------------------------------------
    try:
        raw_pil = Image.open(doc_orig_temp).convert("RGB")

        # === MEMORY GUARD ===
        # Cap image resolution at 2000px on longest side to prevent OOM.
        # Streamlit Cloud has 1 GB RAM; a 4000x3000 image creates a ~36 MB
        # numpy array which compounds during ELA, face cropping, heatmaps etc.
        MAX_SIDE = 2000
        _w, _h = raw_pil.size
        if max(_w, _h) > MAX_SIDE:
            _scale = MAX_SIDE / max(_w, _h)
            raw_pil = raw_pil.resize((int(_w * _scale), int(_h * _scale)), Image.LANCZOS)
            # Re-save the original temp with the resized version so ELA runs on same size
            raw_pil.save(doc_orig_temp, format="JPEG", quality=95)

        if doc_rotation != 0:
            doc_np = rotate_image(np.array(raw_pil), doc_rotation)
            doc_pil = Image.fromarray(doc_np)
        else:
            doc_pil = raw_pil
            doc_np = np.array(doc_pil)

        # F4: Auto-deskew the processing derivative for better OCR
        from perspective import auto_deskew
        doc_np_deskewed, was_deskewed = auto_deskew(doc_np)
        doc_proc_note = (
            f"Rotation {doc_rotation}Â° applied; perspective correction applied."
            if was_deskewed else
            f"Rotation {doc_rotation}Â° applied; no perspective correction needed."
            if doc_rotation != 0 else
            "No rotation or perspective correction applied."
        )

        # Save deskewed derivative for OCR (not for ELA â€” ELA uses orig)
        doc_proc_pil = Image.fromarray(doc_np_deskewed)
        doc_proc_pil.save(doc_proc_temp, format="JPEG", quality=95)

        # Free intermediate PIL objects; keep doc_np for biometrics display
        del raw_pil, doc_proc_pil

    except Exception as exc:
        st.error(f"Could not open document image: {exc}")
        cleanup_temp(doc_orig_temp)
        cleanup_temp(doc_proc_temp)
        cleanup_temp(live_temp)
        st.stop()

    try:
        live_pil = Image.open(live_temp).convert("RGB")
        live_np = np.array(live_pil)
    except Exception as exc:
        st.error(f"Could not open traveller photo: {exc}")
        cleanup_temp(doc_orig_temp)
        cleanup_temp(doc_proc_temp)
        cleanup_temp(live_temp)
        st.stop()

    # -----------------------------------------------------------------------
    # Pre-Flight Document Quality Assessment (uses display array from orig)
    # -----------------------------------------------------------------------
    quality_report = assess_document_quality(doc_np)

    # -----------------------------------------------------------------------
    # Step 1: Document Field Extraction (route by doc_type)
    # -----------------------------------------------------------------------
    from mrz_engine import MRZResult, parse_and_validate, format_date

    visa_result = None
    id_result = None
    mrz_method = ""

    if doc_type == "Passport":
        with st.spinner("Extracting MRZâ€¦"):
            mrz_text, mrz_method, mrz_warning = extract_mrz_text(doc_proc_temp)
        if mrz_warning:
            st.warning(f"MRZ extraction: {mrz_warning}")
        try:
            if mrz_text:
                mrz_result = parse_and_validate(mrz_text)
            else:
                mrz_result = MRZResult(
                    status="MRZ_UNREADABLE",
                    failed_explanations=["MRZ text could not be extracted from the image."],
                )
        except Exception as exc:
            st.warning(f"MRZ validation error: {exc}")
            mrz_result = MRZResult(status="MRZ_UNREADABLE", failed_explanations=[str(exc)])

    elif doc_type == "Visa":
        mrz_method = "EasyOCR (Visa Engine)"
        with st.spinner("Extracting visa fields (EasyOCR)â€¦"):
            from visa_engine import extract_visa_fields
            try:
                reader = get_easy_ocr_reader()
                visa_result = extract_visa_fields(doc_proc_temp, reader=reader)
            except Exception as exc:
                logger.error("Visa engine error: %s", exc)
                from visa_engine import VisaResult
                visa_result = VisaResult(status="UNREADABLE", confidence_note=str(exc))
        mrz_result = MRZResult(
            status="MRZ_UNREADABLE",
            failed_explanations=["MRZ not applicable for Visa documents."],
        )

    else:  # National ID / Driving Licence
        mrz_method = "EasyOCR (ID Engine)"
        with st.spinner("Extracting ID fields (EasyOCR)â€¦"):
            from id_engine import extract_id_fields
            try:
                reader = get_easy_ocr_reader()
                id_result = extract_id_fields(doc_proc_temp, reader=reader)
            except Exception as exc:
                logger.error("ID engine error: %s", exc)
                from id_engine import IDResult
                id_result = IDResult(status="UNREADABLE", confidence_note=str(exc))
        mrz_result = MRZResult(
            status="MRZ_UNREADABLE",
            failed_explanations=["MRZ not applicable for ID/Licence documents."],
        )


    # -----------------------------------------------------------------------
    # Step 2: ELA Forensics (ALWAYS on original bytes â€” never on derivative)
    # -----------------------------------------------------------------------
    with st.spinner("Running ELA forensicsâ€¦"):
        from ela_forensics import perform_ela, heatmap_to_rgb
        try:
            ela_result = perform_ela(doc_orig_temp)
        except Exception as exc:
            logger.error("ELA module error: %s", exc)
            ela_result = {"error": str(exc), "mean_error": None, "alert": False,
                          "heatmap_bgr": None, "ela_gray": None, "caveat": ""}

    # -----------------------------------------------------------------------
    # Step 3: Biometric Verification
    # -----------------------------------------------------------------------
    bio_result = {
        "status": "MODULE_UNAVAILABLE",
        "message": "DeepFace not loaded.",
        "distance": None,
        "passed": False,
        "similarity_pct": 0.0,
        "passport_bbox": None,
        "live_bbox": None,
    }

    if deepface_ok:
        with st.spinner("Running face verification (ArcFace)â€¦"):
            from biometrics import verify_faces, draw_bbox_on_image
            try:
                bio_result = verify_faces(doc_np, live_np)
            except Exception as exc:
                logger.error("Biometrics module error: %s", exc)
                bio_result["status"] = "ERROR"
                bio_result["message"] = str(exc)

    import gc
    gc.collect()

    # -----------------------------------------------------------------------
    # Step 4: Text Consistency (VIZ vs MRZ) â€” specific to Passports
    # -----------------------------------------------------------------------
    text_consistency = {
        "overall_ratio": None, "available": False,
        "name_ratio": None, "dob_ratio": None, "doc_ratio": None,
        "fields_available": {"name": False, "dob": False, "doc_number": False},
    }
    if doc_type == "Passport":
        with st.spinner("Checking text consistencyâ€¦"):
            from risk_engine import extract_viz_fields_with_ocr, compute_text_consistency
            try:
                reader = get_easy_ocr_reader()
                viz_fields = extract_viz_fields_with_ocr(doc_proc_temp, reader=reader)
                text_consistency = compute_text_consistency(
                    viz_fields=viz_fields,
                    mrz_surname=mrz_result.surname,
                    mrz_given=mrz_result.given_names,
                    mrz_dob=mrz_result.date_of_birth,
                    mrz_doc_number=mrz_result.document_number,
                )
            except Exception as exc:
                logger.error("Text consistency error: %s", exc)

    # -----------------------------------------------------------------------
    # Step 5: Composite Risk Score
    # -----------------------------------------------------------------------
    from risk_engine import compute_risk_score

    # For Passports: MRZ fail = security failure. For Aadhaar/DL/Visa: MRZ is
    # not applicable â€” pass None so the weight is redistributed across other checks.
    if doc_type == "Passport":
        mrz_passed_val = True if mrz_result.status == "OK" else False
    else:
        mrz_passed_val = None  # Not applicable â€” no MRZ strip on these documents

    ela_error_val = ela_result.get("mean_error") if not ela_result.get("error") else None
    bio_dist_val = bio_result.get("distance")
    # Text consistency against MRZ only makes sense for passports
    text_ratio_val = text_consistency.get("overall_ratio") if (doc_type == "Passport" and text_consistency.get("available")) else None

    try:
        risk = compute_risk_score(
            mrz_passed=mrz_passed_val,
            ela_mean_error=ela_error_val,
            biometric_distance=bio_dist_val,
            text_ratio=text_ratio_val,
        )
    except Exception as exc:
        logger.error("Risk engine error: %s", exc)
        risk = {
            "score": 0, "score_int": 0, "category": "ERROR",
            "color": "gray", "sub_scores": {}, "sub_weights_used": {},
            "incomplete": ["All checks"], "manual_review": False,
        }

    # =======================================================================
    # DISPLAY RESULTS
    # =======================================================================

    # -- Executive Risk Summary Bar -----------------------------------------
    cat_badge_map = {
        "LOW RISK":      render_badge("LOW RISK", "pass"),
        "SUSPICIOUS":    render_badge("SUSPICIOUS", "suspicious"),
        "CRITICAL RISK": render_badge("CRITICAL RISK", "fail"),
        "MANUAL REVIEW": render_badge("âš  MANUAL REVIEW", "suspicious"),
        "INCOMPLETE":    render_badge("INCOMPLETE", "neutral"),
    }
    cat_badge = cat_badge_map.get(risk["category"], render_badge(risk["category"], "neutral"))

    accent_bar_color = {
        "green":  "#10b981",
        "yellow": "#f59e0b",
        "red":    "#ef4444",
        "orange": "#f97316",
        "gray":   "#64748b",
    }.get(risk.get("color", "gray"), "#64748b")

    st.markdown(
        f"""
        <div class="risk-card">
            <div class="risk-header-row">
                <div>
                    <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; margin-bottom: 2px;">
                        Composite Verification Assessment
                    </div>
                    <div style="display: flex; align-items: baseline; gap: 12px;">
                        <span class="risk-score-num">{risk['score_int']}<span style="font-size: 1rem; color: #64748b; font-weight: 400;"> / 100</span></span>
                        <span>{cat_badge}</span>
                    </div>
                </div>
                <div style="text-align: right; font-size: 0.82rem; color: #94a3b8;">
                    MRZ: <strong>{'VALID' if mrz_result.all_checks_passed else 'FAIL'}</strong> &nbsp;|&nbsp; 
                    ELA: <strong>{'ELEVATED' if ela_result.get('alert') else 'NORMAL'}</strong> &nbsp;|&nbsp; 
                    Face: <strong>{'MATCH' if bio_result.get('passed') else 'MISMATCH'}</strong> &nbsp;|&nbsp; 
                    Quality: <strong>{quality_report.quality_grade}</strong>
                </div>
            </div>
            <div class="risk-bar-bg">
                <div style="width: {risk['score_int']}%; height: 100%; background: {accent_bar_color}; border-radius: 3px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -- F2: MANUAL REVIEW Banner -------------------------------------------
    if risk.get("manual_review"):
        st.warning(
            "âš ï¸ **MANUAL REVIEW REQUIRED** â€” Biometric face verification and text "
            "consistency checks were both unavailable. A definitive automated verdict "
            "cannot be issued. Human review by an authorised officer is mandatory before "
            "making any decision on this document."
        )

    # -- F1: Evidence Anchor (SHA-256) + Processing Note --------------------
    orig_name = getattr(doc_upload, "name", "uploaded_document")
    _proc_cols = st.columns([3, 2])
    with _proc_cols[0]:
        st.caption(f"ðŸ“Ž **Source:** `{orig_name}` | **SHA-256:** `{doc_sha256}`")
    with _proc_cols[1]:
        st.caption(f"ðŸ”„ {doc_proc_note}")

    # -- Pre-Flight Quality Diagnostics Bar ---------------------------------
    q_badge_style = "pass" if quality_report.passed else ("suspicious" if quality_report.quality_grade == "FAIR" else "fail")
    with st.expander(f"ðŸ” Pre-Flight Quality Diagnostics (Grade: {quality_report.quality_grade})", expanded=not quality_report.passed):
        q_c1, q_c2, q_c3, q_c4 = st.columns(4)
        q_c1.metric("Sharpness (Blur)", f"{quality_report.blur_score:.1f}", 
                    delta="Sharp" if not quality_report.is_blurry else "Blurry Alert",
                    delta_color="normal" if not quality_report.is_blurry else "inverse")
        q_c2.metric("Glare / Specular", f"{quality_report.glare_ratio*100:.1f}%",
                    delta="Low Glare" if not quality_report.has_glare else "High Glare",
                    delta_color="normal" if not quality_report.has_glare else "inverse")
        q_c3.metric("Luminance (Light)", f"{quality_report.mean_brightness:.1f} / 255",
                    delta="Good Light" if not quality_report.is_underexposed else "Low Light",
                    delta_color="normal" if not quality_report.is_underexposed else "inverse")
        q_c4.metric("Framing & Orientation", quality_report.orientation_status,
                    delta="Margins Clear" if not quality_report.is_clipped else "Tight Margin",
                    delta_color="normal" if not quality_report.is_clipped else "inverse")

        if quality_report.issues:
            st.markdown("**Identified Quality Factors:**")
            for issue, rec in zip(quality_report.issues, quality_report.recommendations):
                st.markdown(f"- âš ï¸ **{issue}**: {rec}")

    if risk["incomplete"]:
        st.caption(f"â„¹ï¸ Non-fatal: Excluded uncomputed checks from weight: {', '.join(risk['incomplete'])}")

    # -- Three Inspection Modules Grid --------------------------------------
    col1, col2, col3 = st.columns(3)


    # CARD 1: Document Field Analysis (adapts to doc_type)
    with col1:

        # ---- VISA CARD ----
        if doc_type == "Visa" and visa_result is not None:
            v_status_badge = (
                render_badge("EXTRACTED", "pass") if visa_result.status == "OK"
                else render_badge("PARTIAL", "suspicious") if visa_result.status == "PARTIAL"
                else render_badge("UNREADABLE", "fail")
            )
            st.markdown(
                f"""
                <div class="slate-card">
                    <div class="card-title">
                        <span>ðŸ›‚ Visa Field Extraction</span>
                        {v_status_badge}
                    </div>
                """,
                unsafe_allow_html=True,
            )
            if visa_result.status == "UNREADABLE":
                st.error("âŒ Could not extract visa fields from this image.")
                st.caption(visa_result.confidence_note)
            else:
                visa_fields = [
                    ("Visa Number",      visa_result.visa_number or "â€”",    None),
                    ("Visa Type",        visa_result.visa_type or "â€”",      None),
                    ("Country of Issue", visa_result.country_of_issue or "â€”", None),
                    ("Valid From",       visa_result.valid_from or "â€”",     None),
                    ("Valid Until",      visa_result.valid_until or "â€”",    None),
                    ("No. of Entries",   visa_result.entries or "â€”",        None),
                    ("Stay Duration",    visa_result.stay_duration or "â€”",  None),
                    ("Applicant Name",   visa_result.applicant_name or "â€”", None),
                    ("Nationality",      visa_result.nationality or "â€”",    None),
                ]
                rows_html = "".join([render_field_row(l, v, b) for l, v, b in visa_fields])
                st.markdown(rows_html, unsafe_allow_html=True)
                st.caption(visa_result.confidence_note)
                with st.expander("Raw OCR Lines", expanded=False):
                    st.code("\n".join(visa_result.raw_lines), language=None)
            st.markdown("</div>", unsafe_allow_html=True)

        # ---- ID / LICENCE CARD ----
        elif doc_type == "National ID / Driving Licence" and id_result is not None:
            id_status_badge = (
                render_badge("EXTRACTED", "pass") if id_result.status == "OK"
                else render_badge("PARTIAL", "suspicious") if id_result.status == "PARTIAL"
                else render_badge("UNREADABLE", "fail")
            )
            st.markdown(
                f"""
                <div class="slate-card">
                    <div class="card-title">
                        <span>ðŸªª {id_result.doc_subtype} Field Extraction</span>
                        {id_status_badge}
                    </div>
                """,
                unsafe_allow_html=True,
            )
            if id_result.status == "UNREADABLE":
                st.error("âŒ Could not extract ID fields from this image.")
                st.caption(id_result.confidence_note)
            else:
                id_fields = [
                    ("Document Subtype",    id_result.doc_subtype,              None),
                    ("ID / Licence No.",    id_result.id_number or "â€”",         None),
                    ("Full Name",           id_result.full_name or "â€”",         None),
                    ("Date of Birth",       id_result.date_of_birth or "â€”",     None),
                    ("Gender",              id_result.gender or "â€”",            None),
                    ("Blood Group",         id_result.blood_group or "â€”",       None),
                    ("Issue Date",          id_result.issue_date or "â€”",        None),
                    ("Expiry Date",         id_result.expiry_date or "â€”",       None),
                    ("Issuing Authority",   id_result.issuing_authority or "â€”", None),
                    ("Nationality",         id_result.nationality or "â€”",       None),
                    ("Guardian / Relation", id_result.relation_name or "â€”",     None),
                    ("Address",             id_result.address or "â€”",           None),
                ]
                rows_html = "".join([render_field_row(l, v, b) for l, v, b in id_fields])
                st.markdown(rows_html, unsafe_allow_html=True)
                st.caption(id_result.confidence_note)
                with st.expander("Raw OCR Lines", expanded=False):
                    st.code("\n".join(id_result.raw_lines), language=None)
            st.markdown("</div>", unsafe_allow_html=True)

        # ---- PASSPORT / MRZ CARD (unchanged) ----
        else:
            st.markdown(
                f"""
                <div class="slate-card">
                    <div class="card-title">
                        <span>ðŸ“‹ ICAO {getattr(mrz_result, 'mrz_format', 'TD3')} MRZ</span>
                        {render_badge('PASSED', 'pass') if mrz_result.all_checks_passed else render_badge('CHECKSUM FAIL', 'fail')}
                    </div>
                """,
                unsafe_allow_html=True,
            )

            if mrz_result.status == "MRZ_UNREADABLE":
                exp_text = " ".join(mrz_result.failed_explanations).lower() if mrz_result.failed_explanations else ""
                if quality_report.is_blurry:
                    st.error("âŒ MRZ Unreadable â€” Image Too Blurry")
                    st.markdown("- **Fix:** Hold the camera steady and ensure the document is flat.")
                elif quality_report.has_glare:
                    st.error("âŒ MRZ Unreadable â€” Glare / Overexposure Detected")
                    st.markdown("- **Fix:** Tilt the document slightly or diffuse the light source.")
                elif quality_report.is_underexposed:
                    st.error("âŒ MRZ Unreadable â€” Low Light / Underexposed")
                    st.markdown("- **Fix:** Move to a brighter environment before rescanning.")
                elif "length" in exp_text or "44" in exp_text:
                    st.error("âŒ MRZ Unreadable â€” Could Not Find Two 44-Character MRZ Lines")
                    st.markdown(
                        "- The document bottom edge may be cropped. Try adjusting the **Orientation** slider."
                    )
                elif "applicable" in exp_text:
                    st.info("â„¹ï¸ MRZ validation is only applicable to Passport documents.")
                else:
                    st.error("âŒ MRZ Unreadable â€” Non-Standard or Malformed MRZ")
                if mrz_result.failed_explanations and "applicable" not in exp_text:
                    with st.expander("ðŸ” Technical Detail", expanded=False):
                        for exp in mrz_result.failed_explanations:
                            st.caption(f"â€¢ {exp}")
            else:
                from mrz_engine import lookup_country
                fields = [
                    ("Document Type",  mrz_result.document_type,                  None),
                    ("Issuing State",   lookup_country(mrz_result.issuing_state),  None),
                    ("Nationality",     lookup_country(mrz_result.nationality),    None),
                    ("Surname",         mrz_result.surname,                        None),
                    ("Given Names",     mrz_result.given_names,                    None),
                    ("Date of Birth",   format_date(mrz_result.date_of_birth),
                     render_badge("PASS" if (mrz_result.dob_check and mrz_result.dob_check.passed) else "FAIL",
                                  "pass" if (mrz_result.dob_check and mrz_result.dob_check.passed) else "fail")),
                    ("Expiry Date",     format_date(mrz_result.expiration_date, is_expiry=True),
                     render_badge("PASS" if (mrz_result.expiry_check and mrz_result.expiry_check.passed) else "FAIL",
                                  "pass" if (mrz_result.expiry_check and mrz_result.expiry_check.passed) else "fail")),
                    ("Document No.",    mrz_result.document_number,
                     render_badge("PASS" if (mrz_result.document_number_check and mrz_result.document_number_check.passed) else "FAIL",
                                  "pass" if (mrz_result.document_number_check and mrz_result.document_number_check.passed) else "fail")),
                    ("Composite Check", "â€”",
                     render_badge("PASS" if (mrz_result.composite_check and mrz_result.composite_check.passed) else "FAIL",
                                  "pass" if (mrz_result.composite_check and mrz_result.composite_check.passed) else "fail")),
                ]
                rows_html = "".join([render_field_row(l, v, b) for l, v, b in fields])
                st.markdown(rows_html, unsafe_allow_html=True)
                with st.expander("Raw TD3 String", expanded=False):
                    st.code(f"{mrz_result.raw_line1}\n{mrz_result.raw_line2}", language=None)

            st.markdown("</div>", unsafe_allow_html=True)


    # CARD 2: Forensic ELA
    with col2:
        from ela_forensics import ELA_ALERT_THRESHOLD
        ela_badge = render_badge("ELEVATED", "suspicious") if ela_result.get("alert") else render_badge("NORMAL", "pass")
        st.markdown(
            f"""
            <div class="slate-card">
                <div class="card-title">
                    <span>ðŸ”¬ Error Level Analysis</span>
                    {ela_badge}
                </div>
            """,
            unsafe_allow_html=True,
        )

        if ela_result.get("error"):
            st.caption(f"ELA Unavailable: {ela_result['error']}")
        else:
            mean_val = ela_result.get("mean_error", 0.0)
            st.markdown(
                f"""
                <div class="field-row">
                    <span class="field-label">Mean Artifact Intensity</span>
                    <span class="field-value">{mean_val:.2f}</span>
                </div>
                <div class="field-row">
                    <span class="field-label">Alert Threshold</span>
                    <span class="field-value">{ELA_ALERT_THRESHOLD:.1f}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if ela_result.get("caveat"):
                st.caption(f"â„¹ï¸ {ela_result['caveat']}")

            sub_img1, sub_img2 = st.columns(2)
            with sub_img1:
                st.image(doc_pil, caption="Original Doc", width='stretch')
            with sub_img2:
                heatmap_bgr = ela_result.get("heatmap_bgr")
                if heatmap_bgr is not None:
                    st.image(heatmap_to_rgb(heatmap_bgr), caption="ELA Heatmap", width='stretch')
                else:
                    st.caption("Heatmap unavailable")

        st.markdown("</div>", unsafe_allow_html=True)

    # CARD 3: Biometric Face Verification
    with col3:
        from biometrics import ARCFACE_COSINE_THRESHOLD
        bio_status = bio_result.get("status", "ERROR")
        if bio_status == "OK":
            bio_badge = render_badge("MATCH", "pass") if bio_result.get("passed") else render_badge("MISMATCH", "fail")
        else:
            bio_badge = render_badge(bio_status, "neutral")

        st.markdown(
            f"""
            <div class="slate-card">
                <div class="card-title">
                    <span>ðŸ§¬ Face Verification</span>
                    {bio_badge}
                </div>
            """,
            unsafe_allow_html=True,
        )

        if bio_status == "MODULE_UNAVAILABLE":
            st.caption("Biometric engine unavailable in current environment.")
        elif bio_status in ("NO_FACE_PASSPORT", "NO_FACE_LIVE", "MULTIPLE_FACES"):
            st.caption(f"âš ï¸ {bio_result.get('message')}")
        elif bio_status == "ERROR":
            st.caption(f"Verification error: {bio_result.get('message')}")
        else:
            dist = bio_result.get("distance", 1.0)
            sim = bio_result.get("similarity_pct", 0.0)

            st.markdown(
                f"""
                <div class="field-row">
                    <span class="field-label">Similarity Score</span>
                    <span class="field-value">{sim:.1f}%</span>
                </div>
                <div class="field-row">
                    <span class="field-label">Cosine Distance</span>
                    <span class="field-value">{dist:.4f} <span style="font-size:0.75rem;color:#64748b;">(&le; {ARCFACE_COSINE_THRESHOLD})</span></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            passport_photo_crop = bio_result.get("passport_crop")
            l_box = bio_result.get("live_bbox")

            img_b1, img_b2 = st.columns(2)
            with img_b1:
                if passport_photo_crop is not None:
                    st.image(passport_photo_crop, caption="Passport Portrait (Auto-Cropped)", width='stretch')
                else:
                    st.image(doc_pil, caption="Passport Document", width='stretch')

            with img_b2:
                try:
                    annotated_l = draw_bbox_on_image(live_np, l_box, color_rgb=(52, 211, 153))
                    st.image(annotated_l, caption="Live Photo (Face Tracked)", width='stretch')
                except Exception:
                    st.image(live_pil, caption="Live Photo", width='stretch')

        st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Risk Breakdown & Text Consistency Tabular Area
    # -----------------------------------------------------------------------
    with st.expander("Detailed Risk Component Breakdown", expanded=(risk.get("score_int", 0) >= 60)):
        import pandas as pd

        # Plain-English explanations for each check
        check_explanations = {
            "MRZ Checksum":       "Mathematical checksum validation of the Machine Readable Zone (passport only).",
            "ELA Intensity":      "Digital forgery detection via Error Level Analysis â€” checks for pixel-level editing.",
            "Biometric Distance": "ArcFace AI face match between document photo and live/uploaded traveller photo.",
            "Text Consistency":   "Fuzzy cross-check of printed fields vs. MRZ encoded data (passport only).",
        }
        check_status = {
            "MRZ Checksum":       "âœ… All ICAO check digits verified" if mrz_passed_val else ("âž– Not applicable for this document type" if mrz_passed_val is None else "âŒ One or more check digits failed"),
            "ELA Intensity":      f"âœ… Normal compression artifact level ({ela_error_val:.1f})" if ela_error_val is not None and not ela_result.get("alert") else ("âš ï¸ Elevated artifacts detected" if ela_result.get("alert") else "âž– ELA not available"),
            "Biometric Distance": f"âœ… Face match (distance {bio_dist_val:.3f})" if bio_dist_val is not None and bio_result.get("passed") else (f"âŒ Face mismatch (distance {bio_dist_val:.3f})" if bio_dist_val is not None else "âž– No live photo provided"),
            "Text Consistency":   f"âœ… Fields consistent ({text_ratio_val:.0f}% match)" if text_ratio_val is not None and text_ratio_val >= 70 else (f"âš ï¸ Low field match ({text_ratio_val:.0f}%)" if text_ratio_val is not None else "âž– Not applicable for this document type"),
        }

        rows = []
        for name, score in risk["sub_scores"].items():
            w_pct = risk["sub_weights_used"].get(name, 0) * 100
            contrib = score * risk["sub_weights_used"].get(name, 0)
            flag = "ðŸ”´ High" if score >= 60 else ("ðŸŸ¡ Moderate" if score >= 30 else "ðŸŸ¢ Low")
            rows.append({
                "Check": name,
                "Status": check_status.get(name, "â€”"),
                "Risk Level": flag,
                "Risk Score": f"{score:.0f}/100",
                "Weight": f"{w_pct:.0f}%",
                "Contribution": f"+{contrib:.1f}",
            })
        for name in risk["incomplete"]:
            na_reason = "Not applicable for this document type" if name in ["MRZ Checksum", "Text Consistency"] and doc_type != "Passport" else "Check was unavailable or skipped"
            rows.append({
                "Check": name,
                "Status": f"âž– {na_reason}",
                "Risk Level": "âž– N/A",
                "Risk Score": "N/A",
                "Weight": "0% (excluded)",
                "Contribution": "0.0",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

        # Show what's driving the risk if high
        high_checks = [(n, s) for n, s in risk.get("sub_scores", {}).items() if s >= 60]
        if high_checks:
            st.markdown("##### âš ï¸ Risk Factors Detected")
            for name, score in sorted(high_checks, key=lambda x: -x[1]):
                st.warning(f"**{name}** â€” {check_explanations.get(name, '')}  \nRisk contribution: **{score:.0f}/100**. {check_status.get(name, '')}")

        if doc_type != "Passport":
            st.info("â„¹ï¸ **Note:** MRZ Checksum and Text Consistency checks are only applicable to Passports. For Aadhaar / Driving Licences / Visas, the risk score is based on **ELA Forensics** and **Face Verification** only.")

    # -----------------------------------------------------------------------
    # F10: Evidence Report Export
    # -----------------------------------------------------------------------
    st.divider()
    _report_col1, _report_col2 = st.columns([2, 1])
    with _report_col1:
        st.markdown("**ðŸ“„ Evidence Report** â€” Export a structured inspection record for this session.")
    with _report_col2:
        try:
            from report_builder import build_evidence_report
            _orig_name = getattr(doc_upload, "name", "uploaded_document")
            _report_md = build_evidence_report(
                original_filename=_orig_name,
                sha256_hash=doc_sha256,
                mrz_result=mrz_result,
                ela_result=ela_result,
                bio_result=bio_result,
                text_consistency=text_consistency,
                quality_report=quality_report,
                risk=risk,
                mrz_method=mrz_method,
                doc_proc_note=doc_proc_note,
                doc_type=doc_type,
                id_result=id_result,
                visa_result=visa_result,
            )
            st.download_button(
                label="â¬‡ï¸ Download Report (.md)",
                data=_report_md.encode("utf-8"),
                file_name=f"pixelproof_report_{doc_sha256[:8]}.md",
                mime="text/markdown",
                width='stretch',
            )
        except Exception as _rep_exc:
            st.caption(f"Report generation unavailable: {_rep_exc}")

    # -----------------------------------------------------------------------
    # Footer disclaimer
    # -----------------------------------------------------------------------
    st.markdown(
        """
        <p style="color:#64748b;font-size:0.78rem;text-align:center;">
        PixelProof â€” Hackathon Educational Prototype &nbsp;|&nbsp;
        Not validated for real border or immigration use &nbsp;|&nbsp;
        ELA and MRZ checks are heuristic indicators, not proof of forgery or authenticity.
        </p>
        """,
        unsafe_allow_html=True,
    )

finally:
    # Always clean up both original and processing derivative temp files
    cleanup_temp(doc_orig_temp if "doc_orig_temp" in dir() else None)
    cleanup_temp(doc_proc_temp if "doc_proc_temp" in dir() else None)
    cleanup_temp(live_temp if "live_temp" in dir() else None)
