# PixelProof

> **EDUCATIONAL PROTOTYPE — Not validated for, or intended for, real border / immigration decisions.**

A hackathon proof-of-concept demonstrating how multiple computer-vision and
NLP techniques can be combined into a document-verification pipeline.
Built in Python / Streamlit.

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Calibration Guide](#calibration-guide)
5. [Running the Tests](#running-the-tests)
6. [Failure Mode Behaviour](#failure-mode-behaviour)
7. [Limitations & Disclaimer](#limitations--disclaimer)

---

## What It Does

| Step | Module | Technique |
|------|--------|-----------|
| MRZ Validation | `mrz_engine.py` | ICAO 9303 TD3 weighted-mod-10 checksum |
| Forgery Indicator | `ela_forensics.py` | Error Level Analysis (JPEG re-compression) |
| Face Verification | `biometrics.py` | DeepFace + ArcFace + cosine distance |
| Text Consistency | `risk_engine.py` | PaddleOCR + RapidFuzz fuzzy match |
| Risk Score | `risk_engine.py` | Weighted 0-100 composite score |
| Dashboard | `app.py` | Streamlit (colour-coded risk banner + three cards) |

---

## Architecture

```
sih26/
  mrz_engine.py      ICAO 9303 TD3 parser & validator (UI-agnostic)
  ela_forensics.py   JPEG ELA forensics (UI-agnostic)
  biometrics.py      ArcFace facial verification (UI-agnostic)
  risk_engine.py     Composite risk aggregator + VIZ OCR (UI-agnostic)
  app.py             Streamlit dashboard
  requirements.txt   Python dependencies
  README.md          This file
  tests/
    test_mrz_engine.py   pytest unit tests for MRZ engine
```

The four engine modules contain **no Streamlit imports**, making them
independently testable and reusable outside the UI.

---

## Quick Start

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install PaddlePaddle (CPU)

PaddlePaddle must be installed *before* PaddleOCR:

```powershell
pip install paddlepaddle
```

For GPU (CUDA 11.x):
```powershell
pip install paddlepaddle-gpu
```

### 3. Install all other dependencies

```powershell
pip install -r requirements.txt
```

### 4. Launch the app

```powershell
streamlit run app.py
```

The browser should open automatically at `http://localhost:8501`.

---

## Calibration Guide

### ELA Threshold — `ELA_ALERT_THRESHOLD` in `ela_forensics.py`

```python
# ela_forensics.py, near top of file
ELA_ALERT_THRESHOLD: float = 12.0   # <-- adjust this
```

**What it means:** The mean absolute pixel error (0-255 scale, amplified
by `scale=15`) above which ELA flags an "elevated compression-artifact score."

**How to tune:**
1. Gather 20+ genuine scanned passports (your own test set).
2. Run `perform_ela()` on each and note the `mean_error` values.
3. Gather 5-10 known tampered/composited images and do the same.
4. Set the threshold between the two distributions.
   A good starting range is typically **8.0 – 20.0** depending on scanner quality.

> Lower threshold = more sensitive (more false positives on legitimate docs).
> Higher threshold = less sensitive (misses subtle forgeries).

---

### Biometric Threshold — `ARCFACE_COSINE_THRESHOLD` in `biometrics.py`

```python
# biometrics.py, near top of file
ARCFACE_COSINE_THRESHOLD: float = 0.68   # <-- adjust this
```

**What it means:** ArcFace cosine distance at which two faces are declared
a non-match. Lower distance = more similar faces.
- DeepFace default for ArcFace + cosine is ~0.68.
- Do NOT borrow the 0.40 threshold from VGG-Face or FaceNet models.

**How to tune:**
1. Collect matched pairs (same person, passport photo + selfie) and
   non-matched pairs (different people).
2. Run `verify_faces()` on each pair and record distances.
3. Plot a histogram; find the EER (equal error rate) crossing point.
4. Set the threshold at or slightly above the EER point.

> Lower threshold = fewer false accepts (more false rejects).
> Higher threshold = fewer false rejects (more false accepts).

---

## Running the Tests

```powershell
# From the project root
pip install pytest
pytest tests/ -v
```

Expected output: all tests in `tests/test_mrz_engine.py` pass, covering:
- Valid TD3 MRZ (all checks pass).
- Broken DOB check digit (DOB check fails, others unaffected).
- Altered document number (doc number check fails, DOB unaffected).
- Malformed / unreadable inputs (MRZ_UNREADABLE, no exception).

---

## Failure Mode Behaviour

| Scenario | Behaviour |
|----------|-----------|
| PassportEye unavailable | Falls back to PaddleOCR, then regex line-filter; UI shows warning |
| PaddleOCR unavailable | Text consistency check skipped; weight redistributed |
| DeepFace / TF unavailable | Biometric card shows "module unavailable"; weight redistributed |
| No face detected | Explicit "NO_FACE" status; biometric score excluded |
| Multiple faces detected | Explicit "MULTIPLE_FACES" status; user prompted to re-crop |
| ELA file not found / corrupt | ELA card shows error message; score excluded |
| All checks incomplete | Score = 0, category = INCOMPLETE, UI shows prominent warning |

---

## Limitations & Disclaimer

- **This is a hackathon prototype.** No security audit has been performed.
- ELA is a heuristic; it produces false positives and false negatives.
- ArcFace accuracy depends on photo quality, lighting, and angle.
- The MRZ parser supports **TD3 only** (standard passport). TD1 (ID cards) requires a separate implementation.
- No data is stored between sessions; biometric data is processed in memory only.
- **Do not use this system for any real identity verification, border control, or immigration decision.**
