# 🕌 Arabic OCR Project

This is an **ongoing project** aimed at building a robust, modern **Arabic OCR (Optical Character Recognition)** pipeline.  
It focuses on accurately detecting, segmenting, and transcribing Arabic text from historical and printed documents — especially those with diacritics and complex layouts.

---

## 📘 Project Overview

The system is designed in modular stages:

1. **Line Detection**
   - Detects text lines from scanned pages.
   - Uses morphological and projection-based methods.
   - Supports deskewing, background removal, and contrast enhancement.

2. **Ink Percentage Filtering**
   - Computes the percentage of *ink* (non-white pixels) in each line.
   - Filters out overly dark or dense regions (e.g., borders, illustrations).
   - Implemented in `detect_lines_ink_filtered.py`, which discards lines above 75% ink coverage.

3. **Word Segmentation**
   - Splits detected lines into words based on whitespace and vertical projections.
   - Prepares cropped word images for OCR.

4. **Transcription**
   - Uses LLM-assisted OCR (via Tesseract or EasyOCR) to transcribe Arabic text.
   - Handles diacritics and script variations.

5. **LLM-Based Correction**
   - Post-processes OCR output through a public LLM (Groq or OpenAI) to fix spacing, diacritics, and grammar.
   - Ensures a consistent, human-readable Arabic text.

---

## ⚙️ Setup Instructions

### 🧰 1. Environment Setup (macOS / Linux)

Install Python 3 (≥ 3.9) and create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

### 📦 2. Install Dependencies
```bash
pip install opencv-python numpy pillow pytesseract easyocr python-bidi arabic-reshaper
```

**Optional (for LLM correction):**
```bash
pip install groq openai
```

### 🖋️ 3. Install Tesseract (for OCR)
#### macOS:
```bash
brew install tesseract tesseract-lang
```
#### Ubuntu/Debian:
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-ara
```

---

## ▶️ Usage Examples

### **A. Detect Lines and Filter by Ink Percentage**
```bash
python detect_lines_ink_filtered.py   --image path/to/page.png   --out results/   --scale 3.0   --th-max 130   --th-min 45
```
- Outputs:
  - `results/lines_ink.png` — red boxes around detected lines (≤75% ink)
  - `results/lines_ink.json` — line coordinates + ink percentages

---

### **B. Full OCR Pipeline (Work in Progress)**
```bash
python arabic_ocr_pipeline.py   --image path/to/page.png   --out results/   --engine tesseract   --llm groq
```

This runs:
1. Line & word detection  
2. OCR transcription  
3. Optional LLM correction step for final refinement  

---

## 🧩 Current Files

| File | Description |
|------|--------------|
| `detect_lines_ink_filtered.py` | Detects text lines, computes grayscale-based ink percentage, and filters out lines above 75%. |
| `arabic_ocr_pipeline.py` | Full OCR pipeline: detection → segmentation → transcription → LLM correction. *(Work in progress)* |
| `detect_lines.py` / `detect_lines_ink.py` | Earlier experimental versions for line detection and ink analysis. |

---

## 🧠 Status
This is an **active development project**.  
Next milestones include:
- Integrating word-level segmentation with line detection.
- Improving OCR accuracy on degraded Arabic manuscripts.
- Adding debugging and visualization tools.
- Automating evaluation with benchmark datasets.

---

## 📎 Authors & Contributions
Developed by **Taleb Elm**.  
Contributions are welcome as the project evolves. Thanks!

---
