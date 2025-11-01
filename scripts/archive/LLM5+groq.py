# ocr_arabic_spacing_fix_preserve_order.py
# ------------------------------------------------------------
# Requirements:
#   sudo apt-get install -y tesseract-ocr tesseract-ocr-ara
#   git clone https://github.com/tesseract-ocr/tessdata_best   (optional but recommended)
#   pip install pytesseract opencv-python numpy pillow
#   pip install groq   # <-- NEW for LLM correction
# ------------------------------------------------------------

import os
import cv2
import numpy as np
import pytesseract

# ===================== CONFIG ===================== #
IMAGE_PATH = "page2.png"            # input image
SAVE_PREFIX = "page2"               # base name for outputs
LANG = "ara"                        # Arabic language
PSM_TRY = [6, 4, 3]                 # PSM modes to try (single block, columns, auto)
UPSCALE_MAX_DIM = 1500              # upscale small scans up to this max dimension (px)
EXTRA_GAP_FACTOR = 0.60             # 0.5–0.8; lower = more extra spaces, higher = fewer
TESSDATA_DIR = "./tessdata_best" if os.path.isdir("./tessdata_best") else None
USER_WORDS = "user-words.txt" if os.path.isfile("user-words.txt") else None

# === LLM (Groq) Settings === #
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile").strip()
# Safe chunk sizes; you can raise if your model/context allows
LLM_MAX_CHARS = 9000          # per chunk input size (roughly ~3–4k tokens)
LLM_OVERLAP_CHARS = 250       # small overlap to reduce boundary artifacts
RUN_LLM_CORRECTION = bool(GROQ_API_KEY)   # auto-on if API key present
# ================================================== #


def deskew(bgr):
    """Estimate skew angle and rotate to deskew."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    thr = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thr > 0))
    if coords.size == 0:
        return bgr
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    h, w = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def preprocess_variants(bgr):
    """
    Produce several preprocessed variants.
    Avoid closing (it fuses gaps). Use light opening if needed.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)

    # Gentle denoise
    den = cv2.bilateralFilter(g, d=7, sigmaColor=50, sigmaSpace=50)

    # Thresholds
    otsu = cv2.threshold(den, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adapt = cv2.adaptiveThreshold(
        den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )

    # Very light opening to separate touching blobs
    kernel = np.ones((1, 1), np.uint8)
    open_ = cv2.morphologyEx(adapt, cv2.MORPH_OPEN, kernel, iterations=1)

    return {"otsu": otsu, "adapt": adapt, "open": open_}


def tesseract_cfg(psm):
    """Build a config string favoring space preservation."""
    parts = []
    if TESSDATA_DIR:
        parts += [f'--tessdata-dir "{TESSDATA_DIR}"']
    parts += [
        f"-l {LANG}",
        "--oem 1",
        f"--psm {psm}",
        "-c preserve_interword_spaces=1",
        "-c textord_space_size_is_variable=1",
        "-c tessedit_do_invert=0",
        "-c textord_heavy_nr=1",
    ]
    if USER_WORDS:
        parts += [f"-c user_words_suffix={USER_WORDS}"]
    return " ".join(parts)


def run_ocr(img, cfg):
    """Run Tesseract and compute mean confidence from word data."""
    txt = pytesseract.image_to_string(img, config=cfg)
    data = pytesseract.image_to_data(img, config=cfg, output_type=pytesseract.Output.DICT)
    confs = [int(c) for c in data.get("conf", []) if str(c).isdigit() and int(c) >= 0]
    mean_conf = (sum(confs) / len(confs)) if confs else 0.0
    return txt, mean_conf


def pick_best_variant(bgr):
    """Try preprocess variants + PSMs; return best by confidence."""
    variants = preprocess_variants(bgr)
    best = {"conf": -1, "name": None, "psm": None, "cfg": "", "img": None, "txt": ""}

    for name, vimg in variants.items():
        for psm in PSM_TRY:
            cfg = tesseract_cfg(psm)
            txt, conf = run_ocr(vimg, cfg)
            if conf > best["conf"]:
                best.update({"conf": conf, "name": name, "psm": psm, "cfg": cfg, "img": vimg, "txt": txt})
    return best


def reconstruct_preserve_order_with_gaps(img, cfg, extra_gap_factor=EXTRA_GAP_FACTOR):
    """
    Rebuild lines from word boxes while PRESERVING the token order
    produced by Tesseract (no sorting). Insert an EXTRA space when
    the visual gap between consecutive words is large.
    """
    d = pytesseract.image_to_data(img, config=cfg, output_type=pytesseract.Output.DICT)
    n = len(d["text"])

    # group tokens by line in the ORIGINAL order (index i)
    lines = {}
    for i in range(n):
        text_i = d["text"][i]
        conf_i = d["conf"][i]
        if not text_i or not str(conf_i).isdigit() or int(conf_i) < 0:
            continue
        key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        lines.setdefault(key, []).append({
            "t": text_i.strip(),
            "x": int(d["left"][i]),
            "w": int(d["width"][i]),
            "i": i  # keep original order
        })

    out_lines = []
    for key in sorted(lines.keys()):
        words = lines[key]  # already in original order
        # estimate threshold based on median width in this line
        widths = [w["w"] for w in words if w["w"] > 0]
        med_w = np.median(widths) if widths else 15
        gap_thresh = extra_gap_factor * med_w

        s = ""
        for j, w in enumerate(words):
            token = w["t"]
            if not token:
                continue
            if j == 0:
                s += token
            else:
                prev = words[j - 1]
                # compute visual gap using x coordinates
                gap = w["x"] - (prev["x"] + prev["w"])
                # normal space between tokens
                s += " "
                # if gap is large, add an extra space
                if gap > gap_thresh:
                    s += " "
                s += token
        out_lines.append(s)

    return "\n".join(out_lines)

# ===================== NEW: Groq LLM Correction ===================== #
def _chunk_text(text, max_chars=LLM_MAX_CHARS, overlap=LLM_OVERLAP_CHARS):
    """
    Split long text into overlapping chunks to fit model context.
    Splits on paragraph boundaries when possible.
    """
    if len(text) <= max_chars:
        return [text]

    paras = text.split("\n")
    chunks, buf = [], ""
    for p in paras:
        candidate = (buf + "\n" + p) if buf else p
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            # start new buffer; if the single paragraph is huge, hard-split it
            if len(p) > max_chars:
                start = 0
                while start < len(p):
                    end = min(start + max_chars, len(p))
                    part = p[start:end]
                    chunks.append(part)
                    start = end - overlap if end - overlap > start else end
                buf = ""
            else:
                buf = p
    if buf:
        chunks.append(buf)

    # add overlaps between adjacent chunks (simple tail/head stitching)
    stitched = []
    for i, c in enumerate(chunks):
        if i == 0 or overlap <= 0:
            stitched.append(c)
        else:
            head = c[:overlap]
            prev = stitched[-1]
            stitched[-1] = prev  # keep previous intact; overlap context is carried implicitly
            stitched.append(c)   # (we rely on the model to keep coherence across mild overlaps)
    return stitched


def correct_with_groq(text, model=GROQ_MODEL):
    """
    Send OCR text to Groq LLM for Arabic OCR correction.
    Returns corrected text (same overall structure, cleaner tokens).
    """
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)

    system_prompt = (
        "You are an expert Arabic text corrector specializing in fixing OCR output. "
        "You must carefully review the text and follow these rules:\n\n"
        "1. Correct all OCR-specific mistakes: broken words, character substitutions "
        "(e.g., ب/ت/ث, ح/خ, ف/ق), Latin characters instead of Arabic, misplaced diacritics, "
        "and punctuation errors.\n"
        "2. Check every single sentence individually and ensure that it makes sense in its context. "
        "If a word looks wrong, replace it with the correct word that fits the surrounding meaning.\n"
        "3. Repeat the correction process multiple times until the full text is fluent, coherent, and meaningful.\n"
        "4. Preserve the exact line and sentence order from the input. Do not merge or reorder lines.\n"
        "5. Do not remove words, and do not add entirely new sentences. Only adjust or replace words when "
        "necessary for meaning and correctness.\n"
        "6. Keep the original meaning intact and improve readability.\n"
        "7. If diacritics (tashkeel) are present, preserve them whenever possible.\n\n"
        "Return only the fully corrected Arabic text, line by line, with no commentary."
    )

    def _one_call(chunk):
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        "نص OCR المراد تصحيحه:\n\n"
                        + chunk
                        + "\n\n"
                        "التعليمات: صحّح الأخطاء الإملائية الناتجة عن OCR فقط "
                          "مع الحفاظ على المعنى والترتيب والسطر. أعد النص المصحح فقط."
                    ),
                },
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()

    chunks = _chunk_text(text, max_chars=LLM_MAX_CHARS, overlap=LLM_OVERLAP_CHARS)
    corrected_chunks = []
    for ch in chunks:
        corrected_chunks.append(_one_call(ch))

    # Simple join: keep original paragraph breaks between chunks
    corrected = "\n".join(corrected_chunks)

    # Light punctuation tidying (Arabic)
    corrected = (corrected
                 .replace(" »", "»").replace("« ", "«")
                 .replace(" ،", "،").replace(" ؛", "؛")
                 .replace(" .", ".").replace(" !", "!")
                 .replace(" ؟", "؟"))
    return corrected
# =================================================================== #


def main():
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(IMAGE_PATH)

    bgr = cv2.imread(IMAGE_PATH)

    # Upscale small scans so gaps are several pixels wide
    max_dim = max(bgr.shape[:2])
    if max_dim < UPSCALE_MAX_DIM:
        scale = UPSCALE_MAX_DIM / max_dim
        bgr = cv2.resize(
            bgr,
            (int(bgr.shape[1] * scale), int(bgr.shape[0] * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    bgr = deskew(bgr)
    best = pick_best_variant(bgr)

    best_img_path = f"{SAVE_PREFIX}_best_preprocessed.png"
    cv2.imwrite(best_img_path, best["img"])

    print(f"\n[INFO] Best preprocess: {best['name']} | PSM={best['psm']} | mean_conf={best['conf']:.1f}")
    print(f"[INFO] Tesseract config: {best['cfg']}")

    print("\n===== RAW OCR =====\n")
    print(best["txt"].strip())

    rebuilt = reconstruct_preserve_order_with_gaps(best["img"], best["cfg"], EXTRA_GAP_FACTOR)

    # Tidy common punctuation spacing
    rebuilt = (rebuilt
               .replace(" »", "»").replace("« ", "«")
               .replace(" ،", "،").replace(" ؛", "؛")
               .replace(" .", ".").replace(" !", "!")
               .replace(" ؟", "؟"))

    print("\n===== PRESERVE-ORDER GAP-AWARE REBUILT =====\n")
    print(rebuilt.strip())

    # ===================== NEW: LLM Post-processing ===================== #
    corrected = None
    if RUN_LLM_CORRECTION:
        try:
            print("\n[INFO] Running Groq LLM correction...")
            corrected = correct_with_groq(rebuilt)
            print("\n===== LLM-CORRECTED TEXT =====\n")
            print(corrected.strip())
        except Exception as e:
            print(f"\n[WARN] Groq correction failed: {e}")
    else:
        print("\n[INFO] GROQ_API_KEY not set; skipping LLM correction step.")
    # =================================================================== #

    with open(f"{SAVE_PREFIX}_raw.txt", "w", encoding="utf-8") as f:
        f.write(best["txt"].strip() + "\n")
    with open(f"{SAVE_PREFIX}_rebuilt.txt", "w", encoding="utf-8") as f:
        f.write(rebuilt.strip() + "\n")
    if corrected is not None:
        with open(f"{SAVE_PREFIX}_corrected.txt", "w", encoding="utf-8") as f:
            f.write(corrected.strip() + "\n")

    print("\n[INFO] Saved:")
    print(f"  - {best_img_path}")
    print(f"  - {SAVE_PREFIX}_raw.txt")
    print(f"  - {SAVE_PREFIX}_rebuilt.txt")
    if corrected is not None:
        print(f"  - {SAVE_PREFIX}_corrected.txt")


if __name__ == "__main__":
    main()
