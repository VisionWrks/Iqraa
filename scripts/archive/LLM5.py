# ocr_arabic_spacing_fix_preserve_order.py
# ------------------------------------------------------------
# Requirements:
#   sudo apt-get install -y tesseract-ocr tesseract-ocr-ara
#   git clone https://github.com/tesseract-ocr/tessdata_best   (optional but recommended)
#   pip install pytesseract opencv-python numpy pillow
# ------------------------------------------------------------

import os
import cv2
import numpy as np
import pytesseract

# ===================== CONFIG ===================== #
IMAGE_PATH = "page3.png"            # input image
SAVE_PREFIX = "page3"               # base name for outputs
LANG = "ara"                        # Arabic language
PSM_TRY = [6, 4, 3]                 # PSM modes to try (single block, columns, auto)
UPSCALE_MAX_DIM = 1500              # upscale small scans up to this max dimension (px)
EXTRA_GAP_FACTOR = 0.60             # 0.5–0.8; lower = more extra spaces, higher = fewer
TESSDATA_DIR = "./tessdata_best" if os.path.isdir("./tessdata_best") else None
USER_WORDS = "user-words.txt" if os.path.isfile("user-words.txt") else None
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

    with open(f"{SAVE_PREFIX}_raw.txt", "w", encoding="utf-8") as f:
        f.write(best["txt"].strip() + "\n")
    with open(f"{SAVE_PREFIX}_rebuilt.txt", "w", encoding="utf-8") as f:
        f.write(rebuilt.strip() + "\n")

    print("\n[INFO] Saved:")
    print(f"  - {best_img_path}")
    print(f"  - {SAVE_PREFIX}_raw.txt")
    print(f"  - {SAVE_PREFIX}_rebuilt.txt")


if __name__ == "__main__":
    main()
