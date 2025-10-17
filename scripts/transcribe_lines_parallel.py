#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transcribe Arabic lines to letters (parallel) using LLM5.py OCR method.

UPDATE (simplified):
- JSON boxes are already in the **downscaled/scaled detection space**: [x1,y1,x2,y2].
- We DO NOT remap boxes to original; everything happens in the scaled image space.
- Pipeline per line: crop (scaled) -> optional OCR UPSCALE (--scale_ocr) -> TRIM -> ENHANCE -> OCR

Inputs:
  {
    "image": "test.png",
    "params": { "scale": 3.0, "pad": 6, ... },
    "lines": [ { "box": [x1,y1,x2,y2], "ink_pct": ... }, ... ]
  }

CLI:
  python transcribe_lines_parallel.py --input lines.json --output out.json \
    [--image test.png] [--out_dir crops] [--psm 7] [--max_workers 6] \
    [--pad PAD_IN_SCALED] [--scale_det SCALE_USED_FOR_DETECTION] [--scale_ocr 2.0]

Notes:
- pad is interpreted in the SAME (scaled) space as the boxes.
- --scale_det (or JSON params.scale) is used only to reconstruct the detection/working image from the original.
- --scale_ocr is a separate upsample applied to the cropped line before trimming/OCR (default 2.0).
"""
import json
import argparse
import os
import sys
import cv2
import numpy as np
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import OCR helpers from LLM5.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
try:
    from LLM5 import tesseract_cfg, run_ocr
except Exception as e:
    print("[ERROR] Could not import from LLM5.py. Make sure it is alongside this script.", file=sys.stderr)
    raise

# Optional Unicode grapheme clustering
try:
    import regex as re_gr
except Exception:
    re_gr = None


def parse_args():
    p = argparse.ArgumentParser(description="Transcribe Arabic lines in scaled detection space; save crops")
    p.add_argument("--input", "-i", required=True, help="Path to input JSON (your format)")
    p.add_argument("--image", "-m", required=False, help="Override image path from JSON")
    p.add_argument("--output", "-o", required=True, help="Path to output JSON")
    p.add_argument("--out_dir", "-d", default="line_crops", help="Directory to save crops")
    p.add_argument("--psm", type=int, default=7, help="Tesseract PSM (default 7 = single text line)")
    p.add_argument("--max_workers", type=int, default=4, help="Parallel workers")
    p.add_argument("--pad", type=int, default=None, help="Padding in SCALED-SPACE pixels (overrides JSON params.pad)")
    p.add_argument("--scale_det", type=float, default=None, help="Scale used to create detection/scaled image (overrides JSON params.scale)")
    p.add_argument("--scale_ocr", type=float, default=2.0, help="Extra upsample applied AFTER crop for OCR (default 2.0)")
    p.add_argument("--min_height", type=int, default=12, help="Skip lines smaller than this height (pre-ocr-upsample)")
    return p.parse_args()


def to_xywh(x1, y1, x2, y2):
    w = max(1, int(round(x2 - x1)))
    h = max(1, int(round(y2 - y1)))
    return int(round(x1)), int(round(y1)), w, h


def clamp_rect(x, y, w, h, W, H):
    x0 = max(0, x); y0 = max(0, y)
    x1 = min(W, x + w); y1 = min(H, y + h)
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def crop_rect(img, x, y, w, h):
    H, W = img.shape[:2]
    x0, y0, w0, h0 = clamp_rect(x, y, w, h, W, H)
    return img[y0:y0+h0, x0:x0+w0], (x0, y0, w0, h0)


def scale_image(img_bgr, scale):
    if img_bgr is None or img_bgr.size == 0 or scale in (None, 1.0):
        return img_bgr
    h, w = img_bgr.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def autotrim(img_bgr):
    """Trim near-white borders to tight content box. Returns (trimmed_bgr, trim_bbox_relative)."""
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr, [0, 0, 0, 0]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    coords = cv2.findNonZero(th)
    if coords is None:
        h, w = gray.shape[:2]
        return img_bgr, [0, 0, w, h]
    x, y, w, h = cv2.boundingRect(coords)
    pad = 2
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(img_bgr.shape[1], x + w + pad)
    y1 = min(img_bgr.shape[0], y + h + pad)
    trimmed = img_bgr[y0:y1, x0:x1]
    return trimmed, [x0, y0, x1 - x0, y1 - y0]


def enhance_for_ocr(img_bgr):
    """Gentle denoise + contrast + binarize."""
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    mn, mx = np.percentile(gray, [1, 99])
    if mx > mn:
        gray = np.clip((gray - mn) * (255.0 / (mx - mn)), 0, 255).astype(np.uint8)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)


def split_letters_graphemes(text):
    s = text.strip()
    if re_gr:
        clusters = re_gr.findall(r"\X", s)
    else:
        clusters = list(s)
    out = []
    for g in clusters:
        if g.isspace():
            continue
        keep = False
        for ch in g:
            import unicodedata as ud
            cat = ud.category(ch)
            if cat.startswith("L") or cat.startswith("M") or ch == "ـ":
                keep = True
        if keep:
            out.append(g)
    return out


def ocr_line(img_bgr, psm):
    cfg = tesseract_cfg(psm=psm)
    txt, conf = run_ocr(img_bgr, cfg)
    txt_oneline = " ".join(txt.splitlines()).strip()
    return txt_oneline, float(conf)


def process_one(entry, img_scaled, args):
    ln = int(entry["line_number"])

    # pad and crop entirely in SCALED space
    pad_s = int(round(args.pad))
    x1s, y1s, x2s, y2s = entry["box_scaled"]
    x1s_p = int(round(x1s - pad_s))
    y1s_p = int(round(y1s - pad_s))
    x2s_p = int(round(x2s + pad_s))
    y2s_p = int(round(y2s + pad_s))

    xs, ys, ws, hs = to_xywh(x1s_p, y1s_p, x2s_p, y2s_p)
    crop_scaled, (xs0, ys0, ws0, hs0) = crop_rect(img_scaled, xs, ys, ws, hs)

    save_base = os.path.join(args.out_dir, f"line_{ln:03d}")
    os.makedirs(args.out_dir, exist_ok=True)

    out = {
        "line_number": ln,
        "box_scaled": [int(round(x1s)), int(round(y1s)), int(round(x2s)), int(round(y2s))],
        "box_scaled_padded": [x1s_p, y1s_p, x2s_p, y2s_p],
        "rect_scaled_xywh": [xs0, ys0, ws0, hs0],
        "text": "",
        "confidence": 0.0,
        "letters": [],
        "files": {}
    }

    if crop_scaled.size == 0 or crop_scaled.shape[0] < args.min_height:
        return out

    # Save raw scaled crop
    raw_path = f"{save_base}_scaled_raw.png"
    cv2.imwrite(raw_path, crop_scaled)
    out["files"]["scaled_raw"] = os.path.abspath(raw_path)

    # Optional OCR upsample (AFTER crop) in scaled space
    up = crop_scaled
    if args.scale_ocr and args.scale_ocr != 1.0:
        up = scale_image(crop_scaled, float(args.scale_ocr))
        up_path = f"{save_base}_scaled_up.png"
        cv2.imwrite(up_path, up)
        out["files"]["scaled_up"] = os.path.abspath(up_path)

    # Trim and enhance
    trim, trim_rel = autotrim(up)
    trim_path = f"{save_base}_trim.png"
    cv2.imwrite(trim_path, trim)
    out["files"]["trim"] = os.path.abspath(trim_path)
    out["trim_bbox_rel"] = trim_rel

    enh = enhance_for_ocr(trim)
    enh_path = f"{save_base}_enh.png"
    cv2.imwrite(enh_path, enh)
    out["files"]["enhanced"] = os.path.abspath(enh_path)

    # OCR
    text, conf = ocr_line(enh, psm=args.psm)
    out["text"] = text
    out["confidence"] = conf
    out["letters"] = split_letters_graphemes(text)

    return out


def main():
    args = parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        obj = json.load(f)

    image_path = args.image or obj.get("image")
    if not image_path or not os.path.exists(image_path):
        print(f"[ERROR] Image not found. Pass --image or set 'image' in JSON. Got: {image_path}", file=sys.stderr)
        sys.exit(2)

    params = obj.get("params", {}) if isinstance(obj, dict) else {}
    det_scale = float(args.scale_det) if args.scale_det is not None else float(params.get("scale", 1.0))
    pad_scaled = args.pad if args.pad is not None else int(params.get("pad", 6))
    args.pad = pad_scaled

    # Build the SCALED detection image to match your boxes
    img_orig = cv2.imread(image_path)
    if img_orig is None:
        print(f"[ERROR] Could not read image: {image_path}", file=sys.stderr)
        sys.exit(2)
    img_scaled = img_orig if det_scale in (None, 1.0) else scale_image(img_orig, det_scale)

    # Extract boxes (already in SCALED space)
    lines_raw = obj.get("lines", [])
    lines = []
    for i, it in enumerate(lines_raw):
        box = it.get("box") or it.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        x1, y1, x2, y2 = map(float, box)
        lines.append({"line_number": i + 1, "box_scaled": [x1, y1, x2, y2]})

    if not lines:
        print("[ERROR] No valid boxes found.", file=sys.stderr)
        sys.exit(3)

    os.makedirs(args.out_dir, exist_ok=True)

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        fut2ln = {ex.submit(process_one, d, img_scaled, args): d["line_number"] for d in lines}
        for fut in as_completed(fut2ln):
            ln = fut2ln[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"line_number": ln, "error": str(e)}
            results.append(res)

    results = sorted(results, key=lambda r: r["line_number"])

    out = {"lines": results, "meta": {
        "image": os.path.abspath(image_path),
        "input_json": os.path.abspath(args.input),
        "psm": args.psm,
        "max_workers": args.max_workers,
        "pad_scaled_px": args.pad,
        "det_scale": det_scale,
        "scale_ocr": args.scale_ocr,
        "out_dir": os.path.abspath(args.out_dir)
    }}

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote {args.output} with {len(results)} lines.")
    print(f"[OK] Saved crops under: {os.path.abspath(args.out_dir)}")
    print(f"[OK] All operations in SCALED space. det_scale={det_scale}, pad={args.pad}, scale_ocr={args.scale_ocr}")

if __name__ == "__main__":
    main()
