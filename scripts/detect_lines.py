#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text-line detection + ink percentage per line (grayscale-based).

Ink % per line is computed as:
  prct = (# pixels with value < 250) / (total pixels in the line box) * 100

Outputs:
  - lines_ink.png : red boxes with "L{i} {ink:.1f}%"
  - lines_ink.json: {"lines":[{"box":[x1,y1,x2,y2],"ink_pct":...},...], "params": ...}
"""
import os
import json
import argparse
import numpy as np
import cv2

def ensure_dir(d):
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def remove_rule_lines(bw):
    H, W = bw.shape
    kx = max(40, W // 30)
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1)),
                             iterations=1)
    ky = max(40, H // 30)
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky)),
                            iterations=1)
    cleaned = cv2.subtract(bw, cv2.bitwise_or(horiz, vert))
    return cleaned, horiz, vert

def detect_lines_with_mask(img_bgr, scale=3.0, row_frac=0.03, merge_gap=4, pad=6, th_max=130, th_min=45):
    # Upscale + grayscale + CLAHE
    up = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)

    # Margin crop
    margin = max(20, int(0.01 * gray.shape[1]))
    roi = gray[margin:-margin, margin:-margin]

    # Binary (text=255)
    bw = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 51, 12)

    # Remove borders/rules
    bw_clean, horiz, vert = remove_rule_lines(bw)

    H, W = bw_clean.shape
    row_sum = (bw_clean > 0).sum(axis=1).astype(np.int32)
    k = 9
    kernel = np.ones(k, dtype=np.float32) / k
    row_sm = np.convolve(row_sum, kernel, mode='same')

    row_thresh = max(12, int(row_frac * W))
    text_rows = row_sm >= row_thresh

    # Raw bands
    raw_bands = []
    in_band=False; start=0
    for y in range(H):
        if text_rows[y] and not in_band:
            in_band=True; start=y
        elif (not text_rows[y]) and in_band:
            in_band=False; raw_bands.append((start,y))
    if in_band: raw_bands.append((start,H-1))

    # Merge close gaps
    merged = []
    for y1,y2 in raw_bands:
        if not merged: merged.append([y1,y2]); continue
        py1,py2 = merged[-1]
        if y1 - py2 <= merge_gap:
            merged[-1][1] = y2
        else:
            merged.append([y1,y2])
    bands = [(a,b) for a,b in merged]

    # Split to constraints
    def split_to_constraints(y1,y2, row_sm, th_max, th_min):
        segs = []
        cur = y1
        while cur < y2:
            span = y2 - cur
            if span <= th_max:
                if span < th_min:
                    if segs:
                        py1, py2 = segs[-1]
                        segs[-1] = (py1, y2)
                    else:
                        segs.append((cur, y2))
                else:
                    segs.append((cur, y2))
                break
            window_end = min(y2, cur + th_max)
            win = row_sm[cur:window_end]
            split_at = int(np.argmin(win)) if len(win)>0 else (window_end-cur)//2
            split_row = cur + split_at
            if split_row <= cur + th_min:
                split_row = min(cur + th_max, y2)
            segs.append((cur, split_row))
            cur = split_row + 1
        # merge too-thin neighbors
        fixed = []
        for s in segs:
            if not fixed: fixed.append(s); continue
            a,b = fixed[-1]; c,d = s
            if (b-a) < th_min or (d-c) < th_min:
                fixed[-1] = (a,d)
            else:
                fixed.append(s)
        return fixed

    constrained = []
    for (y1,y2) in bands:
        constrained.extend(split_to_constraints(y1,y2, row_sm, th_max, th_min))

    # Candidate boxes from cleaned mask
    pad = int(pad)
    candidates = []
    for (y1,y2) in constrained:
        y1p, y2p = max(0,y1-pad), min(H,y2+pad)
        band = bw_clean[y1p:y2p, :]
        if band.size == 0: continue
        col = (band>0).sum(axis=0)
        xs = np.where(col>0)[0]
        if xs.size == 0: continue
        x1, x2 = int(xs[0]), int(xs[-1])
        candidates.append((y1p,y2p,x1,x2))

    # Border filtering
    filtered = []
    for (y1,y2,x1,x2) in candidates:
        band_full = bw[y1:y2, :]
        width_span = (x2 - x1 + 1) / float(W)
        row_density = (band_full>0).sum(axis=1) / float(W)
        med_density = float(np.median(row_density)) if row_density.size else 0.0
        mean_density = float(np.mean(row_density)) if row_density.size else 0.0

        left_margin  = (x1 <= int(0.01 * W))
        right_margin = (x2 >= int(0.99 * W))

        is_border_like = False
        if width_span >= 0.98 and (left_margin or right_margin):
            is_border_like = True
        if width_span >= 0.97 and med_density >= 0.20:
            is_border_like = True
        if mean_density >= 0.40:
            is_border_like = True

        if not is_border_like and (y2 - y1) >= th_min:
            filtered.append((y1,y2,x1,x2))

    # Map to upscaled image coordinates and return
    boxes = []
    for (y1,y2,x1,x2) in filtered:
        X1, Y1 = x1 + margin, y1 + margin
        X2, Y2 = x2 + margin, y2 + margin
        boxes.append((int(X1), int(Y1), int(X2), int(Y2)))

    return up, boxes, gray

def compute_ink_percentages_gray(gray_up, boxes):
    """
    prct = (# pixels with value < 250) / (total pixels in the box) * 100
    """
    ink_metrics = []
    H, W = gray_up.shape[:2]
    for (x1, y1, x2, y2) in boxes:
        x1c, x2c = max(0, x1), min(W, x2)
        y1c, y2c = max(0, y1), min(H, y2)
        roi = gray_up[y1c:y2c, x1c:x2c]
        total = roi.size if roi.size > 0 else 1
        ink = int((roi < 250).sum())
        ink_pct = 100.0 * ink / float(total)
        ink_metrics.append(ink_pct)
    return ink_metrics

def visualize_and_save(img_up, boxes, ink_pcts, out_png):
    vis = img_up.copy()
    for i, ((x1,y1,x2,y2), ink) in enumerate(zip(boxes, ink_pcts), 1):
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
        label = f"L{i} {ink:.1f}%"
        cv2.putText(vis, label, (x2 + 8, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(out_png, vis)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--scale", type=float, default=3.0, help="Upscale factor")
    ap.add_argument("--row-frac", type=float, default=0.03, help="Fraction of columns that must have ink for a row to be 'text'")
    ap.add_argument("--merge-gap", type=int, default=4, help="Max blank rows to merge bands")
    ap.add_argument("--pad", type=int, default=6, help="Vertical padding added to each band")
    ap.add_argument("--th-max", type=int, default=130, help="Maximum line height on upscaled image")
    ap.add_argument("--th-min", type=int, default=45, help="Minimum line height on upscaled image")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    img_up, boxes, gray_up = detect_lines_with_mask(
        img,
        scale=args.scale,
        row_frac=args.row_frac,
        merge_gap=args.merge_gap,
        pad=args.pad,
        th_max=args.th_max,
        th_min=args.th_min,
    )
    ink_pcts = compute_ink_percentages_gray(gray_up, boxes)

    # ---- Filter lines with >75% ink ----
    filtered_pairs = [(b, p) for (b, p) in zip(boxes, ink_pcts) if p <= 75.0]
    boxes = [b for (b, _) in filtered_pairs]
    ink_pcts = [p for (_, p) in filtered_pairs]
    print(f"Kept {len(boxes)} lines under 75% ink out of {len(filtered_pairs)} kept candidates.")
    # ------------------------------------
    out_png = os.path.join(args.out, "lines_ink.png")
    out_json = os.path.join(args.out, "lines_ink.json")
    visualize_and_save(img_up, boxes, ink_pcts, out_png)

    items = [{"box":[int(x1),int(y1),int(x2),int(y2)], "ink_pct": float(f"{p:.4f}")}
             for (x1,y1,x2,y2), p in zip(boxes, ink_pcts)]
    meta = {
        "image": args.image,
        "params": {
            "scale": args.scale,
            "row_frac": args.row_frac,
            "merge_gap": args.merge_gap,
            "pad": args.pad,
            "th_max": args.th_max,
            "th_min": args.th_min,
        },
        "lines": items
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_json}")
    print(f"Lines detected: {len(boxes)}")

if __name__ == "__main__":
    main()
