#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time
import re
import unicodedata
from typing import Optional

# -----------------------------
# Config
# -----------------------------
BASE_URL   = "https://shamela.ws/book/9472"
START, END = 1, 1703
OUT_PREFIX = "shamela_9472"
PARTS      = 4
DELAY      = 0.8  # be polite to the server
TIMEOUT    = 25

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "ar,ar-EG;q=0.9,en;q=0.8",
}

# -----------------------------
# Utilities
# -----------------------------
ARABIC_RE   = re.compile(r"[\u0600-\u06FF]")
NOISE_TAGS  = ["script", "style", "noscript", "footer", "nav", "header", "form", "aside"]
KASHIDA     = "\u0640"  # tatweel

def page_url(n: int) -> str:
    return f"{BASE_URL.rstrip('/')}/{n}"

def clean_text(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)

def _canon(s: str) -> str:
    """Normalize string for robust comparisons: NFKC, drop tatweel and spaces."""
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace(KASHIDA, "")
    s = s.replace(" ", "")
    return s

SEARCH_NOISE_PREFIXES = (
    "بحث في محتوى الكتب",
    "نتائج البحث",
)

def remove_search_noise(text: str) -> str:
    """Drop lone 'بحــث' lines and known panel headings."""
    cleaned = []
    for ln in (text or "").splitlines():
        c = _canon(ln)
        # drop lines that are just "بحث" (any tatweel/spaces)
        if c == "بحث":
            continue
        # drop known headings / panel crumbs
        if any(c.startswith(_canon(p)) for p in SEARCH_NOISE_PREFIXES):
            continue
        cleaned.append(ln)
    # collapse extra blank lines
    out = "\n".join(l for l in (x.strip() for x in cleaned) if l)
    return out

def looks_empty_or_noise(text: str) -> bool:
    """Consider very short remnants after noise removal as noise."""
    return len((text or "").strip()) < 10

# -----------------------------
# Extraction
# -----------------------------
def extract_shamela_markers(html: str) -> str:
    """
    Site-aware slice for shamela.ws:
      Take full text, start after 'التشكيل', stop before pager/search markers.
    """
    soup = BeautifulSoup(html, "html.parser")
    for t in NOISE_TAGS:
        for tag in soup.find_all(t):
            tag.decompose()

    full = soup.get_text(separator="\n", strip=True)

    # Start at the first 'التشكيل'
    start_idx = None
    for m in re.finditer(r"\bالتشكيل\b", full):
        start_idx = m.end()
        break
    if start_idx is None:
        return ""

    tail = full[start_idx:]
    end_markers = [
        r"<<\s*<\s*ج:",       # pagination 'ج:'
        r"\nص:\s",            # 'ص:' pager
        r"###\s*البحث في:",   # search panel
        r"####\s*البحث في:",
        r"####\s*نتائج البحث:",
    ]
    ends = [re.search(p, tail) for p in end_markers]
    ends = [m for m in ends if m]
    content = tail[:min(m.start() for m in ends)] if ends else tail
    content = content.lstrip()

    if len(ARABIC_RE.findall(content)) < 80:
        return ""
    return clean_text(content)

def longest_arabic_block(soup: BeautifulSoup) -> Optional[str]:
    # remove obvious noise + common sidebars
    for t in NOISE_TAGS:
        for tag in soup.find_all(t):
            tag.decompose()
    for el in soup.find_all(True):
        t = el.get_text(strip=True)
        if t.startswith(("فصول الكتاب", "حجم الكتاب", "مؤلف")):
            el.decompose()

    best = None
    best_score = (-1, -1)
    for el in soup.find_all(["div", "section", "article"], recursive=True):
        txt = el.get_text(separator="\n", strip=True)
        if not txt:
            continue
        ar = len(ARABIC_RE.findall(txt))
        if ar >= 100:
            score = (ar, len(txt))
            if score > best_score:
                best_score = score
                best = txt
    if best:
        return clean_text(best)
    return None

def extract_main_text(html: str) -> str:
    # 1) Markers
    body = extract_shamela_markers(html)
    if body:
        return remove_search_noise(body)

    # 2) Heuristic fallback
    soup = BeautifulSoup(html, "html.parser")
    txt = longest_arabic_block(soup)
    if txt:
        return remove_search_noise(txt)

    # 3) Final fallback: whole body
    body_tag = soup.find("body")
    if body_tag:
        txt = clean_text(body_tag.get_text(separator="\n", strip=True))
        return remove_search_noise(txt)
    return ""

# -----------------------------
# Crawl
# -----------------------------
def crawl():
    total = END - START + 1
    per_part = (total + PARTS - 1) // PARTS
    outs = [Path(f"{OUT_PREFIX}_part{idx+1}.txt").open("w", encoding="utf-8") for idx in range(PARTS)]

    try:
        for i, pg in enumerate(range(START, END + 1)):
            url = page_url(pg)
            out_idx = min(i // per_part, PARTS - 1)
            fh = outs[out_idx]

            try:
                r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
                if r.status_code != 200:
                    print(f"[WARN] {pg}: HTTP {r.status_code}")
                    fh.write(f"\n\n=== PAGE {pg} === {url}\n\n[HTTP {r.status_code}] لم يتم تنزيل الصفحة.\n")
                    continue

                text = extract_main_text(r.text)

                # Always write the banner, but NOT the 'بحــث' noise lines (already removed).
                #fh.write(f"\n\n=== PAGE {pg} === {url}\n\n")
                if looks_empty_or_noise(text):
                    # If you'd rather skip banners for empty pages entirely,
                    # comment the previous write() and uncomment the 'continue' below.
                    # continue
                    fh.write("[فارغ] لم يتم العثور على متن الصفحة (تم حذف واجهة البحث أو لا يوجد نص كافٍ).\n")
                else:
                    fh.write(text + "\n")

            except requests.RequestException as e:
                print(f"[ERR] {pg}: {e}")
                fh.write(f"\n\n=== PAGE {pg} === {url}\n\n[خطأ] فشل التنزيل: {e}\n")

            if DELAY > 0:
                time.sleep(DELAY)

    finally:
        for fh in outs:
            fh.close()

if __name__ == "__main__":
    crawl()
    print("✅ Done.")
