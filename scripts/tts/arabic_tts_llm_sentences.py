#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import edge_tts
import requests
from tqdm import tqdm


# -----------------------------
# Utilities
# -----------------------------
def clean_text(txt: str) -> str:
    lines = [ln.strip() for ln in txt.splitlines()]
    return "\n".join([ln for ln in lines if ln])


def regex_split_sentences(text: str):
    """Simple Arabic/Latin sentence splitter as fallback."""
    sentences = re.split(r"(?<=[\.!\؟\!])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _ollama_base_url(cli_url: Optional[str] = None) -> str:
    """Return the base URL for Ollama."""
    return (cli_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")


# -----------------------------
# JSON parsing helpers
# -----------------------------
def _parse_json_sentences(payload: str):
    """
    Accept:
      1) Array of strings: ["جملة 1.","جملة 2."]
      2) Object: {"sentences": ["جملة 1.","جملة 2."]}
    If the top-level isn't JSON, try to salvage the first array substring.
    """
    def _as_list(data):
        if isinstance(data, list):
            return [s.strip() for s in data if isinstance(s, str) and s.strip()]
        if isinstance(data, dict) and "sentences" in data and isinstance(data["sentences"], list):
            return [s.strip() for s in data["sentences"] if isinstance(s, str) and s.strip()]
        return None

    try:
        data = json.loads(payload)
        arr = _as_list(data)
        if arr:
            return arr
    except Exception:
        pass

    # salvage array substring
    m = re.search(r"\[(?:.|\n)*\]", payload)
    if m:
        try:
            data = json.loads(m.group(0))
            arr = _as_list(data)
            if arr:
                return arr
        except Exception:
            pass
    return None


# -----------------------------
# Ollama-based sentence segmentation
# -----------------------------
def split_with_ollama(
    text: str,
    model: str,
    base_url: Optional[str],
    timeout_s: int,
    endpoint: str,  # "auto" | "generate" | "chat"
):
    """
    Use Ollama to split text into sentences.
    - Tries /api/generate (format=json) and/or /api/chat (format=json).
    - Falls back to regex splitting when needed.
    """
    base = _ollama_base_url(base_url)

    sys_msg = "You are an arabic language expert and a professional text editor. That only resturn JSON"
    
    user_msg = (
        "Split the following Arabic text into sentences that can be spoken compfortabily without cutting the meaning, keeping all of the original text intact. "
        "Do not omit, paraphrase, or summarize anything — only divide it into full sentences. divide more when you are unsure"
        "Go through the whole text and check the result twice"
        "check the words count before and after the split, if they dont match check where you omitted words and restore it"
        "Return the result strictly as JSON only, in one of the two valid formats below:\n"
        "1) [\"Sentence 1.\", \"Sentence 2.\"]\n"
        "2) {\"sentences\": [\"Sentence 1.\", \"Sentence 2.\"]}\n\n"
        f"Text:\n{text}"
    )

    def try_generate():
        r = requests.post(
            f"{base}/api/generate",
            json={
                "model": model,
                "prompt": f"System: {sys_msg}\n\nUser: {user_msg}",
                "stream": False,
                "options": {"temperature": 0.1},
                "format": "json",
            },
            timeout=timeout_s,
        )
        r.raise_for_status()
        raw = r.json().get("response", "")
        return _parse_json_sentences(raw)

    def try_chat():
        r = requests.post(
            f"{base}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg},
                ],
                "stream": False,
                "options": {"temperature": 0.1},
                "format": "json",
            },
            timeout=timeout_s,
        )
        r.raise_for_status()
        j = r.json()
        raw = (j.get("message", {}) or {}).get("content") or j.get("response", "")
        return _parse_json_sentences(raw)

    # Endpoint selection
    if endpoint == "generate":
        try:
            arr = try_generate()
            if arr:
                return arr
            print("[INFO] /api/generate returned non-JSON or empty; using regex fallback.")
        except Exception as e:
            print(f"[INFO] /api/generate failed: {e}")
        return regex_split_sentences(text)

    if endpoint == "chat":
        try:
            arr = try_chat()
            if arr:
                return arr
            print("[INFO] /api/chat returned non-JSON or empty; using regex fallback.")
        except Exception as e:
            print(f"[INFO] /api/chat failed: {e}")
        return regex_split_sentences(text)

    # auto: try generate then chat
    try:
        arr = try_generate()
        if arr:
            return arr
        print("[INFO] /api/generate returned non-JSON or empty; trying /api/chat…")
    except Exception as e:
        print(f"[INFO] /api/generate failed or slow: {e}. Trying /api/chat…")

    try:
        arr = try_chat()
        if arr:
            return arr
        print("[WARN] /api/chat returned non-JSON or empty; using regex fallback.")
    except Exception as e:
        print(f"[WARN] /api/chat failed: {e}")

    return regex_split_sentences(text)


# -----------------------------
# TTS
# -----------------------------
async def speak_sentence(sentence: str, out_file: Path, voice: str, rate: str, volume: str):
    communicate = edge_tts.Communicate(sentence, voice=voice, rate=rate, volume=volume)
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    with out_file.open("ab") as f:
        f.write(audio_data)


# -----------------------------
# Main
# -----------------------------
async def main_async(args):
    text_path = Path(args.file)
    if not text_path.exists():
        print(f"❌ File not found: {text_path}")
        return

    text = clean_text(text_path.read_text(encoding="utf-8"))
    if not text:
        print("⚠️ Input file is empty.")
        return

    print(
        f"🧠 Splitting text using Ollama model: {args.model} @ "
        f"{_ollama_base_url(args.ollama_url)} (endpoint={args.endpoint}, timeout={args.ollama_timeout}s)"
    )
    sentences = split_with_ollama(
        text,
        model=args.model,
        base_url=args.ollama_url,
        timeout_s=args.ollama_timeout,
        endpoint=args.endpoint,
    )
    print(f"🗂 Found {len(sentences)} sentences.")

    out_path = Path(args.out)
    if out_path.exists():
        out_path.unlink()

    for s in tqdm(sentences, desc="Speaking", unit="sentence"):
        await speak_sentence(s, out_path, args.voice, args.rate, args.volume)
        time.sleep(args.pause)

    print(f"✅ Done! Saved to {out_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Arabic TTS with Ollama sentence segmentation")
    parser.add_argument("file", help="Path to Arabic text file (.txt)")
    parser.add_argument("-o", "--out", default="output.mp3", help="Output filename (MP3)")
    parser.add_argument("-v", "--voice", default="ar-SA-HamedNeural", help="Arabic neural voice")
    parser.add_argument("-r", "--rate", default="+0%", help="Rate (e.g., +5%, -10%)")
    parser.add_argument("-V", "--volume", default="+0%", help="Volume (e.g., +10%)")
    parser.add_argument("--pause", type=float, default=0.8, help="Pause (seconds) between sentences")
    parser.add_argument("--model", default="llama3.2:1b", help="Ollama model name")
    parser.add_argument("--ollama-url", default=None, help="Base URL for Ollama (e.g., http://localhost:11434)")
    parser.add_argument("--ollama-timeout", type=int, default=180, help="Ollama HTTP timeout (seconds)")
    parser.add_argument(
        "--endpoint",
        choices=["auto", "generate", "chat"],
        default="auto",
        help="Which Ollama endpoint to use",
    )
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
