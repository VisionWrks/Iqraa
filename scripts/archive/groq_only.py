#!/usr/bin/env python3
import os, sys, json, base64, argparse, mimetypes, math
import requests

API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = os.getenv(
    "GROQ_MODEL",
    "meta-llama/llama-4-maverick-17b-128e-instruct"  # strong vision model
)

SYSTEM_PROMPT = (
    "You are a precise transcription assistant. Extract ALL visible text from each image, "
    "preserving natural reading order and line breaks. Do not summarize or describe. "
    "keep Arabic script as-is. Correct only obvious OCR artifacts when certain; "
    "do not invent words. Return ONLY the transcribed text. once you transcribe the text, "
    "review it all again for grammar and misspellings, and correct them. "
)

def require_key():
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise SystemExit("GROQ_API_KEY is not set")
    return key

def encode_local_image_to_data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        # fall back, most OCR pages are PNG/JPEG
        mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def make_image_block(img: str) -> dict:
    if img.startswith(("http://", "https://")):
        return {"type": "image_url", "image_url": {"url": img}}
    return {"type": "image_url", "image_url": {"url": encode_local_image_to_data_url(img)}}

def call_groq_vision(images, model, lang_hint=None, temperature=0.0, max_out=None) -> str:
    """
    images: list[str] (file paths or URLs), up to 5 per request (API limit).
    """
    key = require_key()

    if lang_hint == "ar":
        user_text = "رجاءً انسخ كل النص الظاهر في الصور بالحروف الأصلية وبنفس ترتيب الأسطر."
    elif lang_hint == "en":
        user_text = "Please transcribe all visible text from the images in their original script and line order."
    else:
        user_text = "Please transcribe all text from these images exactly as seen."

    content = [{"type": "text", "text": user_text}] + [make_image_block(i) for i in images]

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "temperature": temperature,
        "stream": False,
    }
    if max_out is not None:
        payload["max_completion_tokens"] = max_out

    r = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=600,
    )
    if r.status_code >= 400:
        # helpful hint if user picked a text-only model
        if "content must be a string" in r.text:
            raise RuntimeError(
                f"Model likely not vision-capable. Use a vision model like "
                f"'meta-llama/llama-4-maverick-17b-128e-instruct' or '...-scout-...'. "
                f"Full response: {r.text}"
            )
        raise RuntimeError(f"{r.status_code} {r.text}")
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()

def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def main():
    ap = argparse.ArgumentParser(description="Transcribe image(s) via Groq Cloud vision models.")
    ap.add_argument("--image", action="append", required=True,
                    help="Image path or URL. Repeat --image for multiple (≤5 per request).")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="Vision model (default: meta-llama/llama-4-maverick-17b-128e-instruct)")
    ap.add_argument("--lang", choices=["ar","en"], default=None, help="Optional hint")
    ap.add_argument("--max-out", type=int, default=None, help="Optional max completion tokens")
    args = ap.parse_args()

    # API supports up to 5 images per request. If more, batch and concat results.
    results = []
    for batch in chunk(args.image, 5):
        # sanity checks: local files must exist
        for path in batch:
            if not (path.startswith(("http://","https://")) or os.path.exists(path)):
                raise SystemExit(f"Image not found: {path}")
        text = call_groq_vision(batch, model=args.model, lang_hint=args.lang, max_out=args.max_out)
        results.append(text)

    output = "\n".join(results).strip()
    print(output)

if __name__ == "__main__":
    main()
