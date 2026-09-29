import os
import time
import asyncio
import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI()
gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Fallback model list (tried in this order).
# NOTE: gemini-2.5-* models are scheduled for shutdown on 16 Oct 2026,
# so the 3.x models are tried first to keep this working after that date.
MODELS_TO_TRY = [
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------
state = {
    "mode": "auto",
    "interval_ms": 60000,
    "last_auto_capture_ts": 0.0,
    "capture_pending": False,
}

latest = {
    "url": None,
    "analysis": "Waiting for first image...",
    "status": "Unknown",
    "disease_detected": False,
    "timestamp": None,
}

class ImagePayload(BaseModel):
    image_url: str

class ModePayload(BaseModel):
    mode: str

class IntervalPayload(BaseModel):
    interval_ms: int

# ---------------------------------------------------------------------------
# Dashboard & Command Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    with open("index.html") as f:
        return HTMLResponse(f.read())

@app.get("/command")
async def get_command():
    now = time.time()
    should_capture = False

    if state["mode"] == "auto":
        interval_s = state["interval_ms"] / 1000.0
        if now - state["last_auto_capture_ts"] >= interval_s:
            state["last_auto_capture_ts"] = now
            should_capture = True
    else:
        if state["capture_pending"]:
            state["capture_pending"] = False
            should_capture = True

    return {
        "mode": state["mode"],
        "interval_ms": state["interval_ms"],
        "capture": should_capture,
    }

@app.post("/mode")
async def set_mode(payload: ModePayload):
    if payload.mode not in ("auto", "manual"):
        return {"status": "error", "message": "mode must be 'auto' or 'manual'"}
    state["mode"] = payload.mode
    state["capture_pending"] = False
    state["last_auto_capture_ts"] = time.time()
    return {"status": "ok", "mode": state["mode"]}

@app.post("/interval")
async def set_interval(payload: IntervalPayload):
    if payload.interval_ms < 5000:
        return {"status": "error", "message": "interval_ms must be >= 5000"}
    state["interval_ms"] = payload.interval_ms
    return {"status": "ok", "interval_ms": state["interval_ms"]}

@app.post("/capture")
async def trigger_capture():
    state["capture_pending"] = True
    return {"status": "ok", "message": "capture requested"}

@app.get("/status")
async def get_status():
    return state

# ---------------------------------------------------------------------------
# Image upload + Gemini analysis with fallback
# ---------------------------------------------------------------------------
ANALYSIS_PROMPT = (
    "You are an agricultural crop-health inspector analyzing a field image.\n"
    "Respond in EXACTLY this format:\n"
    "STATUS: <one word — HEALTHY, DISEASED, STRESSED, or UNKNOWN>\n"
    "ALERT: <one short sentence, max 12 words, suitable for a small OLED screen>\n"
    "ANALYSIS: <a detailed explanation, under 100 words, describing plant health, "
    "any visible disease/pest/stress signs, and a brief recommendation>"
)

def parse_gemini_response(text: str):
    # Strip stray markdown the model sometimes adds (e.g. "**STATUS:**")
    lines = [ln.replace("*", "").replace("#", "").strip() for ln in text.strip().splitlines()]
    parsed = {}
    current_key = None
    for line in lines:
        matched = False
        for key in ("STATUS:", "ALERT:", "ANALYSIS:"):
            if line.upper().startswith(key):
                current_key = key[:-1]
                parsed[current_key] = line.split(":", 1)[1].strip()
                matched = True
                break
        if not matched and current_key:
            parsed[current_key] = parsed.get(current_key, "") + " " + line

    status = parsed.get("STATUS", "UNKNOWN").upper()
    alert = parsed.get("ALERT", "").strip()
    analysis = parsed.get("ANALYSIS", text.strip()).strip()

    if status not in ("HEALTHY", "DISEASED", "STRESSED", "UNKNOWN"):
        status = "UNKNOWN"

    return status, alert, analysis

def call_gemini_sync(model_name: str, image_bytes: bytes) -> str:
    """Blocking Gemini SDK call — always run this via asyncio.to_thread,
    never awaited directly, or it freezes the whole FastAPI event loop."""
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
    response = gemini.models.generate_content(
        model=model_name,
        contents=[image_part, types.Part(text=ANALYSIS_PROMPT)],
    )
    return response.text

@app.post("/upload")
async def upload(payload: ImagePayload):
    url = payload.image_url

    # --- Download the image, with real error handling ---
    try:
        async with httpx.AsyncClient(timeout=20.0) as http:
            r = await http.get(url)
            r.raise_for_status()
            image_bytes = r.content  # raw JPEG bytes — never base64-encode this
    except Exception as e:
        print(f"Warning: could not download image from ImageKit: {e}")
        latest.update({
            "url": url,
            "analysis": "Could not download the image from ImageKit.",
            "status": "UNKNOWN",
            "disease_detected": False,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        return {
            "status": "UNKNOWN",
            "alert": "IMAGE DOWNLOAD FAILED",
            "disease_detected": False,
            "analysis": "Could not download the image from ImageKit.",
            "used_model": None,
        }

    response_text = None
    used_model = None

    # Fallback loop: try each model in order until one succeeds.
    # The actual Gemini call is offloaded to a worker thread so it
    # doesn't block /command, /latest, /status while it runs.
    for model_name in MODELS_TO_TRY:
        try:
            response_text = await asyncio.to_thread(call_gemini_sync, model_name, image_bytes)
            used_model = model_name
            print(f"Success: Image analyzed using {used_model}")
            break
        except Exception as e:
            print(f"Warning: Model {model_name} failed. Error: {e}")
            continue

    if not response_text:
        status, alert = "UNKNOWN", "API ERROR"
        analysis = "All Gemini models failed to process the image."
        disease_detected = False
    else:
        status, alert, analysis = parse_gemini_response(response_text)
        disease_detected = status == "DISEASED"

    latest["url"] = url
    latest["analysis"] = analysis
    latest["status"] = status
    latest["disease_detected"] = disease_detected
    latest["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    return {
        "status": status,
        "alert": alert or status,
        "disease_detected": disease_detected,
        "analysis": analysis,
        "used_model": used_model,
    }

@app.get("/latest")
async def get_latest():
    return latest
