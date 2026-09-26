import os
import time
import base64
import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI()
gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# ---------------------------------------------------------------------------
# In-memory state (fine for a single ESP32 device / single-instance deploy)
# ---------------------------------------------------------------------------
state = {
    # capture mode: "auto" or "manual"
    "mode": "auto",
    # auto-capture interval in milliseconds, editable from the dashboard
    "interval_ms": 60000,
    # timestamp (server epoch seconds) of the last auto capture that was issued
    "last_auto_capture_ts": 0.0,
    # one-shot flag set by the dashboard "Capture Now" button (manual mode)
    "capture_pending": False,
}

latest = {
    "url": None,
    "analysis": "Waiting for first image...",
    "status": "Unknown",          # HEALTHY / DISEASED / STRESSED / UNKNOWN
    "disease_detected": False,
    "timestamp": None,
}


class ImagePayload(BaseModel):
    image_url: str


class ModePayload(BaseModel):
    mode: str  # "auto" | "manual"


class IntervalPayload(BaseModel):
    interval_ms: int


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    with open("index.html") as f:
        return HTMLResponse(f.read())


# ---------------------------------------------------------------------------
# ESP32 polls this endpoint every few seconds to find out whether it should
# capture right now, and what mode/interval it should be operating under.
# The device itself stays "dumb" — all scheduling logic lives on the server
# so it can be reconfigured live from the web dashboard without reflashing.
# ---------------------------------------------------------------------------
@app.get("/command")
async def get_command():
    now = time.time()
    should_capture = False

    if state["mode"] == "auto":
        interval_s = state["interval_ms"] / 1000.0
        if now - state["last_auto_capture_ts"] >= interval_s:
            state["last_auto_capture_ts"] = now
            should_capture = True
    else:  # manual mode
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
    # reset any pending manual trigger / auto timer so the switch feels immediate
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
    """Dashboard 'Capture Now' button — only meaningful in manual mode."""
    state["capture_pending"] = True
    return {"status": "ok", "message": "capture requested"}


@app.get("/status")
async def get_status():
    return state


# ---------------------------------------------------------------------------
# Image upload + Gemini analysis
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
    lines = text.strip().splitlines()
    parsed = {}
    current_key = None
    for line in lines:
        matched = False
        for key in ("STATUS:", "ALERT:", "ANALYSIS:"):
            if line.strip().upper().startswith(key):
                current_key = key[:-1]
                parsed[current_key] = line.split(":", 1)[1].strip()
                matched = True
                break
        if not matched and current_key:
            parsed[current_key] = parsed.get(current_key, "") + " " + line.strip()

    status = parsed.get("STATUS", "UNKNOWN").upper()
    alert = parsed.get("ALERT", "").strip()
    analysis = parsed.get("ANALYSIS", text.strip()).strip()

    if status not in ("HEALTHY", "DISEASED", "STRESSED", "UNKNOWN"):
        status = "UNKNOWN"

    return status, alert, analysis


@app.post("/upload")
async def upload(payload: ImagePayload):
    url = payload.image_url
    async with httpx.AsyncClient() as http:
        r = await http.get(url)
        img_b64 = base64.b64encode(r.content).decode()

    response = gemini.models.generate_content(
        model="gemini-1.5-flash",
        contents=[
            types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=img_b64)),
            types.Part(text=ANALYSIS_PROMPT),
        ],
    )

    status, alert, analysis = parse_gemini_response(response.text)
    disease_detected = status == "DISEASED"

    latest["url"] = url
    latest["analysis"] = analysis
    latest["status"] = status
    latest["disease_detected"] = disease_detected
    latest["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # This payload is what the ESP32 receives back in the HTTP response body
    # from notifyBackend() — it uses "status" + "alert" to drive the OLED.
    return {
        "status": status,
        "alert": alert or status,
        "disease_detected": disease_detected,
        "analysis": analysis,
    }


@app.get("/latest")
async def get_latest():
    return latest
