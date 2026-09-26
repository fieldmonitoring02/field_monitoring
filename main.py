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

# Fallback-এর জন্য মডেল লিস্ট (যে ক্রমানুসারে ট্রাই করতে চান)
MODELS_TO_TRY = [
    # "gemini-3.6-flash",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite"
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
# Image upload + Gemini analysis with Fallback
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

    response_text = None
    used_model = None

    # Fallback Loop: সিরিয়ালি একেকটি মডেল দিয়ে ট্রাই করবে
    for model_name in MODELS_TO_TRY:
        try:
            response = gemini.models.generate_content(
                model=model_name,
                contents=[
                    types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=img_b64)),
                    types.Part(text=ANALYSIS_PROMPT),
                ],
            )
            response_text = response.text
            used_model = model_name
            print(f"Success: Image analyzed using {used_model}")
            break # সফল হলে লুপ থেকে বেরিয়ে যাবে
        except Exception as e:
            print(f"Warning: Model {model_name} failed. Error: {e}")
            continue # ফেইল করলে পরের মডেলে ট্রাই করবে

    # যদি সবগুলো মডেলই ফেইল করে
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
        "used_model": used_model # কোন মডেল কাজ করেছে তা চেক করার জন্য
    }

@app.get("/latest")
async def get_latest():
    return latest
