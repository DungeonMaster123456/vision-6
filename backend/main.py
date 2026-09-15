"""
Vision 6 backend.
Images: Cloudflare Workers AI (free tier — 10,000 neurons/day, FLUX.1-schnell).
Video: your own Modal GPU deployment (LTX-Video).

Splitting image load onto Cloudflare frees up Modal's $30/month budget
entirely for higher-quality/longer video generations.

Run locally with:  uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
import uuid
import base64
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------- setup ----------

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get these from https://dash.cloudflare.com (Account ID on the overview page,
# API token from My Profile -> API Tokens -> Create Token -> "Workers AI" template)
CLOUDFLARE_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
CLOUDFLARE_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "")
CLOUDFLARE_IMAGE_MODEL = "@cf/black-forest-labs/flux-1-schnell"
CLOUDFLARE_URL = f"https://api.cloudflare.com/client/v4/accounts/{{account_id}}/ai/run/{CLOUDFLARE_IMAGE_MODEL}"

MODAL_VIDEO_GENERATE_URL = os.environ.get("MODAL_VIDEO_GENERATE_URL", "https://monikasic6--vision6-video-generate-video.modal.run")
MODAL_VIDEO_STATUS_URL = os.environ.get("MODAL_VIDEO_STATUS_URL", "https://monikasic6--vision6-video-video-status.modal.run")

app = FastAPI(title="Vision 6 Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/files", StaticFiles(directory=OUTPUT_DIR), name="files")


# ---------- request schemas ----------

class ImageRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024


class VideoRequest(BaseModel):
    prompt: str
    duration: int = 5


# ---------- endpoints ----------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "cloudflare_configured": bool(CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN),
        "video_configured": bool(MODAL_VIDEO_GENERATE_URL),
    }


@app.post("/generate-image")
def generate_image(req: ImageRequest):
    if not req.prompt.strip():
        raise HTTPException(400, "Prompt is empty")
    if not CLOUDFLARE_ACCOUNT_ID or not CLOUDFLARE_API_TOKEN:
        raise HTTPException(400, "CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN not set.")

    print(f"[vision6] Generating image via Cloudflare Workers AI: {req.prompt!r}")

    url = CLOUDFLARE_URL.format(account_id=CLOUDFLARE_ACCOUNT_ID)

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                url,
                headers={"Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}"},
                json={"prompt": req.prompt},
            )
            resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(502, f"Cloudflare image request failed ({e.response.status_code}): {e.response.text[:300]}")
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Cloudflare image request failed: {e}")

    data = resp.json()
    if not data.get("success"):
        raise HTTPException(502, f"Cloudflare returned an error: {data.get('errors')}")

    # Cloudflare's flux-schnell returns base64-encoded PNG data in result.image
    image_b64 = data["result"]["image"]
    image_bytes = base64.b64decode(image_b64)

    filename = f"{uuid.uuid4().hex}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "wb") as f:
        f.write(image_bytes)

    return {"url": f"/files/{filename}"}


@app.post("/generate-video")
def generate_video(req: VideoRequest):
    """
    Starts a video generation job on Modal and returns a call_id immediately.
    The frontend should poll /video-status?call_id=... until status is "done".
    """
    if not req.prompt.strip():
        raise HTTPException(400, "Prompt is empty")
    if not MODAL_VIDEO_GENERATE_URL:
        raise HTTPException(400, "MODAL_VIDEO_GENERATE_URL is not set. Deploy the video Modal app first.")

    print(f"[vision6] Starting video generation via Modal: {req.prompt!r}")

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(MODAL_VIDEO_GENERATE_URL, json={
                "prompt": req.prompt, "duration": req.duration
            })
            resp.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Modal video request failed: {e}")

    return resp.json()  # {"call_id": "fc-..."}


@app.get("/video-status")
def video_status(call_id: str):
    """
    Poll this with the call_id returned from /generate-video.
    Returns {"status": "pending"} while running, or {"status": "done", "url": "..."}
    once the video is ready and saved locally.
    """
    if not MODAL_VIDEO_STATUS_URL:
        raise HTTPException(400, "MODAL_VIDEO_STATUS_URL is not set.")

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(MODAL_VIDEO_STATUS_URL, params={"call_id": call_id})
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Modal status request failed: {e}")

    try:
        data = resp.json()
    except ValueError:
        # Transient empty/invalid response from Modal — treat as still pending
        # rather than crashing, so the frontend's polling loop just tries again.
        return {"status": "pending"}

    if data.get("status") != "done":
        return data  # pending / error / expired — pass through as-is

    # Decode the base64 video and save it locally, same as other outputs
    video_bytes = base64.b64decode(data["video_base64"])
    filename = f"{uuid.uuid4().hex}.mp4"
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "wb") as f:
        f.write(video_bytes)

    return {"status": "done", "url": f"/files/{filename}"}
