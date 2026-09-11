"""
Vision 6 backend — powered by Pollinations.ai
Images: free, no API key, uses the Flux model (good quality).
Video: free, requires one API key from https://enter.pollinations.ai/keys (no credit card).

Run with:  uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
import uuid
import urllib.parse
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------- setup ----------

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get a free key at https://enter.pollinations.ai/keys (only needed for video)
# Set it as an environment variable before running:
#   export POLLINATIONS_API_KEY=your_key_here
POLLINATIONS_API_KEY = os.environ.get("POLLINATIONS_API_KEY", "")

IMAGE_BASE = "https://image.pollinations.ai/prompt"
VIDEO_BASE = "https://gen.pollinations.ai/video"

app = FastAPI(title="Vision 6 Backend (Pollinations)")

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
    duration: int = 5          # seconds
    aspect_ratio: str = "16:9"


# ---------- endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok", "video_key_configured": bool(POLLINATIONS_API_KEY)}


@app.post("/generate-image")
def generate_image(req: ImageRequest):
    if not req.prompt.strip():
        raise HTTPException(400, "Prompt is empty")

    # model=flux gives noticeably better quality than the default model.
    # nologo=true removes the pollinations watermark. seed randomizes each call.
    encoded_prompt = urllib.parse.quote(req.prompt)
    params = {
        "model": "flux",
        "width": req.width,
        "height": req.height,
        "nologo": "true",
        "seed": uuid.uuid4().int % 1_000_000,
    }
    url = f"{IMAGE_BASE}/{encoded_prompt}?{urllib.parse.urlencode(params)}"

    print(f"[vision6] Generating image: {req.prompt!r}")

    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Pollinations image request failed: {e}")

    filename = f"{uuid.uuid4().hex}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "wb") as f:
        f.write(resp.content)

    return {"url": f"/files/{filename}"}


@app.post("/generate-video")
def generate_video(req: VideoRequest):
    if not req.prompt.strip():
        raise HTTPException(400, "Prompt is empty")

    if not POLLINATIONS_API_KEY:
        raise HTTPException(
            400,
            "No Pollinations API key configured. Get a free one at "
            "https://enter.pollinations.ai/keys and set it with: "
            "export POLLINATIONS_API_KEY=your_key_here (then restart the server)."
        )

    encoded_prompt = urllib.parse.quote(req.prompt)
    params = {
        "key": POLLINATIONS_API_KEY,
        "duration": req.duration,
        "aspectRatio": req.aspect_ratio,
    }
    url = f"{VIDEO_BASE}/{encoded_prompt}?{urllib.parse.urlencode(params)}"

    print(f"[vision6] Generating video: {req.prompt!r} (can take a minute or two)")

    try:
        with httpx.Client(timeout=300.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Pollinations video request failed: {e}")

    filename = f"{uuid.uuid4().hex}.mp4"
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "wb") as f:
        f.write(resp.content)

    return {"url": f"/files/{filename}"}

