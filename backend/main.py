"""
Vision 6 backend.
Images: Cloudflare Workers AI (t2i + i2i, free tier).
Video: real stock footage from Pexels + Pixabay (free, instant, unlimited-feel).
Deep: HunyuanVideo 1.5 real generation on Modal H100 — premium, quota-limited
      to keep monthly cost under the $30 free Modal budget (~$0.30/video).

Run locally with:  uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
import uuid
import base64
import datetime
import json as _json
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------- setup ----------

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLOUDFLARE_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
CLOUDFLARE_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "")
CLOUDFLARE_IMAGE_MODEL = "@cf/black-forest-labs/flux-1-schnell"
CLOUDFLARE_IMG2IMG_MODEL = "@cf/runwayml/stable-diffusion-v1-5-img2img"
CLOUDFLARE_URL = f"https://api.cloudflare.com/client/v4/accounts/{{account_id}}/ai/run/{CLOUDFLARE_IMAGE_MODEL}"
CLOUDFLARE_IMG2IMG_URL = f"https://api.cloudflare.com/client/v4/accounts/{{account_id}}/ai/run/{CLOUDFLARE_IMG2IMG_MODEL}"

# Get free keys at https://www.pexels.com/api/ and https://pixabay.com/api/docs/
PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")
PIXABAY_API_KEY = os.environ.get("PIXABAY_API_KEY", "")

# Deep mode (HunyuanVideo 1.5 on Modal H100) — set after `modal deploy` in modal_deep/
MODAL_DEEP_GENERATE_URL = os.environ.get("MODAL_DEEP_GENERATE_URL", "")
MODAL_DEEP_STATUS_URL = os.environ.get("MODAL_DEEP_STATUS_URL", "")

app = FastAPI(title="Vision 6 Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/files", StaticFiles(directory=OUTPUT_DIR), name="files")


# ---------- Public API keys (PER-xxxxxxxx) ----------
# Simple functional key system: generate once, store in a local JSON file,
# validate on requests to public API endpoints. Not enterprise-grade auth,
# but genuinely functional — a valid key is required and checked.
import secrets

API_KEYS_FILE = os.path.join(os.path.dirname(__file__), "api_keys.json")


def _load_api_keys() -> dict:
    if os.path.exists(API_KEYS_FILE):
        try:
            with open(API_KEYS_FILE) as f:
                return _json.load(f)
        except (ValueError, OSError):
            pass
    return {}


def _save_api_keys(keys: dict):
    with open(API_KEYS_FILE, "w") as f:
        _json.dump(keys, f)


def _generate_api_key() -> str:
    keys = _load_api_keys()
    new_key = "PER" + secrets.token_hex(16).upper()
    keys[new_key] = {"created": str(datetime.date.today()), "requests": 0}
    _save_api_keys(keys)
    return new_key


def _validate_api_key(key: str) -> bool:
    keys = _load_api_keys()
    if key not in keys:
        return False
    keys[key]["requests"] = keys[key].get("requests", 0) + 1
    _save_api_keys(keys)
    return True


from fastapi import Header


def require_api_key(x_api_key: str = Header(None)):
    if not x_api_key or not _validate_api_key(x_api_key):
        raise HTTPException(401, "Missing or invalid API key. Pass it as the X-API-Key header.")
    return x_api_key


# ---------- Deep mode quota ----------
# 2 videos/day most months, 3/day in February — keeps monthly usage safely
# under the $30 Modal budget at roughly $0.30/video (H100 rate).
QUOTA_FILE = os.path.join(os.path.dirname(__file__), "deep_quota.json")


def _daily_deep_limit() -> int:
    return 3 if datetime.date.today().month == 2 else 2


def _load_quota_state() -> dict:
    if os.path.exists(QUOTA_FILE):
        try:
            with open(QUOTA_FILE) as f:
                return _json.load(f)
        except (ValueError, OSError):
            pass
    return {}


def _save_quota_state(state: dict):
    with open(QUOTA_FILE, "w") as f:
        _json.dump(state, f)


def _get_remaining_today() -> int:
    state = _load_quota_state()
    today_str = str(datetime.date.today())
    used_today = state.get(today_str, 0)
    return max(0, _daily_deep_limit() - used_today)


def _consume_one_deep_credit():
    state = _load_quota_state()
    today_str = str(datetime.date.today())
    state[today_str] = state.get(today_str, 0) + 1
    cutoff = datetime.date.today() - datetime.timedelta(days=3)
    state = {k: v for k, v in state.items() if k >= str(cutoff)}
    _save_quota_state(state)


# ---------- request schemas ----------

class ImageRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024


class ImageEditRequest(BaseModel):
    prompt: str
    image_base64: str  # source image, base64-encoded, no data: prefix
    strength: float = 0.75  # how much to transform (0 = keep original, 1 = ignore original)


class VideoSearchRequest(BaseModel):
    prompt: str


class DeepVideoRequest(BaseModel):
    prompt: str
    duration: int = 2


# ---------- endpoints ----------

@app.post("/api-keys/generate")
def generate_api_key():
    """Generates a new public API key (PER-xxxxxxxx). No auth needed to create one."""
    key = _generate_api_key()
    return {"api_key": key}


@app.post("/public/generate-image")
def public_generate_image(req: ImageRequest, api_key: str = Header(None, alias="X-API-Key")):
    """Public API version of image generation. Requires a valid X-API-Key header."""
    require_api_key(api_key)
    return generate_image(req)


@app.post("/public/search-video")
def public_search_video(req: VideoSearchRequest, api_key: str = Header(None, alias="X-API-Key")):
    """Public API version of video search. Requires a valid X-API-Key header."""
    require_api_key(api_key)
    return search_video(req)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "cloudflare_configured": bool(CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN),
        "pexels_configured": bool(PEXELS_API_KEY),
        "pixabay_configured": bool(PIXABAY_API_KEY),
        "deep_configured": bool(MODAL_DEEP_GENERATE_URL),
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

    import base64
    data = resp.json()
    if not data.get("success"):
        raise HTTPException(502, f"Cloudflare returned an error: {data.get('errors')}")

    image_bytes = base64.b64decode(data["result"]["image"])
    filename = f"{uuid.uuid4().hex}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "wb") as f:
        f.write(image_bytes)

    return {"url": f"/files/{filename}"}


@app.post("/edit-image")
def edit_image(req: ImageEditRequest):
    """
    Image-to-image: takes an uploaded photo + prompt, returns a transformed
    version. Uses Cloudflare's SD 1.5 img2img model — same account/token as
    the text-to-image endpoint, no extra setup needed.
    """
    if not req.prompt.strip():
        raise HTTPException(400, "Prompt is empty")
    if not req.image_base64:
        raise HTTPException(400, "image_base64 is required")
    if not CLOUDFLARE_ACCOUNT_ID or not CLOUDFLARE_API_TOKEN:
        raise HTTPException(400, "CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN not set.")

    print(f"[vision6] Editing image via Cloudflare img2img: {req.prompt!r}")

    url = CLOUDFLARE_IMG2IMG_URL.format(account_id=CLOUDFLARE_ACCOUNT_ID)

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                url,
                headers={"Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}"},
                json={
                    "prompt": req.prompt,
                    "image_b64": req.image_base64,
                    "strength": req.strength,
                    "guidance": 7.5,
                    "num_steps": 20,
                },
            )
            resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(502, f"Cloudflare img2img request failed ({e.response.status_code}): {e.response.text[:300]}")
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Cloudflare img2img request failed: {e}")

    # This model returns raw image bytes directly (not JSON+base64 like flux-schnell)
    content_type = resp.headers.get("content-type", "")
    if "application/json" in content_type:
        raise HTTPException(502, f"Cloudflare returned an error: {resp.text[:300]}")

    filename = f"{uuid.uuid4().hex}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "wb") as f:
        f.write(resp.content)

    return {"url": f"/files/{filename}"}


@app.post("/search-video")
def search_video(req: VideoSearchRequest):
    """
    Searches Pexels and Pixabay for real stock footage matching the prompt.
    Returns up to 10 results the user can swipe through and pick from —
    real, high-quality footage instead of unreliable AI generation.
    """
    if not req.prompt.strip():
        raise HTTPException(400, "Prompt is empty")

    results = []

    # Pexels
    if PEXELS_API_KEY:
        try:
            with httpx.Client(timeout=20.0) as client:
                resp = client.get(
                    "https://api.pexels.com/videos/search",
                    headers={"Authorization": PEXELS_API_KEY},
                    params={"query": req.prompt, "per_page": 6},
                )
                resp.raise_for_status()
                data = resp.json()
                for v in data.get("videos", []):
                    # pick a reasonably sized file (avoid 4K when possible)
                    files = sorted(v.get("video_files", []), key=lambda f: f.get("width", 0))
                    best = next((f for f in files if f.get("width", 0) >= 640), files[0] if files else None)
                    if best:
                        results.append({
                            "url": best["link"],
                            "thumbnail": v.get("image"),
                            "source": "Pexels",
                            "credit": v.get("user", {}).get("name", "Pexels contributor"),
                            "source_url": v.get("url"),
                        })
        except httpx.HTTPError as e:
            print(f"[vision6] Pexels search failed: {e}")

    # Pixabay
    if PIXABAY_API_KEY:
        try:
            with httpx.Client(timeout=20.0) as client:
                resp = client.get(
                    "https://pixabay.com/api/videos/",
                    params={"key": PIXABAY_API_KEY, "q": req.prompt, "per_page": 6},
                )
                resp.raise_for_status()
                data = resp.json()
                for v in data.get("hits", []):
                    medium = v.get("videos", {}).get("medium", {})
                    if medium.get("url"):
                        results.append({
                            "url": medium["url"],
                            "thumbnail": f"https://i.vimeocdn.com/video/{v.get('picture_id')}_640x360.jpg" if v.get("picture_id") else None,
                            "source": "Pixabay",
                            "credit": v.get("user", "Pixabay contributor"),
                            "source_url": v.get("pageURL"),
                        })
        except httpx.HTTPError as e:
            print(f"[vision6] Pixabay search failed: {e}")

    if not results:
        raise HTTPException(404, "No videos found for that description. Try different wording.")

    return {"results": results[:10]}


@app.get("/deep-quota")
def deep_quota():
    """Returns how many Deep-mode videos are left today."""
    return {
        "remaining": _get_remaining_today(),
        "daily_limit": _daily_deep_limit(),
    }


@app.post("/generate-deep-video")
def generate_deep_video(req: DeepVideoRequest):
    """
    Starts a HunyuanVideo 1.5 generation job on Modal ("Deep" mode).
    Quota-limited: checks and consumes one credit before starting.
    """
    if not req.prompt.strip():
        raise HTTPException(400, "Prompt is empty")
    if not MODAL_DEEP_GENERATE_URL:
        raise HTTPException(400, "MODAL_DEEP_GENERATE_URL is not set. Deploy modal_deep/modal_app.py first.")

    remaining = _get_remaining_today()
    if remaining <= 0:
        raise HTTPException(
            429,
            f"Daily Deep video limit reached (0/{_daily_deep_limit()} remaining today). "
            "Resets at midnight."
        )

    print(f"[vision6] Starting Deep video generation via Modal: {req.prompt!r} ({remaining} left today)")

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(MODAL_DEEP_GENERATE_URL, json={
                "prompt": req.prompt, "duration": req.duration
            })
            resp.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Modal Deep video request failed: {e}")

    # Only consume the credit once the job is confirmed started
    _consume_one_deep_credit()

    data = resp.json()  # {"call_id": "fc-..."}
    data["remaining_after"] = _get_remaining_today()
    return data


@app.get("/deep-video-status")
def deep_video_status(call_id: str):
    """
    Poll this with the call_id returned from /generate-deep-video.
    Returns {"status": "pending"} while running, or {"status": "done", "url": "..."}
    once the video is ready and saved locally.
    """
    if not MODAL_DEEP_STATUS_URL:
        raise HTTPException(400, "MODAL_DEEP_STATUS_URL is not set.")

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(MODAL_DEEP_STATUS_URL, params={"call_id": call_id})
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Modal Deep status request failed: {e}")

    try:
        data = resp.json()
    except ValueError:
        return {"status": "pending"}

    if data.get("status") != "done":
        return data

    video_bytes = base64.b64decode(data["video_base64"])
    filename = f"{uuid.uuid4().hex}.mp4"
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "wb") as f:
        f.write(video_bytes)

    return {"status": "done", "url": f"/files/{filename}"}
