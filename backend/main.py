"""
Vision 6 — "Deep" mode: HunyuanVideo 1.5 generation on Modal H100.
Premium, quota-limited (see backend/main.py for the daily quota logic).

Deploy with:  modal deploy modal_app.py
"""

import modal
import io
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

app = modal.App("vision6-deep")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "sentencepiece",
        "protobuf",
        "fastapi[standard]",
        "imageio",
        "imageio-ffmpeg",
        "hf_transfer",
    )
)

with image.imports():
    import torch
    from diffusers import HunyuanVideoPipeline
    from diffusers.utils import export_to_video

# Kept intentionally modest — H100 is expensive, and this mode is quota-limited
# server-side anyway. Short clips, reasonable resolution.
MAX_FRAMES = 49         # ~2s at 24fps
MAX_STEPS = 30
WIDTH = 768
HEIGHT = 512


@app.cls(
    gpu="H100",
    image=image,
    scaledown_window=120,
    timeout=900,
)
class DeepVideoModel:
    def __init__(self):
        self.pipe = None

    def _load(self):
        if self.pipe is None:
            self.pipe = HunyuanVideoPipeline.from_pretrained(
                "hunyuanvideo-community/HunyuanVideo",
                dtype=torch.bfloat16,
                device_map="balanced",
            )

    @modal.method()
    def generate(self, prompt: str, num_frames: int, steps: int) -> bytes:
        num_frames = min(num_frames, MAX_FRAMES)
        steps = min(steps, MAX_STEPS)

        self._load()
        result = self.pipe(
            prompt=prompt,
            width=WIDTH,
            height=HEIGHT,
            num_frames=num_frames,
            num_inference_steps=steps,
        )
        frames = result.frames[0]

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
        export_to_video(frames, tmp_path, fps=24)
        with open(tmp_path, "rb") as f:
            data = f.read()
        os.unlink(tmp_path)
        return data


@app.function(image=image, timeout=900)
@modal.fastapi_endpoint(method="POST")
def generate_deep_video(item: dict):
    """
    Starts generation asynchronously and returns a call_id immediately.
    Poll /deep-video-status?call_id=... to check progress and get the result.
    NOTE: quota enforcement happens in the main backend (main.py), not here —
    this endpoint trusts that the caller already checked the daily limit.
    """
    from fastapi import Response
    import json

    prompt = item.get("prompt", "").strip()
    if not prompt:
        return Response(content='{"error":"Prompt is empty"}', status_code=400, media_type="application/json")

    duration_seconds = item.get("duration", 2)
    num_frames = max(9, min(int(duration_seconds * 24), MAX_FRAMES))

    model = DeepVideoModel()
    call = model.generate.spawn(
        prompt=prompt,
        num_frames=num_frames,
        steps=MAX_STEPS,
    )
    return Response(
        content=json.dumps({"call_id": call.object_id}),
        media_type="application/json",
    )


@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def deep_video_status(call_id: str):
    from fastapi import Response
    import json
    import base64

    try:
        function_call = modal.FunctionCall.from_id(call_id)
        result = function_call.get(timeout=0)
    except modal.exception.OutputExpiredError:
        return Response(content='{"status":"expired"}', status_code=410, media_type="application/json")
    except TimeoutError:
        return Response(content='{"status":"pending"}', media_type="application/json")
    except Exception as e:
        return Response(
            content=json.dumps({"status": "error", "error": str(e)[:300]}),
            status_code=500,
            media_type="application/json",
        )

    b64_video = base64.b64encode(result).decode("utf-8")
    return Response(
        content=json.dumps({"status": "done", "video_base64": b64_video}),
        media_type="application/json",
    )
