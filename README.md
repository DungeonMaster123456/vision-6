# Vision 6 — Image + Video Generation (Pollinations.ai)

Free, no local GPU needed, no huge downloads. Uses Pollinations.ai:
- **Images:** Flux model via Pollinations — no API key required
- **Video:** Pollinations video API — requires one free API key (no credit card)

---

## 1. One-time setup

You already have Python + venv set up from before. If starting fresh:

```bash
mkdir -p ~/vision6/backend
cd ~/vision6/backend
python3.11 -m venv venv
source venv/bin/activate
```

Install the (much smaller) dependencies:
```bash
pip install -r requirements.txt
```
This installs fastapi, uvicorn, and httpx — takes seconds, no PyTorch, no multi-GB downloads.

### Get a free video API key (skip this if you only want images)
1. Go to https://enter.pollinations.ai/keys
2. Sign up (free, no credit card)
3. Create a key, copy it
4. In your terminal:
   ```bash
   export POLLINATIONS_API_KEY=your_key_here
   ```
   (Run this in the same terminal window before starting the server — you'll need to re-run it each new terminal session, or add it to ~/.zshrc to make it permanent.)

To make it permanent:
```bash
echo 'export POLLINATIONS_API_KEY=your_key_here' >> ~/.zshrc
source ~/.zshrc
```

---

## 2. Run the backend

```bash
cd ~/vision6/backend
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000
```

Starts instantly (no model loading). You'll see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## 3. Open the frontend

Double-click `index.html` to open it in your browser. Make sure the backend is running first.

- **Image mode**: takes ~5-15 seconds, uses the Flux model for good quality.
- **Video mode**: takes ~1-3 minutes depending on length. Requires the API key from step 1.

Generated files also get saved in `~/vision6/backend/outputs/`.

---

## Notes on quality
- Images use model=flux explicitly — this is a real, good diffusion model, not a low-quality fallback.
- If an image still looks off, it's usually the prompt — be specific (subject, style, lighting, composition) rather than vague.
- nologo=true is set so you don't get a watermark.
- Each generation uses a random seed, so re-running the same prompt gives different results — that's expected, not a bug.

## Deploying (optional)
Since there's no local GPU dependency anymore, this backend is light enough to actually deploy for free on Render:
1. Push the backend/ folder to a GitHub repo.
2. On Render, create a new Web Service from that repo.
3. Build command: pip install -r requirements.txt
4. Start command: uvicorn main:app --host 0.0.0.0 --port $PORT
5. Add POLLINATIONS_API_KEY as an environment variable in Render's dashboard (for video).
6. Update BACKEND_URL in index.html to your Render URL instead of localhost:8000.
7. Push index.html to GitHub Pages or Render static hosting.

This gives you a fully hosted, always-on version reachable from anywhere — no need to keep your Mac Mini running.
