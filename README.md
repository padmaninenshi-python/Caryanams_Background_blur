# Remove Background & Number Plate Tool

Standalone Flask app to:
- ✅ Remove background from car images (uses rembg AI or OpenCV fallback)
- ✅ Detect & hide number plates (Caryanams badge / blur / black / white)
- ✅ Apply showroom backgrounds
- ✅ Upload multiple images
- ✅ Download processed results

## Setup

```bash
# 1. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python app.py
```

Open browser at: http://localhost:5000

## Usage

1. **Upload** — drag & drop or click to upload car photos (JPG/PNG/WEBP)
2. **Process** — click "🚀 Process" on individual images OR "Process All" for batch
   - Automatically detects & hides number plate
   - Removes background
   - Applies showroom background
3. **Manual Plate** — click "🎯 Manual Plate" to draw a rectangle over the plate yourself
4. **BG Only** — click "🖼 BG Only" to only remove background (no plate removal)
5. **Download** — click "⬇ Download" on any image or "Download All" for bulk

## Options

- **Quality**: draft (fast) / standard / high / ultra
- **Plate Mode**: Caryanams Badge / Blur / Black Block / White Block  
- **Image Type**: Exterior (removes BG) / Interior (keeps BG) / Plate Only

## Optional: Showroom Background

Place a file called `ba1_studio.jpg` in the `static/custom_bgs/` folder.
This will be used as the showroom background. Without it, a white background is used.

## Deploying on Render

1. Push this folder to a GitHub repo and create a new **Web Service** on Render
   pointing at it.
2. **Build Command:**
   ```
   pip install -r requirements.txt
   ```
3. **Start Command:** leave it blank — Render will auto-detect the `Procfile`
   (`web: gunicorn app:app ...`). If you want to set it manually instead:
   ```
   gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --worker-class gthread --timeout 300
   ```
4. **Do not** set a Start Command of `python app.py` on Render — that runs
   Flask's single-request dev server, which is why one image being processed
   used to block every other request ("ek process chal raha to dusra nahi ho
   raha"). `gunicorn` with `--threads 4` fixes this.
5. Render's free plan has limited CPU/RAM. AI background removal is CPU-heavy,
   so the first request after a deploy/restart (cold start) will still be
   slower than later requests. If requests still time out / show "Bad
   Gateway" under load, try:
   - Selecting **Standard** or **Draft** quality instead of High/Ultra (Ultra
     uses the heaviest model and is the slowest).
   - Upgrading to a paid Render instance with more CPU/RAM.

> ⚠️ Render's free filesystem is **ephemeral** — uploaded/processed images in
> `static/uploads/` and `static/processed/`, and the SQLite DB in `instance/`,
> are wiped on every redeploy/restart. For persistent storage across deploys,
> attach a Render **Persistent Disk** mounted at this app's root, or move to
> S3-compatible object storage + a managed database.

## Notes

- `rembg` is the AI background removal engine. `draft` and `standard` quality
  use the small, fast `u2netp` model (~5MB); `high`/`ultra` use the larger
  `isnet-general-use` model (~170MB, downloaded on first use of that quality).
- Falls back to OpenCV GrabCut if rembg is unavailable.
- All uploaded and processed images are stored in `static/uploads/` and `static/processed/`.
