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

## Notes

- `rembg` is the AI background removal engine (requires ~150MB model download on first run)
- Falls back to OpenCV GrabCut if rembg is unavailable
- All uploaded and processed images are stored in `static/uploads/` and `static/processed/`
