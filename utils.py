"""
Caryanams Studio — Background Removal Utilities
Fixed:
  1. rembg session lazy-loaded with full try/except isolation
  2. numpy/onnxruntime version guard — catches ImportError + AttributeError
  3. Pre-warm thread wrapped in try/except so a crash never kills the server
"""

import os
import io
import math
import base64
import threading
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageEnhance

try:
    import cv2 as cv2
except ImportError:
    cv2 = None


# ─── Static Background Path ───────────────────────────────────────────────────
# Always resolve from this file's location to avoid CWD-dependent failures
STATIC_BG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'static', 'custom_bgs', 'caryanams_studio_bg.jpg')

BACKGROUNDS = {
    'studio': [
        {'id': 'studio_white',    'name': 'Pure White',     'color': '#FFFFFF'},
        {'id': 'studio_grey',     'name': 'Studio Grey',    'color': '#E8E8E8'},
        {'id': 'studio_black',    'name': 'Midnight Black', 'color': '#1A1A1A'},
        {'id': 'studio_blue',     'name': 'Steel Blue',     'color': '#1E3A5F'},
        {'id': 'studio_gradient', 'name': 'Gradient Fog',   'color': '#C8D2E6'},
    ],
    'outdoor': [
        {'id': 'outdoor_road',     'name': 'Open Road',      'color': '#4A7C59'},
        {'id': 'outdoor_mountain', 'name': 'Mountain Pass',  'color': '#5B7FA6'},
        {'id': 'outdoor_sunset',   'name': 'Golden Sunset',  'color': '#FF6B35'},
        {'id': 'outdoor_city',     'name': 'City Night',     'color': '#2C1654'},
        {'id': 'outdoor_showroom', 'name': 'Showroom Floor', 'color': '#C0C0C0'},
    ]
}

SWATCHES = [
    '#FFFFFF', '#E8E8E8', '#1A1A1A', '#1E3A5F', '#C8D2E6', '#4A7C59',
    '#5B7FA6', '#FF6B35', '#2C1654', '#C0C0C0', '#FF0000', '#00FF00',
    '#0000FF', '#FFD700', '#FF1493', '#00CED1', '#8B4513', '#FF4500'
]

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}  # .webp converted on upload

# ─── rembg Session Cache ──────────────────────────────────────────────────────
_rembg_sessions = {}
_session_lock   = threading.Lock()

# FIX: Track whether rembg/onnxruntime is available at all.
# numpy 2.x breaks onnxruntime compiled against numpy 1.x.
# We detect this ONCE on import so we never retry a known-broken install.
_REMBG_AVAILABLE = None


def _check_rembg_available():
    """
    Returns True only if rembg + onnxruntime can actually be imported.
    Catches numpy ABI mismatch (AttributeError on np.bool / np.int etc.)
    and any other ImportError transparently.
    """
    global _REMBG_AVAILABLE
    if _REMBG_AVAILABLE is not None:
        return _REMBG_AVAILABLE
    try:
        # FIX: onnxruntime compiled for numpy 1.x raises AttributeError when
        # numpy 2.x is installed because np.bool, np.int, etc. were removed.
        # Importing onnxruntime here surfaces that error early.
        import onnxruntime  # noqa: F401
        import rembg        # noqa: F401
        _REMBG_AVAILABLE = True
    except (ImportError, AttributeError, Exception) as e:
        print(f'[rembg] NOT available — {type(e).__name__}: {e}')
        print('[rembg] Background removal will fall back to OpenCV GrabCut.')
        _REMBG_AVAILABLE = False
    return _REMBG_AVAILABLE


def get_rembg_session(model='isnet-general-use'):
    """
    Returns a cached rembg session or None if rembg/onnxruntime is broken.
    Safe to call repeatedly — failures are cached so we never retry endlessly.
    """
    if not _check_rembg_available():
        return None

    with _session_lock:
        if model not in _rembg_sessions:
            try:
                from rembg import new_session
                _rembg_sessions[model] = new_session(model)
                print(f'[rembg] session ready: {model}')
            except Exception as e:
                # Cache None so we don't retry on every request
                _rembg_sessions[model] = None
                print(f'[rembg] session failed ({model}): {type(e).__name__}: {e}')
        return _rembg_sessions[model]


def _prewarm():
    """Pre-warm u2net in background. Wrapped so crashes never kill Flask."""
    try:
        get_rembg_session('isnet-general-use')
    except Exception as e:
        print(f'[rembg] pre-warm error (non-fatal): {e}')


# FIX: daemon=True ensures this thread won't block server shutdown
threading.Thread(target=_prewarm, daemon=True).start()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def ensure_static_bg():
    if os.path.exists(STATIC_BG_PATH):
        return True
    os.makedirs(os.path.dirname(STATIC_BG_PATH), exist_ok=True)
    print(f'[static_bg] WARNING: {STATIC_BG_PATH} not found.')
    return False


# ─── Mask Quality Validation ──────────────────────────────────────────────────

def validate_mask_quality(img_rgba, min_fg_ratio=0.10, max_fg_ratio=0.92):
    if img_rgba.mode != 'RGBA':
        return True, 1.0
    alpha = np.array(img_rgba.split()[3])
    fg_pixels = np.sum(alpha > 128)
    fg_ratio  = fg_pixels / alpha.size
    return (min_fg_ratio < fg_ratio < max_fg_ratio), float(fg_ratio)


# ─── BG Removal: rembg ───────────────────────────────────────────────────────

def _resize_for_removal(filepath, max_side=1024):
    img = Image.open(filepath).convert('RGB')
    orig_size = img.size
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, 'PNG')
    return buf.getvalue(), orig_size


def apply_60_percent_background_blur(original_image_path, masked_image_rgba):
    """
    Takes original image and a masked car image (RGBA with transparent background).
    Returns new image with:
    - Car fully sharp (original clarity, EXACT same position)
    - Real background blurred
    - Original composition preserved (no car repositioning)
    
    This creates a professional focus/depth-of-field effect where the car
    stands out with a beautifully blurred background while maintaining
    the exact original composition.
    
    Process:
    1. Load original image (full color, preserves composition)
    2. Use mask to identify car vs background pixels
    3. Apply Gaussian blur to background ONLY
    4. Keep car area from original (sharp)
    5. Return as RGB image (no transparency)
    """
    try:
        import cv2
        
        # Load images
        original = Image.open(original_image_path).convert('RGB')
        masked = masked_image_rgba.convert('RGBA')
        
        # Ensure same size
        if original.size != masked.size:
            original = original.resize(masked.size, Image.LANCZOS)
        
        # Convert to numpy arrays
        orig_cv = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)
        
        # Get alpha channel from masked image (car = 255, background = 0)
        alpha_mask = np.array(masked.split()[3])
        
        # Create binary mask: car = 255, background = 0
        car_mask = (alpha_mask > 128).astype(np.uint8) * 255
        
        # ── Strong, size-adaptive background blur (~80% visible blur) ───────
        # Sigma scales with image size so the blur is clearly visible on
        # both small and large photos. (0,0) kernel size lets OpenCV pick
        # the correct kernel for the given sigma.
        h_img, w_img = orig_cv.shape[:2]
        blur_sigma = max(18, int(min(h_img, w_img) * 0.035))
        blurred_bg = cv2.GaussianBlur(orig_cv, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
        # Second pass for an extra-soft, strongly blurred background
        blurred_bg = cv2.GaussianBlur(blurred_bg, (0, 0), sigmaX=blur_sigma * 0.6, sigmaY=blur_sigma * 0.6)
        
        # Create smooth mask for blending (feathered edges)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        feathered_mask = cv2.morphologyEx(car_mask, cv2.MORPH_CLOSE, k)
        feathered_mask = cv2.GaussianBlur(feathered_mask, (15, 15), 0)
        feathered_mask_3ch = cv2.cvtColor(feathered_mask, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
        
        # IMPORTANT: Keep car pixels from ORIGINAL image (sharp)
        # Only blend background (blurred)
        # This preserves exact original car position and orientation
        final = (orig_cv * feathered_mask_3ch + blurred_bg * (1.0 - feathered_mask_3ch)).astype(np.uint8)
        
        # Convert back to RGB PIL Image
        result_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb, 'RGB')
        
    except Exception as e:
        print(f'[apply_60_percent_background_blur] error: {e}')
        # Fallback: return original
        try:
            return Image.open(original_image_path).convert('RGB')
        except:
            return masked_image_rgba.convert('RGB')


def get_fallback_blur_mask(filepath):
    """
    Last-resort foreground mask for the background-blur feature.

    Used when remove_bg_ai() returns None for BOTH rembg and the full
    OpenCV grabcut+cleanup pipeline (this happens often on Render's free
    tier, where rembg/onnxruntime fails to import due to the numpy ABI
    mismatch, AND the strict validate_mask_quality() check rejects the
    plain grabcut result).

    Without this, the old fallback was `Image.open(...).convert('RGBA')`,
    which has alpha=255 everywhere -> the ENTIRE photo is treated as "car"
    -> apply_60_percent_background_blur() blends 100% sharp pixels and the
    background never gets blurred at all.

    This function runs a quick, lightly-cleaned GrabCut (no person/edge
    cleanup, no quality validation) and ALWAYS returns a usable RGBA mask —
    if GrabCut itself produces something unusable, it falls back to a
    centered rectangle so the blur still has an effect.
    """
    try:
        import cv2
        img_bgr = cv2.imread(filepath)
        if img_bgr is None:
            return Image.open(filepath).convert('RGBA')

        h, w = img_bgr.shape[:2]
        orig_size = (w, h)

        work  = img_bgr
        scale = 1.0
        if max(h, w) > 1000:
            scale = 1000 / max(h, w)
            work  = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
        wh, ww = work.shape[:2]

        margin_x = max(int(ww * 0.08), 5)
        margin_y = max(int(wh * 0.06), 5)
        rect = (margin_x, margin_y, ww - 2 * margin_x, wh - 2 * margin_y)

        mask = np.zeros((wh, ww), np.uint8)
        bgd  = np.zeros((1, 65), np.float64)
        fgd  = np.zeros((1, 65), np.float64)
        cv2.grabCut(work, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)

        fg_mask = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, k, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  k, iterations=1)

        # If grabcut produced almost nothing or almost everything,
        # use a simple centered rectangle so blur still applies.
        fg_ratio = float(fg_mask.mean()) / 255.0
        if fg_ratio < 0.05 or fg_ratio > 0.85:
            fg_mask[:] = 0
            cx0, cx1 = int(ww * 0.12), int(ww * 0.88)
            cy0, cy1 = int(wh * 0.15), int(wh * 0.95)
            fg_mask[cy0:cy1, cx0:cx1] = 255

        if scale != 1.0:
            fg_mask = cv2.resize(fg_mask, orig_size, interpolation=cv2.INTER_LINEAR)

        alpha = cv2.GaussianBlur(fg_mask, (9, 9), 0)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb).convert('RGBA')
        pil_img.putalpha(Image.fromarray(alpha).convert('L'))
        return pil_img

    except Exception as e:
        print(f'[get_fallback_blur_mask] error: {e}')
        try:
            return Image.open(filepath).convert('RGBA')
        except Exception:
            return None


def remove_bg_rembg(filepath, quality='standard'):
    """
    Remove background with rembg. Returns (PIL.Image RGBA, method_str) or
    (None, None) if rembg is unavailable or fails.
    """
    # FIX: gate on availability check — no import attempted if broken
    if not _check_rembg_available():
        return None, None
    try:
        from rembg import remove
        model_map = {
            'draft':    'u2netp',
            'standard': 'isnet-general-use',
            'high':     'isnet-general-use',
            'ultra':    'isnet-general-use',
        }
        model    = model_map.get(quality, 'isnet-general-use')
        session  = get_rembg_session(model)
        # FIX MemoryError: reduce max_side to avoid OOM on large images.
        # alpha_matting needs ~1.86 GiB for 5000px images — only enable for 'ultra'.
        if quality == 'draft':
            max_side = 640
        elif quality == 'ultra':
            max_side = 1024
        else:  # standard / high
            max_side = 800
        img_bytes, orig_size = _resize_for_removal(filepath, max_side)
        # FIX MemoryError: alpha_matting=True on high-res images causes huge allocations.
        # Only enable for 'ultra' quality; standard/high use clean rembg mask (still good).
        alpha_matting = (quality == 'ultra')

        if session:
            result = remove(
                img_bytes, session=session,
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=200,
                alpha_matting_background_threshold=20,
                alpha_matting_erode_size=3
            )
        else:
            result = remove(img_bytes, alpha_matting=alpha_matting)

        img = Image.open(io.BytesIO(result)).convert('RGBA')
        if img.size != orig_size:
            img = img.resize(orig_size, Image.LANCZOS)
        
        # Restore original pixels in tyre zone — rembg corrupts dark tyre colors
        img = _restore_tyre_pixels_from_original(img, filepath, orig_size)

        # Keep only the largest foreground object (removes second car, reflections etc.)
        img = keep_largest_component(img)
        img = remove_persons_and_objects(img)  # Remove humans, persons, side objects
        img = remove_connected_persons(img)    # Remove persons merged/touching the car
        img = trim_side_cars(img)
        img = trim_top_objects(img)   # Remove buildings/signs above car
        img = remove_thin_protrusions(img)  # FIX: Remove wipers/antennas/rods sticking out
        img = restore_tyres(img)      # FIX: re-enabled — now safely fills rear-tyre gaps
        img = restore_windshield(img)  # Fix transparent windshield/windows
        img = clean_edges(img)        # Remove grey border halo

        is_valid, _ = validate_mask_quality(img)
        if not is_valid:
            return None, None
        return img, 'rembg_' + quality

    except ImportError:
        return None, None
    except Exception as e:
        print(f'[rembg] removal error: {type(e).__name__}: {e}')
        return None, None


def keep_largest_component(img_rgba):
    """
    Keep only the main car — removes second cars, humans, ladders, background objects.

    Strategy:
    1. Find all connected components in the alpha mask
    2. Score each by: area × center-proximity × aspect-ratio-car-likeness
    3. Keep the winner (the main car)
    4. Erase every other component completely (no mercy for humans/ladders)
    5. After erasing, re-check for residual blobs attached to car edges and trim
    """
    try:
        import cv2
        alpha = np.array(img_rgba.split()[3])
        H, W  = alpha.shape
        img_cx, img_cy = W / 2.0, H / 2.0

        binary = (alpha > 100).astype(np.uint8) * 255

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8)

        if num_labels <= 2:
            return img_rgba

        # ── Score every foreground component ────────────────────────────────
        best_label  = -1
        best_score  = -1
        best_bbox   = None

        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area < 300:
                continue
            lx  = stats[lbl, cv2.CC_STAT_LEFT]
            ly  = stats[lbl, cv2.CC_STAT_TOP]
            lw  = stats[lbl, cv2.CC_STAT_WIDTH]
            lh  = stats[lbl, cv2.CC_STAT_HEIGHT]
            cx_l, cy_l = centroids[lbl]

            # Center proximity score (0=far, 1=center)
            dist = ((cx_l - img_cx)**2 + (cy_l - img_cy)**2) ** 0.5
            max_dist = ((img_cx)**2 + (img_cy)**2) ** 0.5
            norm_dist = dist / (max_dist + 1e-6)
            center_score = 1.0 - 0.65 * norm_dist

            # Aspect ratio: cars are wide, humans/ladders are tall and narrow
            aspect = lw / max(lh, 1)
            if aspect < 0.35:
                # Very tall narrow object (human, ladder, pole) — heavily penalize
                aspect_score = 0.10
            elif aspect < 0.55:
                # Person-like narrow — strong penalty
                aspect_score = 0.25
            elif aspect < 0.70:
                # Somewhat narrow — moderate penalty
                aspect_score = 0.50
            elif aspect > 0.70:
                # Wide-ish — car-like
                aspect_score = 1.0
            else:
                aspect_score = 0.7

            # Bottom anchor: cars sit at the image bottom, humans often float
            bottom_y = ly + lh
            bottom_ratio = bottom_y / H
            bottom_score = 0.6 + 0.4 * bottom_ratio  # reward bottom-anchored objects

            # Area score (normalized — largest gets best score)
            area_score = area / (W * H)

            score = area_score * center_score * aspect_score * bottom_score * 1e6

            if score > best_score:
                best_score = score
                best_label = lbl
                best_bbox  = (lx, ly, lw, lh)

        if best_label < 0:
            return img_rgba

        # ── Build clean mask: winning component + side mirror blobs ──────────
        keep_mask = (labels == best_label).astype(np.uint8) * 255

        # Generous horizontal margin to preserve side mirrors
        bx, by, bw, bh = best_bbox
        margin_x = max(10, int(bw * 0.12))  # FIX: 0.03→0.12 for side mirrors
        margin_y = max(4, int(bh * 0.03))
        region = np.zeros_like(keep_mask)
        region[
            max(0, by - margin_y) : min(H, by + bh + margin_y),
            max(0, bx - margin_x) : min(W, bx + bw + margin_x)
        ] = 255
        keep_mask = cv2.bitwise_and(keep_mask, region)

        # ── Also keep mirror blobs: small, at left/right of car, body height ─
        main_area = stats[best_label, cv2.CC_STAT_AREA]
        for lbl in range(1, num_labels):
            if lbl == best_label:
                continue
            a = stats[lbl, cv2.CC_STAT_AREA]
            if a < 500 or a > main_area * 0.15:  # only significant mirror-sized blobs
                continue
            lx2 = stats[lbl, cv2.CC_STAT_LEFT]
            ly2 = stats[lbl, cv2.CC_STAT_TOP]
            lw2 = stats[lbl, cv2.CC_STAT_WIDTH]
            lh2 = stats[lbl, cv2.CC_STAT_HEIGHT]
            cx2, cy2 = centroids[lbl]
            # Must be horizontally near car left or right edge (within 20% image width)
            near_left  = cx2 < (bx + bw * 0.25) and (bx - (lx2 + lw2)) < W * 0.20
            near_right = cx2 > (bx + bw * 0.75) and (lx2 - (bx + bw)) < W * 0.20
            in_body    = 0.15 < cy2 / H < 0.90
            not_tall   = lh2 / H < 0.35
            if (near_left or near_right) and in_body and not_tall:
                keep_mask[labels == lbl] = 255

        # Small dilation to recover car edge pixels (2px max)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        keep_mask = cv2.dilate(keep_mask, k, iterations=1)

        new_alpha = np.zeros_like(alpha)
        new_alpha[keep_mask > 0] = alpha[keep_mask > 0]

        # Feather edges but keep interior hard
        feathered = cv2.GaussianBlur(new_alpha, (3, 3), 0.8)
        interior  = (new_alpha > 230)
        feathered[interior] = alpha[interior]

        r, g, b, _ = img_rgba.split()
        return Image.merge('RGBA', (r, g, b, Image.fromarray(feathered)))

    except Exception as e:
        print(f'[keep_largest_component] error: {e}')
        return img_rgba


def remove_persons_and_objects(img_rgba):
    """
    Aggressively remove people, persons, and any non-car objects from the masked image.

    This runs AFTER BG removal. At that point, only foreground blobs remain.
    Strategy:
    1. Find all connected components in the alpha mask
    2. For each blob: classify as CAR or NON-CAR using multiple signals:
       - Aspect ratio (cars are wide, persons are tall/narrow)
       - Position (persons often stand beside/behind car)
       - Blob height vs image height (person blobs are tall fraction of image)
       - Relative size vs main blob (persons are much smaller than car)
    3. Erase ALL non-car blobs unconditionally
    4. For blobs that are partially connected to the car:
       - Detect narrow "neck" connecting person to car edge
       - Cut the connection and erase the person part
    5. Remove residual thin protrusions on car sides (arms, legs sticking out)
    """
    try:
        import cv2
        alpha = np.array(img_rgba.split()[3])
        H, W = alpha.shape

        # ── Step 1: Find all blobs ──────────────────────────────────────────
        binary = (alpha > 80).astype(np.uint8) * 255
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8)

        if num_labels <= 1:
            return img_rgba

        # Collect all blob info
        blobs = []
        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area < 100:
                continue
            lx = stats[lbl, cv2.CC_STAT_LEFT]
            ly = stats[lbl, cv2.CC_STAT_TOP]
            lw = stats[lbl, cv2.CC_STAT_WIDTH]
            lh = stats[lbl, cv2.CC_STAT_HEIGHT]
            cx, cy = centroids[lbl]
            aspect = lw / max(lh, 1)
            blobs.append({'lbl': lbl, 'area': area, 'lx': lx, 'ly': ly,
                          'lw': lw, 'lh': lh, 'cx': cx, 'cy': cy, 'aspect': aspect})

        if not blobs:
            return img_rgba

        # ── Step 2: Find the MAIN CAR blob (largest area with car-like shape) ─
        def car_score(b):
            area_score = b['area']
            # Wide = car-like
            if b['aspect'] >= 0.9:
                shape_score = 2.5
            elif b['aspect'] >= 0.6:
                shape_score = 1.5
            elif b['aspect'] >= 0.35:
                shape_score = 0.6
            else:
                shape_score = 0.1  # Very tall/narrow = person
            # Center proximity bonus
            dist = ((b['cx'] - W / 2) ** 2 + (b['cy'] - H / 2) ** 2) ** 0.5
            center_bonus = 1.0 - 0.5 * (dist / (((W/2)**2 + (H/2)**2)**0.5 + 1e-6))
            return area_score * shape_score * center_bonus

        main_blob = max(blobs, key=car_score)
        main_lbl  = main_blob['lbl']
        main_area = main_blob['area']

        # ── Step 3: Classify each blob as KEEP or ERASE ──────────────────────
        erase_labels = set()
        for b in blobs:
            if b['lbl'] == main_lbl:
                continue

            # Size relative to main car
            size_ratio = b['area'] / max(main_area, 1)

            # If blob is very small (noise/reflection) — erase
            if b['area'] < 500:
                erase_labels.add(b['lbl'])
                continue

            # Person detection signals:
            is_person = False

            # Signal 1: Very tall and narrow (person standing) — aspect < 0.55
            if b['aspect'] < 0.55:
                is_person = True

            # Signal 2: Blob height is >25% of image height AND aspect < 0.8
            height_ratio = b['lh'] / H
            if height_ratio > 0.25 and b['aspect'] < 0.8:
                is_person = True

            # Signal 3: Blob is at left/right 20% of image AND not very wide
            left_edge  = b['lx'] < W * 0.20
            right_edge = (b['lx'] + b['lw']) > W * 0.80
            if (left_edge or right_edge) and b['aspect'] < 1.2:
                blob_cy_ratio = b['cy'] / H
                blob_h_ratio  = b['lh'] / H
                is_mirror_like = (
                    0.20 < blob_cy_ratio < 0.85  # vertically in car body zone
                    and blob_h_ratio < 0.35        # not a tall person-height object
                    and size_ratio < 0.18          # small relative to car
                    and b['aspect'] > 0.25         # not a vertical pole/line
                )
                if not is_mirror_like:
                    is_person = True

            # Signal 4: Much smaller than car AND aspect not car-like
            blob_cy_ratio_s4 = b['cy'] / H
            is_side_mirror_s4 = (
                (b['lx'] < W * 0.20 or (b['lx'] + b['lw']) > W * 0.80)
                and size_ratio < 0.18
                and 0.20 < blob_cy_ratio_s4 < 0.85
                and b['lh'] / H < 0.35
            )
            if size_ratio < 0.15 and b['aspect'] < 1.0 and not is_side_mirror_s4:
                is_person = True

            # Signal 5: Blob width is less than 20% of image width (thin object)
            width_ratio = b['lw'] / W
            is_side_mirror_s5 = (
                (b['lx'] < W * 0.20 or (b['lx'] + b['lw']) > W * 0.80)
                and size_ratio < 0.18
                and 0.20 < (b['cy'] / H) < 0.85
                and b['lh'] / H < 0.35
            )
            if width_ratio < 0.20 and b['aspect'] < 0.9 and not is_side_mirror_s5:
                is_person = True

            # Signal 6: Second car — separate blob that is also car-shaped but smaller
            # Keep second cars ONLY if they are very close in size to main car (>60%)
            # Mirror blobs are always small (<18%) so they won't hit this check
            if not is_person and size_ratio < 0.60:
                erase_labels.add(b['lbl'])
                continue

            if is_person:
                erase_labels.add(b['lbl'])

        # ── Step 4: Build clean alpha — erase all bad blobs ──────────────────
        if not erase_labels:
            # No separate blobs to erase — but still run protrusion trimmer
            pass
        else:
            new_alpha = alpha.copy()
            for erase_lbl in erase_labels:
                new_alpha[labels == erase_lbl] = 0
            r, g, b_ch, _ = img_rgba.split()
            img_rgba = Image.merge('RGBA', (r, g, b_ch, Image.fromarray(new_alpha)))

        # ── Step 5: Remove thin protrusions (person arms/legs connected to car) ─
        # After erasing separate blobs, trim narrow side protrusions from the
        # remaining main mask. A "protrusion" is a side appendage narrower than
        # 12% of the bounding box width.
        try:
            alpha2 = np.array(img_rgba.split()[3])
            binary2 = (alpha2 > 80).astype(np.uint8) * 255

            # Erode left+right sides ONLY to cut thin connections
            # Use a horizontal erosion to disconnect side protrusions
            k_thin = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
            eroded = cv2.erode(binary2, k_thin, iterations=3)

            # Find components in eroded — each should now be the car body only
            n2, labs2, stats2, _ = cv2.connectedComponentsWithStats(eroded, connectivity=8)
            if n2 > 1:
                # Keep only the largest component in eroded (car body)
                best2 = max(range(1, n2), key=lambda l: stats2[l, cv2.CC_STAT_AREA])
                car_body_mask = (labs2 == best2).astype(np.uint8) * 255

                # Dilate back to recover car edges
                k_recover = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                car_body_mask = cv2.dilate(car_body_mask, k_recover, iterations=4)

                # Intersect with original alpha to keep only car pixels
                new_alpha2 = np.minimum(alpha2, car_body_mask)

                # Only apply if we didn't accidentally remove too much of the car
                kept_ratio = new_alpha2.sum() / (alpha2.sum() + 1e-6)
                if kept_ratio > 0.70:  # Only apply if we kept >70% of original
                    r, g, b_ch, _ = img_rgba.split()
                    img_rgba = Image.merge('RGBA', (r, g, b_ch, Image.fromarray(new_alpha2)))
        except Exception as e_prot:
            print(f'[remove_persons] protrusion trim error: {e_prot}')

        return img_rgba

    except Exception as e:
        print(f'[remove_persons_and_objects] error: {e}')
        return img_rgba


def remove_connected_persons(img_rgba):
    """
    Detect and erase person-shaped blobs that are CONNECTED to the car body.

    rembg often merges a bystander/lady into the same alpha region as the car
    because they are touching. This function:
    1. Slices the alpha into vertical columns
    2. For each column, checks the HEIGHT of the foreground span
    3. Tall, narrow spans at the top of the mask = person head/body sticking up
    4. Cuts those spans by zeroing the alpha above the car roof line
    5. Also checks left/right edge columns for person-shaped protrusions:
       - A person standing beside the car shows as a tall thin alpha region
         at left/right edge that rises ABOVE the main car roof
    6. Uses horizontal erosion to break thin neck connections then re-dilates
    """
    try:
        import cv2
        alpha = np.array(img_rgba.split()[3], dtype=np.uint8)
        H, W = alpha.shape
        binary = (alpha > 80).astype(np.uint8)

        # ── Find car roof line (topmost dense row) ────────────────────────────
        row_sum = binary.sum(axis=1).astype(np.float32)
        if row_sum.max() == 0:
            return img_rgba

        # Peak density row (widest part of car)
        peak_row = int(np.argmax(row_sum))
        row_smooth = cv2.GaussianBlur(row_sum.reshape(-1, 1), (31, 1), 0)[:, 0]
        peak_density = float(row_smooth.max())

        # Car roof: scan upward from peak to find where density drops below 35%
        roof_row = 0
        for y in range(peak_row, -1, -1):
            if row_smooth[y] < peak_density * 0.35:
                roof_row = y + 1
                break

        # Car bottom: scan downward from peak
        bottom_row = H - 1
        for y in range(peak_row, H):
            if row_smooth[y] < peak_density * 0.10:
                bottom_row = y - 1
                break

        car_height = max(1, bottom_row - roof_row)
        roof_with_margin = max(0, roof_row - int(car_height * 0.05))

        # ── Column span analysis: find person protrusions above roof ─────────
        new_alpha = alpha.copy()

        col_fg_top = np.full(W, H, dtype=np.int32)   # topmost fg row per column
        col_fg_bot = np.full(W, -1, dtype=np.int32)  # bottommost fg row per column

        for x in range(W):
            col = binary[:, x]
            fg_rows = np.where(col > 0)[0]
            if len(fg_rows) > 0:
                col_fg_top[x] = int(fg_rows[0])
                col_fg_bot[x] = int(fg_rows[-1])

        # For each column: if the fg span starts WAY above the car roof line
        # AND the column is in the left/right 30% of the image, it's likely a person
        left_zone  = int(W * 0.30)
        right_zone = int(W * 0.70)

        for x in range(W):
            top = col_fg_top[x]
            bot = col_fg_bot[x]
            if bot < 0:
                continue

            span_height = bot - top
            # Is the topmost pixel above the car roof by more than 8% of car height?
            above_roof = roof_with_margin - top
            if above_roof <= int(car_height * 0.08):
                continue

            # Only target left/right edge columns (person standing beside car)
            in_side_zone = (x < left_zone) or (x > right_zone)
            if not in_side_zone:
                continue

            # Span should be tall (person-like) relative to car
            person_like = span_height > car_height * 0.30

            # Mirror protection: side mirrors appear as a short span slightly above roof
            # at the left/right edge. They are NOT tall — span_height < 20% of car_height
            # and they don't extend far above the roof (< 25% of car_height).
            is_mirror_protrusion = (
                span_height < car_height * 0.20
                and above_roof < int(car_height * 0.25)
            )
            if is_mirror_protrusion:
                continue  # Skip — this is likely a side mirror, not a person

            if person_like or above_roof > int(car_height * 0.15):
                # Zero out everything above the roof line in this column
                cut_y = max(0, roof_with_margin)
                new_alpha[:cut_y, x] = 0

        # ── Horizontal erosion pass to break thin neck connections ────────────
        # This handles when person's shoulder/neck is thinly connected to car roof
        binary2 = (new_alpha > 80).astype(np.uint8) * 255
        # Horizontal erosion to cut left/right thin connections
        k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
        eroded2 = cv2.erode(binary2, k_h, iterations=2)

        n3, labs3, stats3, _ = cv2.connectedComponentsWithStats(eroded2, connectivity=8)
        if n3 > 2:
            # Find main car blob (largest + most car-like)
            def _car_like(lbl):
                a = stats3[lbl, cv2.CC_STAT_AREA]
                lw = stats3[lbl, cv2.CC_STAT_WIDTH]
                lh = stats3[lbl, cv2.CC_STAT_HEIGHT]
                asp = lw / max(lh, 1)
                return a * (asp if asp > 0.5 else 0.2)

            best3 = max(range(1, n3), key=_car_like)
            car_mask3 = (labs3 == best3).astype(np.uint8) * 255

            # Recover car pixels with dilation
            k_rec = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            car_mask3 = cv2.dilate(car_mask3, k_rec, iterations=3)

            candidate = np.minimum(new_alpha, car_mask3)
            kept_ratio = float(candidate.sum()) / (new_alpha.sum() + 1e-6)
            if kept_ratio > 0.72:
                new_alpha = candidate

        r, g, b, _ = img_rgba.split()
        return Image.merge('RGBA', (r, g, b, Image.fromarray(new_alpha)))

    except Exception as e:
        print(f'[remove_connected_persons] error: {e}')
        return img_rgba


def trim_side_cars(img_rgba):
    """
    After BG removal, erase foreground pixels that belong to:
      - side cars clinging to left/right edges
      - dual-image layouts (two photos stitched side by side)
      - humans or objects at the image edges

    Works by:
    1. Detecting ZERO-density vertical gaps (dual image panels)
    2. Finding density valleys between cars
    3. Keeping only the dominant car column range
    """
    try:
        import cv2
        alpha = np.array(img_rgba.split()[3])
        H, W  = alpha.shape
        binary = (alpha > 100).astype(np.uint8)

        col_sum = binary.sum(axis=0).astype(np.float32)

        if col_sum.max() == 0:
            return img_rgba

        col_smooth = cv2.GaussianBlur(col_sum.reshape(1, -1), (1, 41), 0)[0]
        peak = col_smooth.max()

        # ── Center of mass (x-axis) ──────────────────────────────────────────
        total_fg = col_sum.sum()
        if total_fg > 0:
            x_coords = np.arange(W, dtype=np.float32)
            fg_cx = float(np.dot(col_sum, x_coords) / total_fg)
        else:
            fg_cx = W / 2.0

        # ── Strategy 0: Hard gap detection for dual-image panels ─────────────
        # If there is a near-zero-density vertical strip it is a two-image seam.
        zero_cols = np.where(col_smooth < peak * 0.08)[0]
        if len(zero_cols) > 0:
            # Find contiguous zero-gap regions
            gaps = []
            start_gap = None
            for x in range(W):
                if col_smooth[x] < peak * 0.08:
                    if start_gap is None:
                        start_gap = x
                else:
                    if start_gap is not None:
                        gaps.append((start_gap, x - 1, x - start_gap))
                        start_gap = None
            if start_gap is not None:
                gaps.append((start_gap, W - 1, W - start_gap))

            wide_gaps = [(s, e, w) for s, e, w in gaps if w > W * 0.015]
            if wide_gaps:
                best_gap = max(wide_gaps, key=lambda g: g[2])
                gs, ge, _ = best_gap
                gap_center = (gs + ge) / 2.0
                new_alpha = alpha.copy()
                if fg_cx < gap_center:
                    # Car is left of gap — erase right of gap
                    new_alpha[:, gs:] = 0
                else:
                    # Car is right of gap — erase left of gap
                    new_alpha[:, :ge + 1] = 0
                r, g, b, _ = img_rgba.split()
                result = Image.merge("RGBA", (r, g, b, Image.fromarray(new_alpha)))
                # Recurse once to clean up remaining stray blobs
                return result

        # ── Strategy 1: Valley detection ─────────────────────────────────────
        # Lower threshold to avoid cutting side mirrors (less dense than car body)
        valley_threshold = peak * 0.12

        left_boundary = 0
        for x in range(int(fg_cx), -1, -1):
            if col_smooth[x] < valley_threshold:
                # Before accepting this as a cut point, check if it's a mirror gap:
                # A mirror gap is a valley in the outer 20% of the image with a
                # small foreground blob beyond it (the mirror). Don't cut there.
                if x < W * 0.30:  # FIX: 0.20→0.30 wider mirror zone
                    # Check if there's any foreground to the left of this valley
                    beyond = col_smooth[:x]
                    if len(beyond) > 0 and beyond.max() > peak * 0.02:  # FIX: 0.04→0.02
                        # Small blob beyond — likely side mirror, keep it
                        left_boundary = 0
                        break
                left_boundary = x + 1
                break

        right_boundary = W - 1
        for x in range(int(fg_cx), W):
            if col_smooth[x] < valley_threshold:
                if x > W * 0.70:  # FIX: 0.80→0.70 wider mirror zone
                    beyond = col_smooth[x+1:]
                    if len(beyond) > 0 and beyond.max() > peak * 0.02:  # FIX: 0.04→0.02
                        right_boundary = W - 1
                        break
                right_boundary = x - 1
                break

        # ── Strategy 2: Density threshold fallback ────────────────────────────
        # Lower from 0.20 to 0.06 to preserve side mirrors at car edges
        threshold_cols = np.where(col_smooth > peak * 0.06)[0]
        if len(threshold_cols) > 0:
            thresh_left  = int(threshold_cols[0])
            thresh_right = int(threshold_cols[-1])
        else:
            thresh_left  = 0
            thresh_right = W - 1

        left_col  = max(left_boundary, thresh_left)
        right_col = min(right_boundary, thresh_right)

        if left_col >= right_col:
            left_col  = thresh_left
            right_col = thresh_right

        margin = int((right_col - left_col) * 0.015)
        left_col  = max(0, left_col  - margin)
        right_col = min(W - 1, right_col + margin)

        cut_left  = left_col
        cut_right = W - 1 - right_col
        if cut_left < int(W * 0.015) and cut_right < int(W * 0.015):
            return img_rgba

        new_alpha = alpha.copy()
        if cut_left > 0:
            new_alpha[:, :left_col] = 0
        if cut_right > 0:
            new_alpha[:, right_col + 1:] = 0

        r, g, b, _ = img_rgba.split()
        return Image.merge("RGBA", (r, g, b, Image.fromarray(new_alpha)))

    except Exception as e:
        print(f"[trim_side_cars] error: {e}")
        return img_rgba


def remove_thin_protrusions(img_rgba, min_thickness=6, kept_ratio_threshold=0.72):
    """
    Remove thin diagonal/side protrusions from car mask:
    wipers, antennas, rods, sticks — anything narrow sticking out of the car body.

    Strategy (multi-pass morphological skeleton approach):
    1. Erode the alpha mask with a LARGE circular kernel to dissolve thin appendages
       (wipers ~3-5px wide get wiped; car body ~100px wide survives)
    2. Dilate back by the same amount to recover the true car body shape
    3. Intersect with original alpha → only car body pixels remain
    4. Safety check: if >28% of pixels removed, skip (car too thin / unusual angle)

    Also runs a secondary skeleton-based pass using cross-shaped erosion to catch
    diagonal protrusions that survive circular erosion (e.g. 45° antenna).
    """
    try:
        import cv2
        alpha = np.array(img_rgba.split()[3], dtype=np.uint8)
        H, W  = alpha.shape

        binary = (alpha > 80).astype(np.uint8) * 255

        # ── Estimate car body thickness (minimum dimension of bounding box) ──
        rows_fg = np.where(np.any(binary > 0, axis=1))[0]
        cols_fg = np.where(np.any(binary > 0, axis=0))[0]
        if len(rows_fg) < 10 or len(cols_fg) < 10:
            return img_rgba

        car_h = int(rows_fg[-1]) - int(rows_fg[0]) + 1
        car_w = int(cols_fg[-1]) - int(cols_fg[0]) + 1

        # Erosion radius: 8% of the smaller car dimension but at least 8px, max 22px
        # This dissolves objects thinner than ~16px (wipers, antennas) while
        # preserving the car body which is much wider.
        radius = max(8, min(22, int(min(car_h, car_w) * 0.08)))

        k_circ = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))

        # Pass 1: erode → dissolve thin protrusions
        eroded1 = cv2.erode(binary, k_circ, iterations=1)

        # If erosion wiped everything → car too thin for this method, skip
        if eroded1.sum() == 0:
            return img_rgba

        # Dilate back to recover car body (don't go larger than original)
        recovered1 = cv2.dilate(eroded1, k_circ, iterations=1)

        # Intersect with original
        clean1 = np.minimum(alpha, recovered1)

        ratio1 = clean1.sum() / (alpha.sum() + 1e-6)
        if ratio1 < kept_ratio_threshold:
            # Too aggressive — try a smaller radius (half)
            radius2 = max(5, radius // 2)
            k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius2 * 2 + 1, radius2 * 2 + 1))
            eroded2  = cv2.erode(binary, k2, iterations=1)
            if eroded2.sum() == 0:
                return img_rgba
            recovered2 = cv2.dilate(eroded2, k2, iterations=1)
            clean2 = np.minimum(alpha, recovered2)
            ratio2 = clean2.sum() / (alpha.sum() + 1e-6)
            if ratio2 < kept_ratio_threshold:
                return img_rgba  # Still too aggressive — skip entirely
            clean_alpha = clean2
        else:
            clean_alpha = clean1

        # ── Pass 2: diagonal cross erosion to catch 45° antennas ─────────────
        # A diagonal wiper survives circular erosion if it's slightly thicker
        # than radius. A cross-shaped erosion disconnects it.
        k_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (radius + 1, radius + 1))
        eroded_cross = cv2.erode((clean_alpha > 80).astype(np.uint8) * 255,
                                  k_cross, iterations=1)
        if eroded_cross.sum() > 0:
            # Find components: keep only car body (largest)
            n_c, labs_c, stats_c, _ = cv2.connectedComponentsWithStats(
                eroded_cross, connectivity=8)
            if n_c > 2:
                best_c = max(range(1, n_c), key=lambda l: stats_c[l, cv2.CC_STAT_AREA])
                car_mask_c = (labs_c == best_c).astype(np.uint8) * 255
                # Dilate back
                car_mask_c = cv2.dilate(car_mask_c, k_circ, iterations=1)
                candidate_c = np.minimum(clean_alpha, car_mask_c)
                ratio_c = candidate_c.sum() / (clean_alpha.sum() + 1e-6)
                if ratio_c > kept_ratio_threshold:
                    clean_alpha = candidate_c

        # ── Smooth edges of the cleaned mask ─────────────────────────────────
        clean_alpha = cv2.GaussianBlur(clean_alpha, (3, 3), 0.6)
        # Restore fully opaque interior pixels
        interior = alpha > 230
        clean_alpha[interior] = alpha[interior]

        r, g, b, _ = img_rgba.split()
        return Image.merge('RGBA', (r, g, b, Image.fromarray(clean_alpha)))

    except Exception as e:
        print(f'[remove_thin_protrusions] error: {e}')
        return img_rgba


def trim_top_objects(img_rgba):
    """
    Remove objects above the car (buildings, signs, overhead structures).

    After rembg, anything above the car body (tall objects connected to top edge)
    is clipped using a row-density valley approach — same idea as trim_side_cars
    but applied vertically from the top.

    Strategy:
    1. Compute row density (fg pixels per row)
    2. Find where density drops below 15% of peak going from bottom upward
    3. That valley = top of car body; erase everything above
    """
    try:
        import cv2
        alpha = np.array(img_rgba.split()[3])
        H, W  = alpha.shape
        binary = (alpha > 100).astype(np.uint8)

        row_sum = binary.sum(axis=1).astype(np.float32)  # per-row density
        if row_sum.max() == 0:
            return img_rgba

        row_smooth = cv2.GaussianBlur(row_sum.reshape(-1, 1), (31, 1), 0)[:, 0]
        peak = row_smooth.max()

        # Find center of mass (y-axis)
        total_fg = row_sum.sum()
        if total_fg > 0:
            y_coords = np.arange(H, dtype=np.float32)
            fg_cy = float(np.dot(row_sum, y_coords) / total_fg)
        else:
            fg_cy = H / 2.0

        # Valley threshold: 15% of peak — anything sparser than this is not car body
        valley_thresh = peak * 0.15

        # Search upward from center of mass for valley
        top_boundary = 0
        for y in range(int(fg_cy), -1, -1):
            if row_smooth[y] < valley_thresh:
                top_boundary = y + 1
                break

        # Only cut if we found a meaningful valley (not just top 1%)
        if top_boundary < int(H * 0.01):
            return img_rgba

        new_alpha = alpha.copy()
        new_alpha[:top_boundary, :] = 0

        r, g, b, _ = img_rgba.split()
        return Image.merge('RGBA', (r, g, b, Image.fromarray(new_alpha)))

    except Exception as e:
        print(f'[trim_top_objects] error: {e}')
        return img_rgba


def _restore_tyre_pixels_from_original(img_rgba, filepath, orig_size):
    """
    In the bottom 35% of the car (tyre zone), replace rembg-processed pixels
    with original image pixels where rembg alpha > 30.
    This gives real tyre texture/color instead of rembg artifacts.
    """
    try:
        orig = Image.open(filepath).convert("RGB")
        if orig.size != orig_size:
            orig = orig.resize(orig_size, Image.LANCZOS)

        alpha = np.array(img_rgba.split()[3], dtype=np.float32)
        H, W = alpha.shape

        rows_solid = np.where(np.any(alpha > 80, axis=1))[0]
        if len(rows_solid) < 10:
            return img_rgba

        car_top = int(rows_solid[0])
        car_bottom = int(rows_solid[-1])
        car_height = car_bottom - car_top + 1

        # Tyre zone = bottom 45% (FIX: was 35%, rear tyre on side-view cars was being cut)
        tyre_start = max(0, car_bottom - int(car_height * 0.45))

        orig_np = np.array(orig)
        img_np  = np.array(img_rgba)

        # In tyre zone, where alpha > 30, use original image RGB (real tyre pixels)
        zone_alpha = alpha[tyre_start:car_bottom+1, :]
        fg_mask = zone_alpha > 30

        img_np[tyre_start:car_bottom+1, :, 0][fg_mask] = orig_np[tyre_start:car_bottom+1, :, 0][fg_mask]
        img_np[tyre_start:car_bottom+1, :, 1][fg_mask] = orig_np[tyre_start:car_bottom+1, :, 1][fg_mask]
        img_np[tyre_start:car_bottom+1, :, 2][fg_mask] = orig_np[tyre_start:car_bottom+1, :, 2][fg_mask]

        return Image.fromarray(img_np.astype(np.uint8), 'RGBA')
    except Exception as e:
        print(f'[restore_tyre_pixels] error: {e}')
        return img_rgba


def restore_tyres(img_rgba):
    """
    Clean smudge/artifacts below tyres WITHOUT adding any pixels.
    Only removes junk below the true car bottom — does NOT fill or expand alpha.
    Column-span filling was removed: it was the source of black background bleed.
    Also recovers rear-tyre alpha that rembg clips due to dark tyre colour.
    """
    try:
        import cv2
        alpha = np.array(img_rgba.split()[3], dtype=np.float32)
        H, W  = alpha.shape

        # Find true car bottom using solid threshold only (>128)
        rows_solid = np.where(np.any(alpha > 128, axis=1))[0]
        if len(rows_solid) < 10:
            return img_rgba

        car_bottom_solid = int(rows_solid[-1])

        # Hard-zero everything strictly below the solid car bottom
        if car_bottom_solid + 1 < H:
            alpha[car_bottom_solid + 1:, :] = 0

        # Clean isolated near-invisible noise in bottom 2 rows only
        clean_start = max(0, car_bottom_solid - 1)
        for row in range(clean_start, car_bottom_solid + 1):
            row_alpha = alpha[row, :]
            low = row_alpha < 15   # only truly invisible pixels
            alpha[row, low] = 0

        # ── FIX: Recover rear-tyre alpha gaps ────────────────────────────────
        # rembg sometimes leaves a hollow gap (alpha ~0) inside dark tyre area.
        # Find the tyre zone (bottom 40% of car height) and fill column gaps
        # where a column has fg pixels above AND below a low-alpha gap.
        try:
            rows_any = np.where(np.any(alpha > 30, axis=1))[0]
            if len(rows_any) >= 10:
                car_top_any    = int(rows_any[0])
                car_bot_any    = int(rows_any[-1])
                car_h_any      = car_bot_any - car_top_any + 1
                tyre_zone_top  = max(0, car_bot_any - int(car_h_any * 0.42))

                binary = (alpha > 30).astype(np.uint8)
                for col in range(W):
                    col_slice = binary[tyre_zone_top:car_bot_any + 1, col]
                    fg_rows_in_zone = np.where(col_slice > 0)[0]
                    if len(fg_rows_in_zone) < 2:
                        continue
                    first_fg = int(fg_rows_in_zone[0])
                    last_fg  = int(fg_rows_in_zone[-1])
                    span_h   = last_fg - first_fg + 1
                    fg_count = int(fg_rows_in_zone[-1] - fg_rows_in_zone[0] + 1)
                    fill_ratio = len(fg_rows_in_zone) / max(span_h, 1)
                    # If <65% of span is filled → tyre hollow gap → fill it
                    if fill_ratio < 0.65 and span_h > 4:
                        fill_start = tyre_zone_top + first_fg
                        fill_end   = tyre_zone_top + last_fg + 1
                        # Fill with average of surrounding fg alpha values
                        avg_val = float(alpha[fill_start:fill_end, col][
                            alpha[fill_start:fill_end, col] > 30].mean()) if np.any(
                            alpha[fill_start:fill_end, col] > 30) else 200.0
                        alpha[fill_start:fill_end, col] = np.maximum(
                            alpha[fill_start:fill_end, col],
                            np.where(alpha[fill_start:fill_end, col] < 30,
                                     min(avg_val, 220), alpha[fill_start:fill_end, col])
                        )
        except Exception as e_tyre:
            print(f'[restore_tyres] rear-tyre gap fill error (non-fatal): {e_tyre}')

        r, g, b, _ = img_rgba.split()
        return Image.merge('RGBA', (r, g, b, Image.fromarray(alpha.astype(np.uint8))))

    except Exception as e:
        print(f'[restore_tyres] error: {e}')
        return img_rgba


def restore_windshield(img_rgba):
    """
    Fix transparent windshield/windows that rembg makes see-through.
    Strategy: fill ALL internal holes in the alpha mask — any transparent region
    completely surrounded by opaque car pixels is filled solid.
    This restores windshield, windows, and any other glass areas.
    """
    try:
        import cv2
        alpha = np.array(img_rgba.split()[3])
        H, W  = alpha.shape

        # Binary mask: opaque = 255, transparent = 0
        binary = (alpha > 80).astype(np.uint8) * 255

        # Step 1: Flood fill from ALL 4 borders to find definite background
        # Any region NOT reachable from border = internal hole = fill it
        flood = binary.copy()
        # Pad by 1 to allow flood fill from exact edge
        padded = np.zeros((H+2, W+2), np.uint8)
        padded[1:H+1, 1:W+1] = flood
        cv2.floodFill(padded, None, (0, 0), 255)
        # Remove padding
        flooded = padded[1:H+1, 1:W+1]
        # Internal holes = pixels that are 0 in binary but NOT reached from border
        internal_holes = (binary == 0) & (flooded == 0)

        if internal_holes.any():
            # Fill internal holes with full opacity
            alpha_new = alpha.copy()
            alpha_new[internal_holes] = 255

            # Smooth filled regions edges slightly
            filled_mask = internal_holes.astype(np.uint8) * 255
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            filled_dilated = cv2.dilate(filled_mask, k, iterations=1)
            blend_zone = (filled_dilated > 0) & ~internal_holes
            # Blend edge of filled zone
            alpha_blur = cv2.GaussianBlur(alpha_new.astype(np.float32), (7,7), 2)
            alpha_new[blend_zone] = np.maximum(alpha_new[blend_zone], alpha_blur[blend_zone].astype(np.uint8))

            r, g, b, _ = img_rgba.split()
            return Image.merge('RGBA', (r, g, b, Image.fromarray(alpha_new)))

        return img_rgba
    except Exception as e:
        print(f'[restore_windshield] error: {e}')
        return img_rgba


def clean_edges(img_rgba, erode_px=2):
    """
    Remove grey/white halo after BG removal.
    Uses edge-aware decontamination: removes color bleed from bg into car edges.
    """
    try:
        import cv2
        r, g, b, a = img_rgba.split()
        alpha = np.array(a)
        rgb   = np.array(img_rgba.convert('RGB'), dtype=np.float32)

        # Step 1: Erode alpha to remove fringe
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (erode_px * 2 + 1, erode_px * 2 + 1))
        eroded = cv2.erode(alpha, kernel, iterations=1)
        eroded = cv2.GaussianBlur(eroded, (3, 3), 0.5)
        final_alpha = np.minimum(alpha, eroded)

        # Step 2: Decontaminate edge pixels — remove bg color bleed
        # Any pixel with alpha 10-200 is an edge pixel — adjust its color
        # toward the nearest fully-opaque neighbor color
        edge_mask = (final_alpha > 10) & (final_alpha < 220)
        if edge_mask.any():
            # Dilate fully opaque region to get "safe" car color
            opaque_mask = (final_alpha > 220).astype(np.uint8) * 255
            opaque_dilated = cv2.dilate(opaque_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), iterations=2)
            # For edge pixels, blend toward opaque region color
            # This removes grey/white fringe from bg contamination
            for c_idx, channel in enumerate([np.array(r), np.array(g), np.array(b)]):
                ch = channel.astype(np.float32)
                # Smooth opaque colors to get reference
                ref = cv2.GaussianBlur(ch * (opaque_mask/255.0), (9,9), 3)
                ref_weight = cv2.GaussianBlur((opaque_mask/255.0), (9,9), 3) + 1e-5
                ref_color = ref / ref_weight
                # Blend edge pixels toward reference color
                alpha_ratio = final_alpha.astype(np.float32) / 255.0
                ch[edge_mask] = (ch[edge_mask] * alpha_ratio[edge_mask] +
                                 ref_color[edge_mask] * (1 - alpha_ratio[edge_mask]))
                ch = np.clip(ch, 0, 255).astype(np.uint8)
                if c_idx == 0: r = Image.fromarray(ch)
                elif c_idx == 1: g = Image.fromarray(ch)
                else: b = Image.fromarray(ch)

        return Image.merge('RGBA', (r, g, b, Image.fromarray(final_alpha)))
    except Exception as e:
        print(f'[clean_edges] error: {e}')
        return img_rgba


def fill_holes(mask):
    try:
        import cv2
        h, w = mask.shape
        flood = mask.copy()
        fm = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, fm, (0, 0), 255)
        return mask | cv2.bitwise_not(flood)
    except Exception:
        return mask


def remove_bg_car_opencv(filepath, quality='standard'):
    try:
        import cv2
        img_bgr = cv2.imread(filepath)
        if img_bgr is None:
            raise ValueError('Cannot read image')
        h, w = img_bgr.shape[:2]
        orig_size = (w, h)

        if max(h, w) > 1024:
            sc = 1024 / max(h, w)
            img_bgr = cv2.resize(img_bgr, (int(w * sc), int(h * sc)))
            h, w = img_bgr.shape[:2]

        margin_x = max(int(w * 0.02), 5)
        margin_y = max(int(h * 0.02), 5)
        rect  = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)
        mask  = np.zeros((h, w), np.uint8)
        bgd   = np.zeros((1, 65), np.float64)
        fgd   = np.zeros((1, 65), np.float64)
        cv2.grabCut(img_bgr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)

        cs = max(int(min(w, h) * 0.06), 10)
        # Only mark TOP corners as definite background — tyres are at the bottom
        mask[:cs, :cs] = mask[:cs, -cs:] = cv2.GC_BGD
        # Bottom corners: use probable-background only (not hard BG) to preserve tyres
        mask[-cs:, :cs] = mask[-cs:, -cs:] = cv2.GC_PR_BGD
        s = max(int(min(w, h) * 0.01), 3)
        mask[:s, :] = mask[:, :s] = mask[:, -s:] = cv2.GC_BGD
        # Bottom edge: softer — only mark a 1-pixel border as BG
        mask[-1, :] = cv2.GC_BGD

        # ── Mark outer left/right strips as definite background ───────────────
        # This suppresses side cars that are fully outside the center 80% width
        side_strip = max(int(w * 0.08), 8)
        mask[:, :side_strip]      = cv2.GC_BGD   # left edge strip → BG
        mask[:, w - side_strip:]  = cv2.GC_BGD   # right edge strip → BG

        cv2.grabCut(img_bgr, mask, rect, bgd, fgd, 8, cv2.GC_EVAL)
        fg_mask = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)

        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, k_close, iterations=4)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  k_open,  iterations=1)
        fg_mask = fill_holes(fg_mask)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            max_area = cv2.contourArea(contours[0])
            clean    = np.zeros_like(fg_mask)
            for c in contours:
                if cv2.contourArea(c) > max_area * 0.05:
                    cv2.drawContours(clean, [c], -1, 255, -1)
            fg_mask = fill_holes(clean)

        alpha   = cv2.GaussianBlur(fg_mask.astype(np.uint8), (3, 3), 0.8)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb).convert('RGBA')
        pil_img.putalpha(Image.fromarray(alpha).convert('L'))

        if pil_img.size != orig_size:
            pil_img = pil_img.resize(orig_size, Image.LANCZOS)

        # Keep only the largest foreground object
        pil_img = keep_largest_component(pil_img)
        pil_img = remove_persons_and_objects(pil_img)  # Remove humans, persons, side objects
        pil_img = remove_connected_persons(pil_img)    # Remove persons merged/touching the car
        pil_img = trim_side_cars(pil_img)
        pil_img = trim_top_objects(pil_img)   # Remove buildings/signs above car
        pil_img = remove_thin_protrusions(pil_img)  # FIX: Remove wipers/antennas/rods
        pil_img = restore_tyres(pil_img)      # Restore dark/clipped tyre alpha
        pil_img = restore_windshield(pil_img)  # Fix transparent windshield/windows
        pil_img = clean_edges(pil_img)        # Remove grey border halo

        is_valid, _ = validate_mask_quality(pil_img)
        if not is_valid:
            return None, None
        return pil_img, 'opencv_car'

    except ImportError:
        return None, None
    except Exception as e:
        print(f'[OpenCV] removal error: {type(e).__name__}: {e}')
        return None, None


# ─── Main BG Removal Dispatcher ──────────────────────────────────────────────

def remove_bg_ai(filepath, quality='standard', engine='auto', despill_enable=False):
    result_img = None
    method_used = 'fallback'
    if engine in ('auto', 'rembg'):
        result_img, method_used = remove_bg_rembg(filepath, quality)
    if result_img is None and engine in ('auto', 'grabcut'):
        result_img, method_used = remove_bg_car_opencv(filepath, quality)
    if result_img is None and engine == 'auto':
        result_img, method_used = remove_bg_rembg(filepath, 'draft')
    return result_img, method_used


# ─── Preview Generator ───────────────────────────────────────────────────────

def generate_mask_preview(nobg_img, width=300, height=200):
    checker = Image.new('RGB', (width, height))
    draw    = ImageDraw.Draw(checker)
    tile    = 12
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            col = (200, 200, 200) if (x // tile + y // tile) % 2 == 0 else (240, 240, 240)
            draw.rectangle([x, y, x + tile, y + tile], fill=col)
    preview = nobg_img.copy()
    preview.thumbnail((width, height), Image.LANCZOS)
    px = (width  - preview.width)  // 2
    py = (height - preview.height) // 2
    checker.paste(preview, (px, py), preview if preview.mode == 'RGBA' else None)
    buf = io.BytesIO()
    checker.save(buf, 'PNG')
    return base64.b64encode(buf.getvalue()).decode()


# ─── Caryanams Branding Layer ────────────────────────────────────────────────

def create_watermark_layer(W, H, bg_color_hint=None):
    """
    Place Caryanams branding centered near the top of the image:
      Line 1: 'Caryanams'       — centered, large, semi-transparent grey
      Line 2: 'Driven by Trust' — centered, smaller, golden/amber color
    Matches the reference image watermark style exactly.
    """
    arr = np.zeros((60, 60, 3), dtype=np.uint8)
    if bg_color_hint:
        arr[:] = bg_color_hint
    lum = float(arr.mean()) / 255.0 if bg_color_hint else 0.9  # assume light BG

    # Light background (white/grey): grey main text + golden subtitle
    # Dark background: light grey main text + golden subtitle
    if lum > 0.45:
        color1 = (160, 160, 160, 110)   # semi-transparent grey "Caryanams"
        color2 = (210, 165,  80, 140)   # golden/amber "Driven by Trust"
    else:
        color1 = (210, 210, 210, 120)   # light grey on dark bg
        color2 = (230, 185, 100, 150)   # golden on dark bg

    _font_list = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf',
        '/System/Library/Fonts/Helvetica.ttc',
        'C:/Windows/Fonts/arialbd.ttf',
    ]
    _font_list_reg = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
        '/System/Library/Fonts/Helvetica.ttc',
        'C:/Windows/Fonts/arial.ttf',
    ]

    def _load_font(paths, size):
        for fp in paths:
            try:
                return ImageFont.truetype(fp, int(size))
            except Exception:
                continue
        return ImageFont.load_default()

    # Larger watermark — prominent like reference image (center-top)
    sz1  = max(28, int(min(W, H) * 0.055))   # Caryanams — large
    sz2  = max(16, int(min(W, H) * 0.030))   # Driven by Trust — medium
    font1 = _load_font(_font_list,     sz1)
    font2 = _load_font(_font_list_reg, sz2)

    LINE1 = 'Caryanams'
    LINE2 = 'Driven by Trust'

    layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    ld    = ImageDraw.Draw(layer)
    _tmp  = ImageDraw.Draw(Image.new('RGBA', (1, 1)))

    def tsz(f, t):
        bb = _tmp.textbbox((0, 0), t, font=f)
        return bb[2] - bb[0], bb[3] - bb[1]

    tw1, th1 = tsz(font1, LINE1)
    tw2, th2 = tsz(font2, LINE2)
    gap      = max(4, sz1 // 8)
    total_h  = th1 + gap + th2

    # Center horizontally, position at center of image
    cx = W // 2
    top_y = max(int(H * 0.50), 30)

    x1 = cx - tw1 // 2
    x2 = cx - tw2 // 2
    y1 = top_y
    y2 = y1 + th1 + gap

    # Subtle drop shadow for readability
    shadow_color = (0, 0, 0, 25)
    so = max(1, sz1 // 30)
    ld.text((x1 + so, y1 + so), LINE1, fill=shadow_color, font=font1)
    ld.text((x2 + so, y2 + so), LINE2, fill=shadow_color, font=font2)

    ld.text((x1, y1), LINE1, fill=color1, font=font1)
    ld.text((x2, y2), LINE2, fill=color2, font=font2)

    return layer


# ─── Contact Shadow ──────────────────────────────────────────────────────────

def _add_contact_shadow(canvas, car_x, car_y, car_w, car_h, BG_W, BG_H,
                        fg_img=None, bg_is_light=True):
    """
    Professional showroom ground shadow — 4 layers matching reference image:
      L1: wide soft ambient penumbra  (spreads far, very faint dark)
      L2: mid-range diffuse shadow    (main body impression, dark)
      L3: crisp contact ellipse       (darkest, tight at ground)
      L4: silhouette-derived shadow   (real car shape projected)
    Only below/at the car. Zero side/top shadow bleed.
    On white/light BG: dark shadow. On dark BG: subtle lighter shadow.
    """
    from PIL import ImageFilter, ImageDraw

    floor = car_y + car_h        # y-coordinate of ground line
    cx    = car_x + car_w // 2  # horizontal center

    # Shadow color: dark on light BG, lighter on dark BG
    if bg_is_light:
        # Dark shadow on white background — matches reference image closely
        s1_fill = (0, 0, 0, 30)   # wide ambient
        s2_fill = (0, 0, 0, 55)   # mid diffuse
        s3_fill = (0, 0, 0, 90)   # tight contact
    else:
        # Subtle glow on dark background
        s1_fill = (255, 255, 255, 15)
        s2_fill = (255, 255, 255, 28)
        s3_fill = (255, 255, 255, 50)

    # ── L1: Wide ambient penumbra ─────────────────────────────────────────────
    layer1 = Image.new("RGBA", (BG_W, BG_H), (0, 0, 0, 0))
    d1 = ImageDraw.Draw(layer1)
    w1 = int(car_w * 1.05)
    h1 = max(55, int(car_h * 0.12))
    d1.ellipse([cx - w1 // 2, floor - h1 // 6,
                cx + w1 // 2, floor + h1],
               fill=s1_fill)
    layer1 = layer1.filter(ImageFilter.GaussianBlur(radius=max(38, int(car_w * 0.09))))

    # ── L2: Mid-range diffuse shadow ──────────────────────────────────────────
    layer2 = Image.new("RGBA", (BG_W, BG_H), (0, 0, 0, 0))
    d2 = ImageDraw.Draw(layer2)
    w2 = int(car_w * 0.88)
    h2 = max(36, int(car_h * 0.08))
    d2.ellipse([cx - w2 // 2, floor - h2 // 5,
                cx + w2 // 2, floor + h2],
               fill=s2_fill)
    layer2 = layer2.filter(ImageFilter.GaussianBlur(radius=max(22, int(car_w * 0.05))))

    # ── L3: Crisp contact ellipse — tight at exact ground ─────────────────────
    layer3 = Image.new("RGBA", (BG_W, BG_H), (0, 0, 0, 0))
    d3 = ImageDraw.Draw(layer3)
    w3 = int(car_w * 0.70)
    h3 = max(16, int(car_h * 0.04))
    d3.ellipse([cx - w3 // 2, floor - h3 // 3,
                cx + w3 // 2, floor + h3],
               fill=s3_fill)
    layer3 = layer3.filter(ImageFilter.GaussianBlur(radius=max(8, int(car_w * 0.02))))

    # ── L4: Silhouette shadow (projected from real car shape) ────────────────
    layer4 = None
    if fg_img is not None:
        try:
            fg_a = fg_img.convert("RGBA")
            alpha_ch = fg_a.split()[3]
            alpha_np = np.array(alpha_ch, dtype=np.float32)

            # Project bottom-row density onto ground
            bottom_strip_h = max(8, int(fg_a.height * 0.12))
            strip = alpha_np[-bottom_strip_h:, :]
            col_density = strip.mean(axis=0)

            shadow_h = max(10, int(car_h * 0.035))
            shadow_strip = np.zeros((shadow_h, car_w), dtype=np.float32)
            col_density_resized = np.interp(
                np.linspace(0, len(col_density) - 1, car_w),
                np.arange(len(col_density)), col_density
            )
            for row in range(shadow_h):
                fade = 1.0 - (row / shadow_h) ** 0.6
                shadow_strip[row, :] = col_density_resized * fade * 0.55

            shadow_strip = np.clip(shadow_strip, 0, 255).astype(np.uint8)
            shadow_img = Image.fromarray(shadow_strip, "L")
            shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(radius=4))

            layer4 = Image.new("RGBA", (BG_W, BG_H), (0, 0, 0, 0))
            shadow_rgba = Image.merge("RGBA", [
                Image.new("L", (car_w, shadow_h), 0),
                Image.new("L", (car_w, shadow_h), 0),
                Image.new("L", (car_w, shadow_h), 0),
                shadow_img
            ])
            paste_x = max(0, car_x)
            paste_y = floor
            if paste_y + shadow_h > BG_H:
                shadow_rgba = shadow_rgba.crop((0, 0, car_w, BG_H - paste_y))
            if paste_x + car_w > BG_W:
                shadow_rgba = shadow_rgba.crop((0, 0, BG_W - paste_x, shadow_rgba.height))
            layer4.paste(shadow_rgba, (paste_x, paste_y))
        except Exception as e_shadow:
            print(f"[shadow L4] error: {e_shadow}")
            layer4 = None

    # ── L4b: Extra darkest thin line exactly at tyre contact ─────────────────
    layer_contact = Image.new("RGBA", (BG_W, BG_H), (0, 0, 0, 0))
    d_c = ImageDraw.Draw(layer_contact)
    w_c = int(car_w * 0.55)
    h_c = max(7, int(car_h * 0.015))
    if bg_is_light:
        d_c.ellipse([cx - w_c // 2, floor - 1,
                     cx + w_c // 2, floor + h_c],
                    fill=(0, 0, 0, 100))
    else:
        d_c.ellipse([cx - w_c // 2, floor - 1,
                     cx + w_c // 2, floor + h_c],
                    fill=(255, 255, 255, 55))
    layer_contact = layer_contact.filter(ImageFilter.GaussianBlur(radius=3))

    canvas = Image.alpha_composite(canvas, layer1)
    canvas = Image.alpha_composite(canvas, layer2)
    canvas = Image.alpha_composite(canvas, layer3)
    if layer4 is not None:
        canvas = Image.alpha_composite(canvas, layer4)
    canvas = Image.alpha_composite(canvas, layer_contact)

    # ── L-shape: Subtle left & right side shadow (very light, not dark) ──────
    # Natural car photography shows slight side gradient at car edges —
    # very faint, just to give depth. Much lighter than bottom shadow.
    try:
        if bg_is_light:
            side_alpha = 72   # strong dark side vignette
        else:
            side_alpha = 60   # strong dark side vignette on dark bg

        # Left side shadow — wider and much stronger blur
        layer_left = Image.new("RGBA", (BG_W, BG_H), (0, 0, 0, 0))
        left_w = max(80, int(car_w * 0.18))   # wider shadow band
        left_h = max(60, int(car_h * 0.80))   # taller coverage
        dl = ImageDraw.Draw(layer_left)
        dl.ellipse([car_x - left_w // 2, car_y + car_h // 2 - left_h // 2,
                    car_x + left_w // 2, car_y + car_h // 2 + left_h // 2],
                   fill=(0, 0, 0, side_alpha))
        layer_left = layer_left.filter(ImageFilter.GaussianBlur(radius=max(40, int(car_w * 0.08))))

        # Right side shadow — wider and much stronger blur
        layer_right = Image.new("RGBA", (BG_W, BG_H), (0, 0, 0, 0))
        dr = ImageDraw.Draw(layer_right)
        right_x = car_x + car_w
        dr.ellipse([right_x - left_w // 2, car_y + car_h // 2 - left_h // 2,
                    right_x + left_w // 2, car_y + car_h // 2 + left_h // 2],
                   fill=(0, 0, 0, side_alpha))
        layer_right = layer_right.filter(ImageFilter.GaussianBlur(radius=max(40, int(car_w * 0.08))))

        canvas = Image.alpha_composite(canvas, layer_left)
        canvas = Image.alpha_composite(canvas, layer_right)
    except Exception as e_side:
        print(f"[shadow side] skipped: {e_side}")

    return canvas

# ─── Color Tint ──────────────────────────────────────────────────────────────

def apply_color_tint_to_bg(bg_img, tint_color_rgb, tint_strength=0.45):
    if tint_color_rgb is None:
        return bg_img
    bg_array = np.array(bg_img.convert('RGB'), dtype=np.float32)
    tint     = np.array(tint_color_rgb, dtype=np.float32)
    blended  = bg_array * (1 - tint_strength) + tint * tint_strength
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), 'RGB')


# ─── Caryanams Logo Overlay ───────────────────────────────────────────────────

def _load_logo_clean():
    """
    Load Caryanams logo as RGBA with white background removed.
    Returns clean RGBA logo suitable for pasting, or None if not found.
    """
    logo_candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'images', 'logo.png'),
        os.path.join('static', 'images', 'logo.png'),
    ]
    for lp in logo_candidates:
        if not os.path.exists(lp):
            continue
        try:
            logo = Image.open(lp).convert('RGBA')
            arr  = np.array(logo, dtype=np.float32)
            a    = arr[:,:,3]

            # Always remove white/near-white background for clean overlay
            r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

            # Hard white removal (very near-white pixels -> fully transparent)
            hard_white = (r >= 220) & (g >= 220) & (b >= 220)
            # Soft zone partial fade
            whiteness  = np.minimum(np.minimum(r, g), b)
            soft_zone  = (whiteness >= 180) & (~hard_white)
            new_alpha  = a.copy()
            new_alpha[hard_white] = 0
            fade = np.clip((whiteness[soft_zone] - 180) / 40.0, 0, 1)
            new_alpha[soft_zone] = (a[soft_zone] * (1.0 - fade)).astype(np.float32)
            arr[:,:,3] = new_alpha
            logo = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), 'RGBA')

            # Tight crop to remove transparent border
            ach = np.array(logo)[:,:,3]
            rows_v = np.any(ach > 8, axis=1)
            cols_v = np.any(ach > 8, axis=0)
            if rows_v.any() and cols_v.any():
                r0 = int(np.where(rows_v)[0][0]);  r1 = int(np.where(rows_v)[0][-1])
                c0 = int(np.where(cols_v)[0][0]);  c1 = int(np.where(cols_v)[0][-1])
                logo = logo.crop((c0, r0, c1+1, r1+1))
            return logo
        except Exception as e:
            print(f'[logo] load failed ({lp}): {e}')
    return None


def add_logo_overlay(img_rgb, car_top_y, canvas_w, canvas_h, opacity=0.42):
    """
    Overlay Caryanams logo: full width (left to right), vertically centered.
    Uses actual image size — not canvas_w/h — so logo always fills the real image.
    """
    logo = _load_logo_clean()
    if logo is None:
        return img_rgb

    # Use actual image dimensions
    actual_w, actual_h = img_rgb.size

    # Height = 30% of image height (same visual size as big text watermark)
    logo_h = int(actual_h * 0.30)
    logo_w = int(actual_w)  # full width left to right
    logo   = logo.resize((logo_w, logo_h), Image.LANCZOS)

    # 35% transparent watermark
    la = np.array(logo, dtype=np.float32)
    la[:,:,3] = np.clip(la[:,:,3] * 0.35, 0, 255)
    logo = Image.fromarray(np.clip(la, 0, 255).astype(np.uint8), 'RGBA')

    # x=0 full left to right, positioned in upper 1/3 of image (like reference)
    logo_x = 0
    logo_y = int(actual_h * 0.40)

    result = img_rgb.convert('RGBA')
    result.paste(logo, (logo_x, logo_y), logo)
    return result.convert('RGB')


# ─── Tiled Dark Watermark (Caryanams + Driven by Trust) ──────────────────────

def apply_tiled_watermark(img_pil, opacity=0.12):
    """
    Apply a tiled diagonal dark watermark over the entire image.
    Tile: 'Caryanams' (large, dark black) + 'Driven by Trust' (small, dark black)
    Rendered at ~45° diagonal repeat — exactly matching the HTML CSS watermark style.
    Works on both preview (in-memory) and downloaded files.
    Returns RGB PIL Image.
    """
    img = img_pil.convert('RGBA')
    iw, ih = img.size

    # ── Build one tile ────────────────────────────────────────────────────────
    # Tile dimensions matching the HTML: 300x150 equivalent, scaled to image size
    base_tile_w = max(220, int(iw * 0.28))
    base_tile_h = max(110, int(base_tile_w * 0.50))

    # Font sizes proportional to tile
    sz1 = max(22, int(base_tile_w * 0.155))   # 'Caryanams'
    sz2 = max(12, int(base_tile_w * 0.060))   # 'Driven by Trust'

    _font_bold = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf',
        'C:/Windows/Fonts/arialbd.ttf',
    ]
    _font_reg = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
        'C:/Windows/Fonts/arial.ttf',
    ]

    def _lf(paths, size):
        for fp in paths:
            try:
                return ImageFont.truetype(fp, int(size))
            except Exception:
                pass
        return ImageFont.load_default()

    font1 = _lf(_font_bold, sz1)
    font2 = _lf(_font_reg, sz2)

    LINE1 = 'Caryanams'
    LINE2 = 'Driven by Trust'

    # Measure text
    _tmp_draw = ImageDraw.Draw(Image.new('RGBA', (1, 1)))
    def tsz(f, t):
        bb = _tmp_draw.textbbox((0, 0), t, font=f)
        return bb[2] - bb[0], bb[3] - bb[1]

    tw1, th1 = tsz(font1, LINE1)
    tw2, th2 = tsz(font2, LINE2)

    # Tile: wide enough to hold both lines + padding
    tile_w = max(tw1, tw2) + int(base_tile_w * 0.30)
    gap    = max(3, sz1 // 8)
    tile_h = th1 + gap + th2 + int(base_tile_h * 0.55)

    tile = Image.new('RGBA', (tile_w, tile_h), (0, 0, 0, 0))
    td   = ImageDraw.Draw(tile)

    # Dark black color with controlled opacity
    alpha_val = int(255 * opacity)  # ~30 for 0.12 opacity
    text_color1 = (0, 0, 0, alpha_val)         # 'Caryanams' — pure black
    text_color2 = (0, 0, 0, int(alpha_val * 0.85))  # 'Driven by Trust' — slightly lighter

    # Center both lines in tile
    x1 = (tile_w - tw1) // 2
    x2 = (tile_w - tw2) // 2
    y1 = (tile_h - (th1 + gap + th2)) // 2
    y2 = y1 + th1 + gap

    td.text((x1, y1), LINE1, fill=text_color1, font=font1)
    td.text((x2, y2), LINE2, fill=text_color2, font=font2)

    # ── Rotate tile ~30° for diagonal effect ─────────────────────────────────
    tile_rot = tile.rotate(30, expand=True, resample=Image.BICUBIC)
    rot_w, rot_h = tile_rot.size

    # ── Tile across full image ────────────────────────────────────────────────
    wm_layer = Image.new('RGBA', (iw, ih), (0, 0, 0, 0))
    for ty in range(-rot_h, ih + rot_h, rot_h):
        for tx in range(-rot_w, iw + rot_w, rot_w):
            wm_layer.paste(tile_rot, (tx, ty), tile_rot)

    # ── Composite watermark over image ────────────────────────────────────────
    result = Image.alpha_composite(img, wm_layer)
    return result.convert('RGB')


# ─── Composite: Pure White BG + Perfect Shadow ───────────────────────────────

def _add_mirror_reflection(base_img, car_fg, car_x, car_y, car_w, car_h,
                           BG_W, BG_H, slice_pct=0.28):
    """
    Adds a dark mirror reflection directly below the car.
    - Crops bottom slice_pct of the car
    - Flips it vertically
    - Pastes it starting exactly at car bottom (car_y + car_h)
    - Applies heavy dark gradient so it's a dark shadow reflection
    - Fades out downward
    """
    floor_y = car_y + car_h   # y where car bottom touches ground

    # Source: bottom slice of car
    slice_h = max(10, int(car_h * slice_pct))
    src_top  = car_h - slice_h
    car_slice = car_fg.crop((0, src_top, car_w, car_h))  # RGBA

    # Flip vertically
    reflected = car_slice.transpose(Image.FLIP_TOP_BOTTOM)

    # Clamp reflection height so it doesn't go below image
    max_ref_h = BG_H - floor_y
    if max_ref_h <= 0:
        return base_img
    if slice_h > max_ref_h:
        reflected = reflected.crop((0, 0, car_w, max_ref_h))
        slice_h = max_ref_h

    ref_w, ref_h = reflected.size

    # Build gradient mask: opaque at top (near car), transparent at bottom
    # Top rows = keep most pixels, fade to nothing
    grad_arr = np.zeros((ref_h, ref_w), dtype=np.uint8)
    for row in range(ref_h):
        t = row / ref_h  # 0 at top (car contact), 1 at bottom
        # Starts at 55% opacity, fades to 0% quickly
        alpha = int(max(0, 140 * (1.0 - t ** 0.5)))
        grad_arr[row, :] = alpha

    grad_mask = Image.fromarray(grad_arr, 'L')

    # Apply gradient to alpha channel of reflection
    r, g, b, a = reflected.split()
    # Combine original alpha with gradient mask
    new_a = Image.fromarray(
        np.minimum(np.array(a), np.array(grad_mask)).astype(np.uint8)
    )

    # Darken RGB channels heavily — near-black tint
    r_np = (np.array(r).astype(np.float32) * 0.15).clip(0, 255).astype(np.uint8)
    g_np = (np.array(g).astype(np.float32) * 0.15).clip(0, 255).astype(np.uint8)
    b_np = (np.array(b).astype(np.float32) * 0.15).clip(0, 255).astype(np.uint8)

    reflected_dark = Image.merge("RGBA", (
        Image.fromarray(r_np),
        Image.fromarray(g_np),
        Image.fromarray(b_np),
        new_a
    ))

    # Paste onto base image at floor position
    base_rgba = base_img.convert("RGBA")
    base_rgba.paste(reflected_dark, (car_x, floor_y), reflected_dark)

    return base_rgba.convert("RGB")


def _make_blurred_bg(original_image_path, out_W, out_H):
    """
    Create a blurred background from the original uploaded image.
    Applies heavy gaussian blur + slight darkening for a studio look.
    Falls back to dark navy if the image cannot be loaded.
    """
    try:
        if original_image_path and os.path.exists(original_image_path):
            bg = Image.open(original_image_path).convert('RGBA')
            # Scale up slightly (110%) before blur so edges don't show white
            scale_w = int(out_W * 1.12)
            scale_h = int(out_H * 1.12)
            bg = bg.resize((scale_w, scale_h), Image.LANCZOS)
            # Crop center back to canvas size
            left = (scale_w - out_W) // 2
            top  = (scale_h - out_H) // 2
            bg   = bg.crop((left, top, left + out_W, top + out_H))
            # Heavy blur
            bg = bg.filter(ImageFilter.GaussianBlur(radius=28))
            # Slight darkening for contrast
            bg = ImageEnhance.Brightness(bg).enhance(0.72)
            return bg.convert('RGBA')
    except Exception as _e_blur:
        print(f'[blurred_bg] failed ({_e_blur}), using dark fallback')
    # Fallback: dark navy
    return Image.new('RGBA', (out_W, out_H), (26, 26, 46, 255))


def composite_car_on_static_bg(fg_img, car_size_percent=85, lighting=1.0, tint_color=None,
                               preserve_size=False, original_size=None,
                               original_image_path=None):
    """
    Blurred-original-image background composite:
    - Uses the original uploaded image, heavily blurred, as the background
    - Car tightly fitted, centered, with padding
    - Realistic multi-layer contact shadow ONLY at bottom
    - Zero border artifacts (tight bbox crop + edge cleaning)
    """
    fg = fg_img.copy().convert("RGBA")

    # ── Aggressive edge clean before composite ───────────────────────────────
    fg = clean_edges(fg, erode_px=4)  # FIX: more aggressive halo removal

    # ── Light alpha noise removal — keep tire edges intact ──────────────────
    try:
        r_ch, g_ch, b_ch, a_ch = fg.split()
        a_np = np.array(a_ch, dtype=np.float32)
        a_np[a_np < 8] = 0   # only pure-invisible noise
        fg = Image.merge("RGBA", (r_ch, g_ch, b_ch, Image.fromarray(a_np.astype(np.uint8))))
    except Exception as _e_thresh:
        print(f"[alpha_light_clean] skipped: {_e_thresh}")

    # Remove all transparent padding — tight crop to actual car pixels
    bbox = fg.getbbox()
    if not bbox:
        raise ValueError("Foreground image is fully transparent")
    fg = fg.crop(bbox)

    fw, fh = fg.size
    if fw == 0 or fh == 0:
        raise ValueError("Invalid image size after crop")

    # ── Output canvas size ───────────────────────────────────────────────────
    if preserve_size and original_size:
        out_W, out_H = original_size
    else:
        out_W, out_H = max(fw + 200, 1200), max(fh + 200, 900)

    # ── Size car to fit canvas with padding ──────────────────────────────────
    pad_x = int(out_W * 0.06)   # 6% padding each side
    pad_y_top = int(out_H * 0.06)
    pad_y_bot = int(out_H * 0.16)  # more bottom padding for shadow

    max_car_w = out_W - 2 * pad_x
    max_car_h = out_H - pad_y_top - pad_y_bot

    # Scale car to fill available space (maintain aspect)
    scale_by_w = max_car_w / fw
    scale_by_h = max_car_h / fh
    scale = min(scale_by_w, scale_by_h)

    # Apply car_size_percent as additional scale factor
    size_pct = max(10, min(100, float(car_size_percent))) / 100.0
    scale *= size_pct

    target_width  = max(10, int(fw * scale))
    target_height = max(10, int(fh * scale))

    fg = fg.resize((target_width, target_height), Image.LANCZOS)

    # ── Background canvas: blurred original image ───────────────────────────
    canvas = _make_blurred_bg(original_image_path, out_W, out_H)

    # ── Position: horizontally centered, bottom sits at floor_y ─────────────
    floor_y = out_H - pad_y_bot   # where car bottom touches ground
    car_x   = (out_W - target_width) // 2
    # No shadow offset — car sits directly on the floor line
    car_y   = floor_y - target_height

    # Clamp
    car_x = max(0, min(car_x, out_W - target_width))
    car_y = max(0, min(car_y, out_H - target_height))

    # ── Add dark contact shadow BEFORE pasting car ──────────────────────────
    try:
        canvas = _add_contact_shadow(canvas, car_x, car_y, target_width, target_height,
                                     out_W, out_H, fg_img=fg, bg_is_light=True)
    except Exception as _e_shadow:
        print(f"[shadow] skipped: {_e_shadow}")

    # ── Paste car ────────────────────────────────────────────────────────────
    canvas.paste(fg, (car_x, car_y), fg)

    # ── Post-paste: remove isolated dark blob islands around tyres ──────────
    # rembg misses some dark background pixels near tyres — they appear as
    # black blobs on white BG. Use connected components to find blobs that
    # are NOT connected to the main car body and erase them.
    try:
        import cv2 as _cv2
        r_c, g_c, b_c, a_c = canvas.split()
        a_np = np.array(a_c, dtype=np.uint8)

        # Binary mask of all fg pixels
        binary = (a_np > 30).astype(np.uint8) * 255

        # Connected components
        num_labels, labels, stats, centroids = _cv2.connectedComponentsWithStats(
            binary, connectivity=8)

        if num_labels > 2:
            # Find largest component (main car)
            areas = [stats[l, _cv2.CC_STAT_AREA] for l in range(1, num_labels)]
            main_lbl = int(np.argmax(areas)) + 1
            main_area = stats[main_lbl, _cv2.CC_STAT_AREA]

            # Erase any component smaller than 2% of main car area
            # These are isolated dark blobs around tyres, not real car parts
            for lbl in range(1, num_labels):
                if lbl == main_lbl:
                    continue
                blob_area = stats[lbl, _cv2.CC_STAT_AREA]
                if blob_area < main_area * 0.02:
                    a_np[labels == lbl] = 0

        canvas = Image.merge("RGBA", (r_c, g_c, b_c, Image.fromarray(a_np)))
    except Exception as _e_blob:
        print(f"[blob_clean] skipped: {_e_blob}")

    # Flatten onto blurred background RGB
    blur_base = _make_blurred_bg(original_image_path, out_W, out_H).convert('RGB')
    blur_base.paste(canvas, (0, 0), canvas)

    # ── Mirror reflection: flip bottom 28% of car, paste below car, dark tint ──
    try:
        blur_base = _add_mirror_reflection(blur_base, fg, car_x, car_y,
                                           target_width, target_height, out_W, out_H)
    except Exception as _e_mir:
        print(f"[mirror] skipped: {_e_mir}")

    # ── Logo overlay centered ────────────────────────────────────────────────
    try:
        blur_base = add_logo_overlay(blur_base, car_y, out_W, out_H, opacity=1.0)
    except Exception as _e_logo:
        print(f"[logo_overlay] skipped: {_e_logo}")

    return blur_base


# ─── Composite: Custom / Color BG ────────────────────────────────────────────

def apply_background_color(fg_img, bg_color=None, width=1200, height=800,
                           lighting=1.0, shadow=True,
                           shadow_intensity=0.85, shadow_blur=40,
                           position_y_offset=0.0, bg_image_path=None,
                           car_size_percent=75, preserve_size=False,
                           original_size=None):
    # Use original image size if requested
    if preserve_size and original_size:
        width, height = original_size

    base = bg_image_path if (bg_image_path and os.path.exists(bg_image_path)) else STATIC_BG_PATH

    if os.path.exists(base):
        bg = Image.open(base).convert('RGB').resize((width, height), Image.LANCZOS)
        if bg_color and bg_color != (255, 255, 255):
            bg = apply_color_tint_to_bg(bg, bg_color, tint_strength=0.50)
    else:
        # Fall back to studio BG if plain color requested, so empty space is studio
        if os.path.exists(STATIC_BG_PATH):
            bg = Image.open(STATIC_BG_PATH).convert('RGB').resize((width, height), Image.LANCZOS)
            if bg_color:
                bg = apply_color_tint_to_bg(bg, bg_color, tint_strength=0.50)
        else:
            bg = Image.new('RGB', (width, height), bg_color or (0, 0, 0))

    canvas = bg.convert('RGBA')
    fg     = fg_img.copy().convert('RGBA')
    bbox   = fg.getbbox()
    if bbox:
        fg = fg.crop(bbox)

    fw, fh    = fg.size
    fg_aspect = fw / fh
    size_ratio    = min(90, max(35, car_size_percent)) / 100.0
    target_width  = int(width * size_ratio)
    target_height = int(target_width / fg_aspect)

    max_height = int(height * 0.78)
    if target_height > max_height:
        target_height = max_height
        target_width  = int(target_height * fg_aspect)

    fg = fg.resize((target_width, target_height), Image.LANCZOS)
    # No lighting adjustments — preserve real car colors

    ground_line_y = int(height * 0.92)
    car_x = (width - target_width) // 2
    car_y = ground_line_y - target_height + int(height * position_y_offset)
    car_x = max(10, min(car_x, width  - target_width  - 10))
    car_y = max(10, min(car_y, height - target_height - 30))

    if shadow:
        # Determine if background is light or dark for shadow color
        try:
            bg_arr = np.array(bg.convert('RGB'), dtype=np.float32)
            bg_lum = float(bg_arr.mean()) / 255.0
            _bg_is_light = bg_lum > 0.45
        except Exception:
            _bg_is_light = True
        canvas = _add_contact_shadow(canvas, car_x, car_y, target_width, target_height,
                                     width, height, fg_img=fg, bg_is_light=_bg_is_light)

    canvas.paste(fg, (car_x, car_y), fg)

    # Flatten onto white so no transparent pixels remain
    white_base = Image.new('RGB', (width, height), (255, 255, 255))
    white_base.paste(canvas, (0, 0), canvas)

    # ── Caryanams logo overlay — centered above car, semi-transparent ─────────
    try:
        white_base = add_logo_overlay(white_base, car_y, width, height, opacity=0.90)
    except Exception as _e_logo2:
        print(f"[logo_overlay_bg] skipped: {_e_logo2}")

    return white_base


# ═══════════════════════════════════════════════════════════
#  NUMBER PLATE DETECTION + REMOVAL (ported from caryanams)
# ═══════════════════════════════════════════════════════════

def _score_candidate(x, y, w, h, img_w, img_h):
    """Score a plate candidate. Higher = more likely to be the number plate."""
    if w <= 0 or h <= 0:
        return 0
    aspect     = w / h
    area_ratio = (w * h) / (img_w * img_h)
    cx_ratio   = (x + w / 2) / img_w
    cy_ratio   = (y + h / 2) / img_h
    score = 0

    # Plate must be in lower 70% of image (y center)
    if cy_ratio < 0.30:
        return 0
    # Indian plate aspect: 1.4 to 9.5 (normal 4-5, partially visible 1.5-2)
    if aspect < 1.4 or aspect > 9.5:
        return 0
    # Size limits
    if w > img_w * 0.70 or h > img_h * 0.25:
        return 0
    if w < max(30, img_w * 0.025) or h < max(8, img_h * 0.008):
        return 0

    # Aspect scoring — Indian standard ~4.7:1
    if   3.8 <= aspect <= 5.5:  score += 50
    elif 3.0 <= aspect <= 6.5:  score += 35
    elif 2.0 <= aspect <= 8.0:  score += 18
    elif 1.4 <= aspect <= 9.5:  score += 6

    # Area scoring
    if   0.005 <= area_ratio <= 0.09: score += 30
    elif 0.002 <= area_ratio <= 0.18: score += 14

    # Size reasonability
    if max(30, img_w*0.025) <= w <= img_w*0.65 and max(8, img_h*0.008) <= h <= img_h*0.18:
        score += 20

    # Vertical position
    if   0.65 <= cy_ratio <= 0.95: score += 20
    elif 0.50 <= cy_ratio <= 0.65: score += 12
    elif 0.30 <= cy_ratio <= 0.50: score += 5

    # Horizontal — NO penalty for edge plates (front 3/4 view plates are at right edge)
    if   0.25 <= cx_ratio <= 0.75: score += 10
    elif 0.10 <= cx_ratio <= 0.90: score += 6
    else:                           score += 3   # edge plate still valid

    return score


def _method_edge(img_cv, img_w, img_h):
    candidates = []
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 17, 17)
    for thresh_lo, thresh_hi in [(20, 150), (40, 200), (60, 250)]:
        edges = cv2.Canny(blur, thresh_lo, thresh_hi)
        cnts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:60]
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            s = _score_candidate(x, y, w, h, img_w, img_h)
            if s >= 40:
                candidates.append((s, x, y, w, h))
    return candidates


def _method_morph(img_cv, img_w, img_h):
    candidates = []
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    for kw in [15, 20, 25]:
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 3))
        morph  = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            s = _score_candidate(x, y, w, h, img_w, img_h)
            if s >= 38:
                candidates.append((s, x, y, w, h))
    return candidates


def _method_color(img_cv, img_w, img_h):
    """Detect white plates and yellow plates (Indian vehicles)."""
    candidates = []
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    masks = [
        cv2.inRange(hsv, np.array([0,   0, 170]), np.array([180, 45, 255])),
        cv2.inRange(hsv, np.array([0,   0, 190]), np.array([180, 30, 255])),
        cv2.inRange(hsv, np.array([18,  80,  80]), np.array([35, 255, 255])),
    ]
    for mask in masks:
        for kw in [12, 18]:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 4))
            m2 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            cnts, _ = cv2.findContours(m2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                s = _score_candidate(x, y, w, h, img_w, img_h)
                if s >= 30:
                    candidates.append((s, x, y, w, h))
    return candidates


def _method_sobel(img_cv, img_w, img_h):
    candidates = []
    gray  = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    sobX  = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobX  = cv2.convertScaleAbs(sobX)
    _, th = cv2.threshold(sobX, 45, 255, cv2.THRESH_BINARY)
    for kw in [18, 25]:
        k   = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 4))
        th2 = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)
        cnts, _ = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            s = _score_candidate(x, y, w, h, img_w, img_h)
            if s >= 36:
                candidates.append((s, x, y, w, h))
    return candidates


def _method_white_rect(img_cv, img_w, img_h):
    """Dedicated white rectangle detector for clean white plates."""
    candidates = []
    hsv   = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 55, 255]))
    for kw in [10, 15, 20]:
        k  = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 3))
        w2 = cv2.morphologyEx(white, cv2.MORPH_CLOSE, k)
        cnts, _ = cv2.findContours(w2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            s = _score_candidate(x, y, w, h, img_w, img_h)
            if s >= 35:
                candidates.append((s, x, y, w, h))
    return candidates


def _method_rect_contour(img_cv, img_w, img_h):
    """Find 4-sided rectangular contours with plate proportions."""
    candidates = []
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    for thresh_val in [100, 130, 160, 190]:
        _, th = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < 150:
                continue
            peri  = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(approx)
                s = _score_candidate(x, y, w, h, img_w, img_h)
                if s >= 45:
                    candidates.append((s, x, y, w, h))
    return candidates


def _expand_to_full_plate(img_cv, img_w, img_h, bx, by, bw, bh):
    """Expand a rough detection box to capture the complete plate."""
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    # Search area: wider than the detection
    spx = max(int(bw * 0.8), 40)
    spy = max(int(bh * 1.5), 30)
    sx1 = max(0,     bx - spx)
    sy1 = max(0,     by - spy)
    sx2 = min(img_w, bx + bw + spx)
    sy2 = min(img_h, by + bh + spy)
    roi = gray[sy1:sy2, sx1:sx2]

    best_box   = None
    best_score = -1
    for t in [110, 130, 150, 170, 190]:
        _, roi_th = cv2.threshold(roi, t, 255, cv2.THRESH_BINARY)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (16, 4))
        roi_th = cv2.morphologyEx(roi_th, cv2.MORPH_CLOSE, k)
        cnts2, _ = cv2.findContours(roi_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c2 in cnts2:
            ex, ey, ew, eh = cv2.boundingRect(c2)
            abs_x = sx1 + ex; abs_y = sy1 + ey
            # Must have meaningful overlap with original detection
            int_x1 = max(abs_x, bx); int_y1 = max(abs_y, by)
            int_x2 = min(abs_x+ew, bx+bw); int_y2 = min(abs_y+eh, by+bh)
            ow = max(0, int_x2 - int_x1); oh = max(0, int_y2 - int_y1)
            if bw * bh == 0 or (ow * oh) / (bw * bh) < 0.30:
                continue
            s2 = _score_candidate(abs_x, abs_y, ew, eh, img_w, img_h)
            wb = 15 if ew >= bw * 0.85 else 0
            if s2 + wb > best_score:
                best_score = s2 + wb
                best_box   = (abs_x, abs_y, ew, eh)

    if best_box and best_box[2] >= bw * 0.70:
        return best_box
    return (bx, by, bw, bh)


def detect_number_plate(image_path):
    """
    Detect Indian number plate. Returns (x, y, w, h) in original image pixels, or None.
    Handles: rear view, front view, side 3/4 view, plates at any edge of frame.
    """
    if cv2 is None:
        return None
    try:
        img_pil = Image.open(image_path).convert('RGB')
        img_w, img_h = img_pil.size
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # Run all detection methods
        all_candidates = []
        for fn in [_method_edge, _method_morph, _method_color,
                   _method_sobel, _method_white_rect, _method_rect_contour]:
            try:
                all_candidates.extend(fn(img_cv, img_w, img_h))
            except Exception:
                pass

        if not all_candidates:
            return None

        # De-duplicate: merge candidates that overlap significantly
        all_candidates.sort(key=lambda c: c[0], reverse=True)
        used   = [False] * len(all_candidates)
        merged = []
        for i, (si, xi, yi, wi, hi) in enumerate(all_candidates):
            if used[i]:
                continue
            cx1, cy1, cx2, cy2 = xi, yi, xi+wi, yi+hi
            for j in range(len(all_candidates)):
                if used[j] or j == i:
                    continue
                sj, xj, yj, wj, hj = all_candidates[j]
                jx1, jy1, jx2, jy2 = xj, yj, xj+wj, yj+hj
                inter_x1 = max(cx1, jx1); inter_y1 = max(cy1, jy1)
                inter_x2 = min(cx2, jx2); inter_y2 = min(cy2, jy2)
                inter_a  = max(0, inter_x2-inter_x1) * max(0, inter_y2-inter_y1)
                union_a  = wi*hi + wj*hj - inter_a
                iou = inter_a / union_a if union_a > 0 else 0
                if iou > 0.08:
                    cx1 = min(cx1, jx1); cy1 = min(cy1, jy1)
                    cx2 = max(cx2, jx2); cy2 = max(cy2, jy2)
                    used[j] = True
            used[i] = True
            mw, mh = cx2-cx1, cy2-cy1
            new_s = _score_candidate(cx1, cy1, mw, mh, img_w, img_h)
            merged.append((max(si, new_s), cx1, cy1, mw, mh))

        merged.sort(key=lambda c: c[0], reverse=True)
        _, bx, by, bw, bh = merged[0]

        # Expand to capture the full plate
        bx, by, bw, bh = _expand_to_full_plate(img_cv, img_w, img_h, bx, by, bw, bh)

        # Final padding
        pad = max(6, int(min(bw, bh) * 0.12))
        bx = max(0,      bx - pad)
        by = max(0,      by - pad)
        bw = min(img_w - bx, bw + pad * 2)
        bh = min(img_h - by, bh + pad * 2)
        return (bx, by, bw, bh)

    except Exception as e:
        print(f'[detect_number_plate] error: {e}')
        return None
def _load_font(size):
    from PIL import ImageFont
    for name in ['DejaVuSans-Bold.ttf', 'Arial.ttf', 'FreeSansBold.ttf']:
        for path in ['/usr/share/fonts/truetype/dejavu/', '/usr/share/fonts/', '/usr/share/fonts/truetype/']:
            fp = os.path.join(path, name)
            if os.path.exists(fp):
                try:
                    return ImageFont.truetype(fp, size)
                except Exception:
                    pass
    return ImageFont.load_default()


def _load_logo_rgba():
    """
    Load Caryanams logo as RGBA with white/near-white background made transparent.
    Returns a clean cutout suitable for pasting directly onto car body color.
    """
    logo_candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'images', 'logo.png'),
        os.path.join('static', 'images', 'logo.png'),
        os.path.join(os.path.dirname(__file__), '..', 'static', 'images', 'logo.png'),
    ]
    for lp in logo_candidates:
        if not os.path.exists(lp):
            continue
        try:
            logo = Image.open(lp).convert('RGBA')
            arr  = np.array(logo, dtype=np.float32)
            a    = arr[:,:,3]

            # Only remove white background if logo doesn't already have transparency
            already_transparent = float(np.sum(a < 10)) / (arr.shape[0] * arr.shape[1])
            if already_transparent < 0.15:
                r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
                hard_white = (r >= 228) & (g >= 228) & (b >= 228)
                whiteness   = np.minimum(np.minimum(r, g), b)
                soft_zone   = (whiteness >= 185) & (~hard_white)
                new_alpha = a.copy()
                new_alpha[hard_white] = 0
                fade = np.clip((whiteness[soft_zone] - 185) / 43.0, 0, 1)
                new_alpha[soft_zone] = (a[soft_zone] * (1.0 - fade)).astype(np.float32)
                arr[:,:,3] = new_alpha
                logo = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), 'RGBA')

            # Crop tight to content
            alpha_arr = np.array(logo)[:,:,3]
            rows = np.any(alpha_arr > 8, axis=1)
            cols = np.any(alpha_arr > 8, axis=0)
            if rows.any() and cols.any():
                r0 = int(np.where(rows)[0][0])
                r1 = int(np.where(rows)[0][-1])
                c0 = int(np.where(cols)[0][0])
                c1 = int(np.where(cols)[0][-1])
                r0 = max(0, r0 - 2); r1 = min(logo.height - 1, r1 + 2)
                c0 = max(0, c0 - 2); c1 = min(logo.width  - 1, c1 + 2)
                logo = logo.crop((c0, r0, c1 + 1, r1 + 1))
            return logo
        except Exception as e:
            print(f'[logo] load failed ({lp}): {e}')
    return None


def apply_plate_removal(image_path, output_path, x, y, w, h, mode='caryanams', quad=None):
    """Hide number plate region. mode='caryanams' uses Caryanams logo badge. Returns True on success.
    quad: optional list of 4 points [{'x':..,'y':..}, ..] in TL→TR→BR→BL order (natural image coords).
    When quad is given, perspective-warps the logo patch to fit the quadrilateral exactly.
    """
    from PIL import ImageDraw, ImageFilter
    import numpy as np
    try:
        img = Image.open(image_path).convert('RGBA')
        iw, ih = img.size

        pad = max(4, int(min(w, h) * 0.10))
        rx  = max(0, x - pad)
        ry  = max(0, y - pad)
        rw  = min(iw - rx, w + pad * 2)
        rh  = min(ih - ry, h + pad * 2)

        draw = ImageDraw.Draw(img)

        if mode == 'caryanams':
            # ── Step 1: Sample car body color from AROUND the plate area ───────
            # Use multiple sample regions for a robust median color
            samples = []
            for sy0, sy1 in [
                (max(0, ry - max(20, rh)), max(1, ry - 2)),          # above plate
                (min(ih - 2, ry + rh + 2), min(ih, ry + rh + max(20, rh))),  # below plate
            ]:
                for sx0, sx1 in [
                    (max(0, rx - max(20, rw // 4)), max(1, rx - 2)),   # left of plate
                    (min(iw - 2, rx + rw + 2), min(iw, rx + rw + max(20, rw // 4))),  # right
                ]:
                    try:
                        patch = np.array(img.crop((sx0, sy0, sx1, sy1)))
                        if patch.size > 0 and patch.shape[0] > 0 and patch.shape[1] > 0:
                            samples.append(patch.reshape(-1, 4)[:, :3])
                    except Exception:
                        pass
            # Also sample directly above — most reliable for bumper/body color
            try:
                above_h = max(6, rh)
                above = np.array(img.crop((rx, max(0, ry - above_h), rx + rw, max(1, ry - 1))))
                if above.size > 0:
                    samples.append(above.reshape(-1, 4)[:, :3])
            except Exception:
                pass

            if samples:
                all_px = np.concatenate(samples, axis=0).astype(np.float32)
                avg_r = int(np.median(all_px[:, 0]))
                avg_g = int(np.median(all_px[:, 1]))
                avg_b = int(np.median(all_px[:, 2]))
            else:
                avg_r, avg_g, avg_b = 120, 120, 120

            # ── Step 2 & 3: Replace plate with exact white bg + logo only ──────
            # NO extra painting around plate, NO border, NO body color fill
            plate_w = w
            plate_h = h
            plate_x = x
            plate_y = y

            # White patch EXACTLY plate size — nothing more, nothing less
            white_patch = Image.new('RGBA', (plate_w, plate_h), (255, 255, 255, 255))

            logo = _load_logo_rgba()
            if logo is not None:
                lw_orig, lh_orig = logo.size
                # Fit logo inside plate with small padding, preserve aspect ratio
                pad = max(2, int(min(plate_w, plate_h) * 0.08))
                fit_w = max(1, plate_w - pad * 2)
                fit_h = max(1, plate_h - pad * 2)
                scale = min(fit_w / lw_orig, fit_h / lh_orig)
                new_lw = max(1, int(lw_orig * scale))
                new_lh = max(1, int(lh_orig * scale))
                logo_r = logo.resize((new_lw, new_lh), Image.LANCZOS)
                lx = (plate_w - new_lw) // 2
                ly = (plate_h - new_lh) // 2
                white_patch.paste(logo_r, (lx, ly), logo_r)

            # ── Step 4: Paste — use perspective warp if quad corners provided ──
            if quad and len(quad) == 4:
                # quad: [TL, TR, BR, BL] in natural image pixel coords
                import cv2
                # Source rectangle corners (white_patch canvas)
                src_pts = np.float32([
                    [0,           0          ],
                    [plate_w - 1, 0          ],
                    [plate_w - 1, plate_h - 1],
                    [0,           plate_h - 1],
                ])
                # Destination quad corners in image coords
                dst_pts = np.float32([
                    [quad[0]['x'], quad[0]['y']],  # TL
                    [quad[1]['x'], quad[1]['y']],  # TR
                    [quad[2]['x'], quad[2]['y']],  # BR
                    [quad[3]['x'], quad[3]['y']],  # BL
                ])
                # Compute perspective transform
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)

                # Warp the white+logo patch into a full-image canvas
                patch_cv = cv2.cvtColor(np.array(white_patch), cv2.COLOR_RGBA2BGRA)
                warped   = cv2.warpPerspective(patch_cv, M, (iw, ih),
                                               flags=cv2.INTER_LINEAR,
                                               borderMode=cv2.BORDER_CONSTANT,
                                               borderValue=(0, 0, 0, 0))
                # Convert back to RGBA PIL
                warped_rgba = Image.fromarray(
                    cv2.cvtColor(warped, cv2.COLOR_BGRA2RGBA), 'RGBA')

                # Create a mask from the alpha channel of warped patch
                _, _, _, warped_alpha = warped_rgba.split()
                # Composite: paste warped patch over image using its alpha as mask
                img.paste(warped_rgba, (0, 0), warped_alpha)
            else:
                # Paste white+logo patch at exact plate coordinates (no perspective)
                img.paste(white_patch, (plate_x, plate_y))

        elif mode == 'blur':
            if quad and len(quad) == 4:
                # Perspective-accurate blur: fill polygon with blurred pixels
                import cv2, numpy as np
                img_np = np.array(img)
                pts = np.array([[quad[0]['x'], quad[0]['y']],
                                [quad[1]['x'], quad[1]['y']],
                                [quad[2]['x'], quad[2]['y']],
                                [quad[3]['x'], quad[3]['y']]], dtype=np.int32)
                # Create polygon mask
                mask = np.zeros((ih, iw), dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 255)
                # Blur the entire image then composite using mask
                blurred_full = cv2.GaussianBlur(img_np[:, :, :3],
                                                (max(21, (rw // 4) | 1), max(21, (rh // 4) | 1)), 0)
                img_out = img_np.copy()
                img_out[mask == 255, :3] = blurred_full[mask == 255]
                img = Image.fromarray(img_out, 'RGBA')
            else:
                region  = img.crop((rx, ry, rx+rw, ry+rh))
                blurred = region.filter(ImageFilter.GaussianBlur(radius=max(8, rw//8)))
                img.paste(blurred, (rx, ry))
        elif mode == 'black':
            if quad and len(quad) == 4:
                import cv2, numpy as np
                img_np = np.array(img)
                pts = np.array([[quad[0]['x'], quad[0]['y']],
                                [quad[1]['x'], quad[1]['y']],
                                [quad[2]['x'], quad[2]['y']],
                                [quad[3]['x'], quad[3]['y']]], dtype=np.int32)
                cv2.fillPoly(img_np, [pts], (0, 0, 0, 255))
                img = Image.fromarray(img_np, 'RGBA')
            else:
                draw.rectangle([rx, ry, rx+rw, ry+rh], fill=(0, 0, 0, 255))
        elif mode == 'white':
            if quad and len(quad) == 4:
                import cv2, numpy as np
                img_np = np.array(img)
                pts = np.array([[quad[0]['x'], quad[0]['y']],
                                [quad[1]['x'], quad[1]['y']],
                                [quad[2]['x'], quad[2]['y']],
                                [quad[3]['x'], quad[3]['y']]], dtype=np.int32)
                cv2.fillPoly(img_np, [pts], (255, 255, 255, 255))
                img = Image.fromarray(img_np, 'RGBA')
            else:
                draw.rectangle([rx, ry, rx+rw, ry+rh], fill=(255, 255, 255, 255))
        else:
            sample_y      = max(0, ry - rh)
            sample_region = img.crop((rx, sample_y, rx+rw, sample_y+rh))
            img.paste(sample_region, (rx, ry))

        img.save(output_path, 'PNG')
        return True
    except Exception:
        return False


def process_plate_and_background(image_path, output_path, mode='caryanams', manual=None, car_size_pct=75, image_category='exterior', quad=None, quality='ultra'):
    """
    Full pipeline: detect+remove plate, remove background, apply showroom BG.
    image_category: 'exterior' (default) = BG remove + showroom BG applied
                    'interior'           = only plate hide, no BG change (full-screen preserved)
                    'plate_only'         = only plate hide, keep original image as-is
    Returns (ok, plate_info_dict_or_None).
    """
    import traceback
    try:
        # Step 1: detect plate
        plate = None
        if manual:
            plate = (int(manual.get('x', 0)), int(manual.get('y', 0)),
                     int(manual.get('w', 0)), int(manual.get('h', 0)))
        else:
            plate = detect_number_plate(image_path)

        plate_info = None
        if plate:
            plate_info = {'x': plate[0], 'y': plate[1], 'width': plate[2], 'height': plate[3]}

        # ── INTERIOR / PLATE-ONLY MODE: no BG removal, just hide plate ───────
        # Interior images are kept full-screen as-is; only plate area is masked
        if image_category in ('interior', 'plate_only'):
            if plate:
                import shutil
                work_path = output_path.replace('.png', '_work.png')
                result = apply_plate_removal(image_path, work_path, *plate, mode=mode, quad=quad)
                if result and os.path.exists(work_path):
                    shutil.copy2(work_path, output_path)
                    try:
                        os.remove(work_path)
                    except Exception:
                        pass
                else:
                    # apply_plate_removal failed silently — fall back to original image
                    shutil.copy2(image_path, output_path)
            else:
                import shutil
                shutil.copy2(image_path, output_path)
            return True, plate_info

        # ── EXTERIOR MODE: plate hide → BG remove → showroom composite ────────
        # Step 2: apply plate removal to a temp file
        work_path = output_path.replace('.png', '_work.png')
        try:
            result = apply_plate_removal(image_path, work_path, *(plate or (0, 0, 1, 1)), mode=mode, quad=quad)
            if not result or not os.path.exists(work_path):
                raise RuntimeError('apply_plate_removal did not create work file')
        except Exception:
            import shutil
            shutil.copy2(image_path, work_path)

        # Step 3: remove background
        _has_rembg_local = False
        try:
            from rembg import remove as rembg_remove
            _has_rembg_local = True
        except ImportError:
            pass

        nobg_img = None

        if _has_rembg_local:
            try:
                from rembg import new_session
                # Use isnet-general-use for best quality (ChatGPT-level results)
                _q = quality if quality else 'ultra'
                model_name = 'isnet-general-use'
                session = None
                try:
                    session = new_session(model_name)
                except Exception:
                    session = None
                # FIX MemoryError: reduce max_side and disable alpha_matting to avoid OOM.
                img_bytes, orig_size = _resize_for_removal(work_path, 800)
                alpha_matting = False  # alpha_matting causes MemoryError on large images
                if session:
                    nobg_bytes = rembg_remove(
                        img_bytes, session=session,
                        alpha_matting=alpha_matting,
                        alpha_matting_foreground_threshold=200,
                        alpha_matting_background_threshold=20,
                        alpha_matting_erode_size=3
                    )
                else:
                    nobg_bytes = rembg_remove(img_bytes, alpha_matting=alpha_matting)
                nobg_img = Image.open(io.BytesIO(nobg_bytes)).convert('RGBA')
                # Restore to original size
                if nobg_img.size != orig_size:
                    nobg_img = nobg_img.resize(orig_size, Image.LANCZOS)
                # Restore original tyre pixels (real texture, not rembg artifacts)
                nobg_img = _restore_tyre_pixels_from_original(nobg_img, work_path, orig_size)
            except Exception as e:
                print(f'[rembg] failed: {e}')
                nobg_img = None

        if nobg_img is None:
            # Fallback: opencv-based BG removal (returns tuple: img, method)
            try:
                result_tuple = remove_bg_car_opencv(work_path)
                if isinstance(result_tuple, tuple):
                    nobg_img = result_tuple[0]
                else:
                    nobg_img = result_tuple
            except Exception as e:
                print(f'[opencv] failed: {e}')
                nobg_img = None

        # ── Step 3b: Remove persons and unwanted objects ─────────────────────
        # Remove humans, persons standing beside/behind car, and other non-car
        # objects that survived BG removal. Runs before tyre/edge processing.
        if nobg_img is not None:
            try:
                nobg_img = keep_largest_component(nobg_img)
                nobg_img = remove_persons_and_objects(nobg_img)
                nobg_img = remove_connected_persons(nobg_img)  # Remove persons merged/touching car
                nobg_img = trim_side_cars(nobg_img)
                nobg_img = trim_top_objects(nobg_img)
                nobg_img = remove_thin_protrusions(nobg_img)  # FIX: Remove wipers/antennas/rods
            except Exception as e_clean:
                print(f'[process_plate_and_background] person/object removal error: {e_clean}')

        # ── Edge smoothing + tyre preservation ──────────────────────────────────
        # Applies restore_tyres (handles dark/black tyres with low alpha threshold)
        # then does a gentle blur on non-wheel edges.
        if nobg_img is not None:
            try:
                # Run the dedicated tyre restorer (works for black cars too)
                # restore_tyres DISABLED — causes black ring around tyres
                nobg_img = restore_windshield(nobg_img)  # Fix windshield

                r, g, b, a = nobg_img.split()
                a_np = np.array(a, dtype=np.float32)

                # Find wheel zone extent for post-blur protection
                rows_with_fg = np.where(np.any(a_np > 30, axis=1))[0]
                wheel_zone_start = None
                car_bottom = None
                if len(rows_with_fg) > 0:
                    car_top    = int(rows_with_fg[0])
                    car_bottom = int(rows_with_fg[-1])
                    car_height = car_bottom - car_top + 1
                    # Protect bottom 32% from blur softening
                    wheel_zone_start = max(0, car_bottom - int(car_height * 0.32))

                # Gentle Gaussian blur on alpha to smooth hard edges
                from PIL import ImageFilter
                a_smooth = Image.fromarray(a_np.astype(np.uint8)).filter(
                    ImageFilter.GaussianBlur(radius=0.6)
                )
                a_np2 = np.array(a_smooth, dtype=np.float32)
                a_np2 = np.clip(a_np2, 0, 255)

                # Restore wheel zone alpha after blur
                if wheel_zone_start is not None and car_bottom is not None:
                    a_np2[wheel_zone_start:car_bottom + 1, :] = np.maximum(
                        a_np2[wheel_zone_start:car_bottom + 1, :],
                        a_np[wheel_zone_start:car_bottom + 1, :]
                    )

                nobg_img = Image.merge('RGBA', (r, g, b, Image.fromarray(a_np2.astype(np.uint8))))
            except Exception:
                pass

        if nobg_img is None:
            # Last resort fallback: use image as-is (no BG removal, just plate hide + showroom BG)
            print('[process_plate_and_background] No BG removal available — using original with plate hidden')
            try:
                nobg_img = Image.open(work_path).convert('RGBA')
            except Exception as e_open:
                print(f'[process_plate_and_background] work_path open failed ({e_open}), trying original image')
                try:
                    nobg_img = Image.open(image_path).convert('RGBA')
                except Exception as e_orig:
                    print(f'[process_plate_and_background] original image open also failed: {e_orig}')
                    return False, None

        # Step 4: composite on showroom background (preserve original image size)
        try:
            orig = Image.open(image_path)
            orig_size = orig.size
            orig.close()
            if True:
                final = composite_car_on_static_bg(nobg_img, car_size_percent=car_size_pct,
                                                   preserve_size=True, original_size=orig_size,
                                                   original_image_path=image_path)
            else:
                # No static BG: composite on white background at original size
                print('[process_plate_and_background] static BG missing — using white background')
                white_bg = Image.new('RGB', orig_size, (255, 255, 255))
                canvas   = white_bg.convert('RGBA')
                fg_sized = nobg_img.copy().convert('RGBA')
                if fg_sized.size != orig_size:
                    fg_sized = fg_sized.resize(orig_size, Image.LANCZOS)
                canvas.paste(fg_sized, (0, 0), fg_sized)
                final = canvas.convert('RGB')
            final.save(output_path, 'PNG')
        except Exception as e_composite:
            print(f'[process_plate_and_background] composite failed ({e_composite}), saving nobg direct')
            try:
                nobg_img.save(output_path, 'PNG')
            except Exception as e_save:
                print(f'[process_plate_and_background] final save also failed: {e_save}')
                return False, None

        # Cleanup temp
        try:
            if os.path.exists(work_path):
                os.remove(work_path)
        except Exception:
            pass

        return True, plate_info

    except Exception as e:
        import traceback as tb
        tb.print_exc()
        print(f'[process_plate_and_background] FAILED: {e}')
        return False, None


# ─── Center Logo Stamp ────────────────────────────────────────────────────────

_LOGO_CACHE = {}

def stamp_center_logo(img_pil, logo_path, logo_size_ratio=0.28, opacity=0.88):
    """
    Stamp the Caryanams logo exactly at the center of img_pil.
    - logo_size_ratio: logo width as fraction of image width (default 28%)
    - opacity: 0-1 blend strength
    Returns a new PIL Image (RGB).
    """
    global _LOGO_CACHE
    try:
        img = img_pil.convert('RGBA')
        iw, ih = img.size

        # Load & cache logo
        if logo_path not in _LOGO_CACHE:
            logo_raw = Image.open(logo_path).convert('RGBA')

            # Make white pixels transparent (logo has white background)
            import numpy as np
            data = np.array(logo_raw, dtype=np.float32)
            # Pixels that are "very white" → make transparent
            r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
            whiteness = (r / 255.0 + g / 255.0 + b / 255.0) / 3.0
            # Soft threshold: near-white pixels fade out
            alpha_mask = 1.0 - np.clip((whiteness - 0.80) / 0.18, 0, 1)
            data[:,:,3] = (alpha_mask * 255).astype(np.uint8)
            _LOGO_CACHE[logo_path] = Image.fromarray(data.astype(np.uint8), 'RGBA')

        logo_orig = _LOGO_CACHE[logo_path]

        # Scale logo to logo_size_ratio of image width
        lw = max(80, int(iw * logo_size_ratio))
        lh = int(logo_orig.height * (lw / logo_orig.width))
        logo = logo_orig.resize((lw, lh), Image.LANCZOS)

        # Center position
        lx = (iw - lw) // 2
        ly = (ih - lh) // 2

        # Apply opacity
        logo_data = logo.copy()
        if opacity < 1.0:
            import numpy as np
            arr = np.array(logo_data, dtype=np.float32)
            arr[:,:,3] = arr[:,:,3] * opacity
            logo_data = Image.fromarray(arr.astype(np.uint8), 'RGBA')

        # Composite
        canvas = img.copy()
        canvas.paste(logo_data, (lx, ly), logo_data)
        return canvas.convert('RGB')

    except Exception as e:
        print(f'[stamp_center_logo] failed: {e}')
        return img_pil.convert('RGB')
