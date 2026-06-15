"""
Caryanams Studio Routes — Simplified
Features: 1) Background Blur  2) Draw Plate  3) Crop + Download + Delete
"""

import os
import io
import uuid
import base64
from datetime import datetime

from flask import (
    Blueprint, render_template, request, jsonify,
    send_file, current_app
)
from PIL import Image

from extensions import db
from utils import (
    detect_number_plate, apply_plate_removal,
    apply_60_percent_background_blur,
    remove_bg_ai,
    keep_largest_component, remove_persons_and_objects,
    remove_connected_persons, trim_side_cars,
    trim_top_objects, remove_thin_protrusions,
    restore_tyres, restore_windshield,
    stamp_center_logo, add_logo_overlay,
    apply_tiled_watermark,
    allowed_file
)

def _logo_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'images', 'logo.png')

bp = Blueprint('main', __name__)

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}


# ─── DB Model ─────────────────────────────────────────────────────────────────

class ProcessedImage(db.Model):
    __tablename__ = 'processed_image'
    id             = db.Column(db.String(50), primary_key=True)
    filename       = db.Column(db.String(255))
    original_path  = db.Column(db.String(500))
    nobg_path      = db.Column(db.String(500))
    processed_path = db.Column(db.String(500))
    status         = db.Column(db.String(30), default='uploaded')
    created_at     = db.Column(db.DateTime, default=datetime.utcnow)


# ─── Folder helpers ───────────────────────────────────────────────────────────

def _upload_folder():
    f = os.path.join(current_app.root_path, 'static', 'uploads')
    os.makedirs(f, exist_ok=True)
    return f

def _processed_folder():
    f = os.path.join(current_app.root_path, 'static', 'processed')
    os.makedirs(f, exist_ok=True)
    return f


# ─── Watermark Helper ─────────────────────────────────────────────────────────

def _apply_both_watermarks(img_pil):
    """
    Apply Caryanams tiled dark watermark over the entire image.
    Dark black 'Caryanams' + 'Driven by Trust' repeated diagonally across image.
    Both preview and download get the same watermark.
    Returns RGB PIL Image.
    """
    img = img_pil.convert('RGB')
    try:
        img = apply_tiled_watermark(img, opacity=0.12)
    except Exception as e:
        print(f'[watermark] apply_tiled_watermark failed (non-fatal): {e}')
    return img.convert('RGB')


# ─── Pages ────────────────────────────────────────────────────────────────────

@bp.route('/')
def index():
    return render_template('index.html')


# ─── Upload ───────────────────────────────────────────────────────────────────

@bp.route('/api/upload', methods=['POST'])
def upload():
    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No files received'}), 400

    results, errors = [], []
    for f in files:
        if not f or not f.filename:
            continue
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            errors.append(f'{f.filename}: unsupported type')
            continue
        try:
            uid   = str(uuid.uuid4())[:12]
            fname = f'img_{uid}{ext}'
            path  = os.path.join(_upload_folder(), fname)
            f.save(path)

            # Convert webp to jpg
            if ext == '.webp':
                img = Image.open(path).convert('RGB')
                new_fname = f'img_{uid}.jpg'
                new_path  = os.path.join(_upload_folder(), new_fname)
                img.save(new_path, 'JPEG', quality=92)
                os.remove(path)
                fname, path = new_fname, new_path

            # ── Auto-watermark: dono watermarks upload pe hi lagao ───────────
            wm_path = None
            watermarked_url = None
            try:
                wm_fname = f'wm_{uid}.jpg'
                wm_path  = os.path.join(_processed_folder(), wm_fname)
                img_wm   = _apply_both_watermarks(Image.open(path))
                img_wm.save(wm_path, 'JPEG', quality=92)
                watermarked_url = '/static/processed/' + wm_fname
            except Exception as _wm_err:
                print(f'[upload] watermark failed (non-fatal): {_wm_err}')

            rec = ProcessedImage(
                id=uid, filename=fname, original_path=path,
                processed_path=wm_path if watermarked_url else None,
                status='watermarked' if watermarked_url else 'uploaded'
            )
            db.session.add(rec)
            results.append({
                'id': uid,
                'filename': fname,
                'original_url':  '/static/uploads/' + fname,
                'processed_url': watermarked_url,
                'status': rec.status
            })
        except Exception as e:
            errors.append(str(e))

    if not results and errors:
        db.session.rollback()
        return jsonify({'error': '; '.join(errors)}), 500

    db.session.commit()
    return jsonify({'results': results, 'errors': errors})


# ─── Background Blur ──────────────────────────────────────────────────────────

@bp.route('/api/blur-bg/<image_id>', methods=['POST'])
def blur_bg(image_id):
    """Remove background via AI and apply depth-of-field blur to background only.
    Result: Sharp car + beautifully blurred original background (like DSLR bokeh).
    """
    rec = ProcessedImage.query.get(image_id)
    if not rec:
        return jsonify({'error': 'Image not found'}), 404

    data    = request.get_json(silent=True) or {}
    quality = data.get('quality', 'standard')

    # Use processed image as source if available (e.g. plate already hidden)
    src_path = rec.processed_path or rec.original_path

    # Step 1: AI BG removal to get car mask
    result, method = remove_bg_ai(src_path, quality=quality)
    if result is None:
        try:
            result = Image.open(src_path).convert('RGBA')
            method = 'original_kept'
        except Exception:
            return jsonify({'error': 'BG removal failed'}), 500

    # Step 1b: Cleanup mask
    try:
        result = keep_largest_component(result)
        result = remove_persons_and_objects(result)
        result = remove_connected_persons(result)
        result = trim_side_cars(result)
        result = trim_top_objects(result)
        result = remove_thin_protrusions(result)
        result = restore_tyres(result)
        result = restore_windshield(result)
    except Exception as _e:
        print(f'[blur_bg] cleanup error (non-fatal): {_e}')

    # Step 2: Apply blur — keep car sharp, blur real background
    pf       = _processed_folder()
    out_path = os.path.join(pf, f'blur_{rec.id}.jpg')

    try:
        blurred = apply_60_percent_background_blur(src_path, result)
        # Apply BOTH watermarks
        blurred = _apply_both_watermarks(blurred)
        blurred.save(out_path, 'JPEG', quality=95)
    except Exception as e:
        print(f'[blur_bg] blur failed: {e}')
        try:
            fallback = Image.open(src_path).convert('RGB')
            fallback = _apply_both_watermarks(fallback)
            fallback.save(out_path, 'JPEG', quality=95)
        except Exception:
            return jsonify({'error': 'Save failed'}), 500

    rec.processed_path = out_path
    rec.status         = 'completed'
    db.session.commit()

    return jsonify({
        'success':       True,
        'processed_url': '/static/processed/' + os.path.basename(out_path),
        'status':        'completed',
        'method':        method + '_blur'
    })


# ─── Detect Plate ─────────────────────────────────────────────────────────────

@bp.route('/api/detect-plate/<image_id>')
def detect_plate(image_id):
    rec = ProcessedImage.query.get(image_id)
    if not rec:
        return jsonify({'detected': False, 'message': 'Not found'}), 404
    plate = detect_number_plate(rec.original_path)
    try:
        iw, ih = Image.open(rec.original_path).size
    except Exception:
        return jsonify({'detected': False, 'message': 'Cannot read image'}), 400
    if plate:
        x, y, w, h = plate
        return jsonify({'detected': True, 'x': x, 'y': y, 'width': w, 'height': h,
                        'img_width': iw, 'img_height': ih})
    return jsonify({'detected': False, 'img_width': iw, 'img_height': ih,
                    'message': 'Auto-detection failed. Use manual selection.'})


# ─── Apply Plate (Draw Panel) ─────────────────────────────────────────────────

@bp.route('/api/apply-plate/<image_id>', methods=['POST'])
def apply_plate(image_id):
    """Apply plate overlay from draw panel (manual or auto detected)."""
    rec = ProcessedImage.query.get(image_id)
    if not rec:
        return jsonify({'error': 'Not found'}), 404

    data   = request.get_json(silent=True) or {}
    mode   = data.get('mode', 'caryanams')
    manual = data.get('manual')
    quad   = data.get('quad')

    # Always apply plate hide on original image — blur is applied separately
    src_path = rec.original_path

    if manual:
        plate = (int(manual.get('x', 0)), int(manual.get('y', 0)),
                 int(manual.get('w', 0)), int(manual.get('h', 0)))
    else:
        plate = detect_number_plate(src_path)

    if not plate:
        return jsonify({'success': False, 'message': 'No plate detected. Use draw mode.'}), 400

    pf       = _processed_folder()
    out_path = os.path.join(pf, f'plate_{rec.id}.png')
    ok       = apply_plate_removal(src_path, out_path, *plate, mode=mode, quad=quad)

    if ok and os.path.exists(out_path):
        # Apply BOTH watermarks
        try:
            stamped = _apply_both_watermarks(Image.open(out_path))
            stamped.save(out_path, 'PNG')
        except Exception as _le:
            print(f'[apply_plate] watermark failed (non-fatal): {_le}')
        rec.processed_path = out_path
        rec.status         = 'completed'
        db.session.commit()
        return jsonify({
            'success':       True,
            'processed_url': '/static/processed/' + os.path.basename(out_path),
            'plate':         {'x': plate[0], 'y': plate[1], 'width': plate[2], 'height': plate[3]},
            'message':       '✅ Plate hidden!'
        })

    return jsonify({'success': False, 'message': 'Plate removal failed.'}), 500


# ─── Save Crop ────────────────────────────────────────────────────────────────

@bp.route('/api/save-crop/<image_id>', methods=['POST'])
def save_crop(image_id):
    rec = ProcessedImage.query.get(image_id)
    if not rec:
        return jsonify({'error': 'Not found'}), 404

    data    = request.get_json(silent=True) or {}
    b64data = data.get('image_data', '')
    if not b64data:
        return jsonify({'error': 'No image data'}), 400

    if ',' in b64data:
        b64data = b64data.split(',', 1)[1]

    try:
        img_bytes = base64.b64decode(b64data)
        img       = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        # Apply BOTH watermarks
        img = _apply_both_watermarks(img)
        pf        = _processed_folder()
        out_path  = os.path.join(pf, f'crop_{rec.id}.png')
        img.save(out_path, 'PNG')
        rec.processed_path = out_path
        rec.status         = 'completed'
        db.session.commit()
        return jsonify({
            'success':       True,
            'processed_url': '/static/processed/' + os.path.basename(out_path)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── Download ─────────────────────────────────────────────────────────────────

@bp.route('/api/download/<image_id>')
def download(image_id):
    rec  = ProcessedImage.query.get(image_id)
    if not rec:
        return jsonify({'error': 'Not found'}), 404
    path = rec.processed_path or rec.original_path
    if not path or not os.path.exists(path):
        return jsonify({'error': 'File not found'}), 404
    ext  = os.path.splitext(path)[1]
    name = f'caryanams_{image_id}{ext}'
    return send_file(path, as_attachment=True, download_name=name)


@bp.route('/api/download-all-zip', methods=['POST'])
def download_all_zip():
    import zipfile, io as _io
    data = request.get_json(silent=True) or {}
    ids  = data.get('ids', [])
    if not ids:
        return jsonify({'error': 'No IDs provided'}), 400

    buf = _io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for image_id in ids:
            rec = ProcessedImage.query.get(image_id)
            if not rec:
                continue
            path = rec.processed_path or rec.original_path
            if not path or not os.path.exists(path):
                continue
            ext  = os.path.splitext(path)[1]
            name = f'caryanams_{image_id}{ext}'
            zf.write(path, name)

    buf.seek(0)
    return send_file(buf, mimetype='application/zip', as_attachment=True,
                     download_name='caryanams_processed.zip')


# ─── Delete ───────────────────────────────────────────────────────────────────

@bp.route('/api/delete/<image_id>', methods=['DELETE'])
def delete_image(image_id):
    rec = ProcessedImage.query.get(image_id)
    if not rec:
        return jsonify({'error': 'Not found'}), 404
    for path_attr in ['original_path', 'processed_path', 'nobg_path']:
        p = getattr(rec, path_attr, None)
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass
    db.session.delete(rec)
    db.session.commit()
    return jsonify({'success': True})
