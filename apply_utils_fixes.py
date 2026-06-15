"""
apply_utils_fixes.py
─────────────────────────────────────────────────────────────────────────
Automatically applies the memory/timeout fixes discussed for utils.py:

  1. Adds `import gc`
  2. Removes the startup prewarm thread (rembg model load on boot)
  3. Switches 'standard' quality model from isnet-general-use -> u2netp
  4. Shrinks max_side values (draft/standard/ultra) to reduce memory use
  5. Shrinks the resize size used inside process_plate_and_background

HOW TO USE
─────────────────────────────────────────────────────────────────────────
1. Put this file in the SAME folder as your existing utils.py
2. Run:
       python apply_utils_fixes.py
3. It will:
       - back up your current file as utils.py.bak
       - write the fully updated utils.py in place

After running, utils.py IS the complete, updated file — no manual
editing needed.
─────────────────────────────────────────────────────────────────────────
"""

import os
import shutil

TARGET = 'utils.py'
BACKUP = 'utils.py.bak'


def main():
    if not os.path.exists(TARGET):
        print(f'ERROR: {TARGET} not found in this folder. '
              f'Place this script next to your utils.py and run again.')
        return

    with open(TARGET, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content
    applied = []

    # ── 1. Add `import gc` ──────────────────────────────────────────────
    old = "import threading\nimport numpy as np"
    new = "import threading\nimport gc\nimport numpy as np"
    if old in content:
        content = content.replace(old, new, 1)
        applied.append('1. Added "import gc"')
    else:
        print('  [skip] step 1: import block not found (already patched?)')

    # ── 2. Remove the startup prewarm thread ────────────────────────────
    old = (
        "# FIX: daemon=True ensures this thread won't block server shutdown\n"
        "threading.Thread(target=_prewarm, daemon=True).start()\n"
    )
    if old in content:
        content = content.replace(old, '', 1)
        applied.append('2. Removed startup prewarm thread')
    else:
        print('  [skip] step 2: prewarm thread line not found (already patched?)')

    # ── 3. Switch 'standard' quality model to the lighter u2netp ────────
    old = (
        "        model_map = {\n"
        "            'draft':    'u2netp',\n"
        "            'standard': 'isnet-general-use',\n"
        "            'high':     'isnet-general-use',\n"
        "            'ultra':    'isnet-general-use',\n"
        "        }"
    )
    new = (
        "        model_map = {\n"
        "            'draft':    'u2netp',\n"
        "            'standard': 'u2netp',\n"
        "            'high':     'isnet-general-use',\n"
        "            'ultra':    'isnet-general-use',\n"
        "        }"
    )
    if old in content:
        content = content.replace(old, new, 1)
        applied.append("3. 'standard' quality now uses u2netp (lighter model)")
    else:
        print('  [skip] step 3: model_map block not found (already patched?)')

    # ── 4. Shrink max_side values for rembg removal ─────────────────────
    old = (
        "        if quality == 'draft':\n"
        "            max_side = 640\n"
        "        elif quality == 'ultra':\n"
        "            max_side = 1024\n"
        "        else:  # standard / high\n"
        "            max_side = 800"
    )
    new = (
        "        if quality == 'draft':\n"
        "            max_side = 512\n"
        "        elif quality == 'ultra':\n"
        "            max_side = 768\n"
        "        else:  # standard / high\n"
        "            max_side = 600"
    )
    if old in content:
        content = content.replace(old, new, 1)
        applied.append('4. Reduced max_side sizes (512 / 600 / 768)')
    else:
        print('  [skip] step 4: max_side block not found (already patched?)')

    # ── 5. Shrink resize used in process_plate_and_background ───────────
    old = "img_bytes, orig_size = _resize_for_removal(work_path, 800)"
    new = "img_bytes, orig_size = _resize_for_removal(work_path, 600)"
    if old in content:
        content = content.replace(old, new, 1)
        applied.append('5. Reduced process_plate_and_background resize to 600px')
    else:
        print('  [skip] step 5: _resize_for_removal(work_path, 800) not found (already patched?)')

    if content == original:
        print('\nNo changes were applied — file may already be patched, '
              'or its content has changed from the expected version.')
        return

    # Backup + write
    shutil.copy2(TARGET, BACKUP)
    with open(TARGET, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f'\nBackup saved as {BACKUP}')
    print(f'{TARGET} updated successfully. Changes applied:')
    for a in applied:
        print('  -', a)


if __name__ == '__main__':
    main()
