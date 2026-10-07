#!/usr/bin/env python3
"""Tile screenshots side by side for review: sheet.py out.png scale file1 file2 ..."""
import sys
from PIL import Image
out, scale, files = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
ims = [Image.open(f).convert("RGB") for f in files]
ims = [im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS) for im in ims]
gap = 8
W = sum(i.width for i in ims) + gap * (len(ims) - 1)
H = max(i.height for i in ims)
sheet = Image.new("RGB", (W, H), (120, 120, 120))
x = 0
for im in ims:
    sheet.paste(im, (x, 0)); x += im.width + gap
sheet.save(out)
