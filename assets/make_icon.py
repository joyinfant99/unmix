"""Build assets/Unmix.icns from icons/Unmix2.png (artwork: a light tile on a white backdrop).
The tile is cut out with a squircle mask fitted to its measured edge, placed on the macOS icon grid
(824px on a 1024px canvas) with a soft shadow."""
import subprocess, shutil, os
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "icons", "Unmix2.png")
BOX = (64, 57, 1188, 1182)  # tile edge in the source image (left, top, right, bottom), measured from the rim
N = 5.1                     # squircle exponent that matches the artwork's corners
SS = 4                      # mask supersampling

src = Image.open(SRC).convert("RGB")
l, t, r, b = BOX
w, h = r - l, b - t

# squircle mask via superellipse, drawn supersampled then downsampled for a clean edge
mw, mh = w * SS, h * SS
mask = Image.new("L", (mw, mh), 0)
d = ImageDraw.Draw(mask)
cx, cy, a, bb = mw / 2, mh / 2, mw / 2, mh / 2
import math
pts = []
for i in range(720):
    th = 2 * math.pi * i / 720
    c, s = math.cos(th), math.sin(th)
    pts.append((cx + a * math.copysign(abs(c) ** (2 / N), c), cy + bb * math.copysign(abs(s) ** (2 / N), s)))
d.polygon(pts, fill=255)
mask = mask.resize((w, h), Image.LANCZOS)

tile = src.crop(BOX).convert("RGBA")
tile.putalpha(mask)

S = 1024; body = 824
tile = tile.resize((body, round(body * h / w)), Image.LANCZOS)
canvas = Image.new("RGBA", (S, S), (0, 0, 0, 0))
pos = ((S - tile.width) // 2, (S - tile.height) // 2)
shadow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
shadow.paste((0, 0, 0, 110), (pos[0], pos[1] + 10), tile.split()[3])
canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(14)))
canvas.alpha_composite(tile, pos)
canvas.save(os.path.join(HERE, "icon_1024.png"))

iconset = os.path.join(HERE, "Unmix.iconset")
shutil.rmtree(iconset, ignore_errors=True); os.makedirs(iconset)
for s in (16, 32, 128, 256, 512):
    canvas.resize((s, s), Image.LANCZOS).save(f"{iconset}/icon_{s}x{s}.png")
    canvas.resize((s * 2, s * 2), Image.LANCZOS).save(f"{iconset}/icon_{s}x{s}@2x.png")
subprocess.run(["iconutil", "-c", "icns", iconset, "-o", os.path.join(HERE, "Unmix.icns")], check=True)
shutil.rmtree(iconset)
print("wrote Unmix.icns")
