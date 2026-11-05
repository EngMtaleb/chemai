"""Our plant from above, to scale (0.65 m per pixel), with realistic textures.
The beach and the sea are the AI satellite picture's own pixels (plant_sat_raw.png, right of the shore);
everything else is placed by this script at our design's sizes, with textures cut from that same picture:
panels 70,400 kWp at 0.87 MWdc/ha -> a 900 m square; the RO hall 90 m x 15 m (the racks of 12 trains, 84 m x 10 m,
with room around them); the tank 44,000 m3 at 10 m deep -> 75 m across; the pump building about 30 m x 17 m;
8 m roads; the field 50 m from the hall. Writes plant_to_scale.png/.jpg and plant_to_scale.json (the pixel boxes)."""
# The AI-made textures it reads are not in this repository.

import json, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

M = 0.65
px = lambda m: m / M                                              # noqa: E731
rng = np.random.default_rng(13)
AI = Image.open('plant_sat_raw.png').convert('RGB')
W, H = AI.size
A = np.asarray(AI).astype(np.float32)

# ---- the shore, found in the AI picture (first run of sea pixels right of x = 2000)
r, g, b = A[..., 0], A[..., 1], A[..., 2]
sea = (b > r + 25) & (g > r + 10)
shore = np.zeros(H, int)
for y in range(H):
    run = np.convolve(sea[y, 2000:].astype(int), np.ones(24, int), 'valid')
    idx = np.where(run >= 22)[0]
    shore[y] = int(idx[0]) + 2000 if len(idx) else W
shore = np.convolve(np.pad(shore, 40, mode='edge'), np.ones(81) / 81, 'valid').astype(int)   # smooth it

# ---- synthetic sand, matched to the AI picture's sand
def noise(scale):
    h, w = max(2, H // scale + 2), max(2, W // scale + 2)
    n = rng.normal(0, 1, (h, w)).astype(np.float32)
    return np.asarray(Image.fromarray(n).resize((W, H), Image.BICUBIC))
base = np.array([210.0, 175.0, 142.0], np.float32)
lum = 7.0 * noise(260) + 4.0 * noise(60) + 3.0 * noise(14) + 2.2 * rng.normal(0, 1, (H, W)).astype(np.float32)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
lum += 1.6 * np.sin((xx * 0.8 + yy * 0.35) / 5.5 + 2.0 * noise(120))               # faint wind ripples
sand = base[None, None, :] + lum[..., None] * np.array([1.0, 0.92, 0.85], np.float32)[None, None, :]
sand = np.clip(sand, 0, 255)

# ---- keep the AI beach and sea right of the shore minus ~70 m; blend over 40 px
keep = np.zeros((H, W), np.float32)
for y in range(H):
    x1 = shore[y] - 80
    keep[y, x1:] = 1.0
    keep[y, max(0, x1 - 40):x1] = np.linspace(0, 1, x1 - max(0, x1 - 40), endpoint=False)
out = sand * (1 - keep[..., None]) + A * keep[..., None]
img = Image.fromarray(out.astype(np.uint8))
d = ImageDraw.Draw(img, 'RGBA')

def tracks():                                                       # a few faint tyre tracks in the sand
    for k in range(7):
        x0, y0 = rng.uniform(0, 1900), rng.uniform(0, H)
        ang, pts = rng.uniform(0, math.pi), []
        for t in range(60):
            ang += rng.normal(0, 0.05); x0 += 9 * math.cos(ang); y0 += 9 * math.sin(ang); pts.append((x0, y0))
        d.line(pts, fill=(232, 205, 170, 70), width=3)
tracks()

# ---- the plant: hall, tank, pump building, by the pier the AI drew (pier base at x 2236, y 510)
hall_w, hall_h = px(15), px(90)
hall = (1820, 425, 1820 + hall_w, 425 + hall_h)
tank_r = px(75) / 2; tank_c = (1905, 680)
pump = (1985, 496, 1985 + px(30), 496 + px(17))
road = px(8)
field = (358, 75, 358 + px(math.sqrt(70400 / 1000 / 0.87 * 1e4)), 75 + px(math.sqrt(70400 / 1000 / 0.87 * 1e4)))
assert hall[0] - field[2] >= px(49), 'the field must sit about 50 m from the hall'

road_col = (196, 178, 150, 255)
def road_line(pts):
    d.line(pts, fill=(170, 152, 126, 255), width=int(road) + 4); d.line(pts, fill=road_col, width=int(road))
mid = (field[1] + field[3]) / 2
road_line([(field[0] - 30, field[3] + 22), (field[2] + 22, field[3] + 22), (field[2] + 22, field[1] - 22), (field[0] - 30, field[1] - 22), (field[0] - 30, field[3] + 22)])
road_line([(field[2] + 22, mid), (2085, mid)])
road_line([(2085, 0), (2085, 380), (2075, 900), (2100, H)])
road_line([(hall[2], hall[1] + 30), (2085, hall[1] + 30)])
road_line([(pump[0], pump[3] + 10), (2085, pump[3] + 10)])
# the intake pipe (buried, a faint line) from the pump building to the pier base
d.line([(pump[2], (pump[1] + pump[3]) / 2), (2236, 512)], fill=(120, 108, 92, 150), width=4)

# ---- the solar field: the AI panel rows, turned to run east-west, scaled to a 7.4 m pitch, tiled
P = Image.open('tex_panels.png').convert('RGB')
P = P.crop((15, 15, 668, 586)).rotate(90, expand=True)               # 19 whole rows (pitch 34.37 px), now running east-west
pitch_src = (668 - 15) / 19
s = px(7.4) / pitch_src
tile = P.resize((max(1, int(P.width * s)), max(1, int(P.height * s))), Image.LANCZOS)
fw, fh = int(field[2] - field[0]), int(field[3] - field[1])
F = Image.new('RGB', (fw, fh), (205, 172, 140))
for ty in range(0, fh, tile.height):                                # rows continue from tile to tile; a 3 px break between segments
    for tx in range(0, fw, tile.width + 3):
        F.paste(tile, (tx, ty))
gap = int(road)                                                     # service tracks: one east-west, one north-south
dF = ImageDraw.Draw(F)
dF.rectangle([0, fh // 2 - gap // 2, fw, fh // 2 + gap // 2], fill=(200, 178, 148))
dF.rectangle([fw // 2 - gap // 2, 0, fw // 2 + gap // 2, fh], fill=(200, 178, 148))
for qx in (0.25, 0.75):                                             # inverter cabins
    for qy in (0.25, 0.75):
        cx, cy = int(fw * qx), int(fh * qy)
        dF.rectangle([cx - 5 + 2, cy - 9 + 2, cx + 5 + 2, cy + 9 + 2], fill=(150, 130, 105))
        dF.rectangle([cx - 5, cy - 9, cx + 5, cy + 9], fill=(238, 238, 232), outline=(200, 200, 194))
img.paste(F, (int(field[0]), int(field[1])))
d.rectangle([field[0] - 8, field[1] - 8, field[2] + 8, field[3] + 8], outline=(150, 138, 118, 200), width=2)   # fence

# ---- the hall roof, the tank and the pump building from the AI picture, at our sizes, with shadows
def shadow(box, dx=-7, dy=3, r=4):
    sh = Image.new('L', (W, H), 0); ImageDraw.Draw(sh).rectangle([box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy], fill=110)
    sh = sh.filter(ImageFilter.GaussianBlur(r)); img.paste((70, 58, 46), (0, 0), sh)
roof = Image.open('tex_hall.png').convert('RGB').crop((44, 22, 285, 592)).resize((int(round(hall_w)), int(round(hall_h))), Image.LANCZOS)
shadow(hall, dx=-8, dy=2)
img.paste(roof, (int(hall[0]), int(hall[1])))
T = Image.open('tex_tank_pump.png').convert('RGB')
tank = T.crop((34, 172, 250, 378)).resize((int(2 * tank_r), int(2 * tank_r)), Image.LANCZOS)
mask = Image.new('L', tank.size, 0); ImageDraw.Draw(mask).ellipse([0, 0, tank.width - 1, tank.height - 1], fill=255)
sh = Image.new('L', (W, H), 0)
ImageDraw.Draw(sh).ellipse([tank_c[0] - tank_r - 9, tank_c[1] - tank_r + 3, tank_c[0] + tank_r - 9, tank_c[1] + tank_r + 3], fill=120)
img.paste((70, 58, 46), (0, 0), sh.filter(ImageFilter.GaussianBlur(5)))
img.paste(tank, (int(tank_c[0] - tank_r), int(tank_c[1] - tank_r)), mask)
pb = T.crop((37, 26, 188, 112)).resize((int(pump[2] - pump[0]), int(pump[3] - pump[1])), Image.LANCZOS)
shadow(pump, dx=-5, dy=2, r=3)
img.paste(pb, (int(pump[0]), int(pump[1])))

# ---- one light pass so the pieces sit in the same picture
img = img.filter(ImageFilter.GaussianBlur(0.35))
img.save('plant_to_scale.png'); img.save('plant_to_scale.jpg', quality=90)
img.resize((W // 2, H // 2), Image.LANCZOS).save('plant_to_scale_preview.jpg', quality=88)
json.dump({'m_per_px': M, 'W': W, 'H': H, 'field': field, 'hall': hall, 'tank_c': tank_c, 'tank_r': tank_r, 'pump': pump,
           'pier_base': [2236, 512], 'pier_end': [2500, 498]}, open('plant_to_scale.json', 'w'), indent=1)
print('field', [round(v) for v in field], 'side m', round((field[2] - field[0]) * M), '| hall', [round(v) for v in hall],
      '| tank r m', round(tank_r * M, 1), '| pump m', round((pump[2] - pump[0]) * M), 'x', round((pump[3] - pump[1]) * M))
