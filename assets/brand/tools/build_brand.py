"""Rebuild the rppgkit brand assets (pixel-heart logo).

Usage:
    pip install fonttools pillow
    RPPGKIT_FONT=/path/InterDisplay-SemiBold.otf RPPGKIT_FONT_TAG=/path/Inter-Regular.otf \
        python tools/build_brand.py <output_dir> [contact_sheet.png]

Fonts: Inter Display SemiBold (wordmark) and Inter Regular (tagline),
SIL Open Font License, https://github.com/rsms/inter
"""
import os
import sys

from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from PIL import Image, ImageDraw, ImageFont

OUT = sys.argv[1]
FONT = os.environ.get("RPPGKIT_FONT", "/usr/share/fonts/opentype/inter/InterDisplay-SemiBold.otf")
FONT_TAG = os.environ.get("RPPGKIT_FONT_TAG", "/usr/share/fonts/opentype/inter/Inter-Regular.otf")

RED = "#9F1D35"       # oxblood, primary
RED_DARK = "#D9506A"  # lifted oxblood for dark backgrounds
INK = "#2B2F36"
PAPER = "#F4F5F7"
DARK = "#111418"
MUTED = "#5B616B"
SIGNAL = "#DEB0B9"    # solid tint, used for the signal pixel in colour versions

# 5x4 pixel heart in a 100x100 box: (x, y, is_signal_pixel)
CELLS = [
    (4, 14, 0), (23, 14, 0), (61, 14, 0), (80, 14, 0),
    (4, 33, 0), (23, 33, 0), (42, 33, 0), (61, 33, 1), (80, 33, 0),
    (23, 52, 0), (42, 52, 0), (61, 52, 0),
    (42, 71, 0),
]
CELL, RADIUS, SIGNAL_OPACITY = 16, 3, 0.4

# Wordmark geometry, in units of the 100-unit mark box
WORD = "rppgkit"
WORD_SIZE = 84          # font size
WORD_GAP = 16           # mark box edge to first letter
TRACKING = -0.02        # em
MARK_CENTER_Y = 50.5    # vertical centre of the heart's pixels

font = TTFont(FONT)
UPM = font["head"].unitsPerEm
XH = font["OS/2"].sxHeight
GLYPHS = font.getGlyphSet()
CMAP = font.getBestCmap()
HMTX = font["hmtx"]


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def ensure(*parts):
    p = os.path.join(OUT, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


# ---------- geometry shared by SVG and PNG ----------

def word_layout(size):
    """Return [(char, x_offset)] and total advance for the wordmark at `size`."""
    s = size / UPM
    x, out = 0.0, []
    for ch in WORD:
        out.append((ch, x))
        x += HMTX[CMAP[ord(ch)]][0] * s + TRACKING * size
    return out, x - TRACKING * size


def baseline_for(size):
    return MARK_CENTER_Y + (XH * size / UPM) / 2


def lockup_width():
    _, adv = word_layout(WORD_SIZE)
    return 100 + WORD_GAP + adv + 4


# ---------- SVG ----------

def svg_cells(fill, signal_fill=None, signal_opacity=SIGNAL_OPACITY, dx=0):
    rects = []
    for x, y, sig in CELLS:
        attrs = f'x="{x + dx:g}" y="{y}" width="{CELL}" height="{CELL}" rx="{RADIUS}"'
        if sig and fill in (RED, RED_DARK):
            rects.append(f'<rect {attrs} fill="{SIGNAL}"/>')
        elif sig:
            rects.append(f'<rect {attrs} fill="{fill}" fill-opacity="{signal_opacity}"/>')
        else:
            rects.append(f'<rect {attrs} fill="{fill}"/>')
    return "".join(rects)


def svg_word_path(x0):
    s = WORD_SIZE / UPM
    base = baseline_for(WORD_SIZE)
    d = []
    for ch, xo in word_layout(WORD_SIZE)[0]:
        pen = SVGPathPen(GLYPHS)
        GLYPHS[CMAP[ord(ch)]].draw(TransformPen(pen, (s, 0, 0, -s, x0 + xo, base)))
        d.append(pen.getCommands())
    return " ".join(d)


def write_svg(path, body, w, h, title):
    with open(ensure(path), "w") as f:
        f.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:g} {h:g}" '
            f'role="img" aria-labelledby="t"><title id="t">{title}</title>{body}</svg>\n'
        )


def build_svgs():
    write_svg("svg/rppgkit-mark.svg", svg_cells(RED), 100, 100, "rppgkit")
    write_svg("svg/rppgkit-mark-dark.svg", svg_cells(RED_DARK), 100, 100, "rppgkit")
    write_svg("svg/rppgkit-mark-white.svg", svg_cells("#FFFFFF"), 100, 100, "rppgkit")
    write_svg("svg/rppgkit-mark-black.svg", svg_cells("#000000"), 100, 100, "rppgkit")
    w = round(lockup_width(), 2)
    word = svg_word_path(100 + WORD_GAP)
    write_svg("svg/rppgkit-logo.svg",
              svg_cells(RED) + f'<path fill="{INK}" d="{word}"/>', w, 100, "rppgkit")
    write_svg("svg/rppgkit-logo-dark.svg",
              svg_cells(RED_DARK) + f'<path fill="{PAPER}" d="{word}"/>', w, 100, "rppgkit")
    write_svg("favicon/favicon.svg", svg_cells(RED), 100, 100, "rppgkit")


# ---------- PNG ----------

SS = 4  # supersampling


def draw_mark(img, ox, oy, box, color=RED):
    """Draw the heart into RGBA `img` with its 100-unit box at (ox, oy), `box` px wide."""
    big = Image.new("RGBA", (box * SS, box * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    k = box * SS / 100
    r, g, b = hex_rgb(color)
    for x, y, sig in CELLS:
        fill = (r, g, b, 255)
        if sig:
            fill = hex_rgb(SIGNAL) + (255,) if color in (RED, RED_DARK) else (r, g, b, round(255 * SIGNAL_OPACITY))
        d.rounded_rectangle([x * k, y * k, (x + CELL) * k - 1, (y + CELL) * k - 1],
                            radius=RADIUS * k, fill=fill)
    small = big.resize((box, box), Image.LANCZOS)
    img.alpha_composite(small, (round(ox), round(oy)))


def draw_mark_hinted(size, color=RED):
    """Pixel-snapped heart for 16 and 32 px favicons."""
    cell, gap = {16: (2, 1)}[size]
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    gw, gh = 5 * cell + 4 * gap, 4 * cell + 3 * gap
    ox, oy = (size - gw) // 2 + (size - gw) % 2, (size - gh) // 2 + 1
    r, g, b = hex_rgb(color)
    for x, y, sig in CELLS:
        c, row = round((x - 4) / 19), round((y - 14) / 19)
        fill = hex_rgb(SIGNAL) + (255,) if sig else (r, g, b, 255)
        x0, y0 = ox + c * (cell + gap), oy + row * (cell + gap)
        d.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], fill=fill)
    return img


def draw_lockup(img, ox, oy, height, text_color, mark_color=RED):
    """Mark + wordmark; `height` is the px height of the 100-unit box."""
    k = height / 100
    draw_mark(img, ox, oy, round(height), mark_color)
    layer = Image.new("RGBA", (img.width * SS, img.height * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    size = WORD_SIZE * k
    pil_font = ImageFont.truetype(FONT, round(size * SS))
    x0 = ox + (100 + WORD_GAP) * k
    base = oy + baseline_for(WORD_SIZE) * k
    for ch, xo in word_layout(size)[0]:
        d.text(((x0 + xo) * SS, base * SS), ch, font=pil_font,
               fill=hex_rgb(text_color) + (255,), anchor="ls")
    img.alpha_composite(layer.resize(img.size, Image.LANCZOS))


def build_pngs():
    for s in (32, 48, 64, 128, 256, 512):
        img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        draw_mark(img, 0, 0, s)
        img.save(ensure(f"png/rppgkit-mark-{s}.png"))
    for s in (16,):
        draw_mark_hinted(s).save(ensure(f"png/rppgkit-mark-{s}.png"))

    for h, tag in ((80, "1x"), (160, "2x")):
        w = round(lockup_width() * h / 100)
        for name, col, mc in (("logo", INK, RED), ("logo-dark", PAPER, RED_DARK)):
            img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            draw_lockup(img, 0, 0, h, col, mc)
            img.save(ensure(f"png/rppgkit-{name}@{tag}.png"))

    # favicon set
    i32 = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    draw_mark(i32, 0, 0, 32)
    ico = [draw_mark_hinted(16), i32]
    i48 = Image.new("RGBA", (48, 48), (0, 0, 0, 0))
    draw_mark(i48, 0, 0, 48)
    ico.append(i48)
    ico[2].save(ensure("favicon/favicon.ico"), format="ICO",
                sizes=[(16, 16), (32, 32), (48, 48)], append_images=ico[:2])
    for s, name in ((180, "apple-touch-icon.png"), (192, "icon-192.png"), (512, "icon-512.png")):
        img = Image.new("RGBA", (s, s), (255, 255, 255, 255))
        pad = round(s * 0.14)
        draw_mark(img, pad, pad, s - 2 * pad)
        img.convert("RGB").save(ensure(f"favicon/{name}"))

    # GitHub social preview (1280x640, keep content in the central safe area)
    W, H = 1280, 640
    img = Image.new("RGBA", (W, H), hex_rgb("#FFFFFF") + (255,))
    lh = 230
    lw = lockup_width() * lh / 100
    draw_lockup(img, (W - lw) / 2, 160, lh, INK)
    d = ImageDraw.Draw(img)
    tag = ImageFont.truetype(FONT_TAG, 34)
    d.text((W / 2, 470), "Remote photoplethysmography toolkit for Python",
           font=tag, fill=hex_rgb(MUTED), anchor="ms")
    img.convert("RGB").save(ensure("social/social-preview.png"))


def build_contact_sheet():
    """Preview of everything, for review only (not part of the pack)."""
    sheet = Image.new("RGBA", (1400, 900), (255, 255, 255, 255))
    dark = Image.new("RGBA", (1400, 300), hex_rgb(DARK) + (255,))
    sheet.alpha_composite(dark, (0, 600))
    for i, s in enumerate((512, 128, 64, 48, 32, 16)):
        im = Image.open(os.path.join(OUT, f"png/rppgkit-mark-{s}.png")).convert("RGBA")
        if s == 512:
            im = im.resize((260, 260), Image.LANCZOS)
        sheet.alpha_composite(im, (40 + [0, 300, 450, 540, 610, 670][i], 40))
    big16 = Image.open(os.path.join(OUT, "png/rppgkit-mark-16.png")).resize((128, 128), Image.NEAREST)
    big32 = Image.open(os.path.join(OUT, "png/rppgkit-mark-32.png")).resize((128, 128), Image.NEAREST)
    sheet.alpha_composite(big16.convert("RGBA"), (760, 40))
    sheet.alpha_composite(big32.convert("RGBA"), (920, 40))
    lg = Image.open(os.path.join(OUT, "png/rppgkit-logo@2x.png"))
    sheet.alpha_composite(lg, (40, 360))
    ld = Image.open(os.path.join(OUT, "png/rppgkit-logo-dark@2x.png"))
    sheet.alpha_composite(ld, (40, 670))
    soc = Image.open(os.path.join(OUT, "social/social-preview.png")).convert("RGBA").resize((480, 240))
    sheet.alpha_composite(soc, (880, 330))
    sheet.convert("RGB").save(sys.argv[2])


if __name__ == "__main__":
    build_svgs()
    build_pngs()
    if len(sys.argv) > 2:
        build_contact_sheet()
    print("done")
