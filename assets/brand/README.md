# rppgkit brand assets

Pixel-heart logo (concept C1). The mark is a 5×4 heart built from rounded pixels; one lighter pixel stands for the subtle skin-colour change rPPG measures.

## Contents

```
rppgkit-brand/
├── svg/
│   ├── rppgkit-logo.svg          # long version, for light backgrounds
│   ├── rppgkit-logo-dark.svg     # long version, for dark backgrounds
│   ├── rppgkit-mark.svg          # short version, for light backgrounds
│   ├── rppgkit-mark-dark.svg     # short version, for dark backgrounds
│   ├── rppgkit-mark-white.svg    # short version, single colour, on coloured/photo backgrounds
│   └── rppgkit-mark-black.svg    # short version, single colour, for print / monochrome
├── png/
│   ├── rppgkit-logo@1x.png / @2x.png         # 80 / 160 px tall
│   ├── rppgkit-logo-dark@1x.png / @2x.png
│   └── rppgkit-mark-{16,32,48,64,128,256,512}.png   # 16 px is pixel-snapped
├── favicon/
│   ├── favicon.ico               # 16 + 32 + 48
│   ├── favicon.svg
│   ├── apple-touch-icon.png      # 180, white background
│   └── icon-192.png / icon-512.png
├── social/
│   └── social-preview.png        # 1280×640, GitHub / link previews
└── tools/
    └── build_brand.py            # regenerates everything from code
```

All SVGs are self-contained: the wordmark is converted to paths, so no font is needed to display them.

## Where to put it in the repo

```
rppgkit/
├── assets/brand/        ← copy svg/, png/, favicon/, social/ here
├── docs/_static/        ← copy the files your docs theme needs (see below)
└── tools/build_brand.py ← optional, to regenerate assets
```

## README header

Switches between the light and dark logo with the viewer's GitHub theme:

```html
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/brand/svg/rppgkit-logo-dark.svg">
    <img src="assets/brand/svg/rppgkit-logo.svg" alt="rppgkit" width="360">
  </picture>
</p>
```

**PyPI:** relative paths don't resolve on pypi.org. If `README.md` is also your `long_description`, use absolute URLs to a tagged version, e.g. `https://raw.githubusercontent.com/<user>/rppgkit/v0.1.0/assets/brand/png/rppgkit-logo@2x.png`. PyPI may not render `<picture>`; the `<img>` fallback (light logo) shows instead, which is fine on PyPI's white page.

## GitHub repository

- **Social preview:** Settings → General → Social preview → upload `social/social-preview.png`.
- **Organisation / user avatar:** upload `png/rppgkit-mark-512.png`.

## Documentation sites

**Sphinx – Furo**
```python
# docs/conf.py
html_static_path = ["_static"]
html_favicon = "_static/favicon.ico"
html_theme_options = {
    "light_logo": "rppgkit-logo.svg",
    "dark_logo": "rppgkit-logo-dark.svg",
    "sidebar_hide_name": True,
}
```

**Sphinx – PyData theme**
```python
html_favicon = "_static/favicon.ico"
html_theme_options = {
    "logo": {
        "image_light": "_static/rppgkit-logo.svg",
        "image_dark": "_static/rppgkit-logo-dark.svg",
    }
}
```

**MkDocs – Material**
```yaml
# mkdocs.yml
theme:
  name: material
  logo: assets/rppgkit-mark-white.svg   # header bar is coloured, so use the white mark
  favicon: assets/favicon.svg
```

**Any HTML page**
```html
<link rel="icon" href="/favicon.ico" sizes="any">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="apple-touch-icon" href="/apple-touch-icon.png">
```

## Brand basics

| Role | Colour |
|---|---|
| Oxblood (primary, on light) | `#9F1D35` |
| Oxblood, lifted (on dark) | `#D9506A` |
| Signal pixel | `#DEB0B9` |
| Wordmark on light | `#2B2F36` |
| Wordmark on dark | `#F4F5F7` |
| Dark background | `#111418` |

- **Typeface:** Inter Display SemiBold, lowercase, −2% tracking (SIL Open Font License). Use Inter for docs and slides to match.
- **Dark backgrounds:** always use the `-dark` files. Plain oxblood is too dark to read on near-black; the lifted shade keeps the same hue.
- **Short vs long:** use the long version (mark + wordmark) where there is room — README, docs header, slides. Use the short version (mark only) for favicons, avatars, small UI and when the name already appears next to it.
- **Clear space:** keep at least one pixel-cell (16% of the mark's height) of empty space around the logo.
- **Minimum size:** mark 16 px (use `rppgkit-mark-16.png`, which is pixel-snapped); long version 24 px tall.
- **Don't:** recolour individual pixels, change the signal pixel's position, add outlines or shadows, stretch, or set the wordmark in another font.

## Regenerating

```bash
pip install fonttools pillow
RPPGKIT_FONT=/path/to/InterDisplay-SemiBold.otf \
RPPGKIT_FONT_TAG=/path/to/Inter-Regular.otf \
python tools/build_brand.py assets/brand
```

Colours, pixel layout and wordmark spacing are constants at the top of the script.
