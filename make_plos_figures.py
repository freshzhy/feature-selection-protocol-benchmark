#!/usr/bin/env python3
"""make_plos_figures.py -- renders every manuscript figure to the PLOS ONE
specification by re-running the existing figure scripts with a patched
Figure.savefig:

  * TIFF, LZW compression, RGB (no alpha), 300 dpi, one page
  * width <= 7.5 in (2250 px), height <= 8.75 in
  * no titles inside the figure (PLOS: titles/numbers/captions belong in the
    manuscript); all text 8-12 pt in a sans-serif font
  * file names Fig1.tif ... Fig12.tif and S1_Fig.tif (S1 Fig = the leaky-arm
    critical-difference diagram, formerly Fig. 2b)

Nothing is recomputed: the scripts read the same results/*.csv as before.
Mapping old file -> new label is the one used in plos/manuscript.md.
"""
import io
import runpy
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Text
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "plos" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
MAP = {"cd_nested_20": "Fig2", "cd_leaky_20": "S1_Fig"}  # v2: only the CD diagrams; the rest is make_plos_figures_v2.py
DONE = {}
FONT = ["Liberation Sans", "DejaVu Sans"]
plt.rcParams["font.family"] = FONT
from matplotlib import font_manager as _fm
for _f in (ROOT / "fonts").glob("*.ttf"):
    _fm.fontManager.addfont(str(_f))
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["savefig.bbox"] = "standard"

Axes.set_title = lambda self, *a, **k: None
Figure.suptitle = lambda self, *a, **k: None
_orig = Figure.savefig


def savefig(self, fname, *a, **k):
    stem = Path(str(fname)).stem
    if stem not in MAP:
        return None
    for t in self.findobj(Text):
        t.set_fontfamily(FONT)
        t.set_fontsize(min(12.0, max(8.0, t.get_fontsize())))
    w, h = self.get_size_inches()
    if w > 7.5:
        self.set_size_inches(7.5, h * 7.5 / w)
    try:
        self.tight_layout()
    except Exception:
        pass
    buf = io.BytesIO()
    with matplotlib.rc_context({"savefig.bbox": "standard"}):
        _orig(self, buf, format="png", dpi=300, bbox_inches=None)
    buf.seek(0)
    im = Image.open(buf).convert("RGB")
    path = OUT / f"{MAP[stem]}.tif"
    im.save(path, format="TIFF", compression="tiff_lzw", dpi=(300, 300))
    DONE[MAP[stem]] = im.size
    return None


Figure.savefig = savefig
sys.path.insert(0, str(ROOT))
for script in ("run_statistics_v4.py",):
    try:
        runpy.run_path(str(ROOT / script), run_name="__main__")
    except SystemExit:
        pass
    plt.close("all")
missing = sorted(set(MAP.values()) - set(DONE))
for k in sorted(DONE, key=lambda s: (s.startswith("S"), int("".join(c for c in s if c.isdigit()) or 0))):
    w, h = DONE[k]
    ok = 789 <= w <= 2250 and h <= 2625
    print(f"{k:8s} {w}x{h}px {w/300:.2f}x{h/300:.2f}in {'OK' if ok else 'OUT OF RANGE'}")
print("missing:", missing)
