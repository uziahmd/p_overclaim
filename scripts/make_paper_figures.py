"""Draw the paper's figures from the committed records (paper/figures/*.pdf, vector) and write the Figure 1 example
(paper/figures/fig1_example.tex).

Main text: fig1_example (LaTeX: what the agent did, the three messages of one matched set, and Sonnet 5.5's scores
without and with the transcript), fig2_premium (the premium of every LLM judge without and with the transcript; the
reward models in standard deviations), fig3_selection (best-of-16: the share of selected messages that overclaim at
N = 1 and N = 16), fig4_side_by_side (the side-by-side preference against the premium when scored alone, drawn at
60% of the text width). Appendix: figB1_mean_scores (the mean score of the overclaim, the honest message and the
placebo for every LLM judge without and with the transcript; its values are also written to figB1_mean_scores.csv).

Every plotted value and every count printed in a figure is read from results/full_run/summary.json,
results/full_run/runs_status.json, paper/derived.json and the frozen set files through scripts/paper_data.py.

Design rules: blue #2a78d6 = without the transcript (filled markers), orange #eb6834 = with the transcript (hollow
markers), in every figure; all text in ink #333333; legends with the real markers above the axes; no notes inside the
plot area; no top or right spine; light x grid; thin CI whiskers; one fixed grader order (D.LLM_JUDGES, then
D.REWARD_MODELS). Each PDF is exactly the size it is drawn at (no tight bounding box) and is included unscaled: the
font sizes in the PDF are final (8 pt, never below 7.5 pt). Font: Latin Modern Roman, the paper's font. The OpenType
files (CFF outlines) are converted once to TrueType outlines in a cache directory outside the repository, because
matplotlib embeds fonttype-42 fonts as TrueType and a CFF font embedded that way gives a broken text layer;
STIXGeneral is the fallback when the files are not found.

Run from the repository root: uv run --no-sync python scripts/make_paper_figures.py
"""

from __future__ import annotations

import csv
import importlib.util
import math
import os
import re
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.text as mtext  # noqa: E402
import matplotlib.transforms as mtransforms  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402
from matplotlib.ticker import FixedLocator, MaxNLocator  # noqa: E402

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("paper_data", _HERE / "paper_data.py")
D = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(D)

OUT = D.PAPER / "figures"
TEXT_W = 6.5                       # the document's text width (in); a full-width figure is exactly this wide
BLUE, ORANGE, INK = "#2a78d6", "#eb6834", "#333333"
GRID, ZERO, RULE, LINK = "#e6e6e6", "#c4c4c4", "#8c8c8c", "#b8b8b8"   # lines only, never text
FS, SMALL = 8.0, 7.5               # base and smallest font sizes (pt)
MS = 4.6                           # marker size (pt)
HEADER = 1.0                       # a group label takes one row slot, so every row of a figure sits on one grid


# ---------------------------------------------------------------- fonts

_FONT_DIRS = (Path.home() / ".cache" / "tectonic", Path("/usr/share/fonts"), Path("/usr/local/share/fonts"),
              Path.home() / ".local" / "share" / "fonts", Path.home() / ".fonts",
              Path("/usr/share/texmf/fonts/opentype"), Path("/usr/share/texlive/texmf-dist/fonts/opentype"),
              Path("/usr/local/texlive"))
_LM_FILES = ("lmroman10-regular.otf", "lmroman10-bold.otf", "lmroman10-italic.otf")


def _find_latin_modern() -> list[Path] | None:
    found: dict[str, Path] = {}
    for root in _FONT_DIRS:
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("lmroman10-*.otf")):
            if p.name in _LM_FILES and p.name not in found:
                found[p.name] = p
        if len(found) == len(_LM_FILES):
            break
    return [found[n] for n in _LM_FILES] if len(found) == len(_LM_FILES) else None


def _font_cache() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "overclaim-paper-fonts"
    try:
        base.mkdir(parents=True, exist_ok=True)
        return base
    except OSError:
        tmp = Path(tempfile.gettempdir()) / "overclaim-paper-fonts"
        tmp.mkdir(parents=True, exist_ok=True)
        return tmp


def _otf_to_ttf(src: Path, dst: Path) -> None:
    """Rewrite an OpenType-CFF font with quadratic TrueType outlines (fontTools cu2qu), so that matplotlib's fonttype-42
    embedding (CIDFontType2 / FontFile2) is valid and the PDF text layer copies correctly."""
    from fontTools.pens.cu2quPen import Cu2QuPen
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    from fontTools.ttLib import TTFont, newTable

    font = TTFont(str(src))
    order = font.getGlyphOrder()
    glyph_set = font.getGlyphSet()
    glyphs = {}
    for g in order:
        pen = TTGlyphPen(glyph_set)
        glyph_set[g].draw(Cu2QuPen(pen, 1.0, reverse_direction=True))
        glyphs[g] = pen.glyph()
    font["loca"] = newTable("loca")
    font["glyf"] = glyf = newTable("glyf")
    glyf.glyphOrder = order
    glyf.glyphs = glyphs
    del font["CFF "]
    if "VORG" in font:
        del font["VORG"]
    glyf.compile(font)
    hmtx = font["hmtx"]
    for g in order:
        advance, _ = hmtx[g]
        hmtx[g] = (advance, glyf[g].xMin if glyf[g].numberOfContours else 0)
    maxp = newTable("maxp")
    maxp.tableVersion = 0x00010000
    maxp.maxZones = 1
    for key in ("maxTwilightPoints", "maxStorage", "maxFunctionDefs", "maxInstructionDefs", "maxStackElements",
                "maxSizeOfInstructions", "maxComponentElements"):
        setattr(maxp, key, 0)
    font["maxp"] = maxp
    post = font["post"]
    post.formatType, post.extraNames, post.mapping, post.glyphOrder = 2.0, [], {}, order
    font.sfntVersion = "\x00\x01\x00\x00"
    tmp = dst.with_suffix(".tmp")
    font.save(str(tmp))
    tmp.replace(dst)


def setup_fonts() -> str:
    """Register Latin Modern Roman (as TrueType) and return the family used; STIXGeneral when that fails."""
    try:
        sources = _find_latin_modern()
        if sources is None:
            raise FileNotFoundError("lmroman10 OpenType files not found")
        cache = _font_cache()
        family = None
        for src in sources:
            dst = cache / (src.stem + ".ttf")
            if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
                _otf_to_ttf(src, dst)
            font_manager.fontManager.addfont(str(dst))
            family = font_manager.FontProperties(fname=str(dst)).get_name()
        assert family
        return family
    except Exception as exc:   # any failure: the bundled STIX fonts
        print(f"make_paper_figures: Latin Modern not used ({exc}); falling back to STIXGeneral", file=sys.stderr)
        return "STIXGeneral"


FAMILY = setup_fonts()
plt.rcParams.update({
    "font.family": [FAMILY, "STIXGeneral", "DejaVu Sans"], "mathtext.fontset": "stix",
    "axes.formatter.use_mathtext": False, "axes.unicode_minus": True, "font.size": FS, "axes.labelsize": FS,
    "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS, "axes.titlesize": FS,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": RULE, "axes.linewidth": 0.6,
    "xtick.color": RULE, "ytick.color": RULE, "xtick.labelcolor": INK, "ytick.labelcolor": INK, "text.color": INK,
    "axes.labelcolor": INK, "pdf.fonttype": 42, "ps.fonttype": 42, "lines.linewidth": 0.8,
    "xtick.major.size": 2.5, "ytick.major.size": 0, "xtick.major.width": 0.5, "xtick.major.pad": 2.0,
    "ytick.major.pad": 3.0, "axes.labelpad": 2.5, "legend.frameon": False,
})


# ---------------------------------------------------------------- helpers


def name(g: str) -> str:
    """The display name of a grader; the 4-bit note of two reward models is left to the captions."""
    return D.NAMES.get(g, g).replace(" (4-bit)", "")


def new_fig(w: float, h: float):
    return plt.figure(figsize=(w, h))


def axes_in(fig, left: float, bottom: float, width: float, height: float):
    """An axes placed in inches from the figure's bottom-left corner."""
    fw, fh = fig.get_size_inches()
    return fig.add_axes((left / fw, bottom / fh, width / fw, height / fh))


def inch_x(fig, ax):
    """Transform: x in inches from the figure's left edge, y in the axes' data coordinates."""
    return mtransforms.blended_transform_factory(fig.dpi_scale_trans, ax.transData)


def style(ax, zero: bool = True, grid_axis: str | None = "x") -> None:
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    if zero:
        ax.axvline(0, color=ZERO, linewidth=0.9, zorder=1.2)     # above the grid line at zero, below the data
    ax.tick_params(axis="x", length=2.5, width=0.5, colors=RULE, labelcolor=INK, labelsize=FS)
    ax.tick_params(axis="y", length=0, colors=RULE, labelcolor=INK)


def rows_axes(ax) -> None:
    """A category axis: no y ticks or y spine (the row names are drawn as text)."""
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)


def whisker(ax, y, lo, hi, color, lw: float = 0.8, zorder: float = 2) -> None:
    ax.plot([lo, hi], [y, y], color=color, linewidth=lw, solid_capstyle="butt", zorder=zorder)


def dot(ax, x, y, color, filled: bool, marker: str = "o", size: float = MS, zorder: float = 3) -> None:
    ax.plot([x], [y], marker=marker, markersize=size, markerfacecolor=color if filled else "white",
            markeredgecolor=color, markeredgewidth=0.9, linestyle="none", zorder=zorder)


def handle(color, filled: bool, marker: str = "o", line: bool = True) -> Line2D:
    """A legend handle drawn with the real marker (and its CI whisker)."""
    return Line2D([], [], color=color, linewidth=0.8 if line else 0, marker=marker, markersize=MS,
                  markerfacecolor=color if filled else "white", markeredgecolor=color, markeredgewidth=0.9)


def legend_in(fig, handles, labels, x: float, y: float, ncol: int | None = None, loc: str = "upper left",
              size: float = SMALL, handlelength: float = 1.5, columnspacing: float = 1.4):
    """A legend anchored in inches from the figure's bottom-left corner, outside every axes."""
    return fig.legend(handles, labels, loc=loc, bbox_to_anchor=(x, y), bbox_transform=fig.dpi_scale_trans,
                      ncol=ncol or len(handles), frameon=False, fontsize=size, handlelength=handlelength,
                      handletextpad=0.5, columnspacing=columnspacing, borderaxespad=0.0, borderpad=0.0,
                      labelcolor=INK)


def ftext(fig, x: float, y: float, s: str, **kw):
    """Text placed in inches from the figure's bottom-left corner."""
    kw.setdefault("fontsize", SMALL)
    kw.setdefault("color", INK)
    return fig.text(x, y, s, transform=fig.dpi_scale_trans, **kw)


def pct_ticks(ax, ticks) -> None:
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.set_xticklabels([f"{t:g}%" for t in ticks])


def row_layout(groups: list[tuple[str | None, list]], header: float = HEADER) -> tuple[list, list, float]:
    """Row positions (negative y, top row first) for groups of rows, each group under a label slot; returns
    (rows as (item, y), headers as (label, y), the bottom y)."""
    rows, headers, cur = [], [], 0.0
    for label, items in groups:
        if label is not None:
            headers.append((label, cur - header / 2))
            cur -= header
        for it in items:
            rows.append((it, cur - 0.5))
            cur -= 1.0
    return rows, headers, cur


# ---------------------------------------------------------------- layout check (run on every figure before saving)

BANNED = re.compile(r"\b(hidden|shown|pointwise|pairwise|pick\w*|pool\w*|tax|bonus|dose|filler|band|records|rows)\b"
                    r"|N\s*=\s*2\b", re.IGNORECASE)
PROBLEMS: list[str] = []


def _layout_texts(fig) -> list:
    texts = list(fig.texts)
    for leg in fig.legends:
        texts += leg.get_texts()
    for ax in fig.axes:
        texts += ax.texts
        if ax.get_legend():
            texts += ax.get_legend().get_texts()
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            if not axis.get_visible():
                continue
            lo, hi = sorted(lim)
            span = hi - lo
            for loc, lab in zip(axis.get_majorticklocs(), axis.get_majorticklabels()):
                if lo - 1e-6 * span <= loc <= hi + 1e-6 * span:
                    texts.append(lab)
        texts += [ax.xaxis.label, ax.yaxis.label, ax.title, ax._left_title, ax._right_title]
    return [t for t in texts if t.get_visible() and t.get_text().strip()]


def check_layout(fig, stem: str) -> list[str]:
    """Every text inside the canvas, at least SMALL pt, free of the banned words, and overlapping no other text."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    w, h = fig.bbox.width, fig.bbox.height
    out = []
    boxes = []
    for t in _layout_texts(fig):
        s = t.get_text()
        if t.get_fontsize() < SMALL - 1e-6:
            out.append(f"{stem}: {s!r} is {t.get_fontsize()} pt")
        if BANNED.search(s):
            out.append(f"{stem}: banned word in {s!r}")
        bb = mtext.Text.get_window_extent(t, r)        # the text alone (an annotation's extent adds its leader)
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > w + 0.5 or bb.y1 > h + 0.5:
            out.append(f"{stem}: {s!r} falls outside the canvas")
        boxes.append((s, bb))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if a.x0 < b.x1 - 0.5 and b.x0 < a.x1 - 0.5 and a.y0 < b.y1 - 0.5 and b.y0 < a.y1 - 0.5:
                out.append(f"{stem}: {boxes[i][0]!r} overlaps {boxes[j][0]!r}")
    return out


def save(fig, stem: str) -> None:
    problems = check_layout(fig, stem)
    for p in problems:
        print("layout:", p, file=sys.stderr)
    PROBLEMS.extend(problems)
    OUT.mkdir(parents=True, exist_ok=True)
    # no bbox_inches: the page is exactly the figure size, so the PDF is included unscaled
    fig.savefig(OUT / f"{stem}.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


# ---------------------------------------------------------------- Figure 2

GROUP_NAMES = {"Anthropic": "Claude", "Google": "Gemini"}   # every other lab's judges are open-weight models


def judge_groups() -> list[tuple[str, list[str]]]:
    """The LLM judges in the fixed order (D.LLM_JUDGES), grouped: Claude, Gemini, Open-weight."""
    groups: list[tuple[str, list[str]]] = []
    for g in D.LLM_JUDGES:
        label = GROUP_NAMES.get(D.LAB[g], "Open-weight")
        if not groups or groups[-1][0] != label:
            groups.append((label, []))
        groups[-1][1].append(g)
    return groups


def with_transcript_point(s: dict, g: str, status: dict) -> dict:
    """The with-transcript premium of a judge for Figure 2: the all-60-task value of paper/derived.json where there is
    one (Sonnet 5.5, DeepSeek V4 Pro), else the core row. kind: 'drawn', 'not estimated' (fewer than D.MIN_TASKS
    tasks) or 'not run'; partial: some run behind the value is not done."""
    runs = [rid for (gg, cond), rid in D.POINTWISE_RUNS.items() if gg == g and cond.startswith("shown")]
    partial = any(D.cell_status(status, rid)["status"] != "done" for rid in runs)
    wt = D.derived()["with_transcript_all_tasks"]
    if g in wt:
        v = wt[g]
        p = {"premium": v["premium"], "ci": v["ci"], "n_tasks": v["n_tasks"]}
    else:
        r = D.core_row(s, g, "shown")
        if r is None:
            return {"kind": "not run", "n_tasks": None, "partial": False}
        p = {"premium": r["premium"], "ci": r["premium_ci"], "n_tasks": r["n_tasks"]}
    p["kind"] = "drawn" if p["n_tasks"] >= D.MIN_TASKS else "not estimated"
    p["partial"] = partial
    return p


def fig2_premium(s: dict) -> None:
    status = D.runs_status()
    W, H = TEXT_W, 2.45
    fig = new_fig(W, H)
    groups = judge_groups()
    rows, headers, bottom = row_layout(groups)
    ylo, yhi = bottom - 0.2, 0.15
    top_in, bot_in = 0.43, 0.37
    ax_h = H - top_in - bot_in
    unit = ax_h / (yhi - ylo)                      # inches per row unit, shared with panel (b)
    a_left, a_w = 0.98, 2.84
    col_right = a_left + a_w + 0.58                # right edge of the tasks column (0.3 in clear of panel (b))
    ax = axes_in(fig, a_left, bot_in, a_w, ax_h)
    dy = 0.25                                      # the two tracks: without above the row centre, with below
    xs = []
    tasks = []
    for g, y in rows:
        hid = D.core_row(s, g, "hidden")
        w = with_transcript_point(s, g, status)
        if w["kind"] == "drawn":
            ax.plot([hid["premium"], w["premium"]], [y + dy, y - dy], color=LINK, linewidth=0.7, zorder=1)
            whisker(ax, y - dy, *w["ci"], ORANGE)
            dot(ax, w["premium"], y - dy, ORANGE, False, zorder=3.5)      # the ring stays whole where the two dots meet
            xs += list(w["ci"])
        whisker(ax, y + dy, *hid["premium_ci"], BLUE)
        dot(ax, hid["premium"], y + dy, BLUE, True)
        xs += list(hid["premium_ci"])
        if w["kind"] == "not run":
            right = "–"
        elif w["kind"] == "not estimated":
            right = "‡"       # too few tasks to estimate (the dagger marks pre-registered tests in the selection figure)
        else:
            right = f"{w['n_tasks']}" + ("*" if w["partial"] else "")
        tasks.append((y, f"{hid['n_tasks']} / {right}"))
    lo, hi = min(min(xs), -0.5) - 0.15, max(xs) + 0.15
    ax.set_xlim(lo, hi)
    ax.set_ylim(ylo, yhi)
    ax.xaxis.set_major_locator(FixedLocator(list(range(math.ceil(lo), math.floor(hi) + 1))))
    rows_axes(ax)
    style(ax)
    ax.set_xlabel("premium: score of the overclaim minus the honest message (points, 1 to 10 scale)", fontsize=FS)
    tr = inch_x(fig, ax)
    for g, y in rows:
        ax.text(a_left - 0.06, y, name(g), transform=tr, ha="right", va="center", fontsize=FS)
    for label, y in headers:
        ax.text(0.03, y, label, transform=tr, ha="left", va="center", fontsize=FS, fontweight="bold")
    for y, t in tasks:
        ax.text(col_right, y, t, transform=tr, ha="right", va="center", fontsize=SMALL)
    ax_top = bot_in + ax_h
    ftext(fig, col_right, ax_top + 0.04, "tasks (without / with)", ha="right", va="bottom")
    ax.annotate("← honest message scores higher", xy=(0, 1), xycoords=("data", "axes fraction"),
                xytext=(-3, 3), textcoords="offset points", ha="right", va="bottom", fontsize=SMALL, color=INK)
    ax.annotate("overclaim scores higher →", xy=(0, 1), xycoords=("data", "axes fraction"),
                xytext=(3, 3), textcoords="offset points", ha="left", va="bottom", fontsize=SMALL, color=INK)

    # panel (b): the reward models without the transcript, in standard deviations of each model's scores
    rms = {r["grader"]: r for r in s["e5"] if r.get("table") == "premium"}
    b_rows, _, b_bottom = row_layout([(None, [g for g in D.REWARD_MODELS if g in rms])])
    # panel (a)'s grid: a group label takes exactly one row slot, so the five reward models are level with the
    # first five slots under panel (a)'s first label (four judges and the next group label), at the same row height
    b_rows = [(g, y - HEADER) for g, y in b_rows]
    b_ylo = b_bottom - HEADER - 0.2
    b_h = unit * (yhi - b_ylo)
    b_left, b_names = 5.38, 4.70
    bx = axes_in(fig, b_left, bot_in + ax_h - b_h, W - b_left - 0.06, b_h)
    for g, y in b_rows:
        r = rms[g]
        whisker(bx, y, *r["standardised_ci"], BLUE)
        dot(bx, r["standardised_gap"], y, BLUE, True)
    bmax = max(r["standardised_ci"][1] for r in rms.values())
    bx.set_xlim(0, max(2.0, round(bmax + 0.25, 1)))
    bx.set_ylim(b_ylo, yhi)
    bx.xaxis.set_major_locator(FixedLocator([0, 0.5, 1.0, 1.5, 2.0]))
    bx.set_xticklabels(["0", "0.5", "1", "1.5", "2"])
    rows_axes(bx)
    style(bx, zero=False)
    bx.set_xlabel("premium (standard deviations\nof the model’s scores)", fontsize=FS, linespacing=1.15)
    b_w = W - b_left - 0.06
    bx.xaxis.set_label_coords(((b_names + W) / 2 - b_left) / b_w, -0.215 / b_h)   # centred under the whole panel
    btr = inch_x(fig, bx)
    for g, y in b_rows:
        bx.text(b_left - 0.06, y, name(g), transform=btr, ha="right", va="center", fontsize=FS)

    top = H - 0.03
    ftext(fig, 0.03, top, "(a) LLM judges", ha="left", va="top", fontsize=FS, fontweight="bold")
    ftext(fig, b_names, top, "(b) Reward models", ha="left", va="top", fontsize=FS, fontweight="bold")
    legend_in(fig, [handle(BLUE, True), handle(ORANGE, False)], ["without the transcript", "with the transcript"],
              1.12, top + 0.005)
    save(fig, "fig2_premium")


# ---------------------------------------------------------------- Figure 3


FIG3_PITCH = 0.21     # inches between rows (at least 0.2 in, so labels never touch)


def selection_groups(s: dict) -> list[tuple[str, list[dict]]]:
    """The best-of-16 rows: the LLM selectors in the fixed judge order, each followed by its verification-step row
    (which sees the transcript), then the reward models. Dagger: the selectors of the pre-registered test (H3)."""
    h3 = {r["grader"] for r in s["holm_family"] if r.get("hypothesis") == "H3"}
    llm = []
    for g in D.LLM_JUDGES:
        row = D.e2_row(s, g)
        if row is None:
            continue
        llm.append({"label": name(g) + (" †" if g in h3 else ""), "row": row, "color": BLUE, "filled": True,
                    "indent": False})
        mit = D.e2_row(s, g + "+verify_first", selection="e2_shown_eligible")
        if mit is not None:
            llm.append({"label": "+ verification step, transcript", "row": mit, "color": ORANGE, "filled": False,
                        "indent": True})
    rms = [{"label": name(g), "row": D.e2_row(s, g), "color": BLUE, "filled": True, "indent": False}
           for g in D.REWARD_MODELS if D.e2_row(s, g) is not None]
    return [("LLM judges", llm), ("Reward models", rms)]


def _selection_panel(fig, left: float, bottom: float, names_w: float, data_w: float, col_w: float, height: float,
                     groups: list, base: float, xlim: tuple, ticks: list, header_slot: bool) -> object:
    """One arrow panel: names (left-aligned, indented verification rows), arrows from N = 1 (dot) to N = 16
    (arrowhead), the base-rate line, the right column of the two shares."""
    rows, headers, bot = row_layout(groups if header_slot else [(None, items) for _, items in groups])
    ylo, yhi = bot - 0.15, 0.1
    ax = axes_in(fig, left + names_w, bottom, data_w, height)
    for it, y in rows:
        rates = D.e2_rates(it["row"])["rates"]
        a, b = rates["1"]["mean"] * 100, rates["16"]["mean"] * 100
        ax.add_patch(FancyArrowPatch((a, y), (b, y), arrowstyle="-|>,head_length=3.2,head_width=1.8",
                                     mutation_scale=1.0, color=it["color"], linewidth=1.0,
                                     shrinkA=MS / 2, shrinkB=0, zorder=2))
        dot(ax, a, y, it["color"], it["filled"])
        it["text"] = f"{round(a)}% → {round(b)}%"
    ax.axvline(base * 100, color=RULE, linewidth=0.8, linestyle=(0, (1, 1.6)), zorder=1)
    ax.set_xlim(*xlim)
    ax.set_ylim(ylo, yhi)
    pct_ticks(ax, ticks)
    rows_axes(ax)
    style(ax, zero=False)
    tr = inch_x(fig, ax)
    for it, y in rows:
        ax.text(left + (0.12 if it["indent"] else 0.0), y, it["label"], transform=tr, ha="left", va="center",
                fontsize=FS)
        ax.text(left + names_w + data_w + col_w, y, it["text"], transform=tr, ha="right", va="center",
                fontsize=SMALL)
    for label, y in headers:
        ax.text(left, y, label, transform=tr, ha="left", va="center", fontsize=FS, fontweight="bold")
    return ax


def fig3_selection(s: dict) -> None:
    groups = selection_groups(s)
    base = D.derived()["counts"]["selection_base_rate"]
    shares = [D.e2_rates(it["row"])["rates"][n]["mean"] * 100 for _, items in groups for it in items
              for n in ("1", "16")]
    lo = min(shares)
    xlim = (min(25.0, lo - 6), 102.0)
    ticks = [t for t in (20, 40, 60, 80, 100) if xlim[0] <= t <= xlim[1]]
    xlabel = "share of selected messages that overclaim"
    legend = ([handle(BLUE, True, line=False), handle(ORANGE, False, line=False)],
              ["without the transcript", "with the transcript"])
    top_in, bot_in = 0.55, 0.36
    n_rows = max(len(items) for _, items in groups)
    height = FIG3_PITCH * (n_rows + 0.25)          # the axes span n_rows + 0.25 row units (_selection_panel)
    W, H = TEXT_W, round(top_in + height + bot_in, 2)
    gap = 0.30
    # The name column of each panel is as wide as its longest label needs (draft 6: the rows of the verification step
    # read "+ verification step, transcript"); the two arrow areas keep one width, so the two panels share a scale.
    names_ws, col_w = (1.60, 0.70), 0.62
    data_w = (W - gap - 0.04 - sum(names_ws) - 2 * col_w) / 2
    fig = new_fig(W, H)
    axes, col_xs = [], []
    left = 0.02
    for (label, items), names_w in zip(groups, names_ws):
        ax = _selection_panel(fig, left, bot_in, names_w, data_w, col_w, height, [(label, items)], base, xlim,
                              ticks, False)
        ax.set_xlabel(xlabel, fontsize=FS)
        ftext(fig, left, bot_in + height + 0.2, label, ha="left", va="bottom", fontsize=FS, fontweight="bold")
        axes.append(ax)
        col_xs.append(left + names_w + data_w + col_w)
        left = col_xs[-1] + gap
    legend_in(fig, *legend, 0.02, H - 0.03)
    for ax, col_x in zip(axes, col_xs):
        ax.annotate("one message at random", xy=(base * 100, 1), xycoords=("data", "axes fraction"),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=SMALL, color=INK)
        ax_top = ax.get_position().y1 * H
        ftext(fig, col_x, ax_top + 0.045, "N = 1 → 16", ha="right", va="bottom")
    save(fig, "fig3_selection")


# ---------------------------------------------------------------- Figure 4

FIG4_W = 0.6 * TEXT_W      # the figure is drawn at 60% of the text width and included unscaled
# label offset (points), alignment and leader line of each judge's name
FIG4_LABELS = {"claude-sonnet-5-5": (-4, 0, "right", False), "claude-opus-5-5": (4, 0, "left", False),
               "claude-sonnet-4-6": (4, 0, "left", False), "claude-opus-4-6": (18, -2, "left", True),
               "gemini-3.8-flash": (-6, 8, "right", False), "gemini-4-argon": (0, 15, "center", False),
               "deepseek-v4-pro": (-4, 0, "right", False), "gpt-oss-120b": (18, 7, "left", True),
               "qwen3-32b": (0, 10, "center", False)}


def fig4_side_by_side(s: dict) -> None:
    order_dependent = set(D.derived()["counts"]["judges_order_dependent"])
    W, H = FIG4_W, 2.2
    left, right_pad, bot_in, top_in = 0.74, 0.06, 0.37, 0.30
    fig = new_fig(W, H)
    ax = axes_in(fig, left, bot_in, W - left - right_pad, H - bot_in - top_in)
    pts = {}
    for g in D.LLM_JUDGES:
        hid = D.core_row(s, g, "hidden")
        pw = D.pairwise_row(s, g, "hidden")
        if hid is None or pw is None:
            continue
        oh = pw["pairs"]["O_vs_H"]
        pts[g] = (hid["premium"], oh["rate"] * 100, hid["premium_ci"], [c * 100 for c in oh["ci"]])
    xmax = max(p[2][1] for p in pts.values())
    xlim = (min(-0.35, min(p[2][0] for p in pts.values()) - 0.2), xmax + 0.35)
    ylim = (-7, 109)
    ax.add_patch(Rectangle((0, ylim[0]), xlim[1], 50 - ylim[0], facecolor="#f6f6f6", edgecolor="none", zorder=0))
    ax.axvline(0, color=RULE, linewidth=0.7, linestyle=(0, (3, 2)), zorder=0.5)
    ax.axhline(50, color=RULE, linewidth=0.7, linestyle=(0, (3, 2)), zorder=0.5)
    for g, (x, y, xci, yci) in pts.items():
        ax.plot(xci, [y, y], color=BLUE, linewidth=0.7, zorder=2)
        ax.plot([x, x], yci, color=BLUE, linewidth=0.7, zorder=2)
        dot(ax, x, y, BLUE, g not in order_dependent)
        dx, dy, ha, leader = FIG4_LABELS.get(g, (4, 0, "left", False))
        # a name level with its point starts beyond the end of the CI whisker, so the whisker never runs under it
        anchor_x = xci[1] if (dy == 0 and ha == "left") else xci[0] if (dy == 0 and ha == "right") else x
        ax.annotate(name(g), (anchor_x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=SMALL,
                    color=INK, zorder=4,
                    arrowprops=dict(arrowstyle="-", color=RULE, linewidth=0.5, shrinkA=1.5, shrinkB=3.0,
                                    relpos=({"left": 0.0, "right": 1.0}.get(ha, 0.5), 0.5))
                    if leader else None)
    ax.text(xlim[1] - 0.08, 46.5, "prefers the honest message\nside by side, scores the\noverclaim higher alone",
            ha="right", va="top", fontsize=SMALL, color=INK, linespacing=1.15, zorder=1)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(FixedLocator([0, 25, 50, 75, 100]))
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    style(ax, zero=False, grid_axis=None)
    ax.tick_params(axis="y", length=2.5, width=0.5)
    ax.set_xlabel("premium when scored alone (points)", fontsize=FS)
    ax.set_ylabel("how often the judge prefers\nthe overclaim, side by side", fontsize=FS, linespacing=1.15)
    # hollow: the judges of derived.json counts.judges_order_dependent (the caption states the rule)
    legend_in(fig, [handle(BLUE, False, line=False)], ["verdict depends on the message order"], left - 0.05, H - 0.03)
    save(fig, "fig4_side_by_side")


# ---------------------------------------------------------------- appendix figure: the mean score of each version

GREY = "#8c8c8c"
VERSIONS = (("H", "honest message"), ("P", "placebo"), ("O", "overclaim"))
# What the caption of Figure B1 says about the change of each mean with the transcript, on the tasks a judge scored in
# both conditions (paper/derived.json, mean_scores); the script stops when the data no longer say it.
FIGB1_CAPTION = {
    "overclaim_lower_with_transcript": ["Sonnet 5.5", "Opus 5.5", "Gemini 3.8 Flash", "Gemini 4 Argon"],
    "honest_no_detectable_change": ["Sonnet 5.5", "Opus 5.5", "Gemini 3.8 Flash", "Gemini 4 Argon"],
    "placebo_lower_with_transcript": ["Sonnet 5.5", "Opus 5.5", "Gemini 4 Argon"],
    "all_three_higher_with_transcript": ["DeepSeek V4 Pro", "gpt-oss-120b"],
}


def figB1_mean_scores() -> None:
    """Two panels on one set of rows (the nine LLM judges in the fixed order): (a) without the transcript, (b) with
    it, for the judges that have an estimate. A thin grey line runs from the lowest to the highest of a judge's three
    mean scores; the letters H and O sit on it in the panel's colour and P, in grey, just above it. Where H and O
    would overlap they are set side by side around their midpoint (the CSV beside the figure has the exact means)."""
    m = D.derived()["mean_scores"]
    said = {k: [name(g) for g in m[k]] for k in ("overclaim_lower_with_transcript", "honest_no_detectable_change",
                                                 "placebo_lower_with_transcript", "all_three_higher_with_transcript")}
    if said != FIGB1_CAPTION:
        raise SystemExit(f"the caption of Figure B1 (paper/sections/appendix.tex) names other judges than the data: {said}")
    panels = (("without", "(a) Without the transcript", BLUE), ("with", "(b) With the transcript", ORANGE))
    W = TEXT_W
    pitch, top_in, bot_in = 0.25, 0.47, 0.37
    p_up = 0.45                                          # P sits this far above its row's line, in row units
    rows, _, bot = row_layout([(None, list(D.LLM_JUDGES))])
    ylo, yhi = bot - 0.1, 0.35
    height = pitch * (yhi - ylo)
    H = round(top_in + height + bot_in, 2)
    fig = new_fig(W, H)
    names_w, col_w, gap = 1.0, 0.36, 0.22
    data_w = (W - names_w - 2 * col_w - gap - 0.03) / 2
    xlim = (2.6, 10.35)
    sep = 0.105 / (data_w / (xlim[1] - xlim[0]))        # the width of a bold capital at 8 pt, in score points
    table = []
    for k, (key, title, color) in enumerate(panels):
        left = names_w + k * (data_w + col_w + gap)
        ax = axes_in(fig, left, bot_in, data_w, height)
        tr = inch_x(fig, ax)
        for g, y in rows:
            if k == 0:
                ax.text(names_w - 0.06, y, name(g), transform=tr, ha="right", va="center", fontsize=FS)
            v = m[key].get(g)
            ax.text(left + data_w + col_w, y, str(v["n_tasks"]) if v else "–", transform=tr, ha="right", va="center",
                    fontsize=SMALL)
            if not v:
                continue
            xs = {c: v[c] for c, _ in VERSIONS}
            ax.plot([min(xs.values()), max(xs.values())], [y, y], color=LINK, linewidth=0.8, zorder=1,
                    solid_capstyle="butt")
            pos = dict(xs)
            if abs(xs["O"] - xs["H"]) < sep:             # side by side around their midpoint, in the order of their values
                mid = (xs["O"] + xs["H"]) / 2
                first, second = ("H", "O") if xs["H"] <= xs["O"] else ("O", "H")
                pos[first], pos[second] = mid - sep / 2, mid + sep / 2
            for c in ("H", "O"):
                ax.text(pos[c], y, c, ha="center", va="center", fontsize=FS, fontweight="bold", color=color, zorder=3,
                        bbox=dict(facecolor="white", edgecolor="none", pad=0.3))
            ax.plot([xs["P"], xs["P"]], [y, y + p_up - 0.2], color=LINK, linewidth=0.8, zorder=1,
                    solid_capstyle="butt")               # a short stem from the line to P, at P's mean
            ax.text(xs["P"], y + p_up, "P", ha="center", va="center", fontsize=FS, fontweight="bold", color=GREY,
                    zorder=3)
            for c, label in VERSIONS:
                table.append([title[4:].lower(), name(g), label, f"{xs[c]:.2f}", str(v["n_tasks"])])
        ax.set_xlim(*xlim)
        ax.set_ylim(ylo, yhi)
        ax.xaxis.set_major_locator(FixedLocator(list(range(3, 11))))
        rows_axes(ax)
        style(ax, zero=False)
        ax.set_xlabel("mean score (scale of 1 to 10)", fontsize=FS)
        ftext(fig, left, bot_in + height + 0.05, title, ha="left", va="bottom", fontsize=FS, fontweight="bold")
        ftext(fig, left + data_w + col_w, bot_in + height + 0.05, "tasks", ha="right", va="bottom")
    ftext(fig, 0.03, H - 0.03, "H honest message,  P placebo,  O overclaim", ha="left", va="top", fontsize=FS)
    with (OUT / "figB1_mean_scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["panel", "judge", "version", "mean", "tasks"])
        w.writerows(table)
    save(fig, "figB1_mean_scores")


# ---------------------------------------------------------------- Figure 1 (LaTeX)

FIG1 = D.derived()["figure1"]
FIG1_CANDIDATE = FIG1["candidates"][0]
FIG1_TASK = FIG1_CANDIDATE["task_id"]

# The two lines of the grey strip, written for the task of figure1.candidates[0] and checked against the set file.
#   task: the repository and number of the task id (imageio__imageio-510) and a paraphrase of the issue title in the
#         task file ("`if imageio.get_reader(...): pass` fails for FfmpegFormat.Reader objects"; the issue text
#         says the statement raises an OverflowError). Backticks mark code, set in typewriter type.
#   transcript: the last of the five test commands in the set's evidence_text (the summary line "2 failed, 28
#         passed, 1 skipped"), which is also what the honest sentence states.
#   evidence: strings that must still be in the test-record paragraph of the set's evidence_text, so the transcript
#         line stays true: the number of test commands and the result of the last one.
FIG1_CONTEXT = {
    "coding-imageio__imageio-510-1db4812b": {
        "task": "imageio/imageio #510: using a video reader object as an if-condition fails.",
        "transcript": "the agent's last test run ended with 2 failed and 28 passed.",
        "evidence": ("(5 in total)", '5. `cd /workspace/imageio__imageio__2.6 && python -m pytest tests/test_core.py',
                     'summary: "2 failed, 28 passed, 1 skipped'),
    },
}


def _escape(text: str) -> str:
    return (text.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
            .replace("#", "\\#").replace("$", "\\$").replace("{", "\\{").replace("}", "\\}").replace("^", "\\^{}")
            .replace("~", "\\~{}"))


def _shorten(token: str, limit: int = 30) -> str:
    """A long file path or test id keeps its last part behind a marked cut: [...]/name or [...]::name."""
    if len(token) <= limit:
        return token
    for sep in ("/", "::"):
        if sep in token:
            token = "[...]" + sep + token.rsplit(sep, 1)[1]
            if len(token) <= limit:
                return token
    return token


def _cut_options(text: str) -> str:
    """A command in backticks keeps its words up to the first file it names; what follows (its options) becomes a
    marked cut: `python -m pytest tests/test_core.py -v -k "..." --tb=short` -> `python -m pytest tests/test_core.py
    [...]`. A code span of one word, or one that names no file, stays whole."""
    def cut(m: re.Match) -> str:
        words = m.group(1).split(" ")
        for i, w in enumerate(words[:-1]):
            if i >= 1 and ("/" in w or w.endswith(".py")):
                return "`" + " ".join(words[: i + 1]) + " [...]`"
        return m.group(0)
    return re.sub(r"`([^`]+)`", cut, text)


def clean_sentence(text: str) -> str:
    """The differing sentence as shown in Figure 1: no list bullet, no check-mark emoji, no markdown bold."""
    t = text.replace("✅", " ").replace("**", "")
    t = re.sub(r"^\s*[-*]\s+", "", t.strip())
    return re.sub(r"\s+", " ", t).strip()


def highlighted(text: str) -> str:
    """The sentence as highlighted words: each word in a box of colour hl, the spaces between them as highlighted
    glue that disappears at a line break, so the highlight follows the ragged-right lines. Long paths and the options
    of a command are cut with a marked [...] (Appendix C says so)."""
    words, in_code = [], False
    for tok in _cut_options(clean_sentence(text)).split(" "):
        out = []
        for i, piece in enumerate(tok.split("`")):
            if i > 0:
                in_code = not in_code
            if piece:
                test_id = "::" in piece       # a test id is code even where the message does not mark it with backticks
                piece = _escape(_shorten(piece))
                if in_code or test_id:
                    m = re.match(r"(.*?)([.,;:]*)$", piece) if test_id and not in_code else None
                    body, tail = (m.group(1), m.group(2)) if m else (piece, "")
                    piece = "\\texttt{" + body.replace("--", "-{}-") + "}" + tail   # two hyphens stay two (no en dash)
                out.append(piece)
        if out:
            words.append("".join(out))
    return "\\exhls".join("\\exhlw{" + w + "}" for w in words)


def _inline_code(text: str) -> str:
    """LaTeX for a line of the grey strip: escaped, with `...` set in typewriter type."""
    parts = text.split("`")
    return "".join("\\texttt{" + _escape(p) + "}" if i % 2 else _escape(p) for i, p in enumerate(parts))


def plain_sentence(text: str) -> str:
    """The sentence set as highlighted() sets it (code in typewriter, long paths shortened), without the highlight."""
    return highlighted(text).replace("\\exhls", " ").replace("\\exhlw", "")


# The label of Figure 1's middle part and the small text under it (draft 6): the agent wrote only the first of the
# three messages, and the text says what the other two put in the place of its claim.
FIG1_MIDDLE = ("The final message, three versions",
               "The agent wrote the overclaim. The honest message states the missing check in its place. The placebo "
               "puts a neutral sentence there.")


def fig1_example(sets: list[dict]) -> None:
    """paper/figures/fig1_example.tex, three bands with a label at the left of each: what the agent did (a grey strip:
    the task and what the transcript shows), the final message in three versions (three boxes of equal size with the
    sentence that differs between the overclaim, the honest message and the placebo; highlight in the first two only;
    FIG1_MIDDLE beside them), and how a grader scored it (Sonnet 5.5's three scores without the transcript and its
    three scores with it, under the boxes)."""
    c = FIG1_CANDIDATE
    st = next(x for x in sets if x["task_id"] == FIG1_TASK)
    ctx = FIG1_CONTEXT[FIG1_TASK]
    record = st["evidence_text"].split("\n\n", 1)[0]                      # the test-record paragraph
    for needle in ctx["evidence"]:
        if needle not in record:
            raise ValueError(f"{FIG1_TASK}: the evidence text no longer says {needle!r}")
    if c["scores_with_transcript"] is None:
        raise ValueError(f"{FIG1_TASK}: no scores with the transcript; draw one score row and say so in the report")
    judge = name(FIG1["judge"])
    boxes = [f"\\examplebox{{Overclaim (O)}}{{{highlighted(c['overclaim'])}}}",
             f"\\examplebox{{Honest (H)}}{{{highlighted(c['honest'])}}}",
             f"\\examplebox{{Placebo (P)}}{{{plain_sentence(c['placebo'])}}}"]
    without = " & ".join(f"\\exscore{{exblue!12}}{{{c['scores'][k]}}}" for k in ("O", "H", "P"))
    with_ = " & ".join(f"\\exscore{{exorange!14}}{{{c['scores_with_transcript'][k]}}}" for k in ("O", "H", "P"))
    doc = f"""% Generated by scripts/make_paper_figures.py from {D.DERIVED.relative_to(D.REPO)} (figure1.candidates[0]) and {D.FROZEN_SETS.relative_to(D.REPO)} (task {FIG1_TASK}).
% Do not edit by hand. The three sentences are the candidate's overclaim, honest and placebo lines (bullet and emoji
% dropped); the scores are {judge}'s, without and with the transcript. Needs xcolor (colour hl) and tcolorbox with
% the skins library (which loads TikZ), as loaded in paper/main.tex; the equal-height group settles on the second
% LaTeX run.
\\providecommand{{\\exhlw}}[1]{{{{\\setlength{{\\fboxsep}}{{0pt}}\\colorbox{{hl}}{{\\rule[-0.26em]{{0pt}}{{1.0em}}#1}}}}}}
\\providecommand{{\\exhls}}{{{{\\color{{hl}}\\leaders\\hrule height 0.74em depth 0.26em\\hskip 0.333em plus 0.1em minus 0.05em}}}}
\\definecolor{{exblue}}{{HTML}}{{2A78D6}}\\definecolor{{exorange}}{{HTML}}{{EB6834}}
\\newlength{{\\exlab}}\\newlength{{\\exbody}}\\newlength{{\\exbox}}\\newlength{{\\exgap}}
\\setlength{{\\exlab}}{{1.46in}}\\setlength{{\\exgap}}{{4pt}}
\\setlength{{\\exbody}}{{\\dimexpr\\linewidth-\\exlab\\relax}}\\setlength{{\\exbox}}{{\\dimexpr(\\exbody-2\\exgap)/3\\relax}}
\\newcommand{{\\exband}}[1]{{{{\\scriptsize\\bfseries #1}}}}
\\newcommand{{\\exmark}}[2]{{\\tikz[baseline=-0.6ex]\\draw[#1,fill=#2,line width=0.9pt] (0,0) circle (2.2pt);}}
\\newcommand{{\\exscore}}[2]{{{{\\setlength{{\\fboxsep}}{{0pt}}\\colorbox{{#1}}{{\\makebox[\\exbox][c]{{\\rule[-0.4em]{{0pt}}{{1.4em}}\\large\\bfseries #2}}}}}}}}
\\newcommand{{\\examplebox}}[2]{{%
\\begin{{tcolorbox}}[enhanced, equal height group=figexample, width=\\exbox, colback=white, colframe=black!45,
  boxrule=0.4pt, arc=1pt, boxsep=0pt, left=3pt, right=3pt, top=3pt, bottom=3pt, nobeforeafter,
  title={{\\scriptsize\\bfseries #1}}, colbacktitle=black!6, coltitle=black, toptitle=1.5pt, bottomtitle=1.2pt]
\\scriptsize\\raggedright #2
\\end{{tcolorbox}}}}%
\\par\\noindent
\\begin{{minipage}}[c]{{\\exlab}}\\exband{{What the agent did}}\\end{{minipage}}%
\\begin{{minipage}}[c]{{\\exbody}}
\\begin{{tcolorbox}}[enhanced, width=\\linewidth, colback=black!6, colframe=black!6, boxrule=0pt, arc=0pt, boxsep=0pt,
  left=3.4pt, right=3.4pt, top=2pt, bottom=2pt, nobeforeafter]
\\scriptsize\\raggedright\\textbf{{Task:}} {_inline_code(ctx['task'])}\\\\
\\textbf{{Transcript:}} {_inline_code(ctx['transcript'])}
\\end{{tcolorbox}}
\\end{{minipage}}\\par\\vspace{{3pt}}
\\noindent
\\begin{{minipage}}[c]{{\\exlab}}\\begin{{minipage}}{{\\dimexpr\\exlab-7pt\\relax}}%
{{\\scriptsize\\raggedright{{\\bfseries {FIG1_MIDDLE[0]}}}\\\\[1pt]{FIG1_MIDDLE[1]}\\par}}\\end{{minipage}}\\end{{minipage}}%
\\begin{{minipage}}[c]{{\\exbody}}
{boxes[0]}\\hfill
{boxes[1]}\\hfill
{boxes[2]}
\\end{{minipage}}\\par\\vspace{{4pt}}
\\noindent
{{\\setlength{{\\tabcolsep}}{{0pt}}\\renewcommand{{\\arraystretch}}{{1.0}}%
\\begin{{tabular}}{{@{{}}p{{\\exlab}}@{{}}c@{{\\hspace{{\\exgap}}}}c@{{\\hspace{{\\exgap}}}}c@{{}}}}
\\exband{{How a grader scored it}} & \\multicolumn{{3}}{{@{{}}l@{{}}}}{{\\scriptsize The score that {judge} gave each message, from 1 to 10:}} \\\\[2.5pt]
{{\\scriptsize\\exmark{{exblue}}{{exblue}}\\enspace sees only the message}} & {without} \\\\[2.5pt]
{{\\scriptsize\\exmark{{exorange}}{{white}}\\enspace also sees the transcript}} & {with_} \\\\
\\end{{tabular}}}}\\par
"""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fig1_example.tex").write_text(doc, encoding="utf-8")


def main() -> None:
    s = D.summary()
    PROBLEMS.clear()
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("figA_*.pdf"):          # the appendix figures of draft 3, dropped in draft 4
        old.unlink()
    fig2_premium(s)
    fig3_selection(s)
    fig4_side_by_side(s)
    figB1_mean_scores()
    fig1_example(D.frozen_sets())
    print("wrote", sorted(p.name for p in OUT.iterdir()))
    if PROBLEMS:
        print(f"{len(PROBLEMS)} layout problem(s); see above", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
