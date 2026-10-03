"""
Benchmark 09: Plotting
Matplotlib
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.cluster.hierarchy import dendrogram
from scipy.stats import gaussian_kde
from utils import run, create_suite, header, footer

suite = create_suite("plot", "Matplotlib")
header("Benchmark 09: Plotting", "Matplotlib")

rng = np.random.RandomState(42)

# ── scatter ─────────────────────────────────────────────

x20, y20 = rng.randn(20), rng.randn(20)
x100, y100 = rng.randn(100), rng.randn(100)
x500, y500 = rng.randn(500), rng.randn(500)
x2k, y2k = rng.randn(2000), rng.randn(2000)
x5k, y5k = rng.randn(5000), rng.randn(5000)

def do_scatter(x, y):
    fig, ax = plt.subplots(); ax.scatter(x, y); plt.close(fig)

run(suite, "scatter", "20 pts", lambda: do_scatter(x20, y20))
run(suite, "scatter", "100 pts", lambda: do_scatter(x100, y100))
run(suite, "scatter", "500 pts", lambda: do_scatter(x500, y500))
run(suite, "scatter", "2K pts", lambda: do_scatter(x2k, y2k))
run(suite, "scatter", "5K pts", lambda: do_scatter(x5k, y5k))

# ── line plot ───────────────────────────────────────────

def do_plot(x, y):
    fig, ax = plt.subplots(); ax.plot(x, y); plt.close(fig)

run(suite, "plot (line)", "20 pts", lambda: do_plot(x20, y20))
run(suite, "plot (line)", "100 pts", lambda: do_plot(x100, y100))
run(suite, "plot (line)", "500 pts", lambda: do_plot(x500, y500))
run(suite, "plot (line)", "2K pts", lambda: do_plot(x2k, y2k))
run(suite, "plot (line)", "5K pts", lambda: do_plot(x5k, y5k))

# ── bar ─────────────────────────────────────────────────

bx10, bh10 = np.arange(10), rng.randn(10)
bx50, bh50 = np.arange(50), rng.randn(50)
bx200, bh200 = np.arange(200), rng.randn(200)

def do_bar(x, h):
    fig, ax = plt.subplots(); ax.bar(x, h); plt.close(fig)

run(suite, "bar", "10 bars", lambda: do_bar(bx10, bh10))
run(suite, "bar", "50 bars", lambda: do_bar(bx50, bh50))
run(suite, "bar", "200 bars", lambda: do_bar(bx200, bh200))

group_x20 = np.arange(20)
group_h1 = rng.randn(20)
group_h2 = rng.randn(20)
group_h3 = rng.randn(20)

def do_stacked_bar(x, h1, h2, h3):
    fig, ax = plt.subplots()
    ax.bar(x, h1, label="A")
    ax.bar(x, h2, bottom=h1, label="B")
    ax.bar(x, h3, bottom=h1 + h2, label="C")
    ax.legend()
    plt.close(fig)

def do_grouped_bar(x, h1, h2, h3):
    fig, ax = plt.subplots()
    width = 0.25
    ax.bar(x - width, h1, width=width, label="A")
    ax.bar(x, h2, width=width, label="B")
    ax.bar(x + width, h3, width=width, label="C")
    ax.legend()
    plt.close(fig)

run(suite, "stackedBar", "20 bars × 3 series", lambda: do_stacked_bar(group_x20, group_h1, group_h2, group_h3))
run(suite, "groupedBar", "20 bars × 3 series", lambda: do_grouped_bar(group_x20, group_h1, group_h2, group_h3))

# ── barh ────────────────────────────────────────────────

def do_barh(y, w):
    fig, ax = plt.subplots(); ax.barh(y, w); plt.close(fig)

run(suite, "barh", "10 bars", lambda: do_barh(bx10, bh10))
run(suite, "barh", "50 bars", lambda: do_barh(bx50, bh50))

# ── hist ────────────────────────────────────────────────

def do_hist(x, bins):
    fig, ax = plt.subplots(); ax.hist(x, bins=bins); plt.close(fig)

run(suite, "hist", "100 bins=10", lambda: do_hist(x100, 10))
run(suite, "hist", "500 bins=20", lambda: do_hist(x500, 20))
run(suite, "hist", "2K bins=30", lambda: do_hist(x2k, 30))
run(suite, "hist", "5K bins=50", lambda: do_hist(x5k, 50))

# ── boxplot ─────────────────────────────────────────────

def do_boxplot(x):
    fig, ax = plt.subplots(); ax.boxplot(x); plt.close(fig)

run(suite, "boxplot", "100 pts", lambda: do_boxplot(x100))
run(suite, "boxplot", "500 pts", lambda: do_boxplot(x500))
run(suite, "boxplot", "2K pts", lambda: do_boxplot(x2k))

# ── violinplot ──────────────────────────────────────────

def do_violin(x):
    fig, ax = plt.subplots(); ax.violinplot(x); plt.close(fig)

run(suite, "violinplot", "100 pts", lambda: do_violin(x100))
run(suite, "violinplot", "500 pts", lambda: do_violin(x500))

# ── pie ─────────────────────────────────────────────────

pv5 = [30, 25, 20, 15, 10]
pv10 = [15, 12, 11, 10, 9, 8, 8, 7, 10, 10]

def do_pie(v, labels=None):
    fig, ax = plt.subplots(); ax.pie(v, labels=labels); plt.close(fig)

run(suite, "pie", "5 slices", lambda: do_pie(pv5, ["A","B","C","D","E"]))
run(suite, "pie", "10 slices", lambda: do_pie(pv10))

# ── heatmap ─────────────────────────────────────────────

hm10 = rng.randn(10, 10)
hm25 = rng.randn(25, 25)
hm50 = rng.randn(50, 50)

def do_heatmap(d):
    fig, ax = plt.subplots(); ax.imshow(d, aspect="auto"); plt.close(fig)

run(suite, "heatmap", "10x10", lambda: do_heatmap(hm10))
run(suite, "heatmap", "25x25", lambda: do_heatmap(hm25))
run(suite, "heatmap", "50x50", lambda: do_heatmap(hm50))

# ── imshow ──────────────────────────────────────────────

run(suite, "imshow", "10x10", lambda: do_heatmap(hm10))
run(suite, "imshow", "50x50", lambda: do_heatmap(hm50))

# ── contour ─────────────────────────────────────────────

cs = 20
cx = np.arange(cs); cy = np.arange(cs)
CX, CY = np.meshgrid(cx, cy)
CZ = np.sin(CX / 3) * np.cos(CY / 3)
cs40 = 40
cx40 = np.arange(cs40); cy40 = np.arange(cs40)
CX40, CY40 = np.meshgrid(cx40, cy40)
CZ40 = np.sin(CX40 / 3) * np.cos(CY40 / 3)

def do_contour(X, Y, Z):
    fig, ax = plt.subplots(); ax.contour(X, Y, Z); plt.close(fig)

def do_contourf(X, Y, Z):
    fig, ax = plt.subplots(); ax.contourf(X, Y, Z); plt.close(fig)

run(suite, "contour", "20x20", lambda: do_contour(CX, CY, CZ))
run(suite, "contour", "40x40", lambda: do_contour(CX40, CY40, CZ40))
run(suite, "contourf", "20x20", lambda: do_contourf(CX, CY, CZ))
run(suite, "contourf", "40x40", lambda: do_contourf(CX40, CY40, CZ40))

# ── ML Plots ────────────────────────────────────────────

cm3 = np.array([[45,3,2],[4,40,6],[1,5,44]])

def do_confusion():
    fig, ax = plt.subplots(); ax.imshow(cm3); plt.close(fig)

run(suite, "plotConfusionMatrix", "3x3", do_confusion)

fpr_np = np.linspace(0, 1, 100)
tpr_np = np.minimum(1, (fpr_np) ** 0.5)

def do_roc():
    fig, ax = plt.subplots(); ax.plot(fpr_np, tpr_np); ax.plot([0,1],[0,1],'--'); plt.close(fig)

run(suite, "plotRocCurve", "100 pts", do_roc)

prec_np = 1 - np.linspace(0, 1, 100)
rec_np = np.linspace(0, 1, 100)

def do_pr():
    fig, ax = plt.subplots(); ax.plot(rec_np, prec_np); plt.close(fig)

run(suite, "plotPrecisionRecallCurve", "100 pts", do_pr)

ts = [10, 20, 50, 100, 200]
tsc = [0.6, 0.7, 0.8, 0.85, 0.9]
vsc = [0.5, 0.6, 0.7, 0.75, 0.78]

def do_lc():
    fig, ax = plt.subplots(); ax.plot(ts, tsc); ax.plot(ts, vsc); plt.close(fig)

run(suite, "plotLearningCurve", "5 pts", do_lc)

pr_range = [0.001, 0.01, 0.1, 1, 10]

def do_vc():
    fig, ax = plt.subplots(); ax.plot(pr_range, tsc); ax.plot(pr_range, vsc); plt.close(fig)

run(suite, "plotValidationCurve", "5 pts", do_vc)

# ── v1.0.0 Plotting Additions ───────────────────────────

def do_kde(values, fill=False):
    xs = np.linspace(values.min() - 1, values.max() + 1, 200)
    ys = gaussian_kde(values)(xs)
    fig, ax = plt.subplots()
    ax.plot(xs, ys)
    if fill:
        ax.fill_between(xs, ys, alpha=0.3)
    plt.close(fig)

run(suite, "kdeplot", "500 pts", lambda: do_kde(x500))
run(suite, "kdeplot", "2K pts", lambda: do_kde(x2k, fill=True))

def do_stem(x, y):
    fig, ax = plt.subplots(); ax.stem(x, y); plt.close(fig)

run(suite, "stem", "100 pts", lambda: do_stem(x100, y100))
run(suite, "stem", "500 pts", lambda: do_stem(x500, y500))

qx = np.array([0, 1, 0, 1])
qy = np.array([0, 0, 1, 1])
qu = np.array([1, 0, -1, 0])
qv = np.array([0, 1, 0, -1])
qx25 = np.array([i % 5 for i in range(25)])
qy25 = np.array([i // 5 for i in range(25)])
qu25 = np.array([np.sin(i / 4) for i in range(25)])
qv25 = np.array([np.cos(i / 4) for i in range(25)])

def do_quiver(x, y, u, v):
    fig, ax = plt.subplots(); ax.quiver(x, y, u, v); plt.close(fig)

run(suite, "quiver", "4 vectors", lambda: do_quiver(qx, qy, qu, qv))
run(suite, "quiver", "25 vectors", lambda: do_quiver(qx25, qy25, qu25, qv25))

polar_theta50 = np.array([(2 * np.pi * i) / 50 for i in range(50)])
polar_r50 = np.array([1 + 0.5 * np.sin((6 * np.pi * i) / 50) for i in range(50)])
polar_theta200 = np.array([(2 * np.pi * i) / 200 for i in range(200)])
polar_r200 = np.array([1 + 0.5 * np.sin((6 * np.pi * i) / 200) for i in range(200)])

def do_polar(theta, r, fill=False):
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.plot(theta, r)
    if fill:
        ax.fill(theta, r, alpha=0.3)
    plt.close(fig)

run(suite, "polar", "50 pts", lambda: do_polar(polar_theta50, polar_r50))
run(suite, "polar", "200 pts", lambda: do_polar(polar_theta200, polar_r200, fill=True))

surface_axis20 = np.linspace(-2, 2, 20)
SX20, SY20 = np.meshgrid(surface_axis20, surface_axis20)
SZ20 = np.sin(np.sqrt(SX20 ** 2 + SY20 ** 2))
surface_axis30 = np.linspace(-2, 2, 30)
SX30, SY30 = np.meshgrid(surface_axis30, surface_axis30)
SZ30 = np.sin(np.sqrt(SX30 ** 2 + SY30 ** 2))

def do_surface(X, Y, Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z)
    plt.close(fig)

run(suite, "surface", "20x20", lambda: do_surface(SX20, SY20, SZ20))
run(suite, "surface", "30x30", lambda: do_surface(SX30, SY30, SZ30))

linkage = np.array([
    [0, 1, 1.0, 2],
    [2, 3, 1.5, 2],
    [4, 5, 3.0, 4],
], dtype=float)

def do_dendrogram():
    fig, ax = plt.subplots(); dendrogram(linkage, ax=ax); plt.close(fig)

run(suite, "plotDendrogram", "4 leaves", do_dendrogram)

# ── SVG Rendering ───────────────────────────────────────

def show_svg_scatter():
    fig, ax = plt.subplots(); ax.scatter(x100, y100)
    buf = BytesIO(); fig.savefig(buf, format="svg"); plt.close(fig)

def show_svg_heatmap():
    fig, ax = plt.subplots(); ax.imshow(hm25, aspect="auto")
    buf = BytesIO(); fig.savefig(buf, format="svg"); plt.close(fig)

def show_svg_line():
    fig, ax = plt.subplots(); ax.plot(x500, y500)
    buf = BytesIO(); fig.savefig(buf, format="svg"); plt.close(fig)

run(suite, "show (SVG) scatter", "100 pts", show_svg_scatter)
run(suite, "show (SVG) heatmap", "25x25", show_svg_heatmap)
run(suite, "show (SVG) line", "500 pts", show_svg_line)

def show_png_scatter():
    fig, ax = plt.subplots(); ax.scatter(x100, y100)
    buf = BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)

def show_png_heatmap():
    fig, ax = plt.subplots(); ax.imshow(hm25, aspect="auto")
    buf = BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)

def save_pdf_line():
    fig, ax = plt.subplots(); ax.plot(x500, y500)
    buf = BytesIO(); fig.savefig(buf, format="pdf"); plt.close(fig)

run(suite, "show (PNG) scatter", "100 pts", show_png_scatter)
run(suite, "show (PNG) heatmap", "25x25", show_png_heatmap)
run(suite, "saveFig (PDF) line", "500 pts", save_pdf_line)

# ── Extended coverage (v1.1 benchmark expansion) ────────
x50, y50 = rng.randn(50), rng.randn(50)
x200, y200 = rng.randn(200), rng.randn(200)
x1k, y1k = rng.randn(1000), rng.randn(1000)
x10k, y10k = rng.randn(10000), rng.randn(10000)
for _sz, _xs, _ys in [
    ("50 pts", x50, y50),
    ("200 pts", x200, y200),
    ("1K pts", x1k, y1k),
    ("10K pts", x10k, y10k),
]:
    run(suite, "scatter", _sz, lambda xs=_xs, ys=_ys: do_scatter(xs, ys))
    run(suite, "plot (line)", _sz, lambda xs=_xs, ys=_ys: do_plot(xs, ys))
run(suite, "hist", "1K bins=40", lambda: do_hist(x1k, 40))
run(suite, "hist", "10K bins=60", lambda: do_hist(x10k, 60))
run(suite, "bar", "500 bars", lambda: do_bar(np.arange(500), rng.randn(500)))
run(suite, "barh", "200 bars", lambda: do_barh(np.arange(200), rng.randn(200)))
run(suite, "boxplot", "1K pts", lambda: do_boxplot(x1k))
run(suite, "violinplot", "1K pts", lambda: do_violin(x1k))

footer(suite, "matplotlib-plot.json")
