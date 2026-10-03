/**
 * Benchmark 09: Plotting
 * Deepbox vs Matplotlib
 */

import { arange, linspace, randn, tensor } from "deepbox/ndarray";
import {
  bar,
  barh,
  boxplot,
  contour,
  contourf,
  figure,
  groupedBar,
  heatmap,
  hist,
  imshow,
  kdeplot,
  legend,
  pie,
  plot,
  plotConfusionMatrix,
  plotDendrogram,
  plotLearningCurve,
  plotPrecisionRecallCurve,
  plotRocCurve,
  plotValidationCurve,
  polar,
  quiver,
  radar,
  saveFig,
  scatter,
  show,
  stackedBar,
  stem,
  strip,
  surface,
  violinplot,
  waterfall,
} from "deepbox/plot";
import { createSuite, footer, header, run, runAsync } from "../utils";

const suite = createSuite("plot");
header("Benchmark 09: Plotting");

// ── Helpers ─────────────────────────────────────────────

function freshFig() {
  figure();
}

/**
 * Record a spec-build plotting case as a **local (non-comparable)** benchmark.
 *
 * Deepbox's `scatter`/`plot`/`bar`/… assemble an in-memory figure spec (a plain
 * object graph of series + styling) and defer all rasterization until `show()`.
 * Matplotlib's `ax.scatter(...)` immediately constructs a full Figure/Axes
 * object graph with transforms, artists, and a renderer. The two sides measure
 * fundamentally different work (a spec append versus a figure construction), so
 * counting them head-to-head inflates Deepbox's win rate (the historical
 * "28184x" pie result was never a rendering comparison).
 *
 * These cases are therefore recorded with `comparable: false` and excluded from
 * win totals. The genuine render-vs-render comparison lives in the
 * `show (SVG)` / `show (PNG)` / `saveFig (PDF)` cases below, which serialize on
 * both sides and remain comparable.
 */
function specPlot(operation: string, size: string, fn: () => void): void {
  run(suite, operation, size, fn, {
    comparable: false,
    tags: ["spec-build", "deepbox-only"],
  });
}

// ── scatter ─────────────────────────────────────────────

const x20 = randn([20]);
const y20 = randn([20]);
const x100 = randn([100]);
const y100 = randn([100]);
const x500 = randn([500]);
const y500 = randn([500]);
const x2k = randn([2000]);
const y2k = randn([2000]);
const x5k = randn([5000]);
const y5k = randn([5000]);

specPlot("scatter", "20 pts", () => {
  freshFig();
  scatter(x20, y20);
});
specPlot("scatter", "100 pts", () => {
  freshFig();
  scatter(x100, y100);
});
specPlot("scatter", "500 pts", () => {
  freshFig();
  scatter(x500, y500);
});
specPlot("scatter", "2K pts", () => {
  freshFig();
  scatter(x2k, y2k);
});
specPlot("scatter", "5K pts", () => {
  freshFig();
  scatter(x5k, y5k);
});

// ── line plot ───────────────────────────────────────────

specPlot("plot (line)", "20 pts", () => {
  freshFig();
  plot(x20, y20);
});
specPlot("plot (line)", "100 pts", () => {
  freshFig();
  plot(x100, y100);
});
specPlot("plot (line)", "500 pts", () => {
  freshFig();
  plot(x500, y500);
});
specPlot("plot (line)", "2K pts", () => {
  freshFig();
  plot(x2k, y2k);
});
specPlot("plot (line)", "5K pts", () => {
  freshFig();
  plot(x5k, y5k);
});

// ── bar ─────────────────────────────────────────────────

const barX10 = arange(0, 10);
const barH10 = randn([10]);
const barX50 = arange(0, 50);
const barH50 = randn([50]);
const barX200 = arange(0, 200);
const barH200 = randn([200]);
const groupedX20 = arange(0, 20);
const groupedH1 = randn([20]);
const groupedH2 = randn([20]);
const groupedH3 = randn([20]);

specPlot("bar", "10 bars", () => {
  freshFig();
  bar(barX10, barH10);
});
specPlot("bar", "50 bars", () => {
  freshFig();
  bar(barX50, barH50);
});
specPlot("bar", "200 bars", () => {
  freshFig();
  bar(barX200, barH200);
});
specPlot("stackedBar", "20 bars × 3 series", () => {
  freshFig();
  stackedBar(groupedX20, [groupedH1, groupedH2, groupedH3], {
    labels: ["A", "B", "C"],
  });
  legend();
});
specPlot("groupedBar", "20 bars × 3 series", () => {
  freshFig();
  groupedBar(groupedX20, [groupedH1, groupedH2, groupedH3], {
    labels: ["A", "B", "C"],
  });
  legend();
});

// ── barh ────────────────────────────────────────────────

specPlot("barh", "10 bars", () => {
  freshFig();
  barh(barX10, barH10);
});
specPlot("barh", "50 bars", () => {
  freshFig();
  barh(barX50, barH50);
});

// ── hist ────────────────────────────────────────────────

specPlot("hist", "100 bins=10", () => {
  freshFig();
  hist(x100, 10);
});
specPlot("hist", "500 bins=20", () => {
  freshFig();
  hist(x500, 20);
});
specPlot("hist", "2K bins=30", () => {
  freshFig();
  hist(x2k, 30);
});
specPlot("hist", "5K bins=50", () => {
  freshFig();
  hist(x5k, 50);
});

// ── boxplot ─────────────────────────────────────────────

specPlot("boxplot", "100 pts", () => {
  freshFig();
  boxplot(x100);
});
specPlot("boxplot", "500 pts", () => {
  freshFig();
  boxplot(x500);
});
specPlot("boxplot", "2K pts", () => {
  freshFig();
  boxplot(x2k);
});

// ── violinplot ──────────────────────────────────────────

specPlot("violinplot", "100 pts", () => {
  freshFig();
  violinplot(x100);
});
specPlot("violinplot", "500 pts", () => {
  freshFig();
  violinplot(x500);
});

// ── pie ─────────────────────────────────────────────────

const pieVals5 = tensor([30, 25, 20, 15, 10]);
const pieVals10 = tensor([15, 12, 11, 10, 9, 8, 8, 7, 10, 10]);

specPlot("pie", "5 slices", () => {
  freshFig();
  pie(pieVals5, ["A", "B", "C", "D", "E"]);
});
specPlot("pie", "10 slices", () => {
  freshFig();
  pie(pieVals10);
});

// ── heatmap ─────────────────────────────────────────────

const hm10 = randn([10, 10]);
const hm25 = randn([25, 25]);
const hm50 = randn([50, 50]);

specPlot("heatmap", "10x10", () => {
  freshFig();
  heatmap(hm10);
});
specPlot("heatmap", "25x25", () => {
  freshFig();
  heatmap(hm25);
});
specPlot("heatmap", "50x50", () => {
  freshFig();
  heatmap(hm50);
});

// ── imshow ──────────────────────────────────────────────

specPlot("imshow", "10x10", () => {
  freshFig();
  imshow(hm10);
});
specPlot("imshow", "50x50", () => {
  freshFig();
  imshow(hm50);
});

// ── contour ─────────────────────────────────────────────

const cSize = 20;
const cX = arange(0, cSize);
const cY = arange(0, cSize);
const cZ = tensor(
  Array.from({ length: cSize }, (_, i) =>
    Array.from({ length: cSize }, (_, j) => Math.sin(i / 3) * Math.cos(j / 3))
  )
);
const cSize40 = 40;
const cX40 = arange(0, cSize40);
const cY40 = arange(0, cSize40);
const cZ40 = tensor(
  Array.from({ length: cSize40 }, (_, i) =>
    Array.from({ length: cSize40 }, (_, j) => Math.sin(i / 3) * Math.cos(j / 3))
  )
);

specPlot("contour", "20x20", () => {
  freshFig();
  contour(cX, cY, cZ);
});
specPlot("contour", "40x40", () => {
  freshFig();
  contour(cX40, cY40, cZ40);
});
specPlot("contourf", "20x20", () => {
  freshFig();
  contourf(cX, cY, cZ);
});
specPlot("contourf", "40x40", () => {
  freshFig();
  contourf(cX40, cY40, cZ40);
});

// ── ML Plots ────────────────────────────────────────────

const cm3 = tensor([
  [45, 3, 2],
  [4, 40, 6],
  [1, 5, 44],
]);
specPlot("plotConfusionMatrix", "3x3", () => {
  freshFig();
  plotConfusionMatrix(cm3, ["A", "B", "C"]);
});

const fpr = linspace(0, 1, 100);
const tpr = tensor(Array.from({ length: 100 }, (_, i) => Math.min(1, (i / 100) ** 0.5)));
specPlot("plotRocCurve", "100 pts", () => {
  freshFig();
  plotRocCurve(fpr, tpr, 0.85);
});

const prec = tensor(Array.from({ length: 100 }, (_, i) => 1 - i / 100));
const rec = linspace(0, 1, 100);
specPlot("plotPrecisionRecallCurve", "100 pts", () => {
  freshFig();
  plotPrecisionRecallCurve(prec, rec, 0.75);
});

const trainSizes = tensor([10, 20, 50, 100, 200]);
const trainScores = tensor([0.6, 0.7, 0.8, 0.85, 0.9]);
const valScores = tensor([0.5, 0.6, 0.7, 0.75, 0.78]);
specPlot("plotLearningCurve", "5 pts", () => {
  freshFig();
  plotLearningCurve(trainSizes, trainScores, valScores);
});

const paramRange = tensor([0.001, 0.01, 0.1, 1, 10]);
specPlot("plotValidationCurve", "5 pts", () => {
  freshFig();
  plotValidationCurve(paramRange, trainScores, valScores);
});

// ── v1.0.0 Plotting Additions ───────────────────────────

specPlot("kdeplot", "500 pts", () => {
  freshFig();
  kdeplot(x500);
});
specPlot("kdeplot", "2K pts", () => {
  freshFig();
  kdeplot(x2k, { fill: true });
});

specPlot("stem", "100 pts", () => {
  freshFig();
  stem(x100, y100);
});
specPlot("stem", "500 pts", () => {
  freshFig();
  stem(x500, y500);
});

const qx = tensor([0, 1, 0, 1]);
const qy = tensor([0, 0, 1, 1]);
const qu = tensor([1, 0, -1, 0]);
const qv = tensor([0, 1, 0, -1]);
const qx25 = tensor(Array.from({ length: 25 }, (_, i) => i % 5));
const qy25 = tensor(Array.from({ length: 25 }, (_, i) => Math.floor(i / 5)));
const qu25 = tensor(Array.from({ length: 25 }, (_, i) => Math.sin(i / 4)));
const qv25 = tensor(Array.from({ length: 25 }, (_, i) => Math.cos(i / 4)));

specPlot("quiver", "4 vectors", () => {
  freshFig();
  quiver(qx, qy, qu, qv);
});
specPlot("quiver", "25 vectors", () => {
  freshFig();
  quiver(qx25, qy25, qu25, qv25);
});

const polarTheta50 = tensor(Array.from({ length: 50 }, (_, i) => (2 * Math.PI * i) / 50));
const polarR50 = tensor(
  Array.from({ length: 50 }, (_, i) => 1 + 0.5 * Math.sin((6 * Math.PI * i) / 50))
);
const polarTheta200 = tensor(Array.from({ length: 200 }, (_, i) => (2 * Math.PI * i) / 200));
const polarR200 = tensor(
  Array.from({ length: 200 }, (_, i) => 1 + 0.5 * Math.sin((6 * Math.PI * i) / 200))
);

specPlot("polar", "50 pts", () => {
  freshFig();
  polar(polarTheta50, polarR50);
});
specPlot("polar", "200 pts", () => {
  freshFig();
  polar(polarTheta200, polarR200, { fill: true });
});

const surfaceGrid20 = Array.from({ length: 20 }, (_, i) =>
  Array.from({ length: 20 }, (_, j) => -2 + (4 * (i + j - j)) / 19)
);
const surfaceY20 = Array.from({ length: 20 }, (_, _i) =>
  Array.from({ length: 20 }, (_, j) => -2 + (4 * j) / 19)
);
const surfaceZ20 = Array.from({ length: 20 }, (_, i) =>
  Array.from({ length: 20 }, (_, j) => {
    const x = -2 + (4 * i) / 19;
    const y = -2 + (4 * j) / 19;
    return Math.sin(Math.sqrt(x * x + y * y));
  })
);
const surfaceGrid30 = Array.from({ length: 30 }, (_, i) =>
  Array.from({ length: 30 }, () => -2 + (4 * i) / 29)
);
const surfaceY30 = Array.from({ length: 30 }, (_, _i) =>
  Array.from({ length: 30 }, (_, j) => -2 + (4 * j) / 29)
);
const surfaceZ30 = Array.from({ length: 30 }, (_, i) =>
  Array.from({ length: 30 }, (_, j) => {
    const x = -2 + (4 * i) / 29;
    const y = -2 + (4 * j) / 29;
    return Math.sin(Math.sqrt(x * x + y * y));
  })
);

specPlot("surface", "20x20", () => {
  freshFig();
  surface(surfaceGrid20, surfaceY20, surfaceZ20);
});
specPlot("surface", "30x30", () => {
  freshFig();
  surface(surfaceGrid30, surfaceY30, surfaceZ30);
});

const linkage = [
  [0, 1, 1.0, 2],
  [2, 3, 1.5, 2],
  [4, 5, 3.0, 4],
] as const;
specPlot("plotDendrogram", "4 leaves", () => {
  freshFig();
  plotDendrogram(linkage, 4);
});
run(
  suite,
  "strip",
  "2×500 pts",
  () => {
    freshFig();
    strip([x500, y500], { labels: ["A", "B"] });
  },
  { comparable: false, tags: ["deepbox-only"] }
);
run(
  suite,
  "radar",
  "2×6 axes",
  () => {
    freshFig();
    radar([tensor([0.8, 0.7, 0.9, 0.6, 0.75, 0.85]), tensor([0.6, 0.8, 0.7, 0.9, 0.65, 0.7])], {
      labels: ["Model A", "Model B"],
    });
  },
  { comparable: false, tags: ["deepbox-only"] }
);
run(
  suite,
  "waterfall",
  "6 bars",
  () => {
    freshFig();
    waterfall(
      ["Base", "North", "South", "Returns", "Upsell", "Total"],
      tensor([100, 20, 15, -10, 12, 137])
    );
  },
  { comparable: false, tags: ["deepbox-only"] }
);

// ── SVG Rendering ───────────────────────────────────────

run(suite, "show (SVG) scatter", "100 pts", () => {
  freshFig();
  scatter(x100, y100);
  show();
});
run(suite, "show (SVG) heatmap", "25x25", () => {
  freshFig();
  heatmap(hm25);
  show();
});
run(suite, "show (SVG) line", "500 pts", () => {
  freshFig();
  plot(x500, y500);
  show();
});
await runAsync(suite, "show (PNG) scatter", "100 pts", async () => {
  freshFig();
  scatter(x100, y100);
  return show({ format: "png" });
});
await runAsync(suite, "show (PNG) heatmap", "25x25", async () => {
  freshFig();
  heatmap(hm25);
  return show({ format: "png" });
});
await runAsync(suite, "saveFig (PDF) line", "500 pts", async () => {
  freshFig();
  plot(x500, y500);
  await saveFig("/tmp/deepbox-bench-line.pdf", { format: "pdf" });
  return 1;
});

// ── Extended coverage (v1.1 benchmark expansion) ────────
const x50 = randn([50]);
const y50 = randn([50]);
const x200 = randn([200]);
const y200 = randn([200]);
const x1k = randn([1000]);
const y1k = randn([1000]);
const x10k = randn([10000]);
const y10k = randn([10000]);
const scatterSizes: [string, typeof x50, typeof y50][] = [
  ["50 pts", x50, y50],
  ["200 pts", x200, y200],
  ["1K pts", x1k, y1k],
  ["10K pts", x10k, y10k],
];
for (const [sz, xs, ys] of scatterSizes) {
  specPlot("scatter", sz, () => {
    freshFig();
    scatter(xs, ys);
  });
  specPlot("plot (line)", sz, () => {
    freshFig();
    plot(xs, ys);
  });
}
specPlot("hist", "1K bins=40", () => {
  freshFig();
  hist(x1k, 40);
});
specPlot("hist", "10K bins=60", () => {
  freshFig();
  hist(x10k, 60);
});
specPlot("bar", "500 bars", () => {
  freshFig();
  bar(arange(0, 500), randn([500]));
});
specPlot("barh", "200 bars", () => {
  freshFig();
  barh(arange(0, 200), randn([200]));
});
specPlot("boxplot", "1K pts", () => {
  freshFig();
  boxplot(x1k);
});
specPlot("violinplot", "1K pts", () => {
  freshFig();
  violinplot(x1k);
});

footer(suite, "deepbox-plot.json");
