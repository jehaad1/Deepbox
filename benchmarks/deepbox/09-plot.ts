/**
 * Benchmark 09 — Plotting
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
  heatmap,
  hist,
  imshow,
  pie,
  plot,
  plotConfusionMatrix,
  plotLearningCurve,
  plotPrecisionRecallCurve,
  plotRocCurve,
  plotValidationCurve,
  scatter,
  show,
  violinplot,
} from "deepbox/plot";
import { createSuite, footer, header, run } from "../utils";

const suite = createSuite("plot");
header("Benchmark 09 — Plotting");

// ── Helpers ─────────────────────────────────────────────

function freshFig() {
  figure();
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

run(suite, "scatter", "20 pts", () => {
  freshFig();
  scatter(x20, y20);
});
run(suite, "scatter", "100 pts", () => {
  freshFig();
  scatter(x100, y100);
});
run(suite, "scatter", "500 pts", () => {
  freshFig();
  scatter(x500, y500);
});
run(suite, "scatter", "2K pts", () => {
  freshFig();
  scatter(x2k, y2k);
});
run(suite, "scatter", "5K pts", () => {
  freshFig();
  scatter(x5k, y5k);
});

// ── line plot ───────────────────────────────────────────

run(suite, "plot (line)", "20 pts", () => {
  freshFig();
  plot(x20, y20);
});
run(suite, "plot (line)", "100 pts", () => {
  freshFig();
  plot(x100, y100);
});
run(suite, "plot (line)", "500 pts", () => {
  freshFig();
  plot(x500, y500);
});
run(suite, "plot (line)", "2K pts", () => {
  freshFig();
  plot(x2k, y2k);
});
run(suite, "plot (line)", "5K pts", () => {
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

run(suite, "bar", "10 bars", () => {
  freshFig();
  bar(barX10, barH10);
});
run(suite, "bar", "50 bars", () => {
  freshFig();
  bar(barX50, barH50);
});
run(suite, "bar", "200 bars", () => {
  freshFig();
  bar(barX200, barH200);
});

// ── barh ────────────────────────────────────────────────

run(suite, "barh", "10 bars", () => {
  freshFig();
  barh(barX10, barH10);
});
run(suite, "barh", "50 bars", () => {
  freshFig();
  barh(barX50, barH50);
});

// ── hist ────────────────────────────────────────────────

run(suite, "hist", "100 bins=10", () => {
  freshFig();
  hist(x100, 10);
});
run(suite, "hist", "500 bins=20", () => {
  freshFig();
  hist(x500, 20);
});
run(suite, "hist", "2K bins=30", () => {
  freshFig();
  hist(x2k, 30);
});
run(suite, "hist", "5K bins=50", () => {
  freshFig();
  hist(x5k, 50);
});

// ── boxplot ─────────────────────────────────────────────

run(suite, "boxplot", "100 pts", () => {
  freshFig();
  boxplot(x100);
});
run(suite, "boxplot", "500 pts", () => {
  freshFig();
  boxplot(x500);
});
run(suite, "boxplot", "2K pts", () => {
  freshFig();
  boxplot(x2k);
});

// ── violinplot ──────────────────────────────────────────

run(suite, "violinplot", "100 pts", () => {
  freshFig();
  violinplot(x100);
});
run(suite, "violinplot", "500 pts", () => {
  freshFig();
  violinplot(x500);
});

// ── pie ─────────────────────────────────────────────────

const pieVals5 = tensor([30, 25, 20, 15, 10]);
const pieVals10 = tensor([15, 12, 11, 10, 9, 8, 8, 7, 10, 10]);

run(suite, "pie", "5 slices", () => {
  freshFig();
  pie(pieVals5, ["A", "B", "C", "D", "E"]);
});
run(suite, "pie", "10 slices", () => {
  freshFig();
  pie(pieVals10);
});

// ── heatmap ─────────────────────────────────────────────

const hm10 = randn([10, 10]);
const hm25 = randn([25, 25]);
const hm50 = randn([50, 50]);

run(suite, "heatmap", "10x10", () => {
  freshFig();
  heatmap(hm10);
});
run(suite, "heatmap", "25x25", () => {
  freshFig();
  heatmap(hm25);
});
run(suite, "heatmap", "50x50", () => {
  freshFig();
  heatmap(hm50);
});

// ── imshow ──────────────────────────────────────────────

run(suite, "imshow", "10x10", () => {
  freshFig();
  imshow(hm10);
});
run(suite, "imshow", "50x50", () => {
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

run(suite, "contour", "20x20", () => {
  freshFig();
  contour(cX, cY, cZ);
});
run(suite, "contour", "40x40", () => {
  freshFig();
  contour(cX40, cY40, cZ40);
});
run(suite, "contourf", "20x20", () => {
  freshFig();
  contourf(cX, cY, cZ);
});
run(suite, "contourf", "40x40", () => {
  freshFig();
  contourf(cX40, cY40, cZ40);
});

// ── ML Plots ────────────────────────────────────────────

const cm3 = tensor([
  [45, 3, 2],
  [4, 40, 6],
  [1, 5, 44],
]);
run(suite, "plotConfusionMatrix", "3x3", () => {
  freshFig();
  plotConfusionMatrix(cm3, ["A", "B", "C"]);
});

const fpr = linspace(0, 1, 100);
const tpr = tensor(Array.from({ length: 100 }, (_, i) => Math.min(1, (i / 100) ** 0.5)));
run(suite, "plotRocCurve", "100 pts", () => {
  freshFig();
  plotRocCurve(fpr, tpr, 0.85);
});

const prec = tensor(Array.from({ length: 100 }, (_, i) => 1 - i / 100));
const rec = linspace(0, 1, 100);
run(suite, "plotPrecisionRecallCurve", "100 pts", () => {
  freshFig();
  plotPrecisionRecallCurve(prec, rec, 0.75);
});

const trainSizes = tensor([10, 20, 50, 100, 200]);
const trainScores = tensor([0.6, 0.7, 0.8, 0.85, 0.9]);
const valScores = tensor([0.5, 0.6, 0.7, 0.75, 0.78]);
run(suite, "plotLearningCurve", "5 pts", () => {
  freshFig();
  plotLearningCurve(trainSizes, trainScores, valScores);
});

const paramRange = tensor([0.001, 0.01, 0.1, 1, 10]);
run(suite, "plotValidationCurve", "5 pts", () => {
  freshFig();
  plotValidationCurve(paramRange, trainScores, valScores);
});

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

footer(suite, "deepbox-plot.json");
