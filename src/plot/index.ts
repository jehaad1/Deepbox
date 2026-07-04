/**
 * @see {@link https://deepbox.dev/docs/plot-basic | Deepbox documentation}
 */

// Re-export classes

export type {
  AnimationFrame,
  AnimationOptions,
  AnimationResult,
} from "./animation/Animation";
// Animation
export { Animation, createAnimation } from "./animation/Animation";
export { Axes } from "./figure/Axes";
export { Figure } from "./figure/Figure";
// Re-export state management functions
export { figure, gca, subplot } from "./figure/state";
export type {
  InteractiveOptions,
  InteractiveResult,
  TooltipDataPoint,
} from "./interactive/Interactive";
// Interactive
export {
  createInteractivePlot,
  InteractivePlot,
} from "./interactive/Interactive";
export type {
  Color,
  LegendOptions,
  PlotOptions,
  RenderedPDF,
  RenderedPNG,
  RenderedSVG,
} from "./types";
// Color palettes
export { getPalette, getPaletteColor, listPalettes } from "./utils/colors";

import { InvalidParameterError, ShapeError } from "../core";
// Global plotting functions
import { type Tensor, tensor } from "../ndarray";
import { gaussian_kde } from "../stats/kde";
import type { Figure } from "./figure/Figure";
import { Figure as FigureClass } from "./figure/Figure";
import { gca } from "./figure/state";
import type { LegendOptions, PlotOptions, RenderedPNG, RenderedSVG } from "./types";
import { tensorToFloat64Matrix2D } from "./utils/tensor";

/**
 * Render a figure to SVG or PNG.
 * @param options - Optional figure and format overrides
 */
export function show(
  options: { readonly figure?: Figure; readonly format?: "svg" | "png" } = {}
): RenderedSVG | Promise<RenderedPNG> {
  const fig = options.figure ?? gca().fig;
  if (options.format === "png") return fig.renderPNG();
  return fig.renderSVG();
}

/**
 * Save a figure to disk as SVG, PNG, or PDF.
 * @param path - Output file path (extension must match format)
 * @param options - Optional figure and format overrides
 */
export async function saveFig(
  path: string,
  options: {
    readonly figure?: Figure;
    readonly format?: "svg" | "png" | "pdf";
  } = {}
): Promise<void> {
  if (!path || path.trim().length === 0) {
    throw new InvalidParameterError("path must be a non-empty string", "path", path);
  }
  const dotIndex = path.lastIndexOf(".");
  const ext = dotIndex > 0 ? path.slice(dotIndex + 1).toLowerCase() : undefined;
  const fmt = options.format ?? (ext === "png" ? "png" : ext === "pdf" ? "pdf" : "svg");
  if (ext && ext !== fmt) {
    throw new InvalidParameterError(
      `File extension .${ext} does not match format ${fmt}`,
      "path",
      path
    );
  }
  const fig = options.figure ?? gca().fig;
  if (fmt === "png") {
    const { writeFile } = await import("node:fs/promises");
    const png = await fig.renderPNG();
    await writeFile(path, png.bytes);
  } else if (fmt === "pdf") {
    const { writeFile } = await import("node:fs/promises");
    const pdf = fig.renderPDF();
    await writeFile(path, pdf.bytes);
  } else {
    const { writeFile } = await import("node:fs/promises");
    const svg = fig.renderSVG();
    await writeFile(path, svg.svg, "utf-8");
  }
}

/**
 * Plot a connected line series on the current axes.
 */
export function plot(x: Tensor, y: Tensor, options: PlotOptions = {}): void {
  gca().plot(x, y, options);
}

/**
 * Plot unconnected points on the current axes.
 */
export function scatter(x: Tensor, y: Tensor, options: PlotOptions = {}): void {
  gca().scatter(x, y, options);
}

/**
 * Plot vertical bars on the current axes.
 */
export function bar(x: Tensor, height: Tensor, options: PlotOptions = {}): void {
  gca().bar(x, height, options);
}

/**
 * Plot horizontal bars on the current axes.
 */
export function barh(y: Tensor, width: Tensor, options: PlotOptions = {}): void {
  gca().barh(y, width, options);
}

/**
 * Draw a horizontal line across the current axes at the given y value.
 */
export function axhline(
  y: number,
  options: { color?: string; linewidth?: number; label?: string } = {}
): void {
  gca().axhline(y, options);
}

/**
 * Draw a vertical line across the current axes at the given x value.
 */
export function axvline(
  x: number,
  options: { color?: string; linewidth?: number; label?: string } = {}
): void {
  gca().axvline(x, options);
}

/**
 * Plot stacked vertical bars for multiple series on the current axes.
 */
export function stackedBar(
  x: Tensor,
  heights: readonly Tensor[],
  options: { colors?: readonly string[]; labels?: readonly string[] } = {}
): void {
  gca().stackedBar(x, heights, options);
}

/**
 * Plot grouped (side-by-side) vertical bars for multiple series on the current axes.
 */
export function groupedBar(
  x: Tensor,
  heights: readonly Tensor[],
  options: { colors?: readonly string[]; labels?: readonly string[] } = {}
): void {
  gca().groupedBar(x, heights, options);
}

/**
 * Plot a histogram on the current axes.
 */
export function hist(
  x: Tensor,
  bins?: number | (PlotOptions & { bins?: number }),
  options: PlotOptions = {}
): void {
  let resolvedBins = 10;
  let resolvedOptions = options;
  if (typeof bins === "object" && bins !== null) {
    resolvedBins = bins.bins ?? 10;
    const { bins: _b, ...rest } = bins;
    resolvedOptions = rest;
  } else if (typeof bins === "number") {
    resolvedBins = bins;
  }
  gca().hist(x, resolvedBins, resolvedOptions);
}

/**
 * Plot a box-and-whisker summary on the current axes.
 */
export function boxplot(data: Tensor, options: PlotOptions = {}): void {
  gca().boxplot(data, options);
}

/**
 * Plot a violin summary on the current axes.
 */
export function violinplot(data: Tensor, options: PlotOptions = {}): void {
  gca().violinplot(data, options);
}

/**
 * Plot a pie chart on the current axes.
 */
export function pie(values: Tensor, labels?: readonly string[], options: PlotOptions = {}): void {
  gca().pie(values, labels, options);
}

/**
 * Show or configure a legend on the current axes.
 */
export function legend(options: LegendOptions = {}): void {
  gca().legend(options);
}

/**
 * Plot a heatmap for a 2D tensor.
 */
export function heatmap(data: Tensor, options: PlotOptions = {}): void {
  gca().heatmap(data, options);
}

/**
 * Display a matrix as an image (alias of heatmap).
 */
export function imshow(data: Tensor, options: PlotOptions = {}): void {
  gca().imshow(data, options);
}

/**
 * Plot contour lines for a 2D grid.
 */
export function contour(X: Tensor, Y: Tensor, Z: Tensor, options: PlotOptions = {}): void {
  gca().contour(X, Y, Z, options);
}

/**
 * Plot filled contours for a 2D grid.
 */
export function contourf(X: Tensor, Y: Tensor, Z: Tensor, options: PlotOptions = {}): void {
  gca().contourf(X, Y, Z, options);
}

/**
 * Plot a confusion matrix as a heatmap.
 */
export function plotConfusionMatrix(
  cm: Tensor,
  labels?: readonly string[],
  options: PlotOptions = {}
): void {
  const ax = gca();
  ax.heatmap(cm, options);
  if (cm.ndim === 2) {
    const rows = cm.shape[0] ?? 0;
    const cols = cm.shape[1] ?? 0;
    ax.setTitle("Confusion Matrix");
    ax.setXLabel("Predicted");
    ax.setYLabel("Actual");
    if (labels && labels.length < Math.max(rows, cols)) {
      throw new InvalidParameterError(
        `labels length must be >= ${Math.max(rows, cols)}; received ${labels.length}`,
        "labels",
        labels
      );
    }
    if (labels && rows > 0 && cols > 0) {
      const xLabels = labels.slice(0, cols);
      const yLabels = labels.slice(0, rows);
      const xValues = xLabels.map((_, i) => i + 0.5);
      const yValues = yLabels.map((_, i) => i + 0.5);
      ax.setXTicks(xValues, xLabels);
      ax.setYTicks(yValues, yLabels);
    }
  }
}

/**
 * Plot a ROC curve with optional AUC annotation.
 */
export function plotRocCurve(
  fpr: Tensor,
  tpr: Tensor,
  auc?: number,
  options: PlotOptions = {}
): void {
  const ax = gca();
  ax.plot(fpr, tpr, {
    ...options,
    color: options.color ?? "#1f77b4",
    label: options.label ?? "ROC",
  });
  ax.plot(tensor([0, 1]), tensor([0, 1]), {
    color: "#999999",
    linewidth: 1,
    label: "Chance",
  });
  if (auc !== undefined) {
    ax.setTitle(`ROC Curve (AUC = ${auc.toFixed(3)})`);
  } else {
    ax.setTitle("ROC Curve");
  }
  ax.setXLabel("False Positive Rate");
  ax.setYLabel("True Positive Rate");
}

/**
 * Plot a precision-recall curve with optional AP annotation.
 */
export function plotPrecisionRecallCurve(
  precision: Tensor,
  recall: Tensor,
  averagePrecision?: number,
  options: PlotOptions = {}
): void {
  const ax = gca();
  ax.plot(recall, precision, {
    ...options,
    color: options.color ?? "#1f77b4",
    label: options.label ?? "Precision-Recall",
  });
  if (averagePrecision !== undefined) {
    ax.setTitle(`Precision-Recall Curve (AP = ${averagePrecision.toFixed(3)})`);
  } else {
    ax.setTitle("Precision-Recall Curve");
  }
  ax.setXLabel("Recall");
  ax.setYLabel("Precision");
}

/**
 * Plot training and validation learning curves.
 */
export function plotLearningCurve(
  trainSizes: Tensor,
  trainScores: Tensor,
  valScores: Tensor,
  options: PlotOptions = {}
): void {
  const ax = gca();
  const trainColor = options.colors?.[0] ?? options.color ?? "#1f77b4";
  const valColor = options.colors?.[1] ?? options.color ?? "#ff7f0e";
  ax.plot(trainSizes, trainScores, {
    color: trainColor,
    label: "Training Score",
  });
  ax.plot(trainSizes, valScores, {
    color: valColor,
    label: "Validation Score",
  });
  ax.setTitle("Learning Curve");
  ax.setXLabel("Training Set Size");
  ax.setYLabel("Score");
}

/**
 * Plot training and validation curves.
 */
export function plotValidationCurve(
  paramRange: Tensor,
  trainScores: Tensor,
  valScores: Tensor,
  options: PlotOptions = {}
): void {
  const ax = gca();
  const trainColor = options.colors?.[0] ?? options.color ?? "#1f77b4";
  const valColor = options.colors?.[1] ?? options.color ?? "#ff7f0e";
  ax.plot(paramRange, trainScores, {
    color: trainColor,
    label: "Training Score",
  });
  ax.plot(paramRange, valScores, {
    color: valColor,
    label: "Validation Score",
  });
  ax.setTitle("Validation Curve");
  ax.setXLabel("Parameter Value");
  ax.setYLabel("Score");
}

/**
 * Plot a classifier decision boundary on a 2D feature space.
 */
export function plotDecisionBoundary(
  X: Tensor,
  y: Tensor,
  model: { readonly predict: (x: Tensor) => Tensor },
  options: PlotOptions = {}
): void {
  if (X.dtype === "string") {
    throw new InvalidParameterError("plotDecisionBoundary: X must be numeric", "X", X.dtype);
  }
  if (X.ndim !== 2 || (X.shape[1] ?? 0) !== 2) {
    throw new ShapeError("plotDecisionBoundary: X must be shape [n, 2]");
  }
  const n = X.shape[0] ?? 0;
  if (n === 0) {
    throw new InvalidParameterError("plotDecisionBoundary: X must have at least one row", "X", n);
  }
  if (y.ndim !== 1 || (y.shape[0] ?? -1) !== n) {
    throw new ShapeError("plotDecisionBoundary: y must be shape [n]");
  }

  // Optimized feature reading using flat buffer
  const { data: xData } = tensorToFloat64Matrix2D(X);
  const x0 = new Float64Array(n);
  const x1 = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const vx = xData[i * 2] ?? NaN;
    const vy = xData[i * 2 + 1] ?? NaN;
    if (!Number.isFinite(vx) || !Number.isFinite(vy)) {
      throw new InvalidParameterError("plotDecisionBoundary: X must be finite", "X", {
        index: i,
        x: vx,
        y: vy,
      });
    }
    x0[i] = vx;
    x1[i] = vy;
  }

  const readLabel = (labelTensor: Tensor, index: number): string | number | bigint => {
    const value = labelTensor.at(index);
    if (typeof value === "string") return value;
    if (typeof value === "number") {
      if (!Number.isFinite(value)) {
        throw new InvalidParameterError("plotDecisionBoundary: invalid label value", "y", value);
      }
      return value;
    }
    if (typeof value === "bigint") return value;
    throw new InvalidParameterError("plotDecisionBoundary: invalid label value", "y", value);
  };
  const toFiniteNumber = (value: unknown): number => {
    if (typeof value === "number") {
      if (!Number.isFinite(value)) {
        throw new InvalidParameterError(
          "plotDecisionBoundary: model.predict must return finite numeric scores",
          "predict",
          value
        );
      }
      return value;
    }
    if (typeof value === "bigint") {
      const n = Number(value);
      if (!Number.isFinite(n)) {
        throw new InvalidParameterError(
          "plotDecisionBoundary: model.predict must return finite numeric scores",
          "predict",
          value
        );
      }
      return n;
    }
    throw new InvalidParameterError(
      "plotDecisionBoundary: model.predict must return numeric scores",
      "predict",
      value
    );
  };

  const labelKey = (value: string | number | bigint): string => `${typeof value}:${String(value)}`;

  let xMin = x0[0] ?? 0;
  let xMax = x0[0] ?? 0;
  let yMin = x1[0] ?? 0;
  let yMax = x1[0] ?? 0;
  for (let i = 1; i < n; i++) {
    const vx = x0[i] ?? 0;
    const vy = x1[i] ?? 0;
    if (vx < xMin) xMin = vx;
    if (vx > xMax) xMax = vx;
    if (vy < yMin) yMin = vy;
    if (vy > yMax) yMax = vy;
  }
  const marginX = xMax > xMin ? (xMax - xMin) * 0.05 : 1;
  const marginY = yMax > yMin ? (yMax - yMin) * 0.05 : 1;
  xMin -= marginX;
  xMax += marginX;
  yMin -= marginY;
  yMax += marginY;

  const gridResolution = 100;
  const gridCount = gridResolution * gridResolution;
  const gridFlat = new Float64Array(gridCount * 2);

  for (let gy = 0; gy < gridResolution; gy++) {
    const fy = yMin + ((yMax - yMin) * gy) / Math.max(1, gridResolution - 1);
    for (let gx = 0; gx < gridResolution; gx++) {
      const fx = xMin + ((xMax - xMin) * gx) / Math.max(1, gridResolution - 1);
      const idx = (gy * gridResolution + gx) * 2;
      gridFlat[idx] = fx;
      gridFlat[idx + 1] = fy;
    }
  }

  const gridTensor = tensor(gridFlat).reshape([gridCount, 2]);
  const predictions = model.predict(gridTensor);

  const predictedLabels: Array<string | number | bigint> = new Array(gridCount);
  if (predictions.ndim === 1) {
    if ((predictions.shape[0] ?? -1) !== gridCount) {
      throw new ShapeError("plotDecisionBoundary: model.predict must return [gridSize] labels");
    }
    for (let i = 0; i < gridCount; i++) {
      predictedLabels[i] = readLabel(predictions, i);
    }
  } else if (predictions.ndim === 2) {
    const rows = predictions.shape[0] ?? -1;
    const cols = predictions.shape[1] ?? -1;
    if (rows !== gridCount || cols <= 0) {
      throw new ShapeError("plotDecisionBoundary: model.predict must return [gridSize, k]");
    }
    for (let i = 0; i < rows; i++) {
      let bestCol = 0;
      let bestValue = toFiniteNumber(predictions.at(i, 0));
      for (let c = 1; c < cols; c++) {
        const value = toFiniteNumber(predictions.at(i, c));
        if (value > bestValue) {
          bestValue = value;
          bestCol = c;
        }
      }
      predictedLabels[i] = bestCol;
    }
  } else {
    throw new ShapeError("plotDecisionBoundary: model.predict output must be 1D or 2D");
  }

  const classIndex = new Map<string, number>();
  const classValues: Array<string | number | bigint> = [];
  for (let i = 0; i < n; i++) {
    const label = readLabel(y, i);
    const key = labelKey(label);
    if (!classIndex.has(key)) {
      classIndex.set(key, classValues.length);
      classValues.push(label);
    }
  }
  for (const label of predictedLabels) {
    const key = labelKey(label);
    if (!classIndex.has(key)) {
      classIndex.set(key, classValues.length);
      classValues.push(label);
    }
  }

  const boundaryMatrix: number[][] = Array.from({ length: gridResolution }, () =>
    Array(gridResolution).fill(0)
  );
  for (let gy = 0; gy < gridResolution; gy++) {
    const row = boundaryMatrix[gy];
    if (!row) {
      throw new InvalidParameterError("plotDecisionBoundary: grid row access failed", "gy", gy);
    }
    for (let gx = 0; gx < gridResolution; gx++) {
      const label = predictedLabels[gy * gridResolution + gx];
      const mapped = label === undefined ? 0 : (classIndex.get(labelKey(label)) ?? 0);
      row[gx] = mapped;
    }
  }

  const ax = gca();
  ax.imshow(tensor(boundaryMatrix), {
    colormap: options.colormap ?? "grayscale",
    vmin: options.vmin ?? 0,
    vmax: options.vmax ?? Math.max(1, classValues.length - 1),
    extent: { xmin: xMin, xmax: xMax, ymin: yMin, ymax: yMax },
  });

  const defaultClassColors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
  ];
  const classColors =
    options.colors !== undefined && options.colors.length > 0 ? options.colors : defaultClassColors;
  const byClass = new Map<string, { x: number[]; y: number[]; index: number; label: string }>();
  for (let i = 0; i < n; i++) {
    const label = readLabel(y, i);
    const key = labelKey(label);
    const labelText = String(label);
    const mapped = classIndex.get(key) ?? 0;
    const current = byClass.get(key) ?? {
      x: [],
      y: [],
      index: mapped,
      label: labelText,
    };
    current.x.push(x0[i] ?? 0);
    current.y.push(x1[i] ?? 0);
    byClass.set(key, current);
  }

  for (const group of byClass.values()) {
    const color = classColors[group.index % classColors.length] ?? "#000000";
    ax.scatter(tensor(group.x), tensor(group.y), {
      color,
      size: options.size ?? 4,
      label: group.label,
    });
  }

  ax.setTitle("Decision Boundary");
  ax.setXLabel("Feature 1");
  ax.setYLabel("Feature 2");
}

// ---- KDE Plot ----

/**
 * Plot a kernel density estimate of a 1D dataset.
 *
 * Uses Gaussian KDE to estimate the probability density function,
 * then plots it as a smooth line.
 */
export function kdeplot(
  data: Tensor,
  options: PlotOptions & {
    bw_method?: "scott" | "silverman" | number;
    gridSize?: number;
    fill?: boolean;
  } = {}
): void {
  const n = data.shape[0] ?? 0;
  if (n === 0) {
    throw new InvalidParameterError("kdeplot: data must have at least one element", "data", n);
  }

  const values: number[] = [];
  for (let i = 0; i < n; i++) {
    const v = data.at(i);
    if (typeof v === "number" && Number.isFinite(v)) values.push(v);
  }

  if (values.length === 0) {
    throw new InvalidParameterError(
      "kdeplot: data must contain finite numeric values",
      "data",
      data.dtype
    );
  }

  const bwOpts: { bw_method?: "scott" | "silverman" | number } = {};
  if (options.bw_method !== undefined) bwOpts.bw_method = options.bw_method;
  const kde = gaussian_kde(values, bwOpts);

  let min = values[0] ?? 0;
  let max = values[0] ?? 0;
  for (const v of values) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const range = max - min || 1;
  min -= range * 0.1;
  max += range * 0.1;

  const gridSize = options.gridSize ?? 200;
  const xGrid: number[] = [];
  for (let i = 0; i < gridSize; i++) {
    xGrid.push(min + ((max - min) * i) / (gridSize - 1));
  }

  const density = kde.evaluate(xGrid);

  const ax = gca();
  const xTensor = tensor(xGrid);
  const yTensor = tensor(Array.from(density));

  if (options.fill) {
    ax.area(xTensor, yTensor, {
      ...options,
      color: options.color ?? "#1f77b4",
      label: options.label ?? "KDE",
    });
  } else {
    ax.plot(xTensor, yTensor, {
      ...options,
      color: options.color ?? "#1f77b4",
      label: options.label ?? "KDE",
    });
  }
  ax.setXLabel("Value");
  ax.setYLabel("Density");
}

// ---- ML Plots ----

/**
 * Plot residuals (predicted vs. residual) for regression diagnostics.
 */
export function plotResiduals(yTrue: Tensor, yPred: Tensor, options: PlotOptions = {}): void {
  const n = yTrue.shape[0] ?? 0;
  if (n === 0) {
    throw new InvalidParameterError(
      "plotResiduals: yTrue must have at least one element",
      "yTrue",
      n
    );
  }
  if ((yPred.shape[0] ?? 0) !== n) {
    throw new ShapeError("plotResiduals: yTrue and yPred must have the same length");
  }

  const residuals: number[] = [];
  const predictions: number[] = [];
  for (let i = 0; i < n; i++) {
    const yt = yTrue.at(i);
    const yp = yPred.at(i);
    if (typeof yt === "number" && typeof yp === "number") {
      predictions.push(yp);
      residuals.push(yt - yp);
    }
  }

  const ax = gca();
  ax.scatter(tensor(predictions), tensor(residuals), {
    ...options,
    color: options.color ?? "#1f77b4",
    size: options.size ?? 3,
  });
  ax.axhline(0, { color: "#999999", linewidth: 1 });
  ax.setTitle("Residual Plot");
  ax.setXLabel("Predicted");
  ax.setYLabel("Residual");
}

/**
 * Plot feature importances as a horizontal bar chart.
 */
export function plotFeatureImportance(
  importances: Tensor,
  featureNames?: readonly string[],
  options: PlotOptions = {}
): void {
  const n = importances.shape[0] ?? 0;
  if (n === 0) {
    throw new InvalidParameterError(
      "plotFeatureImportance: importances must have at least one element",
      "importances",
      n
    );
  }

  // Sort by importance descending
  const items: Array<{ index: number; value: number }> = [];
  for (let i = 0; i < n; i++) {
    const v = importances.at(i);
    items.push({ index: i, value: typeof v === "number" ? v : 0 });
  }
  items.sort((a, b) => b.value - a.value);

  const yPos: number[] = [];
  const vals: number[] = [];
  const labels: string[] = [];
  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    if (!item) continue;
    yPos.push(i);
    vals.push(item.value);
    labels.push(featureNames?.[item.index] ?? `Feature ${item.index}`);
  }

  const ax = gca();
  ax.barh(tensor(yPos), tensor(vals), {
    ...options,
    color: options.color ?? "#1f77b4",
  });
  ax.setYTicks(yPos, labels);
  ax.setTitle("Feature Importance");
  ax.setXLabel("Importance");
}

/**
 * Plot an elbow curve for KMeans cluster selection.
 */
export function plotElbowCurve(kValues: Tensor, inertias: Tensor, options: PlotOptions = {}): void {
  const ax = gca();
  ax.plot(kValues, inertias, {
    ...options,
    color: options.color ?? "#1f77b4",
    label: options.label ?? "Inertia",
  });
  ax.setTitle("Elbow Curve");
  ax.setXLabel("Number of Clusters (k)");
  ax.setYLabel("Inertia");
}

/**
 * Plot silhouette scores for clustering evaluation.
 */
export function plotSilhouette(
  kValues: Tensor,
  silhouetteScores: Tensor,
  options: PlotOptions = {}
): void {
  const ax = gca();
  ax.plot(kValues, silhouetteScores, {
    ...options,
    color: options.color ?? "#1f77b4",
    label: options.label ?? "Silhouette Score",
  });
  ax.setTitle("Silhouette Analysis");
  ax.setXLabel("Number of Clusters (k)");
  ax.setYLabel("Silhouette Score");
}

/**
 * Plot a calibration curve (reliability diagram).
 */
export function plotCalibrationCurve(
  fractionPositive: Tensor,
  meanPredicted: Tensor,
  options: PlotOptions = {}
): void {
  const ax = gca();
  ax.plot(meanPredicted, fractionPositive, {
    ...options,
    color: options.color ?? "#1f77b4",
    label: options.label ?? "Calibration",
  });
  // Perfect calibration reference line
  ax.plot(tensor([0, 1]), tensor([0, 1]), {
    color: "#999999",
    linewidth: 1,
    label: "Perfectly Calibrated",
  });
  ax.setTitle("Calibration Curve");
  ax.setXLabel("Mean Predicted Probability");
  ax.setYLabel("Fraction of Positives");
}

// ---- Pair Plot ----

/**
 * Create a pair plot (matrix of scatter plots) for all variable pairs.
 *
 * Similar to Seaborn's pairplot. Creates an n×n grid of subplots where
 * diagonal shows histograms and off-diagonal shows scatter plots.
 */
export function pairplot(
  data: Tensor,
  options: {
    featureNames?: readonly string[];
    color?: string;
    size?: number;
    diagKind?: "hist" | "kde";
  } = {}
): Figure {
  if (data.ndim !== 2) {
    throw new ShapeError("pairplot: data must be a 2D tensor [n_samples, n_features]");
  }

  const nSamples = data.shape[0] ?? 0;
  const nFeatures = data.shape[1] ?? 0;

  if (nSamples === 0 || nFeatures === 0) {
    throw new InvalidParameterError(
      "pairplot: data must have at least one sample and one feature",
      "data",
      data.shape
    );
  }

  const cellSize = 200;
  const totalW = cellSize * nFeatures;
  const totalH = cellSize * nFeatures;
  const fig = new FigureClass({ width: totalW, height: totalH });

  // Extract columns
  const columns: number[][] = [];
  for (let j = 0; j < nFeatures; j++) {
    const col: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      const v = data.at(i, j);
      col.push(typeof v === "number" ? v : 0);
    }
    columns.push(col);
  }

  const featureNames = options.featureNames ?? columns.map((_, i) => `Feature ${i}`);

  for (let row = 0; row < nFeatures; row++) {
    for (let c = 0; c < nFeatures; c++) {
      const vp = {
        x: c * cellSize,
        y: row * cellSize,
        width: cellSize,
        height: cellSize,
      };
      const ax = fig.addAxes({ viewport: vp });

      if (row === c) {
        // Diagonal: histogram
        const colData = columns[row];
        if (colData) {
          ax.hist(tensor(colData), 20, {
            color: options.color ?? "#1f77b4",
          });
        }
      } else {
        // Off-diagonal: scatter
        const xCol = columns[c];
        const yCol = columns[row];
        if (xCol && yCol) {
          ax.scatter(tensor(xCol), tensor(yCol), {
            color: options.color ?? "#1f77b4",
            size: options.size ?? 2,
          });
        }
      }

      // Labels only on edges
      if (row === nFeatures - 1) {
        ax.setXLabel(featureNames[c] ?? "");
      }
      if (c === 0) {
        ax.setYLabel(featureNames[row] ?? "");
      }
    }
  }

  return fig;
}

// ---- Themes / Styles ----

/** A theme configuration for consistent plot styling. */
export interface PlotTheme {
  /** Background color for figure. */
  readonly figureFacecolor: string;
  /** Background color for axes. */
  readonly axesFacecolor: string;
  /** Default line/bar color. */
  readonly primaryColor: string;
  /** Color cycle for multiple series. */
  readonly colorCycle: readonly string[];
  /** Default font size. */
  readonly fontSize: number;
  /** Grid color. */
  readonly gridColor: string;
  /** Whether grid is visible by default. */
  readonly gridVisible: boolean;
}

const themes: Record<string, PlotTheme> = {
  default: {
    figureFacecolor: "#ffffff",
    axesFacecolor: "#ffffff",
    primaryColor: "#1f77b4",
    colorCycle: [
      "#1f77b4",
      "#ff7f0e",
      "#2ca02c",
      "#d62728",
      "#9467bd",
      "#8c564b",
      "#e377c2",
      "#7f7f7f",
      "#bcbd22",
      "#17becf",
    ],
    fontSize: 12,
    gridColor: "#cccccc",
    gridVisible: false,
  },
  dark: {
    figureFacecolor: "#1e1e1e",
    axesFacecolor: "#2d2d2d",
    primaryColor: "#58a6ff",
    colorCycle: [
      "#58a6ff",
      "#f0883e",
      "#3fb950",
      "#f85149",
      "#bc8cff",
      "#db6d28",
      "#f778ba",
      "#8b949e",
      "#d2a825",
      "#39d0d0",
    ],
    fontSize: 12,
    gridColor: "#444444",
    gridVisible: true,
  },
  paper: {
    figureFacecolor: "#ffffff",
    axesFacecolor: "#ffffff",
    primaryColor: "#333333",
    colorCycle: [
      "#333333",
      "#666666",
      "#999999",
      "#bbbbbb",
      "#444444",
      "#777777",
      "#aaaaaa",
      "#555555",
      "#888888",
      "#cccccc",
    ],
    fontSize: 10,
    gridColor: "#e0e0e0",
    gridVisible: true,
  },
  presentation: {
    figureFacecolor: "#ffffff",
    axesFacecolor: "#fafafa",
    primaryColor: "#2563eb",
    colorCycle: [
      "#2563eb",
      "#dc2626",
      "#16a34a",
      "#ea580c",
      "#9333ea",
      "#0891b2",
      "#db2777",
      "#65a30d",
      "#ca8a04",
      "#4f46e5",
    ],
    fontSize: 16,
    gridColor: "#d4d4d4",
    gridVisible: true,
  },
};

// ─── New P2/P3 plot functions ─────────────────────────────────────────────

/**
 * Stem plot: vertical lines from baseline to data points with circle markers.
 */
export function stem(
  x: Tensor,
  y: Tensor,
  options: PlotOptions & { baseline?: number } = {}
): void {
  gca().stem(x, y, options);
}

/**
 * Strip plot: jittered categorical scatter plot (Seaborn-style).
 */
export function strip(
  groups: readonly Tensor[],
  options: {
    colors?: readonly string[];
    labels?: readonly string[];
    size?: number;
    jitter?: number;
  } = {}
): void {
  gca().strip(groups, options);
}

/**
 * Radar / spider chart: multi-dimensional data on radial axes.
 */
export function radar(
  series: readonly Tensor[],
  options: {
    colors?: readonly string[];
    labels?: readonly string[];
    linewidth?: number;
  } = {}
): void {
  gca().radar(series, options);
}

/**
 * Waterfall chart: cumulative positive/negative value changes.
 */
export function waterfall(
  categories: readonly string[],
  values: Tensor,
  options: {
    positiveColor?: string;
    negativeColor?: string;
    totalColor?: string;
    barWidth?: number;
  } = {}
): void {
  gca().waterfall(categories, values, options);
}

/**
 * Quiver plot: vector field arrows.
 */
export function quiver(
  x: Tensor,
  y: Tensor,
  u: Tensor,
  v: Tensor,
  options: PlotOptions & { scale?: number } = {}
): void {
  gca().quiver(x, y, u, v, options);
}

/**
 * Polar plot: data in polar coordinates (theta, r).
 */
export function polar(
  theta: Tensor,
  r: Tensor,
  options: PlotOptions & { fill?: boolean } = {}
): void {
  gca().polar(theta, r, options);
}

/**
 * Joint plot: scatter plot with marginal histograms (Seaborn-style `jointplot`).
 * Creates a new figure with a central scatter plot and marginal histograms on top and right.
 */
export function jointplot(
  x: Tensor,
  y: Tensor,
  options: {
    color?: string;
    bins?: number;
    xlabel?: string;
    ylabel?: string;
    title?: string;
  } = {}
): Figure {
  const fig = new FigureClass({ width: 600, height: 600 });
  const margin = 40;
  const mainW = 400;
  const mainH = 400;
  const margW = 600 - mainW - margin;
  const margH = 600 - mainH - margin;

  // Main scatter axes (bottom-left)
  const mainAx = fig.addAxes({
    padding: margin,
    viewport: { x: 0, y: margH, width: mainW, height: mainH },
  });
  // Top marginal histogram
  const topAx = fig.addAxes({
    padding: margin,
    viewport: { x: 0, y: 0, width: mainW, height: margH },
  });
  // Right marginal histogram
  const rightAx = fig.addAxes({
    padding: margin,
    viewport: { x: mainW, y: margH, width: margW, height: mainH },
  });

  const clr = options.color ?? "#1f77b4";
  const bins = options.bins ?? 20;

  // Main scatter
  mainAx.scatter(x, y, { color: clr, size: 3 });
  if (options.xlabel) mainAx.setXLabel(options.xlabel);
  if (options.ylabel) mainAx.setYLabel(options.ylabel);

  // Top marginal histogram
  topAx.hist(x, bins, { color: clr });

  // Right marginal histogram
  rightAx.hist(y, bins, { color: clr });

  if (options.title) mainAx.setTitle(options.title);

  return fig;
}

/**
 * Plot a dendrogram from a linkage matrix.
 *
 * The linkage matrix has shape (n-1, 4) where each row is
 * [clusterA, clusterB, distance, count].
 *
 * @param linkage - Linkage matrix rows
 * @param nLeaves - Number of original observations
 * @param options - Styling options
 */
export function plotDendrogram(
  linkage: readonly (readonly [number, number, number, number])[],
  nLeaves: number,
  options: PlotOptions = {}
): void {
  gca().dendrogram(linkage, nLeaves, options);
}

let _currentTheme: PlotTheme = themes["default"]!;

/**
 * Set the global plot theme.
 * @param name - Theme name: "default", "dark", "paper", or "presentation"
 */
export function setTheme(name: string): void {
  const theme = themes[name];
  if (!theme) {
    throw new InvalidParameterError(
      `Unknown theme "${name}". Available: ${Object.keys(themes).join(", ")}`,
      "name",
      name
    );
  }
  _currentTheme = theme;
}

/**
 * Get the current plot theme.
 */
export function getTheme(): PlotTheme {
  return _currentTheme;
}

/**
 * Reset the theme to default.
 */
export function resetTheme(): void {
  _currentTheme = themes["default"]!;
}

/**
 * List available theme names.
 */
export function listThemes(): readonly string[] {
  return Object.keys(themes);
}

// ─── 3D Plots ────────────────────────────────────────────────────────────────

export {
  type Plot3DOptions,
  Scatter3D,
  Surface3D,
  Wireframe3D,
} from "./plots/Surface3D";

/**
 * Create a 3D surface plot.
 *
 * @param xGrid - 2D array of x-coordinates (rows × cols)
 * @param yGrid - 2D array of y-coordinates (rows × cols)
 * @param zGrid - 2D array of z-values (rows × cols)
 * @param options - 3D plot options (elevation, azimuth, color, etc.)
 * @returns The Axes containing the surface plot
 *
 * @example
 * ```ts
 * import { surface } from 'deepbox/plot';
 *
 * const x: number[][] = [];
 * const y: number[][] = [];
 * const z: number[][] = [];
 * for (let i = 0; i < 20; i++) {
 *   x.push([]); y.push([]); z.push([]);
 *   for (let j = 0; j < 20; j++) {
 *     const xi = -2 + (4 * i) / 19;
 *     const yj = -2 + (4 * j) / 19;
 *     x[i].push(xi);
 *     y[i].push(yj);
 *     z[i].push(Math.sin(Math.sqrt(xi * xi + yj * yj)));
 *   }
 * }
 * surface(x, y, z);
 * ```
 */
export function surface(
  xGrid: number[][],
  yGrid: number[][],
  zGrid: number[][],
  options: import("./plots/Surface3D").Plot3DOptions = {}
): void {
  const ax = gca();
  const xF = xGrid.map((r) => new Float64Array(r));
  const yF = yGrid.map((r) => new Float64Array(r));
  const zF = zGrid.map((r) => new Float64Array(r));
  ax.surface(xF, yF, zF, options);
}

/**
 * Create a 3D wireframe plot.
 *
 * @param xGrid - 2D array of x-coordinates (rows × cols)
 * @param yGrid - 2D array of y-coordinates (rows × cols)
 * @param zGrid - 2D array of z-values (rows × cols)
 * @param options - 3D plot options (elevation, azimuth, color, etc.)
 *
 * @example
 * ```ts
 * import { wireframe } from 'deepbox/plot';
 * wireframe(x, y, z, { color: '#333', elevation: 45 });
 * ```
 */
export function wireframe(
  xGrid: number[][],
  yGrid: number[][],
  zGrid: number[][],
  options: import("./plots/Surface3D").Plot3DOptions = {}
): void {
  const ax = gca();
  const xF = xGrid.map((r) => new Float64Array(r));
  const yF = yGrid.map((r) => new Float64Array(r));
  const zF = zGrid.map((r) => new Float64Array(r));
  ax.wireframe(xF, yF, zF, options);
}

/**
 * Create a 3D scatter plot.
 *
 * @param x - 1D array of x-coordinates
 * @param y - 1D array of y-coordinates
 * @param z - 1D array of z-coordinates
 * @param options - 3D plot options (elevation, azimuth, color, size, etc.)
 *
 * @example
 * ```ts
 * import { scatter3d } from 'deepbox/plot';
 * scatter3d([1, 2, 3], [4, 5, 6], [7, 8, 9], { color: 'red' });
 * ```
 */
export function scatter3d(
  x: number[],
  y: number[],
  z: number[],
  options: import("./plots/Surface3D").Plot3DOptions = {}
): void {
  const ax = gca();
  ax.scatter3d(new Float64Array(x), new Float64Array(y), new Float64Array(z), options);
}
