// Re-export classes
export { Axes } from "./figure/Axes";
export { Figure } from "./figure/Figure";
// Re-export state management functions
export { figure, gca, subplot } from "./figure/state";
export type {
  Color,
  LegendOptions,
  PlotOptions,
  RenderedPNG,
  RenderedSVG,
} from "./types";

import { InvalidParameterError, ShapeError } from "../core";
// Global plotting functions
import { type Tensor, tensor } from "../ndarray";
import type { Figure } from "./figure/Figure";
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
 * Save a figure to disk as SVG or PNG.
 * @param path - Output file path (extension must match format)
 * @param options - Optional figure and format overrides
 */
export async function saveFig(
  path: string,
  options: { readonly figure?: Figure; readonly format?: "svg" | "png" } = {}
): Promise<void> {
  if (!path || path.trim().length === 0) {
    throw new InvalidParameterError("path must be a non-empty string", "path", path);
  }
  const dotIndex = path.lastIndexOf(".");
  const ext = dotIndex > 0 ? path.slice(dotIndex + 1).toLowerCase() : undefined;
  const fmt = options.format ?? (ext === "png" ? "png" : "svg");
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
 * Plot a histogram on the current axes.
 */
export function hist(x: Tensor, bins = 10, options: PlotOptions = {}): void {
  gca().hist(x, bins, options);
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
