/**
 * @see {@link https://deepbox.dev/docs/plot-basic | Deepbox documentation}
 */

// Re-export classes

export type {
  AnimationEasing,
  AnimationFrame,
  AnimationFrameGenerator,
  AnimationOptions,
  AnimationResult,
} from "./animation/Animation";
// Animation
export { Animation, createAnimation } from "./animation/Animation";
export { Axes } from "./figure/Axes";
export { Figure } from "./figure/Figure";
// Re-export state management functions
export { figure, gca, gcf, sca, subplot } from "./figure/state";
// Themes
export type { PlotTheme } from "./figure/theme";
export { getTheme, listThemes, resetTheme, setTheme } from "./figure/theme";
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
export type { ViolinplotOptions } from "./plots/Violinplot";
export type {
  Color,
  ColormapName,
  LegendOptions,
  PlotOptions,
  RenderedPDF,
  RenderedPNG,
  RenderedSVG,
  TextOptions,
} from "./types";
// Color palettes
export { getPalette, getPaletteColor, listPalettes } from "./utils/colors";

import { InvalidParameterError, ShapeError } from "../core";
// Global plotting functions
import { type Tensor, tensor } from "../ndarray";
import { gaussianKde } from "../stats/kde";
import type { Axes } from "./figure/Axes";
import type { Figure } from "./figure/Figure";
import { Figure as FigureClass } from "./figure/Figure";
import { gca, sca } from "./figure/state";
import { getTheme, themeCycle, themePrimary } from "./figure/theme";
import type { ViolinplotOptions } from "./plots/Violinplot";
import type { LegendOptions, PlotOptions, RenderedPNG, RenderedSVG, TextOptions } from "./types";
import { tensorToFloat64Matrix2D, tensorToFloat64Vector1D } from "./utils/tensor";

/**
 * Convert a 1D tensor to a Float64Array, naming the calling function and
 * argument in the error when the input is not 1D.
 * @internal
 */
function vector1D(t: Tensor, fn: string, name: string): Float64Array {
  if (t.ndim !== 1) {
    throw new ShapeError(`${fn}: ${name} must be a 1D tensor; received ndim=${t.ndim}`);
  }
  return tensorToFloat64Vector1D(t);
}

/**
 * Reduce a learning/validation-curve score tensor to one value per x position.
 * A 1D tensor is used as is; a 2D tensor `[n_points, n_folds]` (the layout
 * returned by scikit-learn's `learning_curve` and `validation_curve`) is
 * averaged over the fold axis.
 * @internal
 */
function meanOverFolds(scores: Tensor, fn: string, name: string): Tensor {
  if (scores.ndim === 1) return scores;
  if (scores.ndim !== 2) {
    throw new ShapeError(
      `${fn}: ${name} must be 1D or 2D [n_points, n_folds]; received ndim=${scores.ndim}`
    );
  }
  const { rows, cols, data } = tensorToFloat64Matrix2D(scores);
  if (cols === 0) {
    throw new InvalidParameterError(`${fn}: ${name} must have at least one fold`, name, cols);
  }
  const out = new Float64Array(rows);
  for (let i = 0; i < rows; i++) {
    let sum = 0;
    const base = i * cols;
    for (let j = 0; j < cols; j++) sum += data[base + j] ?? 0;
    out[i] = sum / cols;
  }
  return tensor(out);
}

/**
 * Render a figure to SVG or PNG.
 *
 * @param options - Optional figure (default: the current figure) and format
 *   (default: `"svg"`). PNG rendering is asynchronous and Node.js only.
 * @returns A rendered SVG synchronously, or a promise of a rendered PNG.
 * @throws {InvalidParameterError} If `format` is not `"svg"` or `"png"`.
 */
export function show(options: {
  readonly figure?: Figure;
  readonly format: "png";
}): Promise<RenderedPNG>;
export function show(options?: { readonly figure?: Figure; readonly format?: "svg" }): RenderedSVG;
export function show(options?: {
  readonly figure?: Figure;
  readonly format?: "svg" | "png";
}): RenderedSVG | Promise<RenderedPNG>;
export function show(
  options: { readonly figure?: Figure; readonly format?: "svg" | "png" } = {}
): RenderedSVG | Promise<RenderedPNG> {
  if (options.format !== undefined && options.format !== "svg" && options.format !== "png") {
    throw new InvalidParameterError(
      `show: format must be "svg" or "png"; received ${String(options.format)}`,
      "format",
      options.format
    );
  }
  const fig = options.figure ?? gca().fig;
  if (options.format === "png") return fig.renderPNG();
  return fig.renderSVG();
}

/**
 * Save a figure to disk as SVG, PNG, or PDF.
 *
 * The format is taken from `options.format` when given, otherwise from the
 * file extension (`.png`, `.pdf`), and falls back to SVG for paths without an
 * extension.
 *
 * @param path - Output file path (extension must match the format)
 * @param options - Optional figure (default: the current figure) and format
 * @throws {InvalidParameterError} If the path is empty, the format is unknown,
 *   or the file extension does not match the format.
 */
export async function saveFig(
  path: string,
  options: {
    readonly figure?: Figure;
    readonly format?: "svg" | "png" | "pdf";
  } = {}
): Promise<void> {
  if (typeof path !== "string" || path.trim().length === 0) {
    throw new InvalidParameterError("path must be a non-empty string", "path", path);
  }
  if (
    options.format !== undefined &&
    options.format !== "svg" &&
    options.format !== "png" &&
    options.format !== "pdf"
  ) {
    throw new InvalidParameterError(
      `saveFig: format must be "svg", "png" or "pdf"; received ${String(options.format)}`,
      "format",
      options.format
    );
  }
  // Only look for an extension in the file name, not in directory names such as "out.v2/plot".
  const baseName = path.slice(Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\")) + 1);
  const dotIndex = baseName.lastIndexOf(".");
  const ext = dotIndex > 0 ? baseName.slice(dotIndex + 1).toLowerCase() : undefined;
  const fmt = options.format ?? (ext === "png" ? "png" : ext === "pdf" ? "pdf" : "svg");
  if (ext && ext !== fmt) {
    throw new InvalidParameterError(
      `File extension .${ext} does not match format ${fmt}`,
      "path",
      path
    );
  }
  const fig = options.figure ?? gca().fig;
  const { writeFile } = await import("node:fs/promises");
  if (fmt === "png") {
    const png = await fig.renderPNG();
    await writeFile(path, png.bytes);
  } else if (fmt === "pdf") {
    const pdf = fig.renderPDF();
    await writeFile(path, pdf.bytes);
  } else {
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
 *
 * @param x - 1D tensor of sample values
 * @param bins - Number of bins (default 10), or an options object that may
 *   carry `bins`. When an options object is passed here it is merged over the
 *   third argument.
 * @param options - Styling options
 */
export function hist(
  x: Tensor,
  bins?: number | (PlotOptions & { bins?: number }),
  options: PlotOptions = {}
): void {
  let resolvedBins = 10;
  let resolvedOptions = options;
  if (typeof bins === "object" && bins !== null) {
    const { bins: nestedBins, ...rest } = bins;
    const { bins: optionBins, ...baseOptions } = options;
    resolvedBins = nestedBins ?? optionBins ?? 10;
    resolvedOptions = { ...baseOptions, ...rest };
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
 * Plot a violin summary on the current axes. `options` may carry `position`, `width` and
 * `bandwidth` ("silverman", "scott" or a number) besides the usual styling options.
 * @see {@link Axes.violinplot}
 */
export function violinplot(data: Tensor, options: ViolinplotOptions = {}): void {
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
 * Plot a heatmap for a 2D tensor. Row 0 is drawn at the bottom of the y axis unless
 * `options.origin` is "upper".
 */
export function heatmap(data: Tensor, options: PlotOptions = {}): void {
  gca().heatmap(data, options);
}

/**
 * Display a matrix as an image (alias of heatmap). Unlike matplotlib's `imshow`, row 0 is
 * drawn at the bottom of the y axis unless `options.origin` is "upper".
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
 * Plot a step function on the current axes.
 *
 * @param x - 1D tensor of x coordinates
 * @param y - 1D tensor of y coordinates
 * @param options - Styling options and `where`: "post" (default), "pre" or "mid"
 * @see {@link Axes.step}
 */
export function step(
  x: Tensor,
  y: Tensor,
  options: PlotOptions & { where?: "pre" | "post" | "mid" } = {}
): void {
  gca().step(x, y, options);
}

/**
 * Plot error bars on the current axes.
 *
 * @param x - 1D tensor of x coordinates
 * @param y - 1D tensor of y coordinates
 * @param yerr - Non-negative errors: a 1D tensor for symmetric bars, or a `[2, n]` tensor with
 *   the lower and upper errors
 * @param options - Styling options
 */
export function errorbar(x: Tensor, y: Tensor, yerr: Tensor, options: PlotOptions = {}): void {
  gca().errorbar(x, y, yerr, options);
}

/**
 * Fill the area between two curves on the current axes (like `plt.fill_between`).
 *
 * @param x - 1D tensor of x coordinates
 * @param y1 - 1D tensor of the first curve
 * @param y2 - 1D tensor of the second curve
 * @param options - Styling options
 */
export function fillBetween(x: Tensor, y1: Tensor, y2: Tensor, options: PlotOptions = {}): void {
  gca().fillBetween(x, y1, y2, options);
}

/**
 * Fill the area between a curve and y = 0 on the current axes.
 *
 * @param x - 1D tensor of x coordinates
 * @param y - 1D tensor of y values
 * @param options - Styling options
 */
export function area(x: Tensor, y: Tensor, options: PlotOptions = {}): void {
  gca().area(x, y, options);
}

/**
 * Set the x-axis limits of the current axes. `min` may be larger than `max` to reverse
 * the axis.
 */
export function xlim(min: number, max: number): void {
  gca().xlim(min, max);
}

/**
 * Set the y-axis limits of the current axes. `min` may be larger than `max` to reverse
 * the axis.
 */
export function ylim(min: number, max: number): void {
  gca().ylim(min, max);
}

/**
 * Show or hide the grid of the current axes.
 *
 * @param visible - Whether the grid is drawn (default: true)
 * @param options - Grid line color
 */
export function grid(visible = true, options: { color?: string } = {}): void {
  gca().grid(visible, options);
}

/**
 * Add text at data coordinates on the current axes.
 *
 * @param x - X position in data coordinates
 * @param y - Y position in data coordinates
 * @param s - The text
 * @param options - Color, font size and alignment (`ha`, `va`)
 * @see {@link Axes.annotate}
 */
export function text(x: number, y: number, s: string, options: TextOptions = {}): void {
  gca().text(x, y, s, options);
}

/**
 * Add a text annotation at data coordinates on the current axes. Same as {@link text}, with
 * the text as the first argument.
 *
 * @param label - The text
 * @param x - X position in data coordinates
 * @param y - Y position in data coordinates
 * @param options - Color, font size and alignment (`ha`, `va`)
 */
export function annotate(label: string, x: number, y: number, options: TextOptions = {}): void {
  gca().annotate(label, x, y, options);
}

/**
 * Set the title of the current axes.
 */
export function title(label: string): void {
  gca().setTitle(label);
}

/**
 * Set the x-axis label of the current axes.
 */
export function xlabel(label: string): void {
  gca().setXLabel(label);
}

/**
 * Set the y-axis label of the current axes.
 */
export function ylabel(label: string): void {
  gca().setYLabel(label);
}

/**
 * Create a twin of the current axes that shares its x axis and has its own y axis on the
 * right, and make the twin the current axes (like `plt.twinx`). Call {@link sca} with the
 * original axes to draw on it again.
 *
 * @returns The new axes
 */
export function twinx(): Axes {
  return sca(gca().twinx());
}

/**
 * Plot a confusion matrix as a heatmap.
 *
 * Rows are the actual classes and columns the predicted classes. Class 0 is
 * drawn in the top row, matching `sklearn.metrics.ConfusionMatrixDisplay`.
 * Tick labels default to the class indices.
 *
 * @param cm - 2D confusion matrix `[n_actual, n_predicted]`
 * @param labels - Optional class names (at least as many as rows and columns)
 * @param options - Heatmap options (colormap, vmin, vmax, extent)
 * @throws {ShapeError} If `cm` is not 2D.
 * @throws {InvalidParameterError} If `labels` is shorter than the matrix.
 */
export function plotConfusionMatrix(
  cm: Tensor,
  labels?: readonly string[],
  options: PlotOptions = {}
): void {
  if (cm.ndim !== 2) {
    throw new ShapeError(`plotConfusionMatrix: cm must be a 2D tensor; received ndim=${cm.ndim}`);
  }
  const { rows, cols } = tensorToFloat64Matrix2D(cm);
  if (labels && labels.length < Math.max(rows, cols)) {
    throw new InvalidParameterError(
      `labels length must be >= ${Math.max(rows, cols)}; received ${labels.length}`,
      "labels",
      labels
    );
  }

  const ax = gca();
  // origin "upper" draws row 0 (actual class 0) at the top, whatever `options.origin` says.
  ax.heatmap(cm, { ...options, origin: "upper" });
  ax.setTitle("Confusion Matrix");
  ax.setXLabel("Predicted");
  ax.setYLabel("Actual");

  const extent = options.extent;
  const x0 = extent?.xmin ?? 0;
  const xSpan = (extent?.xmax ?? cols) - x0;
  const y0 = extent?.ymin ?? 0;
  const ySpan = (extent?.ymax ?? rows) - y0;
  const xValues: number[] = [];
  const xLabels: string[] = [];
  for (let j = 0; j < cols; j++) {
    xValues.push(x0 + ((j + 0.5) / cols) * xSpan);
    xLabels.push(labels?.[j] ?? String(j));
  }
  const yValues: number[] = [];
  const yLabels: string[] = [];
  for (let k = 0; k < rows; k++) {
    // Tick k (bottom to top) belongs to actual class rows - 1 - k.
    yValues.push(y0 + ((k + 0.5) / rows) * ySpan);
    yLabels.push(labels?.[rows - 1 - k] ?? String(rows - 1 - k));
  }
  ax.setXTicks(xValues, xLabels);
  ax.setYTicks(yValues, yLabels);
}

/**
 * Plot a ROC curve with optional AUC annotation.
 *
 * @param fpr - False positive rates (x axis)
 * @param tpr - True positive rates (y axis)
 * @param auc - Optional area under the curve, shown in the title
 * @param options - Line styling options
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
    color: options.color ?? themePrimary(),
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
 *
 * @param precision - Precision values (y axis)
 * @param recall - Recall values (x axis)
 * @param averagePrecision - Optional average precision, shown in the title
 * @param options - Line styling options
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
    color: options.color ?? themePrimary(),
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
 *
 * Scores may be 1D (one value per training size) or 2D
 * `[n_sizes, n_folds]` as returned by `sklearn.model_selection.learning_curve`;
 * 2D scores are averaged over the folds.
 *
 * @param trainSizes - Training set sizes (x axis)
 * @param trainScores - Training scores
 * @param valScores - Validation scores
 * @param options - Line options. `colors[0]` / `colors[1]` (or `color` for the
 *   training line) set the line colors.
 */
export function plotLearningCurve(
  trainSizes: Tensor,
  trainScores: Tensor,
  valScores: Tensor,
  options: PlotOptions = {}
): void {
  const train = meanOverFolds(trainScores, "plotLearningCurve", "trainScores");
  const val = meanOverFolds(valScores, "plotLearningCurve", "valScores");
  const ax = gca();
  const trainColor = options.colors?.[0] ?? options.color ?? themeCycle(0);
  const valColor = options.colors?.[1] ?? themeCycle(1);
  ax.plot(trainSizes, train, { ...options, color: trainColor, label: "Training Score" });
  ax.plot(trainSizes, val, { ...options, color: valColor, label: "Validation Score" });
  ax.setTitle("Learning Curve");
  ax.setXLabel("Training Set Size");
  ax.setYLabel("Score");
}

/**
 * Plot training and validation curves against a hyperparameter range.
 *
 * Scores may be 1D or 2D `[n_values, n_folds]` as returned by
 * `sklearn.model_selection.validation_curve`; 2D scores are averaged over the
 * folds.
 *
 * @param paramRange - Hyperparameter values (x axis)
 * @param trainScores - Training scores
 * @param valScores - Validation scores
 * @param options - Line options. `colors[0]` / `colors[1]` (or `color` for the
 *   training line) set the line colors.
 */
export function plotValidationCurve(
  paramRange: Tensor,
  trainScores: Tensor,
  valScores: Tensor,
  options: PlotOptions = {}
): void {
  const train = meanOverFolds(trainScores, "plotValidationCurve", "trainScores");
  const val = meanOverFolds(valScores, "plotValidationCurve", "valScores");
  const ax = gca();
  const trainColor = options.colors?.[0] ?? options.color ?? themeCycle(0);
  const valColor = options.colors?.[1] ?? themeCycle(1);
  ax.plot(paramRange, train, { ...options, color: trainColor, label: "Training Score" });
  ax.plot(paramRange, val, { ...options, color: valColor, label: "Validation Score" });
  ax.setTitle("Validation Curve");
  ax.setXLabel("Parameter Value");
  ax.setYLabel("Score");
}

/**
 * Plot a classifier decision boundary on a 2D feature space.
 *
 * The model is evaluated on a 100 x 100 grid that covers the data range plus a
 * 5% margin. `model.predict` may return either class labels (`[n]`, or `[n, 1]`)
 * or per-class scores (`[n, k]`); with scores, the column index of the largest
 * score is used as the class label, so `y` must use the labels `0..k-1`.
 * Number and bigint labels that are equal (for example `1` and `1n`) are
 * treated as the same class.
 *
 * @param X - Feature matrix of shape `[n, 2]` with finite values
 * @param y - Class labels of shape `[n]`
 * @param model - Object with a `predict` method
 * @param options - `colors` for the class markers, `size` for the marker size,
 *   `colormap`, `vmin` and `vmax` for the background
 * @throws {InvalidParameterError} If `X` is not numeric, empty or non-finite, or
 *   if the model returns non-numeric scores.
 * @throws {ShapeError} If `X`, `y` or the predictions have the wrong shape.
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

  const readLabel = (labelTensor: Tensor, ...index: number[]): string | number | bigint => {
    const value = labelTensor.at(...index);
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

  // Strings keep their own namespace; numbers and bigints share one so that an int64
  // label tensor and a float prediction tensor agree on what "class 1" is.
  const labelKey = (value: string | number | bigint): string =>
    typeof value === "string" ? `s:${value}` : `n:${String(value)}`;

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
    const fy = yMin + ((yMax - yMin) * gy) / (gridResolution - 1);
    for (let gx = 0; gx < gridResolution; gx++) {
      const fx = xMin + ((xMax - xMin) * gx) / (gridResolution - 1);
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
    if (cols === 1) {
      // A column of labels, as returned by models that keep predictions 2D.
      for (let i = 0; i < rows; i++) {
        predictedLabels[i] = readLabel(predictions, i, 0);
      }
    } else {
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
    }
  } else {
    throw new ShapeError("plotDecisionBoundary: model.predict output must be 1D or 2D");
  }

  const yLabels: Array<string | number | bigint> = new Array(n);
  for (let i = 0; i < n; i++) yLabels[i] = readLabel(y, i);

  const classIndex = new Map<string, number>();
  const classValues: Array<string | number | bigint> = [];
  for (const label of yLabels) {
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

  const boundaryFlat = new Float64Array(gridCount);
  for (let i = 0; i < gridCount; i++) {
    const label = predictedLabels[i];
    boundaryFlat[i] = label === undefined ? 0 : (classIndex.get(labelKey(label)) ?? 0);
  }

  const ax = gca();
  ax.imshow(tensor(boundaryFlat).reshape([gridResolution, gridResolution]), {
    colormap: options.colormap ?? "grayscale",
    vmin: options.vmin ?? 0,
    vmax: options.vmax ?? Math.max(1, classValues.length - 1),
    extent: { xmin: xMin, xmax: xMax, ymin: yMin, ymax: yMax },
  });

  const byClass = new Map<string, { x: number[]; y: number[]; index: number; label: string }>();
  for (let i = 0; i < n; i++) {
    const label = yLabels[i] ?? 0;
    const key = labelKey(label);
    let current = byClass.get(key);
    if (current === undefined) {
      current = { x: [], y: [], index: classIndex.get(key) ?? 0, label: String(label) };
      byClass.set(key, current);
    }
    current.x.push(x0[i] ?? 0);
    current.y.push(x1[i] ?? 0);
  }

  const classColors =
    options.colors !== undefined && options.colors.length > 0
      ? options.colors
      : getTheme().colorCycle;
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
 * Uses a Gaussian KDE to estimate the probability density function, then plots
 * it as a smooth line (or, with `fill: true`, as a shaded area under the curve). The grid
 * extends three kernel bandwidths beyond the data on both sides, so the tails
 * of the density are visible.
 *
 * Non-finite values (NaN, Infinity) are ignored.
 *
 * @param data - 1D tensor of observations
 * @param options - Line styling options plus `bwMethod` (`"scott"`,
 *   `"silverman"` or a positive bandwidth, default `"scott"`; `bw_method` is a deprecated
 *   spelling, and `bwMethod` wins when both are given), `gridSize`
 *   (number of evaluation points, an integer >= 2, default 200) and `fill` (shade the area
 *   between the curve and zero)
 * @throws {ShapeError} If `data` is not 1D.
 * @throws {InvalidParameterError} If there are no finite values or `gridSize`
 *   is invalid.
 */
export function kdeplot(
  data: Tensor,
  options: PlotOptions & {
    bwMethod?: "scott" | "silverman" | number;
    /** @deprecated Prefer `bwMethod`. */
    bw_method?: "scott" | "silverman" | number;
    gridSize?: number;
    fill?: boolean;
  } = {}
): void {
  if (data.ndim !== 1) {
    throw new ShapeError(`kdeplot: data must be a 1D tensor; received ndim=${data.ndim}`);
  }
  const n = data.shape[0] ?? 0;
  if (n === 0) {
    throw new InvalidParameterError("kdeplot: data must have at least one element", "data", n);
  }

  const {
    bwMethod: bwMethodOption,
    bw_method: bwMethodSnake,
    gridSize: gridSizeOption,
    fill,
    ...plotOptions
  } = options;
  const bwMethod = bwMethodOption ?? bwMethodSnake;
  const gridSize = gridSizeOption ?? 200;
  if (!Number.isInteger(gridSize) || gridSize < 2) {
    throw new InvalidParameterError(
      `kdeplot: gridSize must be an integer >= 2; received ${gridSize}`,
      "gridSize",
      gridSize
    );
  }

  const all = tensorToFloat64Vector1D(data);
  const values = new Float64Array(n);
  let count = 0;
  for (let i = 0; i < n; i++) {
    const v = all[i] ?? NaN;
    if (Number.isFinite(v)) values[count++] = v;
  }
  if (count === 0) {
    throw new InvalidParameterError(
      "kdeplot: data must contain finite numeric values",
      "data",
      data.dtype
    );
  }
  const finite = values.subarray(0, count);

  const bwOpts: { bwMethod?: "scott" | "silverman" | number } = {};
  if (bwMethod !== undefined) bwOpts.bwMethod = bwMethod;
  const kde = gaussianKde(finite, bwOpts);

  let min = finite[0] ?? 0;
  let max = finite[0] ?? 0;
  for (let i = 1; i < count; i++) {
    const v = finite[i] ?? 0;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const lo = min - 3 * kde.bandwidth;
  const hi = max + 3 * kde.bandwidth;

  const xGrid = new Float64Array(gridSize);
  for (let i = 0; i < gridSize; i++) {
    xGrid[i] = lo + ((hi - lo) * i) / (gridSize - 1);
  }

  const density = kde.evaluate(xGrid);

  const ax = gca();
  const xTensor = tensor(xGrid);
  const yTensor = tensor(density);
  const style = {
    ...plotOptions,
    color: plotOptions.color ?? themePrimary(),
    label: plotOptions.label ?? "KDE",
  };

  if (fill) {
    ax.area(xTensor, yTensor, style);
  } else {
    ax.plot(xTensor, yTensor, style);
  }
  ax.setXLabel("Value");
  ax.setYLabel("Density");
}

// ---- ML Plots ----

/**
 * Plot residuals (residual against predicted value) for regression diagnostics.
 *
 * The residual is `yTrue - yPred`, matching the convention of
 * `sklearn.metrics.PredictionErrorDisplay` with `kind="residual_vs_predicted"`.
 *
 * @param yTrue - 1D tensor of observed values
 * @param yPred - 1D tensor of predicted values, same length as `yTrue`
 * @param options - Marker styling options
 * @throws {InvalidParameterError} If `yTrue` is empty.
 * @throws {ShapeError} If the inputs are not 1D or differ in length.
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

  const yt = vector1D(yTrue, "plotResiduals", "yTrue");
  const yp = vector1D(yPred, "plotResiduals", "yPred");
  const residuals = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    residuals[i] = (yt[i] ?? 0) - (yp[i] ?? 0);
  }

  const ax = gca();
  ax.scatter(tensor(yp), tensor(residuals), {
    ...options,
    color: options.color ?? themePrimary(),
    size: options.size ?? 3,
  });
  ax.axhline(0, { color: "#999999", linewidth: 1 });
  ax.setTitle("Residual Plot");
  ax.setXLabel("Predicted");
  ax.setYLabel("Residual");
}

/**
 * Plot feature importances as a horizontal bar chart.
 *
 * Features are sorted by importance, with the most important feature at the
 * top.
 *
 * @param importances - 1D tensor of finite importance values
 * @param featureNames - Optional names, one per feature (default `Feature i`)
 * @param options - Bar styling options
 * @throws {InvalidParameterError} If `importances` is empty or non-finite, or
 *   `featureNames` does not have one entry per feature.
 * @throws {ShapeError} If `importances` is not 1D.
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
  const values = vector1D(importances, "plotFeatureImportance", "importances");
  if (featureNames !== undefined && featureNames.length !== n) {
    throw new InvalidParameterError(
      `plotFeatureImportance: featureNames must have one entry per feature (${n}); received ${featureNames.length}`,
      "featureNames",
      featureNames
    );
  }

  // Sort by importance descending
  const order: number[] = [];
  for (let i = 0; i < n; i++) {
    if (!Number.isFinite(values[i])) {
      throw new InvalidParameterError(
        "plotFeatureImportance: importances must be finite",
        "importances",
        { index: i, value: values[i] }
      );
    }
    order.push(i);
  }
  order.sort((a, b) => (values[b] ?? 0) - (values[a] ?? 0));

  // The y axis grows upwards, so give the most important feature the largest position.
  const yPos: number[] = [];
  const vals: number[] = [];
  const labels: string[] = [];
  for (let rank = 0; rank < n; rank++) {
    const index = order[rank] ?? 0;
    yPos.push(n - 1 - rank);
    vals.push(values[index] ?? 0);
    labels.push(featureNames?.[index] ?? `Feature ${index}`);
  }

  const ax = gca();
  ax.barh(tensor(yPos), tensor(vals), {
    ...options,
    color: options.color ?? themePrimary(),
  });
  ax.setYTicks(yPos, labels);
  ax.setTitle("Feature Importance");
  ax.setXLabel("Importance");
}

/**
 * Plot an elbow curve for KMeans cluster selection.
 *
 * @param kValues - Numbers of clusters (x axis)
 * @param inertias - Inertia for each k (y axis)
 * @param options - Line styling options
 */
export function plotElbowCurve(kValues: Tensor, inertias: Tensor, options: PlotOptions = {}): void {
  const ax = gca();
  ax.plot(kValues, inertias, {
    ...options,
    color: options.color ?? themePrimary(),
    label: options.label ?? "Inertia",
  });
  ax.setTitle("Elbow Curve");
  ax.setXLabel("Number of Clusters (k)");
  ax.setYLabel("Inertia");
}

/**
 * Plot silhouette scores for clustering evaluation.
 *
 * @param kValues - Numbers of clusters (x axis)
 * @param silhouetteScores - Mean silhouette score for each k (y axis)
 * @param options - Line styling options
 */
export function plotSilhouette(
  kValues: Tensor,
  silhouetteScores: Tensor,
  options: PlotOptions = {}
): void {
  const ax = gca();
  ax.plot(kValues, silhouetteScores, {
    ...options,
    color: options.color ?? themePrimary(),
    label: options.label ?? "Silhouette Score",
  });
  ax.setTitle("Silhouette Analysis");
  ax.setXLabel("Number of Clusters (k)");
  ax.setYLabel("Silhouette Score");
}

/**
 * Plot a calibration curve (reliability diagram).
 *
 * The argument order follows `sklearn.calibration.calibration_curve`, which
 * returns `(prob_true, prob_pred)`.
 *
 * @param fractionPositive - Fraction of positives in each bin (y axis)
 * @param meanPredicted - Mean predicted probability in each bin (x axis)
 * @param options - Line styling options
 */
export function plotCalibrationCurve(
  fractionPositive: Tensor,
  meanPredicted: Tensor,
  options: PlotOptions = {}
): void {
  const ax = gca();
  ax.plot(meanPredicted, fractionPositive, {
    ...options,
    color: options.color ?? themePrimary(),
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
 * Similar to Seaborn's pairplot. Creates an n x n grid of subplots where the
 * diagonal shows a histogram (or a kernel density estimate) of each variable
 * and the off-diagonal cells show scatter plots. Axis labels are drawn on the
 * left column and bottom row only.
 *
 * @param data - 2D tensor `[n_samples, n_features]`
 * @param options - `featureNames` (one per column), `color`, scatter `size`,
 *   `diagKind` (`"hist"` or `"kde"`, default `"hist"`) and `bins` (histogram
 *   bins on the diagonal, default 20)
 * @returns The figure holding the grid, row-major
 * @throws {ShapeError} If `data` is not 2D.
 * @throws {InvalidParameterError} If `data` is empty, `featureNames` has the
 *   wrong length, or `diagKind` is unknown.
 */
export function pairplot(
  data: Tensor,
  options: {
    featureNames?: readonly string[];
    color?: string;
    size?: number;
    diagKind?: "hist" | "kde";
    bins?: number;
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
  const diagKind = options.diagKind ?? "hist";
  if (diagKind !== "hist" && diagKind !== "kde") {
    throw new InvalidParameterError(
      `pairplot: diagKind must be "hist" or "kde"; received ${String(diagKind)}`,
      "diagKind",
      diagKind
    );
  }
  if (options.featureNames !== undefined && options.featureNames.length !== nFeatures) {
    throw new InvalidParameterError(
      `pairplot: featureNames must have one entry per column (${nFeatures}); received ${options.featureNames.length}`,
      "featureNames",
      options.featureNames
    );
  }

  const cellSize = 200;
  const totalW = cellSize * nFeatures;
  const totalH = cellSize * nFeatures;
  const fig = new FigureClass({ width: totalW, height: totalH });

  // Extract columns
  const { data: flat } = tensorToFloat64Matrix2D(data);
  const columns: Float64Array[] = [];
  for (let j = 0; j < nFeatures; j++) {
    const col = new Float64Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      col[i] = flat[i * nFeatures + j] ?? 0;
    }
    columns.push(col);
  }

  const featureNames = options.featureNames ?? columns.map((_, i) => `Feature ${i}`);
  const color = options.color ?? themePrimary();

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
        const colData = columns[row];
        if (colData) {
          if (diagKind === "kde") {
            const finite = colData.filter(Number.isFinite);
            if (finite.length > 0) {
              const kde = gaussianKde(finite);
              let colMin = finite[0] ?? 0;
              let colMax = colMin;
              for (const v of finite) {
                if (v < colMin) colMin = v;
                if (v > colMax) colMax = v;
              }
              const lo = colMin - 3 * kde.bandwidth;
              const hi = colMax + 3 * kde.bandwidth;
              const steps = 100;
              const grid = new Float64Array(steps);
              for (let k = 0; k < steps; k++) grid[k] = lo + ((hi - lo) * k) / (steps - 1);
              ax.plot(tensor(grid), tensor(kde.evaluate(grid)), { color });
            }
          } else {
            ax.hist(tensor(colData), options.bins ?? 20, { color });
          }
        }
      } else {
        // Off-diagonal: scatter
        const xCol = columns[c];
        const yCol = columns[row];
        if (xCol && yCol) {
          ax.scatter(tensor(xCol), tensor(yCol), {
            color,
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

// ---- Additional plot types ----

/**
 * Stem plot: vertical lines from baseline to data points with circle markers.
 *
 * @param x - 1D tensor of x positions
 * @param y - 1D tensor of values
 * @param options - Styling options plus `baseline` (default 0)
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
 *
 * @param groups - One 1D tensor per category
 * @param options - Colors and labels per group, marker `size` and `jitter` width
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
 *
 * @param series - One 1D tensor per series; all series have one value per axis
 * @param options - Colors and labels per series, and `linewidth`
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
 *
 * @param categories - One label per step
 * @param values - 1D tensor of step values, same length as `categories`
 * @param options - Bar colors and `barWidth`
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
 *
 * @param x - 1D tensor of arrow x positions
 * @param y - 1D tensor of arrow y positions
 * @param u - 1D tensor of arrow x components
 * @param v - 1D tensor of arrow y components
 * @param options - Styling options plus an arrow length `scale`
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
 *
 * @param theta - 1D tensor of angles in radians
 * @param r - 1D tensor of radii
 * @param options - Styling options plus `fill` to shade the enclosed area
 */
export function polar(
  theta: Tensor,
  r: Tensor,
  options: PlotOptions & { fill?: boolean } = {}
): void {
  gca().polar(theta, r, options);
}

/**
 * Equal-width histogram bins for `values`, using the same binning rule as the
 * `hist` drawable (non-finite values are ignored, constant data fills the
 * middle bin of a unit-width range).
 * @internal
 */
function histogramBins(
  values: Float64Array,
  bins: number
): { readonly edges: number[]; readonly counts: number[] } {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let finiteCount = 0;
  for (const v of values) {
    if (!Number.isFinite(v)) continue;
    finiteCount++;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (finiteCount === 0) {
    throw new InvalidParameterError(
      "jointplot: y must contain at least one finite value",
      "y",
      values.length
    );
  }
  const span = max - min;
  const width = span > 0 ? span / bins : 1;
  const start = span > 0 ? min : min - (bins * width) / 2;
  const counts = new Array<number>(bins).fill(0);
  const edges: number[] = [];
  for (let k = 0; k <= bins; k++) edges.push(start + k * width);
  if (span === 0) {
    counts[Math.floor(bins / 2)] = finiteCount;
  } else {
    for (const v of values) {
      if (!Number.isFinite(v)) continue;
      const k = Math.min(Math.max(0, Math.floor((v - min) / width)), bins - 1);
      counts[k] = (counts[k] ?? 0) + 1;
    }
  }
  return { edges, counts };
}

/**
 * Joint plot: scatter plot with marginal histograms (Seaborn-style `jointplot`).
 *
 * Creates a new figure with a central scatter plot, a histogram of `x` above it
 * and a histogram of `y` to its right. The marginal axes use the same data range
 * as the scatter axes, so the bins line up with the points. The histogram of
 * `y` is drawn as horizontal bar outlines.
 *
 * @param x - 1D tensor of x values
 * @param y - 1D tensor of y values, same length as `x`
 * @param options - `color`, number of `bins` (default 20), axis labels and title
 * @returns The new figure (it is not made the current figure)
 * @throws {ShapeError} If `x` and `y` are not 1D tensors of the same length.
 * @throws {InvalidParameterError} If `bins` is not a positive integer.
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
  const xv = vector1D(x, "jointplot", "x");
  const yv = vector1D(y, "jointplot", "y");
  if (xv.length !== yv.length) {
    throw new ShapeError("jointplot: x and y must have the same length");
  }
  const bins = options.bins ?? 20;
  if (!Number.isInteger(bins) || bins <= 0) {
    throw new InvalidParameterError(
      `jointplot: bins must be a positive integer; received ${bins}`,
      "bins",
      bins
    );
  }

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

  const clr = options.color ?? themePrimary();

  // Main scatter
  mainAx.scatter(x, y, { color: clr, size: 3 });
  if (options.xlabel) mainAx.setXLabel(options.xlabel);
  if (options.ylabel) mainAx.setYLabel(options.ylabel);

  // Top marginal histogram
  topAx.hist(x, bins, { color: clr });

  // Right marginal histogram: horizontal bar outlines over the y bins.
  const yBins = histogramBins(yv, bins);
  for (let k = 0; k < bins; k++) {
    const count = yBins.counts[k] ?? 0;
    const lo = yBins.edges[k] ?? 0;
    const hi = yBins.edges[k + 1] ?? 0;
    rightAx.plot(tensor([0, count, count, 0, 0]), tensor([lo, lo, hi, hi, lo]), {
      color: clr,
      linewidth: 1,
    });
  }

  // Share the data ranges so that marginal bins sit over/next to the points they count.
  const xRange = finiteRange(xv);
  if (xRange) {
    mainAx.xlim(xRange.lo, xRange.hi);
    topAx.xlim(xRange.lo, xRange.hi);
  }
  const yRange = finiteRange(yv);
  if (yRange) {
    mainAx.ylim(yRange.lo, yRange.hi);
    rightAx.ylim(yRange.lo, yRange.hi);
  }

  if (options.title) mainAx.setTitle(options.title);

  return fig;
}

/**
 * Finite min/max of `values` widened by 5% of the span on each side, or null
 * when there are no finite values or all of them are equal.
 * @internal
 */
function finiteRange(values: Float64Array): { readonly lo: number; readonly hi: number } | null {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (const v of values) {
    if (!Number.isFinite(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (!(max > min)) return null;
  const pad = (max - min) * 0.05;
  return { lo: min - pad, hi: max + pad };
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

// ─── 3D Plots ────────────────────────────────────────────────────────────────

export {
  type Plot3DOptions,
  Scatter3D,
  Surface3D,
  Wireframe3D,
} from "./plots/Surface3D";

/**
 * Copy three rectangular grids into Float64Arrays, checking that they are
 * non-empty and share one shape.
 * @internal
 */
function toGrids(
  fn: string,
  xGrid: ArrayLike<ArrayLike<number>>,
  yGrid: ArrayLike<ArrayLike<number>>,
  zGrid: ArrayLike<ArrayLike<number>>
): [Float64Array[], Float64Array[], Float64Array[]] {
  const rows = zGrid.length;
  if (rows === 0 || xGrid.length !== rows || yGrid.length !== rows) {
    throw new ShapeError(
      `${fn}: xGrid, yGrid and zGrid must have the same, non-zero number of rows; received ${xGrid.length}, ${yGrid.length}, ${zGrid.length}`
    );
  }
  const cols = zGrid[0]?.length ?? 0;
  if (cols === 0) {
    throw new ShapeError(`${fn}: grids must have at least one column`);
  }
  const copy = (grid: ArrayLike<ArrayLike<number>>, name: string): Float64Array[] => {
    const out: Float64Array[] = [];
    for (let r = 0; r < rows; r++) {
      const row = grid[r];
      if (row === undefined || row.length !== cols) {
        throw new ShapeError(
          `${fn}: ${name} must be a rectangular ${rows} x ${cols} grid; row ${r} has length ${row?.length ?? 0}`
        );
      }
      out.push(Float64Array.from(row));
    }
    return out;
  };
  return [copy(xGrid, "xGrid"), copy(yGrid, "yGrid"), copy(zGrid, "zGrid")];
}

/**
 * Create a 3D surface plot on the current axes.
 *
 * @param xGrid - 2D array of x-coordinates (rows x cols)
 * @param yGrid - 2D array of y-coordinates (rows x cols)
 * @param zGrid - 2D array of z-values (rows x cols)
 * @param options - 3D plot options (elevation, azimuth, color, etc.)
 * @throws {ShapeError} If the grids are empty, ragged, or differ in shape.
 *
 * @example
 * ```ts
 * import { surface } from 'deepbox/plot';
 *
 * const n = 20;
 * const grid = (f: (x: number, y: number) => number): number[][] =>
 *   Array.from({ length: n }, (_, i) =>
 *     Array.from({ length: n }, (_, j) => f(-2 + (4 * i) / (n - 1), -2 + (4 * j) / (n - 1)))
 *   );
 * surface(
 *   grid((x) => x),
 *   grid((_, y) => y),
 *   grid((x, y) => Math.sin(Math.hypot(x, y)))
 * );
 * ```
 */
export function surface(
  xGrid: ArrayLike<ArrayLike<number>>,
  yGrid: ArrayLike<ArrayLike<number>>,
  zGrid: ArrayLike<ArrayLike<number>>,
  options: import("./plots/Surface3D").Plot3DOptions = {}
): void {
  const [xF, yF, zF] = toGrids("surface", xGrid, yGrid, zGrid);
  gca().surface(xF, yF, zF, options);
}

/**
 * Create a 3D wireframe plot on the current axes.
 *
 * @param xGrid - 2D array of x-coordinates (rows x cols)
 * @param yGrid - 2D array of y-coordinates (rows x cols)
 * @param zGrid - 2D array of z-values (rows x cols)
 * @param options - 3D plot options (elevation, azimuth, color, etc.)
 * @throws {ShapeError} If the grids are empty, ragged, or differ in shape.
 *
 * @example
 * ```ts
 * import { wireframe } from 'deepbox/plot';
 *
 * const x = [[0, 1], [0, 1]];
 * const y = [[0, 0], [1, 1]];
 * const z = [[0, 1], [1, 2]];
 * wireframe(x, y, z, { color: '#333333', elevation: 45 });
 * ```
 */
export function wireframe(
  xGrid: ArrayLike<ArrayLike<number>>,
  yGrid: ArrayLike<ArrayLike<number>>,
  zGrid: ArrayLike<ArrayLike<number>>,
  options: import("./plots/Surface3D").Plot3DOptions = {}
): void {
  const [xF, yF, zF] = toGrids("wireframe", xGrid, yGrid, zGrid);
  gca().wireframe(xF, yF, zF, options);
}

/**
 * Create a 3D scatter plot on the current axes.
 *
 * @param x - 1D array of x-coordinates
 * @param y - 1D array of y-coordinates
 * @param z - 1D array of z-coordinates
 * @param options - 3D plot options (elevation, azimuth, color, size, etc.)
 * @throws {ShapeError} If the arrays differ in length.
 *
 * @example
 * ```ts
 * import { scatter3d } from 'deepbox/plot';
 * scatter3d([1, 2, 3], [4, 5, 6], [7, 8, 9], { color: 'red' });
 * ```
 */
export function scatter3d(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  z: ArrayLike<number>,
  options: import("./plots/Surface3D").Plot3DOptions = {}
): void {
  gca().scatter3d(Float64Array.from(x), Float64Array.from(y), Float64Array.from(z), options);
}
