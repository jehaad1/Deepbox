/**
 * @see {@link https://deepbox.dev/docs/plot-basic | Deepbox documentation}
 */

import { InvalidParameterError, ShapeError } from "../../core";
import type { AnyTensor } from "../../ndarray";
import type { RasterCanvas } from "../canvas/RasterCanvas";
import { Bar2D } from "../plots/Bar2D";
import { Boxplot } from "../plots/Boxplot";
import { Contour2D } from "../plots/Contour2D";
import { ContourF2D } from "../plots/ContourF2D";
import { Dendrogram2D, type LinkageRow } from "../plots/Dendrogram2D";
import { Heatmap2D } from "../plots/Heatmap2D";
import { Histogram } from "../plots/Histogram";
import { HorizontalBar2D } from "../plots/HorizontalBar2D";
import { Line2D } from "../plots/Line2D";
import { Pie } from "../plots/Pie";
import { Polar2D } from "../plots/Polar2D";
import { Quiver2D } from "../plots/Quiver2D";
import { Radar2D } from "../plots/Radar2D";
import { Scatter2D } from "../plots/Scatter2D";
import { Stem2D } from "../plots/Stem2D";
import { Strip2D } from "../plots/Strip2D";
import { type Plot3DOptions, Scatter3D, Surface3D, Wireframe3D } from "../plots/Surface3D";
import { Violinplot, type ViolinplotOptions } from "../plots/Violinplot";
import { Waterfall2D } from "../plots/Waterfall2D";
import type {
  Color,
  Drawable,
  LegendEntry,
  LegendOptions,
  PlotOptions,
  RasterDrawContext,
  SvgDrawContext,
  TextOptions,
  Viewport,
} from "../types";
import { normalizeColor, parseHexColorToRGBA } from "../utils/colors";
import { buildContourGrid } from "../utils/contours";
import { buildLegendEntry } from "../utils/legend";
import { tensorToFloat64Matrix2D, tensorToFloat64Vector1D } from "../utils/tensor";
import { estimateTextWidth } from "../utils/text";
import { generateLogTicks, generateTicks, type Tick } from "../utils/ticks";
import { computeAutoRange, makeTransform } from "../utils/transforms";
import { escapeXml } from "../utils/xml";
import type { Figure } from "./Figure";
import { getTheme } from "./theme";

type AxisRange = {
  xmin: number;
  xmax: number;
  ymin: number;
  ymax: number;
};

type StepPosition = "pre" | "post" | "mid";

/** Colors cycled through by the multi-series bar helpers (the tab10 palette). */
const SERIES_COLORS: readonly string[] = [
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

/** log10 position given to values that are not positive on a log axis (far off screen). */
const LOG_CLIP = -1000;

/** Drawable kinds that live in a projected 3D view, where data-space ticks have no meaning. */
const THREE_D_KINDS: ReadonlySet<string> = new Set(["surface3d", "wireframe3d", "scatter3d"]);

/** Rounds to two decimals for SVG attributes, dropping trailing zeros. */
function fmt(v: number): string {
  return String(Number(v.toFixed(2)));
}

function assertSameLength(nameA: string, lenA: number, nameB: string, lenB: number): void {
  if (lenA !== lenB) {
    throw new ShapeError(
      `${nameA} and ${nameB} must have the same length; received ${lenA} and ${lenB}`
    );
  }
}

function assertFinitePositive(name: string, v: number): void {
  if (typeof v !== "number" || !Number.isFinite(v) || v <= 0) {
    throw new InvalidParameterError(`${name} must be a positive number; received ${v}`, name, v);
  }
}

/**
 * A closed polygon that is filled with its color and outlined with a line.
 * It extends Line2D so the fill helpers keep returning a Line2D.
 */
class FilledPolygon extends Line2D {
  override drawSVG(ctx: SvgDrawContext): void {
    const cmds: string[] = [];
    for (let i = 0; i < this.x.length; i++) {
      const xi = this.x[i] ?? Number.NaN;
      const yi = this.y[i] ?? Number.NaN;
      if (!Number.isFinite(xi) || !Number.isFinite(yi)) continue;
      cmds.push(
        `${cmds.length === 0 ? "M" : "L"}${ctx.transform.xToPx(xi).toFixed(2)} ${ctx.transform.yToPx(yi).toFixed(2)}`
      );
    }
    if (cmds.length >= 3) {
      ctx.push(
        `<path d="${cmds.join(" ")} Z" fill="${escapeXml(this.color)}" fill-rule="evenodd" stroke="none" />`
      );
    }
    super.drawSVG(ctx);
  }

  override drawRaster(ctx: RasterDrawContext): void {
    const px: number[] = [];
    const py: number[] = [];
    for (let i = 0; i < this.x.length; i++) {
      const xi = this.x[i] ?? Number.NaN;
      const yi = this.y[i] ?? Number.NaN;
      if (!Number.isFinite(xi) || !Number.isFinite(yi)) continue;
      px.push(ctx.transform.xToPx(xi));
      py.push(ctx.transform.yToPx(yi));
    }
    const c = parseHexColorToRGBA(this.color);
    ctx.canvas.fillPolygonRGBA(px, py, c.r, c.g, c.b, c.a);
    super.drawRaster(ctx);
  }

  override getLegendEntries(): readonly LegendEntry[] | null {
    const entry = buildLegendEntry(this.label, { color: this.color, shape: "box" });
    return entry ? [entry] : null;
  }
}

/** Closed outline of the polygon with the given vertices (first vertex repeated at the end). */
function closePolygon(xs: readonly number[], ys: readonly number[]): [Float64Array, Float64Array] {
  const n = xs.length;
  if (n === 0) return [new Float64Array(0), new Float64Array(0)];
  const cx = new Float64Array(n + 1);
  const cy = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) {
    cx[i] = xs[i] ?? 0;
    cy[i] = ys[i] ?? 0;
  }
  cx[n] = xs[0] ?? 0;
  cy[n] = ys[0] ?? 0;
  return [cx, cy];
}

/** Reference-line pseudo-drawable: only contributes its position to the auto range. */
function referenceRange(kind: "axhline" | "axvline", value: number): Drawable {
  const inf = Number.POSITIVE_INFINITY;
  const range =
    kind === "axhline"
      ? { xmin: inf, xmax: -inf, ymin: value, ymax: value }
      : { xmin: value, xmax: value, ymin: inf, ymax: -inf };
  return {
    kind,
    getDataRange: () => range,
    drawSVG: () => {},
    drawRaster: () => {},
  };
}

/**
 * Auto range of one axis on a log scale, padded by 5% of the span in log10 space.
 * Values that are not positive cannot be shown. A drawable that reports its smallest
 * positive coordinate (`getPositiveMin`, as lines and scatter plots do) is bounded by
 * that value, like matplotlib. For any other drawable whose minimum is not positive the
 * lower bound falls back to a decade below the maximum (at most 0.1).
 */
function logAutoRange(drawables: readonly Drawable[], axis: "x" | "y"): [number, number] {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  let unbounded = false;
  for (const d of drawables) {
    const r = d.getDataRange();
    if (!r) continue;
    const a = axis === "x" ? r.xmin : r.ymin;
    const b = axis === "x" ? r.xmax : r.ymax;
    if (Number.isFinite(b) && b > hi) hi = b;
    const positive = d.getPositiveMin?.();
    if (positive !== undefined) {
      const p = axis === "x" ? positive.x : positive.y;
      if (Number.isFinite(p) && p > 0 && p < lo) lo = p;
    } else if (Number.isFinite(a)) {
      if (a > 0) {
        if (a < lo) lo = a;
      } else {
        unbounded = true;
      }
    }
  }
  if (unbounded && Number.isFinite(hi) && hi > 0) lo = Math.min(lo, 0.1, hi / 10);
  if (!(hi > 0) || !Number.isFinite(lo)) return [0.1, 10];
  let a = Math.log10(lo);
  let b = Math.log10(hi);
  if (a === b) {
    a -= 0.5;
    b += 0.5;
  }
  const pad = (b - a) * 0.05;
  return [10 ** (a - pad), 10 ** (b + pad)];
}

/**
 * An Axes represents a single plot area within a Figure.
 *
 * Create one with `fig.addAxes()`. Plotting methods take 1D (or, for matrices,
 * 2D) tensors and add a series to the axes; rendering happens when the figure
 * is rendered. The axes is drawn inside its viewport, inset by `padding`
 * pixels on every side to leave room for ticks, labels and the title.
 */
export class Axes {
  public readonly fig: Figure;
  private readonly padding: number;
  private readonly paddingProvided: boolean;
  private readonly facecolor: Color;
  private readonly drawables: Drawable[];
  private title: string;
  private xlabel: string;
  private ylabel: string;
  private legendOptions: LegendOptions | null;
  private xTicksOverride: readonly Tick[] | null;
  private yTicksOverride: readonly Tick[] | null;
  private readonly baseViewport: Viewport | undefined;
  private xLimOverride: [number, number] | null;
  private yLimOverride: [number, number] | null;
  private gridVisible: boolean;
  private gridColor: Color;
  private annotations: Array<{
    text: string;
    x: number;
    y: number;
    color: Color;
    fontSize: number;
    ha: "left" | "center" | "right";
    va: "bottom" | "center" | "top";
  }>;
  // Look taken from the active theme when the axes is created.
  private readonly textColor: Color;
  private readonly tickFontSize: number;
  private readonly labelFontSize: number;
  private readonly titleFontSize: number;
  private readonly legendFontSize: number;
  private readonly legendBackground: Color;
  private readonly defaultGridVisible: boolean;
  private isTwinX: boolean;
  private _parent: Axes | null;
  private hlines: Array<{
    y: number;
    color: Color;
    linewidth: number;
    label: string | null;
  }>;
  private vlines: Array<{
    x: number;
    color: Color;
    linewidth: number;
    label: string | null;
  }>;
  private xScale: "linear" | "log";
  private yScale: "linear" | "log";

  constructor(
    fig: Figure,
    options: {
      readonly padding?: number;
      readonly facecolor?: Color;
      readonly viewport?: Viewport;
    }
  ) {
    this.fig = fig;
    this.paddingProvided = options.padding !== undefined;
    const base: Viewport = options.viewport ?? {
      x: 0,
      y: 0,
      width: this.fig.width,
      height: this.fig.height,
    };

    if (
      !Number.isFinite(base.x) ||
      !Number.isFinite(base.y) ||
      !Number.isFinite(base.width) ||
      !Number.isFinite(base.height) ||
      base.width <= 0 ||
      base.height <= 0
    ) {
      throw new InvalidParameterError(
        "viewport must have positive finite width/height",
        "viewport",
        base
      );
    }

    const p =
      options.padding ??
      Math.min(50, Math.max(0, Math.floor(Math.min(base.width, base.height) / 4)));

    if (!Number.isFinite(p) || p < 0) {
      throw new InvalidParameterError(`padding must be non-negative; received ${p}`, "padding", p);
    }

    if (2 * p >= base.width || 2 * p >= base.height) {
      if (this.paddingProvided) {
        throw new InvalidParameterError("padding is too large", "padding", p);
      }
      const maxSafe = Math.max(0, Math.floor((Math.min(base.width, base.height) - 1) / 2));
      this.padding = maxSafe;
    } else {
      this.padding = p;
    }

    const theme = getTheme();
    this.facecolor = normalizeColor(options.facecolor, theme.axesFacecolor);
    this.textColor = theme.textColor;
    this.tickFontSize = Math.max(1, theme.fontSize - 2);
    this.labelFontSize = theme.fontSize;
    this.titleFontSize = theme.fontSize + 2;
    this.legendFontSize = theme.fontSize;
    this.legendBackground = theme.axesFacecolor;
    this.defaultGridVisible = theme.gridVisible;
    this.baseViewport = options.viewport;
    this.drawables = [];
    this.title = "";
    this.xlabel = "";
    this.ylabel = "";
    this.legendOptions = null;
    this.xTicksOverride = null;
    this.yTicksOverride = null;
    this.xLimOverride = null;
    this.yLimOverride = null;
    this.gridVisible = theme.gridVisible;
    this.gridColor = theme.gridColor;
    this.annotations = [];
    this.isTwinX = false;
    this._parent = null;
    this.hlines = [];
    this.vlines = [];
    this.xScale = "linear";
    this.yScale = "linear";
  }

  /** The parent axes when this is a twin (null otherwise). */
  get parent(): Axes | null {
    return this._parent;
  }

  /**
   * The axes that owns the shared x-axis (scale, limits and tick overrides):
   * the first non-twin ancestor, or this axes when it is not a twin.
   */
  private root(): Axes {
    return this._parent ? this._parent.root() : this;
  }

  /** This axes plus every axes that shares its x-axis through `twinx()`. */
  private xGroup(): readonly Axes[] {
    const root = this.root();
    if (!this.fig.axesList.some((a) => a._parent !== null)) return [this];
    return this.fig.axesList.filter((a) => a === root || a.root() === root);
  }

  /**
   * True when nothing has been drawn or configured on this axes yet. Used by
   * the pyplot-style helpers to drop the default axes that `figure()` creates
   * when a subplot grid is requested.
   * @internal
   */
  isBlank(): boolean {
    return (
      this.drawables.length === 0 &&
      this.title === "" &&
      this.xlabel === "" &&
      this.ylabel === "" &&
      this.legendOptions === null &&
      this.xTicksOverride === null &&
      this.yTicksOverride === null &&
      this.xLimOverride === null &&
      this.yLimOverride === null &&
      this.gridVisible === this.defaultGridVisible &&
      this.annotations.length === 0 &&
      this.hlines.length === 0 &&
      this.vlines.length === 0 &&
      this.xScale === "linear" &&
      this.yScale === "linear" &&
      this.xGroup().length === 1
    );
  }

  /** Set the axes title text. */
  setTitle(title: string): void {
    this.title = title;
  }

  /** Set the x-axis label text. */
  setXLabel(label: string): void {
    this.xlabel = label;
  }

  /** Set the y-axis label text. */
  setYLabel(label: string): void {
    this.ylabel = label;
  }

  /**
   * Set custom x-axis tick positions and labels. Pass an empty array to go back
   * to automatic ticks. On a twin axes this sets the ticks of the shared x-axis.
   * @param values - Tick positions in data coordinates
   * @param labels - Optional tick labels (default: the values)
   */
  setXTicks(values: readonly number[], labels?: readonly string[]): void {
    this.root().xTicksOverride = this.buildTickOverride(values, labels, "xTicks");
  }

  /**
   * Set custom y-axis tick positions and labels. Pass an empty array to go back
   * to automatic ticks.
   * @param values - Tick positions in data coordinates
   * @param labels - Optional tick labels (default: the values)
   */
  setYTicks(values: readonly number[], labels?: readonly string[]): void {
    this.yTicksOverride = this.buildTickOverride(values, labels, "yTicks");
  }

  /**
   * Show or configure the legend for this axes.
   * @param options - Legend display options
   */
  legend(options: LegendOptions = {}): void {
    if (
      options.location !== undefined &&
      !["upper-right", "upper-left", "lower-right", "lower-left"].includes(options.location)
    ) {
      throw new InvalidParameterError(
        `legend location must be one of upper-right, upper-left, lower-right, lower-left; received ${options.location}`,
        "location",
        options.location
      );
    }
    if (options.fontSize !== undefined) {
      if (!Number.isFinite(options.fontSize) || options.fontSize <= 0) {
        throw new InvalidParameterError(
          `legend fontSize must be positive; received ${options.fontSize}`,
          "fontSize",
          options.fontSize
        );
      }
    }
    if (options.padding !== undefined) {
      if (!Number.isFinite(options.padding) || options.padding < 0) {
        throw new InvalidParameterError(
          `legend padding must be non-negative; received ${options.padding}`,
          "padding",
          options.padding
        );
      }
    }
    this.legendOptions = options;
  }

  /**
   * Plot a connected line series.
   * @param x - 1D tensor of x coordinates
   * @param y - 1D tensor of y coordinates
   * @param options - Styling options
   */
  plot(x: AnyTensor, y: AnyTensor, options: PlotOptions = {}): Line2D {
    const xv = tensorToFloat64Vector1D(x);
    const yv = tensorToFloat64Vector1D(y);
    const d = new Line2D(xv, yv, options);
    this.drawables.push(d);
    return d;
  }

  /**
   * Plot unconnected points.
   * @param x - 1D tensor of x coordinates
   * @param y - 1D tensor of y coordinates
   * @param options - Styling options
   */
  scatter(x: AnyTensor, y: AnyTensor, options: PlotOptions = {}): Scatter2D {
    const xv = tensorToFloat64Vector1D(x);
    const yv = tensorToFloat64Vector1D(y);
    const d = new Scatter2D(xv, yv, options);
    this.drawables.push(d);
    return d;
  }

  /**
   * Plot vertical bars.
   * @param x - 1D tensor of bar centers
   * @param height - 1D tensor of bar heights
   * @param options - Styling options
   */
  bar(x: AnyTensor, height: AnyTensor, options: PlotOptions = {}): Bar2D {
    const xv = tensorToFloat64Vector1D(x);
    const hv = tensorToFloat64Vector1D(height);
    const d = new Bar2D(xv, hv, options);
    this.drawables.push(d);
    return d;
  }

  /**
   * Plot a histogram.
   * @param x - 1D tensor of sample values
   * @param bins - Number of bins
   * @param options - Styling options
   */
  hist(x: AnyTensor, bins = 10, options: PlotOptions = {}): Histogram {
    const xv = tensorToFloat64Vector1D(x);
    const resolvedBins = options.bins ?? bins;
    const d = new Histogram(xv, resolvedBins, options);
    this.drawables.push(d);
    return d;
  }

  /**
   * Plot horizontal bars.
   * @param y - 1D tensor of bar centers
   * @param width - 1D tensor of bar widths
   * @param options - Styling options
   */
  barh(y: AnyTensor, width: AnyTensor, options: PlotOptions = {}): HorizontalBar2D {
    const yv = tensorToFloat64Vector1D(y);
    const wv = tensorToFloat64Vector1D(width);
    const d = new HorizontalBar2D(yv, wv, options);
    this.drawables.push(d);
    return d;
  }

  /**
   * Plot a heatmap for 2D data.
   *
   * By default row 0 of `data` is drawn at the bottom of the y axis (matplotlib's
   * `origin="lower"`) and column 0 at the left. Pass `origin: "upper"` to put row 0 at
   * the top, as `imshow` does in matplotlib. Values are mapped from `[vmin, vmax]` onto the
   * colormap; non-finite cells are left blank.
   * @param data - 2D tensor of values
   * @param options - Styling and scale options (`colormap`, `vmin`, `vmax`, `extent`,
   *   `origin`)
   * @throws {ShapeError} If `data` is not a rectangular 2D tensor.
   * @throws {InvalidParameterError} If an option is invalid or `data` has no finite value.
   */
  heatmap(data: AnyTensor, options: PlotOptions = {}): Heatmap2D {
    const mat = tensorToFloat64Matrix2D(data);
    const d = new Heatmap2D(mat.data, mat.rows, mat.cols, options);
    this.drawables.push(d);
    return d;
  }

  /**
   * Plot a dendrogram from a linkage matrix.
   *
   * The linkage matrix has shape (n-1, 4) where each row is
   * [clusterA, clusterB, distance, count]. Leaves are placed left to right in the order of a
   * depth-first walk from the root, like `scipy.cluster.hierarchy.dendrogram`; the order is
   * available as `leaves` on the returned drawable.
   *
   * @param linkage - Linkage matrix rows
   * @param nLeaves - Number of original observations
   * @param options - Styling options
   * @throws {InvalidParameterError} If `nLeaves` is not a positive integer, or a row refers to
   *   an unknown cluster, merges a cluster twice or with itself, or has a non-finite distance.
   */
  dendrogram(
    linkage: readonly LinkageRow[],
    nLeaves: number,
    options: PlotOptions = {}
  ): Dendrogram2D {
    const d = new Dendrogram2D(linkage, nLeaves, options);
    this.drawables.push(d);
    return d;
  }

  /**
   * Display a matrix as an image. This is an alias of {@link Axes.heatmap}, so unlike
   * matplotlib's `imshow` row 0 is drawn at the bottom of the y axis unless you pass
   * `origin: "upper"`.
   * @param data - 2D tensor of values
   * @param options - Styling and scale options (`colormap`, `vmin`, `vmax`, `extent`,
   *   `origin`)
   * @throws {ShapeError} If `data` is not a rectangular 2D tensor.
   * @throws {InvalidParameterError} If an option is invalid or `data` has no finite value.
   */
  imshow(data: AnyTensor, options: PlotOptions = {}): Heatmap2D {
    const mat = tensorToFloat64Matrix2D(data);
    const d = new Heatmap2D(mat.data, mat.rows, mat.cols, options);
    this.drawables.push(d);
    return d;
  }

  /**
   * Plot contour lines for a 2D grid.
   * @param X - 1D/2D tensor of x coordinates (or empty)
   * @param Y - 1D/2D tensor of y coordinates (or empty)
   * @param Z - 2D tensor of values
   * @param options - Styling and level options
   */
  contour(X: AnyTensor, Y: AnyTensor, Z: AnyTensor, options: PlotOptions = {}): Contour2D {
    const grid = buildContourGrid(X, Y, Z, options.extent);
    const d = new Contour2D(grid, options);
    this.drawables.push(d);
    return d;
  }

  /**
   * Plot filled contours for a 2D grid.
   * @param X - 1D/2D tensor of x coordinates (or empty)
   * @param Y - 1D/2D tensor of y coordinates (or empty)
   * @param Z - 2D tensor of values
   * @param options - Styling and level options
   */
  contourf(X: AnyTensor, Y: AnyTensor, Z: AnyTensor, options: PlotOptions = {}): ContourF2D {
    const grid = buildContourGrid(X, Y, Z, options.extent);
    const d = new ContourF2D(grid, options);
    this.drawables.push(d);
    return d;
  }

  /**
   * Plot a box-and-whisker summary for a 1D dataset.
   * @param data - 1D tensor of values
   * @param options - Styling options
   */
  boxplot(data: AnyTensor, options: PlotOptions = {}): Boxplot {
    const values = tensorToFloat64Vector1D(data);
    const d = new Boxplot(1, values, options);
    this.drawables.push(d);
    return d;
  }

  /**
   * Plot a violin distribution summary for a 1D dataset: a Gaussian kernel density
   * estimate mirrored around `position`, with the quartiles marked.
   *
   * The bandwidth rule is Silverman's by default (`(3n/4)^(-1/5) s`); pass
   * `bandwidth: "scott"` for matplotlib's default rule (`n^(-1/5) s`, about 6% wider)
   * or a positive number for the kernel standard deviation in data units.
   * @param data - 1D tensor of values
   * @param options - Styling options plus `position` (x of the violin, default 1),
   *   `width` (largest width in x units, default 0.8) and `bandwidth`
   * @throws {InvalidParameterError} If `position`, `width` or `bandwidth` is invalid,
   *   or `data` is not empty but has no finite value.
   */
  violinplot(data: AnyTensor, options: ViolinplotOptions = {}): Violinplot {
    const values = tensorToFloat64Vector1D(data);
    const d = new Violinplot(options.position ?? 1, values, options);
    this.drawables.push(d);
    return d;
  }

  /**
   * Plot a pie chart.
   * @param values - 1D tensor of non-negative values
   * @param labels - Optional labels (must match values length)
   * @param options - Styling options
   */
  pie(values: AnyTensor, labels?: readonly string[], options: PlotOptions = {}): Pie {
    const data = tensorToFloat64Vector1D(values);
    const range = { xmin: 0, xmax: 1, ymin: 0, ymax: 1 };
    const d = new Pie(0.5, 0.5, 0.35, data, labels, options, range);
    this.drawables.push(d);
    return d;
  }

  /**
   * Create a twin Axes that shares the x-axis but has an independent y-axis
   * rendered on the right side. The returned axes overlays this one so that
   * two different y-scales can be shown side-by-side.
   *
   * Both axes use one x range (the union of their data, unless `xlim` is set),
   * one x scale and one set of x ticks; `xlim`, `setXScale` and `setXTicks`
   * called on either of them apply to both.
   *
   * @example
   * ```ts
   * const ax1 = fig.addAxes();
   * ax1.plot(x, y1, { color: 'blue', label: 'Temperature' });
   * ax1.setYLabel('Temperature (°C)');
   *
   * const ax2 = ax1.twinx();
   * ax2.plot(x, y2, { color: 'red', label: 'Humidity' });
   * ax2.setYLabel('Humidity (%)');
   * ```
   */
  twinx(): Axes {
    const opts: { padding: number; facecolor: Color; viewport?: Viewport } = {
      padding: this.padding,
      facecolor: this.facecolor,
    };
    if (this.baseViewport !== undefined) opts.viewport = this.baseViewport;
    const twin = new Axes(this.fig, opts);
    twin.isTwinX = true;
    twin._parent = this;
    // The grid of the parent already covers the shared x axis; a second one would cut
    // across it at the twin's own y ticks.
    twin.gridVisible = false;
    this.fig.axesList.push(twin);
    return twin;
  }

  /**
   * Set explicit x-axis limits. `min` may be larger than `max` to reverse the
   * axis. On a twin axes this sets the limits of the shared x-axis.
   * @param min - Value at the left edge
   * @param max - Value at the right edge
   */
  xlim(min: number, max: number): void {
    const root = this.root();
    this.assertLimits("xlim", min, max, root.xScale);
    root.xLimOverride = [min, max];
  }

  /**
   * Set explicit y-axis limits. `min` may be larger than `max` to reverse the axis.
   * @param min - Value at the bottom edge
   * @param max - Value at the top edge
   */
  ylim(min: number, max: number): void {
    this.assertLimits("ylim", min, max, this.yScale);
    this.yLimOverride = [min, max];
  }

  private assertLimits(name: string, min: number, max: number, scale: "linear" | "log"): void {
    if (
      typeof min !== "number" ||
      typeof max !== "number" ||
      !Number.isFinite(min) ||
      !Number.isFinite(max)
    ) {
      throw new InvalidParameterError(
        `${name} limits must be finite numbers; received ${String(min)}, ${String(max)}`,
        name,
        [min, max]
      );
    }
    if (min === max) {
      throw new InvalidParameterError(
        `${name} limits must differ; received ${min} for both`,
        name,
        [min, max]
      );
    }
    if (scale === "log" && (min <= 0 || max <= 0)) {
      throw new InvalidParameterError(
        `${name} limits must be positive on a log scale; received ${min}, ${max}`,
        name,
        [min, max]
      );
    }
  }

  /**
   * Show or hide grid lines.
   * @param visible - Whether the grid is drawn (default: true)
   * @param options - Grid line color
   */
  grid(visible = true, options: { color?: Color } = {}): void {
    this.gridVisible = visible;
    if (options.color) this.gridColor = normalizeColor(options.color, "#cccccc");
  }

  /**
   * Add a text annotation at data coordinates. By default the text starts at
   * (x, y) and sits on that baseline. `ha` and `va` change which point of the
   * text is placed at (x, y), for example `{ ha: "center", va: "center" }` to
   * center a label on a heatmap cell. Annotations are drawn in SVG, PNG and PDF
   * output.
   * @param text - Annotation text
   * @param x - X position in data coordinates
   * @param y - Y position in data coordinates
   * @param options - Text color (default: the axes text color), font size in
   *   pixels (default 10), horizontal alignment `ha` ("left", "center" or
   *   "right"; default "left") and vertical alignment `va` ("bottom", "center"
   *   or "top"; default "bottom")
   * @throws {InvalidParameterError} If x or y is not finite, the font size is
   *   not positive, or `ha` or `va` is not one of the listed values.
   */
  annotate(text: string, x: number, y: number, options: TextOptions = {}): void {
    if (
      typeof x !== "number" ||
      !Number.isFinite(x) ||
      typeof y !== "number" ||
      !Number.isFinite(y)
    ) {
      throw new InvalidParameterError(
        `annotate coordinates must be finite numbers; received x=${String(x)}, y=${String(y)}`,
        "x/y",
        [x, y]
      );
    }
    const fontSize = options.fontSize ?? 10;
    assertFinitePositive("fontSize", fontSize);
    const ha = options.ha ?? "left";
    if (ha !== "left" && ha !== "center" && ha !== "right") {
      throw new InvalidParameterError(
        `ha must be "left", "center" or "right"; received ${String(ha)}`,
        "ha",
        ha
      );
    }
    const va = options.va ?? "bottom";
    if (va !== "bottom" && va !== "center" && va !== "top") {
      throw new InvalidParameterError(
        `va must be "bottom", "center" or "top"; received ${String(va)}`,
        "va",
        va
      );
    }
    this.annotations.push({
      text,
      x,
      y,
      color: normalizeColor(options.color ?? this.textColor, "#000000"),
      fontSize,
      ha,
      va,
    });
  }

  /**
   * Draw a dashed horizontal line across the axes at the given y value. The
   * value is included when the y range is chosen automatically, and a labeled
   * line gets a legend entry.
   * @param y - Y-coordinate in data space
   * @param options - Color, linewidth (default 1), and optional label
   */
  axhline(y: number, options: { color?: Color; linewidth?: number; label?: string } = {}): void {
    if (typeof y !== "number" || !Number.isFinite(y)) {
      throw new InvalidParameterError(
        `axhline y must be a finite number; received ${String(y)}`,
        "y",
        y
      );
    }
    const linewidth = options.linewidth ?? 1;
    assertFinitePositive("linewidth", linewidth);
    this.hlines.push({
      y,
      color: normalizeColor(options.color ?? this.textColor, "#000000"),
      linewidth,
      label: options.label ?? null,
    });
  }

  /**
   * Draw a dashed vertical line across the axes at the given x value. The
   * value is included when the x range is chosen automatically, and a labeled
   * line gets a legend entry.
   * @param x - X-coordinate in data space
   * @param options - Color, linewidth (default 1), and optional label
   */
  axvline(x: number, options: { color?: Color; linewidth?: number; label?: string } = {}): void {
    if (typeof x !== "number" || !Number.isFinite(x)) {
      throw new InvalidParameterError(
        `axvline x must be a finite number; received ${String(x)}`,
        "x",
        x
      );
    }
    const linewidth = options.linewidth ?? 1;
    assertFinitePositive("linewidth", linewidth);
    this.vlines.push({
      x,
      color: normalizeColor(options.color ?? this.textColor, "#000000"),
      linewidth,
      label: options.label ?? null,
    });
  }

  /**
   * Set the x-axis scale. On a twin axes this sets the shared x-axis.
   * @param scale - "linear" or "log"
   */
  setXScale(scale: "linear" | "log"): void {
    this.assertScale("setXScale", scale);
    const root = this.root();
    if (
      scale === "log" &&
      root.xLimOverride &&
      (root.xLimOverride[0] <= 0 || root.xLimOverride[1] <= 0)
    ) {
      throw new InvalidParameterError(
        "x limits must be positive to use a log scale",
        "scale",
        scale
      );
    }
    root.xScale = scale;
  }

  /**
   * Set the y-axis scale.
   * @param scale - "linear" or "log"
   */
  setYScale(scale: "linear" | "log"): void {
    this.assertScale("setYScale", scale);
    if (
      scale === "log" &&
      this.yLimOverride &&
      (this.yLimOverride[0] <= 0 || this.yLimOverride[1] <= 0)
    ) {
      throw new InvalidParameterError(
        "y limits must be positive to use a log scale",
        "scale",
        scale
      );
    }
    this.yScale = scale;
  }

  private assertScale(name: string, scale: string): void {
    if (scale !== "linear" && scale !== "log") {
      throw new InvalidParameterError(
        `${name} expects "linear" or "log"; received ${String(scale)}`,
        "scale",
        scale
      );
    }
  }

  /** Add text at data coordinates (alias of annotate, with the text as the third argument). */
  text(x: number, y: number, s: string, options: TextOptions = {}): void {
    this.annotate(s, x, y, options);
  }

  /**
   * Plot a step function.
   *
   * `where` selects how the steps are placed between samples:
   * - "post" (default): y[i] holds from x[i] up to x[i+1].
   * - "pre": y[i] holds from x[i-1] up to x[i] (matplotlib's default).
   * - "mid": the jump happens halfway between x[i] and x[i+1].
   *
   * @param x - 1D tensor of x coordinates
   * @param y - 1D tensor of y coordinates
   * @param options - Styling options and the step position
   */
  step(x: AnyTensor, y: AnyTensor, options: PlotOptions & { where?: StepPosition } = {}): Line2D {
    const xv = tensorToFloat64Vector1D(x);
    const yv = tensorToFloat64Vector1D(y);
    assertSameLength("x", xv.length, "y", yv.length);
    const where = options.where ?? "post";
    if (where !== "pre" && where !== "post" && where !== "mid") {
      throw new InvalidParameterError(
        `step where must be "pre", "post" or "mid"; received ${String(where)}`,
        "where",
        where
      );
    }
    const sx: number[] = [];
    const sy: number[] = [];
    for (let i = 0; i < xv.length; i++) {
      const xi = xv[i] ?? 0;
      const yi = yv[i] ?? 0;
      if (i > 0) {
        const xp = xv[i - 1] ?? 0;
        const yp = yv[i - 1] ?? 0;
        if (where === "post") {
          sx.push(xi);
          sy.push(yp);
        } else if (where === "pre") {
          sx.push(xp);
          sy.push(yi);
        } else {
          const xm = (xp + xi) / 2;
          sx.push(xm, xm);
          sy.push(yp, yi);
        }
      }
      // Under "mid" the interior samples lie on the flat runs, so only the ends are kept.
      if (where !== "mid" || i === 0 || i === xv.length - 1) {
        sx.push(xi);
        sy.push(yi);
      }
    }
    const d = new Line2D(new Float64Array(sx), new Float64Array(sy), options);
    this.drawables.push(d);
    return d;
  }

  /**
   * Plot error bars: the series as a line plus a vertical bar at each point.
   * @param x - 1D tensor of x coordinates
   * @param y - 1D tensor of y coordinates
   * @param yerr - Error magnitudes (non-negative): a 1D tensor of length n for symmetric
   *   bars, or a 2D tensor of shape [2, n] holding the lower and upper errors
   * @param options - Styling options
   */
  errorbar(x: AnyTensor, y: AnyTensor, yerr: AnyTensor, options: PlotOptions = {}): Line2D {
    const xv = tensorToFloat64Vector1D(x);
    const yv = tensorToFloat64Vector1D(y);
    assertSameLength("x", xv.length, "y", yv.length);
    const errTensor = "tensor" in yerr ? yerr.tensor : yerr;
    let lower: Float64Array;
    let upper: Float64Array;
    if (errTensor.ndim === 2) {
      const mat = tensorToFloat64Matrix2D(yerr);
      if (mat.rows !== 2) {
        throw new ShapeError(
          `2D yerr must have shape [2, n] (lower, upper); received [${mat.rows}, ${mat.cols}]`
        );
      }
      lower = mat.data.subarray(0, mat.cols);
      upper = mat.data.subarray(mat.cols);
    } else {
      lower = upper = tensorToFloat64Vector1D(yerr);
    }
    assertSameLength("x", xv.length, "yerr", lower.length);
    for (const side of [lower, upper]) {
      for (let i = 0; i < side.length; i++) {
        if ((side[i] ?? 0) < 0) {
          throw new InvalidParameterError(
            `yerr must not contain negative values; received ${side[i]} at index ${i}`,
            "yerr",
            side[i]
          );
        }
      }
    }
    // Plot the main line
    const mainLine = new Line2D(xv, yv, options);
    this.drawables.push(mainLine);
    // One vertical segment per point
    const color = normalizeColor(options.color, "#1f77b4");
    const lw = options.linewidth ?? 1;
    for (let i = 0; i < xv.length; i++) {
      const xi = xv[i] ?? 0;
      const yi = yv[i] ?? 0;
      const barLine = new Line2D(
        new Float64Array([xi, xi]),
        new Float64Array([yi - (lower[i] ?? 0), yi + (upper[i] ?? 0)]),
        { color, linewidth: lw }
      );
      this.drawables.push(barLine);
    }
    return mainLine;
  }

  /**
   * Fill the area between two y-curves. Samples where x, y1 or y2 is not finite are dropped.
   * @param x - 1D tensor of x coordinates
   * @param y1 - 1D tensor of lower y values
   * @param y2 - 1D tensor of upper y values
   * @param options - Styling options
   */
  fillBetween(x: AnyTensor, y1: AnyTensor, y2: AnyTensor, options: PlotOptions = {}): Line2D {
    const xv = tensorToFloat64Vector1D(x);
    const y1v = tensorToFloat64Vector1D(y1);
    const y2v = tensorToFloat64Vector1D(y2);
    assertSameLength("x", xv.length, "y1", y1v.length);
    assertSameLength("x", xv.length, "y2", y2v.length);
    return this.addFilledBetween(xv, y1v, y2v, options);
  }

  /**
   * Fill the area between two y-curves.
   * @deprecated Prefer {@link Axes.fillBetween}.
   */
  fill_between(x: AnyTensor, y1: AnyTensor, y2: AnyTensor, options: PlotOptions = {}): Line2D {
    return this.fillBetween(x, y1, y2, options);
  }

  private addFilledBetween(
    xv: Float64Array,
    y1v: Float64Array,
    y2v: Float64Array,
    options: PlotOptions
  ): Line2D {
    const kx: number[] = [];
    const k1: number[] = [];
    const k2: number[] = [];
    for (let i = 0; i < xv.length; i++) {
      const xi = xv[i] ?? Number.NaN;
      const a = y1v[i] ?? Number.NaN;
      const b = y2v[i] ?? Number.NaN;
      if (Number.isFinite(xi) && Number.isFinite(a) && Number.isFinite(b)) {
        kx.push(xi);
        k1.push(a);
        k2.push(b);
      }
    }
    // Forward along y2, backward along y1.
    const [px, py] = closePolygon([...kx, ...[...kx].reverse()], [...k2, ...[...k1].reverse()]);
    const d = new FilledPolygon(px, py, options);
    this.drawables.push(d);
    return d;
  }

  /**
   * Filled area between a curve and y = 0.
   * @param x - 1D tensor of x coordinates
   * @param y - 1D tensor of y values
   * @param options - Styling options
   */
  area(x: AnyTensor, y: AnyTensor, options: PlotOptions = {}): Line2D {
    const xv = tensorToFloat64Vector1D(x);
    const yv = tensorToFloat64Vector1D(y);
    assertSameLength("x", xv.length, "y", yv.length);
    return this.addFilledBetween(xv, new Float64Array(xv.length), yv, options);
  }

  /**
   * Plot stacked vertical bars for multiple series. Series are drawn bottom to
   * top; a non-finite height counts as 0 and its bar is not drawn.
   * @param x - 1D tensor of bar centers
   * @param heights - Array of 1D tensors (each as long as `x`), one per series
   * @param options - Colors and labels arrays for each series
   */
  stackedBar(
    x: AnyTensor,
    heights: readonly AnyTensor[],
    options: { colors?: readonly Color[]; labels?: readonly string[] } = {}
  ): void {
    const xv = tensorToFloat64Vector1D(x);
    const cumulative = new Float64Array(xv.length);
    const barW = 0.8;

    for (let s = 0; s < heights.length; s++) {
      const hv = tensorToFloat64Vector1D(heights[s] as AnyTensor);
      assertSameLength("x", xv.length, `heights[${s}]`, hv.length);
      const bottom = new Float64Array(cumulative);
      const clr = options.colors?.[s] ?? SERIES_COLORS[s % SERIES_COLORS.length] ?? "#000000";
      const lbl = options.labels?.[s];

      for (let i = 0; i < xv.length; i++) {
        const h = hv[i] ?? 0;
        if (Number.isFinite(h)) cumulative[i] = (cumulative[i] ?? 0) + h;
      }

      let labeled = false;
      for (let i = 0; i < xv.length; i++) {
        const xi = xv[i] ?? Number.NaN;
        if (!Number.isFinite(xi) || !Number.isFinite(hv[i] ?? Number.NaN)) continue;
        const lo = bottom[i] ?? 0;
        const hi = cumulative[i] ?? 0;
        const [rpx, rpy] = closePolygon(
          [xi - barW / 2, xi + barW / 2, xi + barW / 2, xi - barW / 2],
          [lo, lo, hi, hi]
        );
        const lineOpts: PlotOptions = !labeled && lbl ? { color: clr, label: lbl } : { color: clr };
        labeled = true;
        this.drawables.push(new FilledPolygon(rpx, rpy, lineOpts));
      }
    }
  }

  /**
   * Plot grouped (side-by-side) vertical bars for multiple series. The bars of
   * one group share a total width of 0.8 data units around each x.
   * @param x - 1D tensor of bar centers
   * @param heights - Array of 1D tensors (each as long as `x`), one per series
   * @param options - Colors and labels arrays for each series
   */
  groupedBar(
    x: AnyTensor,
    heights: readonly AnyTensor[],
    options: { colors?: readonly Color[]; labels?: readonly string[] } = {}
  ): void {
    const xv = tensorToFloat64Vector1D(x);
    const nGroups = heights.length;
    if (nGroups === 0) return;
    const totalWidth = 0.8;
    const barWidth = totalWidth / nGroups;

    for (let s = 0; s < nGroups; s++) {
      const hv = tensorToFloat64Vector1D(heights[s] as AnyTensor);
      assertSameLength("x", xv.length, `heights[${s}]`, hv.length);
      const clr = options.colors?.[s] ?? SERIES_COLORS[s % SERIES_COLORS.length] ?? "#000000";
      const lbl = options.labels?.[s];
      const offset = -totalWidth / 2 + barWidth * s + barWidth / 2;
      const offsetX = new Float64Array(xv.length);
      for (let i = 0; i < xv.length; i++) {
        offsetX[i] = (xv[i] ?? 0) + offset;
      }
      const barOpts: PlotOptions = lbl ? { color: clr, label: lbl } : { color: clr };
      this.drawables.push(new Bar2D(offsetX, hv, { ...barOpts, barWidth }));
    }
  }

  /**
   * Stem plot: vertical lines from baseline to data points with markers.
   */
  stem(x: AnyTensor, y: AnyTensor, options: PlotOptions & { baseline?: number } = {}): Stem2D {
    const xv = tensorToFloat64Vector1D(x);
    const yv = tensorToFloat64Vector1D(y);
    const d = new Stem2D(xv, yv, options);
    this.drawables.push(d);
    return d;
  }

  /**
   * Strip plot: jittered categorical scatter plot.
   */
  strip(
    groups: readonly AnyTensor[],
    options: {
      colors?: readonly Color[];
      labels?: readonly string[];
      size?: number;
      jitter?: number;
    } = {}
  ): Strip2D {
    const gs = groups.map((g) => tensorToFloat64Vector1D(g));
    const d = new Strip2D(gs, options);
    this.drawables.push(d);
    return d;
  }

  /**
   * Radar (spider) chart: multi-dimensional data on radial axes. Axis 0 points straight up
   * and the others follow clockwise. Every series is divided by the largest absolute value
   * over all series, so the outer ring is that maximum.
   * @throws {InvalidParameterError} If there are no series, fewer than 3 axes, series of
   *   different lengths, or a value that is not finite.
   */
  radar(
    series: readonly AnyTensor[],
    options: {
      colors?: readonly Color[];
      labels?: readonly string[];
      linewidth?: number;
    } = {}
  ): Radar2D {
    const ss = series.map((s) => tensorToFloat64Vector1D(s));
    const d = new Radar2D(ss, options);
    this.drawables.push(d);
    return d;
  }

  /**
   * Waterfall chart: cumulative positive/negative value changes. Step `i` is a bar
   * centered at x = i, and the category names become the x tick labels (replacing
   * custom x ticks set earlier; call `setXTicks` afterwards to override them).
   */
  waterfall(
    categories: readonly string[],
    values: AnyTensor,
    options: {
      positiveColor?: Color;
      negativeColor?: Color;
      totalColor?: Color;
      barWidth?: number;
    } = {}
  ): Waterfall2D {
    const vv = tensorToFloat64Vector1D(values);
    const d = new Waterfall2D([...categories], vv, options);
    this.drawables.push(d);
    if (categories.length > 0) {
      this.setXTicks(
        categories.map((_, i) => i),
        categories
      );
    }
    return d;
  }

  /**
   * Quiver plot: vector field arrows.
   */
  quiver(
    x: AnyTensor,
    y: AnyTensor,
    u: AnyTensor,
    v: AnyTensor,
    options: PlotOptions & { scale?: number } = {}
  ): Quiver2D {
    const xv = tensorToFloat64Vector1D(x);
    const yv = tensorToFloat64Vector1D(y);
    const uv = tensorToFloat64Vector1D(u);
    const vv = tensorToFloat64Vector1D(v);
    const d = new Quiver2D(xv, yv, uv, vv, options);
    this.drawables.push(d);
    return d;
  }

  /**
   * Polar plot: data in polar coordinates (theta, r).
   */
  polar(theta: AnyTensor, r: AnyTensor, options: PlotOptions & { fill?: boolean } = {}): Polar2D {
    const tv = tensorToFloat64Vector1D(theta);
    const rv = tensorToFloat64Vector1D(r);
    const d = new Polar2D(tv, rv, options);
    this.drawables.push(d);
    return d;
  }

  /**
   * 3D surface plot.
   */
  surface(
    xGrid: Float64Array[],
    yGrid: Float64Array[],
    zGrid: Float64Array[],
    options: Plot3DOptions = {}
  ): Surface3D {
    const d = new Surface3D(xGrid, yGrid, zGrid, options);
    this.drawables.push(d);
    return d;
  }

  /**
   * 3D wireframe plot.
   */
  wireframe(
    xGrid: Float64Array[],
    yGrid: Float64Array[],
    zGrid: Float64Array[],
    options: Plot3DOptions = {}
  ): Wireframe3D {
    const d = new Wireframe3D(xGrid, yGrid, zGrid, options);
    this.drawables.push(d);
    return d;
  }

  /**
   * 3D scatter plot.
   */
  scatter3d(
    x: Float64Array,
    y: Float64Array,
    z: Float64Array,
    options: Plot3DOptions = {}
  ): Scatter3D {
    const d = new Scatter3D(x, y, z, options);
    this.drawables.push(d);
    return d;
  }

  /**
   * Axis ranges in data coordinates: explicit limits where set, otherwise the
   * automatic range (5% margin; in log10 space on log axes). Axes that share an
   * x-axis through `twinx()` use one common x range; each keeps its own y range.
   */
  private resolveRange(): AxisRange {
    const root = this.root();
    const group = this.xGroup();
    const own = this.rangeSources();
    const shared = group.length === 1 ? own : group.flatMap((ax) => ax.rangeSources());
    const autoY = computeAutoRange(own);
    const autoX = group.length === 1 ? autoY : computeAutoRange(shared);
    let xmin = autoX.xmin;
    let xmax = autoX.xmax;
    let ymin = autoY.ymin;
    let ymax = autoY.ymax;
    if (root.xLimOverride) {
      [xmin, xmax] = root.xLimOverride;
    } else if (root.xScale === "log") {
      [xmin, xmax] = logAutoRange(shared, "x");
    }
    if (this.yLimOverride) {
      [ymin, ymax] = this.yLimOverride;
    } else if (this.yScale === "log") {
      [ymin, ymax] = logAutoRange(own, "y");
    }
    return { xmin, xmax, ymin, ymax };
  }

  /**
   * True when every drawable is a 3D plot (surface, wireframe or scatter3d). Those are drawn
   * in a projected view whose screen coordinates (about -0.7 to 0.7) are not data values, so
   * the axes shows no tick labels or grid lines for them.
   */
  private isProjected3D(): boolean {
    return this.drawables.length > 0 && this.drawables.every((d) => THREE_D_KINDS.has(d.kind));
  }

  /** Drawables plus reference lines, so `axhline`/`axvline` positions enter the auto range. */
  private rangeSources(): readonly Drawable[] {
    if (this.hlines.length === 0 && this.vlines.length === 0) return this.drawables;
    return [
      ...this.drawables,
      ...this.hlines.filter((h) => this.canShowY(h.y)).map((h) => referenceRange("axhline", h.y)),
      ...this.vlines.filter((v) => this.canShowX(v.x)).map((v) => referenceRange("axvline", v.x)),
    ];
  }

  // A log axis cannot show a non-positive reference line.
  private canShowX(x: number): boolean {
    return this.root().xScale !== "log" || x > 0;
  }

  private canShowY(y: number): boolean {
    return this.yScale !== "log" || y > 0;
  }

  /**
   * Ticks for one axis, respecting log scale. On a log axis, ticks are placed
   * at decade boundaries (..., 0.1, 1, 10, 100, ...) with the data VALUE as the
   * label. The x-axis state is owned by the root axes of a twin group.
   */
  private ticksFor(axis: "x" | "y", lo: number, hi: number, maxTicks: number): readonly Tick[] {
    const root = this.root();
    const override = axis === "x" ? root.xTicksOverride : this.yTicksOverride;
    if (override) return override;
    const isLog = axis === "x" ? root.xScale === "log" : this.yScale === "log";
    if (!isLog) return generateTicks(lo, hi, maxTicks);
    return generateLogTicks(lo, hi);
  }

  /** Ticks that fall inside the visible range (with a tiny tolerance for rounding). */
  private computeTicks(
    vp: Viewport,
    range: AxisRange
  ): { readonly xTicks: readonly Tick[]; readonly yTicks: readonly Tick[] } {
    const maxXTicks = Math.max(2, Math.floor(vp.width / 80));
    const maxYTicks = Math.max(2, Math.floor(vp.height / 60));
    const within = (value: number, a: number, b: number): boolean => {
      const lo = Math.min(a, b);
      const hi = Math.max(a, b);
      const eps = (hi - lo) * 1e-9;
      return value >= lo - eps && value <= hi + eps;
    };
    return {
      xTicks: this.ticksFor("x", range.xmin, range.xmax, maxXTicks).filter((t) =>
        within(t.value, range.xmin, range.xmax)
      ),
      yTicks: this.ticksFor("y", range.ymin, range.ymax, maxYTicks).filter((t) =>
        within(t.value, range.ymin, range.ymax)
      ),
    };
  }

  private applyLogRange(range: AxisRange): AxisRange {
    const r = { ...range };
    if (this.root().xScale === "log") {
      r.xmin = r.xmin > 0 ? Math.log10(r.xmin) : -1;
      r.xmax = r.xmax > 0 ? Math.log10(r.xmax) : 1;
    }
    if (this.yScale === "log") {
      r.ymin = r.ymin > 0 ? Math.log10(r.ymin) : -1;
      r.ymax = r.ymax > 0 ? Math.log10(r.ymax) : 1;
    }
    return r;
  }

  private wrapLogTransform(base: {
    readonly xToPx: (x: number) => number;
    readonly yToPx: (y: number) => number;
  }): {
    readonly xToPx: (x: number) => number;
    readonly yToPx: (y: number) => number;
  } {
    const logX = this.root().xScale === "log";
    const logY = this.yScale === "log";
    if (!logX && !logY) return base;
    // Like matplotlib's nonpositive="clip", a value that is not positive is sent far outside
    // the visible range (log10 = -1000) instead of being drawn at some visible position.
    const logOf = (v: number): number => (v > 0 ? Math.log10(v) : LOG_CLIP);
    return {
      xToPx: logX ? (x: number) => base.xToPx(logOf(x)) : base.xToPx,
      yToPx: logY ? (y: number) => base.yToPx(logOf(y)) : base.yToPx,
    };
  }

  private viewport(): Viewport {
    const p = this.padding;
    const base: Viewport = this.baseViewport ?? {
      x: 0,
      y: 0,
      width: this.fig.width,
      height: this.fig.height,
    };
    return {
      x: base.x + p,
      y: base.y + p,
      width: base.width - 2 * p,
      height: base.height - 2 * p,
    };
  }

  private collectLegendEntries(): readonly LegendEntry[] {
    const entries: LegendEntry[] = [];
    const seen = new Set<string>();
    const add = (entry: LegendEntry): void => {
      const trimmed = entry.label.trim();
      if (!trimmed || seen.has(trimmed)) return;
      seen.add(trimmed);
      entries.push({ ...entry, label: trimmed });
    };
    for (const drawable of this.drawables) {
      const list = drawable.getLegendEntries?.();
      if (!list) continue;
      for (const entry of list) add(entry);
    }
    for (const line of [...this.hlines, ...this.vlines]) {
      if (line.label === null) continue;
      add({ label: line.label, color: line.color, shape: "line", lineWidth: line.linewidth });
    }
    return entries;
  }

  private resolveLegendOptions(): Required<LegendOptions> {
    const options = this.legendOptions ?? {};
    return {
      visible: options.visible ?? true,
      location: options.location ?? "upper-right",
      fontSize: options.fontSize ?? this.legendFontSize,
      padding: options.padding ?? 6,
      background: normalizeColor(options.background ?? this.legendBackground, "#ffffff"),
      borderColor: normalizeColor(options.borderColor ?? this.textColor, "#000000"),
    };
  }

  /**
   * Distance in pixels from the axes frame to the center line of an axis label. It grows with
   * the label font size above the default of 12 so that larger labels keep clear of the ticks.
   */
  private labelGap(): number {
    return 35 + Math.max(0, this.labelFontSize - 12);
  }

  private buildTickOverride(
    values: readonly number[],
    labels: readonly string[] | undefined,
    name: string
  ): readonly Tick[] | null {
    if (values.length === 0) return null;
    if (labels && labels.length !== values.length) {
      throw new InvalidParameterError(
        `${name} labels length must match values length (${values.length}); received ${labels.length}`,
        name,
        labels
      );
    }
    const ticks: Tick[] = [];
    for (const [i, value] of values.entries()) {
      if (!Number.isFinite(value)) {
        throw new InvalidParameterError(`${name} values must be finite`, name, value);
      }
      const label = labels ? (labels[i] ?? "") : String(value);
      ticks.push({ value, label: label.trim() });
    }
    return ticks;
  }

  /** Legend box geometry; `measure` returns the rendered width of a label. */
  private layoutLegend(vp: Viewport, measure: (label: string, fontSize: number) => number) {
    if (!this.legendOptions) return null;
    const entries = this.collectLegendEntries();
    if (entries.length === 0) return null;
    const options = this.resolveLegendOptions();
    if (!options.visible) return null;

    const fontSize = options.fontSize;
    const padding = options.padding;
    const symbolSize = Math.max(8, Math.round(fontSize * 0.9));
    const gap = 6;
    let maxLabelWidth = 0;
    for (const entry of entries) {
      maxLabelWidth = Math.max(maxLabelWidth, measure(entry.label, fontSize));
    }
    const lineHeight = fontSize + 4;
    const boxWidth = padding * 2 + symbolSize + gap + maxLabelWidth;
    const boxHeight = padding * 2 + entries.length * lineHeight;
    const margin = 6;
    const isRight = options.location.includes("right");
    const isUpper = options.location.includes("upper");
    const boxX = isRight ? vp.x + vp.width - boxWidth - margin : vp.x + margin;
    const boxY = isUpper ? vp.y + margin : vp.y + vp.height - boxHeight - margin;
    return {
      entries,
      options,
      fontSize,
      padding,
      symbolSize,
      gap,
      lineHeight,
      boxWidth,
      boxHeight,
      boxX,
      boxY,
    };
  }

  private renderTicksSVG(
    elements: string[],
    vp: Viewport,
    range: AxisRange,
    transform: {
      readonly xToPx: (x: number) => number;
      readonly yToPx: (y: number) => number;
    }
  ): void {
    const { xTicks, yTicks } = this.computeTicks(vp, range);
    const tickLength = 5;
    const labelOffset = 2;

    // Twin axes skip x-ticks (shared with parent)
    if (!this.isTwinX) {
      for (const tick of xTicks) {
        const px = transform.xToPx(tick.value);
        elements.push(
          `<line class="x-tick" x1="${px.toFixed(2)}" y1="${(vp.y + vp.height).toFixed(
            2
          )}" x2="${px.toFixed(2)}" y2="${(vp.y + vp.height + tickLength).toFixed(
            2
          )}" stroke="${escapeXml(this.textColor)}" />`
        );
        elements.push(
          `<text class="tick-label tick-label-x" x="${px.toFixed(2)}" y="${(
            vp.y + vp.height + tickLength + labelOffset
          ).toFixed(
            2
          )}" text-anchor="middle" dominant-baseline="hanging" font-size="${this.tickFontSize}" fill="${escapeXml(this.textColor)}">${escapeXml(
            tick.label
          )}</text>`
        );
      }
    }

    for (const tick of yTicks) {
      const py = transform.yToPx(tick.value);
      if (this.isTwinX) {
        // Render y-ticks on the right side
        elements.push(
          `<line class="y-tick" x1="${(vp.x + vp.width).toFixed(2)}" y1="${py.toFixed(2)}" x2="${(
            vp.x + vp.width + tickLength
          ).toFixed(2)}" y2="${py.toFixed(2)}" stroke="${escapeXml(this.textColor)}" />`
        );
        elements.push(
          `<text class="tick-label tick-label-y" x="${(
            vp.x + vp.width + tickLength + labelOffset
          ).toFixed(
            2
          )}" y="${py.toFixed(2)}" text-anchor="start" dominant-baseline="middle" font-size="${this.tickFontSize}" fill="${escapeXml(this.textColor)}">${escapeXml(
            tick.label
          )}</text>`
        );
      } else {
        elements.push(
          `<line class="y-tick" x1="${vp.x.toFixed(2)}" y1="${py.toFixed(2)}" x2="${(
            vp.x - tickLength
          ).toFixed(2)}" y2="${py.toFixed(2)}" stroke="${escapeXml(this.textColor)}" />`
        );
        elements.push(
          `<text class="tick-label tick-label-y" x="${(vp.x - tickLength - labelOffset).toFixed(
            2
          )}" y="${py.toFixed(2)}" text-anchor="end" dominant-baseline="middle" font-size="${this.tickFontSize}" fill="${escapeXml(this.textColor)}">${escapeXml(
            tick.label
          )}</text>`
        );
      }
    }
  }

  private renderLegendSVG(elements: string[], vp: Viewport): void {
    const layout = this.layoutLegend(vp, estimateTextWidth);
    if (!layout) return;
    const {
      entries,
      options,
      fontSize,
      padding,
      symbolSize,
      gap,
      lineHeight,
      boxWidth,
      boxHeight,
      boxX,
      boxY,
    } = layout;

    elements.push(`<g class="legend">`);
    elements.push(
      `<rect class="legend-box" x="${boxX.toFixed(2)}" y="${boxY.toFixed(
        2
      )}" width="${boxWidth.toFixed(2)}" height="${boxHeight.toFixed(
        2
      )}" fill="${escapeXml(options.background)}" stroke="${escapeXml(options.borderColor)}" />`
    );

    for (let i = 0; i < entries.length; i++) {
      const entry = entries[i];
      if (!entry) continue;
      const rowY = boxY + padding + i * lineHeight + lineHeight / 2;
      const symbolX = boxX + padding;
      const symbolY = rowY - symbolSize / 2;

      if (entry.shape === "line") {
        const lineY = rowY;
        const lineWidth = entry.lineWidth ?? 2;
        elements.push(
          `<line class="legend-line" x1="${symbolX.toFixed(2)}" y1="${lineY.toFixed(
            2
          )}" x2="${(symbolX + symbolSize).toFixed(2)}" y2="${lineY.toFixed(
            2
          )}" stroke="${escapeXml(entry.color)}" stroke-width="${lineWidth}" />`
        );
      } else if (entry.shape === "marker") {
        const radius = Math.max(2, Math.min(symbolSize / 2, entry.markerSize ?? symbolSize / 2));
        const cx = symbolX + symbolSize / 2;
        const cy = rowY;
        elements.push(
          `<circle class="legend-marker" cx="${cx.toFixed(2)}" cy="${cy.toFixed(
            2
          )}" r="${radius.toFixed(2)}" fill="${escapeXml(entry.color)}" />`
        );
      } else {
        elements.push(
          `<rect class="legend-swatch" x="${symbolX.toFixed(2)}" y="${symbolY.toFixed(
            2
          )}" width="${symbolSize.toFixed(2)}" height="${symbolSize.toFixed(
            2
          )}" fill="${escapeXml(entry.color)}" stroke="${escapeXml(this.textColor)}" />`
        );
      }

      const textX = symbolX + symbolSize + gap;
      elements.push(
        `<text class="legend-label" x="${textX.toFixed(2)}" y="${rowY.toFixed(
          2
        )}" text-anchor="start" dominant-baseline="middle" font-size="${fontSize}" fill="${escapeXml(this.textColor)}">${escapeXml(
          entry.label
        )}</text>`
      );
    }

    elements.push(`</g>`);
  }

  private renderTicksRaster(
    canvas: RasterCanvas,
    vp: Viewport,
    range: AxisRange,
    transform: {
      readonly xToPx: (x: number) => number;
      readonly yToPx: (y: number) => number;
    }
  ): void {
    const { xTicks, yTicks } = this.computeTicks(vp, range);
    const tickLength = 5;
    const labelOffset = 2;
    const text = parseHexColorToRGBA(this.textColor);

    // Twin axes skip x-ticks (shared with parent)
    if (!this.isTwinX) {
      for (const tick of xTicks) {
        const px = Math.round(transform.xToPx(tick.value));
        const y0 = Math.round(vp.y + vp.height);
        const y1 = y0 + tickLength;
        canvas.drawLineRGBA(px, y0, px, y1, text.r, text.g, text.b, text.a);
        canvas.drawTextRGBA(tick.label, px, y1 + labelOffset, text.r, text.g, text.b, text.a, {
          fontSize: this.tickFontSize,
          align: "middle",
          baseline: "top",
        });
      }
    }

    for (const tick of yTicks) {
      const py = Math.round(transform.yToPx(tick.value));
      if (this.isTwinX) {
        // Render y-ticks on the right side
        const x0 = Math.round(vp.x + vp.width);
        const x1 = x0 + tickLength;
        canvas.drawLineRGBA(x0, py, x1, py, text.r, text.g, text.b, text.a);
        canvas.drawTextRGBA(tick.label, x1 + labelOffset, py, text.r, text.g, text.b, text.a, {
          fontSize: this.tickFontSize,
          align: "start",
          baseline: "middle",
        });
      } else {
        const x0 = Math.round(vp.x);
        const x1 = x0 - tickLength;
        canvas.drawLineRGBA(x0, py, x1, py, text.r, text.g, text.b, text.a);
        canvas.drawTextRGBA(tick.label, x1 - labelOffset, py, text.r, text.g, text.b, text.a, {
          fontSize: this.tickFontSize,
          align: "end",
          baseline: "middle",
        });
      }
    }
  }

  private renderLegendRaster(canvas: RasterCanvas, vp: Viewport): void {
    // Size the box from the bitmap font so labels never spill out of it.
    const layout = this.layoutLegend(
      vp,
      (label, fontSize) => canvas.measureText(label, fontSize).width
    );
    if (!layout) return;
    const {
      entries,
      options,
      fontSize,
      padding,
      symbolSize,
      gap,
      lineHeight,
      boxWidth,
      boxHeight,
      boxX,
      boxY,
    } = layout;

    const bg = parseHexColorToRGBA(options.background);
    const border = parseHexColorToRGBA(options.borderColor);
    const bx = Math.round(boxX);
    const by = Math.round(boxY);
    const bw = Math.round(boxWidth);
    const bh = Math.round(boxHeight);
    canvas.fillRectRGBA(bx, by, bw, bh, bg.r, bg.g, bg.b, bg.a);
    canvas.drawLineRGBA(bx, by, bx + bw, by, border.r, border.g, border.b, border.a);
    canvas.drawLineRGBA(bx, by + bh, bx + bw, by + bh, border.r, border.g, border.b, border.a);
    canvas.drawLineRGBA(bx, by, bx, by + bh, border.r, border.g, border.b, border.a);
    canvas.drawLineRGBA(bx + bw, by, bx + bw, by + bh, border.r, border.g, border.b, border.a);

    const text = parseHexColorToRGBA(this.textColor);
    for (let i = 0; i < entries.length; i++) {
      const entry = entries[i];
      if (!entry) continue;
      const rowY = boxY + padding + i * lineHeight + lineHeight / 2;
      const symbolX = boxX + padding;
      const symbolY = rowY - symbolSize / 2;
      const color = parseHexColorToRGBA(entry.color);

      if (entry.shape === "line") {
        const y = Math.round(rowY);
        const x0 = Math.round(symbolX);
        const x1 = Math.round(symbolX + symbolSize);
        canvas.drawLineRGBA(x0, y, x1, y, color.r, color.g, color.b, color.a);
      } else if (entry.shape === "marker") {
        const radius = Math.max(2, Math.min(symbolSize / 2, entry.markerSize ?? symbolSize / 2));
        const cx = Math.round(symbolX + symbolSize / 2);
        const cy = Math.round(rowY);
        canvas.drawCircleRGBA(cx, cy, Math.round(radius), color.r, color.g, color.b, color.a);
      } else {
        canvas.fillRectRGBA(
          Math.round(symbolX),
          Math.round(symbolY),
          Math.round(symbolSize),
          Math.round(symbolSize),
          color.r,
          color.g,
          color.b,
          color.a
        );
      }

      const textX = symbolX + symbolSize + gap;
      canvas.drawTextRGBA(
        entry.label,
        Math.round(textX),
        Math.round(rowY),
        text.r,
        text.g,
        text.b,
        text.a,
        { fontSize, align: "start", baseline: "middle" }
      );
    }
  }

  /** Dashed (4 on, 3 off) axis-aligned reference line, `linewidth` pixels thick. */
  private drawReferenceLineRaster(
    canvas: RasterCanvas,
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    color: Color,
    linewidth: number
  ): void {
    const c = parseHexColorToRGBA(color);
    const thickness = Math.max(1, Math.round(linewidth));
    const first = -Math.floor((thickness - 1) / 2);
    const horizontal = y0 === y1;
    const length = horizontal ? Math.abs(x1 - x0) : Math.abs(y1 - y0);
    const start = horizontal ? Math.min(x0, x1) : Math.min(y0, y1);
    for (let k = 0; k < thickness; k++) {
      const off = first + k;
      for (let d = 0; d <= length; d += 7) {
        const e = Math.min(d + 3, length);
        if (horizontal) {
          canvas.drawLineRGBA(start + d, y0 + off, start + e, y0 + off, c.r, c.g, c.b, c.a);
        } else {
          canvas.drawLineRGBA(x0 + off, start + d, x0 + off, start + e, c.r, c.g, c.b, c.a);
        }
      }
    }
  }

  /**
   * Render axes to SVG elements.
   * @internal
   */
  renderSVGInto(elements: string[]): void {
    const vp = this.viewport();
    const range = this.resolveRange();
    const baseTransform = makeTransform(this.applyLogRange(range), vp);
    const transform = this.wrapLogTransform(baseTransform);
    // Twin axes overlay the parent: skip background and frame
    if (!this.isTwinX) {
      elements.push(
        `<rect x="${fmt(vp.x)}" y="${fmt(vp.y)}" width="${fmt(vp.width)}" height="${fmt(vp.height)}" fill="${escapeXml(this.facecolor)}" stroke="${escapeXml(this.textColor)}" />`
      );
    }

    if (this.gridVisible && !this.isProjected3D()) {
      const { xTicks, yTicks } = this.computeTicks(vp, range);
      for (const t of xTicks) {
        const px = transform.xToPx(t.value);
        elements.push(
          `<line x1="${px.toFixed(2)}" y1="${fmt(vp.y)}" x2="${px.toFixed(2)}" y2="${(vp.y + vp.height).toFixed(2)}" stroke="${escapeXml(this.gridColor)}" stroke-width="0.5" />`
        );
      }
      for (const t of yTicks) {
        const py = transform.yToPx(t.value);
        elements.push(
          `<line x1="${fmt(vp.x)}" y1="${py.toFixed(2)}" x2="${(vp.x + vp.width).toFixed(2)}" y2="${py.toFixed(2)}" stroke="${escapeXml(this.gridColor)}" stroke-width="0.5" />`
        );
      }
    }

    const ctx: SvgDrawContext = {
      transform,
      push: (el) => {
        elements.push(el);
      },
    };

    // Clip data series to the axes viewport so out-of-range geometry (e.g.
    // under ylim/xlim overrides) doesn't bleed over the title band or into
    // neighbouring subplots.
    const clipId = `dbclip-${Math.round(vp.x)}-${Math.round(vp.y)}-${Math.round(vp.width)}-${Math.round(vp.height)}`;
    // Use a <path> (not <rect>) for the clip shape so it doesn't perturb tests
    // or tooling that count data <rect> elements.
    const cx0 = fmt(vp.x);
    const cy0 = fmt(vp.y);
    const cx1 = fmt(vp.x + vp.width);
    const cy1 = fmt(vp.y + vp.height);
    elements.push(
      `<clipPath id="${clipId}"><path d="M${cx0} ${cy0} L${cx1} ${cy0} L${cx1} ${cy1} L${cx0} ${cy1} Z" /></clipPath>`,
      `<g clip-path="url(#${clipId})">`
    );
    for (const d of this.drawables) d.drawSVG(ctx);
    elements.push("</g>");

    // Render horizontal reference lines (those outside the axes are not drawn)
    for (const hl of this.hlines) {
      if (!this.canShowY(hl.y)) continue;
      const pyNum = transform.yToPx(hl.y);
      if (!(pyNum >= vp.y - 1e-6 && pyNum <= vp.y + vp.height + 1e-6)) continue;
      const py = pyNum.toFixed(2);
      elements.push(
        `<line class="axhline" x1="${vp.x.toFixed(2)}" y1="${py}" x2="${(vp.x + vp.width).toFixed(2)}" y2="${py}" stroke="${escapeXml(hl.color)}" stroke-width="${hl.linewidth}" stroke-dasharray="4,3" />`
      );
    }

    // Render vertical reference lines
    for (const vl of this.vlines) {
      if (!this.canShowX(vl.x)) continue;
      const pxNum = transform.xToPx(vl.x);
      if (!(pxNum >= vp.x - 1e-6 && pxNum <= vp.x + vp.width + 1e-6)) continue;
      const px = pxNum.toFixed(2);
      elements.push(
        `<line class="axvline" x1="${px}" y1="${vp.y.toFixed(2)}" x2="${px}" y2="${(vp.y + vp.height).toFixed(2)}" stroke="${escapeXml(vl.color)}" stroke-width="${vl.linewidth}" stroke-dasharray="4,3" />`
      );
    }

    for (const ann of this.annotations) {
      const px = transform.xToPx(ann.x).toFixed(2);
      const py = transform.yToPx(ann.y).toFixed(2);
      const anchor =
        ann.ha === "center"
          ? ' text-anchor="middle"'
          : ann.ha === "right"
            ? ' text-anchor="end"'
            : "";
      const baseline =
        ann.va === "center"
          ? ' dominant-baseline="middle"'
          : ann.va === "top"
            ? ' dominant-baseline="hanging"'
            : "";
      elements.push(
        `<text x="${px}" y="${py}"${anchor}${baseline} font-size="${ann.fontSize}" fill="${escapeXml(ann.color)}">${escapeXml(ann.text)}</text>`
      );
    }

    const shouldRenderTicks =
      (this.drawables.length > 0 || this.hlines.length > 0 || this.vlines.length > 0) &&
      !this.isProjected3D();
    if (shouldRenderTicks) {
      this.renderTicksSVG(elements, vp, range, transform);
    }

    if (this.title) {
      const titleX = fmt(vp.x + vp.width / 2);
      const titleY = fmt(vp.y - 10);
      elements.push(
        `<text x="${titleX}" y="${titleY}" text-anchor="middle" font-size="${this.titleFontSize}" font-weight="bold" fill="${escapeXml(this.textColor)}">${escapeXml(this.title)}</text>`
      );
    }

    if (this.xlabel) {
      const xlabelX = fmt(vp.x + vp.width / 2);
      const xlabelY = fmt(vp.y + vp.height + this.labelGap());
      elements.push(
        `<text x="${xlabelX}" y="${xlabelY}" text-anchor="middle" font-size="${this.labelFontSize}" fill="${escapeXml(this.textColor)}">${escapeXml(this.xlabel)}</text>`
      );
    }

    if (this.ylabel) {
      if (this.isTwinX) {
        // Render y-label on the right side for twin axes
        const ylabelX = fmt(vp.x + vp.width + this.labelGap());
        const ylabelY = fmt(vp.y + vp.height / 2);
        elements.push(
          `<text x="${ylabelX}" y="${ylabelY}" text-anchor="middle" font-size="${this.labelFontSize}" fill="${escapeXml(this.textColor)}" transform="rotate(90 ${ylabelX} ${ylabelY})">${escapeXml(this.ylabel)}</text>`
        );
      } else {
        const ylabelX = fmt(vp.x - this.labelGap());
        const ylabelY = fmt(vp.y + vp.height / 2);
        elements.push(
          `<text x="${ylabelX}" y="${ylabelY}" text-anchor="middle" font-size="${this.labelFontSize}" fill="${escapeXml(this.textColor)}" transform="rotate(-90 ${ylabelX} ${ylabelY})">${escapeXml(this.ylabel)}</text>`
        );
      }
    }

    this.renderLegendSVG(elements, vp);
  }

  /**
   * Render axes to raster canvas.
   * @internal
   */
  renderRasterInto(canvas: RasterCanvas): void {
    const vp = this.viewport();
    const range = this.resolveRange();
    const baseTransform = makeTransform(this.applyLogRange(range), vp);
    const transform = this.wrapLogTransform(baseTransform);
    const ctx = { transform, canvas };

    // Twin axes overlay the parent: skip background and frame
    if (!this.isTwinX) {
      const bg = parseHexColorToRGBA(this.facecolor);
      const x0 = Math.round(vp.x);
      const y0 = Math.round(vp.y);
      const w = Math.max(0, Math.round(vp.width));
      const h = Math.max(0, Math.round(vp.height));
      if (w > 0 && h > 0) {
        canvas.fillRectRGBA(x0, y0, w, h, bg.r, bg.g, bg.b, bg.a);
        const edge = parseHexColorToRGBA(this.textColor);
        const x1 = x0 + w;
        const y1 = y0 + h;
        canvas.drawLineRGBA(x0, y0, x1, y0, edge.r, edge.g, edge.b, edge.a);
        canvas.drawLineRGBA(x0, y1, x1, y1, edge.r, edge.g, edge.b, edge.a);
        canvas.drawLineRGBA(x0, y0, x0, y1, edge.r, edge.g, edge.b, edge.a);
        canvas.drawLineRGBA(x1, y0, x1, y1, edge.r, edge.g, edge.b, edge.a);
      }
    }

    if (this.gridVisible && !this.isProjected3D()) {
      const { xTicks, yTicks } = this.computeTicks(vp, range);
      const gc = parseHexColorToRGBA(this.gridColor);
      const gx0 = Math.round(vp.x);
      const gx1 = Math.round(vp.x + vp.width);
      const gy0 = Math.round(vp.y);
      const gy1 = Math.round(vp.y + vp.height);
      for (const t of xTicks) {
        const px = Math.round(transform.xToPx(t.value));
        canvas.drawLineRGBA(px, gy0, px, gy1, gc.r, gc.g, gc.b, gc.a);
      }
      for (const t of yTicks) {
        const py = Math.round(transform.yToPx(t.value));
        canvas.drawLineRGBA(gx0, py, gx1, py, gc.r, gc.g, gc.b, gc.a);
      }
    }

    // Clip data series to the axes viewport (data must not paint over the
    // title band or neighbouring subplots).
    canvas.setClipRect(
      Math.round(vp.x),
      Math.round(vp.y),
      Math.round(vp.x + vp.width),
      Math.round(vp.y + vp.height)
    );
    try {
      for (const d of this.drawables) d.drawRaster(ctx);
    } finally {
      canvas.clearClip();
    }

    // Render horizontal reference lines (those outside the axes are not drawn)
    for (const hl of this.hlines) {
      if (!this.canShowY(hl.y)) continue;
      const pyNum = transform.yToPx(hl.y);
      if (!(pyNum >= vp.y - 1e-6 && pyNum <= vp.y + vp.height + 1e-6)) continue;
      const py = Math.round(pyNum);
      this.drawReferenceLineRaster(
        canvas,
        Math.round(vp.x),
        py,
        Math.round(vp.x + vp.width),
        py,
        hl.color,
        hl.linewidth
      );
    }

    // Render vertical reference lines
    for (const vl of this.vlines) {
      if (!this.canShowX(vl.x)) continue;
      const pxNum = transform.xToPx(vl.x);
      if (!(pxNum >= vp.x - 1e-6 && pxNum <= vp.x + vp.width + 1e-6)) continue;
      const px = Math.round(pxNum);
      this.drawReferenceLineRaster(
        canvas,
        px,
        Math.round(vp.y),
        px,
        Math.round(vp.y + vp.height),
        vl.color,
        vl.linewidth
      );
    }

    for (const ann of this.annotations) {
      const c = parseHexColorToRGBA(ann.color);
      canvas.drawTextRGBA(
        ann.text,
        transform.xToPx(ann.x),
        transform.yToPx(ann.y),
        c.r,
        c.g,
        c.b,
        c.a,
        {
          fontSize: ann.fontSize,
          align: ann.ha === "center" ? "middle" : ann.ha === "right" ? "end" : "start",
          baseline: ann.va === "center" ? "middle" : ann.va === "top" ? "top" : "bottom",
        }
      );
    }

    const shouldRenderTicks =
      (this.drawables.length > 0 || this.hlines.length > 0 || this.vlines.length > 0) &&
      !this.isProjected3D();
    if (shouldRenderTicks) {
      this.renderTicksRaster(canvas, vp, range, transform);
    }

    const text = parseHexColorToRGBA(this.textColor);
    if (this.title) {
      const titleX = vp.x + vp.width / 2;
      const titleY = vp.y - 10;
      canvas.drawTextRGBA(this.title, titleX, titleY, text.r, text.g, text.b, text.a, {
        fontSize: this.titleFontSize,
        align: "middle",
        baseline: "bottom",
      });
    }

    if (this.xlabel) {
      const xlabelX = vp.x + vp.width / 2;
      const xlabelY = vp.y + vp.height + this.labelGap();
      canvas.drawTextRGBA(this.xlabel, xlabelX, xlabelY, text.r, text.g, text.b, text.a, {
        fontSize: this.labelFontSize,
        align: "middle",
        baseline: "top",
      });
    }

    if (this.ylabel) {
      if (this.isTwinX) {
        const ylabelX = vp.x + vp.width + this.labelGap();
        const ylabelY = vp.y + vp.height / 2;
        canvas.drawTextRGBA(this.ylabel, ylabelX, ylabelY, text.r, text.g, text.b, text.a, {
          fontSize: this.labelFontSize,
          align: "middle",
          baseline: "middle",
          rotation: 90,
        });
      } else {
        const ylabelX = vp.x - this.labelGap();
        const ylabelY = vp.y + vp.height / 2;
        canvas.drawTextRGBA(this.ylabel, ylabelX, ylabelY, text.r, text.g, text.b, text.a, {
          fontSize: this.labelFontSize,
          align: "middle",
          baseline: "middle",
          rotation: -90,
        });
      }
    }

    this.renderLegendRaster(canvas, vp);
  }
}
