/**
 * @see {@link https://deepbox.dev/docs/plot-basic | Deepbox documentation}
 */

import { InvalidParameterError } from "../../core";
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
import { Violinplot } from "../plots/Violinplot";
import { Waterfall2D } from "../plots/Waterfall2D";
import type {
  Color,
  Drawable,
  LegendEntry,
  LegendOptions,
  PlotOptions,
  SvgDrawContext,
  Viewport,
} from "../types";
import { normalizeColor, parseHexColorToRGBA } from "../utils/colors";
import { buildContourGrid } from "../utils/contours";
import { tensorToFloat64Matrix2D, tensorToFloat64Vector1D } from "../utils/tensor";
import { estimateTextWidth } from "../utils/text";
import { generateLogTicks, generateTicks, type Tick } from "../utils/ticks";
import { computeAutoRange, makeTransform } from "../utils/transforms";
import { escapeXml } from "../utils/xml";
import type { Figure } from "./Figure";

/**
 * An Axes represents a single plot area within a Figure.
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
  }>;
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

    this.facecolor = normalizeColor(options.facecolor, "#ffffff");
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
    this.gridVisible = false;
    this.gridColor = "#cccccc";
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
   * Set custom x-axis tick positions and labels.
   * @param values - Tick positions in data coordinates
   * @param labels - Optional tick labels
   */
  setXTicks(values: readonly number[], labels?: readonly string[]): void {
    this.xTicksOverride = this.buildTickOverride(values, labels, "xTicks");
  }

  /**
   * Set custom y-axis tick positions and labels.
   * @param values - Tick positions in data coordinates
   * @param labels - Optional tick labels
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
   * @param data - 2D tensor of values
   * @param options - Styling and scale options
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
   * [clusterA, clusterB, distance, count].
   *
   * @param linkage - Linkage matrix rows
   * @param nLeaves - Number of original observations
   * @param options - Styling options
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
   * Display a matrix as an image (alias of heatmap).
   * @param data - 2D tensor of values
   * @param options - Styling and scale options
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
   * Plot a violin distribution summary for a 1D dataset.
   * @param data - 1D tensor of values
   * @param options - Styling options
   */
  violinplot(data: AnyTensor, options: PlotOptions = {}): Violinplot {
    const values = tensorToFloat64Vector1D(data);
    const d = new Violinplot(1, values, options);
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
   * rendered on the right side.  The returned axes overlays this one so that
   * two different y-scales can be shown side-by-side.
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
    this.fig.axesList.push(twin);
    return twin;
  }

  /** Set explicit x-axis limits. */
  xlim(min: number, max: number): void {
    this.xLimOverride = [min, max];
  }

  /** Set explicit y-axis limits. */
  ylim(min: number, max: number): void {
    this.yLimOverride = [min, max];
  }

  /** Show or hide grid lines. */
  grid(visible = true, options: { color?: Color } = {}): void {
    this.gridVisible = visible;
    if (options.color) this.gridColor = options.color;
  }

  /** Add a text annotation at data coordinates. */
  annotate(
    text: string,
    x: number,
    y: number,
    options: { color?: Color; fontSize?: number } = {}
  ): void {
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
    this.annotations.push({
      text,
      x,
      y,
      color: normalizeColor(options.color, "#000000"),
      fontSize: options.fontSize ?? 10,
    });
  }

  /**
   * Draw a horizontal line across the axes at the given y value.
   * @param y - Y-coordinate in data space
   * @param options - Color, linewidth, and optional label
   */
  axhline(y: number, options: { color?: Color; linewidth?: number; label?: string } = {}): void {
    this.hlines.push({
      y,
      color: normalizeColor(options.color, "#000000"),
      linewidth: options.linewidth ?? 1,
      label: options.label ?? null,
    });
  }

  /**
   * Draw a vertical line across the axes at the given x value.
   * @param x - X-coordinate in data space
   * @param options - Color, linewidth, and optional label
   */
  axvline(x: number, options: { color?: Color; linewidth?: number; label?: string } = {}): void {
    this.vlines.push({
      x,
      color: normalizeColor(options.color, "#000000"),
      linewidth: options.linewidth ?? 1,
      label: options.label ?? null,
    });
  }

  /**
   * Set the x-axis scale.
   * @param scale - "linear" or "log"
   */
  setXScale(scale: "linear" | "log"): void {
    this.xScale = scale;
  }

  /**
   * Set the y-axis scale.
   * @param scale - "linear" or "log"
   */
  setYScale(scale: "linear" | "log"): void {
    this.yScale = scale;
  }

  /** Add text at data coordinates (alias of annotate). */
  text(x: number, y: number, s: string, options: { color?: Color; fontSize?: number } = {}): void {
    this.annotate(s, x, y, options);
  }

  /**
   * Plot a step function.
   * @param x - 1D tensor of x coordinates
   * @param y - 1D tensor of y coordinates
   * @param options - Styling options
   */
  step(x: AnyTensor, y: AnyTensor, options: PlotOptions = {}): Line2D {
    const xv = tensorToFloat64Vector1D(x);
    const yv = tensorToFloat64Vector1D(y);
    // Convert to step: for each pair, insert intermediate horizontal segment
    const sx: number[] = [];
    const sy: number[] = [];
    for (let i = 0; i < xv.length; i++) {
      if (i > 0) {
        sx.push(xv[i]!);
        sy.push(yv[i - 1]!);
      }
      sx.push(xv[i]!);
      sy.push(yv[i]!);
    }
    const d = new Line2D(new Float64Array(sx), new Float64Array(sy), options);
    this.drawables.push(d);
    return d;
  }

  /**
   * Plot error bars.
   * @param x - 1D tensor of x coordinates
   * @param y - 1D tensor of y coordinates
   * @param yerr - 1D tensor of y error magnitudes (symmetric) or [ylo, yhi] pair
   * @param options - Styling options
   */
  errorbar(x: AnyTensor, y: AnyTensor, yerr: AnyTensor, options: PlotOptions = {}): Line2D {
    const xv = tensorToFloat64Vector1D(x);
    const yv = tensorToFloat64Vector1D(y);
    const ev = tensorToFloat64Vector1D(yerr);
    // Plot the main line
    const mainLine = new Line2D(xv, yv, options);
    this.drawables.push(mainLine);
    // Plot error bar caps as individual line segments
    const color = normalizeColor(options.color, "#1f77b4");
    const lw = options.linewidth ?? 1;
    for (let i = 0; i < xv.length; i++) {
      const xi = xv[i]!;
      const yi = yv[i]!;
      const ei = ev[i] ?? 0;
      // Vertical error bar
      const barLine = new Line2D(new Float64Array([xi, xi]), new Float64Array([yi - ei, yi + ei]), {
        color,
        linewidth: lw,
      });
      this.drawables.push(barLine);
    }
    return mainLine;
  }

  /**
   * Fill the area between two y-curves.
   * @param x - 1D tensor of x coordinates
   * @param y1 - 1D tensor of lower y values
   * @param y2 - 1D tensor of upper y values
   * @param options - Styling options
   */
  fill_between(x: AnyTensor, y1: AnyTensor, y2: AnyTensor, options: PlotOptions = {}): Line2D {
    const xv = tensorToFloat64Vector1D(x);
    const y1v = tensorToFloat64Vector1D(y1);
    const y2v = tensorToFloat64Vector1D(y2);
    // Create a closed polygon path: forward along y2, backward along y1
    const px = [...Array.from(xv), ...Array.from(xv).reverse()];
    const py = [...Array.from(y2v), ...Array.from(y1v).reverse()];
    const d = new Line2D(new Float64Array(px), new Float64Array(py), options);
    this.drawables.push(d);
    return d;
  }

  /** Alias: filled area plot. */
  area(x: AnyTensor, y: AnyTensor, options: PlotOptions = {}): Line2D {
    const xv = tensorToFloat64Vector1D(x);
    const yv = tensorToFloat64Vector1D(y);
    // Fill between y and 0
    const zeros = new Array(xv.length).fill(0) as number[];
    const color = normalizeColor(options.color, "#1f77b4");
    const px = [...Array.from(xv), ...Array.from(xv).reverse()];
    const py = [...Array.from(yv), ...zeros.reverse()];
    const d = new Line2D(new Float64Array(px), new Float64Array(py), {
      ...options,
      color,
    });
    this.drawables.push(d);
    return d;
  }

  /**
   * Plot stacked vertical bars for multiple series.
   * @param x - 1D tensor of bar centers
   * @param heights - Array of 1D tensors, one per series (stacked bottom-to-top)
   * @param options - Colors and labels arrays for each series
   */
  stackedBar(
    x: AnyTensor,
    heights: readonly AnyTensor[],
    options: { colors?: readonly Color[]; labels?: readonly string[] } = {}
  ): void {
    const xv = tensorToFloat64Vector1D(x);
    const stackColors = [
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
    const cumulative = new Float64Array(xv.length);
    const barW = 0.8;

    for (let s = 0; s < heights.length; s++) {
      const hv = tensorToFloat64Vector1D(heights[s]!);
      const bottom = new Float64Array(cumulative);
      const clr = options.colors?.[s] ?? stackColors[s % stackColors.length] ?? "#000000";
      const lbl = options.labels?.[s];

      for (let i = 0; i < xv.length; i++) {
        cumulative[i] = (cumulative[i] ?? 0) + (hv[i] ?? 0);
      }

      for (let i = 0; i < xv.length; i++) {
        const xi = xv[i] ?? 0;
        const lo = bottom[i] ?? 0;
        const hi = cumulative[i] ?? 0;
        const rpx = new Float64Array([xi - barW / 2, xi + barW / 2, xi + barW / 2, xi - barW / 2]);
        const rpy = new Float64Array([lo, lo, hi, hi]);
        const lineOpts: PlotOptions = i === 0 && lbl ? { color: clr, label: lbl } : { color: clr };
        const rect = new Line2D(rpx, rpy, lineOpts);
        this.drawables.push(rect);
      }
    }
  }

  /**
   * Plot grouped (side-by-side) vertical bars for multiple series.
   * @param x - 1D tensor of bar centers
   * @param heights - Array of 1D tensors, one per series
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
    const grpColors = [
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
    const totalWidth = 0.8;
    const barWidth = totalWidth / nGroups;

    for (let s = 0; s < nGroups; s++) {
      const hv = tensorToFloat64Vector1D(heights[s]!);
      const clr = options.colors?.[s] ?? grpColors[s % grpColors.length] ?? "#000000";
      const lbl = options.labels?.[s];
      const offset = -totalWidth / 2 + barWidth * s + barWidth / 2;
      const offsetX = new Float64Array(xv.length);
      for (let i = 0; i < xv.length; i++) {
        offsetX[i] = (xv[i] ?? 0) + offset;
      }
      const barOpts: PlotOptions = lbl ? { color: clr, label: lbl } : { color: clr };
      const d = new Bar2D(offsetX, hv, barOpts);
      this.drawables.push(d);
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
   * Radar (spider) chart: multi-dimensional data on radial axes.
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
   * Waterfall chart: cumulative positive/negative value changes.
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
   * Ticks for one axis, respecting log scale. On a log axis, ticks are placed
   * at decade boundaries (…, 0.1, 1, 10, 100, …) with the data VALUE as the
   * label — generating linear ticks and positioning them through the log
   * transform (the previous behavior) produced wrong labels crammed against
   * one edge plus a spurious "0" tick.
   */
  private ticksFor(
    axis: "x" | "y",
    lo: number,
    hi: number,
    maxTicks: number,
    override: readonly Tick[] | null | undefined
  ): readonly Tick[] {
    if (override) return override;
    const isLog = axis === "x" ? this.xScale === "log" : this.yScale === "log";
    if (!isLog) return generateTicks(lo, hi, maxTicks);
    return generateLogTicks(lo, hi);
  }

  private applyLogRange(range: { xmin: number; xmax: number; ymin: number; ymax: number }): {
    xmin: number;
    xmax: number;
    ymin: number;
    ymax: number;
  } {
    const r = { ...range };
    if (this.xScale === "log") {
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
    if (this.xScale === "linear" && this.yScale === "linear") return base;
    const logX = this.xScale === "log";
    const logY = this.yScale === "log";
    return {
      xToPx: logX ? (x: number) => base.xToPx(x > 0 ? Math.log10(x) : -1) : base.xToPx,
      yToPx: logY ? (y: number) => base.yToPx(y > 0 ? Math.log10(y) : -1) : base.yToPx,
    };
  }

  private applyLimOverrides(range: { xmin: number; xmax: number; ymin: number; ymax: number }): {
    xmin: number;
    xmax: number;
    ymin: number;
    ymax: number;
  } {
    return {
      xmin: this.xLimOverride ? this.xLimOverride[0] : range.xmin,
      xmax: this.xLimOverride ? this.xLimOverride[1] : range.xmax,
      ymin: this.yLimOverride ? this.yLimOverride[0] : range.ymin,
      ymax: this.yLimOverride ? this.yLimOverride[1] : range.ymax,
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
    for (const drawable of this.drawables) {
      const list = drawable.getLegendEntries?.();
      if (!list) continue;
      for (const entry of list) {
        const trimmed = entry.label.trim();
        if (!trimmed) continue;
        if (seen.has(trimmed)) continue;
        seen.add(trimmed);
        entries.push({ ...entry, label: trimmed });
      }
    }
    return entries;
  }

  private resolveLegendOptions(): Required<LegendOptions> {
    const options = this.legendOptions ?? {};
    return {
      visible: options.visible ?? true,
      location: options.location ?? "upper-right",
      fontSize: options.fontSize ?? 12,
      padding: options.padding ?? 6,
      background: normalizeColor(options.background, "#ffffff"),
      borderColor: normalizeColor(options.borderColor, "#000000"),
    };
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

  private renderTicksSVG(
    elements: string[],
    vp: Viewport,
    range: {
      readonly xmin: number;
      readonly xmax: number;
      readonly ymin: number;
      readonly ymax: number;
    },
    transform: {
      readonly xToPx: (x: number) => number;
      readonly yToPx: (y: number) => number;
    }
  ): void {
    const maxXTicks = Math.max(2, Math.floor(vp.width / 80));
    const maxYTicks = Math.max(2, Math.floor(vp.height / 60));
    const xTicks = this.ticksFor(
      "x",
      range.xmin,
      range.xmax,
      maxXTicks,
      this.xTicksOverride
    ).filter((tick) => tick.value >= range.xmin && tick.value <= range.xmax);
    const yTicks = this.ticksFor(
      "y",
      range.ymin,
      range.ymax,
      maxYTicks,
      this.yTicksOverride
    ).filter((tick) => tick.value >= range.ymin && tick.value <= range.ymax);
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
          )}" stroke="#000" />`
        );
        elements.push(
          `<text class="tick-label tick-label-x" x="${px.toFixed(2)}" y="${(
            vp.y + vp.height + tickLength + labelOffset
          ).toFixed(
            2
          )}" text-anchor="middle" dominant-baseline="hanging" font-size="10" fill="#000">${escapeXml(
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
          ).toFixed(2)}" y2="${py.toFixed(2)}" stroke="#000" />`
        );
        elements.push(
          `<text class="tick-label tick-label-y" x="${(
            vp.x + vp.width + tickLength + labelOffset
          ).toFixed(
            2
          )}" y="${py.toFixed(2)}" text-anchor="start" dominant-baseline="middle" font-size="10" fill="#000">${escapeXml(
            tick.label
          )}</text>`
        );
      } else {
        elements.push(
          `<line class="y-tick" x1="${vp.x.toFixed(2)}" y1="${py.toFixed(2)}" x2="${(
            vp.x - tickLength
          ).toFixed(2)}" y2="${py.toFixed(2)}" stroke="#000" />`
        );
        elements.push(
          `<text class="tick-label tick-label-y" x="${(vp.x - tickLength - labelOffset).toFixed(
            2
          )}" y="${py.toFixed(2)}" text-anchor="end" dominant-baseline="middle" font-size="10" fill="#000">${escapeXml(
            tick.label
          )}</text>`
        );
      }
    }
  }

  private renderLegendSVG(elements: string[], vp: Viewport): void {
    if (!this.legendOptions) return;
    const entries = this.collectLegendEntries();
    if (entries.length === 0) return;
    const options = this.resolveLegendOptions();
    if (!options.visible) return;

    const fontSize = options.fontSize;
    const padding = options.padding;
    const symbolSize = Math.max(8, Math.round(fontSize * 0.9));
    const gap = 6;
    let maxLabelWidth = 0;
    for (const entry of entries) {
      const width = estimateTextWidth(entry.label, fontSize);
      maxLabelWidth = Math.max(maxLabelWidth, width);
    }
    const lineHeight = fontSize + 4;
    const boxWidth = padding * 2 + symbolSize + gap + maxLabelWidth;
    const boxHeight = padding * 2 + entries.length * lineHeight;
    const margin = 6;

    const isRight = options.location.includes("right");
    const isUpper = options.location.includes("upper");
    const boxX = isRight ? vp.x + vp.width - boxWidth - margin : vp.x + margin;
    const boxY = isUpper ? vp.y + margin : vp.y + vp.height - boxHeight - margin;

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
          )}" fill="${escapeXml(entry.color)}" stroke="#000" />`
        );
      }

      const textX = symbolX + symbolSize + gap;
      elements.push(
        `<text class="legend-label" x="${textX.toFixed(2)}" y="${rowY.toFixed(
          2
        )}" text-anchor="start" dominant-baseline="middle" font-size="${fontSize}" fill="#000">${escapeXml(
          entry.label
        )}</text>`
      );
    }

    elements.push(`</g>`);
  }

  private renderTicksRaster(
    canvas: RasterCanvas,
    vp: Viewport,
    range: {
      readonly xmin: number;
      readonly xmax: number;
      readonly ymin: number;
      readonly ymax: number;
    },
    transform: {
      readonly xToPx: (x: number) => number;
      readonly yToPx: (y: number) => number;
    }
  ): void {
    const maxXTicks = Math.max(2, Math.floor(vp.width / 80));
    const maxYTicks = Math.max(2, Math.floor(vp.height / 60));
    const xTicks = this.ticksFor(
      "x",
      range.xmin,
      range.xmax,
      maxXTicks,
      this.xTicksOverride
    ).filter((tick) => tick.value >= range.xmin && tick.value <= range.xmax);
    const yTicks = this.ticksFor(
      "y",
      range.ymin,
      range.ymax,
      maxYTicks,
      this.yTicksOverride
    ).filter((tick) => tick.value >= range.ymin && tick.value <= range.ymax);
    const tickLength = 5;
    const labelOffset = 2;
    const text = parseHexColorToRGBA("#000000");

    // Twin axes skip x-ticks (shared with parent)
    if (!this.isTwinX) {
      for (const tick of xTicks) {
        const px = Math.round(transform.xToPx(tick.value));
        const y0 = Math.round(vp.y + vp.height);
        const y1 = y0 + tickLength;
        canvas.drawLineRGBA(px, y0, px, y1, text.r, text.g, text.b, text.a);
        canvas.drawTextRGBA(tick.label, px, y1 + labelOffset, text.r, text.g, text.b, text.a, {
          fontSize: 10,
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
          fontSize: 10,
          align: "start",
          baseline: "middle",
        });
      } else {
        const x0 = Math.round(vp.x);
        const x1 = x0 - tickLength;
        canvas.drawLineRGBA(x0, py, x1, py, text.r, text.g, text.b, text.a);
        canvas.drawTextRGBA(tick.label, x1 - labelOffset, py, text.r, text.g, text.b, text.a, {
          fontSize: 10,
          align: "end",
          baseline: "middle",
        });
      }
    }
  }

  private renderLegendRaster(canvas: RasterCanvas, vp: Viewport): void {
    if (!this.legendOptions) return;
    const entries = this.collectLegendEntries();
    if (entries.length === 0) return;
    const options = this.resolveLegendOptions();
    if (!options.visible) return;

    const fontSize = options.fontSize;
    const padding = options.padding;
    const symbolSize = Math.max(8, Math.round(fontSize * 0.9));
    const gap = 6;
    let maxLabelWidth = 0;
    for (const entry of entries) {
      maxLabelWidth = Math.max(maxLabelWidth, estimateTextWidth(entry.label, fontSize));
    }
    const lineHeight = fontSize + 4;
    const boxWidth = padding * 2 + symbolSize + gap + maxLabelWidth;
    const boxHeight = padding * 2 + entries.length * lineHeight;
    const margin = 6;
    const isRight = options.location.includes("right");
    const isUpper = options.location.includes("upper");
    const boxX = isRight ? vp.x + vp.width - boxWidth - margin : vp.x + margin;
    const boxY = isUpper ? vp.y + margin : vp.y + vp.height - boxHeight - margin;

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

    const text = parseHexColorToRGBA("#000000");
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

  /**
   * Render axes to SVG elements.
   * @internal
   */
  renderSVGInto(elements: string[]): void {
    const vp = this.viewport();
    const autoRange = computeAutoRange(this.drawables);
    const range = this.applyLimOverrides(autoRange);
    const baseTransform = makeTransform(this.applyLogRange(range), vp);
    const transform = this.wrapLogTransform(baseTransform);
    // Twin axes overlay the parent — skip background and frame
    if (!this.isTwinX) {
      elements.push(
        `<rect x="${vp.x}" y="${vp.y}" width="${vp.width}" height="${vp.height}" fill="${escapeXml(this.facecolor)}" stroke="#000" />`
      );
    }

    if (this.gridVisible) {
      const maxXT = Math.max(2, Math.floor(vp.width / 80));
      const maxYT = Math.max(2, Math.floor(vp.height / 60));
      const gxTicks = this.ticksFor("x", range.xmin, range.xmax, maxXT, this.xTicksOverride).filter(
        (t) => t.value >= range.xmin && t.value <= range.xmax
      );
      const gyTicks = this.ticksFor("y", range.ymin, range.ymax, maxYT, this.yTicksOverride).filter(
        (t) => t.value >= range.ymin && t.value <= range.ymax
      );
      for (const t of gxTicks) {
        const px = transform.xToPx(t.value);
        elements.push(
          `<line x1="${px.toFixed(2)}" y1="${vp.y}" x2="${px.toFixed(2)}" y2="${(vp.y + vp.height).toFixed(2)}" stroke="${escapeXml(this.gridColor)}" stroke-width="0.5" />`
        );
      }
      for (const t of gyTicks) {
        const py = transform.yToPx(t.value);
        elements.push(
          `<line x1="${vp.x}" y1="${py.toFixed(2)}" x2="${(vp.x + vp.width).toFixed(2)}" y2="${py.toFixed(2)}" stroke="${escapeXml(this.gridColor)}" stroke-width="0.5" />`
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
    const cx0 = vp.x;
    const cy0 = vp.y;
    const cx1 = vp.x + vp.width;
    const cy1 = vp.y + vp.height;
    elements.push(
      `<clipPath id="${clipId}"><path d="M${cx0} ${cy0} L${cx1} ${cy0} L${cx1} ${cy1} L${cx0} ${cy1} Z" /></clipPath>`,
      `<g clip-path="url(#${clipId})">`
    );
    for (const d of this.drawables) d.drawSVG(ctx);
    elements.push("</g>");

    // Render horizontal reference lines
    for (const hl of this.hlines) {
      const py = transform.yToPx(hl.y).toFixed(2);
      elements.push(
        `<line class="axhline" x1="${vp.x.toFixed(2)}" y1="${py}" x2="${(vp.x + vp.width).toFixed(2)}" y2="${py}" stroke="${escapeXml(hl.color)}" stroke-width="${hl.linewidth}" stroke-dasharray="4,3" />`
      );
    }

    // Render vertical reference lines
    for (const vl of this.vlines) {
      const px = transform.xToPx(vl.x).toFixed(2);
      elements.push(
        `<line class="axvline" x1="${px}" y1="${vp.y.toFixed(2)}" x2="${px}" y2="${(vp.y + vp.height).toFixed(2)}" stroke="${escapeXml(vl.color)}" stroke-width="${vl.linewidth}" stroke-dasharray="4,3" />`
      );
    }

    for (const ann of this.annotations) {
      const px = transform.xToPx(ann.x).toFixed(2);
      const py = transform.yToPx(ann.y).toFixed(2);
      elements.push(
        `<text x="${px}" y="${py}" font-size="${ann.fontSize}" fill="${escapeXml(ann.color)}">${escapeXml(ann.text)}</text>`
      );
    }

    const shouldRenderTicks =
      this.drawables.length > 0 || this.hlines.length > 0 || this.vlines.length > 0;
    if (shouldRenderTicks) {
      this.renderTicksSVG(elements, vp, range, transform);
    }

    if (this.title) {
      const titleX = vp.x + vp.width / 2;
      const titleY = vp.y - 10;
      elements.push(
        `<text x="${titleX}" y="${titleY}" text-anchor="middle" font-size="14" font-weight="bold" fill="#000">${escapeXml(this.title)}</text>`
      );
    }

    if (this.xlabel) {
      const xlabelX = vp.x + vp.width / 2;
      const xlabelY = vp.y + vp.height + 35;
      elements.push(
        `<text x="${xlabelX}" y="${xlabelY}" text-anchor="middle" font-size="12" fill="#000">${escapeXml(this.xlabel)}</text>`
      );
    }

    if (this.ylabel) {
      if (this.isTwinX) {
        // Render y-label on the right side for twin axes
        const ylabelX = vp.x + vp.width + 35;
        const ylabelY = vp.y + vp.height / 2;
        elements.push(
          `<text x="${ylabelX}" y="${ylabelY}" text-anchor="middle" font-size="12" fill="#000" transform="rotate(90 ${ylabelX} ${ylabelY})">${escapeXml(this.ylabel)}</text>`
        );
      } else {
        const ylabelX = vp.x - 35;
        const ylabelY = vp.y + vp.height / 2;
        elements.push(
          `<text x="${ylabelX}" y="${ylabelY}" text-anchor="middle" font-size="12" fill="#000" transform="rotate(-90 ${ylabelX} ${ylabelY})">${escapeXml(this.ylabel)}</text>`
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
    const autoRange = computeAutoRange(this.drawables);
    const range = this.applyLimOverrides(autoRange);
    const baseTransform = makeTransform(this.applyLogRange(range), vp);
    const transform = this.wrapLogTransform(baseTransform);
    const ctx = { transform, canvas };

    // Twin axes overlay the parent — skip background and frame
    if (!this.isTwinX) {
      const bg = parseHexColorToRGBA(this.facecolor);
      const x0 = Math.round(vp.x);
      const y0 = Math.round(vp.y);
      const w = Math.max(0, Math.round(vp.width));
      const h = Math.max(0, Math.round(vp.height));
      if (w > 0 && h > 0) {
        canvas.fillRectRGBA(x0, y0, w, h, bg.r, bg.g, bg.b, bg.a);
        const edge = parseHexColorToRGBA("#000000");
        const x1 = x0 + w;
        const y1 = y0 + h;
        canvas.drawLineRGBA(x0, y0, x1, y0, edge.r, edge.g, edge.b, edge.a);
        canvas.drawLineRGBA(x0, y1, x1, y1, edge.r, edge.g, edge.b, edge.a);
        canvas.drawLineRGBA(x0, y0, x0, y1, edge.r, edge.g, edge.b, edge.a);
        canvas.drawLineRGBA(x1, y0, x1, y1, edge.r, edge.g, edge.b, edge.a);
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
    for (const d of this.drawables) d.drawRaster(ctx);
    canvas.clearClip();

    // Render horizontal reference lines
    for (const hl of this.hlines) {
      const py = Math.round(transform.yToPx(hl.y));
      const hlColor = parseHexColorToRGBA(hl.color);
      canvas.drawLineRGBA(
        Math.round(vp.x),
        py,
        Math.round(vp.x + vp.width),
        py,
        hlColor.r,
        hlColor.g,
        hlColor.b,
        hlColor.a
      );
    }

    // Render vertical reference lines
    for (const vl of this.vlines) {
      const px = Math.round(transform.xToPx(vl.x));
      const vlColor = parseHexColorToRGBA(vl.color);
      canvas.drawLineRGBA(
        px,
        Math.round(vp.y),
        px,
        Math.round(vp.y + vp.height),
        vlColor.r,
        vlColor.g,
        vlColor.b,
        vlColor.a
      );
    }

    const shouldRenderTicks =
      this.drawables.length > 0 || this.hlines.length > 0 || this.vlines.length > 0;
    if (shouldRenderTicks) {
      this.renderTicksRaster(canvas, vp, range, transform);
    }

    const text = parseHexColorToRGBA("#000000");
    if (this.title) {
      const titleX = vp.x + vp.width / 2;
      const titleY = vp.y - 10;
      canvas.drawTextRGBA(this.title, titleX, titleY, text.r, text.g, text.b, text.a, {
        fontSize: 14,
        align: "middle",
        baseline: "bottom",
      });
    }

    if (this.xlabel) {
      const xlabelX = vp.x + vp.width / 2;
      const xlabelY = vp.y + vp.height + 35;
      canvas.drawTextRGBA(this.xlabel, xlabelX, xlabelY, text.r, text.g, text.b, text.a, {
        fontSize: 12,
        align: "middle",
        baseline: "top",
      });
    }

    if (this.ylabel) {
      if (this.isTwinX) {
        const ylabelX = vp.x + vp.width + 35;
        const ylabelY = vp.y + vp.height / 2;
        canvas.drawTextRGBA(this.ylabel, ylabelX, ylabelY, text.r, text.g, text.b, text.a, {
          fontSize: 12,
          align: "middle",
          baseline: "middle",
          rotation: 90,
        });
      } else {
        const ylabelX = vp.x - 35;
        const ylabelY = vp.y + vp.height / 2;
        canvas.drawTextRGBA(this.ylabel, ylabelX, ylabelY, text.r, text.g, text.b, text.a, {
          fontSize: 12,
          align: "middle",
          baseline: "middle",
          rotation: -90,
        });
      }
    }

    this.renderLegendRaster(canvas, vp);
  }
}
