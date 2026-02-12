import { InvalidParameterError } from "../../core";
import type { AnyTensor } from "../../ndarray";
import type { RasterCanvas } from "../canvas/RasterCanvas";
import { Bar2D } from "../plots/Bar2D";
import { Boxplot } from "../plots/Boxplot";
import { Contour2D } from "../plots/Contour2D";
import { ContourF2D } from "../plots/ContourF2D";
import { Heatmap2D } from "../plots/Heatmap2D";
import { Histogram } from "../plots/Histogram";
import { HorizontalBar2D } from "../plots/HorizontalBar2D";
import { Line2D } from "../plots/Line2D";
import { Pie } from "../plots/Pie";
import { Scatter2D } from "../plots/Scatter2D";
import { Violinplot } from "../plots/Violinplot";
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
import { generateTicks, type Tick } from "../utils/ticks";
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
    const xTicks = (this.xTicksOverride ?? generateTicks(range.xmin, range.xmax, maxXTicks)).filter(
      (tick) => tick.value >= range.xmin && tick.value <= range.xmax
    );
    const yTicks = (this.yTicksOverride ?? generateTicks(range.ymin, range.ymax, maxYTicks)).filter(
      (tick) => tick.value >= range.ymin && tick.value <= range.ymax
    );
    const tickLength = 5;
    const labelOffset = 2;

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

    for (const tick of yTicks) {
      const py = transform.yToPx(tick.value);
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
    const xTicks = (this.xTicksOverride ?? generateTicks(range.xmin, range.xmax, maxXTicks)).filter(
      (tick) => tick.value >= range.xmin && tick.value <= range.xmax
    );
    const yTicks = (this.yTicksOverride ?? generateTicks(range.ymin, range.ymax, maxYTicks)).filter(
      (tick) => tick.value >= range.ymin && tick.value <= range.ymax
    );
    const tickLength = 5;
    const labelOffset = 2;
    const text = parseHexColorToRGBA("#000000");

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

    for (const tick of yTicks) {
      const py = Math.round(transform.yToPx(tick.value));
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
    const range = computeAutoRange(this.drawables);
    const transform = makeTransform(range, vp);
    elements.push(
      `<rect x="${vp.x}" y="${vp.y}" width="${vp.width}" height="${vp.height}" fill="${escapeXml(this.facecolor)}" stroke="#000" />`
    );

    const ctx: SvgDrawContext = {
      transform,
      push: (el) => {
        elements.push(el);
      },
    };

    for (const d of this.drawables) d.drawSVG(ctx);

    const shouldRenderTicks = this.drawables.length > 0;
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
      const ylabelX = vp.x - 35;
      const ylabelY = vp.y + vp.height / 2;
      elements.push(
        `<text x="${ylabelX}" y="${ylabelY}" text-anchor="middle" font-size="12" fill="#000" transform="rotate(-90 ${ylabelX} ${ylabelY})">${escapeXml(this.ylabel)}</text>`
      );
    }

    this.renderLegendSVG(elements, vp);
  }

  /**
   * Render axes to raster canvas.
   * @internal
   */
  renderRasterInto(canvas: RasterCanvas): void {
    const vp = this.viewport();
    const range = computeAutoRange(this.drawables);
    const transform = makeTransform(range, vp);
    const ctx = { transform, canvas };

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
    for (const d of this.drawables) d.drawRaster(ctx);

    const shouldRenderTicks = this.drawables.length > 0;
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
      const ylabelX = vp.x - 35;
      const ylabelY = vp.y + vp.height / 2;
      canvas.drawTextRGBA(this.ylabel, ylabelX, ylabelY, text.r, text.g, text.b, text.a, {
        fontSize: 12,
        align: "middle",
        baseline: "middle",
        rotation: -90,
      });
    }

    this.renderLegendRaster(canvas, vp);
  }
}
