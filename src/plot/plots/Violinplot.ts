/**
 * Violin plot drawable: a Gaussian kernel density estimate mirrored around a position, with
 * the quartiles marked.
 *
 * @module plot/plots/Violinplot
 * @see {@link https://deepbox.dev/docs/plot-basic | Deepbox documentation}
 */

import { InvalidParameterError } from "../../core";
import type {
  Color,
  DataRange,
  Drawable,
  LegendEntry,
  PlotOptions,
  RasterDrawContext,
  SvgDrawContext,
} from "../types";
import { normalizeColor, parseHexColorToRGBA } from "../utils/colors";
import { buildLegendEntry, normalizeLegendLabel } from "../utils/legend";
import {
  calculateQuartiles,
  kernelDensityEstimation,
  scottBandwidth,
  silvermanBandwidth,
} from "../utils/statistics";
import { escapeXml } from "../utils/xml";

/** Number of points at which the density is evaluated. */
const KDE_POINTS = 100;

/**
 * Options of a violin plot: the common {@link PlotOptions} plus the layout and the density
 * estimate.
 */
export type ViolinplotOptions = PlotOptions & {
  /** x position of the violin (default 1). Used by `Axes.violinplot`; must be finite. */
  readonly position?: number;
  /** Largest width of the violin in x units (default 0.8). Must be positive and finite. */
  readonly width?: number;
  /**
   * Kernel bandwidth of the density estimate: "silverman" (default, `(3n/4)^(-1/5) s`),
   * "scott" (`n^(-1/5) s`, matplotlib's default) or a positive number, the standard deviation
   * of the Gaussian kernel in data units.
   */
  readonly bandwidth?: "silverman" | "scott" | number;
};

/**
 * Violin plot drawable.
 *
 * Non-finite input values are ignored. An empty input draws nothing; a non-empty input with no
 * finite value throws. The density uses `options.bandwidth` (Silverman's rule by default) and is
 * evaluated at 100 points spanning the data range extended by 10% on each side (by three
 * bandwidths when all values are equal, so a constant sample still has a visible shape).
 * The violin is centered on the `position` argument; `options.position` is only read by
 * `Axes.violinplot`.
 * @internal
 */
export class Violinplot implements Drawable {
  readonly kind = "violinplot";
  readonly position: number;
  readonly q1: number;
  readonly median: number;
  readonly q3: number;
  readonly kdePoints: readonly number[];
  readonly kdeValues: readonly number[];
  readonly color: Color;
  readonly edgecolor: Color;
  readonly violinWidth: number;
  readonly label: string | null;
  readonly hasData: boolean;

  constructor(position: number, data: Float64Array, options: ViolinplotOptions) {
    if (!Number.isFinite(position)) {
      throw new InvalidParameterError(
        `violinplot position must be finite; received ${position}`,
        "position",
        position
      );
    }
    const width = options.width ?? 0.8;
    if (!Number.isFinite(width) || width <= 0) {
      throw new InvalidParameterError(
        `violinplot width must be a positive finite number; received ${width}`,
        "width",
        width
      );
    }
    const rule = options.bandwidth ?? "silverman";
    if (
      rule !== "silverman" &&
      rule !== "scott" &&
      (typeof rule !== "number" || !Number.isFinite(rule) || rule <= 0)
    ) {
      throw new InvalidParameterError(
        `violinplot bandwidth must be "silverman", "scott" or a positive number; received ${String(rule)}`,
        "bandwidth",
        rule
      );
    }
    this.position = position;
    this.color = normalizeColor(options.color, "#8c564b");
    this.edgecolor = normalizeColor(options.edgecolor, "#000000");
    this.violinWidth = width;
    this.label = normalizeLegendLabel(options.label);

    const finite = new Float64Array(data.length);
    let n = 0;
    for (let i = 0; i < data.length; i++) {
      const v = data[i] ?? Number.NaN;
      if (Number.isFinite(v)) finite[n++] = v;
    }
    const sorted = finite.subarray(0, n).sort();

    if (n === 0) {
      if (data.length > 0) {
        throw new InvalidParameterError(
          "violinplot data must contain at least one finite value",
          "data",
          data
        );
      }
      this.hasData = false;
      this.q1 = 0;
      this.median = 0;
      this.q3 = 0;
      this.kdeValues = [];
      this.kdePoints = [];
      return;
    }
    this.hasData = true;

    const { q1, median, q3 } = calculateQuartiles(sorted);
    this.q1 = q1;
    this.median = median;
    this.q3 = q3;

    // Evaluation grid: data range plus padding.
    const firstVal = sorted[0] ?? 0;
    const lastVal = sorted[n - 1] ?? 0;
    const dataRange = lastVal - firstVal;
    const bandwidth =
      typeof rule === "number"
        ? rule
        : rule === "scott"
          ? scottBandwidth(sorted)
          : silvermanBandwidth(sorted);
    const padding = dataRange > 0 ? dataRange * 0.1 : 3 * bandwidth;
    const min = firstVal - padding;
    const max = lastVal + padding;
    if (!Number.isFinite(min) || !Number.isFinite(max)) {
      throw new InvalidParameterError(
        "violinplot data span is too large to evaluate a density",
        "data",
        data
      );
    }

    const kdePoints: number[] = [];
    for (let i = 0; i < KDE_POINTS; i++) {
      kdePoints.push(min + (i / (KDE_POINTS - 1)) * (max - min));
    }

    this.kdeValues = kernelDensityEstimation(sorted, kdePoints, bandwidth);
    this.kdePoints = kdePoints;
  }

  getDataRange(): DataRange | null {
    if (!this.hasData) return null;
    if (this.kdePoints.length === 0) return null;

    const minY = this.kdePoints[0] ?? 0;
    const maxY = this.kdePoints[this.kdePoints.length - 1] ?? 0;

    return {
      xmin: this.position - this.violinWidth / 2,
      xmax: this.position + this.violinWidth / 2,
      ymin: minY,
      ymax: maxY,
    };
  }

  drawSVG(ctx: SvgDrawContext): void {
    if (!this.hasData) return;
    const x = this.position;

    // Find max KDE value for scaling
    let maxKDE = 0;
    for (let i = 0; i < this.kdeValues.length; i++) {
      const v = this.kdeValues[i] ?? 0;
      if (v > maxKDE) maxKDE = v;
    }
    if (maxKDE === 0) return;

    // Build violin path
    const pathPoints: string[] = [];

    // Left side of violin (from bottom to top)
    for (let i = 0; i < this.kdePoints.length; i++) {
      const y = this.kdePoints[i] ?? 0;
      const kde = this.kdeValues[i] ?? 0;
      const width = (kde / maxKDE) * this.violinWidth;
      const xLeft = x - width / 2;

      const px = ctx.transform.xToPx(xLeft);
      const py = ctx.transform.yToPx(y);
      pathPoints.push(`${px.toFixed(2)},${py.toFixed(2)}`);
    }

    // Right side of violin (from top to bottom)
    for (let i = this.kdePoints.length - 1; i >= 0; i--) {
      const y = this.kdePoints[i] ?? 0;
      const kde = this.kdeValues[i] ?? 0;
      const width = (kde / maxKDE) * this.violinWidth;
      const xRight = x + width / 2;

      const px = ctx.transform.xToPx(xRight);
      const py = ctx.transform.yToPx(y);
      pathPoints.push(`${px.toFixed(2)},${py.toFixed(2)}`);
    }

    // Draw violin shape
    if (pathPoints.length > 0) {
      ctx.push(
        `<path d="M ${pathPoints.join(" L ")} Z" fill="${escapeXml(this.color)}" stroke="${escapeXml(this.edgecolor)}" stroke-width="1" />`
      );
    }

    // Draw quartile indicators
    const yq1 = ctx.transform.yToPx(this.q1);
    const ymed = ctx.transform.yToPx(this.median);
    const yq3 = ctx.transform.yToPx(this.q3);

    const indicatorWidth = this.violinWidth * 0.8;
    const xLeft = ctx.transform.xToPx(x - indicatorWidth / 2);
    const xRight = ctx.transform.xToPx(x + indicatorWidth / 2);

    // Draw quartile lines
    ctx.push(
      `<line x1="${xLeft.toFixed(2)}" y1="${yq1.toFixed(2)}" x2="${xRight.toFixed(2)}" y2="${yq1.toFixed(2)}" stroke="${escapeXml(this.edgecolor)}" stroke-width="2" />`
    );
    ctx.push(
      `<line x1="${xLeft.toFixed(2)}" y1="${ymed.toFixed(2)}" x2="${xRight.toFixed(2)}" y2="${ymed.toFixed(2)}" stroke="${escapeXml(this.edgecolor)}" stroke-width="3" />`
    );
    ctx.push(
      `<line x1="${xLeft.toFixed(2)}" y1="${yq3.toFixed(2)}" x2="${xRight.toFixed(2)}" y2="${yq3.toFixed(2)}" stroke="${escapeXml(this.edgecolor)}" stroke-width="2" />`
    );
  }

  drawRaster(ctx: RasterDrawContext): void {
    if (!this.hasData) return;
    const rgba = parseHexColorToRGBA(this.color);
    const edge = parseHexColorToRGBA(this.edgecolor);
    const x = this.position;

    // Find max KDE value for scaling
    let maxKDE = 0;
    for (let i = 0; i < this.kdeValues.length; i++) {
      const v = this.kdeValues[i] ?? 0;
      if (v > maxKDE) maxKDE = v;
    }
    if (maxKDE === 0) return;

    // Violin outline: left edge bottom to top, then right edge top to bottom.
    const count = this.kdePoints.length;
    const xs = new Float64Array(2 * count);
    const ys = new Float64Array(2 * count);
    for (let i = 0; i < count; i++) {
      const half = (((this.kdeValues[i] ?? 0) / maxKDE) * this.violinWidth) / 2;
      const py = ctx.transform.yToPx(this.kdePoints[i] ?? 0);
      xs[i] = ctx.transform.xToPx(x - half);
      ys[i] = py;
      xs[2 * count - 1 - i] = ctx.transform.xToPx(x + half);
      ys[2 * count - 1 - i] = py;
    }
    ctx.canvas.fillPolygonRGBA(xs, ys, rgba.r, rgba.g, rgba.b, rgba.a);
    for (let i = 0; i < 2 * count; i++) {
      const j = (i + 1) % (2 * count);
      ctx.canvas.drawLineRGBA(
        Math.round(xs[i] ?? 0),
        Math.round(ys[i] ?? 0),
        Math.round(xs[j] ?? 0),
        Math.round(ys[j] ?? 0),
        edge.r,
        edge.g,
        edge.b,
        edge.a
      );
    }

    // Draw quartile indicators
    const yq1 = Math.round(ctx.transform.yToPx(this.q1));
    const ymed = Math.round(ctx.transform.yToPx(this.median));
    const yq3 = Math.round(ctx.transform.yToPx(this.q3));

    const indicatorWidth = this.violinWidth * 0.8;
    const xLeft = Math.round(ctx.transform.xToPx(x - indicatorWidth / 2));
    const xRight = Math.round(ctx.transform.xToPx(x + indicatorWidth / 2));

    // Draw quartile lines
    ctx.canvas.drawLineRGBA(xLeft, yq1, xRight, yq1, edge.r, edge.g, edge.b, edge.a);
    ctx.canvas.drawLineRGBA(xLeft, ymed, xRight, ymed, edge.r, edge.g, edge.b, edge.a);
    ctx.canvas.drawLineRGBA(xLeft, yq3, xRight, yq3, edge.r, edge.g, edge.b, edge.a);
  }

  getLegendEntries(): readonly LegendEntry[] | null {
    const entry = buildLegendEntry(this.label, {
      color: this.color,
      shape: "box",
    });
    return entry ? [entry] : null;
  }
}
