/**
 * @see {@link https://deepbox.dev/docs/plot-basic | Deepbox documentation}
 */

import { InvalidParameterError, ShapeError } from "../../core";
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
import { isFiniteNumber } from "../utils/validation";
import { escapeXml } from "../utils/xml";

/**
 * Stem plot: draws vertical lines from a baseline to data points with markers.
 * A dashed baseline spans the smallest to the largest finite x. Points with a
 * non-finite x or y are skipped.
 * @internal
 */
export class Stem2D implements Drawable {
  readonly kind = "stem";
  readonly x: Float64Array;
  readonly y: Float64Array;
  readonly color: Color;
  readonly linewidth: number;
  readonly markerSize: number;
  readonly baseline: number;
  readonly label: string | null;

  constructor(x: Float64Array, y: Float64Array, options: PlotOptions & { baseline?: number }) {
    if (x.length !== y.length) throw new ShapeError("x and y must have the same length");
    this.x = x;
    this.y = y;
    this.color = normalizeColor(options.color, "#1f77b4");
    const lw = options.linewidth ?? 1.5;
    if (!Number.isFinite(lw) || lw <= 0) {
      throw new InvalidParameterError(
        `linewidth must be a positive number; received ${lw}`,
        "linewidth",
        lw
      );
    }
    this.linewidth = lw;
    const size = options.size ?? 4;
    if (!Number.isFinite(size) || size <= 0) {
      throw new InvalidParameterError(
        `size must be a positive number; received ${size}`,
        "size",
        size
      );
    }
    this.markerSize = size;
    const baseline = options.baseline ?? 0;
    if (!Number.isFinite(baseline)) {
      throw new InvalidParameterError(
        `baseline must be finite; received ${baseline}`,
        "baseline",
        baseline
      );
    }
    this.baseline = baseline;
    this.label = normalizeLegendLabel(options.label);
  }

  /** Smallest and largest x among points that are drawn, or null if there are none. */
  private finiteXSpan(): readonly [number, number] | null {
    let lo = Number.POSITIVE_INFINITY;
    let hi = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < this.x.length; i++) {
      const xi = this.x[i] ?? 0;
      const yi = this.y[i] ?? 0;
      if (!isFiniteNumber(xi) || !isFiniteNumber(yi)) continue;
      if (xi < lo) lo = xi;
      if (xi > hi) hi = xi;
    }
    return lo <= hi ? [lo, hi] : null;
  }

  getDataRange(): DataRange | null {
    let xmin = Infinity;
    let xmax = -Infinity;
    let ymin = this.baseline;
    let ymax = this.baseline;

    for (let i = 0; i < this.x.length; i++) {
      const xi = this.x[i] ?? 0;
      const yi = this.y[i] ?? 0;
      if (!isFiniteNumber(xi) || !isFiniteNumber(yi)) continue;
      xmin = Math.min(xmin, xi);
      xmax = Math.max(xmax, xi);
      ymin = Math.min(ymin, yi);
      ymax = Math.max(ymax, yi);
    }

    if (!isFiniteNumber(xmin) || !isFiniteNumber(xmax)) return null;
    return { xmin, xmax, ymin, ymax };
  }

  drawSVG(ctx: SvgDrawContext): void {
    const basePy = ctx.transform.yToPx(this.baseline);
    const ec = escapeXml(this.color);

    // Draw baseline across the finite x extent
    const span = this.finiteXSpan();
    if (span) {
      const x0 = ctx.transform.xToPx(span[0]);
      const xn = ctx.transform.xToPx(span[1]);
      ctx.push(
        `<line x1="${x0.toFixed(2)}" y1="${basePy.toFixed(2)}" x2="${xn.toFixed(2)}" y2="${basePy.toFixed(2)}" stroke="${ec}" stroke-width="0.8" stroke-dasharray="4,2" />`
      );
    }

    // Draw stems and markers
    for (let i = 0; i < this.x.length; i++) {
      const xi = this.x[i] ?? 0;
      const yi = this.y[i] ?? 0;
      if (!isFiniteNumber(xi) || !isFiniteNumber(yi)) continue;
      const px = ctx.transform.xToPx(xi);
      const py = ctx.transform.yToPx(yi);
      ctx.push(
        `<line x1="${px.toFixed(2)}" y1="${basePy.toFixed(2)}" x2="${px.toFixed(2)}" y2="${py.toFixed(2)}" stroke="${ec}" stroke-width="${this.linewidth}" />`
      );
      ctx.push(
        `<circle cx="${px.toFixed(2)}" cy="${py.toFixed(2)}" r="${this.markerSize}" fill="${ec}" />`
      );
    }
  }

  drawRaster(ctx: RasterDrawContext): void {
    const rgba = parseHexColorToRGBA(this.color);
    const basePy = Math.round(ctx.transform.yToPx(this.baseline));
    const span = this.finiteXSpan();
    if (span) {
      const x0 = Math.round(ctx.transform.xToPx(span[0]));
      const xn = Math.round(ctx.transform.xToPx(span[1]));
      // Dashed like the SVG baseline: 4 px on, 2 px off.
      const left = Math.min(x0, xn);
      const right = Math.max(x0, xn);
      for (let xs = left; xs <= right; xs += 6) {
        const xe = Math.min(xs + 4, right);
        ctx.canvas.drawLineRGBA(xs, basePy, xe, basePy, rgba.r, rgba.g, rgba.b, rgba.a);
      }
    }
    for (let i = 0; i < this.x.length; i++) {
      const xi = this.x[i] ?? 0;
      const yi = this.y[i] ?? 0;
      if (!isFiniteNumber(xi) || !isFiniteNumber(yi)) continue;
      const px = Math.round(ctx.transform.xToPx(xi));
      const py = Math.round(ctx.transform.yToPx(yi));
      ctx.canvas.drawLineRGBA(px, basePy, px, py, rgba.r, rgba.g, rgba.b, rgba.a);
      ctx.canvas.drawCircleRGBA(px, py, this.markerSize, rgba.r, rgba.g, rgba.b, rgba.a);
    }
  }

  getLegendEntries(): readonly LegendEntry[] | null {
    const entry = buildLegendEntry(this.label, {
      color: this.color,
      shape: "line",
      lineWidth: this.linewidth,
    });
    return entry ? [entry] : null;
  }
}
