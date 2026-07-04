/**
 * @see {@link https://deepbox.dev/docs/plot-basic | Deepbox documentation}
 */

import { ShapeError } from "../../core";
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
    this.linewidth = options.linewidth ?? 1.5;
    this.markerSize = options.size ?? 4;
    this.baseline = options.baseline ?? 0;
    this.label = normalizeLegendLabel(options.label);
  }

  getDataRange(): DataRange | null {
    let xmin = Infinity;
    let xmax = -Infinity;
    let ymin = Math.min(this.baseline, Infinity);
    let ymax = Math.max(this.baseline, -Infinity);

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

    // Draw baseline
    const x0 = this.x.length > 0 ? ctx.transform.xToPx(this.x[0] ?? 0) : 0;
    const xn = this.x.length > 0 ? ctx.transform.xToPx(this.x[this.x.length - 1] ?? 0) : 0;
    ctx.push(
      `<line x1="${x0.toFixed(2)}" y1="${basePy.toFixed(2)}" x2="${xn.toFixed(2)}" y2="${basePy.toFixed(2)}" stroke="${ec}" stroke-width="0.8" stroke-dasharray="4,2" />`
    );

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
