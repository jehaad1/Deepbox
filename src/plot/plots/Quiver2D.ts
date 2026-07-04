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
 * Quiver plot: draws arrows representing a vector field.
 * @internal
 */
export class Quiver2D implements Drawable {
  readonly kind = "quiver";
  readonly x: Float64Array;
  readonly y: Float64Array;
  readonly u: Float64Array;
  readonly v: Float64Array;
  readonly color: Color;
  readonly linewidth: number;
  readonly scale: number;
  readonly label: string | null;

  constructor(
    x: Float64Array,
    y: Float64Array,
    u: Float64Array,
    v: Float64Array,
    options: PlotOptions & { scale?: number }
  ) {
    const n = x.length;
    if (y.length !== n || u.length !== n || v.length !== n)
      throw new ShapeError("x, y, u, v must all have the same length");
    this.x = x;
    this.y = y;
    this.u = u;
    this.v = v;
    this.color = normalizeColor(options.color, "#1f77b4");
    this.linewidth = options.linewidth ?? 1.5;
    this.scale = options.scale ?? 1;
    this.label = normalizeLegendLabel(options.label);
  }

  getDataRange(): DataRange | null {
    let xmin = Infinity;
    let xmax = -Infinity;
    let ymin = Infinity;
    let ymax = -Infinity;

    for (let i = 0; i < this.x.length; i++) {
      const xi = this.x[i] ?? 0;
      const yi = this.y[i] ?? 0;
      const ui = (this.u[i] ?? 0) * this.scale;
      const vi = (this.v[i] ?? 0) * this.scale;
      if (!isFiniteNumber(xi) || !isFiniteNumber(yi)) continue;
      xmin = Math.min(xmin, xi, xi + ui);
      xmax = Math.max(xmax, xi, xi + ui);
      ymin = Math.min(ymin, yi, yi + vi);
      ymax = Math.max(ymax, yi, yi + vi);
    }

    if (!isFiniteNumber(xmin) || !isFiniteNumber(xmax)) return null;
    return { xmin, xmax, ymin, ymax };
  }

  drawSVG(ctx: SvgDrawContext): void {
    const ec = escapeXml(this.color);
    const headLen = 6;
    const headAngle = Math.PI / 6;

    for (let i = 0; i < this.x.length; i++) {
      const xi = this.x[i] ?? 0;
      const yi = this.y[i] ?? 0;
      const ui = (this.u[i] ?? 0) * this.scale;
      const vi = (this.v[i] ?? 0) * this.scale;
      if (!isFiniteNumber(xi) || !isFiniteNumber(yi)) continue;

      const x0 = ctx.transform.xToPx(xi);
      const y0 = ctx.transform.yToPx(yi);
      const x1 = ctx.transform.xToPx(xi + ui);
      const y1 = ctx.transform.yToPx(yi + vi);

      // Shaft
      ctx.push(
        `<line x1="${x0.toFixed(2)}" y1="${y0.toFixed(2)}" x2="${x1.toFixed(2)}" y2="${y1.toFixed(2)}" stroke="${ec}" stroke-width="${this.linewidth}" />`
      );

      // Arrowhead
      const dx = x1 - x0;
      const dy = y1 - y0;
      const angle = Math.atan2(dy, dx);
      const ax1 = x1 - headLen * Math.cos(angle - headAngle);
      const ay1 = y1 - headLen * Math.sin(angle - headAngle);
      const ax2 = x1 - headLen * Math.cos(angle + headAngle);
      const ay2 = y1 - headLen * Math.sin(angle + headAngle);
      ctx.push(
        `<polygon points="${x1.toFixed(2)},${y1.toFixed(2)} ${ax1.toFixed(2)},${ay1.toFixed(2)} ${ax2.toFixed(2)},${ay2.toFixed(2)}" fill="${ec}" />`
      );
    }
  }

  drawRaster(ctx: RasterDrawContext): void {
    const rgba = parseHexColorToRGBA(this.color);
    for (let i = 0; i < this.x.length; i++) {
      const xi = this.x[i] ?? 0;
      const yi = this.y[i] ?? 0;
      const ui = (this.u[i] ?? 0) * this.scale;
      const vi = (this.v[i] ?? 0) * this.scale;
      if (!isFiniteNumber(xi) || !isFiniteNumber(yi)) continue;

      const x0 = Math.round(ctx.transform.xToPx(xi));
      const y0 = Math.round(ctx.transform.yToPx(yi));
      const x1 = Math.round(ctx.transform.xToPx(xi + ui));
      const y1 = Math.round(ctx.transform.yToPx(yi + vi));
      ctx.canvas.drawLineRGBA(x0, y0, x1, y1, rgba.r, rgba.g, rgba.b, rgba.a);
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
