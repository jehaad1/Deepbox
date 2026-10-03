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
 * Quiver plot: draws arrows representing a vector field. The arrow at `(x, y)`
 * runs to `(x + u * scale, y + v * scale)` in data coordinates. Entries with a
 * non-finite x, y, u or v are skipped, and zero-length arrows draw nothing.
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
    const lw = options.linewidth ?? 1.5;
    if (!Number.isFinite(lw) || lw <= 0) {
      throw new InvalidParameterError(
        `linewidth must be a positive number; received ${lw}`,
        "linewidth",
        lw
      );
    }
    this.linewidth = lw;
    const scale = options.scale ?? 1;
    if (!Number.isFinite(scale)) {
      throw new InvalidParameterError(`scale must be finite; received ${scale}`, "scale", scale);
    }
    this.scale = scale;
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
      if (!isFiniteNumber(ui) || !isFiniteNumber(vi)) continue;
      xmin = Math.min(xmin, xi, xi + ui);
      xmax = Math.max(xmax, xi, xi + ui);
      ymin = Math.min(ymin, yi, yi + vi);
      ymax = Math.max(ymax, yi, yi + vi);
    }

    if (!isFiniteNumber(xmin) || !isFiniteNumber(xmax)) return null;
    if (!isFiniteNumber(ymin) || !isFiniteNumber(ymax)) return null;
    return { xmin, xmax, ymin, ymax };
  }

  /**
   * Arrowhead corner points for a shaft ending at (x1, y1) that started at (x0, y0),
   * in pixels. The head is 6 px long (at most half the shaft) with a 30 degree
   * half-angle. Returns null for a zero-length shaft.
   */
  private static arrowhead(
    x0: number,
    y0: number,
    x1: number,
    y1: number
  ): readonly [number, number, number, number] | null {
    const dx = x1 - x0;
    const dy = y1 - y0;
    const length = Math.hypot(dx, dy);
    if (!(length > 0)) return null;
    const headLen = Math.min(6, length / 2);
    const headAngle = Math.PI / 6;
    const angle = Math.atan2(dy, dx);
    return [
      x1 - headLen * Math.cos(angle - headAngle),
      y1 - headLen * Math.sin(angle - headAngle),
      x1 - headLen * Math.cos(angle + headAngle),
      y1 - headLen * Math.sin(angle + headAngle),
    ];
  }

  drawSVG(ctx: SvgDrawContext): void {
    const ec = escapeXml(this.color);

    for (let i = 0; i < this.x.length; i++) {
      const xi = this.x[i] ?? 0;
      const yi = this.y[i] ?? 0;
      const ui = (this.u[i] ?? 0) * this.scale;
      const vi = (this.v[i] ?? 0) * this.scale;
      if (!isFiniteNumber(xi) || !isFiniteNumber(yi)) continue;
      if (!isFiniteNumber(ui) || !isFiniteNumber(vi)) continue;

      const x0 = ctx.transform.xToPx(xi);
      const y0 = ctx.transform.yToPx(yi);
      const x1 = ctx.transform.xToPx(xi + ui);
      const y1 = ctx.transform.yToPx(yi + vi);

      const head = Quiver2D.arrowhead(x0, y0, x1, y1);
      if (head === null) continue;

      // Shaft
      ctx.push(
        `<line x1="${x0.toFixed(2)}" y1="${y0.toFixed(2)}" x2="${x1.toFixed(2)}" y2="${y1.toFixed(2)}" stroke="${ec}" stroke-width="${this.linewidth}" />`
      );

      // Arrowhead
      const [ax1, ay1, ax2, ay2] = head;
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
      if (!isFiniteNumber(ui) || !isFiniteNumber(vi)) continue;

      const px0 = ctx.transform.xToPx(xi);
      const py0 = ctx.transform.yToPx(yi);
      const px1 = ctx.transform.xToPx(xi + ui);
      const py1 = ctx.transform.yToPx(yi + vi);
      const head = Quiver2D.arrowhead(px0, py0, px1, py1);
      if (head === null) continue;

      const x0 = Math.round(px0);
      const y0 = Math.round(py0);
      const x1 = Math.round(px1);
      const y1 = Math.round(py1);
      ctx.canvas.drawLineRGBA(x0, y0, x1, y1, rgba.r, rgba.g, rgba.b, rgba.a);
      ctx.canvas.fillTriangleRGBA(
        x1,
        y1,
        Math.round(head[0]),
        Math.round(head[1]),
        Math.round(head[2]),
        Math.round(head[3]),
        rgba.r,
        rgba.g,
        rgba.b,
        rgba.a
      );
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
