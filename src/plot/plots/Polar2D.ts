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
 * Polar plot: plots data in polar coordinates (theta, r) as Cartesian (x, y).
 * Converts polar (theta, r) to Cartesian and draws as connected line with optional fill.
 * @internal
 */
export class Polar2D implements Drawable {
  readonly kind = "polar";
  readonly theta: Float64Array;
  readonly r: Float64Array;
  readonly color: Color;
  readonly linewidth: number;
  readonly fill: boolean;
  readonly label: string | null;

  constructor(theta: Float64Array, r: Float64Array, options: PlotOptions & { fill?: boolean }) {
    if (theta.length !== r.length) throw new ShapeError("theta and r must have the same length");
    this.theta = theta;
    this.r = r;
    this.color = normalizeColor(options.color, "#1f77b4");
    this.linewidth = options.linewidth ?? 2;
    this.fill = options.fill ?? false;
    this.label = normalizeLegendLabel(options.label);
  }

  getDataRange(): DataRange | null {
    let maxR = 0;
    for (let i = 0; i < this.r.length; i++) {
      const ri = this.r[i] ?? 0;
      if (isFiniteNumber(ri)) maxR = Math.max(maxR, Math.abs(ri));
    }
    if (maxR === 0) maxR = 1;
    const margin = maxR * 0.1;
    return {
      xmin: -(maxR + margin),
      xmax: maxR + margin,
      ymin: -(maxR + margin),
      ymax: maxR + margin,
    };
  }

  drawSVG(ctx: SvgDrawContext): void {
    const ec = escapeXml(this.color);

    // Draw polar grid (circles and radial lines)
    let maxR = 0;
    for (let i = 0; i < this.r.length; i++) {
      const ri = this.r[i] ?? 0;
      if (isFiniteNumber(ri)) maxR = Math.max(maxR, Math.abs(ri));
    }
    if (maxR === 0) maxR = 1;

    // Grid circles
    const nCircles = 4;
    for (let c = 1; c <= nCircles; c++) {
      const gr = (maxR * c) / nCircles;
      const points: string[] = [];
      const steps = 60;
      for (let s = 0; s <= steps; s++) {
        const a = (2 * Math.PI * s) / steps;
        const gx = ctx.transform.xToPx(gr * Math.cos(a));
        const gy = ctx.transform.yToPx(gr * Math.sin(a));
        points.push(`${gx.toFixed(2)},${gy.toFixed(2)}`);
      }
      ctx.push(
        `<polyline points="${points.join(" ")}" fill="none" stroke="#ddd" stroke-width="0.5" />`
      );
    }

    // Radial lines
    for (let a = 0; a < 8; a++) {
      const angle = (a * Math.PI) / 4;
      const x0 = ctx.transform.xToPx(0);
      const y0 = ctx.transform.yToPx(0);
      const x1 = ctx.transform.xToPx(maxR * Math.cos(angle));
      const y1 = ctx.transform.yToPx(maxR * Math.sin(angle));
      ctx.push(
        `<line x1="${x0.toFixed(2)}" y1="${y0.toFixed(2)}" x2="${x1.toFixed(2)}" y2="${y1.toFixed(2)}" stroke="#ddd" stroke-width="0.5" />`
      );
    }

    // Draw data
    if (this.theta.length === 0) return;
    const points: string[] = [];
    for (let i = 0; i < this.theta.length; i++) {
      const t = this.theta[i] ?? 0;
      const ri = this.r[i] ?? 0;
      if (!isFiniteNumber(t) || !isFiniteNumber(ri)) continue;
      const gx = ctx.transform.xToPx(ri * Math.cos(t));
      const gy = ctx.transform.yToPx(ri * Math.sin(t));
      points.push(`${gx.toFixed(2)},${gy.toFixed(2)}`);
    }

    if (this.fill) {
      ctx.push(
        `<polygon points="${points.join(" ")}" fill="${ec}" fill-opacity="0.2" stroke="${ec}" stroke-width="${this.linewidth}" />`
      );
    } else {
      ctx.push(
        `<polyline points="${points.join(" ")}" fill="none" stroke="${ec}" stroke-width="${this.linewidth}" />`
      );
    }
  }

  drawRaster(ctx: RasterDrawContext): void {
    const rgba = parseHexColorToRGBA(this.color);
    for (let i = 0; i < this.theta.length - 1; i++) {
      const t0 = this.theta[i] ?? 0;
      const r0 = this.r[i] ?? 0;
      const t1 = this.theta[i + 1] ?? 0;
      const r1 = this.r[i + 1] ?? 0;
      if (!isFiniteNumber(t0) || !isFiniteNumber(r0)) continue;
      if (!isFiniteNumber(t1) || !isFiniteNumber(r1)) continue;

      const x0 = Math.round(ctx.transform.xToPx(r0 * Math.cos(t0)));
      const y0 = Math.round(ctx.transform.yToPx(r0 * Math.sin(t0)));
      const x1 = Math.round(ctx.transform.xToPx(r1 * Math.cos(t1)));
      const y1 = Math.round(ctx.transform.yToPx(r1 * Math.sin(t1)));
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
