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
 * Polar plot: plots data in polar coordinates (theta, r) as Cartesian (x, y).
 * `theta` is in radians, measured counter-clockwise from the positive x axis.
 * Draws a polar grid (four circles and eight radial lines) and the data as a
 * connected line with optional fill. Samples with a non-finite theta or r are
 * not drawn and break the line.
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
    const lw = options.linewidth ?? 2;
    if (!Number.isFinite(lw) || lw <= 0) {
      throw new InvalidParameterError(
        `linewidth must be a positive number; received ${lw}`,
        "linewidth",
        lw
      );
    }
    this.linewidth = lw;
    this.fill = options.fill ?? false;
    this.label = normalizeLegendLabel(options.label);
  }

  /** Largest finite |r|, or 1 when there is nothing to scale by. */
  private maxRadius(): number {
    let maxR = 0;
    for (let i = 0; i < this.r.length; i++) {
      const ri = this.r[i] ?? 0;
      if (isFiniteNumber(ri)) maxR = Math.max(maxR, Math.abs(ri));
    }
    return maxR === 0 ? 1 : maxR;
  }

  getDataRange(): DataRange | null {
    const maxR = this.maxRadius();
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
    const maxR = this.maxRadius();

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
    let points: string[] = [];
    const flush = (): void => {
      if (points.length === 0) return;
      if (this.fill) {
        ctx.push(
          `<polygon points="${points.join(" ")}" fill="${ec}" fill-opacity="0.2" stroke="${ec}" stroke-width="${this.linewidth}" />`
        );
      } else {
        ctx.push(
          `<polyline points="${points.join(" ")}" fill="none" stroke="${ec}" stroke-width="${this.linewidth}" />`
        );
      }
      points = [];
    };
    for (let i = 0; i < this.theta.length; i++) {
      const t = this.theta[i] ?? 0;
      const ri = this.r[i] ?? 0;
      if (!isFiniteNumber(t) || !isFiniteNumber(ri)) {
        flush();
        continue;
      }
      const gx = ctx.transform.xToPx(ri * Math.cos(t));
      const gy = ctx.transform.yToPx(ri * Math.sin(t));
      points.push(`${gx.toFixed(2)},${gy.toFixed(2)}`);
    }
    flush();
  }

  drawRaster(ctx: RasterDrawContext): void {
    const rgba = parseHexColorToRGBA(this.color);
    const maxR = this.maxRadius();

    // Polar grid, same layout as the SVG output.
    const nCircles = 4;
    const steps = 60;
    for (let c = 1; c <= nCircles; c++) {
      const gr = (maxR * c) / nCircles;
      let px = Math.round(ctx.transform.xToPx(gr));
      let py = Math.round(ctx.transform.yToPx(0));
      for (let s = 1; s <= steps; s++) {
        const a = (2 * Math.PI * s) / steps;
        const nx = Math.round(ctx.transform.xToPx(gr * Math.cos(a)));
        const ny = Math.round(ctx.transform.yToPx(gr * Math.sin(a)));
        ctx.canvas.drawLineRGBA(px, py, nx, ny, 221, 221, 221, 255);
        px = nx;
        py = ny;
      }
    }
    const ox = Math.round(ctx.transform.xToPx(0));
    const oy = Math.round(ctx.transform.yToPx(0));
    for (let a = 0; a < 8; a++) {
      const angle = (a * Math.PI) / 4;
      const x1 = Math.round(ctx.transform.xToPx(maxR * Math.cos(angle)));
      const y1 = Math.round(ctx.transform.yToPx(maxR * Math.sin(angle)));
      ctx.canvas.drawLineRGBA(ox, oy, x1, y1, 221, 221, 221, 255);
    }

    // Data: one run of consecutive finite samples at a time.
    let xs: number[] = [];
    let ys: number[] = [];
    const flush = (): void => {
      if (this.fill && xs.length >= 3) {
        ctx.canvas.fillPolygonRGBA(xs, ys, rgba.r, rgba.g, rgba.b, Math.round(rgba.a * 0.2));
      }
      const last = xs.length - 1;
      for (let k = 0; k < last; k++) {
        ctx.canvas.drawLineRGBA(
          xs[k] ?? 0,
          ys[k] ?? 0,
          xs[k + 1] ?? 0,
          ys[k + 1] ?? 0,
          rgba.r,
          rgba.g,
          rgba.b,
          rgba.a
        );
      }
      if (this.fill && last >= 1) {
        ctx.canvas.drawLineRGBA(
          xs[last] ?? 0,
          ys[last] ?? 0,
          xs[0] ?? 0,
          ys[0] ?? 0,
          rgba.r,
          rgba.g,
          rgba.b,
          rgba.a
        );
      }
      xs = [];
      ys = [];
    };
    for (let i = 0; i < this.theta.length; i++) {
      const t = this.theta[i] ?? 0;
      const ri = this.r[i] ?? 0;
      if (!isFiniteNumber(t) || !isFiniteNumber(ri)) {
        flush();
        continue;
      }
      xs.push(Math.round(ctx.transform.xToPx(ri * Math.cos(t))));
      ys.push(Math.round(ctx.transform.yToPx(ri * Math.sin(t))));
    }
    flush();
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
