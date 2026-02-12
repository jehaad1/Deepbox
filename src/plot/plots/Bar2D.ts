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
 * @internal
 */
export class Bar2D implements Drawable {
  readonly kind = "bar";
  readonly x: Float64Array;
  readonly height: Float64Array;
  readonly color: Color;
  readonly edgecolor: Color;
  readonly barWidth: number;
  readonly label: string | null;

  constructor(x: Float64Array, height: Float64Array, options: PlotOptions) {
    if (x.length !== height.length) throw new ShapeError("x and height must have the same length");
    this.x = x;
    this.height = height;
    this.color = normalizeColor(options.color, "#2ca02c");
    this.edgecolor = normalizeColor(options.edgecolor, "#000000");
    this.barWidth = 0.8;
    this.label = normalizeLegendLabel(options.label);
  }

  getDataRange(): DataRange | null {
    let xmin = Number.POSITIVE_INFINITY;
    let xmax = Number.NEGATIVE_INFINITY;
    let ymin = 0;
    let ymax = 0;
    let sawValue = false;

    for (let i = 0; i < this.x.length; i++) {
      const xi = this.x[i] ?? 0;
      const hi = this.height[i] ?? 0;
      if (!isFiniteNumber(xi) || !isFiniteNumber(hi)) continue;
      sawValue = true;
      xmin = Math.min(xmin, xi - this.barWidth / 2);
      xmax = Math.max(xmax, xi + this.barWidth / 2);
      ymin = Math.min(ymin, hi);
      ymax = Math.max(ymax, hi);
    }

    if (!sawValue || !Number.isFinite(xmin) || !Number.isFinite(xmax)) {
      return null;
    }

    return { xmin, xmax, ymin, ymax };
  }

  drawSVG(ctx: SvgDrawContext): void {
    for (let i = 0; i < this.x.length; i++) {
      const xi = this.x[i] ?? 0;
      const hi = this.height[i] ?? 0;
      if (!isFiniteNumber(xi) || !isFiniteNumber(hi)) continue;
      const x0 = ctx.transform.xToPx(xi - this.barWidth / 2);
      const x1 = ctx.transform.xToPx(xi + this.barWidth / 2);
      const y0 = ctx.transform.yToPx(0);
      const y1 = ctx.transform.yToPx(hi);
      const w = Math.abs(x1 - x0);
      const h = Math.abs(y1 - y0);
      const rx = Math.min(x0, x1);
      const ry = Math.min(y0, y1);
      ctx.push(
        `<rect x="${rx.toFixed(2)}" y="${ry.toFixed(2)}" width="${w.toFixed(2)}" height="${h.toFixed(2)}" fill="${escapeXml(this.color)}" stroke="${escapeXml(this.edgecolor)}" />`
      );
    }
  }

  drawRaster(ctx: RasterDrawContext): void {
    const rgba = parseHexColorToRGBA(this.color);
    const edge = parseHexColorToRGBA(this.edgecolor);
    for (let i = 0; i < this.x.length; i++) {
      const xi = this.x[i] ?? 0;
      const hi = this.height[i] ?? 0;
      if (!isFiniteNumber(xi) || !isFiniteNumber(hi)) continue;
      const x0 = Math.round(ctx.transform.xToPx(xi - this.barWidth / 2));
      const x1 = Math.round(ctx.transform.xToPx(xi + this.barWidth / 2));
      const y0 = Math.round(ctx.transform.yToPx(0));
      const y1 = Math.round(ctx.transform.yToPx(hi));
      const w = Math.abs(x1 - x0);
      const h = Math.abs(y1 - y0);
      const rx = Math.min(x0, x1);
      const ry = Math.min(y0, y1);

      // Fill
      ctx.canvas.fillRectRGBA(rx, ry, w, h, rgba.r, rgba.g, rgba.b, rgba.a);

      // Stroke border
      const r_e = edge.r;
      const g_e = edge.g;
      const b_e = edge.b;
      const a_e = edge.a;

      ctx.canvas.drawLineRGBA(rx, ry, rx + w, ry, r_e, g_e, b_e, a_e);
      ctx.canvas.drawLineRGBA(rx + w, ry, rx + w, ry + h, r_e, g_e, b_e, a_e);
      ctx.canvas.drawLineRGBA(rx + w, ry + h, rx, ry + h, r_e, g_e, b_e, a_e);
      ctx.canvas.drawLineRGBA(rx, ry + h, rx, ry, r_e, g_e, b_e, a_e);
    }
  }

  getLegendEntries(): readonly LegendEntry[] | null {
    const entry = buildLegendEntry(this.label, { color: this.color, shape: "box" });
    return entry ? [entry] : null;
  }
}
