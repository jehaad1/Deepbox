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
export class HorizontalBar2D implements Drawable {
  readonly kind = "barh";
  readonly y: Float64Array;
  readonly width: Float64Array;
  readonly color: Color;
  readonly edgecolor: Color;
  readonly barHeight: number;
  readonly label: string | null;

  constructor(y: Float64Array, width: Float64Array, options: PlotOptions) {
    if (y.length !== width.length) throw new ShapeError("y and width must have the same length");
    this.y = y;
    this.width = width;
    this.color = normalizeColor(options.color, "#9467bd");
    this.edgecolor = normalizeColor(options.edgecolor, "#000000");
    this.barHeight = 0.8;
    this.label = normalizeLegendLabel(options.label);
  }

  getDataRange(): DataRange | null {
    let ymin = Number.POSITIVE_INFINITY;
    let ymax = Number.NEGATIVE_INFINITY;
    let xmin = 0;
    let xmax = 0;
    let sawValue = false;

    for (let i = 0; i < this.y.length; i++) {
      const yi = this.y[i] ?? 0;
      const wi = this.width[i] ?? 0;
      if (!isFiniteNumber(yi) || !isFiniteNumber(wi)) continue;
      sawValue = true;
      ymin = Math.min(ymin, yi - this.barHeight / 2);
      ymax = Math.max(ymax, yi + this.barHeight / 2);
      xmin = Math.min(xmin, wi);
      xmax = Math.max(xmax, wi);
    }

    if (!sawValue || !Number.isFinite(ymin) || !Number.isFinite(ymax)) {
      return null;
    }

    return { xmin, xmax, ymin, ymax };
  }

  drawSVG(ctx: SvgDrawContext): void {
    for (let i = 0; i < this.y.length; i++) {
      const yi = this.y[i] ?? 0;
      const wi = this.width[i] ?? 0;
      if (!isFiniteNumber(yi) || !isFiniteNumber(wi)) continue;
      const x0 = ctx.transform.xToPx(0);
      const x1 = ctx.transform.xToPx(wi);
      const y0 = ctx.transform.yToPx(yi - this.barHeight / 2);
      const y1 = ctx.transform.yToPx(yi + this.barHeight / 2);
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
    for (let i = 0; i < this.y.length; i++) {
      const yi = this.y[i] ?? 0;
      const wi = this.width[i] ?? 0;
      if (!isFiniteNumber(yi) || !isFiniteNumber(wi)) continue;
      const x0 = Math.round(ctx.transform.xToPx(0));
      const x1 = Math.round(ctx.transform.xToPx(wi));
      const y0 = Math.round(ctx.transform.yToPx(yi - this.barHeight / 2));
      const y1 = Math.round(ctx.transform.yToPx(yi + this.barHeight / 2));
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
