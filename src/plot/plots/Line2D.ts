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
 * @internal
 */
export class Line2D implements Drawable {
  readonly kind = "line";
  readonly x: Float64Array;
  readonly y: Float64Array;
  readonly color: Color;
  readonly linewidth: number;
  readonly label: string | null;

  constructor(x: Float64Array, y: Float64Array, options: PlotOptions) {
    if (x.length !== y.length) throw new ShapeError("x and y must have the same length");
    this.x = x;
    this.y = y;
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
    this.label = normalizeLegendLabel(options.label);
  }

  getDataRange(): DataRange | null {
    let xmin = Number.POSITIVE_INFINITY;
    let xmax = Number.NEGATIVE_INFINITY;
    let ymin = Number.POSITIVE_INFINITY;
    let ymax = Number.NEGATIVE_INFINITY;

    for (let i = 0; i < this.x.length; i++) {
      const xi = this.x[i] ?? 0;
      const yi = this.y[i] ?? 0;
      if (!isFiniteNumber(xi) || !isFiniteNumber(yi)) continue;
      xmin = Math.min(xmin, xi);
      xmax = Math.max(xmax, xi);
      ymin = Math.min(ymin, yi);
      ymax = Math.max(ymax, yi);
    }

    if (
      !Number.isFinite(xmin) ||
      !Number.isFinite(xmax) ||
      !Number.isFinite(ymin) ||
      !Number.isFinite(ymax)
    ) {
      return null;
    }

    return { xmin, xmax, ymin, ymax };
  }

  drawSVG(ctx: SvgDrawContext): void {
    const pts: string[] = [];
    for (let i = 0; i < this.x.length; i++) {
      const xi = this.x[i] ?? 0;
      const yi = this.y[i] ?? 0;
      if (!isFiniteNumber(xi) || !isFiniteNumber(yi)) continue;
      pts.push(`${ctx.transform.xToPx(xi).toFixed(2)},${ctx.transform.yToPx(yi).toFixed(2)}`);
    }
    ctx.push(
      `<polyline fill="none" stroke="${escapeXml(this.color)}" stroke-width="${this.linewidth}" points="${pts.join(" ")}" />`
    );
  }

  drawRaster(ctx: RasterDrawContext): void {
    const rgba = parseHexColorToRGBA(this.color);
    for (let i = 1; i < this.x.length; i++) {
      const x0 = this.x[i - 1] ?? 0;
      const y0 = this.y[i - 1] ?? 0;
      const x1 = this.x[i] ?? 0;
      const y1 = this.y[i] ?? 0;
      if (!isFiniteNumber(x0) || !isFiniteNumber(y0) || !isFiniteNumber(x1) || !isFiniteNumber(y1))
        continue;
      ctx.canvas.drawLineRGBA(
        Math.round(ctx.transform.xToPx(x0)),
        Math.round(ctx.transform.yToPx(y0)),
        Math.round(ctx.transform.xToPx(x1)),
        Math.round(ctx.transform.yToPx(y1)),
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
