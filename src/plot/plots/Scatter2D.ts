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
export class Scatter2D implements Drawable {
  readonly kind = "scatter";
  readonly x: Float64Array;
  readonly y: Float64Array;
  readonly color: Color;
  readonly size: number;
  readonly label: string | null;

  constructor(x: Float64Array, y: Float64Array, options: PlotOptions) {
    if (x.length !== y.length) throw new ShapeError("x and y must have the same length");
    this.x = x;
    this.y = y;
    this.color = normalizeColor(options.color, "#ff7f0e");
    const size = options.size ?? 5;
    if (!Number.isFinite(size) || size <= 0) {
      throw new InvalidParameterError(
        `size must be a positive number; received ${size}`,
        "size",
        size
      );
    }
    this.size = size;
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
    for (let i = 0; i < this.x.length; i++) {
      const xi = this.x[i] ?? 0;
      const yi = this.y[i] ?? 0;
      if (!isFiniteNumber(xi) || !isFiniteNumber(yi)) continue;
      const px = ctx.transform.xToPx(xi);
      const py = ctx.transform.yToPx(yi);
      ctx.push(
        `<circle cx="${px.toFixed(2)}" cy="${py.toFixed(2)}" r="${this.size}" fill="${escapeXml(this.color)}" />`
      );
    }
  }

  drawRaster(ctx: RasterDrawContext): void {
    const rgba = parseHexColorToRGBA(this.color);
    for (let i = 0; i < this.x.length; i++) {
      const xi = this.x[i] ?? 0;
      const yi = this.y[i] ?? 0;
      if (!isFiniteNumber(xi) || !isFiniteNumber(yi)) continue;
      const px = Math.round(ctx.transform.xToPx(xi));
      const py = Math.round(ctx.transform.yToPx(yi));
      ctx.canvas.drawCircleRGBA(px, py, this.size, rgba.r, rgba.g, rgba.b, rgba.a);
    }
  }

  getLegendEntries(): readonly LegendEntry[] | null {
    const entry = buildLegendEntry(this.label, {
      color: this.color,
      shape: "marker",
      markerSize: this.size,
    });
    return entry ? [entry] : null;
  }
}
