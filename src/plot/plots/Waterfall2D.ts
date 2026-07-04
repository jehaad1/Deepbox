/**
 * @see {@link https://deepbox.dev/docs/plot-basic | Deepbox documentation}
 */

import { ShapeError } from "../../core";
import type {
  Color,
  DataRange,
  Drawable,
  LegendEntry,
  RasterDrawContext,
  SvgDrawContext,
} from "../types";
import { normalizeColor, parseHexColorToRGBA } from "../utils/colors";
import { escapeXml } from "../utils/xml";

/**
 * Waterfall chart: shows cumulative effect of sequential positive/negative values.
 * @internal
 */
export class Waterfall2D implements Drawable {
  readonly kind = "waterfall";
  readonly categories: string[];
  readonly values: Float64Array;
  readonly positiveColor: Color;
  readonly negativeColor: Color;
  readonly totalColor: Color;
  readonly barWidth: number;

  constructor(
    categories: string[],
    values: Float64Array,
    options: {
      positiveColor?: Color;
      negativeColor?: Color;
      totalColor?: Color;
      barWidth?: number;
    } = {}
  ) {
    if (categories.length !== values.length)
      throw new ShapeError("categories and values must have the same length");
    this.categories = categories;
    this.values = values;
    this.positiveColor = normalizeColor(options.positiveColor, "#2ca02c");
    this.negativeColor = normalizeColor(options.negativeColor, "#d62728");
    this.totalColor = normalizeColor(options.totalColor, "#1f77b4");
    this.barWidth = options.barWidth ?? 0.6;
  }

  getDataRange(): DataRange | null {
    if (this.values.length === 0) return null;
    let cumulative = 0;
    let ymin = 0;
    let ymax = 0;
    for (let i = 0; i < this.values.length; i++) {
      const prev = cumulative;
      cumulative += this.values[i] ?? 0;
      ymin = Math.min(ymin, prev, cumulative);
      ymax = Math.max(ymax, prev, cumulative);
    }
    return {
      xmin: -0.5,
      xmax: this.values.length - 0.5,
      ymin,
      ymax,
    };
  }

  drawSVG(ctx: SvgDrawContext): void {
    const halfW = this.barWidth / 2;
    let cumulative = 0;

    for (let i = 0; i < this.values.length; i++) {
      const val = this.values[i] ?? 0;
      const bottom = cumulative;
      cumulative += val;
      const top = cumulative;

      const isLast = i === this.values.length - 1;
      const color = isLast ? this.totalColor : val >= 0 ? this.positiveColor : this.negativeColor;
      const ec = escapeXml(color);

      const x1 = ctx.transform.xToPx(i - halfW);
      const x2 = ctx.transform.xToPx(i + halfW);
      const y1 = ctx.transform.yToPx(Math.max(bottom, top));
      const y2 = ctx.transform.yToPx(Math.min(bottom, top));

      const w = Math.abs(x2 - x1);
      const h = Math.abs(y2 - y1);
      const rx = Math.min(x1, x2);
      const ry = Math.min(y1, y2);

      ctx.push(
        `<rect x="${rx.toFixed(2)}" y="${ry.toFixed(2)}" width="${w.toFixed(2)}" height="${Math.max(h, 0.5).toFixed(2)}" fill="${ec}" />`
      );

      // Draw connector line to next bar
      if (i < this.values.length - 1) {
        const connY = ctx.transform.yToPx(cumulative);
        const connX1 = ctx.transform.xToPx(i + halfW);
        const connX2 = ctx.transform.xToPx(i + 1 - halfW);
        ctx.push(
          `<line x1="${connX1.toFixed(2)}" y1="${connY.toFixed(2)}" x2="${connX2.toFixed(2)}" y2="${connY.toFixed(2)}" stroke="#999" stroke-width="0.8" stroke-dasharray="3,2" />`
        );
      }
    }
  }

  drawRaster(ctx: RasterDrawContext): void {
    const halfW = this.barWidth / 2;
    let cumulative = 0;

    for (let i = 0; i < this.values.length; i++) {
      const val = this.values[i] ?? 0;
      const bottom = cumulative;
      cumulative += val;
      const top = cumulative;

      const isLast = i === this.values.length - 1;
      const color = isLast ? this.totalColor : val >= 0 ? this.positiveColor : this.negativeColor;
      const rgba = parseHexColorToRGBA(color);

      const x1 = Math.round(ctx.transform.xToPx(i - halfW));
      const x2 = Math.round(ctx.transform.xToPx(i + halfW));
      const y1 = Math.round(ctx.transform.yToPx(Math.max(bottom, top)));
      const y2 = Math.round(ctx.transform.yToPx(Math.min(bottom, top)));

      const rx = Math.min(x1, x2);
      const ry = Math.min(y1, y2);
      const w = Math.abs(x2 - x1);
      const h = Math.max(Math.abs(y2 - y1), 1);

      // Fill rectangle using horizontal lines
      for (let row = ry; row < ry + h; row++) {
        ctx.canvas.drawLineRGBA(rx, row, rx + w, row, rgba.r, rgba.g, rgba.b, rgba.a);
      }
    }
  }

  getLegendEntries(): readonly LegendEntry[] | null {
    return null;
  }
}
