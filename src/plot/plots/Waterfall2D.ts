/**
 * Waterfall chart drawable.
 *
 * @module plot/plots/Waterfall2D
 * @see {@link https://deepbox.dev/docs/plot-basic | Deepbox documentation}
 */

import { InvalidParameterError, ShapeError } from "../../core";
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

type Bar = {
  readonly index: number;
  /** Cumulative value before this step. */
  readonly start: number;
  /** Cumulative value after this step. */
  readonly end: number;
  readonly color: Color;
};

/**
 * Waterfall chart: shows cumulative effect of sequential positive/negative values.
 *
 * Step `i` is a bar spanning the running total before and after `values[i]`, centered at x = i.
 * Positive steps use `positiveColor` and negative steps `negativeColor`. The last bar always uses
 * `totalColor`; it is still the last step (it starts at the previous running total), not a
 * separate bar from zero. A dashed connector joins the end of each bar to the start of the next.
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

  /**
   * @param categories - One label per step
   * @param values - Step values, same length as `categories`; every value must be finite
   * @param options - Bar colors and `barWidth` (in x units, default 0.6, must be positive)
   * @throws {ShapeError} If `categories` and `values` differ in length.
   * @throws {InvalidParameterError} If a value is not finite or `barWidth` is not a positive
   *   finite number.
   */
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
    if (categories.length !== values.length) {
      throw new ShapeError(
        `categories and values must have the same length; received ${categories.length} and ${values.length}`
      );
    }
    for (let i = 0; i < values.length; i++) {
      const v = values[i] ?? Number.NaN;
      if (!Number.isFinite(v)) {
        throw new InvalidParameterError(
          `waterfall values must be finite; received ${v} at index ${i}`,
          "values",
          v
        );
      }
    }
    const barWidth = options.barWidth ?? 0.6;
    if (!Number.isFinite(barWidth) || barWidth <= 0) {
      throw new InvalidParameterError(
        `barWidth must be a positive finite number; received ${barWidth}`,
        "barWidth",
        barWidth
      );
    }
    this.categories = categories.slice();
    this.values = values.slice();
    this.positiveColor = normalizeColor(options.positiveColor, "#2ca02c");
    this.negativeColor = normalizeColor(options.negativeColor, "#d62728");
    this.totalColor = normalizeColor(options.totalColor, "#1f77b4");
    this.barWidth = barWidth;
  }

  private bars(): Bar[] {
    const bars: Bar[] = [];
    const last = this.values.length - 1;
    let cumulative = 0;
    for (let i = 0; i <= last; i++) {
      const val = this.values[i] ?? 0;
      const start = cumulative;
      cumulative += val;
      const color =
        i === last ? this.totalColor : val >= 0 ? this.positiveColor : this.negativeColor;
      bars.push({ index: i, start, end: cumulative, color });
    }
    return bars;
  }

  getDataRange(): DataRange | null {
    if (this.values.length === 0) return null;
    let ymin = 0;
    let ymax = 0;
    for (const bar of this.bars()) {
      ymin = Math.min(ymin, bar.start, bar.end);
      ymax = Math.max(ymax, bar.start, bar.end);
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
    const bars = this.bars();

    for (const bar of bars) {
      const i = bar.index;
      const x1 = ctx.transform.xToPx(i - halfW);
      const x2 = ctx.transform.xToPx(i + halfW);
      const y1 = ctx.transform.yToPx(Math.max(bar.start, bar.end));
      const y2 = ctx.transform.yToPx(Math.min(bar.start, bar.end));

      const w = Math.abs(x2 - x1);
      const h = Math.abs(y2 - y1);
      const rx = Math.min(x1, x2);
      const ry = Math.min(y1, y2);

      ctx.push(
        `<rect x="${rx.toFixed(2)}" y="${ry.toFixed(2)}" width="${w.toFixed(2)}" height="${Math.max(h, 0.5).toFixed(2)}" fill="${escapeXml(bar.color)}" />`
      );

      // Draw connector line to next bar
      if (i < bars.length - 1) {
        const connY = ctx.transform.yToPx(bar.end);
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
    const bars = this.bars();
    const connector = parseHexColorToRGBA("#999999");

    for (const bar of bars) {
      const i = bar.index;
      const rgba = parseHexColorToRGBA(bar.color);

      const x1 = Math.round(ctx.transform.xToPx(i - halfW));
      const x2 = Math.round(ctx.transform.xToPx(i + halfW));
      const y1 = Math.round(ctx.transform.yToPx(Math.max(bar.start, bar.end)));
      const y2 = Math.round(ctx.transform.yToPx(Math.min(bar.start, bar.end)));

      const rx = Math.min(x1, x2);
      const ry = Math.min(y1, y2);
      const w = Math.abs(x2 - x1);
      const h = Math.max(Math.abs(y2 - y1), 1);

      // Pixel-inclusive on the x edges, matching the line-based fill used before.
      ctx.canvas.fillRectRGBA(rx, ry, w + 1, h, rgba.r, rgba.g, rgba.b, rgba.a);

      if (i < bars.length - 1) {
        const connY = Math.round(ctx.transform.yToPx(bar.end));
        const connX1 = Math.round(ctx.transform.xToPx(i + halfW));
        const connX2 = Math.round(ctx.transform.xToPx(i + 1 - halfW));
        // Dashed: 3 px on, 2 px off, like the SVG output.
        for (let x = Math.min(connX1, connX2); x < Math.max(connX1, connX2); x += 5) {
          const xEnd = Math.min(x + 2, Math.max(connX1, connX2));
          ctx.canvas.drawLineRGBA(
            x,
            connY,
            xEnd,
            connY,
            connector.r,
            connector.g,
            connector.b,
            255
          );
        }
      }
    }
  }

  getLegendEntries(): readonly LegendEntry[] | null {
    return null;
  }
}
