import { InvalidParameterError } from "../../core";
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
import { calculateQuartiles, calculateWhiskers } from "../utils/statistics";
import { isFiniteNumber } from "../utils/validation";
import { escapeXml } from "../utils/xml";

/**
 * @internal
 */
export class Boxplot implements Drawable {
  readonly kind = "boxplot";
  readonly position: number;
  readonly q1: number;
  readonly median: number;
  readonly q3: number;
  readonly whiskerLow: number;
  readonly whiskerHigh: number;
  readonly outliers: readonly number[];
  readonly color: Color;
  readonly edgecolor: Color;
  readonly boxWidth: number;
  readonly label: string | null;
  readonly hasData: boolean;

  constructor(position: number, data: Float64Array, options: PlotOptions) {
    this.position = position;
    this.color = normalizeColor(options.color, "#8c564b");
    this.edgecolor = normalizeColor(options.edgecolor, "#000000");
    this.boxWidth = 0.5;
    this.label = normalizeLegendLabel(options.label);

    const sorted = Array.from(data)
      .filter(isFiniteNumber)
      .sort((a, b) => a - b);
    const n = sorted.length;

    if (n === 0) {
      if (data.length > 0) {
        throw new InvalidParameterError(
          "boxplot data must contain at least one finite value",
          "data",
          data
        );
      }
      this.hasData = false;
      this.q1 = 0;
      this.median = 0;
      this.q3 = 0;
      this.whiskerLow = 0;
      this.whiskerHigh = 0;
      this.outliers = [];
      return;
    }
    this.hasData = true;

    // Use proper statistical calculations
    const { q1, median, q3 } = calculateQuartiles(sorted);
    this.q1 = q1;
    this.median = median;
    this.q3 = q3;

    const { lowerWhisker, upperWhisker, outliers } = calculateWhiskers(sorted, q1, q3);
    this.whiskerLow = lowerWhisker;
    this.whiskerHigh = upperWhisker;
    this.outliers = outliers;
  }

  getDataRange(): DataRange | null {
    if (!this.hasData) return null;
    let ymin = this.whiskerLow;
    let ymax = this.whiskerHigh;
    for (const o of this.outliers) {
      if (o < ymin) ymin = o;
      if (o > ymax) ymax = o;
    }
    return {
      xmin: this.position - this.boxWidth,
      xmax: this.position + this.boxWidth,
      ymin,
      ymax,
    };
  }

  drawSVG(ctx: SvgDrawContext): void {
    if (!this.hasData) return;
    const x = this.position;
    const x0 = ctx.transform.xToPx(x - this.boxWidth / 2);
    const x1 = ctx.transform.xToPx(x + this.boxWidth / 2);
    const xc = ctx.transform.xToPx(x);

    const yq1 = ctx.transform.yToPx(this.q1);
    const ymed = ctx.transform.yToPx(this.median);
    const yq3 = ctx.transform.yToPx(this.q3);
    const ywl = ctx.transform.yToPx(this.whiskerLow);
    const ywh = ctx.transform.yToPx(this.whiskerHigh);

    // Draw box
    ctx.push(
      `<rect x="${Math.min(x0, x1).toFixed(2)}" y="${Math.min(yq1, yq3).toFixed(2)}" width="${Math.abs(x1 - x0).toFixed(2)}" height="${Math.abs(yq3 - yq1).toFixed(2)}" fill="${escapeXml(this.color)}" stroke="${escapeXml(this.edgecolor)}" />`
    );

    // Draw median line
    ctx.push(
      `<line x1="${x0.toFixed(2)}" y1="${ymed.toFixed(2)}" x2="${x1.toFixed(2)}" y2="${ymed.toFixed(2)}" stroke="${escapeXml(this.edgecolor)}" stroke-width="2" />`
    );

    // Draw whiskers
    ctx.push(
      `<line x1="${xc.toFixed(2)}" y1="${yq1.toFixed(2)}" x2="${xc.toFixed(2)}" y2="${ywl.toFixed(2)}" stroke="${escapeXml(this.edgecolor)}" />`
    );
    ctx.push(
      `<line x1="${xc.toFixed(2)}" y1="${yq3.toFixed(2)}" x2="${xc.toFixed(2)}" y2="${ywh.toFixed(2)}" stroke="${escapeXml(this.edgecolor)}" />`
    );

    // Draw whisker caps (half the box width in pixel space)
    const capHalf = Math.abs(x1 - x0) / 4;
    ctx.push(
      `<line x1="${(xc - capHalf).toFixed(2)}" y1="${ywl.toFixed(2)}" x2="${(xc + capHalf).toFixed(2)}" y2="${ywl.toFixed(2)}" stroke="${escapeXml(this.edgecolor)}" />`
    );
    ctx.push(
      `<line x1="${(xc - capHalf).toFixed(2)}" y1="${ywh.toFixed(2)}" x2="${(xc + capHalf).toFixed(2)}" y2="${ywh.toFixed(2)}" stroke="${escapeXml(this.edgecolor)}" />`
    );

    // Draw outliers
    for (const outlier of this.outliers) {
      const yo = ctx.transform.yToPx(outlier);
      ctx.push(
        `<circle cx="${xc.toFixed(2)}" cy="${yo.toFixed(2)}" r="3" fill="${escapeXml(this.edgecolor)}" stroke="none" />`
      );
    }
  }

  drawRaster(ctx: RasterDrawContext): void {
    if (!this.hasData) return;
    const rgba = parseHexColorToRGBA(this.color);
    const edge = parseHexColorToRGBA(this.edgecolor);
    const x = this.position;
    const x0 = Math.round(ctx.transform.xToPx(x - this.boxWidth / 2));
    const x1 = Math.round(ctx.transform.xToPx(x + this.boxWidth / 2));
    const xc = Math.round(ctx.transform.xToPx(x));

    const yq1 = Math.round(ctx.transform.yToPx(this.q1));
    const ymed = Math.round(ctx.transform.yToPx(this.median));
    const yq3 = Math.round(ctx.transform.yToPx(this.q3));
    const ywl = Math.round(ctx.transform.yToPx(this.whiskerLow));
    const ywh = Math.round(ctx.transform.yToPx(this.whiskerHigh));

    // Draw box
    const rx = Math.min(x0, x1);
    const ry = Math.min(yq1, yq3);
    const w = Math.abs(x1 - x0);
    const h = Math.abs(yq3 - yq1);

    ctx.canvas.fillRectRGBA(rx, ry, w, h, rgba.r, rgba.g, rgba.b, rgba.a);

    // Stroke box border
    ctx.canvas.drawLineRGBA(rx, ry, rx + w, ry, edge.r, edge.g, edge.b, edge.a);
    ctx.canvas.drawLineRGBA(rx + w, ry, rx + w, ry + h, edge.r, edge.g, edge.b, edge.a);
    ctx.canvas.drawLineRGBA(rx + w, ry + h, rx, ry + h, edge.r, edge.g, edge.b, edge.a);
    ctx.canvas.drawLineRGBA(rx, ry + h, rx, ry, edge.r, edge.g, edge.b, edge.a);

    // Draw median line
    ctx.canvas.drawLineRGBA(x0, ymed, x1, ymed, edge.r, edge.g, edge.b, edge.a);

    // Draw whiskers
    ctx.canvas.drawLineRGBA(xc, yq1, xc, ywl, edge.r, edge.g, edge.b, edge.a);
    ctx.canvas.drawLineRGBA(xc, yq3, xc, ywh, edge.r, edge.g, edge.b, edge.a);

    // Draw whisker caps (half the box width in pixel space)
    const capHalf = Math.round(Math.abs(x1 - x0) / 4);
    ctx.canvas.drawLineRGBA(xc - capHalf, ywl, xc + capHalf, ywl, edge.r, edge.g, edge.b, edge.a);
    ctx.canvas.drawLineRGBA(xc - capHalf, ywh, xc + capHalf, ywh, edge.r, edge.g, edge.b, edge.a);

    // Draw outliers
    for (const outlier of this.outliers) {
      const yo = Math.round(ctx.transform.yToPx(outlier));
      ctx.canvas.drawCircleRGBA(xc, yo, 3, edge.r, edge.g, edge.b, edge.a);
    }
  }

  getLegendEntries(): readonly LegendEntry[] | null {
    const entry = buildLegendEntry(this.label, {
      color: this.color,
      shape: "box",
    });
    return entry ? [entry] : null;
  }
}
