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
import { calculateQuartiles, kernelDensityEstimation } from "../utils/statistics";
import { isFiniteNumber } from "../utils/validation";
import { escapeXml } from "../utils/xml";

/**
 * @internal
 */
export class Violinplot implements Drawable {
  readonly kind = "violinplot";
  readonly position: number;
  readonly q1: number;
  readonly median: number;
  readonly q3: number;
  readonly kdePoints: readonly number[];
  readonly kdeValues: readonly number[];
  readonly color: Color;
  readonly edgecolor: Color;
  readonly violinWidth: number;
  readonly label: string | null;
  readonly hasData: boolean;

  constructor(position: number, data: Float64Array, options: PlotOptions) {
    this.position = position;
    this.color = normalizeColor(options.color, "#8c564b");
    this.edgecolor = normalizeColor(options.edgecolor, "#000000");
    this.violinWidth = 0.8;
    this.label = normalizeLegendLabel(options.label);

    const sorted = Array.from(data)
      .filter(isFiniteNumber)
      .sort((a, b) => a - b);
    const n = sorted.length;

    if (n === 0) {
      if (data.length > 0) {
        throw new InvalidParameterError(
          "violinplot data must contain at least one finite value",
          "data",
          data
        );
      }
      this.hasData = false;
      this.q1 = 0;
      this.median = 0;
      this.q3 = 0;
      this.kdeValues = [];
      this.kdePoints = [];
      return;
    }
    this.hasData = true;

    // Calculate quartiles
    const { q1, median, q3 } = calculateQuartiles(sorted);
    this.q1 = q1;
    this.median = median;
    this.q3 = q3;

    // Calculate KDE for violin shape
    const firstVal = sorted[0] ?? 0;
    const lastVal = sorted[sorted.length - 1] ?? 0;
    const dataRange = lastVal - firstVal;
    const padding = dataRange * 0.1;
    const min = firstVal - padding;
    const max = lastVal + padding;

    // Create points for KDE evaluation
    const numPoints = 100;
    const kdePoints: number[] = [];
    for (let i = 0; i < numPoints; i++) {
      kdePoints.push(min + (i / (numPoints - 1)) * (max - min));
    }

    // Calculate KDE values
    this.kdeValues = kernelDensityEstimation(sorted, kdePoints, 0);
    this.kdePoints = kdePoints;
  }

  getDataRange(): DataRange | null {
    if (!this.hasData) return null;
    if (this.kdePoints.length === 0) return null;

    const minY = this.kdePoints[0] ?? 0;
    const maxY = this.kdePoints[this.kdePoints.length - 1] ?? 0;

    return {
      xmin: this.position - this.violinWidth / 2,
      xmax: this.position + this.violinWidth / 2,
      ymin: minY,
      ymax: maxY,
    };
  }

  drawSVG(ctx: SvgDrawContext): void {
    if (!this.hasData) return;
    const x = this.position;

    // Find max KDE value for scaling
    let maxKDE = 0;
    for (let i = 0; i < this.kdeValues.length; i++) {
      const v = this.kdeValues[i] ?? 0;
      if (v > maxKDE) maxKDE = v;
    }
    if (maxKDE === 0) return;

    // Build violin path
    const pathPoints: string[] = [];

    // Left side of violin (from bottom to top)
    for (let i = 0; i < this.kdePoints.length; i++) {
      const y = this.kdePoints[i] ?? 0;
      const kde = this.kdeValues[i] ?? 0;
      const width = (kde / maxKDE) * this.violinWidth;
      const xLeft = x - width / 2;

      const px = ctx.transform.xToPx(xLeft);
      const py = ctx.transform.yToPx(y);
      pathPoints.push(`${px.toFixed(2)},${py.toFixed(2)}`);
    }

    // Right side of violin (from top to bottom)
    for (let i = this.kdePoints.length - 1; i >= 0; i--) {
      const y = this.kdePoints[i] ?? 0;
      const kde = this.kdeValues[i] ?? 0;
      const width = (kde / maxKDE) * this.violinWidth;
      const xRight = x + width / 2;

      const px = ctx.transform.xToPx(xRight);
      const py = ctx.transform.yToPx(y);
      pathPoints.push(`${px.toFixed(2)},${py.toFixed(2)}`);
    }

    // Draw violin shape
    if (pathPoints.length > 0) {
      ctx.push(
        `<path d="M ${pathPoints.join(" L ")} Z" fill="${escapeXml(this.color)}" stroke="${escapeXml(this.edgecolor)}" stroke-width="1" />`
      );
    }

    // Draw quartile indicators
    const yq1 = ctx.transform.yToPx(this.q1);
    const ymed = ctx.transform.yToPx(this.median);
    const yq3 = ctx.transform.yToPx(this.q3);

    const indicatorWidth = this.violinWidth * 0.8;
    const xLeft = ctx.transform.xToPx(x - indicatorWidth / 2);
    const xRight = ctx.transform.xToPx(x + indicatorWidth / 2);

    // Draw quartile lines
    ctx.push(
      `<line x1="${xLeft.toFixed(2)}" y1="${yq1.toFixed(2)}" x2="${xRight.toFixed(2)}" y2="${yq1.toFixed(2)}" stroke="${escapeXml(this.edgecolor)}" stroke-width="2" />`
    );
    ctx.push(
      `<line x1="${xLeft.toFixed(2)}" y1="${ymed.toFixed(2)}" x2="${xRight.toFixed(2)}" y2="${ymed.toFixed(2)}" stroke="${escapeXml(this.edgecolor)}" stroke-width="3" />`
    );
    ctx.push(
      `<line x1="${xLeft.toFixed(2)}" y1="${yq3.toFixed(2)}" x2="${xRight.toFixed(2)}" y2="${yq3.toFixed(2)}" stroke="${escapeXml(this.edgecolor)}" stroke-width="2" />`
    );
  }

  drawRaster(ctx: RasterDrawContext): void {
    if (!this.hasData) return;
    const rgba = parseHexColorToRGBA(this.color);
    const edge = parseHexColorToRGBA(this.edgecolor);
    const x = this.position;

    // Find max KDE value for scaling
    let maxKDE = 0;
    for (let i = 0; i < this.kdeValues.length; i++) {
      const v = this.kdeValues[i] ?? 0;
      if (v > maxKDE) maxKDE = v;
    }
    if (maxKDE === 0) return;

    // Simple raster rendering - draw filled polygons for violin
    for (let i = 0; i < this.kdePoints.length - 1; i++) {
      const y1 = this.kdePoints[i] ?? 0;
      const y2 = this.kdePoints[i + 1] ?? 0;
      const kde1 = this.kdeValues[i] ?? 0;
      const kde2 = this.kdeValues[i + 1] ?? 0;

      const width1 = (kde1 / maxKDE) * this.violinWidth;
      const width2 = (kde2 / maxKDE) * this.violinWidth;

      const xLeft1 = x - width1 / 2;
      const xRight1 = x + width1 / 2;
      const xLeft2 = x - width2 / 2;
      const xRight2 = x + width2 / 2;

      const pxLeft1 = Math.round(ctx.transform.xToPx(xLeft1));
      const pxRight1 = Math.round(ctx.transform.xToPx(xRight1));
      const pxLeft2 = Math.round(ctx.transform.xToPx(xLeft2));
      const pxRight2 = Math.round(ctx.transform.xToPx(xRight2));

      const py1 = Math.round(ctx.transform.yToPx(y1));
      const py2 = Math.round(ctx.transform.yToPx(y2));

      // Draw filled quadrilateral as two triangles
      ctx.canvas.drawLineRGBA(pxLeft1, py1, pxRight1, py1, rgba.r, rgba.g, rgba.b, rgba.a);
      ctx.canvas.drawLineRGBA(pxRight1, py1, pxRight2, py2, rgba.r, rgba.g, rgba.b, rgba.a);
      ctx.canvas.drawLineRGBA(pxRight2, py2, pxLeft2, py2, rgba.r, rgba.g, rgba.b, rgba.a);
      ctx.canvas.drawLineRGBA(pxLeft2, py2, pxLeft1, py1, rgba.r, rgba.g, rgba.b, rgba.a);

      // Fill the area by drawing horizontal lines
      const minY = Math.min(py1, py2);
      const maxY = Math.max(py1, py2);
      if (py1 === py2) {
        const leftX = Math.min(pxLeft1, pxLeft2);
        const rightX = Math.max(pxRight1, pxRight2);
        ctx.canvas.drawLineRGBA(leftX, py1, rightX, py1, rgba.r, rgba.g, rgba.b, rgba.a);
      } else {
        for (let y = minY; y <= maxY; y++) {
          const t = (y - py1) / (py2 - py1);
          const leftX = pxLeft1 + t * (pxLeft2 - pxLeft1);
          const rightX = pxRight1 + t * (pxRight2 - pxRight1);
          ctx.canvas.drawLineRGBA(
            Math.round(leftX),
            y,
            Math.round(rightX),
            y,
            rgba.r,
            rgba.g,
            rgba.b,
            rgba.a
          );
        }
      }
    }

    // Draw quartile indicators
    const yq1 = Math.round(ctx.transform.yToPx(this.q1));
    const ymed = Math.round(ctx.transform.yToPx(this.median));
    const yq3 = Math.round(ctx.transform.yToPx(this.q3));

    const indicatorWidth = this.violinWidth * 0.8;
    const xLeft = Math.round(ctx.transform.xToPx(x - indicatorWidth / 2));
    const xRight = Math.round(ctx.transform.xToPx(x + indicatorWidth / 2));

    // Draw quartile lines
    ctx.canvas.drawLineRGBA(xLeft, yq1, xRight, yq1, edge.r, edge.g, edge.b, edge.a);
    ctx.canvas.drawLineRGBA(xLeft, ymed, xRight, ymed, edge.r, edge.g, edge.b, edge.a);
    ctx.canvas.drawLineRGBA(xLeft, yq3, xRight, yq3, edge.r, edge.g, edge.b, edge.a);
  }

  getLegendEntries(): readonly LegendEntry[] | null {
    const entry = buildLegendEntry(this.label, {
      color: this.color,
      shape: "box",
    });
    return entry ? [entry] : null;
  }
}
