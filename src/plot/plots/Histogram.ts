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
import { isFiniteNumber } from "../utils/validation";
import { escapeXml } from "../utils/xml";

/**
 * @internal
 */
export class Histogram implements Drawable {
  readonly kind = "histogram";
  readonly bins: Float64Array;
  readonly counts: Float64Array;
  readonly binWidth: number;
  readonly color: Color;
  readonly edgecolor: Color;
  readonly label: string | null;

  constructor(data: Float64Array, numBins: number, options: PlotOptions) {
    if (!Number.isFinite(numBins) || Math.trunc(numBins) !== numBins || numBins <= 0) {
      throw new InvalidParameterError(
        `bins must be a positive integer; received ${numBins}`,
        "bins",
        numBins
      );
    }
    this.color = normalizeColor(options.color, "#d62728");
    this.edgecolor = normalizeColor(options.edgecolor, "#000000");
    this.label = normalizeLegendLabel(options.label);

    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    let hasFinite = false;
    let hasInfinity = false;
    for (let i = 0; i < data.length; i++) {
      const v = data[i] ?? 0;
      if (isFiniteNumber(v)) {
        hasFinite = true;
        min = Math.min(min, v);
        max = Math.max(max, v);
      } else if (v === Infinity || v === -Infinity) {
        hasInfinity = true;
      }
    }

    if (!hasFinite) {
      if (hasInfinity) {
        throw new InvalidParameterError(
          "histogram data must contain at least one finite value",
          "data",
          data
        );
      }
      this.bins = new Float64Array(0);
      this.counts = new Float64Array(0);
      this.binWidth = 1;
      return;
    }

    const span = max - min;
    const binWidth = span > 0 ? span / numBins : 1;
    const binStart = span > 0 ? min : min - (numBins * binWidth) / 2;
    this.bins = new Float64Array(numBins);
    this.counts = new Float64Array(numBins);
    this.binWidth = binWidth;

    for (let i = 0; i < numBins; i++) {
      this.bins[i] = binStart + i * binWidth;
      this.counts[i] = 0;
    }

    if (span === 0) {
      let finiteCount = 0;
      for (let i = 0; i < data.length; i++) {
        const v = data[i] ?? 0;
        if (isFiniteNumber(v)) finiteCount++;
      }
      const mid = Math.floor(numBins / 2);
      this.counts[mid] = finiteCount;
      return;
    }

    for (let i = 0; i < data.length; i++) {
      const v = data[i] ?? 0;
      if (!isFiniteNumber(v)) continue;
      const rawIdx = Math.floor((v - min) / binWidth);
      const binIdx = Math.min(Math.max(0, rawIdx), numBins - 1);
      this.counts[binIdx] = (this.counts[binIdx] ?? 0) + 1;
    }
  }

  getDataRange(): DataRange | null {
    if (this.bins.length === 0) return null;
    const xmin = this.bins[0] ?? 0;
    const xmax = (this.bins[this.bins.length - 1] ?? 0) + this.binWidth;
    let ymax = 0;
    for (let i = 0; i < this.counts.length; i++) {
      ymax = Math.max(ymax, this.counts[i] ?? 0);
    }
    return { xmin, xmax, ymin: 0, ymax };
  }

  drawSVG(ctx: SvgDrawContext): void {
    if (this.bins.length === 0) return;
    const binWidth = this.binWidth;
    for (let i = 0; i < this.bins.length; i++) {
      const x0 = this.bins[i] ?? 0;
      const count = this.counts[i] ?? 0;
      const px0 = ctx.transform.xToPx(x0);
      const px1 = ctx.transform.xToPx(x0 + binWidth);
      const py0 = ctx.transform.yToPx(0);
      const py1 = ctx.transform.yToPx(count);
      const w = Math.abs(px1 - px0);
      const h = Math.abs(py1 - py0);
      const rx = Math.min(px0, px1);
      const ry = Math.min(py0, py1);
      ctx.push(
        `<rect x="${rx.toFixed(2)}" y="${ry.toFixed(2)}" width="${w.toFixed(2)}" height="${h.toFixed(2)}" fill="${escapeXml(this.color)}" stroke="${escapeXml(this.edgecolor)}" />`
      );
    }
  }

  drawRaster(ctx: RasterDrawContext): void {
    if (this.bins.length === 0) return;
    const rgba = parseHexColorToRGBA(this.color);
    const edge = parseHexColorToRGBA(this.edgecolor);
    const binWidth = this.binWidth;
    for (let i = 0; i < this.bins.length; i++) {
      const x0 = this.bins[i] ?? 0;
      const count = this.counts[i] ?? 0;
      const px0 = Math.round(ctx.transform.xToPx(x0));
      const px1 = Math.round(ctx.transform.xToPx(x0 + binWidth));
      const py0 = Math.round(ctx.transform.yToPx(0));
      const py1 = Math.round(ctx.transform.yToPx(count));
      const w = Math.abs(px1 - px0);
      const h = Math.abs(py1 - py0);
      const rx = Math.min(px0, px1);
      const ry = Math.min(py0, py1);

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
