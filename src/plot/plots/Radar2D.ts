/**
 * @see {@link https://deepbox.dev/docs/plot-basic | Deepbox documentation}
 */

import { InvalidParameterError } from "../../core";
import type {
  Color,
  DataRange,
  Drawable,
  LegendEntry,
  RasterDrawContext,
  SvgDrawContext,
} from "../types";
import { normalizeColor, parseHexColorToRGBA } from "../utils/colors";
import { buildLegendEntry, normalizeLegendLabel } from "../utils/legend";
import { escapeXml } from "../utils/xml";

/**
 * Radar (spider) chart: plots multi-dimensional data on radial axes.
 * @internal
 */
export class Radar2D implements Drawable {
  readonly kind = "radar";
  readonly series: Float64Array[];
  readonly axisCount: number;
  readonly colors: Color[];
  readonly labels: (string | null)[];
  readonly linewidth: number;

  constructor(
    series: Float64Array[],
    options: {
      colors?: readonly Color[];
      labels?: readonly string[];
      linewidth?: number;
    } = {}
  ) {
    if (series.length === 0)
      throw new InvalidParameterError("At least one series is required", "series", 0);

    const n = series[0]?.length ?? 0;
    if (n < 3) throw new InvalidParameterError("Radar chart requires at least 3 axes", "axes", n);

    for (const s of series) {
      if (s.length !== n)
        throw new InvalidParameterError(
          "All series must have the same number of values",
          "series",
          s.length
        );
    }

    this.series = series;
    this.axisCount = n;
    this.linewidth = options.linewidth ?? 2;
    this.colors = series.map((_, i) =>
      normalizeColor(
        options.colors?.[i],
        ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"][i % 6] ?? "#1f77b4"
      )
    );
    this.labels = series.map((_, i) => normalizeLegendLabel(options.labels?.[i]));
  }

  getDataRange(): DataRange | null {
    // Radar is drawn in a fixed [-1.2, 1.2] coordinate space
    return { xmin: -1.3, xmax: 1.3, ymin: -1.3, ymax: 1.3 };
  }

  drawSVG(ctx: SvgDrawContext): void {
    const n = this.axisCount;
    const angleStep = (2 * Math.PI) / n;

    // Find max value for normalization
    let maxVal = 0;
    for (const s of this.series) {
      for (let i = 0; i < s.length; i++) {
        maxVal = Math.max(maxVal, Math.abs(s[i] ?? 0));
      }
    }
    if (maxVal === 0) maxVal = 1;

    // Draw grid circles
    for (let level = 1; level <= 4; level++) {
      const r = level / 4;
      const points: string[] = [];
      for (let i = 0; i < n; i++) {
        const angle = angleStep * i - Math.PI / 2;
        const gx = ctx.transform.xToPx(r * Math.cos(angle));
        const gy = ctx.transform.yToPx(r * Math.sin(angle));
        points.push(`${gx.toFixed(2)},${gy.toFixed(2)}`);
      }
      ctx.push(
        `<polygon points="${points.join(" ")}" fill="none" stroke="#ccc" stroke-width="0.5" />`
      );
    }

    // Draw axes
    for (let i = 0; i < n; i++) {
      const angle = angleStep * i - Math.PI / 2;
      const x0 = ctx.transform.xToPx(0);
      const y0 = ctx.transform.yToPx(0);
      const x1 = ctx.transform.xToPx(Math.cos(angle));
      const y1 = ctx.transform.yToPx(Math.sin(angle));
      ctx.push(
        `<line x1="${x0.toFixed(2)}" y1="${y0.toFixed(2)}" x2="${x1.toFixed(2)}" y2="${y1.toFixed(2)}" stroke="#999" stroke-width="0.5" />`
      );
    }

    // Draw data polygons
    for (let s = 0; s < this.series.length; s++) {
      const data = this.series[s];
      if (!data) continue;
      const ec = escapeXml(this.colors[s] ?? "#1f77b4");
      const points: string[] = [];
      for (let i = 0; i < n; i++) {
        const val = (data[i] ?? 0) / maxVal;
        const angle = angleStep * i - Math.PI / 2;
        const gx = ctx.transform.xToPx(val * Math.cos(angle));
        const gy = ctx.transform.yToPx(val * Math.sin(angle));
        points.push(`${gx.toFixed(2)},${gy.toFixed(2)}`);
      }
      ctx.push(
        `<polygon points="${points.join(" ")}" fill="${ec}" fill-opacity="0.15" stroke="${ec}" stroke-width="${this.linewidth}" />`
      );
    }
  }

  drawRaster(ctx: RasterDrawContext): void {
    // Minimal raster fallback — draw series as filled polygons
    let maxVal = 0;
    for (const s of this.series) {
      for (let i = 0; i < s.length; i++) {
        maxVal = Math.max(maxVal, Math.abs(s[i] ?? 0));
      }
    }
    if (maxVal === 0) maxVal = 1;

    const n = this.axisCount;
    const angleStep = (2 * Math.PI) / n;

    for (let s = 0; s < this.series.length; s++) {
      const data = this.series[s];
      if (!data) continue;
      const rgba = parseHexColorToRGBA(this.colors[s] ?? "#1f77b4");
      for (let i = 0; i < n; i++) {
        const val = (data[i] ?? 0) / maxVal;
        const angle = angleStep * i - Math.PI / 2;
        const px = Math.round(ctx.transform.xToPx(val * Math.cos(angle)));
        const py = Math.round(ctx.transform.yToPx(val * Math.sin(angle)));

        const nextI = (i + 1) % n;
        const nextVal = (data[nextI] ?? 0) / maxVal;
        const nextAngle = angleStep * nextI - Math.PI / 2;
        const npx = Math.round(ctx.transform.xToPx(nextVal * Math.cos(nextAngle)));
        const npy = Math.round(ctx.transform.yToPx(nextVal * Math.sin(nextAngle)));
        ctx.canvas.drawLineRGBA(px, py, npx, npy, rgba.r, rgba.g, rgba.b, rgba.a);
      }
    }
  }

  getLegendEntries(): readonly LegendEntry[] | null {
    const entries: LegendEntry[] = [];
    for (let i = 0; i < this.series.length; i++) {
      const entry = buildLegendEntry(this.labels[i] ?? null, {
        color: this.colors[i] ?? "#1f77b4",
        shape: "line",
        lineWidth: this.linewidth,
      });
      if (entry) entries.push(entry);
    }
    return entries.length > 0 ? entries : null;
  }
}
