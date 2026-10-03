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
 *
 * Axis 0 points straight up and the axes follow clockwise. Every series is
 * divided by the largest absolute value over all series, so the outer ring
 * corresponds to that maximum. All values must be finite.
 * @internal
 */
export class Radar2D implements Drawable {
  readonly kind = "radar";
  readonly series: Float64Array[];
  readonly axisCount: number;
  /** Value that maps to the outer ring (largest absolute value, or 1 if all are 0). */
  readonly maxValue: number;
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

    let maxVal = 0;
    for (const s of series) {
      if (s.length !== n)
        throw new InvalidParameterError(
          "All series must have the same number of values",
          "series",
          s.length
        );
      for (let i = 0; i < n; i++) {
        const v = s[i] ?? 0;
        if (!Number.isFinite(v)) {
          throw new InvalidParameterError(
            `radar values must be finite; received ${v}`,
            "series",
            v
          );
        }
        if (Math.abs(v) > maxVal) maxVal = Math.abs(v);
      }
    }

    const lw = options.linewidth ?? 2;
    if (!Number.isFinite(lw) || lw <= 0) {
      throw new InvalidParameterError(
        `linewidth must be a positive number; received ${lw}`,
        "linewidth",
        lw
      );
    }

    this.series = series;
    this.axisCount = n;
    this.maxValue = maxVal === 0 ? 1 : maxVal;
    this.linewidth = lw;
    this.colors = series.map((_, i) =>
      normalizeColor(
        options.colors?.[i],
        ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"][i % 6] ?? "#1f77b4"
      )
    );
    this.labels = series.map((_, i) => normalizeLegendLabel(options.labels?.[i]));
  }

  getDataRange(): DataRange | null {
    // Radar is drawn in a fixed [-1.3, 1.3] coordinate space (unit ring plus margin)
    return { xmin: -1.3, xmax: 1.3, ymin: -1.3, ymax: 1.3 };
  }

  drawSVG(ctx: SvgDrawContext): void {
    const n = this.axisCount;
    const angleStep = (2 * Math.PI) / n;

    const maxVal = this.maxValue;

    // Draw grid rings
    for (let level = 1; level <= 4; level++) {
      const r = level / 4;
      const points: string[] = [];
      for (let i = 0; i < n; i++) {
        const angle = Math.PI / 2 - angleStep * i;
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
      const angle = Math.PI / 2 - angleStep * i;
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
        const angle = Math.PI / 2 - angleStep * i;
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
    const maxVal = this.maxValue;
    const n = this.axisCount;
    const angleStep = (2 * Math.PI) / n;
    const px = (r: number, i: number): number =>
      Math.round(ctx.transform.xToPx(r * Math.cos(Math.PI / 2 - angleStep * i)));
    const py = (r: number, i: number): number =>
      Math.round(ctx.transform.yToPx(r * Math.sin(Math.PI / 2 - angleStep * i)));

    // Grid rings and axes, same layout as the SVG output.
    for (let level = 1; level <= 4; level++) {
      const r = level / 4;
      for (let i = 0; i < n; i++) {
        const j = (i + 1) % n;
        ctx.canvas.drawLineRGBA(px(r, i), py(r, i), px(r, j), py(r, j), 204, 204, 204, 255);
      }
    }
    const ox = Math.round(ctx.transform.xToPx(0));
    const oy = Math.round(ctx.transform.yToPx(0));
    for (let i = 0; i < n; i++) {
      ctx.canvas.drawLineRGBA(ox, oy, px(1, i), py(1, i), 153, 153, 153, 255);
    }

    for (let s = 0; s < this.series.length; s++) {
      const data = this.series[s];
      if (!data) continue;
      const rgba = parseHexColorToRGBA(this.colors[s] ?? "#1f77b4");
      const xs = new Array<number>(n);
      const ys = new Array<number>(n);
      for (let i = 0; i < n; i++) {
        const val = (data[i] ?? 0) / maxVal;
        xs[i] = px(val, i);
        ys[i] = py(val, i);
      }
      ctx.canvas.fillPolygonRGBA(xs, ys, rgba.r, rgba.g, rgba.b, Math.round(rgba.a * 0.15));
      for (let i = 0; i < n; i++) {
        const j = (i + 1) % n;
        ctx.canvas.drawLineRGBA(
          xs[i] ?? 0,
          ys[i] ?? 0,
          xs[j] ?? 0,
          ys[j] ?? 0,
          rgba.r,
          rgba.g,
          rgba.b,
          rgba.a
        );
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
