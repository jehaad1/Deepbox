import { InvalidParameterError } from "../../core";
import type {
  DataRange,
  Drawable,
  LegendEntry,
  PlotOptions,
  RasterDrawContext,
  SvgDrawContext,
} from "../types";
import { applyColormap } from "../utils/colormaps";
import { normalizeColor } from "../utils/colors";
import { buildLegendEntry, normalizeLegendLabel } from "../utils/legend";
import { isFiniteNumber } from "../utils/validation";

/**
 * @internal
 */
export class Heatmap2D implements Drawable {
  readonly kind = "heatmap";
  readonly data: Float64Array;
  readonly rows: number;
  readonly cols: number;
  readonly vmin: number;
  readonly vmax: number;
  readonly colormap: "viridis" | "plasma" | "inferno" | "magma" | "grayscale";
  readonly xMin: number;
  readonly xMax: number;
  readonly yMin: number;
  readonly yMax: number;
  readonly label: string | null;

  constructor(data: Float64Array, rows: number, cols: number, options: PlotOptions) {
    this.data = data;
    this.rows = rows;
    this.cols = cols;
    const colormap = options.colormap ?? "viridis";
    if (!["viridis", "plasma", "inferno", "magma", "grayscale"].includes(colormap)) {
      throw new InvalidParameterError(
        `colormap must be one of viridis, plasma, inferno, magma, grayscale; received ${colormap}`,
        "colormap",
        colormap
      );
    }
    this.colormap = colormap;
    this.label = normalizeLegendLabel(options.label);

    const extent = options.extent;
    if (extent) {
      const { xmin, xmax, ymin, ymax } = extent;
      if (
        !Number.isFinite(xmin) ||
        !Number.isFinite(xmax) ||
        !Number.isFinite(ymin) ||
        !Number.isFinite(ymax)
      ) {
        throw new InvalidParameterError("extent values must be finite", "extent", extent);
      }
      if (xmax <= xmin || ymax <= ymin) {
        throw new InvalidParameterError("extent ranges must be positive", "extent", extent);
      }
      this.xMin = xmin;
      this.xMax = xmax;
      this.yMin = ymin;
      this.yMax = ymax;
    } else {
      this.xMin = 0;
      this.xMax = cols;
      this.yMin = 0;
      this.yMax = rows;
    }

    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < data.length; i++) {
      const v = data[i] ?? 0;
      if (isFiniteNumber(v)) {
        min = Math.min(min, v);
        max = Math.max(max, v);
      }
    }
    if (!Number.isFinite(min) || !Number.isFinite(max)) {
      throw new InvalidParameterError(
        "heatmap data must contain at least one finite value",
        "data",
        data
      );
    }

    if (options.vmin !== undefined && !Number.isFinite(options.vmin)) {
      throw new InvalidParameterError(
        `vmin must be finite; received ${options.vmin}`,
        "vmin",
        options.vmin
      );
    }
    if (options.vmax !== undefined && !Number.isFinite(options.vmax)) {
      throw new InvalidParameterError(
        `vmax must be finite; received ${options.vmax}`,
        "vmax",
        options.vmax
      );
    }

    const vmin = options.vmin ?? min;
    const vmax = options.vmax ?? max;
    if (Number.isFinite(vmin) && Number.isFinite(vmax) && vmin > vmax) {
      throw new InvalidParameterError(
        `vmin must be <= vmax; received vmin=${vmin} vmax=${vmax}`,
        "vmin/vmax",
        { vmin, vmax }
      );
    }
    this.vmin = Number.isFinite(vmin) ? vmin : 0;
    this.vmax = Number.isFinite(vmax) ? vmax : 1;
  }

  getDataRange(): DataRange | null {
    return { xmin: this.xMin, xmax: this.xMax, ymin: this.yMin, ymax: this.yMax };
  }

  drawSVG(ctx: SvgDrawContext): void {
    if (this.rows <= 0 || this.cols <= 0) return;
    const range = this.vmax - this.vmin;
    const xSpan = this.xMax - this.xMin;
    const ySpan = this.yMax - this.yMin;
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        const v = this.data[i * this.cols + j] ?? 0;
        if (!isFiniteNumber(v)) continue;
        const normalized = range !== 0 ? (v - this.vmin) / range : 0;
        const intensity = Math.max(0, Math.min(1, normalized));
        const [r, g, b] = applyColormap(intensity, this.colormap);
        const color = `rgb(${r},${g},${b})`;
        const x0 = ctx.transform.xToPx(this.xMin + (j / this.cols) * xSpan);
        const x1 = ctx.transform.xToPx(this.xMin + ((j + 1) / this.cols) * xSpan);
        const y0 = ctx.transform.yToPx(this.yMin + (i / this.rows) * ySpan);
        const y1 = ctx.transform.yToPx(this.yMin + ((i + 1) / this.rows) * ySpan);
        const w = Math.abs(x1 - x0);
        const h = Math.abs(y1 - y0);
        const rx = Math.min(x0, x1);
        const ry = Math.min(y0, y1);
        ctx.push(
          `<rect x="${rx.toFixed(2)}" y="${ry.toFixed(2)}" width="${w.toFixed(2)}" height="${h.toFixed(2)}" fill="${color}" />`
        );
      }
    }
  }

  drawRaster(ctx: RasterDrawContext): void {
    if (this.rows <= 0 || this.cols <= 0) return;
    const range = this.vmax - this.vmin;
    const xSpan = this.xMax - this.xMin;
    const ySpan = this.yMax - this.yMin;
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        const v = this.data[i * this.cols + j] ?? 0;
        if (!isFiniteNumber(v)) continue;
        const normalized = range !== 0 ? (v - this.vmin) / range : 0;
        const intensity = Math.max(0, Math.min(1, normalized));
        const [r, g, b] = applyColormap(intensity, this.colormap);
        const x0 = Math.round(ctx.transform.xToPx(this.xMin + (j / this.cols) * xSpan));
        const x1 = Math.round(ctx.transform.xToPx(this.xMin + ((j + 1) / this.cols) * xSpan));
        const y0 = Math.round(ctx.transform.yToPx(this.yMin + (i / this.rows) * ySpan));
        const y1 = Math.round(ctx.transform.yToPx(this.yMin + ((i + 1) / this.rows) * ySpan));
        const w = Math.abs(x1 - x0);
        const h = Math.abs(y1 - y0);
        const rx = Math.min(x0, x1);
        const ry = Math.min(y0, y1);
        ctx.canvas.fillRectRGBA(rx, ry, w, h, r, g, b, 255);
      }
    }
  }

  getLegendEntries(): readonly LegendEntry[] | null {
    const entry = buildLegendEntry(this.label, {
      color: normalizeColor(`rgb(${applyColormap(0.5, this.colormap).join(",")})`, "#1f77b4"),
      shape: "box",
    });
    return entry ? [entry] : null;
  }
}
