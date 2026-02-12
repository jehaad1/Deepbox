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
import { applyColormap } from "../utils/colormaps";
import { normalizeColor, parseHexColorToRGBA } from "../utils/colors";
import type { ContourGrid } from "../utils/contours";
import { buildLegendEntry, normalizeLegendLabel } from "../utils/legend";
import { isFiniteNumber } from "../utils/validation";
import { escapeXml } from "../utils/xml";

const DEFAULT_COLORS: readonly Color[] = [
  "#1f77b4",
  "#ff7f0e",
  "#2ca02c",
  "#d62728",
  "#9467bd",
  "#8c564b",
  "#e377c2",
  "#7f7f7f",
  "#bcbd22",
  "#17becf",
];

type Vertex = {
  readonly x: number;
  readonly y: number;
  readonly v: number;
};

type BandTriangle = {
  readonly x1: number;
  readonly y1: number;
  readonly x2: number;
  readonly y2: number;
  readonly x3: number;
  readonly y3: number;
  readonly band: number;
};

function interpolateVertex(a: Vertex, b: Vertex, threshold: number): Vertex {
  const dv = b.v - a.v;
  const t = dv === 0 ? 0.5 : Math.max(0, Math.min(1, (threshold - a.v) / dv));
  return {
    x: a.x + t * (b.x - a.x),
    y: a.y + t * (b.y - a.y),
    v: threshold,
  };
}

function clipPolygonByThreshold(
  points: readonly Vertex[],
  threshold: number,
  keepAbove: boolean
): Vertex[] {
  if (points.length === 0) return [];
  const output: Vertex[] = [];
  for (let i = 0; i < points.length; i++) {
    const current = points[i];
    const next = points[(i + 1) % points.length];
    if (!current || !next) continue;
    const currentInside = keepAbove ? current.v >= threshold : current.v <= threshold;
    const nextInside = keepAbove ? next.v >= threshold : next.v <= threshold;

    if (currentInside && nextInside) {
      output.push(next);
    } else if (currentInside && !nextInside) {
      output.push(interpolateVertex(current, next, threshold));
    } else if (!currentInside && nextInside) {
      output.push(interpolateVertex(current, next, threshold));
      output.push(next);
    }
  }
  return output;
}

function triangulatePolygon(polygon: readonly Vertex[], band: number, out: BandTriangle[]): void {
  if (polygon.length < 3) return;
  const origin = polygon[0];
  if (!origin) return;
  for (let i = 1; i < polygon.length - 1; i++) {
    const v1 = polygon[i];
    const v2 = polygon[i + 1];
    if (!v1 || !v2) continue;
    out.push({
      x1: origin.x,
      y1: origin.y,
      x2: v1.x,
      y2: v1.y,
      x3: v2.x,
      y3: v2.y,
      band,
    });
  }
}

function addBandTriangles(
  tri: readonly Vertex[],
  levels: readonly number[],
  out: BandTriangle[]
): void {
  if (tri.length !== 3 || levels.length < 2) return;
  const v0 = tri[0]?.v ?? 0;
  const v1 = tri[1]?.v ?? 0;
  const v2 = tri[2]?.v ?? 0;
  const triMin = Math.min(v0, v1, v2);
  const triMax = Math.max(v0, v1, v2);
  const lastLevel = levels[levels.length - 1] ?? 0;
  if (triMax < (levels[0] ?? 0) || triMin > lastLevel) return;

  const bandCount = levels.length - 1;
  for (let band = 0; band < bandCount; band++) {
    const lo = levels[band] ?? 0;
    const hi = levels[band + 1] ?? lo;
    if (triMax < lo || triMin > hi) continue;
    let poly = clipPolygonByThreshold(tri, lo, true);
    if (poly.length === 0) continue;
    poly = clipPolygonByThreshold(poly, hi, false);
    if (poly.length < 3) continue;
    triangulatePolygon(poly, band, out);
  }
}

function resolveFillLevels(
  min: number,
  max: number,
  levels: number | readonly number[] | undefined
): number[] {
  if (Array.isArray(levels)) {
    const filtered = levels.filter((v) => Number.isFinite(v));
    const unique = Array.from(new Set(filtered));
    unique.sort((a, b) => a - b);
    if (unique.length < 2) {
      throw new InvalidParameterError(
        "levels must contain at least two finite values",
        "levels",
        levels
      );
    }
    return unique;
  }

  const numLevels = typeof levels === "number" ? levels : 10;
  if (!Number.isFinite(numLevels) || Math.trunc(numLevels) !== numLevels || numLevels <= 0) {
    throw new InvalidParameterError(
      `levels must be a positive integer; received ${numLevels}`,
      "levels",
      numLevels
    );
  }

  if (min === max) {
    return [min, max];
  }

  const step = (max - min) / numLevels;
  const boundaries = new Array<number>(numLevels + 1);
  for (let i = 0; i <= numLevels; i++) {
    boundaries[i] = min + i * step;
  }
  if (boundaries.length < 2) {
    return [min, max];
  }
  return boundaries;
}

function resolveBandColors(
  bandCount: number,
  options: PlotOptions
): {
  readonly colors: readonly Color[];
  readonly rgba: readonly { r: number; g: number; b: number; a: number }[];
} {
  const colors: Color[] = [];

  if (options.colors && options.colors.length > 0) {
    for (let i = 0; i < bandCount; i++) {
      const color = options.colors[i % options.colors.length];
      colors.push(normalizeColor(color, DEFAULT_COLORS[i % DEFAULT_COLORS.length] ?? "#1f77b4"));
    }
  } else if (options.color) {
    const normalized = normalizeColor(options.color, "#1f77b4");
    for (let i = 0; i < bandCount; i++) colors.push(normalized);
  } else if (options.colormap) {
    if (!["viridis", "plasma", "inferno", "magma", "grayscale"].includes(options.colormap)) {
      throw new InvalidParameterError(
        `colormap must be one of viridis, plasma, inferno, magma, grayscale; received ${options.colormap}`,
        "colormap",
        options.colormap
      );
    }
    const denom = Math.max(1, bandCount - 1);
    for (let i = 0; i < bandCount; i++) {
      const t = i / denom;
      const [r, g, b] = applyColormap(t, options.colormap);
      colors.push(`rgb(${r},${g},${b})`);
    }
  } else {
    const colormap = "viridis";
    const denom = Math.max(1, bandCount - 1);
    for (let i = 0; i < bandCount; i++) {
      const t = i / denom;
      const [r, g, b] = applyColormap(t, colormap);
      colors.push(`rgb(${r},${g},${b})`);
    }
  }

  const rgba = colors.map((c) => parseHexColorToRGBA(c));
  return { colors, rgba };
}

/**
 * @internal
 */
export class ContourF2D implements Drawable {
  readonly kind = "contourf";
  readonly rows: number;
  readonly cols: number;
  readonly xCoords: Float64Array;
  readonly yCoords: Float64Array;
  readonly levels: readonly number[];
  readonly bandColors: readonly Color[];
  readonly bandRGBA: readonly { r: number; g: number; b: number; a: number }[];
  readonly triangles: readonly BandTriangle[];
  readonly xmin: number;
  readonly xmax: number;
  readonly ymin: number;
  readonly ymax: number;
  readonly label: string | null;

  constructor(grid: ContourGrid, options: PlotOptions = {}) {
    this.rows = grid.rows;
    this.cols = grid.cols;
    this.xCoords = grid.xCoords;
    this.yCoords = grid.yCoords;
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
    const min = options.vmin ?? grid.dataMin;
    const max = options.vmax ?? grid.dataMax;
    if (Number.isFinite(min) && Number.isFinite(max) && min > max) {
      throw new InvalidParameterError(
        `vmin must be <= vmax; received vmin=${min} vmax=${max}`,
        "vmin/vmax",
        { vmin: min, vmax: max }
      );
    }
    this.levels = resolveFillLevels(min, max, options.levels);

    const bandCount = this.levels.length - 1;
    if (bandCount <= 0) {
      throw new InvalidParameterError(
        "levels must define at least one fill band",
        "levels",
        this.levels
      );
    }

    const { colors, rgba } = resolveBandColors(bandCount, options);
    this.bandColors = colors;
    this.bandRGBA = rgba;
    this.label = normalizeLegendLabel(options.label);

    let xMin = Number.POSITIVE_INFINITY;
    let xMax = Number.NEGATIVE_INFINITY;
    let yMin = Number.POSITIVE_INFINITY;
    let yMax = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < this.xCoords.length; i++) {
      const v = this.xCoords[i] ?? 0;
      if (v < xMin) xMin = v;
      if (v > xMax) xMax = v;
    }
    for (let i = 0; i < this.yCoords.length; i++) {
      const v = this.yCoords[i] ?? 0;
      if (v < yMin) yMin = v;
      if (v > yMax) yMax = v;
    }
    this.xmin = xMin;
    this.xmax = xMax;
    this.ymin = yMin;
    this.ymax = yMax;

    const cellRows = Math.max(0, this.rows - 1);
    const cellCols = Math.max(0, this.cols - 1);
    const triangles: BandTriangle[] = [];

    if (cellRows > 0 && cellCols > 0) {
      const data = grid.data;
      for (let i = 0; i < cellRows; i++) {
        const rowOffset = i * this.cols;
        const rowOffsetNext = (i + 1) * this.cols;
        const y0 = this.yCoords[i] ?? 0;
        const y1 = this.yCoords[i + 1] ?? 0;
        for (let j = 0; j < cellCols; j++) {
          const x0 = this.xCoords[j] ?? 0;
          const x1 = this.xCoords[j + 1] ?? 0;
          const v00 = data[rowOffset + j] ?? 0;
          const v10 = data[rowOffset + j + 1] ?? 0;
          const v11 = data[rowOffsetNext + j + 1] ?? 0;
          const v01 = data[rowOffsetNext + j] ?? 0;

          if (
            !isFiniteNumber(v00) ||
            !isFiniteNumber(v10) ||
            !isFiniteNumber(v11) ||
            !isFiniteNumber(v01)
          ) {
            continue;
          }

          const t0: Vertex = { x: x0, y: y0, v: v00 };
          const t1: Vertex = { x: x1, y: y0, v: v10 };
          const t2: Vertex = { x: x1, y: y1, v: v11 };
          const t3: Vertex = { x: x0, y: y1, v: v01 };

          addBandTriangles([t0, t1, t2], this.levels, triangles);
          addBandTriangles([t0, t2, t3], this.levels, triangles);
        }
      }
    }

    this.triangles = triangles;
  }

  getDataRange(): DataRange | null {
    if (!Number.isFinite(this.xmin) || !Number.isFinite(this.xmax)) return null;
    if (!Number.isFinite(this.ymin) || !Number.isFinite(this.ymax)) return null;
    return {
      xmin: this.xmin,
      xmax: this.xmax,
      ymin: this.ymin,
      ymax: this.ymax,
    };
  }

  drawSVG(ctx: SvgDrawContext): void {
    if (this.triangles.length === 0) return;
    for (const tri of this.triangles) {
      const color = this.bandColors[tri.band] ?? "#1f77b4";
      const x1 = ctx.transform.xToPx(tri.x1);
      const y1 = ctx.transform.yToPx(tri.y1);
      const x2 = ctx.transform.xToPx(tri.x2);
      const y2 = ctx.transform.yToPx(tri.y2);
      const x3 = ctx.transform.xToPx(tri.x3);
      const y3 = ctx.transform.yToPx(tri.y3);
      ctx.push(
        `<path d="M ${x1.toFixed(2)} ${y1.toFixed(2)} L ${x2.toFixed(2)} ${y2.toFixed(
          2
        )} L ${x3.toFixed(2)} ${y3.toFixed(2)} Z" fill="${escapeXml(color)}" />`
      );
    }
  }

  drawRaster(ctx: RasterDrawContext): void {
    if (this.triangles.length === 0) return;
    for (const tri of this.triangles) {
      const rgba = this.bandRGBA[tri.band] ?? { r: 0, g: 0, b: 0, a: 255 };
      ctx.canvas.fillTriangleRGBA(
        Math.round(ctx.transform.xToPx(tri.x1)),
        Math.round(ctx.transform.yToPx(tri.y1)),
        Math.round(ctx.transform.xToPx(tri.x2)),
        Math.round(ctx.transform.yToPx(tri.y2)),
        Math.round(ctx.transform.xToPx(tri.x3)),
        Math.round(ctx.transform.yToPx(tri.y3)),
        rgba.r,
        rgba.g,
        rgba.b,
        rgba.a
      );
    }
  }

  getLegendEntries(): readonly LegendEntry[] | null {
    const color = this.bandColors[Math.floor(this.bandColors.length / 2)] ?? "#1f77b4";
    const entry = buildLegendEntry(this.label, { color, shape: "box" });
    return entry ? [entry] : null;
  }
}
