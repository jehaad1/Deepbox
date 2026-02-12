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

type Segment = {
  readonly x1: number;
  readonly y1: number;
  readonly x2: number;
  readonly y2: number;
  readonly levelIndex: number;
};

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

function resolveLevels(
  min: number,
  max: number,
  levels: number | readonly number[] | undefined
): number[] {
  if (Array.isArray(levels)) {
    const filtered = levels.filter((v) => Number.isFinite(v));
    if (filtered.length === 0) {
      throw new InvalidParameterError(
        "levels must contain at least one finite value",
        "levels",
        levels
      );
    }
    const unique = Array.from(new Set(filtered));
    unique.sort((a, b) => a - b);
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
  if (numLevels === 1) {
    return [min === max ? min : (min + max) / 2];
  }
  const step = (max - min) / (numLevels - 1);
  const values = new Array<number>(numLevels);
  for (let i = 0; i < numLevels; i++) {
    values[i] = min + i * step;
  }
  return values;
}

function resolveLevelColors(
  levels: readonly number[],
  options: PlotOptions
): {
  readonly colors: readonly Color[];
  readonly rgba: readonly { r: number; g: number; b: number; a: number }[];
} {
  const count = levels.length;
  const colors: Color[] = [];

  if (options.colors && options.colors.length > 0) {
    for (let i = 0; i < count; i++) {
      const color = options.colors[i % options.colors.length];
      colors.push(normalizeColor(color, DEFAULT_COLORS[i % DEFAULT_COLORS.length] ?? "#1f77b4"));
    }
  } else if (options.color) {
    const normalized = normalizeColor(options.color, "#1f77b4");
    for (let i = 0; i < count; i++) colors.push(normalized);
  } else if (options.colormap) {
    if (!["viridis", "plasma", "inferno", "magma", "grayscale"].includes(options.colormap)) {
      throw new InvalidParameterError(
        `colormap must be one of viridis, plasma, inferno, magma, grayscale; received ${options.colormap}`,
        "colormap",
        options.colormap
      );
    }
    const denom = Math.max(1, count - 1);
    for (let i = 0; i < count; i++) {
      const t = i / denom;
      const [r, g, b] = applyColormap(t, options.colormap);
      colors.push(normalizeColor(`rgb(${r},${g},${b})`, "#1f77b4"));
    }
  } else {
    for (let i = 0; i < count; i++) {
      const color = DEFAULT_COLORS[i % DEFAULT_COLORS.length] ?? "#1f77b4";
      colors.push(i === 0 ? color : normalizeColor(color, "#1f77b4"));
    }
  }

  const rgba = colors.map((c) => parseHexColorToRGBA(c));
  return { colors, rgba };
}

function interpolate(level: number, v0: number, v1: number, c0: number, c1: number): number {
  if (v0 === v1) return (c0 + c1) / 2;
  const t = Math.max(0, Math.min(1, (level - v0) / (v1 - v0)));
  return c0 + t * (c1 - c0);
}

/**
 * @internal
 */
export class Contour2D implements Drawable {
  readonly kind = "contour";
  readonly segments: readonly Segment[];
  readonly levelColors: readonly Color[];
  readonly levelRGBA: readonly { r: number; g: number; b: number; a: number }[];
  readonly linewidth: number;
  readonly xmin: number;
  readonly xmax: number;
  readonly ymin: number;
  readonly ymax: number;
  readonly label: string | null;

  constructor(grid: ContourGrid, options: PlotOptions = {}) {
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

    const levels = resolveLevels(min, max, options.levels);
    const { colors, rgba } = resolveLevelColors(levels, options);
    let levelColors = colors;
    let levelRGBA = rgba;
    this.label = normalizeLegendLabel(options.label);

    const lw = options.linewidth ?? 1;
    if (!Number.isFinite(lw) || lw <= 0) {
      throw new InvalidParameterError(
        `linewidth must be a positive number; received ${lw}`,
        "linewidth",
        lw
      );
    }
    this.linewidth = lw;

    let xMin = Number.POSITIVE_INFINITY;
    let xMax = Number.NEGATIVE_INFINITY;
    let yMin = Number.POSITIVE_INFINITY;
    let yMax = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < grid.xCoords.length; i++) {
      const v = grid.xCoords[i] ?? 0;
      if (v < xMin) xMin = v;
      if (v > xMax) xMax = v;
    }
    for (let i = 0; i < grid.yCoords.length; i++) {
      const v = grid.yCoords[i] ?? 0;
      if (v < yMin) yMin = v;
      if (v > yMax) yMax = v;
    }
    this.xmin = xMin;
    this.xmax = xMax;
    this.ymin = yMin;
    this.ymax = yMax;

    let segments: Segment[] = [];
    const rows = grid.rows;
    const cols = grid.cols;
    const data = grid.data;
    const xCoords = grid.xCoords;
    const yCoords = grid.yCoords;

    if (rows >= 2 && cols >= 2) {
      for (let levelIndex = 0; levelIndex < levels.length; levelIndex++) {
        const level = levels[levelIndex] ?? 0;
        for (let i = 0; i < rows - 1; i++) {
          const y0 = yCoords[i] ?? 0;
          const y1 = yCoords[i + 1] ?? 0;
          const rowOffset = i * cols;
          const rowOffsetNext = (i + 1) * cols;
          for (let j = 0; j < cols - 1; j++) {
            const x0 = xCoords[j] ?? 0;
            const x1 = xCoords[j + 1] ?? 0;

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

            const idx =
              (v00 >= level ? 1 : 0) |
              (v10 >= level ? 2 : 0) |
              (v11 >= level ? 4 : 0) |
              (v01 >= level ? 8 : 0);

            if (idx === 0 || idx === 15) continue;

            const edgePoint = (edge: number): { readonly x: number; readonly y: number } => {
              switch (edge) {
                case 0:
                  return { x: interpolate(level, v00, v10, x0, x1), y: y0 };
                case 1:
                  return { x: x1, y: interpolate(level, v10, v11, y0, y1) };
                case 2:
                  return { x: interpolate(level, v11, v01, x1, x0), y: y1 };
                case 3:
                  return { x: x0, y: interpolate(level, v01, v00, y1, y0) };
                default:
                  return { x: x0, y: y0 };
              }
            };

            const pushSegment = (e1: number, e2: number): void => {
              const p1 = edgePoint(e1);
              const p2 = edgePoint(e2);
              segments.push({
                x1: p1.x,
                y1: p1.y,
                x2: p2.x,
                y2: p2.y,
                levelIndex,
              });
            };

            switch (idx) {
              case 1:
                pushSegment(3, 0);
                break;
              case 2:
                pushSegment(0, 1);
                break;
              case 3:
                pushSegment(3, 1);
                break;
              case 4:
                pushSegment(1, 2);
                break;
              case 5: {
                // Saddle point: disambiguate using center value
                const center = (v00 + v10 + v11 + v01) / 4;
                if (center >= level) {
                  pushSegment(3, 2);
                  pushSegment(0, 1);
                } else {
                  pushSegment(3, 0);
                  pushSegment(1, 2);
                }
                break;
              }
              case 6:
                pushSegment(0, 2);
                break;
              case 7:
                pushSegment(3, 2);
                break;
              case 8:
                pushSegment(2, 3);
                break;
              case 9:
                pushSegment(0, 2);
                break;
              case 10: {
                // Saddle point: disambiguate using center value
                const center = (v00 + v10 + v11 + v01) / 4;
                if (center >= level) {
                  pushSegment(0, 3);
                  pushSegment(2, 1);
                } else {
                  pushSegment(0, 1);
                  pushSegment(2, 3);
                }
                break;
              }
              case 11:
                pushSegment(1, 2);
                break;
              case 12:
                pushSegment(1, 3);
                break;
              case 13:
                pushSegment(0, 1);
                break;
              case 14:
                pushSegment(3, 0);
                break;
              default:
                break;
            }
          }
        }
      }
    }

    if (segments.length > 0) {
      const used = Array.from(new Set(segments.map((seg) => seg.levelIndex))).sort((a, b) => a - b);
      if (used.length > 0 && used.length < levels.length) {
        const usedLevels = used.map((idx) => levels[idx] ?? 0);
        const resolved = resolveLevelColors(usedLevels, options);
        levelColors = resolved.colors;
        levelRGBA = resolved.rgba;
        const map = new Map<number, number>();
        for (let i = 0; i < used.length; i++) {
          const idx = used[i];
          if (idx !== undefined) {
            map.set(idx, i);
          }
        }
        segments = segments.map((seg) => ({
          x1: seg.x1,
          y1: seg.y1,
          x2: seg.x2,
          y2: seg.y2,
          levelIndex: map.get(seg.levelIndex) ?? 0,
        }));
      }
    }

    this.levelColors = levelColors;
    this.levelRGBA = levelRGBA;
    this.segments = segments;
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
    for (const seg of this.segments) {
      const color = this.levelColors[seg.levelIndex] ?? "#1f77b4";
      ctx.push(
        `<line x1="${ctx.transform.xToPx(seg.x1).toFixed(2)}" y1="${ctx.transform
          .yToPx(seg.y1)
          .toFixed(2)}" x2="${ctx.transform.xToPx(seg.x2).toFixed(2)}" y2="${ctx.transform
          .yToPx(seg.y2)
          .toFixed(
            2
          )}" stroke="${escapeXml(color)}" stroke-width="${this.linewidth}" fill="none" />`
      );
    }
  }

  drawRaster(ctx: RasterDrawContext): void {
    for (const seg of this.segments) {
      const rgba = this.levelRGBA[seg.levelIndex] ?? {
        r: 0,
        g: 0,
        b: 0,
        a: 255,
      };
      ctx.canvas.drawLineRGBA(
        Math.round(ctx.transform.xToPx(seg.x1)),
        Math.round(ctx.transform.yToPx(seg.y1)),
        Math.round(ctx.transform.xToPx(seg.x2)),
        Math.round(ctx.transform.yToPx(seg.y2)),
        rgba.r,
        rgba.g,
        rgba.b,
        rgba.a
      );
    }
  }

  getLegendEntries(): readonly LegendEntry[] | null {
    const color = this.levelColors[0] ?? "#1f77b4";
    const entry = buildLegendEntry(this.label, {
      color,
      shape: "line",
      lineWidth: this.linewidth,
    });
    return entry ? [entry] : null;
  }
}
