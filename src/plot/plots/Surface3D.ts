/**
 * 3D plotting drawables: Surface, Wireframe, and Scatter3D.
 *
 * Uses orthographic projection with configurable elevation and azimuth
 * angles to render 3D data as 2D SVG/raster output.
 *
 * @module plot/plots/Surface3D
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
import { normalizeLegendLabel } from "../utils/legend";
import { escapeXml } from "../utils/xml";

type Vec3 = { x: number; y: number; z: number };
type Vec2 = { x: number; y: number };

/**
 * Project a 3D point to 2D using rotation by elevation and azimuth.
 */
function project(p: Vec3, elev: number, azim: number): Vec2 {
  const cosE = Math.cos(elev);
  const sinE = Math.sin(elev);
  const cosA = Math.cos(azim);
  const sinA = Math.sin(azim);

  // Rotate around Z axis (azimuth), then tilt (elevation)
  const x1 = p.x * cosA - p.y * sinA;
  const y1 = p.x * sinA + p.y * cosA;
  const z1 = p.z;

  const xProj = x1;
  const yProj = -z1 * cosE + y1 * sinE;

  return { x: xProj, y: yProj };
}

function normalizeGrid(
  xGrid: Float64Array[],
  yGrid: Float64Array[],
  zGrid: Float64Array[]
): {
  rows: number;
  cols: number;
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
  zMin: number;
  zMax: number;
} {
  const rows = xGrid.length;
  if (rows === 0) throw new ShapeError("3D grid must have at least 1 row");
  const cols = xGrid[0]!.length;

  let xMin = Infinity;
  let xMax = -Infinity;
  let yMin = Infinity;
  let yMax = -Infinity;
  let zMin = Infinity;
  let zMax = -Infinity;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = xGrid[r]![c] ?? 0;
      const y = yGrid[r]![c] ?? 0;
      const z = zGrid[r]![c] ?? 0;
      if (x < xMin) xMin = x;
      if (x > xMax) xMax = x;
      if (y < yMin) yMin = y;
      if (y > yMax) yMax = y;
      if (z < zMin) zMin = z;
      if (z > zMax) zMax = z;
    }
  }

  return { rows, cols, xMin, xMax, yMin, yMax, zMin, zMax };
}

function normalizePoint(
  x: number,
  y: number,
  z: number,
  xMin: number,
  xMax: number,
  yMin: number,
  yMax: number,
  zMin: number,
  zMax: number
): Vec3 {
  const xSpan = xMax - xMin || 1;
  const ySpan = yMax - yMin || 1;
  const zSpan = zMax - zMin || 1;
  return {
    x: (x - xMin) / xSpan - 0.5,
    y: (y - yMin) / ySpan - 0.5,
    z: (z - zMin) / zSpan - 0.5,
  };
}

function zToColor(
  z: number,
  zMin: number,
  zMax: number,
  baseColor: { r: number; g: number; b: number }
): string {
  const t = zMax > zMin ? (z - zMin) / (zMax - zMin) : 0.5;
  const brightness = 0.3 + t * 0.7;
  const r = Math.round(baseColor.r * brightness);
  const g = Math.round(baseColor.g * brightness);
  const b = Math.round(baseColor.b * brightness);
  return `rgb(${r},${g},${b})`;
}

/** Rendering options for 3D plot elements. */
export type Plot3DOptions = {
  readonly label?: string;
  readonly color?: Color;
  readonly linewidth?: number;
  readonly size?: number;
  readonly elevation?: number;
  readonly azimuth?: number;
  readonly alpha?: number;
};

/**
 * 3D Surface plot drawable.
 * @internal
 */
export class Surface3D implements Drawable {
  readonly kind = "surface3d";
  private readonly xGrid: Float64Array[];
  private readonly yGrid: Float64Array[];
  private readonly zGrid: Float64Array[];
  private readonly color: Color;
  private readonly label: string | null;
  private readonly elevation: number;
  private readonly azimuth: number;
  private readonly alpha: number;

  constructor(
    xGrid: Float64Array[],
    yGrid: Float64Array[],
    zGrid: Float64Array[],
    options: Plot3DOptions = {}
  ) {
    if (xGrid.length !== yGrid.length || xGrid.length !== zGrid.length) {
      throw new ShapeError("xGrid, yGrid, zGrid must have the same number of rows");
    }
    this.xGrid = xGrid;
    this.yGrid = yGrid;
    this.zGrid = zGrid;
    this.color = normalizeColor(options.color, "#1f77b4");
    this.label = normalizeLegendLabel(options.label);
    this.elevation = ((options.elevation ?? 30) * Math.PI) / 180;
    this.azimuth = ((options.azimuth ?? -60) * Math.PI) / 180;
    this.alpha = options.alpha ?? 0.8;
  }

  getDataRange(): DataRange | null {
    const info = normalizeGrid(this.xGrid, this.yGrid, this.zGrid);
    const projected: Vec2[] = [];
    for (let r = 0; r < info.rows; r++) {
      for (let c = 0; c < info.cols; c++) {
        const p = normalizePoint(
          this.xGrid[r]![c] ?? 0,
          this.yGrid[r]![c] ?? 0,
          this.zGrid[r]![c] ?? 0,
          info.xMin,
          info.xMax,
          info.yMin,
          info.yMax,
          info.zMin,
          info.zMax
        );
        projected.push(project(p, this.elevation, this.azimuth));
      }
    }
    let xmin = Infinity;
    let xmax = -Infinity;
    let ymin = Infinity;
    let ymax = -Infinity;
    for (const pt of projected) {
      if (pt.x < xmin) xmin = pt.x;
      if (pt.x > xmax) xmax = pt.x;
      if (pt.y < ymin) ymin = pt.y;
      if (pt.y > ymax) ymax = pt.y;
    }
    return { xmin, xmax, ymin, ymax };
  }

  drawSVG(ctx: SvgDrawContext): void {
    const info = normalizeGrid(this.xGrid, this.yGrid, this.zGrid);
    const baseRGB = parseHexColorToRGBA(this.color);

    // Draw filled quads from back to front (painter's algorithm)
    type Quad = { points: Vec2[]; z: number; fillColor: string };
    const quads: Quad[] = [];

    for (let r = 0; r < info.rows - 1; r++) {
      for (let c = 0; c < info.cols - 1; c++) {
        const corners = [
          { r, c },
          { r, c: c + 1 },
          { r: r + 1, c: c + 1 },
          { r: r + 1, c },
        ];
        const pts3d = corners.map((cr) =>
          normalizePoint(
            this.xGrid[cr.r]![cr.c] ?? 0,
            this.yGrid[cr.r]![cr.c] ?? 0,
            this.zGrid[cr.r]![cr.c] ?? 0,
            info.xMin,
            info.xMax,
            info.yMin,
            info.yMax,
            info.zMin,
            info.zMax
          )
        );
        const pts2d = pts3d.map((p) => project(p, this.elevation, this.azimuth));
        const avgZ =
          ((this.zGrid[r]![c] ?? 0) +
            (this.zGrid[r]![c + 1] ?? 0) +
            (this.zGrid[r + 1]![c + 1] ?? 0) +
            (this.zGrid[r + 1]![c] ?? 0)) /
          4;
        const fillColor = zToColor(avgZ, info.zMin, info.zMax, baseRGB);
        const screenPts = pts2d.map((p) => ({
          x: ctx.transform.xToPx(p.x),
          y: ctx.transform.yToPx(p.y),
        }));
        quads.push({ points: screenPts, z: avgZ, fillColor });
      }
    }

    // Sort by depth (painter's algorithm)
    quads.sort((a, b) => a.z - b.z);

    for (const quad of quads) {
      const pathData = quad.points
        .map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`)
        .join(" ");
      ctx.push(
        `<path d="${pathData} Z" fill="${escapeXml(quad.fillColor)}" stroke="#333" stroke-width="0.5" opacity="${this.alpha}" />`
      );
    }
  }

  drawRaster(ctx: RasterDrawContext): void {
    // Raster fallback: draw wireframe only
    const info = normalizeGrid(this.xGrid, this.yGrid, this.zGrid);
    const baseRGB = parseHexColorToRGBA(this.color);

    for (let r = 0; r < info.rows; r++) {
      for (let c = 0; c < info.cols - 1; c++) {
        const p1 = normalizePoint(
          this.xGrid[r]![c] ?? 0,
          this.yGrid[r]![c] ?? 0,
          this.zGrid[r]![c] ?? 0,
          info.xMin,
          info.xMax,
          info.yMin,
          info.yMax,
          info.zMin,
          info.zMax
        );
        const p2 = normalizePoint(
          this.xGrid[r]![c + 1] ?? 0,
          this.yGrid[r]![c + 1] ?? 0,
          this.zGrid[r]![c + 1] ?? 0,
          info.xMin,
          info.xMax,
          info.yMin,
          info.yMax,
          info.zMin,
          info.zMax
        );
        const s1 = project(p1, this.elevation, this.azimuth);
        const s2 = project(p2, this.elevation, this.azimuth);
        ctx.canvas.drawLineRGBA(
          Math.round(ctx.transform.xToPx(s1.x)),
          Math.round(ctx.transform.yToPx(s1.y)),
          Math.round(ctx.transform.xToPx(s2.x)),
          Math.round(ctx.transform.yToPx(s2.y)),
          baseRGB.r,
          baseRGB.g,
          baseRGB.b,
          255
        );
      }
    }
    for (let c = 0; c < info.cols; c++) {
      for (let r = 0; r < info.rows - 1; r++) {
        const p1 = normalizePoint(
          this.xGrid[r]![c] ?? 0,
          this.yGrid[r]![c] ?? 0,
          this.zGrid[r]![c] ?? 0,
          info.xMin,
          info.xMax,
          info.yMin,
          info.yMax,
          info.zMin,
          info.zMax
        );
        const p2 = normalizePoint(
          this.xGrid[r + 1]![c] ?? 0,
          this.yGrid[r + 1]![c] ?? 0,
          this.zGrid[r + 1]![c] ?? 0,
          info.xMin,
          info.xMax,
          info.yMin,
          info.yMax,
          info.zMin,
          info.zMax
        );
        const s1 = project(p1, this.elevation, this.azimuth);
        const s2 = project(p2, this.elevation, this.azimuth);
        ctx.canvas.drawLineRGBA(
          Math.round(ctx.transform.xToPx(s1.x)),
          Math.round(ctx.transform.yToPx(s1.y)),
          Math.round(ctx.transform.xToPx(s2.x)),
          Math.round(ctx.transform.yToPx(s2.y)),
          baseRGB.r,
          baseRGB.g,
          baseRGB.b,
          255
        );
      }
    }
  }

  getLegendEntries(): readonly LegendEntry[] | null {
    if (!this.label) return null;
    return [{ label: this.label, color: this.color, shape: "box" }];
  }
}

/**
 * 3D Wireframe plot drawable.
 * @internal
 */
export class Wireframe3D implements Drawable {
  readonly kind = "wireframe3d";
  private readonly xGrid: Float64Array[];
  private readonly yGrid: Float64Array[];
  private readonly zGrid: Float64Array[];
  private readonly color: Color;
  private readonly linewidth: number;
  private readonly label: string | null;
  private readonly elevation: number;
  private readonly azimuth: number;

  constructor(
    xGrid: Float64Array[],
    yGrid: Float64Array[],
    zGrid: Float64Array[],
    options: Plot3DOptions = {}
  ) {
    this.xGrid = xGrid;
    this.yGrid = yGrid;
    this.zGrid = zGrid;
    this.color = normalizeColor(options.color, "#333333");
    this.linewidth = options.linewidth ?? 1;
    this.label = normalizeLegendLabel(options.label);
    this.elevation = ((options.elevation ?? 30) * Math.PI) / 180;
    this.azimuth = ((options.azimuth ?? -60) * Math.PI) / 180;
  }

  getDataRange(): DataRange | null {
    const info = normalizeGrid(this.xGrid, this.yGrid, this.zGrid);
    const projected: Vec2[] = [];
    for (let r = 0; r < info.rows; r++) {
      for (let c = 0; c < info.cols; c++) {
        const p = normalizePoint(
          this.xGrid[r]![c] ?? 0,
          this.yGrid[r]![c] ?? 0,
          this.zGrid[r]![c] ?? 0,
          info.xMin,
          info.xMax,
          info.yMin,
          info.yMax,
          info.zMin,
          info.zMax
        );
        projected.push(project(p, this.elevation, this.azimuth));
      }
    }
    let xmin = Infinity;
    let xmax = -Infinity;
    let ymin = Infinity;
    let ymax = -Infinity;
    for (const pt of projected) {
      if (pt.x < xmin) xmin = pt.x;
      if (pt.x > xmax) xmax = pt.x;
      if (pt.y < ymin) ymin = pt.y;
      if (pt.y > ymax) ymax = pt.y;
    }
    return { xmin, xmax, ymin, ymax };
  }

  drawSVG(ctx: SvgDrawContext): void {
    const info = normalizeGrid(this.xGrid, this.yGrid, this.zGrid);
    const escapedColor = escapeXml(this.color);

    // Draw row lines
    for (let r = 0; r < info.rows; r++) {
      const pts: string[] = [];
      for (let c = 0; c < info.cols; c++) {
        const p = normalizePoint(
          this.xGrid[r]![c] ?? 0,
          this.yGrid[r]![c] ?? 0,
          this.zGrid[r]![c] ?? 0,
          info.xMin,
          info.xMax,
          info.yMin,
          info.yMax,
          info.zMin,
          info.zMax
        );
        const s = project(p, this.elevation, this.azimuth);
        const px = ctx.transform.xToPx(s.x);
        const py = ctx.transform.yToPx(s.y);
        pts.push(`${px.toFixed(1)},${py.toFixed(1)}`);
      }
      ctx.push(
        `<polyline points="${pts.join(" ")}" fill="none" stroke="${escapedColor}" stroke-width="${this.linewidth}" />`
      );
    }

    // Draw column lines
    for (let c = 0; c < info.cols; c++) {
      const pts: string[] = [];
      for (let r = 0; r < info.rows; r++) {
        const p = normalizePoint(
          this.xGrid[r]![c] ?? 0,
          this.yGrid[r]![c] ?? 0,
          this.zGrid[r]![c] ?? 0,
          info.xMin,
          info.xMax,
          info.yMin,
          info.yMax,
          info.zMin,
          info.zMax
        );
        const s = project(p, this.elevation, this.azimuth);
        const px = ctx.transform.xToPx(s.x);
        const py = ctx.transform.yToPx(s.y);
        pts.push(`${px.toFixed(1)},${py.toFixed(1)}`);
      }
      ctx.push(
        `<polyline points="${pts.join(" ")}" fill="none" stroke="${escapedColor}" stroke-width="${this.linewidth}" />`
      );
    }
  }

  drawRaster(ctx: RasterDrawContext): void {
    const info = normalizeGrid(this.xGrid, this.yGrid, this.zGrid);
    const rgb = parseHexColorToRGBA(this.color);

    for (let r = 0; r < info.rows; r++) {
      for (let c = 0; c < info.cols - 1; c++) {
        const p1 = normalizePoint(
          this.xGrid[r]![c] ?? 0,
          this.yGrid[r]![c] ?? 0,
          this.zGrid[r]![c] ?? 0,
          info.xMin,
          info.xMax,
          info.yMin,
          info.yMax,
          info.zMin,
          info.zMax
        );
        const p2 = normalizePoint(
          this.xGrid[r]![c + 1] ?? 0,
          this.yGrid[r]![c + 1] ?? 0,
          this.zGrid[r]![c + 1] ?? 0,
          info.xMin,
          info.xMax,
          info.yMin,
          info.yMax,
          info.zMin,
          info.zMax
        );
        const s1 = project(p1, this.elevation, this.azimuth);
        const s2 = project(p2, this.elevation, this.azimuth);
        ctx.canvas.drawLineRGBA(
          Math.round(ctx.transform.xToPx(s1.x)),
          Math.round(ctx.transform.yToPx(s1.y)),
          Math.round(ctx.transform.xToPx(s2.x)),
          Math.round(ctx.transform.yToPx(s2.y)),
          rgb.r,
          rgb.g,
          rgb.b,
          255
        );
      }
    }
    for (let c = 0; c < info.cols; c++) {
      for (let r = 0; r < info.rows - 1; r++) {
        const p1 = normalizePoint(
          this.xGrid[r]![c] ?? 0,
          this.yGrid[r]![c] ?? 0,
          this.zGrid[r]![c] ?? 0,
          info.xMin,
          info.xMax,
          info.yMin,
          info.yMax,
          info.zMin,
          info.zMax
        );
        const p2 = normalizePoint(
          this.xGrid[r + 1]![c] ?? 0,
          this.yGrid[r + 1]![c] ?? 0,
          this.zGrid[r + 1]![c] ?? 0,
          info.xMin,
          info.xMax,
          info.yMin,
          info.yMax,
          info.zMin,
          info.zMax
        );
        const s1 = project(p1, this.elevation, this.azimuth);
        const s2 = project(p2, this.elevation, this.azimuth);
        ctx.canvas.drawLineRGBA(
          Math.round(ctx.transform.xToPx(s1.x)),
          Math.round(ctx.transform.yToPx(s1.y)),
          Math.round(ctx.transform.xToPx(s2.x)),
          Math.round(ctx.transform.yToPx(s2.y)),
          rgb.r,
          rgb.g,
          rgb.b,
          255
        );
      }
    }
  }

  getLegendEntries(): readonly LegendEntry[] | null {
    if (!this.label) return null;
    return [{ label: this.label, color: this.color, shape: "line" }];
  }
}

/**
 * 3D Scatter plot drawable.
 * @internal
 */
export class Scatter3D implements Drawable {
  readonly kind = "scatter3d";
  private readonly x: Float64Array;
  private readonly y: Float64Array;
  private readonly z: Float64Array;
  private readonly color: Color;
  private readonly size: number;
  private readonly label: string | null;
  private readonly elevation: number;
  private readonly azimuth: number;

  constructor(x: Float64Array, y: Float64Array, z: Float64Array, options: Plot3DOptions = {}) {
    if (x.length !== y.length || x.length !== z.length) {
      throw new ShapeError("x, y, z must have the same length");
    }
    this.x = x;
    this.y = y;
    this.z = z;
    this.color = normalizeColor(options.color, "#ff7f0e");
    this.size = options.size ?? 4;
    this.label = normalizeLegendLabel(options.label);
    this.elevation = ((options.elevation ?? 30) * Math.PI) / 180;
    this.azimuth = ((options.azimuth ?? -60) * Math.PI) / 180;

    if (this.size <= 0) {
      throw new InvalidParameterError("size must be > 0", "size", this.size);
    }
  }

  getDataRange(): DataRange | null {
    let xMin = Infinity;
    let xMax = -Infinity;
    let yMin = Infinity;
    let yMax = -Infinity;
    let zMin = Infinity;
    let zMax = -Infinity;
    for (let i = 0; i < this.x.length; i++) {
      const xi = this.x[i] ?? 0;
      const yi = this.y[i] ?? 0;
      const zi = this.z[i] ?? 0;
      if (xi < xMin) xMin = xi;
      if (xi > xMax) xMax = xi;
      if (yi < yMin) yMin = yi;
      if (yi > yMax) yMax = yi;
      if (zi < zMin) zMin = zi;
      if (zi > zMax) zMax = zi;
    }

    let pxmin = Infinity;
    let pxmax = -Infinity;
    let pymin = Infinity;
    let pymax = -Infinity;
    for (let i = 0; i < this.x.length; i++) {
      const p = normalizePoint(
        this.x[i] ?? 0,
        this.y[i] ?? 0,
        this.z[i] ?? 0,
        xMin,
        xMax,
        yMin,
        yMax,
        zMin,
        zMax
      );
      const s = project(p, this.elevation, this.azimuth);
      if (s.x < pxmin) pxmin = s.x;
      if (s.x > pxmax) pxmax = s.x;
      if (s.y < pymin) pymin = s.y;
      if (s.y > pymax) pymax = s.y;
    }
    return { xmin: pxmin, xmax: pxmax, ymin: pymin, ymax: pymax };
  }

  drawSVG(ctx: SvgDrawContext): void {
    let xMin = Infinity;
    let xMax = -Infinity;
    let yMin = Infinity;
    let yMax = -Infinity;
    let zMin = Infinity;
    let zMax = -Infinity;
    for (let i = 0; i < this.x.length; i++) {
      const xi = this.x[i] ?? 0;
      const yi = this.y[i] ?? 0;
      const zi = this.z[i] ?? 0;
      if (xi < xMin) xMin = xi;
      if (xi > xMax) xMax = xi;
      if (yi < yMin) yMin = yi;
      if (yi > yMax) yMax = yi;
      if (zi < zMin) zMin = zi;
      if (zi > zMax) zMax = zi;
    }

    const escapedColor = escapeXml(this.color);
    for (let i = 0; i < this.x.length; i++) {
      const p = normalizePoint(
        this.x[i] ?? 0,
        this.y[i] ?? 0,
        this.z[i] ?? 0,
        xMin,
        xMax,
        yMin,
        yMax,
        zMin,
        zMax
      );
      const s = project(p, this.elevation, this.azimuth);
      const px = ctx.transform.xToPx(s.x);
      const py = ctx.transform.yToPx(s.y);
      ctx.push(
        `<circle cx="${px.toFixed(1)}" cy="${py.toFixed(1)}" r="${this.size}" fill="${escapedColor}" />`
      );
    }
  }

  drawRaster(ctx: RasterDrawContext): void {
    let xMin = Infinity;
    let xMax = -Infinity;
    let yMin = Infinity;
    let yMax = -Infinity;
    let zMin = Infinity;
    let zMax = -Infinity;
    for (let i = 0; i < this.x.length; i++) {
      const xi = this.x[i] ?? 0;
      const yi = this.y[i] ?? 0;
      const zi = this.z[i] ?? 0;
      if (xi < xMin) xMin = xi;
      if (xi > xMax) xMax = xi;
      if (yi < yMin) yMin = yi;
      if (yi > yMax) yMax = yi;
      if (zi < zMin) zMin = zi;
      if (zi > zMax) zMax = zi;
    }

    const rgb = parseHexColorToRGBA(this.color);
    for (let i = 0; i < this.x.length; i++) {
      const p = normalizePoint(
        this.x[i] ?? 0,
        this.y[i] ?? 0,
        this.z[i] ?? 0,
        xMin,
        xMax,
        yMin,
        yMax,
        zMin,
        zMax
      );
      const s = project(p, this.elevation, this.azimuth);
      const px = Math.round(ctx.transform.xToPx(s.x));
      const py = Math.round(ctx.transform.yToPx(s.y));
      ctx.canvas.drawCircleRGBA(px, py, Math.round(this.size), rgb.r, rgb.g, rgb.b, 255);
    }
  }

  getLegendEntries(): readonly LegendEntry[] | null {
    if (!this.label) return null;
    return [{ label: this.label, color: this.color, shape: "marker" }];
  }
}
