/**
 * 3D plotting drawables: Surface, Wireframe, and Scatter3D.
 *
 * Uses orthographic projection with configurable elevation and azimuth
 * angles to render 3D data as 2D SVG/raster output. The camera follows the Matplotlib
 * convention: the viewer sits in the direction `(cos(el)cos(az), cos(el)sin(az), sin(el))`
 * from the origin and looks at it with the z axis pointing up, so `azimuth = -60` and
 * `elevation = 30` (the defaults) reproduce Matplotlib's default 3D view.
 *
 * Each drawable scales its own x, y and z extents to a unit cube before projecting, so
 * several 3D drawables on one axes do not share a coordinate system.
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

/** Rendering options for 3D plot elements. */
export type Plot3DOptions = {
  /** Legend label. */
  readonly label?: string;
  /** Surface base color, wireframe line color or scatter marker color. */
  readonly color?: Color;
  /** Wireframe line width in pixels (must be positive). */
  readonly linewidth?: number;
  /** Scatter marker radius in pixels (must be positive). */
  readonly size?: number;
  /** Camera elevation above the xy plane in degrees (default 30). */
  readonly elevation?: number;
  /** Camera azimuth in degrees, measured from the +x axis towards +y (default -60). */
  readonly azimuth?: number;
  /** Opacity from 0 to 1. Defaults to 0.8 for surfaces; other 3D plots are opaque by default. */
  readonly alpha?: number;
};

type View = {
  readonly cosE: number;
  readonly sinE: number;
  readonly cosA: number;
  readonly sinA: number;
};

function resolveAngle(value: number | undefined, fallback: number, name: string): number {
  const deg = value ?? fallback;
  if (!Number.isFinite(deg)) {
    throw new InvalidParameterError(`${name} must be a finite number; received ${deg}`, name, deg);
  }
  return deg;
}

function makeView(options: Plot3DOptions): View {
  const elev = (resolveAngle(options.elevation, 30, "elevation") * Math.PI) / 180;
  const azim = (resolveAngle(options.azimuth, -60, "azimuth") * Math.PI) / 180;
  return { cosE: Math.cos(elev), sinE: Math.sin(elev), cosA: Math.cos(azim), sinA: Math.sin(azim) };
}

function resolveAlpha(alpha: number | undefined, fallback: number): number {
  const a = alpha ?? fallback;
  if (!Number.isFinite(a) || a < 0 || a > 1) {
    throw new InvalidParameterError(`alpha must be between 0 and 1; received ${a}`, "alpha", a);
  }
  return a;
}

function resolvePositive(value: number | undefined, fallback: number, name: string): number {
  const v = value ?? fallback;
  if (!Number.isFinite(v) || v <= 0) {
    throw new InvalidParameterError(`${name} must be > 0; received ${v}`, name, v);
  }
  return v;
}

/**
 * Projected, depth-annotated points shared by the three drawables.
 *
 * `right`/`up` are the orthographic screen coordinates (up is positive), `depth` grows towards
 * the viewer, and `valid` is 0 for points with a non-finite coordinate.
 */
type Cloud = {
  readonly count: number;
  readonly right: Float64Array;
  readonly up: Float64Array;
  readonly depth: Float64Array;
  readonly z: Float64Array;
  readonly valid: Uint8Array;
  readonly validCount: number;
  readonly zMin: number;
  readonly zMax: number;
};

function buildCloud(x: ArrayLike<number>, y: ArrayLike<number>, z: ArrayLike<number>, view: View) {
  const count = x.length;
  const valid = new Uint8Array(count);
  let xMin = Infinity;
  let xMax = -Infinity;
  let yMin = Infinity;
  let yMax = -Infinity;
  let zMin = Infinity;
  let zMax = -Infinity;
  let validCount = 0;
  for (let i = 0; i < count; i++) {
    const xi = x[i] ?? Number.NaN;
    const yi = y[i] ?? Number.NaN;
    const zi = z[i] ?? Number.NaN;
    if (!Number.isFinite(xi) || !Number.isFinite(yi) || !Number.isFinite(zi)) continue;
    valid[i] = 1;
    validCount++;
    if (xi < xMin) xMin = xi;
    if (xi > xMax) xMax = xi;
    if (yi < yMin) yMin = yi;
    if (yi > yMax) yMax = yi;
    if (zi < zMin) zMin = zi;
    if (zi > zMax) zMax = zi;
  }

  // Scale each axis to [-0.5, 0.5]; a flat axis (zero span) is centered.
  const xSpan = xMax - xMin || 1;
  const ySpan = yMax - yMin || 1;
  const zSpan = zMax - zMin || 1;
  const right = new Float64Array(count);
  const up = new Float64Array(count);
  const depth = new Float64Array(count);
  const zOut = new Float64Array(count);
  const { cosE, sinE, cosA, sinA } = view;
  for (let i = 0; i < count; i++) {
    if (valid[i] === 0) continue;
    const nx = ((x[i] ?? 0) - xMin) / xSpan - 0.5;
    const ny = ((y[i] ?? 0) - yMin) / ySpan - 0.5;
    const nz = ((z[i] ?? 0) - zMin) / zSpan - 0.5;
    const horizontal = nx * cosA + ny * sinA; // distance along the viewing direction
    right[i] = -nx * sinA + ny * cosA;
    up[i] = nz * cosE - horizontal * sinE;
    depth[i] = horizontal * cosE + nz * sinE;
    zOut[i] = z[i] ?? 0;
  }
  const cloud: Cloud = { count, right, up, depth, z: zOut, valid, validCount, zMin, zMax };
  return cloud;
}

function cloudRange(cloud: Cloud): DataRange | null {
  if (cloud.validCount === 0) return null;
  let xmin = Infinity;
  let xmax = -Infinity;
  let ymin = Infinity;
  let ymax = -Infinity;
  for (let i = 0; i < cloud.count; i++) {
    if (cloud.valid[i] === 0) continue;
    const r = cloud.right[i] ?? 0;
    const u = cloud.up[i] ?? 0;
    if (r < xmin) xmin = r;
    if (r > xmax) xmax = r;
    if (u < ymin) ymin = u;
    if (u > ymax) ymax = u;
  }
  return { xmin, xmax, ymin, ymax };
}

/** Flattens row-major grids into one cloud after checking that all three have the same shape. */
function buildGridCloud(
  xGrid: readonly Float64Array[],
  yGrid: readonly Float64Array[],
  zGrid: readonly Float64Array[],
  view: View
): { readonly rows: number; readonly cols: number; readonly cloud: Cloud } {
  const rows = xGrid.length;
  if (rows !== yGrid.length || rows !== zGrid.length) {
    throw new ShapeError("xGrid, yGrid, zGrid must have the same number of rows");
  }
  if (rows === 0) throw new ShapeError("3D grid must have at least 1 row");
  const cols = xGrid[0]?.length ?? 0;
  for (let r = 0; r < rows; r++) {
    if (
      (xGrid[r]?.length ?? -1) !== cols ||
      (yGrid[r]?.length ?? -1) !== cols ||
      (zGrid[r]?.length ?? -1) !== cols
    ) {
      throw new ShapeError(
        `3D grid rows must all have ${cols} columns (the length of the first row of xGrid); ` +
          `row ${r} differs in xGrid, yGrid or zGrid`
      );
    }
  }
  const x = new Float64Array(rows * cols);
  const y = new Float64Array(rows * cols);
  const z = new Float64Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    x.set(xGrid[r] ?? [], r * cols);
    y.set(yGrid[r] ?? [], r * cols);
    z.set(zGrid[r] ?? [], r * cols);
  }
  return { rows, cols, cloud: buildCloud(x, y, z, view) };
}

/** Color with brightness scaled from 30% (lowest z) to 100% (highest z). */
function shade(
  z: number,
  zMin: number,
  zMax: number,
  base: { readonly r: number; readonly g: number; readonly b: number }
): { readonly r: number; readonly g: number; readonly b: number } {
  const t = zMax > zMin ? (z - zMin) / (zMax - zMin) : 0.5;
  const brightness = 0.3 + t * 0.7;
  return {
    r: Math.round(base.r * brightness),
    g: Math.round(base.g * brightness),
    b: Math.round(base.b * brightness),
  };
}

/** Quadrilateral cells of a grid whose four corners are all valid, sorted far to near. */
function sortedQuads(
  rows: number,
  cols: number,
  cloud: Cloud
): { readonly corners: readonly [number, number, number, number]; readonly zMean: number }[] {
  const quads: {
    corners: readonly [number, number, number, number];
    zMean: number;
    depth: number;
  }[] = [];
  for (let r = 0; r < rows - 1; r++) {
    for (let c = 0; c < cols - 1; c++) {
      const i0 = r * cols + c;
      const i1 = i0 + 1;
      const i2 = i0 + cols + 1;
      const i3 = i0 + cols;
      if (
        cloud.valid[i0] === 0 ||
        cloud.valid[i1] === 0 ||
        cloud.valid[i2] === 0 ||
        cloud.valid[i3] === 0
      ) {
        continue;
      }
      quads.push({
        corners: [i0, i1, i2, i3],
        zMean:
          ((cloud.z[i0] ?? 0) + (cloud.z[i1] ?? 0) + (cloud.z[i2] ?? 0) + (cloud.z[i3] ?? 0)) / 4,
        depth:
          ((cloud.depth[i0] ?? 0) +
            (cloud.depth[i1] ?? 0) +
            (cloud.depth[i2] ?? 0) +
            (cloud.depth[i3] ?? 0)) /
          4,
      });
    }
  }
  // Painter's algorithm: the farthest cells (smallest depth) are drawn first.
  quads.sort((a, b) => a.depth - b.depth);
  return quads;
}

/**
 * 3D Surface plot drawable.
 *
 * Draws shaded quadrilaterals far to near; cells with a non-finite corner are skipped. The grids
 * are copied when the drawable is created, so later changes to the input arrays have no effect.
 * @internal
 */
export class Surface3D implements Drawable {
  readonly kind = "surface3d";
  private readonly rows: number;
  private readonly cols: number;
  private readonly cloud: Cloud;
  private readonly color: Color;
  private readonly label: string | null;
  private readonly alpha: number;

  /**
   * @param xGrid - Row-major x coordinates, one `Float64Array` per row
   * @param yGrid - y coordinates with the same shape as `xGrid`
   * @param zGrid - Heights with the same shape as `xGrid`
   * @param options - Color, label, camera angles and opacity
   * @throws {ShapeError} If the grids are empty or differ in shape or have ragged rows.
   * @throws {InvalidParameterError} If an angle is not finite or `alpha` is outside [0, 1].
   */
  constructor(
    xGrid: Float64Array[],
    yGrid: Float64Array[],
    zGrid: Float64Array[],
    options: Plot3DOptions = {}
  ) {
    const view = makeView(options);
    const grid = buildGridCloud(xGrid, yGrid, zGrid, view);
    this.rows = grid.rows;
    this.cols = grid.cols;
    this.cloud = grid.cloud;
    this.color = normalizeColor(options.color, "#1f77b4");
    this.label = normalizeLegendLabel(options.label);
    this.alpha = resolveAlpha(options.alpha, 0.8);
  }

  getDataRange(): DataRange | null {
    return cloudRange(this.cloud);
  }

  drawSVG(ctx: SvgDrawContext): void {
    const { cloud } = this;
    const base = parseHexColorToRGBA(this.color);
    const opacity = this.alpha * (base.a / 255);

    for (const quad of sortedQuads(this.rows, this.cols, cloud)) {
      const fill = shade(quad.zMean, cloud.zMin, cloud.zMax, base);
      const pathData = quad.corners
        .map((i, k) => {
          const px = ctx.transform.xToPx(cloud.right[i] ?? 0);
          const py = ctx.transform.yToPx(cloud.up[i] ?? 0);
          return `${k === 0 ? "M" : "L"}${px.toFixed(1)},${py.toFixed(1)}`;
        })
        .join(" ");
      ctx.push(
        `<path d="${pathData} Z" fill="${escapeXml(`rgb(${fill.r},${fill.g},${fill.b})`)}" stroke="#333" stroke-width="0.5" opacity="${opacity}" />`
      );
    }
  }

  drawRaster(ctx: RasterDrawContext): void {
    const { cloud } = this;
    const base = parseHexColorToRGBA(this.color);
    const a = Math.round(this.alpha * base.a);
    const edgeAlpha = Math.round(a * 0.8);
    const xs = new Float64Array(4);
    const ys = new Float64Array(4);

    for (const quad of sortedQuads(this.rows, this.cols, cloud)) {
      const fill = shade(quad.zMean, cloud.zMin, cloud.zMax, base);
      for (let k = 0; k < 4; k++) {
        const i = quad.corners[k] ?? 0;
        xs[k] = ctx.transform.xToPx(cloud.right[i] ?? 0);
        ys[k] = ctx.transform.yToPx(cloud.up[i] ?? 0);
      }
      ctx.canvas.fillPolygonRGBA(xs, ys, fill.r, fill.g, fill.b, a);
      for (let k = 0; k < 4; k++) {
        const n = (k + 1) % 4;
        ctx.canvas.drawLineRGBA(
          Math.round(xs[k] ?? 0),
          Math.round(ys[k] ?? 0),
          Math.round(xs[n] ?? 0),
          Math.round(ys[n] ?? 0),
          0x33,
          0x33,
          0x33,
          edgeAlpha
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
 *
 * Draws the grid lines along both grid directions; a line is broken at points with a non-finite
 * coordinate. The grids are copied when the drawable is created.
 * @internal
 */
export class Wireframe3D implements Drawable {
  readonly kind = "wireframe3d";
  private readonly rows: number;
  private readonly cols: number;
  private readonly cloud: Cloud;
  private readonly color: Color;
  private readonly linewidth: number;
  private readonly label: string | null;
  private readonly alpha: number;

  /**
   * @param xGrid - Row-major x coordinates, one `Float64Array` per row
   * @param yGrid - y coordinates with the same shape as `xGrid`
   * @param zGrid - Heights with the same shape as `xGrid`
   * @param options - Color, line width, label, camera angles and opacity
   * @throws {ShapeError} If the grids are empty or differ in shape or have ragged rows.
   * @throws {InvalidParameterError} If an angle is not finite, `linewidth` is not positive or
   *   `alpha` is outside [0, 1].
   */
  constructor(
    xGrid: Float64Array[],
    yGrid: Float64Array[],
    zGrid: Float64Array[],
    options: Plot3DOptions = {}
  ) {
    const view = makeView(options);
    const grid = buildGridCloud(xGrid, yGrid, zGrid, view);
    this.rows = grid.rows;
    this.cols = grid.cols;
    this.cloud = grid.cloud;
    this.color = normalizeColor(options.color, "#333333");
    this.linewidth = resolvePositive(options.linewidth, 1, "linewidth");
    this.label = normalizeLegendLabel(options.label);
    this.alpha = resolveAlpha(options.alpha, 1);
  }

  getDataRange(): DataRange | null {
    return cloudRange(this.cloud);
  }

  /** Index lists of the maximal runs of valid points along each row, then each column. */
  private lines(): number[][] {
    const { rows, cols, cloud } = this;
    const lines: number[][] = [];
    const collect = (outer: number, inner: number, indexOf: (o: number, i: number) => number) => {
      for (let o = 0; o < outer; o++) {
        let run: number[] = [];
        for (let i = 0; i < inner; i++) {
          const idx = indexOf(o, i);
          if (cloud.valid[idx] === 1) {
            run.push(idx);
          } else {
            if (run.length > 1) lines.push(run);
            run = [];
          }
        }
        if (run.length > 1) lines.push(run);
      }
    };
    collect(rows, cols, (r, c) => r * cols + c);
    collect(cols, rows, (c, r) => r * cols + c);
    return lines;
  }

  drawSVG(ctx: SvgDrawContext): void {
    const { cloud } = this;
    const base = parseHexColorToRGBA(this.color);
    const opacity = this.alpha * (base.a / 255);
    const opacityAttr = opacity < 1 ? ` stroke-opacity="${opacity}"` : "";
    const escapedColor = escapeXml(this.color);
    for (const line of this.lines()) {
      const pts = line.map(
        (i) =>
          `${ctx.transform.xToPx(cloud.right[i] ?? 0).toFixed(1)},${ctx.transform
            .yToPx(cloud.up[i] ?? 0)
            .toFixed(1)}`
      );
      ctx.push(
        `<polyline points="${pts.join(" ")}" fill="none" stroke="${escapedColor}" stroke-width="${this.linewidth}"${opacityAttr} />`
      );
    }
  }

  drawRaster(ctx: RasterDrawContext): void {
    const { cloud } = this;
    const rgb = parseHexColorToRGBA(this.color);
    const a = Math.round(this.alpha * rgb.a);
    for (const line of this.lines()) {
      for (let k = 0; k + 1 < line.length; k++) {
        const i = line[k] ?? 0;
        const j = line[k + 1] ?? 0;
        ctx.canvas.drawLineRGBA(
          Math.round(ctx.transform.xToPx(cloud.right[i] ?? 0)),
          Math.round(ctx.transform.yToPx(cloud.up[i] ?? 0)),
          Math.round(ctx.transform.xToPx(cloud.right[j] ?? 0)),
          Math.round(ctx.transform.yToPx(cloud.up[j] ?? 0)),
          rgb.r,
          rgb.g,
          rgb.b,
          a
        );
      }
    }
  }

  getLegendEntries(): readonly LegendEntry[] | null {
    if (!this.label) return null;
    return [{ label: this.label, color: this.color, shape: "line", lineWidth: this.linewidth }];
  }
}

/**
 * 3D Scatter plot drawable.
 *
 * Points are drawn far to near; points with a non-finite coordinate are skipped. The coordinates
 * are copied when the drawable is created.
 * @internal
 */
export class Scatter3D implements Drawable {
  readonly kind = "scatter3d";
  private readonly cloud: Cloud;
  private readonly color: Color;
  private readonly size: number;
  private readonly label: string | null;
  private readonly alpha: number;

  /**
   * @param x - x coordinates
   * @param y - y coordinates, same length as `x`
   * @param z - z coordinates, same length as `x`
   * @param options - Color, marker radius `size`, label, camera angles and opacity
   * @throws {ShapeError} If `x`, `y` and `z` differ in length.
   * @throws {InvalidParameterError} If an angle is not finite, `size` is not positive or `alpha`
   *   is outside [0, 1].
   */
  constructor(x: Float64Array, y: Float64Array, z: Float64Array, options: Plot3DOptions = {}) {
    if (x.length !== y.length || x.length !== z.length) {
      throw new ShapeError(
        `x, y, z must have the same length; received ${x.length}, ${y.length}, ${z.length}`
      );
    }
    const view = makeView(options);
    this.size = resolvePositive(options.size, 4, "size");
    this.alpha = resolveAlpha(options.alpha, 1);
    this.color = normalizeColor(options.color, "#ff7f0e");
    this.label = normalizeLegendLabel(options.label);
    this.cloud = buildCloud(x, y, z, view);
  }

  getDataRange(): DataRange | null {
    return cloudRange(this.cloud);
  }

  /** Indices of the valid points, farthest first. */
  private order(): number[] {
    const { cloud } = this;
    const idx: number[] = [];
    for (let i = 0; i < cloud.count; i++) if (cloud.valid[i] === 1) idx.push(i);
    idx.sort((a, b) => (cloud.depth[a] ?? 0) - (cloud.depth[b] ?? 0));
    return idx;
  }

  drawSVG(ctx: SvgDrawContext): void {
    const { cloud } = this;
    const base = parseHexColorToRGBA(this.color);
    const opacity = this.alpha * (base.a / 255);
    const opacityAttr = opacity < 1 ? ` fill-opacity="${opacity}"` : "";
    const escapedColor = escapeXml(this.color);
    for (const i of this.order()) {
      const px = ctx.transform.xToPx(cloud.right[i] ?? 0);
      const py = ctx.transform.yToPx(cloud.up[i] ?? 0);
      ctx.push(
        `<circle cx="${px.toFixed(1)}" cy="${py.toFixed(1)}" r="${this.size}" fill="${escapedColor}"${opacityAttr} />`
      );
    }
  }

  drawRaster(ctx: RasterDrawContext): void {
    const { cloud } = this;
    const rgb = parseHexColorToRGBA(this.color);
    const a = Math.round(this.alpha * rgb.a);
    for (const i of this.order()) {
      const px = Math.round(ctx.transform.xToPx(cloud.right[i] ?? 0));
      const py = Math.round(ctx.transform.yToPx(cloud.up[i] ?? 0));
      ctx.canvas.drawCircleRGBA(px, py, this.size, rgb.r, rgb.g, rgb.b, a);
    }
  }

  getLegendEntries(): readonly LegendEntry[] | null {
    if (!this.label) return null;
    return [{ label: this.label, color: this.color, shape: "marker", markerSize: this.size }];
  }
}
