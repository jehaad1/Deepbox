/**
 * @see {@link https://deepbox.dev/docs/plot-basic | Deepbox documentation}
 */

import { InvalidParameterError, MemoryError } from "../../core";
import { assertPositiveInt } from "../utils/validation";

/** Largest pixel buffer (in bytes) a canvas may allocate. */
const MAX_CANVAS_BYTES = 2 ** 31 - 1;
/** Segments are clipped to the drawable area expanded by this many pixels. */
const LINE_CLIP_MARGIN = 4096;
/** Upper bound on the integer glyph scale, so huge font sizes cannot stall rendering. */
const MAX_TEXT_SCALE = 64;

const LITTLE_ENDIAN = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;

function toByte(v: number): number {
  if (v >= 255) return 255;
  return v > 0 ? Math.round(v) : 0;
}

/**
 * Software RGBA raster surface used by the PNG renderer.
 *
 * Pixel writes use source-over alpha compositing, so colors with an alpha
 * below 255 blend with what is already on the canvas (matching the SVG output).
 *
 * @internal
 */
export class RasterCanvas {
  readonly width: number;
  readonly height: number;
  readonly data: Uint8ClampedArray;
  private readonly words: Uint32Array;
  // Writable pixel area: the canvas intersected with the optional clip rectangle
  // (set via setClipRect, used to keep data series inside the axes viewport).
  // Pixels outside [bx0, bx1) x [by0, by1) are discarded.
  private bx0 = 0;
  private by0 = 0;
  private bx1: number;
  private by1: number;

  constructor(width: number, height: number) {
    assertPositiveInt("width", width);
    assertPositiveInt("height", height);
    const bytes = width * height * 4;
    if (bytes > MAX_CANVAS_BYTES) {
      throw new MemoryError(
        `Canvas of ${width}x${height} pixels needs ${bytes} bytes, which exceeds the limit of ${MAX_CANVAS_BYTES} bytes`,
        { requestedBytes: bytes, availableBytes: MAX_CANVAS_BYTES }
      );
    }
    this.width = width;
    this.height = height;
    this.data = new Uint8ClampedArray(bytes);
    this.words = new Uint32Array(this.data.buffer);
    this.bx1 = width;
    this.by1 = height;
  }

  /**
   * Restrict subsequent pixel writes to `[x0,x1) x [y0,y1)`.
   * An empty or inverted rectangle discards all writes until the clip is cleared.
   */
  setClipRect(x0: number, y0: number, x1: number, y1: number): void {
    if (
      !Number.isFinite(x0) ||
      !Number.isFinite(y0) ||
      !Number.isFinite(x1) ||
      !Number.isFinite(y1)
    ) {
      throw new InvalidParameterError(
        `clip rectangle must be finite; received [${x0}, ${y0}, ${x1}, ${y1}]`,
        "clip",
        [x0, y0, x1, y1]
      );
    }
    this.bx0 = Math.max(0, Math.ceil(x0));
    this.by0 = Math.max(0, Math.ceil(y0));
    this.bx1 = Math.min(this.width, Math.ceil(x1));
    this.by1 = Math.min(this.height, Math.ceil(y1));
  }

  /** Remove any active clip rectangle. */
  clearClip(): void {
    this.bx0 = 0;
    this.by0 = 0;
    this.bx1 = this.width;
    this.by1 = this.height;
  }

  /** Overwrite every pixel (no blending, no clipping) with the given color. */
  clearRGBA(r: number, g: number, b: number, a: number): void {
    this.words.fill(this.pack(r, g, b, a));
  }

  private pack(r: number, g: number, b: number, a: number): number {
    const rr = toByte(r);
    const gg = toByte(g);
    const bb = toByte(b);
    const aa = toByte(a);
    return LITTLE_ENDIAN
      ? ((aa << 24) | (bb << 16) | (gg << 8) | rr) >>> 0
      : ((rr << 24) | (gg << 16) | (bb << 8) | aa) >>> 0;
  }

  // Source-over composite of one pixel; `idx` is the byte offset of the pixel.
  private blendAt(idx: number, r: number, g: number, b: number, a: number): void {
    const d = this.data;
    const sa = a >= 255 ? 255 : a > 0 ? a : 0;
    if (sa === 0) return;
    const da = d[idx + 3] ?? 0;
    if (sa === 255 || da === 0) {
      d[idx] = r;
      d[idx + 1] = g;
      d[idx + 2] = b;
      d[idx + 3] = sa;
      return;
    }
    const wd = (da * (255 - sa)) / 255;
    const ra = sa + wd;
    d[idx] = (r * sa + (d[idx] ?? 0) * wd) / ra;
    d[idx + 1] = (g * sa + (d[idx + 1] ?? 0) * wd) / ra;
    d[idx + 2] = (b * sa + (d[idx + 2] ?? 0) * wd) / ra;
    d[idx + 3] = ra;
  }

  /**
   * Composite one pixel. Fractional coordinates address the pixel that contains
   * them; coordinates outside the canvas or clip rectangle (or NaN) are ignored.
   */
  setPixelRGBA(x: number, y: number, r: number, g: number, b: number, a: number): void {
    const ix = Math.floor(x);
    const iy = Math.floor(y);
    if (!(ix >= this.bx0 && ix < this.bx1 && iy >= this.by0 && iy < this.by1)) return;
    this.blendAt((iy * this.width + ix) * 4, r, g, b, a);
  }

  // Fill the integer pixel run [xa, xb) on row y, clipped to the writable area.
  private fillSpan(
    y: number,
    xa: number,
    xb: number,
    r: number,
    g: number,
    b: number,
    a: number
  ): void {
    if (!(y >= this.by0 && y < this.by1)) return;
    const lo = Math.max(xa, this.bx0);
    const hi = Math.min(xb, this.bx1);
    if (!(hi > lo)) return;
    const rowStart = y * this.width;
    if (a >= 255) {
      this.words.fill(this.pack(r, g, b, 255), rowStart + lo, rowStart + hi);
      return;
    }
    for (let x = lo; x < hi; x++) this.blendAt((rowStart + x) * 4, r, g, b, a);
  }

  fillRectRGBA(
    x0: number,
    y0: number,
    w: number,
    h: number,
    r: number,
    g: number,
    b: number,
    a: number
  ): void {
    const xa = Math.round(x0);
    const xb = Math.round(x0 + w);
    const ya = Math.max(Math.round(y0), this.by0);
    const yb = Math.min(Math.round(y0 + h), this.by1);
    if (Number.isNaN(xa) || Number.isNaN(xb)) return;
    for (let y = ya; y < yb; y++) this.fillSpan(y, xa, xb, r, g, b, a);
  }

  /**
   * Draw a one pixel wide Bresenham line between two points (inclusive).
   *
   * Segments are clipped against the drawable area (plus a safety margin) before
   * rasterizing, so very distant endpoints keep their slope and cost nothing
   * extra. Segments with a non-finite endpoint are skipped.
   */
  drawLineRGBA(
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    r: number,
    g: number,
    b: number,
    a: number
  ): void {
    if (
      !Number.isFinite(x0) ||
      !Number.isFinite(y0) ||
      !Number.isFinite(x1) ||
      !Number.isFinite(y1)
    )
      return;
    if (this.bx0 >= this.bx1 || this.by0 >= this.by1) return;

    let ax = x0;
    let ay = y0;
    let bx = x1;
    let by = y1;
    const minX = this.bx0 - LINE_CLIP_MARGIN;
    const maxX = this.bx1 - 1 + LINE_CLIP_MARGIN;
    const minY = this.by0 - LINE_CLIP_MARGIN;
    const maxY = this.by1 - 1 + LINE_CLIP_MARGIN;
    const aIn = ax >= minX && ax <= maxX && ay >= minY && ay <= maxY;
    const bIn = bx >= minX && bx <= maxX && by >= minY && by <= maxY;
    if (!aIn || !bIn) {
      // Liang-Barsky clipping against the expanded box.
      const dx = x1 - x0;
      const dy = y1 - y0;
      const p = [-dx, dx, -dy, dy];
      const q = [x0 - minX, maxX - x0, y0 - minY, maxY - y0];
      let t0 = 0;
      let t1 = 1;
      for (let k = 0; k < 4; k++) {
        const pk = p[k] ?? 0;
        const qk = q[k] ?? 0;
        if (pk === 0) {
          if (qk < 0) return;
          continue;
        }
        const t = qk / pk;
        if (pk < 0) {
          if (t > t1) return;
          if (t > t0) t0 = t;
        } else {
          if (t < t0) return;
          if (t < t1) t1 = t;
        }
      }
      if (t0 > 0) {
        ax = x0 + t0 * dx;
        ay = y0 + t0 * dy;
      }
      if (t1 < 1) {
        bx = x0 + t1 * dx;
        by = y0 + t1 * dy;
      }
    }

    let x = Math.round(ax);
    let y = Math.round(ay);
    const xEnd = Math.round(bx);
    const yEnd = Math.round(by);
    const dxp = Math.abs(xEnd - x);
    const sx = x < xEnd ? 1 : -1;
    const dyp = -Math.abs(yEnd - y);
    const sy = y < yEnd ? 1 : -1;
    let err = dxp + dyp;

    for (;;) {
      this.setPixelRGBA(x, y, r, g, b, a);
      if (x === xEnd && y === yEnd) break;
      const e2 = 2 * err;
      if (e2 >= dyp) {
        err += dyp;
        x += sx;
      }
      if (e2 <= dxp) {
        err += dxp;
        y += sy;
      }
    }
  }

  fillTriangleRGBA(
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    x2: number,
    y2: number,
    r: number,
    g: number,
    b: number,
    a: number
  ): void {
    if (
      !Number.isFinite(x0) ||
      !Number.isFinite(y0) ||
      !Number.isFinite(x1) ||
      !Number.isFinite(y1) ||
      !Number.isFinite(x2) ||
      !Number.isFinite(y2)
    ) {
      return;
    }
    // Sort vertices by Y
    if (y0 > y1) {
      [x0, x1] = [x1, x0];
      [y0, y1] = [y1, y0];
    }
    if (y0 > y2) {
      [x0, x2] = [x2, x0];
      [y0, y2] = [y2, y0];
    }
    if (y1 > y2) {
      [x1, x2] = [x2, x1];
      [y1, y2] = [y2, y1];
    }

    const totalHeight = y2 - y0;
    if (totalHeight === 0) return;

    // Only visit scanlines that can land inside the writable area.
    const iStart = Math.max(0, Math.ceil(this.by0 - y0));
    const iEnd = Math.min(Math.ceil(totalHeight), Math.ceil(this.by1 - y0));

    for (let i = iStart; i < iEnd; i++) {
      const secondHalf = i > y1 - y0 || y1 === y0;
      const segmentHeight = secondHalf ? y2 - y1 : y1 - y0;
      const alpha = i / totalHeight;
      const beta = (i - (secondHalf ? y1 - y0 : 0)) / segmentHeight;

      let ax = x0 + (x2 - x0) * alpha;
      let bx = secondHalf ? x1 + (x2 - x1) * beta : x0 + (x1 - x0) * beta;

      if (ax > bx) {
        [ax, bx] = [bx, ax];
      }

      this.fillSpan(Math.floor(y0 + i), Math.floor(ax), Math.ceil(bx), r, g, b, a);
    }
  }

  /**
   * Fill a polygon given by pixel-space vertices using the even-odd rule.
   * A pixel is filled when its center lies inside the polygon. Polygons with
   * fewer than three vertices or a non-finite vertex are skipped.
   */
  fillPolygonRGBA(
    xs: ArrayLike<number>,
    ys: ArrayLike<number>,
    r: number,
    g: number,
    b: number,
    a: number
  ): void {
    const n = Math.min(xs.length, ys.length);
    if (n < 3) return;
    let minY = Number.POSITIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < n; i++) {
      const x = xs[i] ?? Number.NaN;
      const y = ys[i] ?? Number.NaN;
      if (!Number.isFinite(x) || !Number.isFinite(y)) return;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    const yStart = Math.max(this.by0, Math.ceil(minY - 0.5));
    const yEnd = Math.min(this.by1 - 1, Math.ceil(maxY - 0.5) - 1);
    if (yEnd < yStart) return;

    // Bucket the x position where each edge crosses a scanline center.
    const buckets: Array<number[] | undefined> = new Array(yEnd - yStart + 1);
    for (let i = 0, j = n - 1; i < n; j = i++) {
      const xi = xs[i] ?? 0;
      const yi = ys[i] ?? 0;
      const xj = xs[j] ?? 0;
      const yj = ys[j] ?? 0;
      if (yi === yj) continue;
      const lo = Math.min(yi, yj);
      const hi = Math.max(yi, yj);
      const first = Math.max(yStart, Math.ceil(lo - 0.5));
      const last = Math.min(yEnd, Math.ceil(hi - 0.5) - 1);
      const slope = (xj - xi) / (yj - yi);
      for (let py = first; py <= last; py++) {
        const bucket = buckets[py - yStart] ?? [];
        bucket.push(xi + (py + 0.5 - yi) * slope);
        buckets[py - yStart] = bucket;
      }
    }
    for (let py = yStart; py <= yEnd; py++) {
      const crossings = buckets[py - yStart];
      if (!crossings) continue;
      crossings.sort((p, q) => p - q);
      for (let k = 0; k + 1 < crossings.length; k += 2) {
        const xa = Math.ceil((crossings[k] ?? 0) - 0.5);
        const xb = Math.ceil((crossings[k + 1] ?? 0) - 0.5);
        this.fillSpan(py, xa, xb, r, g, b, a);
      }
    }
  }

  /** Fill a disc of the given radius (pixels whose centers lie within it). */
  drawCircleRGBA(
    cx: number,
    cy: number,
    radius: number,
    r: number,
    g: number,
    b: number,
    a: number
  ): void {
    const rr = Math.max(0, radius);
    const r2 = rr * rr;
    const x0 = Math.max(Math.floor(cx - rr), this.bx0);
    const x1 = Math.min(Math.floor(cx + rr) + 1, this.bx1);
    const y0 = Math.max(Math.floor(cy - rr), this.by0);
    const y1 = Math.min(Math.floor(cy + rr) + 1, this.by1);

    for (let y = y0; y < y1; y++) {
      const dy = y - cy;
      const rowBase = y * this.width;
      for (let x = x0; x < x1; x++) {
        const dx = x - cx;
        if (dx * dx + dy * dy <= r2) this.blendAt((rowBase + x) * 4, r, g, b, a);
      }
    }
  }

  /**
   * Size of `text` in pixels when drawn with {@link drawTextRGBA} (built-in
   * 5x7 bitmap font scaled by an integer factor derived from `fontSize`).
   */
  measureText(text: string, fontSize = 12): { readonly width: number; readonly height: number } {
    const scale = fontScale(fontSize);
    const charWidth = FONT_WIDTH * scale;
    const charHeight = FONT_HEIGHT * scale;
    const spacing = scale;
    let count = 0;
    for (const _ of text) count++;
    if (count === 0) return { width: 0, height: charHeight };
    return { width: count * charWidth + (count - 1) * spacing, height: charHeight };
  }

  /**
   * Draw text with the built-in bitmap font. Lowercase letters are drawn as
   * uppercase and characters without a glyph are drawn as "?".
   */
  drawTextRGBA(
    text: string,
    x: number,
    y: number,
    r: number,
    g: number,
    b: number,
    a: number,
    options: {
      readonly fontSize?: number;
      readonly align?: "start" | "middle" | "end";
      readonly baseline?: "top" | "middle" | "bottom";
      readonly rotation?: 0 | 90 | -90;
    } = {}
  ): void {
    if (text.length === 0) return;
    const fontSize = options.fontSize ?? 12;
    const align = options.align ?? "start";
    const baseline = options.baseline ?? "top";
    const rotation = options.rotation ?? 0;
    if (rotation !== 0 && rotation !== 90 && rotation !== -90) {
      throw new InvalidParameterError(
        `rotation must be 0, 90 or -90; received ${rotation}`,
        "rotation",
        rotation
      );
    }
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;

    const chars = Array.from(text);
    const scale = fontScale(fontSize);
    const charWidth = FONT_WIDTH * scale;
    const charHeight = FONT_HEIGHT * scale;
    const spacing = scale;
    const textWidth = chars.length * charWidth + (chars.length - 1) * spacing;
    const textHeight = charHeight;

    const boxWidth = rotation === 0 ? textWidth : textHeight;
    const boxHeight = rotation === 0 ? textHeight : textWidth;

    let originX = Math.round(x);
    let originY = Math.round(y);
    if (align === "middle") originX -= Math.round(boxWidth / 2);
    else if (align === "end") originX -= boxWidth;
    if (baseline === "middle") originY -= Math.round(boxHeight / 2);
    else if (baseline === "bottom") originY -= boxHeight;

    for (let i = 0; i < chars.length; i++) {
      const rows = glyphRows(chars[i] ?? "");
      const xOffset = i * (charWidth + spacing);
      for (let row = 0; row < rows.length; row++) {
        const mask = rows[row] ?? 0;
        for (let col = 0; col < FONT_WIDTH; col++) {
          if ((mask & (1 << (FONT_WIDTH - 1 - col))) === 0) continue;
          const baseX = xOffset + col * scale;
          const baseY = row * scale;
          for (let sy = 0; sy < scale; sy++) {
            for (let sx = 0; sx < scale; sx++) {
              const px = baseX + sx;
              const py = baseY + sy;
              let rx = px;
              let ry = py;
              if (rotation === -90) {
                rx = py;
                ry = textWidth - 1 - px;
              } else if (rotation === 90) {
                rx = textHeight - 1 - py;
                ry = px;
              }
              this.setPixelRGBA(originX + rx, originY + ry, r, g, b, a);
            }
          }
        }
      }
    }
  }
}

const FONT_WIDTH = 5;
const FONT_HEIGHT = 7;

const FONT: Readonly<Record<string, readonly number[]>> = {
  " ": [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
  "!": [0x04, 0x04, 0x04, 0x04, 0x04, 0x00, 0x04],
  '"': [0x0a, 0x0a, 0x00, 0x00, 0x00, 0x00, 0x00],
  "'": [0x04, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00],
  "(": [0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02],
  ")": [0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08],
  "*": [0x00, 0x0a, 0x04, 0x1f, 0x04, 0x0a, 0x00],
  "+": [0x00, 0x04, 0x04, 0x1f, 0x04, 0x04, 0x00],
  ",": [0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x08],
  "-": [0x00, 0x00, 0x00, 0x1f, 0x00, 0x00, 0x00],
  ".": [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04],
  "/": [0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10],
  "0": [0x0e, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0e],
  "1": [0x04, 0x0c, 0x04, 0x04, 0x04, 0x04, 0x0e],
  "2": [0x0e, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1f],
  "3": [0x1f, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0e],
  "4": [0x02, 0x06, 0x0a, 0x12, 0x1f, 0x02, 0x02],
  "5": [0x1f, 0x10, 0x1e, 0x01, 0x01, 0x11, 0x0e],
  "6": [0x06, 0x08, 0x10, 0x1e, 0x11, 0x11, 0x0e],
  "7": [0x1f, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
  "8": [0x0e, 0x11, 0x11, 0x0e, 0x11, 0x11, 0x0e],
  "9": [0x0e, 0x11, 0x11, 0x0f, 0x01, 0x02, 0x0c],
  ":": [0x00, 0x04, 0x00, 0x00, 0x00, 0x04, 0x00],
  ";": [0x00, 0x04, 0x00, 0x00, 0x00, 0x04, 0x08],
  "<": [0x02, 0x04, 0x08, 0x10, 0x08, 0x04, 0x02],
  "=": [0x00, 0x1f, 0x00, 0x1f, 0x00, 0x00, 0x00],
  ">": [0x10, 0x08, 0x04, 0x02, 0x04, 0x08, 0x10],
  "?": [0x0e, 0x11, 0x01, 0x02, 0x04, 0x00, 0x04],
  "%": [0x18, 0x19, 0x02, 0x04, 0x08, 0x13, 0x03],
  _: [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1f],
  A: [0x0e, 0x11, 0x11, 0x1f, 0x11, 0x11, 0x11],
  B: [0x1e, 0x11, 0x11, 0x1e, 0x11, 0x11, 0x1e],
  C: [0x0e, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0e],
  D: [0x1e, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1e],
  E: [0x1f, 0x10, 0x10, 0x1e, 0x10, 0x10, 0x1f],
  F: [0x1f, 0x10, 0x10, 0x1e, 0x10, 0x10, 0x10],
  G: [0x0e, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0e],
  H: [0x11, 0x11, 0x11, 0x1f, 0x11, 0x11, 0x11],
  I: [0x0e, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0e],
  J: [0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0c],
  K: [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
  L: [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1f],
  M: [0x11, 0x1b, 0x15, 0x11, 0x11, 0x11, 0x11],
  N: [0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11],
  O: [0x0e, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e],
  P: [0x1e, 0x11, 0x11, 0x1e, 0x10, 0x10, 0x10],
  Q: [0x0e, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0d],
  R: [0x1e, 0x11, 0x11, 0x1e, 0x14, 0x12, 0x11],
  S: [0x0f, 0x10, 0x10, 0x0e, 0x01, 0x01, 0x1e],
  T: [0x1f, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
  U: [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e],
  V: [0x11, 0x11, 0x11, 0x11, 0x11, 0x0a, 0x04],
  W: [0x11, 0x11, 0x11, 0x11, 0x15, 0x1b, 0x11],
  X: [0x11, 0x11, 0x0a, 0x04, 0x0a, 0x11, 0x11],
  Y: [0x11, 0x11, 0x0a, 0x04, 0x04, 0x04, 0x04],
  Z: [0x1f, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1f],
  "[": [0x0e, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0e],
  "]": [0x0e, 0x02, 0x02, 0x02, 0x02, 0x02, 0x0e],
  "#": [0x0a, 0x0a, 0x1f, 0x0a, 0x1f, 0x0a, 0x0a],
  $: [0x04, 0x0f, 0x14, 0x0e, 0x05, 0x1e, 0x04],
  "&": [0x0c, 0x12, 0x14, 0x08, 0x15, 0x12, 0x0d],
  "@": [0x0e, 0x11, 0x17, 0x15, 0x17, 0x10, 0x0e],
  "\\": [0x10, 0x10, 0x08, 0x04, 0x02, 0x01, 0x01],
  "^": [0x04, 0x0a, 0x11, 0x00, 0x00, 0x00, 0x00],
  "`": [0x08, 0x04, 0x02, 0x00, 0x00, 0x00, 0x00],
  "{": [0x02, 0x04, 0x04, 0x08, 0x04, 0x04, 0x02],
  "|": [0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
  "}": [0x08, 0x04, 0x04, 0x02, 0x04, 0x04, 0x08],
  "~": [0x00, 0x00, 0x08, 0x15, 0x02, 0x00, 0x00],
};

function fontScale(fontSize: number): number {
  if (!Number.isFinite(fontSize) || fontSize <= 0) {
    throw new InvalidParameterError(
      `fontSize must be a positive finite number; received ${fontSize}`,
      "fontSize",
      fontSize
    );
  }
  return Math.min(MAX_TEXT_SCALE, Math.max(1, Math.round(fontSize / FONT_HEIGHT)));
}

function glyphRows(ch: string): readonly number[] {
  const direct = FONT[ch];
  if (direct) return direct;
  const upper = ch.toUpperCase();
  return FONT[upper] ?? FONT["?"] ?? [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
}
