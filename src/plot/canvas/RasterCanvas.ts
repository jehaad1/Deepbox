import { assertPositiveInt, clampInt } from "../utils/validation";

/**
 * @internal
 */
export class RasterCanvas {
  readonly width: number;
  readonly height: number;
  readonly data: Uint8ClampedArray;

  constructor(width: number, height: number) {
    assertPositiveInt("width", width);
    assertPositiveInt("height", height);
    this.width = width;
    this.height = height;
    this.data = new Uint8ClampedArray(width * height * 4);
  }

  clearRGBA(r: number, g: number, b: number, a: number): void {
    const n = this.width * this.height;
    for (let i = 0; i < n; i++) {
      const k = i * 4;
      this.data[k] = r;
      this.data[k + 1] = g;
      this.data[k + 2] = b;
      this.data[k + 3] = a;
    }
  }

  setPixelRGBA(x: number, y: number, r: number, g: number, b: number, a: number): void {
    if (x < 0 || y < 0 || x >= this.width || y >= this.height) return;
    const idx = (y * this.width + x) * 4;
    this.data[idx] = r;
    this.data[idx + 1] = g;
    this.data[idx + 2] = b;
    this.data[idx + 3] = a;
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
    const x1 = x0 + w;
    const y1 = y0 + h;
    const cx0 = clampInt(x0, 0, this.width);
    const cy0 = clampInt(y0, 0, this.height);
    const cx1 = clampInt(x1, 0, this.width);
    const cy1 = clampInt(y1, 0, this.height);
    for (let y = cy0; y < cy1; y++) {
      for (let x = cx0; x < cx1; x++) this.setPixelRGBA(x, y, r, g, b, a);
    }
  }

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
    let x = clampInt(x0, -1_000_000, 1_000_000);
    let y = clampInt(y0, -1_000_000, 1_000_000);
    const xEnd = clampInt(x1, -1_000_000, 1_000_000);
    const yEnd = clampInt(y1, -1_000_000, 1_000_000);
    const dx = Math.abs(xEnd - x);
    const sx = x < xEnd ? 1 : -1;
    const dy = -Math.abs(yEnd - y);
    const sy = y < yEnd ? 1 : -1;
    let err = dx + dy;

    for (;;) {
      this.setPixelRGBA(x, y, r, g, b, a);
      if (x === xEnd && y === yEnd) break;
      const e2 = 2 * err;
      if (e2 >= dy) {
        err += dy;
        x += sx;
      }
      if (e2 <= dx) {
        err += dx;
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

    for (let i = 0; i < totalHeight; i++) {
      const secondHalf = i > y1 - y0 || y1 === y0;
      const segmentHeight = secondHalf ? y2 - y1 : y1 - y0;
      const alpha = i / totalHeight;
      const beta = (i - (secondHalf ? y1 - y0 : 0)) / segmentHeight;

      let ax = x0 + (x2 - x0) * alpha;
      let bx = secondHalf ? x1 + (x2 - x1) * beta : x0 + (x1 - x0) * beta;

      if (ax > bx) {
        [ax, bx] = [bx, ax];
      }

      const y = Math.floor(y0 + i);
      const startX = Math.floor(ax);
      const endX = Math.ceil(bx);

      for (let x = startX; x < endX; x++) {
        this.setPixelRGBA(x, y, r, g, b, a);
      }
    }
  }

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
    const x0 = clampInt(cx - rr, 0, this.width);
    const x1 = clampInt(cx + rr + 1, 0, this.width);
    const y0 = clampInt(cy - rr, 0, this.height);
    const y1 = clampInt(cy + rr + 1, 0, this.height);

    for (let y = y0; y < y1; y++) {
      const dy = y - cy;
      for (let x = x0; x < x1; x++) {
        const dx = x - cx;
        if (dx * dx + dy * dy <= r2) this.setPixelRGBA(x, y, r, g, b, a);
      }
    }
  }

  measureText(text: string, fontSize = 12): { readonly width: number; readonly height: number } {
    const scale = Math.max(1, Math.round(fontSize / FONT_HEIGHT));
    const charWidth = FONT_WIDTH * scale;
    const charHeight = FONT_HEIGHT * scale;
    const spacing = scale;
    if (text.length === 0) return { width: 0, height: charHeight };
    return { width: text.length * charWidth + (text.length - 1) * spacing, height: charHeight };
  }

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

    const scale = Math.max(1, Math.round(fontSize / FONT_HEIGHT));
    const charWidth = FONT_WIDTH * scale;
    const charHeight = FONT_HEIGHT * scale;
    const spacing = scale;
    const textWidth = text.length * charWidth + (text.length - 1) * spacing;
    const textHeight = charHeight;

    const boxWidth = rotation === 0 ? textWidth : textHeight;
    const boxHeight = rotation === 0 ? textHeight : textWidth;

    let originX = Math.round(x);
    let originY = Math.round(y);
    if (align === "middle") originX -= Math.round(boxWidth / 2);
    else if (align === "end") originX -= boxWidth;
    if (baseline === "middle") originY -= Math.round(boxHeight / 2);
    else if (baseline === "bottom") originY -= boxHeight;

    for (let i = 0; i < text.length; i++) {
      const ch = text[i] ?? "";
      const rows = glyphRows(ch);
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
  "/": [0x01, 0x02, 0x04, 0x08, 0x10, 0x00, 0x00],
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
  "%": [0x19, 0x19, 0x02, 0x04, 0x08, 0x13, 0x13],
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
  J: [0x07, 0x02, 0x02, 0x02, 0x12, 0x12, 0x0c],
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
};

function glyphRows(ch: string): readonly number[] {
  const direct = FONT[ch];
  if (direct) return direct;
  const upper = ch.toUpperCase();
  return FONT[upper] ?? FONT["?"] ?? [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
}
