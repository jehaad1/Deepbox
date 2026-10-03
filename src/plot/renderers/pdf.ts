/**
 * Minimal, dependency-free SVG to PDF converter.
 *
 * `svgToPdf` reads the SVG that Deepbox figures produce and rewrites its drawing elements as
 * PDF content-stream operators, so the PDF stays vector. Elements are drawn in document order.
 *
 * Supported SVG: `rect`, `line`, `circle`, `ellipse`, `polyline`, `polygon`, `path`
 * (`M L H V C S Q T A Z`, absolute and relative), `text`, and `g` / `clipPath`. Supported
 * presentation attributes: `fill`, `stroke`, `stroke-width`, `stroke-dasharray`,
 * `stroke-linecap`, `stroke-linejoin`, `fill-rule`, `opacity`, `fill-opacity`,
 * `stroke-opacity`, `transform` (`translate`, `scale`, `rotate`, `skewX`, `skewY`, `matrix`),
 * `clip-path` on groups, a `style` attribute with the same properties, `font-size`,
 * `font-weight`, `text-anchor` and `dominant-baseline`. Gradients, patterns, filters, images,
 * CSS stylesheets, `use` and rounded rectangle corners are not supported and are skipped.
 *
 * Text uses the built-in Helvetica and Helvetica-Bold fonts with WinAnsi encoding, so characters
 * outside Windows-1252 are written as `?`.
 *
 * @module plot/renderers/pdf
 * @see {@link https://deepbox.dev/docs/plot-basic | Deepbox documentation}
 */

import { InvalidParameterError } from "../../core";
import { parseHexColorToRGBA } from "../utils/colors";

/** Advance widths (1/1000 em) of Helvetica for Windows-1252 codes 32..255. */
const HELVETICA_WIDTHS: readonly number[] = [
  278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278, 556, 556, 556,
  556, 556, 556, 556, 556, 556, 556, 278, 278, 584, 584, 584, 556, 1015, 667, 667, 722, 722, 667,
  611, 778, 722, 278, 500, 667, 556, 833, 722, 778, 667, 778, 722, 667, 611, 722, 667, 944, 667,
  667, 611, 278, 278, 278, 469, 556, 333, 556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500,
  222, 833, 556, 556, 556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500, 334, 260, 334, 584,
  761, 556, 0, 222, 556, 333, 1000, 556, 556, 333, 1000, 667, 333, 1000, 0, 611, 0, 0, 222, 222,
  333, 333, 350, 556, 1000, 333, 1000, 500, 333, 944, 0, 500, 667, 278, 333, 556, 556, 556, 556,
  260, 556, 333, 737, 370, 556, 584, 333, 737, 333, 400, 584, 333, 333, 333, 556, 537, 278, 333,
  333, 365, 556, 834, 834, 834, 611, 667, 667, 667, 667, 667, 667, 1000, 722, 667, 667, 667, 667,
  278, 278, 278, 278, 722, 722, 778, 778, 778, 778, 778, 584, 778, 722, 722, 722, 722, 667, 667,
  611, 556, 556, 556, 556, 556, 556, 889, 500, 556, 556, 556, 556, 278, 278, 278, 278, 556, 556,
  556, 556, 556, 556, 556, 584, 611, 556, 556, 556, 556, 500, 556, 500,
];

/** Advance widths (1/1000 em) of Helvetica-Bold for Windows-1252 codes 32..255. */
const HELVETICA_BOLD_WIDTHS: readonly number[] = [
  278, 333, 474, 556, 556, 889, 722, 238, 333, 333, 389, 584, 278, 333, 278, 278, 556, 556, 556,
  556, 556, 556, 556, 556, 556, 556, 333, 333, 584, 584, 584, 611, 975, 722, 722, 722, 722, 667,
  611, 778, 722, 278, 556, 722, 611, 833, 722, 778, 667, 778, 722, 667, 611, 722, 667, 944, 667,
  667, 611, 333, 278, 333, 584, 556, 333, 556, 611, 556, 611, 556, 333, 611, 611, 278, 278, 556,
  278, 889, 611, 611, 611, 611, 389, 556, 333, 611, 556, 778, 556, 556, 500, 389, 280, 389, 584,
  761, 556, 0, 278, 556, 500, 1000, 556, 556, 333, 1000, 667, 333, 1000, 0, 611, 0, 0, 278, 278,
  500, 500, 350, 556, 1000, 333, 1000, 556, 333, 944, 0, 500, 667, 278, 333, 556, 556, 556, 556,
  280, 556, 333, 737, 370, 556, 584, 333, 737, 333, 400, 584, 333, 333, 333, 611, 556, 278, 333,
  333, 365, 556, 834, 834, 834, 611, 722, 722, 722, 722, 722, 722, 1000, 722, 667, 667, 667, 667,
  278, 278, 278, 278, 722, 722, 778, 778, 778, 778, 778, 584, 778, 722, 722, 722, 722, 667, 667,
  611, 556, 556, 556, 556, 556, 556, 889, 556, 556, 556, 556, 556, 278, 278, 278, 278, 611, 611,
  611, 611, 611, 611, 611, 584, 611, 611, 611, 611, 611, 556, 611, 556,
];

/** Windows-1252 code points for bytes 0x80-0x9F (0 marks an unassigned byte). */
const CP1252_HIGH: readonly number[] = [
  0x20ac, 0, 0x201a, 0x0192, 0x201e, 0x2026, 0x2020, 0x2021, 0x02c6, 0x2030, 0x0160, 0x2039, 0x0152,
  0, 0x017d, 0, 0, 0x2018, 0x2019, 0x201c, 0x201d, 0x2022, 0x2013, 0x2014, 0x02dc, 0x2122, 0x0161,
  0x203a, 0x0153, 0, 0x017e, 0x0178,
];

type Attrs = Record<string, string>;

type Style = {
  readonly fill: string;
  readonly stroke: string;
  readonly strokeWidth: number;
  readonly fillOpacity: number;
  readonly strokeOpacity: number;
  /** Product of the `opacity` of this element and its ancestors. */
  readonly opacity: number;
  readonly dash: string;
  readonly cap: string;
  readonly join: string;
  readonly fillRule: string;
  readonly fontSize: number;
  readonly bold: boolean;
  readonly anchor: string;
  readonly baseline: string;
  readonly hidden: boolean;
};

const ROOT_STYLE: Style = {
  fill: "black",
  stroke: "none",
  strokeWidth: 1,
  fillOpacity: 1,
  strokeOpacity: 1,
  opacity: 1,
  dash: "none",
  cap: "butt",
  join: "miter",
  fillRule: "nonzero",
  fontSize: 16,
  bold: false,
  anchor: "start",
  baseline: "alphabetic",
  hidden: false,
};

/** Number formatting for PDF: finite, no exponent, at most 4 decimals. */
function num(n: number): string {
  if (!Number.isFinite(n)) return "0";
  const clamped = Math.max(-1e9, Math.min(1e9, n));
  const s = clamped.toFixed(4);
  const trimmed = s.includes(".") ? s.replace(/0+$/, "").replace(/\.$/, "") : s;
  return trimmed === "-0" || trimmed === "" ? "0" : trimmed;
}

function decodeEntities(s: string): string {
  if (!s.includes("&")) return s;
  return s.replace(/&(#x[0-9a-fA-F]+|#\d+|amp|lt|gt|quot|apos);/g, (match, body: string) => {
    switch (body) {
      case "amp":
        return "&";
      case "lt":
        return "<";
      case "gt":
        return ">";
      case "quot":
        return '"';
      case "apos":
        return "'";
      default: {
        const code =
          body[1] === "x" || body[1] === "X"
            ? Number.parseInt(body.slice(2), 16)
            : Number.parseInt(body.slice(1), 10);
        return Number.isFinite(code) && code > 0 && code <= 0x10ffff
          ? String.fromCodePoint(code)
          : match;
      }
    }
  });
}

function parseAttrs(src: string): Attrs {
  const attrs: Attrs = {};
  const re = /([^\s=/>"']+)\s*=\s*(?:"([^"]*)"|'([^']*)')/g;
  let m = re.exec(src);
  while (m) {
    attrs[m[1] ?? ""] = decodeEntities(m[2] ?? m[3] ?? "");
    m = re.exec(src);
  }
  return attrs;
}

/** Plain number from an attribute such as "12", "12.5px"; `fallback` when absent or invalid. */
function numAttr(value: string | undefined, fallback: number): number {
  if (value === undefined) return fallback;
  const v = Number.parseFloat(value);
  return Number.isFinite(v) ? v : fallback;
}

function opacityAttr(value: string | undefined): number | null {
  if (value === undefined) return null;
  const trimmed = value.trim();
  const v = Number.parseFloat(trimmed);
  if (!Number.isFinite(v)) return null;
  return Math.min(1, Math.max(0, trimmed.endsWith("%") ? v / 100 : v));
}

const STYLE_KEYS = [
  "fill",
  "stroke",
  "stroke-width",
  "stroke-dasharray",
  "stroke-linecap",
  "stroke-linejoin",
  "fill-rule",
  "opacity",
  "fill-opacity",
  "stroke-opacity",
  "font-size",
  "font-weight",
  "text-anchor",
  "dominant-baseline",
  "display",
  "visibility",
];

/** Presentation attributes overlaid by the `style="a:b;c:d"` declarations. */
function effectiveProps(attrs: Attrs): Attrs {
  const styleAttr = attrs["style"];
  if (!styleAttr) return attrs;
  const merged: Attrs = { ...attrs };
  for (const decl of styleAttr.split(";")) {
    const colon = decl.indexOf(":");
    if (colon < 0) continue;
    const key = decl.slice(0, colon).trim().toLowerCase();
    const value = decl.slice(colon + 1).trim();
    if (STYLE_KEYS.includes(key) && value.length > 0) merged[key] = value;
  }
  return merged;
}

function deriveStyle(parent: Style, rawAttrs: Attrs): Style {
  const a = effectiveProps(rawAttrs);
  const weight = a["font-weight"];
  const weightNum = weight === undefined ? Number.NaN : Number.parseInt(weight, 10);
  const bold =
    weight === undefined
      ? parent.bold
      : weight === "bold" ||
        weight === "bolder" ||
        (Number.isFinite(weightNum) && weightNum >= 600);
  const ownOpacity = opacityAttr(a["opacity"]);
  return {
    fill: a["fill"]?.trim() ?? parent.fill,
    stroke: a["stroke"]?.trim() ?? parent.stroke,
    strokeWidth: numAttr(a["stroke-width"], parent.strokeWidth),
    fillOpacity: opacityAttr(a["fill-opacity"]) ?? parent.fillOpacity,
    strokeOpacity: opacityAttr(a["stroke-opacity"]) ?? parent.strokeOpacity,
    opacity: parent.opacity * (ownOpacity ?? 1),
    dash: a["stroke-dasharray"]?.trim() ?? parent.dash,
    cap: a["stroke-linecap"]?.trim() ?? parent.cap,
    join: a["stroke-linejoin"]?.trim() ?? parent.join,
    fillRule: a["fill-rule"]?.trim() ?? parent.fillRule,
    fontSize: numAttr(a["font-size"], parent.fontSize),
    bold,
    anchor: a["text-anchor"]?.trim() ?? parent.anchor,
    baseline: a["dominant-baseline"]?.trim() ?? parent.baseline,
    hidden:
      parent.hidden || a["display"]?.trim() === "none" || a["visibility"]?.trim() === "hidden",
  };
}

type Paint = { readonly rgb: string; readonly alpha: number };

/** Resolves an SVG paint to PDF RGB components; null for `none`, gradients and the like. */
function resolvePaint(value: string): Paint | null {
  const v = value.trim().toLowerCase();
  if (v === "" || v === "none" || v === "transparent" || v.startsWith("url(")) return null;
  const rgba = parseHexColorToRGBA(v === "currentcolor" ? "#000000" : v);
  if (rgba.a === 0) return null;
  return {
    rgb: `${num(rgba.r / 255)} ${num(rgba.g / 255)} ${num(rgba.b / 255)}`,
    alpha: rgba.a / 255,
  };
}

/** Collects the graphics states (constant alpha) used by the page. */
class AlphaStates {
  private readonly names = new Map<string, string>();

  /** Name of the graphics state for the given alphas, or null when both are fully opaque. */
  use(fillAlpha: number, strokeAlpha: number): string | null {
    const ca = Math.min(1, Math.max(0, fillAlpha));
    const cs = Math.min(1, Math.max(0, strokeAlpha));
    if (ca >= 0.9995 && cs >= 0.9995) return null;
    const key = `${num(ca)} ${num(cs)}`;
    let name = this.names.get(key);
    if (name === undefined) {
      name = `GS${this.names.size + 1}`;
      this.names.set(key, name);
    }
    return name;
  }

  resources(): string {
    if (this.names.size === 0) return "";
    const entries: string[] = [];
    for (const [key, name] of this.names) {
      const [ca = "1", cs = "1"] = key.split(" ");
      entries.push(`/${name} << /Type /ExtGState /ca ${ca} /CA ${cs} >>`);
    }
    return ` /ExtGState << ${entries.join(" ")} >>`;
  }
}

// ---------------------------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------------------------

const KAPPA = 0.5522847498307936;

function ellipseOps(cx: number, cy: number, rx: number, ry: number): string[] {
  const kx = rx * KAPPA;
  const ky = ry * KAPPA;
  return [
    `${num(cx + rx)} ${num(cy)} m`,
    `${num(cx + rx)} ${num(cy + ky)} ${num(cx + kx)} ${num(cy + ry)} ${num(cx)} ${num(cy + ry)} c`,
    `${num(cx - kx)} ${num(cy + ry)} ${num(cx - rx)} ${num(cy + ky)} ${num(cx - rx)} ${num(cy)} c`,
    `${num(cx - rx)} ${num(cy - ky)} ${num(cx - kx)} ${num(cy - ry)} ${num(cx)} ${num(cy - ry)} c`,
    `${num(cx + kx)} ${num(cy - ry)} ${num(cx + rx)} ${num(cy - ky)} ${num(cx + rx)} ${num(cy)} c`,
    "h",
  ];
}

/**
 * Converts an SVG elliptical arc (endpoint parameterization, SVG spec F.6.5) into cubic Bezier
 * segments of at most 90 degrees each. Returns the `c` operators.
 */
function arcToCubics(
  x0: number,
  y0: number,
  rxIn: number,
  ryIn: number,
  rotationDeg: number,
  largeArc: boolean,
  sweep: boolean,
  x: number,
  y: number
): string[] {
  let rx = Math.abs(rxIn);
  let ry = Math.abs(ryIn);
  if (x0 === x && y0 === y) return [];
  if (rx === 0 || ry === 0) return [`${num(x)} ${num(y)} l`];

  const phi = (rotationDeg * Math.PI) / 180;
  const cosPhi = Math.cos(phi);
  const sinPhi = Math.sin(phi);
  const dx2 = (x0 - x) / 2;
  const dy2 = (y0 - y) / 2;
  const x1p = cosPhi * dx2 + sinPhi * dy2;
  const y1p = -sinPhi * dx2 + cosPhi * dy2;

  const lambda = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry);
  if (lambda > 1) {
    const s = Math.sqrt(lambda);
    rx *= s;
    ry *= s;
  }

  const rx2 = rx * rx;
  const ry2 = ry * ry;
  const num_ = rx2 * ry2 - rx2 * y1p * y1p - ry2 * x1p * x1p;
  const den = rx2 * y1p * y1p + ry2 * x1p * x1p;
  let coef = den === 0 ? 0 : Math.sqrt(Math.max(0, num_ / den));
  if (largeArc === sweep) coef = -coef;
  const cxp = (coef * rx * y1p) / ry;
  const cyp = (-coef * ry * x1p) / rx;
  const cx = cosPhi * cxp - sinPhi * cyp + (x0 + x) / 2;
  const cy = sinPhi * cxp + cosPhi * cyp + (y0 + y) / 2;

  const angle = (ux: number, uy: number, vx: number, vy: number): number => {
    const dot = ux * vx + uy * vy;
    const len = Math.hypot(ux, uy) * Math.hypot(vx, vy);
    const a = Math.acos(Math.max(-1, Math.min(1, dot / len)));
    return ux * vy - uy * vx < 0 ? -a : a;
  };
  const theta1 = angle(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry);
  let delta = angle((x1p - cxp) / rx, (y1p - cyp) / ry, (-x1p - cxp) / rx, (-y1p - cyp) / ry);
  if (!sweep && delta > 0) delta -= 2 * Math.PI;
  else if (sweep && delta < 0) delta += 2 * Math.PI;

  const segments = Math.max(1, Math.ceil(Math.abs(delta) / (Math.PI / 2) - 1e-9));
  const step = delta / segments;
  const t = (4 / 3) * Math.tan(step / 4);
  const ops: string[] = [];
  const point = (ang: number): [number, number] => {
    const ex = rx * Math.cos(ang);
    const ey = ry * Math.sin(ang);
    return [cosPhi * ex - sinPhi * ey + cx, sinPhi * ex + cosPhi * ey + cy];
  };
  const deriv = (ang: number): [number, number] => {
    const ex = -rx * Math.sin(ang);
    const ey = ry * Math.cos(ang);
    return [cosPhi * ex - sinPhi * ey, sinPhi * ex + cosPhi * ey];
  };
  for (let i = 0; i < segments; i++) {
    const a1 = theta1 + i * step;
    const a2 = a1 + step;
    const [p1x, p1y] = point(a1);
    const [d1x, d1y] = deriv(a1);
    const [p2x, p2y] = i === segments - 1 ? [x, y] : point(a2);
    const [d2x, d2y] = deriv(a2);
    ops.push(
      `${num(p1x + t * d1x)} ${num(p1y + t * d1y)} ${num(p2x - t * d2x)} ${num(p2y - t * d2y)} ${num(p2x)} ${num(p2y)} c`
    );
  }
  return ops;
}

/** Cursor over SVG path data. */
class PathReader {
  private i = 0;
  constructor(private readonly s: string) {}

  private skipSeparators(): void {
    while (this.i < this.s.length && /[\s,]/.test(this.s[this.i] ?? "")) this.i++;
  }

  /** Next command letter, consumed; null when the next token is not a letter. */
  command(): string | null {
    this.skipSeparators();
    const ch = this.s[this.i];
    if (ch !== undefined && /[A-Za-z]/.test(ch)) {
      this.i++;
      return ch;
    }
    return null;
  }

  atEnd(): boolean {
    this.skipSeparators();
    return this.i >= this.s.length;
  }

  hasNumber(): boolean {
    this.skipSeparators();
    return /[-+.\d]/.test(this.s[this.i] ?? "");
  }

  number(): number | null {
    this.skipSeparators();
    const re = /[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?/y;
    re.lastIndex = this.i;
    const m = re.exec(this.s);
    if (!m) return null;
    this.i = re.lastIndex;
    const v = Number.parseFloat(m[0]);
    return Number.isFinite(v) ? v : null;
  }

  /** Arc flag: a single `0` or `1`, which may directly precede the next number. */
  flag(): boolean | null {
    this.skipSeparators();
    const ch = this.s[this.i];
    if (ch === "0" || ch === "1") {
      this.i++;
      return ch === "1";
    }
    return null;
  }
}

/** Converts SVG path data to PDF path operators. Parsing stops quietly at the first error. */
function pathDataToOps(d: string): string[] {
  const ops: string[] = [];
  const r = new PathReader(d);
  let cx = 0;
  let cy = 0;
  let startX = 0;
  let startY = 0;
  let lastCubic: [number, number] | null = null;
  let lastQuad: [number, number] | null = null;
  let cmd: string | null = null;
  let first = true;

  while (!r.atEnd()) {
    const next = r.command();
    if (next !== null) {
      cmd = next;
    } else if (cmd === null || !r.hasNumber() || cmd === "Z" || cmd === "z") {
      break;
    }
    if (cmd === null) break;
    if (first && cmd !== "M" && cmd !== "m") break;
    first = false;

    const rel: boolean = cmd === cmd.toLowerCase();
    const upper: string = cmd.toUpperCase();
    const read = (count: number): number[] | null => {
      const out: number[] = [];
      for (let k = 0; k < count; k++) {
        const v = r.number();
        if (v === null) return null;
        out.push(v);
      }
      return out;
    };

    if (upper === "Z") {
      ops.push("h");
      cx = startX;
      cy = startY;
      lastCubic = null;
      lastQuad = null;
      continue;
    }

    if (upper === "M") {
      const p = read(2);
      if (!p) break;
      cx = rel ? cx + (p[0] ?? 0) : (p[0] ?? 0);
      cy = rel ? cy + (p[1] ?? 0) : (p[1] ?? 0);
      startX = cx;
      startY = cy;
      ops.push(`${num(cx)} ${num(cy)} m`);
      cmd = rel ? "l" : "L"; // further coordinate pairs are implicit line-tos
      lastCubic = null;
      lastQuad = null;
    } else if (upper === "L") {
      const p = read(2);
      if (!p) break;
      cx = rel ? cx + (p[0] ?? 0) : (p[0] ?? 0);
      cy = rel ? cy + (p[1] ?? 0) : (p[1] ?? 0);
      ops.push(`${num(cx)} ${num(cy)} l`);
      lastCubic = null;
      lastQuad = null;
    } else if (upper === "H") {
      const p = read(1);
      if (!p) break;
      cx = rel ? cx + (p[0] ?? 0) : (p[0] ?? 0);
      ops.push(`${num(cx)} ${num(cy)} l`);
      lastCubic = null;
      lastQuad = null;
    } else if (upper === "V") {
      const p = read(1);
      if (!p) break;
      cy = rel ? cy + (p[0] ?? 0) : (p[0] ?? 0);
      ops.push(`${num(cx)} ${num(cy)} l`);
      lastCubic = null;
      lastQuad = null;
    } else if (upper === "C" || upper === "S") {
      const p = read(upper === "C" ? 6 : 4);
      if (!p) break;
      const ox = rel ? cx : 0;
      const oy = rel ? cy : 0;
      let x1: number;
      let y1: number;
      let k = 0;
      if (upper === "C") {
        x1 = ox + (p[0] ?? 0);
        y1 = oy + (p[1] ?? 0);
        k = 2;
      } else {
        x1 = lastCubic ? 2 * cx - lastCubic[0] : cx;
        y1 = lastCubic ? 2 * cy - lastCubic[1] : cy;
      }
      const x2 = ox + (p[k] ?? 0);
      const y2 = oy + (p[k + 1] ?? 0);
      const x = ox + (p[k + 2] ?? 0);
      const y = oy + (p[k + 3] ?? 0);
      ops.push(`${num(x1)} ${num(y1)} ${num(x2)} ${num(y2)} ${num(x)} ${num(y)} c`);
      lastCubic = [x2, y2];
      lastQuad = null;
      cx = x;
      cy = y;
    } else if (upper === "Q" || upper === "T") {
      const p = read(upper === "Q" ? 4 : 2);
      if (!p) break;
      const ox = rel ? cx : 0;
      const oy = rel ? cy : 0;
      let qx: number;
      let qy: number;
      let k = 0;
      if (upper === "Q") {
        qx = ox + (p[0] ?? 0);
        qy = oy + (p[1] ?? 0);
        k = 2;
      } else {
        qx = lastQuad ? 2 * cx - lastQuad[0] : cx;
        qy = lastQuad ? 2 * cy - lastQuad[1] : cy;
      }
      const x = ox + (p[k] ?? 0);
      const y = oy + (p[k + 1] ?? 0);
      ops.push(
        `${num(cx + (2 / 3) * (qx - cx))} ${num(cy + (2 / 3) * (qy - cy))} ` +
          `${num(x + (2 / 3) * (qx - x))} ${num(y + (2 / 3) * (qy - y))} ${num(x)} ${num(y)} c`
      );
      lastQuad = [qx, qy];
      lastCubic = null;
      cx = x;
      cy = y;
    } else if (upper === "A") {
      const rx = r.number();
      const ry = r.number();
      const rot = r.number();
      const large = r.flag();
      const sweep = r.flag();
      const px = r.number();
      const py = r.number();
      if (
        rx === null ||
        ry === null ||
        rot === null ||
        large === null ||
        sweep === null ||
        px === null ||
        py === null
      ) {
        break;
      }
      const x = rel ? cx + px : px;
      const y = rel ? cy + py : py;
      ops.push(...arcToCubics(cx, cy, rx, ry, rot, large, sweep, x, y));
      cx = x;
      cy = y;
      lastCubic = null;
      lastQuad = null;
    } else {
      break;
    }
  }
  return ops;
}

function pointsToOps(points: string, close: boolean): string[] {
  const nums = points
    .trim()
    .split(/[\s,]+/)
    .map(Number.parseFloat);
  const ops: string[] = [];
  for (let i = 0; i + 1 < nums.length; i += 2) {
    const x = nums[i] ?? Number.NaN;
    const y = nums[i + 1] ?? Number.NaN;
    if (!Number.isFinite(x) || !Number.isFinite(y)) break;
    ops.push(`${num(x)} ${num(y)} ${i === 0 ? "m" : "l"}`);
  }
  if (close && ops.length > 0) ops.push("h");
  return ops;
}

/** Path operators for a shape element, or null when the element is not a drawable shape. */
function shapeOps(name: string, a: Attrs): string[] | null {
  switch (name) {
    case "rect": {
      const w = numAttr(a["width"], 0);
      const h = numAttr(a["height"], 0);
      if (!(w > 0) || !(h > 0)) return [];
      return [`${num(numAttr(a["x"], 0))} ${num(numAttr(a["y"], 0))} ${num(w)} ${num(h)} re`];
    }
    case "circle": {
      const r = numAttr(a["r"], 0);
      return r > 0 ? ellipseOps(numAttr(a["cx"], 0), numAttr(a["cy"], 0), r, r) : [];
    }
    case "ellipse": {
      const rx = numAttr(a["rx"], 0);
      const ry = numAttr(a["ry"], 0);
      return rx > 0 && ry > 0 ? ellipseOps(numAttr(a["cx"], 0), numAttr(a["cy"], 0), rx, ry) : [];
    }
    case "line":
      return [
        `${num(numAttr(a["x1"], 0))} ${num(numAttr(a["y1"], 0))} m`,
        `${num(numAttr(a["x2"], 0))} ${num(numAttr(a["y2"], 0))} l`,
      ];
    case "polyline":
      return pointsToOps(a["points"] ?? "", false);
    case "polygon":
      return pointsToOps(a["points"] ?? "", true);
    case "path":
      return pathDataToOps(a["d"] ?? "");
    default:
      return null;
  }
}

// ---------------------------------------------------------------------------------------------
// Transforms
// ---------------------------------------------------------------------------------------------

type Matrix = readonly [number, number, number, number, number, number];

const IDENTITY: Matrix = [1, 0, 0, 1, 0, 0];

function multiply(m1: Matrix, m2: Matrix): Matrix {
  const [a1, b1, c1, d1, e1, f1] = m1;
  const [a2, b2, c2, d2, e2, f2] = m2;
  return [
    a1 * a2 + c1 * b2,
    b1 * a2 + d1 * b2,
    a1 * c2 + c1 * d2,
    b1 * c2 + d1 * d2,
    a1 * e2 + c1 * f2 + e1,
    b1 * e2 + d1 * f2 + f1,
  ];
}

/** Parses an SVG `transform` list into one matrix; null if absent or not understood. */
function parseTransform(value: string | undefined): Matrix | null {
  if (!value) return null;
  const re = /([a-zA-Z]+)\s*\(([^)]*)\)/g;
  let result: Matrix = IDENTITY;
  let found = false;
  let m = re.exec(value);
  while (m) {
    const args = (m[2] ?? "")
      .trim()
      .split(/[\s,]+/)
      .filter((s) => s.length > 0)
      .map(Number.parseFloat);
    if (args.some((v) => !Number.isFinite(v))) return null;
    const at = (i: number, fallback: number): number => args[i] ?? fallback;
    let t: Matrix | null = null;
    switch (m[1]) {
      case "translate":
        t = [1, 0, 0, 1, at(0, 0), at(1, 0)];
        break;
      case "scale":
        t = [at(0, 1), 0, 0, at(1, at(0, 1)), 0, 0];
        break;
      case "rotate": {
        const ang = (at(0, 0) * Math.PI) / 180;
        const cos = Math.cos(ang);
        const sin = Math.sin(ang);
        const cx = at(1, 0);
        const cy = at(2, 0);
        t = [cos, sin, -sin, cos, cx - cos * cx + sin * cy, cy - sin * cx - cos * cy];
        break;
      }
      case "skewX":
        t = [1, 0, Math.tan((at(0, 0) * Math.PI) / 180), 1, 0, 0];
        break;
      case "skewY":
        t = [1, Math.tan((at(0, 0) * Math.PI) / 180), 0, 1, 0, 0];
        break;
      case "matrix":
        if (args.length === 6) t = [at(0, 1), at(1, 0), at(2, 0), at(3, 1), at(4, 0), at(5, 0)];
        break;
      default:
        break;
    }
    if (t === null) return null;
    result = multiply(result, t);
    found = true;
    m = re.exec(value);
  }
  return found ? result : null;
}

function cmOp(m: Matrix): string {
  return `${m.map(num).join(" ")} cm`;
}

// ---------------------------------------------------------------------------------------------
// Text
// ---------------------------------------------------------------------------------------------

/** Windows-1252 byte for a code point, or 0x3F ("?") when it has none. */
function toWinAnsi(cp: number): number {
  if (cp >= 0x20 && cp < 0x7f) return cp;
  if (cp >= 0xa0 && cp <= 0xff) return cp;
  if (cp === 0x2212) return 0x2d; // minus sign -> hyphen-minus
  for (let i = 0; i < CP1252_HIGH.length; i++) {
    if (CP1252_HIGH[i] === cp) return 0x80 + i;
  }
  return 0x3f;
}

function winAnsiBytes(text: string): number[] {
  const bytes: number[] = [];
  for (const ch of text) bytes.push(toWinAnsi(ch.codePointAt(0) ?? 0x3f));
  return bytes;
}

function pdfString(bytes: readonly number[]): string {
  let out = "(";
  for (const b of bytes) {
    if (b === 0x5c || b === 0x28 || b === 0x29) out += `\\${String.fromCharCode(b)}`;
    else if (b < 0x20 || b > 0x7e) out += `\\${b.toString(8).padStart(3, "0")}`;
    else out += String.fromCharCode(b);
  }
  return `${out})`;
}

function textWidth(bytes: readonly number[], bold: boolean, size: number): number {
  const table = bold ? HELVETICA_BOLD_WIDTHS : HELVETICA_WIDTHS;
  let units = 0;
  for (const b of bytes) units += table[b - 32] ?? 0;
  return (units * size) / 1000;
}

/** Vertical shift (in em) from the SVG `y` to the alphabetic baseline for `dominant-baseline`. */
function baselineShift(baseline: string): number {
  switch (baseline) {
    case "middle":
    case "central":
      return 0.35;
    case "hanging":
    case "text-before-edge":
    case "text-top":
      return 0.72;
    case "text-after-edge":
    case "text-bottom":
      return -0.2;
    default:
      return 0;
  }
}

// ---------------------------------------------------------------------------------------------
// Conversion
// ---------------------------------------------------------------------------------------------

const TAG_RE =
  /<!--[\s\S]*?-->|<\?[\s\S]*?\?>|<!\[CDATA\[[\s\S]*?\]\]>|<![^>]*>|<(\/?)([A-Za-z_][\w:.-]*)((?:"[^"]*"|'[^']*'|[^>"'])*?)(\/?)>/g;

/** Elements whose character content must not be scanned for tags. */
const RAW_TEXT_ELEMENTS = new Set(["style", "script", "title", "desc"]);

/** Grouping elements that are drawn like `<g>`. */
const GROUP_ELEMENTS = new Set(["g", "a"]);

/** Containers whose children are definitions, never drawn directly (unsupported here). */
const DEFINITION_ELEMENTS = new Set([
  "defs",
  "symbol",
  "mask",
  "pattern",
  "marker",
  "lineargradient",
  "radialgradient",
  "filter",
  "switch",
]);

type Group = {
  readonly style: Style;
  /** Number of `Q` operators to emit when the group closes. */
  readonly restores: number;
};

function buildContentStream(
  svg: string,
  pageWidth: number,
  pageHeight: number
): {
  readonly content: string;
  readonly resources: string;
} {
  const ops: string[] = [];
  const alphas = new AlphaStates();
  const clips = new Map<string, string[]>();

  // White page background.
  ops.push("q", "1 1 1 rg", `0 0 ${num(pageWidth)} ${num(pageHeight)} re f`, "Q");

  // Root transform: SVG user space (y down) -> PDF page space (y up), fitted like
  // preserveAspectRatio="xMidYMid meet".
  let viewBox: { x: number; y: number; w: number; h: number } | null = null;
  const stack: Group[] = [];
  let currentClip: { id: string; ops: string[] } | null = null;
  let rootDone = false;

  const currentStyle = (): Style => stack[stack.length - 1]?.style ?? ROOT_STYLE;

  const paintShape = (pathOps: readonly string[], style: Style, closesArea: boolean): void => {
    if (pathOps.length === 0 || style.hidden) return;
    const fill = closesArea ? resolvePaint(style.fill) : null;
    const stroke = style.strokeWidth > 0 ? resolvePaint(style.stroke) : null;
    if (!fill && !stroke) return;
    const fillAlpha = fill ? style.opacity * style.fillOpacity * fill.alpha : 1;
    const strokeAlpha = stroke ? style.opacity * style.strokeOpacity * stroke.alpha : 1;
    ops.push("q");
    const gs = alphas.use(fillAlpha, strokeAlpha);
    if (gs) ops.push(`/${gs} gs`);
    if (fill) ops.push(`${fill.rgb} rg`);
    if (stroke) {
      ops.push(`${stroke.rgb} RG`, `${num(style.strokeWidth)} w`);
      ops.push(`${style.cap === "round" ? 1 : style.cap === "square" ? 2 : 0} J`);
      ops.push(`${style.join === "round" ? 1 : style.join === "bevel" ? 2 : 0} j`);
      const dashes = style.dash
        .split(/[\s,]+/)
        .map(Number.parseFloat)
        .filter((v) => Number.isFinite(v) && v >= 0);
      if (style.dash !== "none" && dashes.length > 0 && dashes.some((v) => v > 0)) {
        const arr = dashes.length % 2 === 1 ? [...dashes, ...dashes] : dashes;
        ops.push(`[${arr.map(num).join(" ")}] 0 d`);
      }
    }
    ops.push(...pathOps);
    const evenOdd = style.fillRule === "evenodd" ? "*" : "";
    if (fill && stroke) ops.push(`B${evenOdd}`);
    else if (fill) ops.push(`f${evenOdd}`);
    else ops.push("S");
    ops.push("Q");
  };

  const paintText = (a: Attrs, content: string, style: Style): void => {
    if (style.hidden) return;
    const collapsed = content.replace(/\s+/g, " ").trim();
    if (collapsed.length === 0) return;
    const fill = resolvePaint(style.fill);
    if (!fill) return;
    const bytes = winAnsiBytes(collapsed);
    const size = style.fontSize;
    if (!(size > 0)) return;
    const width = textWidth(bytes, style.bold, size);
    const x0 = numAttr(a["x"]?.split(/[\s,]+/)[0], 0) + numAttr(a["dx"]?.split(/[\s,]+/)[0], 0);
    const y0 = numAttr(a["y"]?.split(/[\s,]+/)[0], 0) + numAttr(a["dy"]?.split(/[\s,]+/)[0], 0);
    const x = style.anchor === "middle" ? x0 - width / 2 : style.anchor === "end" ? x0 - width : x0;
    const y = y0 + baselineShift(style.baseline) * size;

    ops.push("q");
    const gs = alphas.use(style.opacity * style.fillOpacity * fill.alpha, 1);
    if (gs) ops.push(`/${gs} gs`);
    ops.push(
      "BT",
      `/${style.bold ? "F2" : "F1"} ${num(size)} Tf`,
      `${fill.rgb} rg`,
      // The page is flipped (y down), so the text matrix flips it back to draw upright glyphs.
      `1 0 0 -1 ${num(x)} ${num(y)} Tm`,
      `${pdfString(bytes)} Tj`,
      "ET",
      "Q"
    );
  };

  TAG_RE.lastIndex = 0;
  let m = TAG_RE.exec(svg);
  while (m) {
    const closing = m[1] === "/";
    const name = m[2];
    if (name === undefined) {
      m = TAG_RE.exec(svg); // comment, processing instruction, CDATA or doctype
      continue;
    }
    const selfClosing = m[4] === "/";
    const tag = name.toLowerCase();

    if (closing) {
      if (GROUP_ELEMENTS.has(tag) || DEFINITION_ELEMENTS.has(tag)) {
        const g = stack.pop();
        for (let i = 0; g && i < g.restores; i++) ops.push("Q");
      } else if (tag === "clippath" && currentClip) {
        clips.set(currentClip.id, currentClip.ops);
        currentClip = null;
      }
      m = TAG_RE.exec(svg);
      continue;
    }

    const attrs = parseAttrs(m[3] ?? "");

    // Skip the character content of elements that never contain drawing commands.
    if (RAW_TEXT_ELEMENTS.has(tag) && !selfClosing) {
      const end = svg.indexOf(`</${name}`, TAG_RE.lastIndex);
      if (end >= 0) TAG_RE.lastIndex = end;
      m = TAG_RE.exec(svg);
      continue;
    }

    if (tag === "svg" && !rootDone) {
      rootDone = true;
      const vb = (attrs["viewBox"] ?? "")
        .trim()
        .split(/[\s,]+/)
        .map(Number.parseFloat);
      if (
        vb.length === 4 &&
        vb.every((v) => Number.isFinite(v)) &&
        (vb[2] ?? 0) > 0 &&
        (vb[3] ?? 0) > 0
      ) {
        viewBox = { x: vb[0] ?? 0, y: vb[1] ?? 0, w: vb[2] ?? 1, h: vb[3] ?? 1 };
      } else {
        const w = numAttr(attrs["width"], pageWidth);
        const h = numAttr(attrs["height"], pageHeight);
        viewBox = { x: 0, y: 0, w: w > 0 ? w : pageWidth, h: h > 0 ? h : pageHeight };
      }
      const scale = Math.min(pageWidth / viewBox.w, pageHeight / viewBox.h);
      const offX = (pageWidth - viewBox.w * scale) / 2;
      const offY = (pageHeight - viewBox.h * scale) / 2;
      ops.push(
        cmOp([scale, 0, 0, -scale, offX - viewBox.x * scale, pageHeight - offY + viewBox.y * scale])
      );
      stack.push({ style: deriveStyle(ROOT_STYLE, attrs), restores: 0 });
      m = TAG_RE.exec(svg);
      continue;
    }

    if (tag === "clippath") {
      if (attrs["id"] !== undefined && !selfClosing) currentClip = { id: attrs["id"], ops: [] };
      m = TAG_RE.exec(svg);
      continue;
    }

    if (currentClip) {
      const clipOps = shapeOps(tag, attrs);
      if (clipOps) currentClip.ops.push(...clipOps);
      m = TAG_RE.exec(svg);
      continue;
    }

    if (DEFINITION_ELEMENTS.has(tag)) {
      if (!selfClosing) {
        stack.push({ style: { ...deriveStyle(currentStyle(), attrs), hidden: true }, restores: 0 });
      }
      m = TAG_RE.exec(svg);
      continue;
    }

    if (GROUP_ELEMENTS.has(tag)) {
      const style = deriveStyle(currentStyle(), attrs);
      let restores = 0;
      const matrix = parseTransform(attrs["transform"]);
      const clipRef = /^url\(\s*#([^)\s]+)\s*\)$/.exec(attrs["clip-path"] ?? "");
      const clipOps = clipRef ? clips.get(clipRef[1] ?? "") : undefined;
      if (matrix || (clipOps && clipOps.length > 0)) {
        ops.push("q");
        restores = 1;
        if (matrix) ops.push(cmOp(matrix));
        if (clipOps && clipOps.length > 0) ops.push(...clipOps, "W n");
      }
      if (selfClosing) {
        for (let i = 0; i < restores; i++) ops.push("Q");
      } else {
        stack.push({ style, restores });
      }
      m = TAG_RE.exec(svg);
      continue;
    }

    if (tag === "text") {
      let content = "";
      if (!selfClosing) {
        const end = svg.indexOf("</text", TAG_RE.lastIndex);
        const stop = end >= 0 ? end : svg.length;
        content = decodeEntities(svg.slice(TAG_RE.lastIndex, stop).replace(/<[^>]*>/g, ""));
        TAG_RE.lastIndex = stop;
        const close = svg.indexOf(">", stop);
        if (close >= 0) TAG_RE.lastIndex = close + 1;
      }
      const style = deriveStyle(currentStyle(), attrs);
      const matrix = parseTransform(attrs["transform"]);
      if (matrix) ops.push("q", cmOp(matrix));
      paintText(attrs, content, style);
      if (matrix) ops.push("Q");
      m = TAG_RE.exec(svg);
      continue;
    }

    const pathOps = shapeOps(tag, attrs);
    if (pathOps) {
      const style = deriveStyle(currentStyle(), attrs);
      const matrix = parseTransform(attrs["transform"]);
      if (matrix) ops.push("q", cmOp(matrix));
      paintShape(pathOps, style, tag !== "line");
      if (matrix) ops.push("Q");
    }
    m = TAG_RE.exec(svg);
  }

  // Close anything left open by malformed input so q/Q stay balanced.
  for (const g of stack.reverse()) for (let i = 0; i < g.restores; i++) ops.push("Q");

  return { content: ops.join("\n"), resources: alphas.resources() };
}

/** One byte per UTF-16 unit (every character the converter emits is Latin-1). */
function latin1(s: string): Uint8Array {
  const out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i) & 0xff;
  return out;
}

/**
 * Convert an SVG document to a single-page PDF.
 *
 * The page is `width` x `height` points (1 pt = 1/72 inch) and the SVG is scaled to fit it,
 * keeping its aspect ratio (using the `viewBox`, or the SVG `width`/`height`). See the module
 * description for the supported subset of SVG. Output is PDF 1.4 with uncompressed content.
 *
 * @param svgString - Complete SVG document string
 * @param width - Page width in points; must be positive and finite
 * @param height - Page height in points; must be positive and finite
 * @returns PDF file bytes
 * @throws {InvalidParameterError} If `width` or `height` is not a positive finite number.
 */
export function svgToPdf(svgString: string, width: number, height: number): Uint8Array {
  if (!Number.isFinite(width) || width <= 0) {
    throw new InvalidParameterError(
      `width must be a positive finite number; received ${width}`,
      "width",
      width
    );
  }
  if (!Number.isFinite(height) || height <= 0) {
    throw new InvalidParameterError(
      `height must be a positive finite number; received ${height}`,
      "height",
      height
    );
  }

  const { content, resources } = buildContentStream(svgString, width, height);

  const font = (base: string): string =>
    `<< /Type /Font /Subtype /Type1 /BaseFont /${base} /Encoding /WinAnsiEncoding >>`;
  const bodies = [
    "<< /Type /Catalog /Pages 2 0 R >>",
    "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${num(width)} ${num(height)}] /Contents 4 0 R ` +
      `/Resources << /Font << /F1 5 0 R /F2 6 0 R >>${resources} >> >>`,
    `<< /Length ${content.length} >>\nstream\n${content}\nendstream`,
    font("Helvetica"),
    font("Helvetica-Bold"),
  ];

  const chunks: Uint8Array[] = [latin1("%PDF-1.4\n%âãÏÓ\n")];
  const offsets: number[] = [];
  let position = chunks[0]?.length ?? 0;
  bodies.forEach((body, i) => {
    const bytes = latin1(`${i + 1} 0 obj\n${body}\nendobj\n`);
    offsets.push(position);
    chunks.push(bytes);
    position += bytes.length;
  });

  const xref = [`xref\n0 ${bodies.length + 1}\n`, "0000000000 65535 f \n"];
  for (const offset of offsets) xref.push(`${String(offset).padStart(10, "0")} 00000 n \n`);
  xref.push(
    `trailer\n<< /Size ${bodies.length + 1} /Root 1 0 R >>\nstartxref\n${position}\n%%EOF\n`
  );
  chunks.push(latin1(xref.join("")));

  const total = chunks.reduce((sum, c) => sum + c.length, 0);
  const out = new Uint8Array(total);
  let at = 0;
  for (const c of chunks) {
    out.set(c, at);
    at += c.length;
  }
  return out;
}
