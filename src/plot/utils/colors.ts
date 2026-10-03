/**
 * @see {@link https://deepbox.dev/docs/plot-basic | Deepbox documentation}
 */

import { InvalidParameterError } from "../../core";
import type { Color } from "../types";

const colorCache = new Map<
  string,
  {
    readonly r: number;
    readonly g: number;
    readonly b: number;
    readonly a: number;
  }
>();

const namedColors: Record<string, string> = {
  aliceblue: "#f0f8ff",
  antiquewhite: "#faebd7",
  aqua: "#00ffff",
  aquamarine: "#7fffd4",
  azure: "#f0ffff",
  beige: "#f5f5dc",
  bisque: "#ffe4c4",
  black: "#000000",
  blanchedalmond: "#ffebcd",
  blue: "#0000ff",
  blueviolet: "#8a2be2",
  brown: "#a52a2a",
  burlywood: "#deb887",
  cadetblue: "#5f9ea0",
  chartreuse: "#7fff00",
  chocolate: "#d2691e",
  coral: "#ff7f50",
  cornflowerblue: "#6495ed",
  cornsilk: "#fff8dc",
  crimson: "#dc143c",
  cyan: "#00ffff",
  darkblue: "#00008b",
  darkcyan: "#008b8b",
  darkgoldenrod: "#b8860b",
  darkgray: "#a9a9a9",
  darkgrey: "#a9a9a9",
  darkgreen: "#006400",
  darkkhaki: "#bdb76b",
  darkmagenta: "#8b008b",
  darkolivegreen: "#556b2f",
  darkorange: "#ff8c00",
  darkorchid: "#9932cc",
  darkred: "#8b0000",
  darksalmon: "#e9967a",
  darkseagreen: "#8fbc8f",
  darkslateblue: "#483d8b",
  darkslategray: "#2f4f4f",
  darkslategrey: "#2f4f4f",
  darkturquoise: "#00ced1",
  darkviolet: "#9400d3",
  deeppink: "#ff1493",
  deepskyblue: "#00bfff",
  dimgray: "#696969",
  dimgrey: "#696969",
  dodgerblue: "#1e90ff",
  firebrick: "#b22222",
  floralwhite: "#fffaf0",
  forestgreen: "#228b22",
  fuchsia: "#ff00ff",
  gainsboro: "#dcdcdc",
  ghostwhite: "#f8f8ff",
  gold: "#ffd700",
  goldenrod: "#daa520",
  gray: "#808080",
  grey: "#808080",
  green: "#008000",
  greenyellow: "#adff2f",
  honeydew: "#f0fff0",
  hotpink: "#ff69b4",
  indianred: "#cd5c5c",
  indigo: "#4b0082",
  ivory: "#fffff0",
  khaki: "#f0e68c",
  lavender: "#e6e6fa",
  lavenderblush: "#fff0f5",
  lawngreen: "#7cfc00",
  lemonchiffon: "#fffacd",
  lightblue: "#add8e6",
  lightcoral: "#f08080",
  lightcyan: "#e0ffff",
  lightgoldenrodyellow: "#fafad2",
  lightgray: "#d3d3d3",
  lightgrey: "#d3d3d3",
  lightgreen: "#90ee90",
  lightpink: "#ffb6c1",
  lightsalmon: "#ffa07a",
  lightseagreen: "#20b2aa",
  lightskyblue: "#87cefa",
  lightslategray: "#778899",
  lightslategrey: "#778899",
  lightsteelblue: "#b0c4de",
  lightyellow: "#ffffe0",
  lime: "#00ff00",
  limegreen: "#32cd32",
  linen: "#faf0e6",
  magenta: "#ff00ff",
  maroon: "#800000",
  mediumaquamarine: "#66cdaa",
  mediumblue: "#0000cd",
  mediumorchid: "#ba55d3",
  mediumpurple: "#9370db",
  mediumseagreen: "#3cb371",
  mediumslateblue: "#7b68ee",
  mediumspringgreen: "#00fa9a",
  mediumturquoise: "#48d1cc",
  mediumvioletred: "#c71585",
  midnightblue: "#191970",
  mintcream: "#f5fffa",
  mistyrose: "#ffe4e1",
  moccasin: "#ffe4b5",
  navajowhite: "#ffdead",
  navy: "#000080",
  oldlace: "#fdf5e6",
  olive: "#808000",
  olivedrab: "#6b8e23",
  orange: "#ffa500",
  orangered: "#ff4500",
  orchid: "#da70d6",
  palegoldenrod: "#eee8aa",
  palegreen: "#98fb98",
  paleturquoise: "#afeeee",
  palevioletred: "#db7093",
  papayawhip: "#ffefd5",
  peachpuff: "#ffdab9",
  peru: "#cd853f",
  pink: "#ffc0cb",
  plum: "#dda0dd",
  powderblue: "#b0e0e6",
  purple: "#800080",
  rebeccapurple: "#663399",
  red: "#ff0000",
  rosybrown: "#bc8f8f",
  royalblue: "#4169e1",
  saddlebrown: "#8b4513",
  salmon: "#fa8072",
  sandybrown: "#f4a460",
  seagreen: "#2e8b57",
  seashell: "#fff5ee",
  sienna: "#a0522d",
  silver: "#c0c0c0",
  skyblue: "#87ceeb",
  slateblue: "#6a5acd",
  slategray: "#708090",
  slategrey: "#708090",
  snow: "#fffafa",
  springgreen: "#00ff7f",
  steelblue: "#4682b4",
  tan: "#d2b48c",
  teal: "#008080",
  thistle: "#d8bfd8",
  tomato: "#ff6347",
  turquoise: "#40e0d0",
  violet: "#ee82ee",
  wheat: "#f5deb3",
  white: "#ffffff",
  whitesmoke: "#f5f5f5",
  yellow: "#ffff00",
  yellowgreen: "#9acd32",
};

type RGBA = {
  readonly r: number;
  readonly g: number;
  readonly b: number;
  readonly a: number;
};

/** Color returned for anything that cannot be parsed: opaque black. */
const BLACK: RGBA = { r: 0, g: 0, b: 0, a: 255 };
const TRANSPARENT: RGBA = { r: 0, g: 0, b: 0, a: 0 };

/** Upper bound on cached parse results, so per-point color strings cannot grow memory forever. */
const COLOR_CACHE_LIMIT = 1024;

function cacheResult(key: string, value: RGBA): RGBA {
  if (colorCache.size >= COLOR_CACHE_LIMIT) colorCache.clear();
  colorCache.set(key, value);
  return value;
}

/**
 * Normalizes a color value to hex format: `#rrggbb`, or `#rrggbbaa` when the color is not
 * fully opaque. Returns `fallback` for `undefined` or an empty string. Text that is not a
 * recognized color becomes black (`#000000`), like {@link parseHexColorToRGBA}.
 * @internal
 */
export function normalizeColor(c: Color | undefined, fallback: Color): Color {
  if (!c) return fallback;
  const s = c.trim();
  if (s.length === 0) return fallback;
  const hex8 = s.match(/^#([0-9a-fA-F]{8})$/);
  if (hex8 && hex8[1] !== undefined) {
    return `#${hex8[1].toLowerCase()}`;
  }

  // Convert to RGBA then back to hex for consistent output
  const rgba = parseHexColorToRGBA(s);
  const r = rgba.r.toString(16).padStart(2, "0");
  const g = rgba.g.toString(16).padStart(2, "0");
  const b = rgba.b.toString(16).padStart(2, "0");

  if (rgba.a === 255) {
    return `#${r}${g}${b}`;
  }
  const a = rgba.a.toString(16).padStart(2, "0");
  return `#${r}${g}${b}${a}`;
}

const NUMBER_TOKEN = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/;

/** Parses "12", "12.5" or "50%"; percentages are scaled so 100% equals `percentBase`. */
function parseComponent(token: string, percentBase: number): number | null {
  const isPercent = token.endsWith("%");
  const body = isPercent ? token.slice(0, -1) : token;
  if (!NUMBER_TOKEN.test(body)) return null;
  const v = Number.parseFloat(body);
  if (!Number.isFinite(v)) return null;
  return isPercent ? (v / 100) * percentBase : v;
}

function parseHue(token: string): number | null {
  const body = token.endsWith("deg") ? token.slice(0, -3) : token;
  if (!NUMBER_TOKEN.test(body)) return null;
  const v = Number.parseFloat(body);
  return Number.isFinite(v) ? v : null;
}

function hue2rgb(p: number, q: number, t: number): number {
  let u = t;
  if (u < 0) u += 1;
  if (u > 1) u -= 1;
  if (u < 1 / 6) return p + (q - p) * 6 * u;
  if (u < 1 / 2) return q;
  if (u < 2 / 3) return p + (q - p) * (2 / 3 - u) * 6;
  return p;
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  if (s === 0) {
    const v = Math.round(l * 255);
    return [v, v, v];
  }
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  return [
    Math.round(hue2rgb(p, q, h + 1 / 3) * 255),
    Math.round(hue2rgb(p, q, h) * 255),
    Math.round(hue2rgb(p, q, h - 1 / 3) * 255),
  ];
}

const clampByte = (value: number): number => Math.min(255, Math.max(0, Math.round(value)));
const clampUnit = (value: number): number => Math.min(1, Math.max(0, value));

function parseFunctional(s: string): RGBA | null {
  const m = s.match(/^(rgba?|hsla?)\(\s*([^()]*?)\s*\)$/);
  if (!m) return null;
  const kind = m[1] ?? "";
  const args = (m[2] ?? "").split(/\s*[,/]\s*|\s+/).filter((a) => a.length > 0);
  if (args.length !== 3 && args.length !== 4) return null;

  let alpha = 1;
  if (args.length === 4) {
    const a = parseComponent(args[3] ?? "", 1);
    if (a === null) return null;
    alpha = clampUnit(a);
  }
  const a255 = Math.round(alpha * 255);

  if (kind.startsWith("rgb")) {
    const r = parseComponent(args[0] ?? "", 255);
    const g = parseComponent(args[1] ?? "", 255);
    const b = parseComponent(args[2] ?? "", 255);
    if (r === null || g === null || b === null) return null;
    return { r: clampByte(r), g: clampByte(g), b: clampByte(b), a: a255 };
  }

  const hue = parseHue(args[0] ?? "");
  const sat = parseComponent(args[1] ?? "", 100);
  const light = parseComponent(args[2] ?? "", 100);
  if (hue === null || sat === null || light === null) return null;
  const h = (((hue % 360) + 360) % 360) / 360;
  const [r, g, b] = hslToRgb(h, clampUnit(sat / 100), clampUnit(light / 100));
  return { r: clampByte(r), g: clampByte(g), b: clampByte(b), a: a255 };
}

function parseColorString(s: string): RGBA | null {
  if (s === "transparent") return TRANSPARENT;

  const named = Object.hasOwn(namedColors, s) ? namedColors[s] : undefined;
  if (named !== undefined) return parseColorString(named);

  if (s.startsWith("#")) {
    let hex = s.slice(1);
    if (!/^[0-9a-f]+$/.test(hex)) return null;
    if (hex.length === 3 || hex.length === 4) {
      hex = Array.from(hex, (ch) => ch + ch).join("");
    }
    if (hex.length !== 6 && hex.length !== 8) return null;
    return {
      r: Number.parseInt(hex.slice(0, 2), 16),
      g: Number.parseInt(hex.slice(2, 4), 16),
      b: Number.parseInt(hex.slice(4, 6), 16),
      a: hex.length === 8 ? Number.parseInt(hex.slice(6, 8), 16) : 255,
    };
  }

  return parseFunctional(s);
}

/**
 * Parses a CSS color to 8-bit RGBA (alpha 0-255).
 *
 * Supported forms: `#rgb`, `#rgba`, `#rrggbb`, `#rrggbbaa`, `rgb()`/`rgba()` (numbers or
 * percentages, comma or space separated, optional `/ alpha`), `hsl()`/`hsla()`, the 148 CSS
 * color names and `transparent`. Matching is case-insensitive. Anything else, including a
 * non-string value, yields opaque black.
 * @internal
 */
export function parseHexColorToRGBA(c: Color): {
  readonly r: number;
  readonly g: number;
  readonly b: number;
  readonly a: number;
} {
  if (typeof c !== "string") return BLACK;
  const cached = colorCache.get(c);
  if (cached) return cached;
  return cacheResult(c, parseColorString(c.trim().toLowerCase()) ?? BLACK);
}

/**
 * Named color palettes for plotting.
 */
const palettes: Readonly<Record<string, readonly string[]>> = {
  tab10: [
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
  ],
  Set1: [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf",
    "#999999",
  ],
  Set2: ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3"],
  Paired: [
    "#a6cee3",
    "#1f78b4",
    "#b2df8a",
    "#33a02c",
    "#fb9a99",
    "#e31a1c",
    "#fdbf6f",
    "#ff7f00",
    "#cab2d6",
    "#6a3d9a",
    "#ffff99",
    "#b15928",
  ],
  viridis: [
    "#440154",
    "#482878",
    "#3e4989",
    "#31688e",
    "#26828e",
    "#1f9e89",
    "#35b779",
    "#6ece58",
    "#b5de2b",
    "#fde725",
  ],
  plasma: [
    "#0d0887",
    "#46039f",
    "#7201a8",
    "#9c179e",
    "#bd3786",
    "#d8576b",
    "#ed7953",
    "#fb9f3a",
    "#fdca26",
    "#f0f921",
  ],
  inferno: [
    "#000004",
    "#1b0c41",
    "#4a0c6b",
    "#781c6d",
    "#a52c60",
    "#cf4446",
    "#ed6925",
    "#fb9b06",
    "#f7d13d",
    "#fcffa4",
  ],
  magma: [
    "#000004",
    "#180f3d",
    "#440f76",
    "#721f81",
    "#9e2f7f",
    "#cd4071",
    "#f1605d",
    "#fd9668",
    "#feca8d",
    "#fcfdbf",
  ],
  cividis: [
    "#00224e",
    "#123570",
    "#3b496c",
    "#575d6d",
    "#707173",
    "#8a8678",
    "#a59c74",
    "#c3b369",
    "#e1cc55",
    "#fee838",
  ],
};

for (const colors of Object.values(palettes)) Object.freeze(colors);

function lookupPalette(name: string): readonly string[] {
  const p = Object.hasOwn(palettes, name) ? palettes[name] : undefined;
  if (!p) {
    const available = Object.keys(palettes).join(", ");
    throw new InvalidParameterError(
      `Unknown palette '${name}'. Available: ${available}`,
      "name",
      name
    );
  }
  return p;
}

/**
 * Get a named color palette.
 *
 * @param name - Palette name (tab10, Set1, Set2, Paired, viridis, plasma, inferno, magma, cividis)
 * @returns A new array of hex color strings; changing it does not affect the palette
 * @throws {InvalidParameterError} If the palette name is unknown.
 */
export function getPalette(name: string): readonly string[] {
  return lookupPalette(name).slice();
}

/**
 * Get a color from a named palette by index (wraps around in both directions, so index -1 is
 * the last color).
 *
 * @param name - Palette name
 * @param index - Integer color index
 * @returns Hex color string
 * @throws {InvalidParameterError} If the palette name is unknown or `index` is not an integer.
 */
export function getPaletteColor(name: string, index: number): string {
  const p = lookupPalette(name);
  if (!Number.isInteger(index)) {
    throw new InvalidParameterError(`index must be an integer; received ${index}`, "index", index);
  }
  return p[((index % p.length) + p.length) % p.length] ?? "#000000";
}

/**
 * List all available palette names.
 */
export function listPalettes(): string[] {
  return Object.keys(palettes);
}
