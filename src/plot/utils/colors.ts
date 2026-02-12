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

/**
 * Normalizes a color value to hex format.
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

/**
 * Parses color to RGBA.
 * @internal
 */
export function parseHexColorToRGBA(c: Color): {
  readonly r: number;
  readonly g: number;
  readonly b: number;
  readonly a: number;
} {
  const cached = colorCache.get(c);
  if (cached) return cached;

  const s = c.trim().toLowerCase();
  const clampByte = (value: number): number => {
    if (!Number.isFinite(value)) return 0;
    return Math.min(255, Math.max(0, Math.round(value)));
  };
  const clampAlpha = (value: number): number => {
    if (!Number.isFinite(value)) return 1;
    return Math.min(1, Math.max(0, value));
  };

  if (namedColors[s]) {
    const namedColor = namedColors[s];
    if (typeof namedColor === "string") {
      const result = parseHexColorToRGBA(namedColor);
      colorCache.set(c, result);
      return result;
    }
    const result = { r: 0, g: 0, b: 0, a: 255 };
    colorCache.set(c, result);
    return result;
  }

  if (s.startsWith("#")) {
    const hex = s.slice(1);
    if (hex.length === 6 || hex.length === 8) {
      const r = Number.parseInt(hex.slice(0, 2), 16);
      const g = Number.parseInt(hex.slice(2, 4), 16);
      const b = Number.parseInt(hex.slice(4, 6), 16);
      const a = hex.length === 8 ? Number.parseInt(hex.slice(6, 8), 16) : 255;
      if (Number.isFinite(r) && Number.isFinite(g) && Number.isFinite(b) && Number.isFinite(a)) {
        const result = { r, g, b, a };
        colorCache.set(c, result);
        return result;
      }
    }
  }

  const rgbMatch = s.match(/^rgba?\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*([\d.]+)\s*)?\)$/);
  if (rgbMatch) {
    const rStr = rgbMatch[1];
    const gStr = rgbMatch[2];
    const bStr = rgbMatch[3];
    if (!rStr || !gStr || !bStr) {
      const result = { r: 0, g: 0, b: 0, a: 255 };
      colorCache.set(c, result);
      return result;
    }
    const r = clampByte(Number.parseInt(rStr, 10));
    const g = clampByte(Number.parseInt(gStr, 10));
    const b = clampByte(Number.parseInt(bStr, 10));
    const alpha = rgbMatch[4] ? clampAlpha(Number.parseFloat(rgbMatch[4])) : 1;
    const result = { r, g, b, a: Math.round(alpha * 255) };
    colorCache.set(c, result);
    return result;
  }

  const hslMatch = s.match(
    /^hsla?\s*\(\s*([\d.]+)\s*,\s*([\d.]+)%\s*,\s*([\d.]+)%\s*(?:,\s*([\d.]+)\s*)?\)$/
  );
  if (hslMatch) {
    const hRaw = hslMatch[1];
    const sRaw = hslMatch[2];
    const lRaw = hslMatch[3];
    if (!hRaw || !sRaw || !lRaw) {
      const result = { r: 0, g: 0, b: 0, a: 255 };
      colorCache.set(c, result);
      return result;
    }
    const hueDegrees = Number.parseFloat(hRaw);
    const sat = Number.parseFloat(sRaw);
    const light = Number.parseFloat(lRaw);
    if (!Number.isFinite(hueDegrees) || !Number.isFinite(sat) || !Number.isFinite(light)) {
      const result = { r: 0, g: 0, b: 0, a: 255 };
      colorCache.set(c, result);
      return result;
    }
    const h = (((hueDegrees % 360) + 360) % 360) / 360;
    const sl = Math.min(1, Math.max(0, sat / 100));
    const l = Math.min(1, Math.max(0, light / 100));
    const alpha = hslMatch[4] ? clampAlpha(Number.parseFloat(hslMatch[4])) : 1;

    const hslToRgb = (h: number, s: number, l: number): [number, number, number] => {
      let r: number, g: number, b: number;
      if (s === 0) {
        r = g = b = l;
      } else {
        const hue2rgb = (p: number, q: number, t: number): number => {
          if (t < 0) t += 1;
          if (t > 1) t -= 1;
          if (t < 1 / 6) return p + (q - p) * 6 * t;
          if (t < 1 / 2) return q;
          if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
          return p;
        };
        const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        const p = 2 * l - q;
        r = hue2rgb(p, q, h + 1 / 3);
        g = hue2rgb(p, q, h);
        b = hue2rgb(p, q, h - 1 / 3);
      }
      return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
    };

    const [r, g, b] = hslToRgb(h, sl, l);
    const result = {
      r: clampByte(r),
      g: clampByte(g),
      b: clampByte(b),
      a: Math.round(alpha * 255),
    };
    colorCache.set(c, result);
    return result;
  }

  const result = { r: 0, g: 0, b: 0, a: 255 };
  colorCache.set(c, result);
  return result;
}
