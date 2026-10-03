/**
 * DataFrame conditional formatting / styling accessor.
 *
 * Provides a pandas-like `df.style` API for applying conditional
 * formatting rules and rendering styled HTML tables.
 *
 * @module dataframe/StyleAccessor
 * @see {@link https://deepbox.dev/docs/dataframe-overview | Deepbox documentation}
 */

import { InvalidParameterError } from "../core/errors/index";

/**
 * A single cell style rule.
 */
export type CellStyle = {
  readonly color?: string;
  readonly backgroundColor?: string;
  readonly fontWeight?: string;
  readonly fontStyle?: string;
  readonly textAlign?: string;
};

/**
 * A formatting function that receives a cell value and returns a CSS-like style object.
 */
export type StyleFunction = (value: unknown, row: number, col: number) => CellStyle;

/** Columnar view of a DataFrame that the accessor reads from. */
type StyleData = {
  readonly columns: string[];
  readonly getColumn: (name: string) => unknown[];
  /** Row labels shown as row headers; row numbers are used when missing. */
  readonly index?: readonly (string | number)[];
  readonly nRows: number;
};

/**
 * The numeric value of a cell, or NaN when the cell is not a number. null,
 * undefined, strings, booleans and Dates are not numeric here (`Number(null)` would be 0).
 */
function cellNumber(value: unknown): number {
  if (typeof value === "number") return value;
  if (typeof value === "bigint") return Number(value);
  return Number.NaN;
}

/** Smallest and largest finite number of every column, keyed by column position. */
function columnRanges(data: StyleData): Map<number, { min: number; max: number }> {
  const ranges = new Map<number, { min: number; max: number }>();
  for (let c = 0; c < data.columns.length; c++) {
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    for (const v of data.getColumn(data.columns[c] as string)) {
      const n = cellNumber(v);
      if (Number.isFinite(n)) {
        if (n < min) min = n;
        if (n > max) max = n;
      }
    }
    if (min <= max) ranges.set(c, { min, max });
  }
  return ranges;
}

/**
 * Style accessor for DataFrame.
 *
 * Access via `df.style`. Allows chaining conditional formatting
 * rules and rendering to HTML or plain-text tables. Rules that look at a whole
 * column (`highlightMax`, `backgroundGradient`, ...) read the DataFrame when
 * they are added. Rules apply in the order they were added, and a later rule
 * overrides an earlier one for the same CSS property.
 *
 * @example
 * ```ts
 * import { DataFrame } from 'deepbox/dataframe';
 *
 * const df = new DataFrame({ a: [1, 2, 3], b: [4, 5, 6] });
 * const html = df.style
 *   .highlightMax({ color: 'green' })
 *   .highlightMin({ color: 'red' })
 *   .toHTML();
 * ```
 */
export class StyleAccessor {
  private readonly getData: () => StyleData;
  private readonly rules: StyleFunction[] = [];
  private formatters: Map<string, (value: unknown) => string> = new Map();
  private caption_?: string;

  /** @internal */
  constructor(getData: () => StyleData) {
    this.getData = getData;
  }

  /**
   * Set a caption for the styled table.
   */
  setCaption(caption: string): this {
    this.caption_ = caption;
    return this;
  }

  /**
   * Apply a generic style function to all cells.
   */
  applymap(fn: StyleFunction): this {
    this.rules.push(fn);
    return this;
  }

  /**
   * Alias of {@link StyleAccessor.applymap}, the name pandas uses since 2.1.
   */
  map(fn: StyleFunction): this {
    return this.applymap(fn);
  }

  /**
   * Highlight the maximum value in each column. When several cells share the
   * maximum, all of them are highlighted. Non-numeric cells are ignored.
   */
  highlightMax(style: CellStyle = { backgroundColor: "#d4edda" }): this {
    const ranges = columnRanges(this.getData());
    this.rules.push((value, _row, col) => {
      const range = ranges.get(col);
      return range !== undefined && cellNumber(value) === range.max ? style : {};
    });
    return this;
  }

  /**
   * Same as {@link StyleAccessor.highlightMax}.
   *
   * @deprecated Prefer {@link StyleAccessor.highlightMax}.
   */
  highlight_max(style: CellStyle = { backgroundColor: "#d4edda" }): this {
    return this.highlightMax(style);
  }

  /**
   * Highlight the minimum value in each column. When several cells share the
   * minimum, all of them are highlighted. Non-numeric cells are ignored.
   */
  highlightMin(style: CellStyle = { backgroundColor: "#f8d7da" }): this {
    const ranges = columnRanges(this.getData());
    this.rules.push((value, _row, col) => {
      const range = ranges.get(col);
      return range !== undefined && cellNumber(value) === range.min ? style : {};
    });
    return this;
  }

  /**
   * Same as {@link StyleAccessor.highlightMin}.
   *
   * @deprecated Prefer {@link StyleAccessor.highlightMin}.
   */
  highlight_min(style: CellStyle = { backgroundColor: "#f8d7da" }): this {
    return this.highlightMin(style);
  }

  /**
   * Highlight missing values: null, undefined and NaN. Infinite values are not
   * treated as missing.
   */
  highlightNull(style: CellStyle = { backgroundColor: "#fff3cd" }): this {
    this.rules.push((value, _row, _col) => {
      if (value === null || value === undefined) return style;
      if (typeof value === "number" && Number.isNaN(value)) return style;
      return {};
    });
    return this;
  }

  /**
   * Same as {@link StyleAccessor.highlightNull}.
   *
   * @deprecated Prefer {@link StyleAccessor.highlightNull}.
   */
  highlight_null(style: CellStyle = { backgroundColor: "#fff3cd" }): this {
    return this.highlightNull(style);
  }

  /**
   * Apply a color gradient (background) based on numeric values. Each column is
   * scaled between its own minimum and maximum. A column whose values are all
   * equal gets the midpoint color.
   *
   * @param low - Hex color for the lowest value, `#rgb` or `#rrggbb` (default: white)
   * @param high - Hex color for the highest value (default: blue)
   * @throws {InvalidParameterError} If a color is not a hex color
   */
  backgroundGradient(low = "#ffffff", high = "#4472c4"): this {
    const lowRGB = parseHexColor(low, "low");
    const highRGB = parseHexColor(high, "high");
    const ranges = columnRanges(this.getData());

    this.rules.push((value, _row, col) => {
      const range = ranges.get(col);
      if (!range) return {};
      const n = cellNumber(value);
      if (!Number.isFinite(n)) return {};
      const span = range.max - range.min;
      const t = span > 0 ? (n - range.min) / span : 0.5;
      const r = Math.round(lowRGB.r + t * (highRGB.r - lowRGB.r));
      const g = Math.round(lowRGB.g + t * (highRGB.g - lowRGB.g));
      const b = Math.round(lowRGB.b + t * (highRGB.b - lowRGB.b));
      return { backgroundColor: `rgb(${r},${g},${b})` };
    });
    return this;
  }

  /**
   * Same as {@link StyleAccessor.backgroundGradient}.
   *
   * @deprecated Prefer {@link StyleAccessor.backgroundGradient}.
   */
  background_gradient(low = "#ffffff", high = "#4472c4"): this {
    return this.backgroundGradient(low, high);
  }

  /**
   * Draw an inline bar in each numeric cell. The bar length is the value's
   * position between its column's minimum and maximum (50% when they are equal).
   *
   * @param color - CSS color of the bar (default: green)
   */
  bar(color = "#5fba7d"): this {
    if (typeof color !== "string" || color.trim() === "") {
      throw new InvalidParameterError("color must be a non-empty string", "color", color);
    }
    const ranges = columnRanges(this.getData());

    this.rules.push((value, _row, col) => {
      const range = ranges.get(col);
      if (!range) return {};
      const n = cellNumber(value);
      if (!Number.isFinite(n)) return {};
      const span = range.max - range.min;
      const pct = span > 0 ? ((n - range.min) / span) * 100 : 50;
      return {
        backgroundColor: `linear-gradient(90deg, ${color} ${pct.toFixed(1)}%, transparent ${pct.toFixed(1)}%)`,
      };
    });
    return this;
  }

  /**
   * Set a formatter for a specific column. The formatter receives the raw cell
   * value and returns the text to show.
   *
   * @throws {InvalidParameterError} If the column does not exist
   */
  format(column: string, formatter: (value: unknown) => string): this {
    const { columns } = this.getData();
    if (!columns.includes(column)) {
      throw new InvalidParameterError(
        `Column '${String(column)}' not found; available columns: ${columns.join(", ")}`,
        "column",
        column
      );
    }
    this.formatters.set(column, formatter);
    return this;
  }

  /**
   * Text shown for one cell: the column formatter if set, otherwise the value as
   * a string (empty for null and undefined, ISO 8601 for Dates).
   */
  private display(colName: string, value: unknown): string {
    const formatter = this.formatters.get(colName);
    if (formatter) return formatter(value);
    if (value === null || value === undefined) return "";
    if (value instanceof Date) {
      return Number.isNaN(value.getTime()) ? "Invalid Date" : value.toISOString();
    }
    return String(value);
  }

  /**
   * Render the styled DataFrame as an HTML table string. Cell text, column names
   * and the caption are HTML-escaped. Row headers show the index labels of the DataFrame
   * (the row number when the accessor has no index).
   */
  toHTML(): string {
    const data = this.getData();
    const lines: string[] = [];
    lines.push("<table>");

    if (this.caption_) {
      lines.push(`  <caption>${escapeHTML(this.caption_)}</caption>`);
    }

    // Header
    lines.push("  <thead><tr>");
    lines.push("    <th></th>");
    for (const col of data.columns) {
      lines.push(`    <th>${escapeHTML(col)}</th>`);
    }
    lines.push("  </tr></thead>");

    // Body
    lines.push("  <tbody>");
    const columnData: unknown[][] = data.columns.map((c) => data.getColumn(c));

    for (let r = 0; r < data.nRows; r++) {
      lines.push("  <tr>");
      lines.push(`    <th>${escapeHTML(String(data.index?.[r] ?? r))}</th>`);
      for (let c = 0; c < data.columns.length; c++) {
        const value = columnData[c]?.[r];
        const colName = data.columns[c] as string;

        // Compute merged styles
        const merged: Record<string, string> = {};
        for (const rule of this.rules) {
          const s = rule(value, r, c);
          if (s.color) merged["color"] = s.color;
          if (s.backgroundColor) merged["background"] = s.backgroundColor;
          if (s.fontWeight) merged["font-weight"] = s.fontWeight;
          if (s.fontStyle) merged["font-style"] = s.fontStyle;
          if (s.textAlign) merged["text-align"] = s.textAlign;
        }

        const styleStr = Object.entries(merged)
          .map(([k, v]) => `${k}:${v}`)
          .join(";");

        const display = this.display(colName, value);

        if (styleStr) {
          lines.push(`    <td style="${escapeHTML(styleStr)}">${escapeHTML(display)}</td>`);
        } else {
          lines.push(`    <td>${escapeHTML(display)}</td>`);
        }
      }
      lines.push("  </tr>");
    }
    lines.push("  </tbody>");
    lines.push("</table>");
    return lines.join("\n");
  }

  /**
   * Render a plain-text table with ANSI escape codes.
   *
   * Only bold, italic and the named text colors red, green, yellow, blue,
   * magenta, cyan, black, white and gray are used; other styles, including
   * background colors, are ignored. Unlike {@link StyleAccessor.toHTML}, no row
   * header column is printed.
   */
  toANSI(): string {
    const data = this.getData();
    const columnData: unknown[][] = data.columns.map((c) => data.getColumn(c));

    // Compute column widths
    const widths: number[] = data.columns.map((c) => c.length);
    for (let c = 0; c < data.columns.length; c++) {
      const colName = data.columns[c] as string;
      for (let r = 0; r < data.nRows; r++) {
        const display = this.display(colName, columnData[c]?.[r]);
        if (display.length > (widths[c] ?? 0)) widths[c] = display.length;
      }
    }

    const lines: string[] = [];

    // Header
    const headerParts = data.columns.map((col, i) => col.padEnd(widths[i] ?? 0));
    lines.push(headerParts.join("  "));
    lines.push(widths.map((w) => "-".repeat(w)).join("  "));

    // Rows
    for (let r = 0; r < data.nRows; r++) {
      const parts: string[] = [];
      for (let c = 0; c < data.columns.length; c++) {
        const value = columnData[c]?.[r];
        const colName = data.columns[c] as string;
        const padded = this.display(colName, value).padEnd(widths[c] ?? 0);

        // Apply ANSI styles
        let styled = padded;
        for (const rule of this.rules) {
          const s = rule(value, r, c);
          if (s.fontWeight === "bold") styled = `\x1b[1m${styled}\x1b[0m`;
          if (s.fontStyle === "italic") styled = `\x1b[3m${styled}\x1b[0m`;
          const code = s.color === undefined ? undefined : ANSI_COLORS[s.color];
          if (code !== undefined) styled = `\x1b[${code}m${styled}\x1b[0m`;
        }
        parts.push(styled);
      }
      lines.push(parts.join("  "));
    }

    return lines.join("\n");
  }
}

const ANSI_COLORS: Readonly<Record<string, number>> = {
  black: 30,
  red: 31,
  green: 32,
  yellow: 33,
  blue: 34,
  magenta: 35,
  cyan: 36,
  white: 37,
  gray: 90,
};

function escapeHTML(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/** Parse `#rgb` or `#rrggbb` (the `#` is optional). */
function parseHexColor(color: string, param: string): { r: number; g: number; b: number } {
  const h = typeof color === "string" ? color.trim().replace(/^#/, "") : "";
  if (!/^(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$/.test(h)) {
    throw new InvalidParameterError(
      `${param} must be a hex color such as '#fff' or '#4472c4'; received ${String(color)}`,
      param,
      color
    );
  }
  const full = h.length === 3 ? `${h[0]}${h[0]}${h[1]}${h[1]}${h[2]}${h[2]}` : h;
  return {
    r: Number.parseInt(full.slice(0, 2), 16),
    g: Number.parseInt(full.slice(2, 4), 16),
    b: Number.parseInt(full.slice(4, 6), 16),
  };
}
