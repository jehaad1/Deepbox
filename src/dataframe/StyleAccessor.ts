/**
 * DataFrame conditional formatting / styling accessor.
 *
 * Provides a pandas-like `df.style` API for applying conditional
 * formatting rules and rendering styled HTML tables.
 *
 * @module dataframe/StyleAccessor
 * @see {@link https://deepbox.dev/docs/dataframe-overview | Deepbox documentation}
 */

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

/**
 * Style accessor for DataFrame.
 *
 * Access via `df.style`. Allows chaining conditional formatting
 * rules and rendering to HTML or plain-text tables.
 *
 * @example
 * ```ts
 * import { DataFrame } from 'deepbox/dataframe';
 *
 * const df = new DataFrame({ a: [1, 2, 3], b: [4, 5, 6] });
 * const html = df.style
 *   .highlight_max({ color: 'green' })
 *   .highlight_min({ color: 'red' })
 *   .toHTML();
 * ```
 */
export class StyleAccessor {
  private readonly getData: () => {
    readonly columns: string[];
    readonly getColumn: (name: string) => unknown[];
    readonly nRows: number;
  };
  private readonly rules: StyleFunction[] = [];
  private formatters: Map<string, (value: unknown) => string> = new Map();
  private caption_?: string;

  /** @internal */
  constructor(
    getData: () => {
      readonly columns: string[];
      readonly getColumn: (name: string) => unknown[];
      readonly nRows: number;
    }
  ) {
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
   * Highlight the maximum value in each column.
   */
  highlight_max(style: CellStyle = { backgroundColor: "#d4edda" }): this {
    const data = this.getData();
    const maxMap = new Map<number, number>();
    for (let c = 0; c < data.columns.length; c++) {
      const colName = data.columns[c]!;
      const colData = data.getColumn(colName);
      let maxVal = -Infinity;
      let maxRow = 0;
      for (let r = 0; r < colData.length; r++) {
        const v = Number(colData[r]);
        if (Number.isFinite(v) && v > maxVal) {
          maxVal = v;
          maxRow = r;
        }
      }
      if (maxVal > -Infinity) maxMap.set(c, maxRow);
    }
    this.rules.push((_value, row, col) => {
      if (maxMap.get(col) === row) return style;
      return {};
    });
    return this;
  }

  /**
   * Highlight the minimum value in each column.
   */
  highlight_min(style: CellStyle = { backgroundColor: "#f8d7da" }): this {
    const data = this.getData();
    const minMap = new Map<number, number>();
    for (let c = 0; c < data.columns.length; c++) {
      const colName = data.columns[c]!;
      const colData = data.getColumn(colName);
      let minVal = Infinity;
      let minRow = 0;
      for (let r = 0; r < colData.length; r++) {
        const v = Number(colData[r]);
        if (Number.isFinite(v) && v < minVal) {
          minVal = v;
          minRow = r;
        }
      }
      if (minVal < Infinity) minMap.set(c, minRow);
    }
    this.rules.push((_value, row, col) => {
      if (minMap.get(col) === row) return style;
      return {};
    });
    return this;
  }

  /**
   * Highlight null/NaN values.
   */
  highlight_null(style: CellStyle = { backgroundColor: "#fff3cd" }): this {
    this.rules.push((value, _row, _col) => {
      if (value === null || value === undefined) return style;
      if (typeof value === "number" && !Number.isFinite(value)) return style;
      return {};
    });
    return this;
  }

  /**
   * Apply a color gradient (background) based on numeric values.
   *
   * @param low - CSS color for the lowest value (default: white)
   * @param high - CSS color for the highest value (default: blue)
   */
  background_gradient(low = "#ffffff", high = "#4472c4"): this {
    const data = this.getData();
    const colRanges = new Map<number, { min: number; max: number }>();
    for (let c = 0; c < data.columns.length; c++) {
      const colName = data.columns[c]!;
      const colData = data.getColumn(colName);
      let min = Infinity;
      let max = -Infinity;
      for (const v of colData) {
        const n = Number(v);
        if (Number.isFinite(n)) {
          if (n < min) min = n;
          if (n > max) max = n;
        }
      }
      if (min < Infinity && max > -Infinity) {
        colRanges.set(c, { min, max });
      }
    }

    const lowRGB = parseSimpleColor(low);
    const highRGB = parseSimpleColor(high);

    this.rules.push((value, _row, col) => {
      const range = colRanges.get(col);
      if (!range) return {};
      const n = Number(value);
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
   * Apply conditional bar rendering (inline bars in cells).
   */
  bar(color = "#5fba7d"): this {
    const data = this.getData();
    const colRanges = new Map<number, { min: number; max: number }>();
    for (let c = 0; c < data.columns.length; c++) {
      const colName = data.columns[c]!;
      const colData = data.getColumn(colName);
      let min = Infinity;
      let max = -Infinity;
      for (const v of colData) {
        const n = Number(v);
        if (Number.isFinite(n)) {
          if (n < min) min = n;
          if (n > max) max = n;
        }
      }
      if (min < Infinity) colRanges.set(c, { min, max });
    }

    this.rules.push((value, _row, col) => {
      const range = colRanges.get(col);
      if (!range) return {};
      const n = Number(value);
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
   * Set a formatter for a specific column.
   */
  format(column: string, formatter: (value: unknown) => string): this {
    this.formatters.set(column, formatter);
    return this;
  }

  /**
   * Render the styled DataFrame as an HTML table string.
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
      lines.push(`    <th>${r}</th>`);
      for (let c = 0; c < data.columns.length; c++) {
        const value = columnData[c]![r];
        const colName = data.columns[c]!;

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

        const formatter = this.formatters.get(colName);
        const display = formatter ? formatter(value) : String(value ?? "");

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
   * Render a plain-text representation with ANSI color codes.
   *
   * Note: Only supports basic color/bold/italic formatting.
   * Returns a plain-text table string with ANSI escape codes.
   */
  toANSI(): string {
    const data = this.getData();
    const columnData: unknown[][] = data.columns.map((c) => data.getColumn(c));

    // Compute column widths
    const widths: number[] = data.columns.map((c) => c.length);
    for (let c = 0; c < data.columns.length; c++) {
      for (let r = 0; r < data.nRows; r++) {
        const value = columnData[c]![r];
        const colName = data.columns[c]!;
        const formatter = this.formatters.get(colName);
        const display = formatter ? formatter(value) : String(value ?? "");
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
        const value = columnData[c]![r];
        const colName = data.columns[c]!;
        const formatter = this.formatters.get(colName);
        const display = formatter ? formatter(value) : String(value ?? "");
        const padded = display.padEnd(widths[c] ?? 0);

        // Apply ANSI styles
        let styled = padded;
        for (const rule of this.rules) {
          const s = rule(value, r, c);
          if (s.fontWeight === "bold") styled = `\x1b[1m${styled}\x1b[0m`;
          if (s.fontStyle === "italic") styled = `\x1b[3m${styled}\x1b[0m`;
          if (s.color === "red") styled = `\x1b[31m${styled}\x1b[0m`;
          if (s.color === "green") styled = `\x1b[32m${styled}\x1b[0m`;
          if (s.color === "yellow") styled = `\x1b[33m${styled}\x1b[0m`;
          if (s.color === "blue") styled = `\x1b[34m${styled}\x1b[0m`;
        }
        parts.push(styled);
      }
      lines.push(parts.join("  "));
    }

    return lines.join("\n");
  }
}

function escapeHTML(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function parseSimpleColor(hex: string): { r: number; g: number; b: number } {
  const h = hex.replace("#", "");
  if (h.length === 3) {
    return {
      r: parseInt(h[0]! + h[0]!, 16),
      g: parseInt(h[1]! + h[1]!, 16),
      b: parseInt(h[2]! + h[2]!, 16),
    };
  }
  return {
    r: parseInt(h.slice(0, 2), 16),
    g: parseInt(h.slice(2, 4), 16),
    b: parseInt(h.slice(4, 6), 16),
  };
}
