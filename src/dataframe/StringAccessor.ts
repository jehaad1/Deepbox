/**
 * @see {@link https://deepbox.dev/docs/dataframe-overview | Deepbox documentation}
 */

import { DataValidationError, InvalidParameterError } from "../core/errors/index";
import { DataFrame } from "./DataFrame";
import { Series } from "./Series";
import type { SeriesOptions } from "./types";

/**
 * Coerce a Series element to string or null.
 * Null/undefined become null; everything else is validated as string.
 */
function toStringOrNull(value: unknown, index: number): string | null {
  if (value === null || value === undefined) return null;
  if (typeof value === "string") return value;
  throw new DataValidationError(
    `StringAccessor: element at index ${index} is not a string (got ${typeof value})`
  );
}

/**
 * Build a new Series by mapping each element through a function.
 * Null inputs produce null outputs.
 */
function mapStringSeries<T>(
  data: readonly unknown[],
  indexLabels: readonly (string | number)[],
  name: string | undefined,
  fn: (value: string) => T
): Series<T | null> {
  const result: Array<T | null> = new Array(data.length);
  for (let i = 0; i < data.length; i++) {
    const s = toStringOrNull(data[i], i);
    result[i] = s === null ? null : fn(s);
  }
  const opts: SeriesOptions = { index: [...indexLabels] };
  if (name !== undefined) opts.name = name;
  return new Series(result, opts);
}

/**
 * String accessor for Series, providing vectorized string operations.
 *
 * Accessed via `series.str`. Operates element-wise on string Series.
 * Null/undefined elements propagate as null in the output.
 *
 * @example
 * ```ts
 * const s = new Series(['hello', 'WORLD', null]);
 * s.str.upper();       // Series(['HELLO', 'WORLD', null])
 * s.str.lower();       // Series(['hello', 'world', null])
 * s.str.contains('lo'); // Series([true, false, null])
 * ```
 */
export class StringAccessor {
  private readonly _data: readonly unknown[];
  private readonly _index: readonly (string | number)[];
  private readonly _name: string | undefined;

  constructor(series: Series<unknown>) {
    this._data = series.data;
    this._index = series.index;
    this._name = series.name;
  }

  // ─── Case transforms ───────────────────────────────────────────

  upper(): Series<string | null> {
    return mapStringSeries(this._data, this._index, this._name, (s) => s.toUpperCase());
  }

  lower(): Series<string | null> {
    return mapStringSeries(this._data, this._index, this._name, (s) => s.toLowerCase());
  }

  title(): Series<string | null> {
    return mapStringSeries(this._data, this._index, this._name, (s) =>
      s.replace(/\b\w/g, (c) => c.toUpperCase())
    );
  }

  capitalize(): Series<string | null> {
    return mapStringSeries(this._data, this._index, this._name, (s) =>
      s.length === 0 ? s : s.charAt(0).toUpperCase() + s.slice(1).toLowerCase()
    );
  }

  // ─── Trimming ──────────────────────────────────────────────────

  strip(chars?: string): Series<string | null> {
    if (chars === undefined) {
      return mapStringSeries(this._data, this._index, this._name, (s) => s.trim());
    }
    const pattern = buildCharClassPattern(chars);
    const re = new RegExp(`^${pattern}+|${pattern}+$`, "g");
    return mapStringSeries(this._data, this._index, this._name, (s) => s.replace(re, ""));
  }

  lstrip(chars?: string): Series<string | null> {
    if (chars === undefined) {
      return mapStringSeries(this._data, this._index, this._name, (s) => s.replace(/^\s+/, ""));
    }
    const pattern = buildCharClassPattern(chars);
    const re = new RegExp(`^${pattern}+`);
    return mapStringSeries(this._data, this._index, this._name, (s) => s.replace(re, ""));
  }

  rstrip(chars?: string): Series<string | null> {
    if (chars === undefined) {
      return mapStringSeries(this._data, this._index, this._name, (s) => s.replace(/\s+$/, ""));
    }
    const pattern = buildCharClassPattern(chars);
    const re = new RegExp(`${pattern}+$`);
    return mapStringSeries(this._data, this._index, this._name, (s) => s.replace(re, ""));
  }

  // ─── Search / Match ────────────────────────────────────────────

  contains(pat: string | RegExp, regex: boolean = true): Series<boolean | null> {
    const re = toRegExp(pat, regex, "contains");
    return mapStringSeries(this._data, this._index, this._name, (s) => re.test(s));
  }

  startswith(pat: string): Series<boolean | null> {
    validateString(pat, "pat");
    return mapStringSeries(this._data, this._index, this._name, (s) => s.startsWith(pat));
  }

  endswith(pat: string): Series<boolean | null> {
    validateString(pat, "pat");
    return mapStringSeries(this._data, this._index, this._name, (s) => s.endsWith(pat));
  }

  match(pat: string | RegExp): Series<boolean | null> {
    const re = toRegExp(pat, true, "match");
    return mapStringSeries(this._data, this._index, this._name, (s) => re.test(s));
  }

  // ─── Replace / Split ──────────────────────────────────────────

  replace(pat: string | RegExp, repl: string, regex: boolean = true): Series<string | null> {
    if (regex) {
      const re = toRegExp(pat, true, "replace");
      const globalRe = re.global ? re : new RegExp(re.source, `${re.flags}g`);
      return mapStringSeries(this._data, this._index, this._name, (s) => s.replace(globalRe, repl));
    }
    validateString(pat, "pat");
    const literal = pat;
    return mapStringSeries(this._data, this._index, this._name, (s) => s.split(literal).join(repl));
  }

  split(pat: string | RegExp = /\s+/, n?: number): Series<string[] | null> {
    return mapStringSeries(this._data, this._index, this._name, (s) => {
      const parts = s.split(pat);
      if (n !== undefined && n >= 0 && parts.length > n + 1) {
        const first = parts.slice(0, n);
        const rest = parts.slice(n).join(typeof pat === "string" ? pat : " ");
        return [...first, rest];
      }
      return parts;
    });
  }

  // ─── Length / Slice ────────────────────────────────────────────

  len(): Series<number | null> {
    return mapStringSeries(this._data, this._index, this._name, (s) => s.length);
  }

  slice(start?: number, stop?: number): Series<string | null> {
    return mapStringSeries(this._data, this._index, this._name, (s) => s.slice(start, stop));
  }

  // ─── Extract / FindAll ─────────────────────────────────────────

  extract(pat: string | RegExp, group: number = 0): Series<string | null> {
    const re = toRegExp(pat, true, "extract");
    return mapStringSeries(this._data, this._index, this._name, (s) => {
      const m = re.exec(s);
      if (m === null) return null;
      const val = m[group];
      return val === undefined ? null : val;
    });
  }

  findall(pat: string | RegExp): Series<string[] | null> {
    const re = toRegExp(pat, true, "findall");
    const globalRe = re.global ? re : new RegExp(re.source, `${re.flags}g`);
    return mapStringSeries(this._data, this._index, this._name, (s) => {
      const matches = s.match(globalRe);
      return matches === null ? [] : [...matches];
    });
  }

  // ─── Padding / Alignment ───────────────────────────────────────

  pad(
    width: number,
    side: "left" | "right" | "both" = "left",
    fillchar: string = " "
  ): Series<string | null> {
    validatePositiveInt(width, "width");
    if (fillchar.length !== 1) {
      throw new InvalidParameterError("fillchar must be a single character", "fillchar", fillchar);
    }
    return mapStringSeries(this._data, this._index, this._name, (s) => {
      if (s.length >= width) return s;
      const diff = width - s.length;
      if (side === "left") return fillchar.repeat(diff) + s;
      if (side === "right") return s + fillchar.repeat(diff);
      // both
      const leftPad = Math.floor(diff / 2);
      const rightPad = diff - leftPad;
      return fillchar.repeat(leftPad) + s + fillchar.repeat(rightPad);
    });
  }

  center(width: number, fillchar: string = " "): Series<string | null> {
    return this.pad(width, "both", fillchar);
  }

  zfill(width: number): Series<string | null> {
    validatePositiveInt(width, "width");
    return mapStringSeries(this._data, this._index, this._name, (s) => {
      if (s.length >= width) return s;
      // Preserve leading sign
      if (s.length > 0 && (s.charAt(0) === "-" || s.charAt(0) === "+")) {
        return s.charAt(0) + s.slice(1).padStart(width - 1, "0");
      }
      return s.padStart(width, "0");
    });
  }

  // ─── Concatenation ─────────────────────────────────────────────

  cat(sep: string = ""): string {
    const parts: string[] = [];
    for (let i = 0; i < this._data.length; i++) {
      const s = toStringOrNull(this._data[i], i);
      if (s !== null) parts.push(s);
    }
    return parts.join(sep);
  }

  // ─── get_dummies ───────────────────────────────────────────────

  get_dummies(sep: string = "|"): DataFrame {
    validateString(sep, "sep");
    // Collect all unique tokens
    const allTokens = new Set<string>();
    const tokensByRow: Array<Set<string> | null> = new Array(this._data.length);

    for (let i = 0; i < this._data.length; i++) {
      const s = toStringOrNull(this._data[i], i);
      if (s === null) {
        tokensByRow[i] = null;
      } else {
        const tokens = new Set(s.split(sep));
        tokensByRow[i] = tokens;
        for (const t of tokens) allTokens.add(t);
      }
    }

    // Sort tokens for deterministic column order
    const sortedTokens = [...allTokens].sort();

    // Build column data
    const columns: Record<string, number[]> = {};
    for (const token of sortedTokens) {
      const col = new Array<number>(this._data.length);
      for (let i = 0; i < this._data.length; i++) {
        const row = tokensByRow[i];
        col[i] = row?.has(token) ? 1 : 0;
      }
      columns[token] = col;
    }

    return new DataFrame(columns, {
      index: [...this._index],
      columns: sortedTokens,
    });
  }

  // ─── Repeat ────────────────────────────────────────────────────

  repeat(times: number): Series<string | null> {
    validatePositiveInt(times, "times");
    return mapStringSeries(this._data, this._index, this._name, (s) => s.repeat(times));
  }

  // ─── Count occurrences ─────────────────────────────────────────

  count(pat: string | RegExp): Series<number | null> {
    const re = toRegExp(pat, true, "count");
    const globalRe = re.global ? re : new RegExp(re.source, `${re.flags}g`);
    return mapStringSeries(this._data, this._index, this._name, (s) => {
      const matches = s.match(globalRe);
      return matches === null ? 0 : matches.length;
    });
  }

  // ─── Boolean checks ───────────────────────────────────────────

  isalpha(): Series<boolean | null> {
    return mapStringSeries(
      this._data,
      this._index,
      this._name,
      (s) => s.length > 0 && /^[a-zA-Z]+$/.test(s)
    );
  }

  isdigit(): Series<boolean | null> {
    return mapStringSeries(
      this._data,
      this._index,
      this._name,
      (s) => s.length > 0 && /^\d+$/.test(s)
    );
  }

  isalnum(): Series<boolean | null> {
    return mapStringSeries(
      this._data,
      this._index,
      this._name,
      (s) => s.length > 0 && /^[a-zA-Z0-9]+$/.test(s)
    );
  }

  isspace(): Series<boolean | null> {
    return mapStringSeries(
      this._data,
      this._index,
      this._name,
      (s) => s.length > 0 && /^\s+$/.test(s)
    );
  }

  isupper(): Series<boolean | null> {
    return mapStringSeries(
      this._data,
      this._index,
      this._name,
      (s) => s.length > 0 && s === s.toUpperCase() && s !== s.toLowerCase()
    );
  }

  islower(): Series<boolean | null> {
    return mapStringSeries(
      this._data,
      this._index,
      this._name,
      (s) => s.length > 0 && s === s.toLowerCase() && s !== s.toUpperCase()
    );
  }
}

// ─── Internal helpers ───────────────────────────────────────────

function validateString(value: unknown, name: string): asserts value is string {
  if (typeof value !== "string") {
    throw new InvalidParameterError(`${name} must be a string`, name, value);
  }
}

function validatePositiveInt(value: number, name: string): void {
  if (!Number.isFinite(value) || !Number.isInteger(value) || value < 0) {
    throw new InvalidParameterError(`${name} must be a non-negative integer`, name, value);
  }
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function buildCharClassPattern(chars: string): string {
  return `[${chars.replace(/[\]\\^-]/g, "\\$&")}]`;
}

function toRegExp(pat: string | RegExp, regex: boolean, functionName: string): RegExp {
  if (pat instanceof RegExp) return pat;
  if (!regex) return new RegExp(escapeRegExp(pat));
  try {
    return new RegExp(pat);
  } catch {
    throw new InvalidParameterError(`${functionName}: invalid regex pattern`, "pat", pat);
  }
}
