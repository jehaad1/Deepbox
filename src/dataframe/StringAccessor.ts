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
 * Null/undefined elements propagate as null in the output. Any other
 * non-string element raises a DataValidationError.
 *
 * Lengths, padding widths and slice positions count Unicode code points (not
 * UTF-16 units), as in pandas, so an emoji counts as one character.
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

  private map<T>(fn: (value: string) => T): Series<T | null> {
    return mapStringSeries(this._data, this._index, this._name, fn);
  }

  // ─── Case transforms ───────────────────────────────────────────

  /** Convert every string to upper case. */
  upper(): Series<string | null> {
    return this.map((s) => s.toUpperCase());
  }

  /** Convert every string to lower case. */
  lower(): Series<string | null> {
    return this.map((s) => s.toLowerCase());
  }

  /**
   * Capitalize the first letter of every word and lower-case the rest, like
   * Python's `str.title()`. A word is a run of cased letters, so digits and
   * punctuation start a new word: `"it's 3rd"` becomes `"It'S 3Rd"`.
   */
  title(): Series<string | null> {
    return this.map(titleCase);
  }

  /** Upper-case the first character and lower-case the rest. */
  capitalize(): Series<string | null> {
    return this.map((s) => {
      if (s.length === 0) return s;
      const first = String.fromCodePoint(s.codePointAt(0) as number);
      return first.toUpperCase() + s.slice(first.length).toLowerCase();
    });
  }

  /** Swap the case of every character. */
  swapcase(): Series<string | null> {
    return this.map((s) => {
      let out = "";
      for (const ch of s) {
        out +=
          ch === ch.toUpperCase() && ch !== ch.toLowerCase() ? ch.toLowerCase() : ch.toUpperCase();
      }
      return out;
    });
  }

  // ─── Trimming ──────────────────────────────────────────────────

  /**
   * Remove characters from both ends.
   * @param chars - Characters to remove (each character of the string is one candidate).
   *   Defaults to whitespace.
   */
  strip(chars?: string): Series<string | null> {
    if (chars === undefined) return this.map((s) => s.trim());
    validateString(chars, "chars");
    return this.map((s) => stripChars(s, chars, true, true));
  }

  /**
   * Remove characters from the start.
   * @param chars - Characters to remove. Defaults to whitespace.
   */
  lstrip(chars?: string): Series<string | null> {
    if (chars === undefined) return this.map((s) => s.trimStart());
    validateString(chars, "chars");
    return this.map((s) => stripChars(s, chars, true, false));
  }

  /**
   * Remove characters from the end.
   * @param chars - Characters to remove. Defaults to whitespace.
   */
  rstrip(chars?: string): Series<string | null> {
    if (chars === undefined) return this.map((s) => s.trimEnd());
    validateString(chars, "chars");
    return this.map((s) => stripChars(s, chars, false, true));
  }

  // ─── Search / Match ────────────────────────────────────────────

  /**
   * Test whether each string contains a pattern anywhere.
   *
   * @param pat - Pattern: a string (regular expression source, or literal text when `regex` is false) or a RegExp
   * @param regex - Treat a string pattern as a regular expression (default: true)
   * @param caseSensitive - Match case (default: true)
   * @throws {InvalidParameterError} If a string pattern is not a valid regular expression
   */
  contains(
    pat: string | RegExp,
    regex: boolean = true,
    caseSensitive: boolean = true
  ): Series<boolean | null> {
    const re = statelessRegExp(toRegExp(pat, regex, "contains", !caseSensitive));
    return this.map((s) => re.test(s));
  }

  /** Test whether each string starts with `pat` (a literal string). */
  startswith(pat: string): Series<boolean | null> {
    validateString(pat, "pat");
    return this.map((s) => s.startsWith(pat));
  }

  /** Test whether each string ends with `pat` (a literal string). */
  endswith(pat: string): Series<boolean | null> {
    validateString(pat, "pat");
    return this.map((s) => s.endsWith(pat));
  }

  /**
   * Test whether the pattern matches at the start of each string, like
   * Python's `re.match`. Use {@link StringAccessor.contains} to search anywhere
   * and {@link StringAccessor.fullmatch} to require the whole string to match.
   */
  match(pat: string | RegExp): Series<boolean | null> {
    const re = anchoredRegExp(toRegExp(pat, true, "match"), false);
    return this.map((s) => {
      re.lastIndex = 0;
      return re.test(s);
    });
  }

  /** Test whether the pattern matches each whole string. */
  fullmatch(pat: string | RegExp): Series<boolean | null> {
    const re = anchoredRegExp(toRegExp(pat, true, "fullmatch"), true);
    return this.map((s) => {
      re.lastIndex = 0;
      return re.test(s);
    });
  }

  // ─── Replace / Split ──────────────────────────────────────────

  /**
   * Replace every occurrence of a pattern.
   *
   * With `regex` true (the default, unlike pandas) a string pattern is a regular
   * expression and `repl` follows JavaScript replacement syntax (`$1`, `$&`,
   * `$<name>`). With `regex` false, `pat` is literal text and `repl` is inserted
   * as is. An empty literal pattern inserts `repl` between every character and at
   * both ends.
   *
   * @throws {InvalidParameterError} If the pattern is invalid, or is not a string when `regex` is false
   */
  replace(pat: string | RegExp, repl: string, regex: boolean = true): Series<string | null> {
    validateString(repl, "repl");
    if (regex) {
      const re = globalRegExp(toRegExp(pat, true, "replace"));
      return this.map((s) => s.replace(re, repl));
    }
    validateString(pat, "pat");
    const literal = pat;
    if (literal.length === 0) {
      // Insert between code points, not between UTF-16 units.
      return this.map((s) => `${repl}${Array.from(s).join(repl)}${s.length > 0 ? repl : ""}`);
    }
    return this.map((s) => s.replaceAll(literal, () => repl));
  }

  /**
   * Split each string into a list of parts.
   *
   * - With no pattern, splits on runs of whitespace and ignores leading and
   *   trailing whitespace, like Python's `str.split()`.
   * - A string pattern is a literal separator. An empty string splits the text
   *   into single characters.
   * - A RegExp splits at each non-empty match. Text captured by groups is kept in
   *   the output between the parts, as in Python's `re.split`; zero-length
   *   matches are ignored.
   *
   * @param pat - Separator (default: whitespace)
   * @param n - Maximum number of splits. The remainder stays in the last part
   *   unchanged. Zero or negative (and undefined) means no limit.
   * @throws {InvalidParameterError} If n is not an integer or pat is not a string or RegExp
   */
  split(pat?: string | RegExp, n?: number): Series<string[] | null> {
    if (pat !== undefined && typeof pat !== "string" && !(pat instanceof RegExp)) {
      throw new InvalidParameterError("pat must be a string or a RegExp", "pat", pat);
    }
    if (n !== undefined && !Number.isInteger(n)) {
      throw new InvalidParameterError("n must be an integer", "n", n);
    }
    const max = n === undefined || n <= 0 ? Number.POSITIVE_INFINITY : n;
    const re = pat instanceof RegExp ? globalRegExp(pat) : undefined;
    return this.map((s) => {
      if (pat === undefined) return splitWhitespace(s, max);
      if (re) return splitRegExp(s, re, max);
      return splitLiteral(s, pat as string, max);
    });
  }

  // ─── Length / Slice ────────────────────────────────────────────

  /** Number of characters (Unicode code points) in each string. */
  len(): Series<number | null> {
    return this.map(codePointLength);
  }

  /**
   * Slice each string with Python semantics (negative positions count from the end).
   *
   * @param start - First position (default: 0, or the last character for a negative step)
   * @param stop - Position to stop before (default: the end)
   * @param step - Step between characters (default: 1). May be negative but not zero.
   * @throws {InvalidParameterError} If a position is not an integer or step is zero
   */
  slice(start?: number, stop?: number, step?: number): Series<string | null> {
    for (const [value, label] of [
      [start, "start"],
      [stop, "stop"],
      [step, "step"],
    ] as const) {
      if (value !== undefined && !Number.isInteger(value)) {
        throw new InvalidParameterError(`${label} must be an integer`, label, value);
      }
    }
    if (step === 0) {
      throw new InvalidParameterError("slice step cannot be zero", "step", step);
    }
    return this.map((s) => {
      if ((step === undefined || step === 1) && !hasSurrogates(s)) return s.slice(start, stop);
      return pySlice(Array.from(s), start, stop, step ?? 1).join("");
    });
  }

  /**
   * Get the character at a position (negative positions count from the end).
   * Returns null when the position is out of range.
   */
  get(position: number): Series<string | null> {
    if (!Number.isInteger(position)) {
      throw new InvalidParameterError("position must be an integer", "position", position);
    }
    return this.map((s) => {
      const chars = Array.from(s);
      const i = position < 0 ? chars.length + position : position;
      return chars[i] ?? null;
    });
  }

  // ─── Extract / FindAll ─────────────────────────────────────────

  /**
   * Extract one capture group of the first match in each string.
   *
   * @param pat - Regular expression
   * @param group - Capture group number; 0 is the whole match (default: 0)
   * @returns The matched text, or null where the pattern does not match
   * @throws {InvalidParameterError} If group is not a valid group number for the pattern
   */
  extract(pat: string | RegExp, group: number = 0): Series<string | null> {
    const re = statelessRegExp(toRegExp(pat, true, "extract"));
    if (!Number.isInteger(group) || group < 0 || group > countGroups(re)) {
      throw new InvalidParameterError(
        `group must be an integer between 0 and ${countGroups(re)} for this pattern`,
        "group",
        group
      );
    }
    return this.map((s) => {
      const m = re.exec(s);
      if (m === null) return null;
      const val = m[group];
      return val === undefined ? null : val;
    });
  }

  /**
   * Find all non-overlapping matches of a pattern in each string. Returns the
   * full matches (capture groups are not returned separately).
   */
  findall(pat: string | RegExp): Series<string[] | null> {
    const re = globalRegExp(toRegExp(pat, true, "findall"));
    return this.map((s) => {
      const matches = s.match(re);
      return matches === null ? [] : [...matches];
    });
  }

  /**
   * Lowest position (in characters) at which `sub` occurs, or -1 when absent.
   * `sub` is literal text.
   */
  find(sub: string): Series<number | null> {
    validateString(sub, "sub");
    return this.map((s) => {
      const i = s.indexOf(sub);
      return i < 0 ? -1 : codePointLength(s.slice(0, i));
    });
  }

  /**
   * Highest position (in characters) at which `sub` occurs, or -1 when absent.
   * `sub` is literal text.
   */
  rfind(sub: string): Series<number | null> {
    validateString(sub, "sub");
    return this.map((s) => {
      const i = s.lastIndexOf(sub);
      return i < 0 ? -1 : codePointLength(s.slice(0, i));
    });
  }

  /** Remove `prefix` from the start of each string when present. */
  removeprefix(prefix: string): Series<string | null> {
    validateString(prefix, "prefix");
    return this.map((s) =>
      prefix.length > 0 && s.startsWith(prefix) ? s.slice(prefix.length) : s
    );
  }

  /** Remove `suffix` from the end of each string when present. */
  removesuffix(suffix: string): Series<string | null> {
    validateString(suffix, "suffix");
    return this.map((s) =>
      suffix.length > 0 && s.endsWith(suffix) ? s.slice(0, s.length - suffix.length) : s
    );
  }

  // ─── Padding / Alignment ───────────────────────────────────────

  /**
   * Pad each string to a minimum width. Longer strings are returned unchanged.
   *
   * @param width - Minimum width in characters
   * @param side - Where to add the fill: "left" (default, right-aligns the text),
   *   "right" (left-aligns the text) or "both" (centers it; with an odd amount of
   *   fill the extra character goes on the left when the width is odd and on the right
   *   when it is even, as in Python's `str.center`)
   * @param fillchar - Single fill character (default: space)
   * @throws {InvalidParameterError} If width is not a non-negative integer, side is unknown, or fillchar is not one character
   */
  pad(
    width: number,
    side: "left" | "right" | "both" = "left",
    fillchar: string = " "
  ): Series<string | null> {
    validateNonNegativeInt(width, "width");
    if (side !== "left" && side !== "right" && side !== "both") {
      throw new InvalidParameterError(
        `side must be 'left', 'right' or 'both'; received ${String(side)}`,
        "side",
        side
      );
    }
    if (typeof fillchar !== "string" || codePointLength(fillchar) !== 1) {
      throw new InvalidParameterError("fillchar must be a single character", "fillchar", fillchar);
    }
    return this.map((s) => {
      const diff = width - codePointLength(s);
      if (diff <= 0) return s;
      if (side === "left") return fillchar.repeat(diff) + s;
      if (side === "right") return s + fillchar.repeat(diff);
      const leftPad = Math.floor(diff / 2) + (diff & width & 1);
      return fillchar.repeat(leftPad) + s + fillchar.repeat(diff - leftPad);
    });
  }

  /** Center each string in a field of the given width. See {@link StringAccessor.pad}. */
  center(width: number, fillchar: string = " "): Series<string | null> {
    return this.pad(width, "both", fillchar);
  }

  /** Left-align each string in a field of the given width, filling on the right. */
  ljust(width: number, fillchar: string = " "): Series<string | null> {
    return this.pad(width, "right", fillchar);
  }

  /** Right-align each string in a field of the given width, filling on the left. */
  rjust(width: number, fillchar: string = " "): Series<string | null> {
    return this.pad(width, "left", fillchar);
  }

  /**
   * Pad each string with zeros on the left to the given width. A leading `+` or
   * `-` stays in front of the zeros.
   */
  zfill(width: number): Series<string | null> {
    validateNonNegativeInt(width, "width");
    return this.map((s) => {
      const len = codePointLength(s);
      if (len >= width) return s;
      const sign = s.charAt(0);
      if (sign === "-" || sign === "+") {
        return sign + "0".repeat(width - len) + s.slice(1);
      }
      return "0".repeat(width - len) + s;
    });
  }

  // ─── Concatenation ─────────────────────────────────────────────

  /**
   * Join all non-null strings into one string.
   * @param sep - Separator placed between elements (default: empty)
   */
  cat(sep: string = ""): string {
    validateString(sep, "sep");
    const parts: string[] = [];
    for (let i = 0; i < this._data.length; i++) {
      const s = toStringOrNull(this._data[i], i);
      if (s !== null) parts.push(s);
    }
    return parts.join(sep);
  }

  // ─── getDummies ────────────────────────────────────────────────

  /**
   * Split each string on `sep` and build 0/1 indicator columns, one per distinct
   * token, ordered alphabetically. Null rows are all zeros.
   *
   * @param sep - Token separator (default: "|"). An empty string makes every
   *   character a token.
   * @throws {InvalidParameterError} If sep is not a string
   */
  getDummies(sep: string = "|"): DataFrame {
    validateString(sep, "sep");
    // Collect all unique tokens
    const allTokens = new Set<string>();
    const tokensByRow: Array<Set<string> | null> = new Array(this._data.length);

    for (let i = 0; i < this._data.length; i++) {
      const s = toStringOrNull(this._data[i], i);
      if (s === null) {
        tokensByRow[i] = null;
      } else {
        const tokens = new Set(sep.length === 0 ? Array.from(s) : s.split(sep));
        tokensByRow[i] = tokens;
        for (const t of tokens) allTokens.add(t);
      }
    }

    // Sort tokens for deterministic column order
    const sortedTokens = [...allTokens].sort();

    // Build column data. A null-prototype object keeps tokens such as
    // "__proto__" as ordinary column names.
    const columns: Record<string, number[]> = Object.create(null) as Record<string, number[]>;
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

  /**
   * Same as {@link StringAccessor.getDummies}.
   *
   * @deprecated Prefer {@link StringAccessor.getDummies}.
   */
  get_dummies(sep: string = "|"): DataFrame {
    return this.getDummies(sep);
  }

  // ─── Repeat ────────────────────────────────────────────────────

  /** Repeat each string `times` times. */
  repeat(times: number): Series<string | null> {
    validateNonNegativeInt(times, "times");
    return this.map((s) => s.repeat(times));
  }

  // ─── Count occurrences ─────────────────────────────────────────

  /** Count the non-overlapping matches of a pattern in each string. */
  count(pat: string | RegExp): Series<number | null> {
    const re = globalRegExp(toRegExp(pat, true, "count"));
    return this.map((s) => {
      const matches = s.match(re);
      return matches === null ? 0 : matches.length;
    });
  }

  // ─── Boolean checks ───────────────────────────────────────────
  // Each check is false for the empty string, like Python.

  /** True where every character is a letter (any script) and the string is not empty. */
  isalpha(): Series<boolean | null> {
    return this.map((s) => /^\p{L}+$/u.test(s));
  }

  /** True where every character is a digit (decimal digits of any script, plus superscript and subscript digits). */
  isdigit(): Series<boolean | null> {
    return this.map((s) => /^[\p{Nd}²³¹⁰⁴-⁹₀-₉]+$/u.test(s));
  }

  /** True where every character is a decimal digit (Unicode category Nd). */
  isdecimal(): Series<boolean | null> {
    return this.map((s) => /^\p{Nd}+$/u.test(s));
  }

  /** True where every character is numeric (Unicode category N, such as digits, fractions and Roman numerals). */
  isnumeric(): Series<boolean | null> {
    return this.map((s) => /^\p{N}+$/u.test(s));
  }

  /** True where every character is a letter or a number. */
  isalnum(): Series<boolean | null> {
    return this.map((s) => /^[\p{L}\p{N}]+$/u.test(s));
  }

  /** True where every character is whitespace. */
  isspace(): Series<boolean | null> {
    return this.map((s) => {
      if (s.length === 0) return false;
      for (let i = 0; i < s.length; i++) {
        if (!isPythonWhitespace(s.charCodeAt(i))) return false;
      }
      return true;
    });
  }

  /** True where there is at least one cased letter and all cased letters are upper case. */
  isupper(): Series<boolean | null> {
    return this.map((s) => s.length > 0 && s === s.toUpperCase() && s !== s.toLowerCase());
  }

  /** True where there is at least one cased letter and all cased letters are lower case. */
  islower(): Series<boolean | null> {
    return this.map((s) => s.length > 0 && s === s.toLowerCase() && s !== s.toUpperCase());
  }
}

// ─── Internal helpers ───────────────────────────────────────────

function validateString(value: unknown, name: string): asserts value is string {
  if (typeof value !== "string") {
    throw new InvalidParameterError(`${name} must be a string`, name, value);
  }
}

function validateNonNegativeInt(value: number, name: string): void {
  if (!Number.isFinite(value) || !Number.isInteger(value) || value < 0) {
    throw new InvalidParameterError(`${name} must be a non-negative integer`, name, value);
  }
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function hasSurrogates(s: string): boolean {
  return /[\ud800-\udfff]/.test(s);
}

function codePointLength(s: string): number {
  if (!hasSurrogates(s)) return s.length;
  let n = 0;
  for (const _ of s) n++;
  return n;
}

/** Python-style `str.title()`: upper-case a cased letter that follows an uncased character. */
function titleCase(s: string): string {
  let out = "";
  let prevCased = false;
  for (const ch of s) {
    const lower = ch.toLowerCase();
    const upper = ch.toUpperCase();
    if (lower !== upper) {
      out += prevCased ? lower : upper;
      prevCased = true;
    } else {
      out += ch;
      prevCased = false;
    }
  }
  return out;
}

/** Remove any of the characters in `chars` from the chosen ends of `s`. */
function stripChars(s: string, chars: string, left: boolean, right: boolean): string {
  if (chars.length === 0) return s;
  if (!hasSurrogates(chars)) {
    // No surrogate in the set, so stripping code units can never split a pair.
    let a = 0;
    let b = s.length;
    if (left) while (a < b && chars.includes(s.charAt(a))) a++;
    if (right) while (b > a && chars.includes(s.charAt(b - 1))) b--;
    return s.slice(a, b);
  }
  const set = new Set(Array.from(chars));
  const cps = Array.from(s);
  let a = 0;
  let b = cps.length;
  if (left) while (a < b && set.has(cps[a] as string)) a++;
  if (right) while (b > a && set.has(cps[b - 1] as string)) b--;
  return cps.slice(a, b).join("");
}

/** Python slice semantics over an array of characters. */
function pySlice(
  chars: readonly string[],
  start: number | undefined,
  stop: number | undefined,
  step: number
): string[] {
  const len = chars.length;
  const out: string[] = [];
  if (step > 0) {
    let lo = start === undefined ? 0 : start < 0 ? Math.max(start + len, 0) : Math.min(start, len);
    const hi = stop === undefined ? len : stop < 0 ? Math.max(stop + len, 0) : Math.min(stop, len);
    for (; lo < hi; lo += step) out.push(chars[lo] as string);
  } else {
    let hi =
      start === undefined
        ? len - 1
        : start < 0
          ? Math.max(start + len, -1)
          : Math.min(start, len - 1);
    const lo =
      stop === undefined ? -1 : stop < 0 ? Math.max(stop + len, -1) : Math.min(stop, len - 1);
    for (; hi > lo; hi += step) out.push(chars[hi] as string);
  }
  return out;
}

function splitLiteral(s: string, sep: string, max: number): string[] {
  if (sep.length === 0) {
    // Split into characters; the remainder after `max` splits stays together.
    const chars = Array.from(s);
    if (chars.length <= max + 1) return chars;
    return [...chars.slice(0, max), chars.slice(max).join("")];
  }
  const out: string[] = [];
  let pos = 0;
  while (out.length < max) {
    const i = s.indexOf(sep, pos);
    if (i < 0) break;
    out.push(s.slice(pos, i));
    pos = i + sep.length;
  }
  out.push(s.slice(pos));
  return out;
}

/** Code units Python's `str.isspace()` accepts (differs from JS `\s` on U+FEFF, U+0085 and U+001C-U+001F). */
function isPythonWhitespace(code: number): boolean {
  return (
    (code >= 0x09 && code <= 0x0d) ||
    (code >= 0x1c && code <= 0x20) ||
    code === 0x85 ||
    code === 0xa0 ||
    code === 0x1680 ||
    (code >= 0x2000 && code <= 0x200a) ||
    code === 0x2028 ||
    code === 0x2029 ||
    code === 0x202f ||
    code === 0x205f ||
    code === 0x3000
  );
}

function splitWhitespace(s: string, max: number): string[] {
  const out: string[] = [];
  const len = s.length;
  let i = 0;
  for (;;) {
    while (i < len && isPythonWhitespace(s.charCodeAt(i))) i++;
    if (i >= len) break;
    if (out.length >= max) {
      out.push(s.slice(i));
      break;
    }
    let j = i;
    while (j < len && !isPythonWhitespace(s.charCodeAt(j))) j++;
    out.push(s.slice(i, j));
    i = j;
  }
  return out;
}

function splitRegExp(s: string, re: RegExp, max: number): string[] {
  const out: string[] = [];
  let pos = 0;
  let splits = 0;
  re.lastIndex = 0;
  while (splits < max) {
    const m = re.exec(s);
    if (m === null) break;
    if (m[0].length === 0) {
      // Zero-length match: step past it to avoid looping forever.
      re.lastIndex = m.index + 1;
      if (re.lastIndex > s.length) break;
      continue;
    }
    out.push(s.slice(pos, m.index));
    for (let g = 1; g < m.length; g++) {
      const captured = m[g];
      if (captured !== undefined) out.push(captured);
    }
    pos = m.index + m[0].length;
    splits++;
  }
  out.push(s.slice(pos));
  return out;
}

/** Number of capture groups in a regular expression. */
function countGroups(re: RegExp): number {
  // An empty alternative always matches, so the match array lists every group.
  const m = new RegExp(`${re.source}|`, re.flags.replace(/[gy]/g, "")).exec("");
  return m === null ? 0 : m.length - 1;
}

function toRegExp(
  pat: string | RegExp,
  regex: boolean,
  functionName: string,
  ignoreCase = false
): RegExp {
  if (pat instanceof RegExp) {
    return ignoreCase && !pat.ignoreCase ? new RegExp(pat.source, `${pat.flags}i`) : pat;
  }
  validateString(pat, "pat");
  const flags = ignoreCase ? "i" : "";
  if (!regex) return new RegExp(escapeRegExp(pat), flags);
  try {
    return new RegExp(pat, flags);
  } catch {
    throw new InvalidParameterError(`${functionName}: invalid regex pattern`, "pat", pat);
  }
}

/**
 * A copy without the `g` and `y` flags. `test` and `exec` on a global or sticky
 * RegExp depend on `lastIndex`, which would leak state from one element to the next.
 */
function statelessRegExp(re: RegExp): RegExp {
  return re.global || re.sticky ? new RegExp(re.source, re.flags.replace(/[gy]/g, "")) : re;
}

/** A global copy (no sticky flag) for replace, match-all and split. */
function globalRegExp(re: RegExp): RegExp {
  return new RegExp(re.source, `${re.flags.replace(/[gy]/g, "")}g`);
}

/**
 * A sticky copy that can only match at index 0. With `full`, the match must also
 * reach the end of the string, with normal backtracking.
 */
function anchoredRegExp(re: RegExp, full: boolean): RegExp {
  const flags = `${re.flags.replace(/[gy]/g, "")}y`;
  return new RegExp(full ? `(?:${re.source})(?![\\s\\S])` : re.source, flags);
}
