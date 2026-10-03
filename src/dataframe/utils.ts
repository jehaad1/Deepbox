/**
 * Internal utilities for DataFrame and Series.
 * @see {@link https://deepbox.dev/docs/dataframe-overview | Deepbox documentation}
 */

import { InvalidParameterError } from "../core/errors/index";

/**
 * Checks if a value is a Record (object but not null or array).
 *
 * Other object types such as `Date`, `Map` and class instances also pass;
 * check for those first when they need separate handling.
 */
export const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/**
 * True for object literals and `Object.create(null)` objects, but not arrays, dates or
 * class instances.
 */
export const isPlainObject = (value: unknown): value is Record<string, unknown> => {
  if (typeof value !== "object" || value === null) return false;
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
};

const keyOf = (value: unknown, outer: unknown[] | undefined): string => {
  if (value === null) return "null";
  if (value === undefined) return "undefined";

  switch (typeof value) {
    case "number":
      if (Number.isNaN(value)) return "NaN";
      if (value === Infinity) return "Infinity";
      if (value === -Infinity) return "-Infinity";
      return `n:${value}`;
    case "string":
      // The length prefix keeps keys of nested values unambiguous: ["a,s1:b"] and
      // ["a", "b"] must not produce the same text.
      return `s${value.length}:${value}`;
    case "boolean":
      return `b:${value}`;
    case "bigint":
      return `bi:${value.toString()}`;
    case "symbol":
      return `y:${String(value)}`;
    case "function":
      return `f:${String(value)}`;
    default:
      break;
  }

  // Dates are keyed by their timestamp; without this branch Object.keys(date) is
  // empty and every Date would hash to "{}".
  if (value instanceof Date) {
    return `d:${value.getTime()}`;
  }
  if (value instanceof RegExp) {
    const text = String(value);
    return `r${text.length}:${text}`;
  }

  // The ancestor stack is only needed for containers, so primitives never allocate one.
  const ancestors = outer ?? [];
  if (ancestors.includes(value)) return "circular";
  ancestors.push(value);
  try {
    if (Array.isArray(value)) {
      return `[${value.map((item) => keyOf(item, ancestors)).join(",")}]`;
    }

    if (ArrayBuffer.isView(value) && !(value instanceof DataView)) {
      const items = Array.from(value as unknown as ArrayLike<unknown>);
      return `t[${items.map((item) => keyOf(item, ancestors)).join(",")}]`;
    }

    if (value instanceof Map) {
      const parts = [...value].map(([k, v]) => `${keyOf(k, ancestors)}=>${keyOf(v, ancestors)}`);
      return `m{${parts.sort().join(",")}}`;
    }

    if (value instanceof Set) {
      const parts = [...value].map((item) => keyOf(item, ancestors));
      return `st{${parts.sort().join(",")}}`;
    }

    // For objects, sort the keys so {a: 1, b: 2} and {b: 2, a: 1} share a key.
    if (isRecord(value)) {
      const keys = Object.keys(value).sort();
      const parts = keys.map((k) => `${keyOf(k, ancestors)}:${keyOf(value[k], ancestors)}`);
      return `{${parts.join(",")}}`;
    }
  } finally {
    ancestors.pop();
  }

  return String(value);
};

/**
 * Generates a unique string key for a value or array of values.
 *
 * Two values get the same key exactly when they are equal in the sense used
 * by join, groupBy and dropDuplicates. The key distinguishes:
 * - null vs undefined
 * - NaN vs other numbers (NaN matches NaN; 0 matches -0)
 * - Infinity vs -Infinity
 * - 1 vs "1" vs 1n vs true
 * - Dates (by timestamp), nested arrays, typed arrays, Maps, Sets, plain objects
 *   (key order does not matter)
 *
 * Strings are length-prefixed, so composite keys such as `["a,b"]` and
 * `["a", "b"]` cannot collide. Circular references are keyed as `"circular"`
 * instead of overflowing the stack.
 */
export const createKey = (value: unknown): string => keyOf(value, undefined);

/**
 * Checks if a value is a finite number (not NaN, not +/-Infinity).
 */
export const isValidNumber = (value: unknown): value is number => Number.isFinite(value);

/**
 * Neumaier compensated sum shared by Series, DataFrame and groupby so the same
 * values always add up to the same bits. An infinite or overflowing partial sum
 * skips the compensation term, so an Infinity in the input is returned as is
 * instead of turning into NaN.
 */
export const compensatedSum = (values: ArrayLike<number>): number => {
  let sum = 0;
  let comp = 0;
  for (let i = 0; i < values.length; i++) {
    const x = values[i] as number;
    const t = sum + x;
    if (Number.isFinite(t)) {
      comp += Math.abs(sum) >= Math.abs(x) ? sum - t + x : x - t + sum;
    }
    sum = t;
  }
  return Number.isFinite(sum) ? sum + comp : sum;
};

/**
 * Sum of squared deviations from the mean, using the corrected two-pass
 * algorithm so the result stays accurate when the mean is large relative to
 * the spread. The input must not be empty.
 */
export const sumSquaredDeviations = (values: ArrayLike<number>): number => {
  const n = values.length;
  const mean = compensatedSum(values) / n;
  let sq = 0;
  let dev = 0;
  for (let i = 0; i < n; i++) {
    const d = (values[i] as number) - mean;
    sq += d * d;
    dev += d;
  }
  return sq - (dev * dev) / n;
};

/**
 * Ascending order of two strings by UTF-16 code unit, independent of the locale.
 * This is the order of JavaScript's `<` operator and matches pandas for text
 * outside the supplementary planes.
 */
export const compareStrings = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);

/** Text of one cell in toString(): "null" for missing, ISO 8601 for dates. */
export const cellText = (value: unknown): string => {
  if (value === null || value === undefined) return "null";
  if (value instanceof Date) {
    return Number.isNaN(value.getTime()) ? "Invalid Date" : value.toISOString();
  }
  return String(value);
};

/**
 * Validates the `limit` option of a fill: a positive integer, or undefined for no limit.
 * @throws {InvalidParameterError} If the limit is not a positive integer
 */
export const checkFillLimit = (limit: number | undefined): number | undefined => {
  if (limit === undefined) return undefined;
  if (typeof limit !== "number" || !Number.isInteger(limit) || limit < 1) {
    throw new InvalidParameterError("limit must be a positive integer", "limit", limit);
  }
  return limit;
};

/**
 * Resolves a fill method name, accepting the pandas aliases `"pad"` and `"backfill"`.
 * @throws {InvalidParameterError} If the name is not a known method
 */
export const checkFillMethod = (method: unknown): "ffill" | "bfill" => {
  if (method === "ffill" || method === "pad") return "ffill";
  if (method === "bfill" || method === "backfill") return "bfill";
  throw new InvalidParameterError(
    'method must be "ffill", "bfill", "pad" or "backfill"',
    "method",
    method
  );
};

/**
 * Fills missing entries by propagating the nearest valid neighbour: the previous one for
 * `"ffill"`, the next one for `"bfill"`. At most `limit` consecutive missing entries are
 * filled in each gap, counted from the valid value that is propagated. Entries with no
 * valid neighbour on the propagating side keep their value. Returns a new array.
 */
export const propagateFill = (
  values: readonly unknown[],
  direction: "ffill" | "bfill",
  limit: number | undefined,
  isMissing: (value: unknown) => boolean
): unknown[] => {
  const n = values.length;
  const out = values.slice();
  const forward = direction === "ffill";
  let carried: unknown;
  let have = false;
  let run = 0;
  for (let step = 0; step < n; step++) {
    const i = forward ? step : n - 1 - step;
    const value = values[i];
    if (!isMissing(value)) {
      carried = value;
      have = true;
      run = 0;
    } else if (have && (limit === undefined || run < limit)) {
      out[i] = carried;
      run++;
    }
  }
  return out;
};
