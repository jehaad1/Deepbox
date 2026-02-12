/**
 * Internal utilities for DataFrame and Series.
 */

/**
 * Checks if a value is a Record (object but not null or array).
 */
export const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/**
 * Generates a unique string key for a value or array of values.
 * Handles distinctions between:
 * - null vs undefined
 * - NaN vs other numbers
 * - Infinity vs -Infinity
 * - 1 vs "1"
 * - Nested arrays
 */
export const createKey = (value: unknown): string => {
  if (value === null) return "null";
  if (value === undefined) return "undefined";

  const type = typeof value;

  if (type === "number") {
    if (Number.isNaN(value)) return "NaN";
    if (value === Infinity) return "Infinity";
    if (value === -Infinity) return "-Infinity";
    return `n:${value}`;
  }

  if (type === "string") {
    return `s:${value}`;
  }

  if (type === "boolean") {
    return `b:${value}`;
  }

  if (type === "bigint") {
    return `bi:${value.toString()}`;
  }

  if (Array.isArray(value)) {
    return `[${value.map(createKey).join(",")}]`;
  }

  // For objects, we'll do a stable sort of keys to ensure {a:1, b:2} === {b:2, a:1}
  if (isRecord(value)) {
    const keys = Object.keys(value).sort();
    const parts = keys.map((k) => `${createKey(k)}:${createKey(value[k])}`);
    return `{${parts.join(",")}}`;
  }

  return String(value);
};

/**
 * Checks if a value is a valid number (not NaN, not Infinity).
 */
export const isValidNumber = (value: unknown): value is number => {
  return typeof value === "number" && !Number.isNaN(value) && Number.isFinite(value);
};
