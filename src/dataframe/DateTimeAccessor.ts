/**
 * @see {@link https://deepbox.dev/docs/dataframe-overview | Deepbox documentation}
 */

import { DataValidationError, InvalidParameterError } from "../core/errors/index";
import { Series } from "./Series";
import type { SeriesOptions } from "./types";

/**
 * Represents a date/time value internally as a UTC millisecond timestamp.
 * This is a lightweight wrapper that provides Pandas-like `.dt` accessor
 * fields without requiring a heavy date library.
 */

/**
 * Parse a single value to a Date, returning null for null/undefined.
 */
function toDateOrNull(value: unknown, index: number): Date | null {
  if (value === null || value === undefined) return null;
  if (value instanceof Date) {
    if (Number.isNaN(value.getTime())) {
      throw new DataValidationError(
        `DateTimeAccessor: element at index ${index} is an invalid Date`
      );
    }
    return value;
  }
  if (typeof value === "string") {
    const str = value.trim();
    // A bare ISO date ('2024-01-01') is parsed by JS as UTC midnight, but the
    // component getters below read LOCAL time — so west of UTC the day/month
    // come out off by one. Parse date-only and naive-datetime strings as
    // LOCAL time so parsing and extraction stay consistent (matching pandas'
    // naive-datetime semantics). Strings with an explicit Z/offset keep their
    // timezone (handled by the fallback Date parse).
    const dateOnly = /^(\d{4})-(\d{2})-(\d{2})$/.exec(str);
    if (dateOnly) {
      return new Date(Number(dateOnly[1]), Number(dateOnly[2]) - 1, Number(dateOnly[3]));
    }
    const naive = /^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::(\d{2})(?:\.(\d+))?)?$/.exec(str);
    if (naive) {
      const ms = naive[7] ? Math.round(Number(`0.${naive[7]}`) * 1000) : 0;
      return new Date(
        Number(naive[1]),
        Number(naive[2]) - 1,
        Number(naive[3]),
        Number(naive[4]),
        Number(naive[5]),
        naive[6] ? Number(naive[6]) : 0,
        ms
      );
    }
    const d = new Date(str);
    if (Number.isNaN(d.getTime())) {
      throw new DataValidationError(
        `DateTimeAccessor: cannot parse '${value}' at index ${index} as a date`
      );
    }
    return d;
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new DataValidationError(
        `DateTimeAccessor: element at index ${index} is not a finite number`
      );
    }
    return new Date(value);
  }
  throw new DataValidationError(
    `DateTimeAccessor: element at index ${index} has unsupported type ${typeof value}`
  );
}

/**
 * Build a new Series by mapping each element through a function.
 * Null inputs produce null outputs.
 */
function mapDateSeries<T>(
  data: readonly unknown[],
  indexLabels: readonly (string | number)[],
  name: string | undefined,
  fn: (date: Date) => T
): Series<T | null> {
  const result: Array<T | null> = new Array(data.length);
  for (let i = 0; i < data.length; i++) {
    const d = toDateOrNull(data[i], i);
    result[i] = d === null ? null : fn(d);
  }
  const opts: SeriesOptions = { index: [...indexLabels] };
  if (name !== undefined) opts.name = name;
  return new Series(result, opts);
}

/**
 * DateTime accessor for Series, providing vectorized date/time operations.
 *
 * Accessed via `series.dt`. Operates element-wise on Series containing
 * Date objects, ISO date strings, or epoch milliseconds.
 * Null/undefined elements propagate as null in the output.
 *
 * @example
 * ```ts
 * const s = new Series([new Date('2024-01-15'), new Date('2024-06-20')]);
 * s.dt.year();       // Series([2024, 2024])
 * s.dt.month();      // Series([1, 6])
 * s.dt.dayofweek();  // Series([1, 4])  (Mon=0 ... Sun=6)
 * ```
 */
export class DateTimeAccessor {
  private readonly _data: readonly unknown[];
  private readonly _index: readonly (string | number)[];
  private readonly _name: string | undefined;

  constructor(series: Series<unknown>) {
    this._data = series.data;
    this._index = series.index;
    this._name = series.name;
  }

  // ─── Component extraction ──────────────────────────────────────

  year(): Series<number | null> {
    return mapDateSeries(this._data, this._index, this._name, (d) => d.getFullYear());
  }

  month(): Series<number | null> {
    return mapDateSeries(this._data, this._index, this._name, (d) => d.getMonth() + 1);
  }

  day(): Series<number | null> {
    return mapDateSeries(this._data, this._index, this._name, (d) => d.getDate());
  }

  hour(): Series<number | null> {
    return mapDateSeries(this._data, this._index, this._name, (d) => d.getHours());
  }

  minute(): Series<number | null> {
    return mapDateSeries(this._data, this._index, this._name, (d) => d.getMinutes());
  }

  second(): Series<number | null> {
    return mapDateSeries(this._data, this._index, this._name, (d) => d.getSeconds());
  }

  millisecond(): Series<number | null> {
    return mapDateSeries(this._data, this._index, this._name, (d) => d.getMilliseconds());
  }

  /**
   * Day of week: Monday=0, Tuesday=1, ..., Sunday=6 (ISO convention).
   */
  dayofweek(): Series<number | null> {
    return mapDateSeries(this._data, this._index, this._name, (d) => {
      const jsDay = d.getDay(); // 0=Sun, 1=Mon, ... 6=Sat
      return jsDay === 0 ? 6 : jsDay - 1; // Convert to Mon=0 ... Sun=6
    });
  }

  /**
   * Day of year: 1-366.
   */
  dayofyear(): Series<number | null> {
    return mapDateSeries(this._data, this._index, this._name, (d) => {
      const start = new Date(d.getFullYear(), 0, 0);
      const diff = d.getTime() - start.getTime();
      return Math.floor(diff / 86400000);
    });
  }

  /**
   * ISO week number (1-53).
   */
  weekofyear(): Series<number | null> {
    return mapDateSeries(this._data, this._index, this._name, (d) => {
      const target = new Date(d.getFullYear(), d.getMonth(), d.getDate());
      // ISO: week starts on Monday; Jan 4 is always in week 1
      target.setDate(target.getDate() + 3 - ((target.getDay() + 6) % 7));
      const jan4 = new Date(target.getFullYear(), 0, 4);
      jan4.setDate(jan4.getDate() + 3 - ((jan4.getDay() + 6) % 7));
      const diff = target.getTime() - jan4.getTime();
      return 1 + Math.round(diff / 604800000);
    });
  }

  /**
   * Quarter: 1-4.
   */
  quarter(): Series<number | null> {
    return mapDateSeries(
      this._data,
      this._index,
      this._name,
      (d) => Math.floor(d.getMonth() / 3) + 1
    );
  }

  /**
   * Whether the year is a leap year.
   */
  is_leap_year(): Series<boolean | null> {
    return mapDateSeries(this._data, this._index, this._name, (d) => {
      const y = d.getFullYear();
      return (y % 4 === 0 && y % 100 !== 0) || y % 400 === 0;
    });
  }

  /**
   * Days in the month of each date.
   */
  days_in_month(): Series<number | null> {
    return mapDateSeries(this._data, this._index, this._name, (d) =>
      new Date(d.getFullYear(), d.getMonth() + 1, 0).getDate()
    );
  }

  // ─── Conversion ────────────────────────────────────────────────

  /**
   * Convert each date to epoch milliseconds.
   */
  timestamp(): Series<number | null> {
    return mapDateSeries(this._data, this._index, this._name, (d) => d.getTime());
  }

  /**
   * Convert each date to ISO string format.
   */
  isoformat(): Series<string | null> {
    return mapDateSeries(this._data, this._index, this._name, (d) => d.toISOString());
  }

  /**
   * Format each date using a strftime-like pattern.
   * Supports: %Y (year), %m (month), %d (day), %H (hour), %M (minute), %S (second)
   */
  strftime(fmt: string): Series<string | null> {
    if (typeof fmt !== "string") {
      throw new InvalidParameterError("fmt must be a string", "fmt", fmt);
    }
    return mapDateSeries(this._data, this._index, this._name, (d) => {
      let result = fmt;
      result = result.replace(/%Y/g, String(d.getFullYear()));
      result = result.replace(/%m/g, String(d.getMonth() + 1).padStart(2, "0"));
      result = result.replace(/%d/g, String(d.getDate()).padStart(2, "0"));
      result = result.replace(/%H/g, String(d.getHours()).padStart(2, "0"));
      result = result.replace(/%M/g, String(d.getMinutes()).padStart(2, "0"));
      result = result.replace(/%S/g, String(d.getSeconds()).padStart(2, "0"));
      return result;
    });
  }

  // ─── Date-only (truncation) ────────────────────────────────────

  /**
   * Truncate to date only (midnight).
   */
  date(): Series<Date | null> {
    return mapDateSeries(
      this._data,
      this._index,
      this._name,
      (d) => new Date(d.getFullYear(), d.getMonth(), d.getDate())
    );
  }

  /**
   * Extract the time portion as "HH:MM:SS".
   */
  time(): Series<string | null> {
    return mapDateSeries(this._data, this._index, this._name, (d) => {
      const hh = String(d.getHours()).padStart(2, "0");
      const mm = String(d.getMinutes()).padStart(2, "0");
      const ss = String(d.getSeconds()).padStart(2, "0");
      return `${hh}:${mm}:${ss}`;
    });
  }

  // ─── Floor / Ceil / Round ──────────────────────────────────────

  /**
   * Floor each date to the start of the given frequency.
   * @param freq - 'D' (day), 'H' (hour), 'min' (minute), 'S' (second)
   */
  floor(freq: "D" | "H" | "min" | "S"): Series<Date | null> {
    return mapDateSeries(this._data, this._index, this._name, (d) => floorDate(d, freq));
  }

  /**
   * Ceil each date to the next boundary of the given frequency.
   */
  ceil(freq: "D" | "H" | "min" | "S"): Series<Date | null> {
    return mapDateSeries(this._data, this._index, this._name, (d) => {
      const floored = floorDate(d, freq);
      if (floored.getTime() === d.getTime()) return new Date(d.getTime());
      return new Date(floored.getTime() + freqToMs(freq));
    });
  }

  /**
   * Round each date to the nearest boundary of the given frequency.
   */
  round(freq: "D" | "H" | "min" | "S"): Series<Date | null> {
    const ms = freqToMs(freq);
    return mapDateSeries(this._data, this._index, this._name, (d) => {
      const floored = floorDate(d, freq);
      const diff = d.getTime() - floored.getTime();
      if (diff < ms / 2) return floored;
      return new Date(floored.getTime() + ms);
    });
  }
}

// ─── Standalone helper functions ──────────────────────────────────

function freqToMs(freq: "D" | "H" | "min" | "S"): number {
  switch (freq) {
    case "D":
      return 86400000;
    case "H":
      return 3600000;
    case "min":
      return 60000;
    case "S":
      return 1000;
  }
}

function floorDate(d: Date, freq: "D" | "H" | "min" | "S"): Date {
  switch (freq) {
    case "D":
      return new Date(d.getFullYear(), d.getMonth(), d.getDate());
    case "H":
      return new Date(d.getFullYear(), d.getMonth(), d.getDate(), d.getHours());
    case "min":
      return new Date(d.getFullYear(), d.getMonth(), d.getDate(), d.getHours(), d.getMinutes());
    case "S":
      return new Date(
        d.getFullYear(),
        d.getMonth(),
        d.getDate(),
        d.getHours(),
        d.getMinutes(),
        d.getSeconds()
      );
  }
}

// ─── Standalone functions ──────────────────────────────────────────

type DateFreq = "D" | "H" | "min" | "S" | "MS" | "YS";

/**
 * Parse an array of values to Date Series.
 *
 * Accepts strings (ISO 8601), epoch milliseconds (numbers), or Date objects.
 *
 * @param data - Array of date-like values
 * @param options - Series options (name, index)
 * @returns Series of Date objects
 *
 * @example
 * ```ts
 * import { to_datetime } from 'deepbox/dataframe';
 * const dates = to_datetime(['2024-01-01', '2024-06-15', '2024-12-31']);
 * dates.dt.month();  // Series([1, 6, 12])
 * ```
 */
export function to_datetime(
  data: ReadonlyArray<string | number | Date | null | undefined>,
  options: SeriesOptions = {}
): Series<Date | null> {
  const result: Array<Date | null> = new Array(data.length);
  for (let i = 0; i < data.length; i++) {
    result[i] = toDateOrNull(data[i], i);
  }
  return new Series(result, options);
}

/**
 * Generate a range of dates at a fixed frequency.
 *
 * @param start - Start date (string, Date, or epoch ms)
 * @param periods - Number of periods to generate
 * @param freq - Frequency: 'D' (day), 'H' (hour), 'min' (minute), 'S' (second), 'MS' (month start), 'YS' (year start)
 * @returns Series of Dates
 *
 * @example
 * ```ts
 * import { date_range } from 'deepbox/dataframe';
 * const dates = date_range('2024-01-01', 5, 'D');
 * // Series([Date(Jan 1), Date(Jan 2), ..., Date(Jan 5)])
 * ```
 */
export function date_range(
  start: string | Date | number,
  periods: number,
  freq: DateFreq = "D"
): Series<Date> {
  if (!Number.isFinite(periods) || !Number.isInteger(periods) || periods < 0) {
    throw new InvalidParameterError("periods must be a non-negative integer", "periods", periods);
  }

  const startDate = toDateOrNull(start, 0);
  if (startDate === null) {
    throw new InvalidParameterError("start must be a valid date value", "start", start);
  }

  const dates: Date[] = new Array(periods);
  for (let i = 0; i < periods; i++) {
    dates[i] = addFreq(startDate, freq, i);
  }

  return new Series(dates);
}

function addFreq(base: Date, freq: DateFreq, n: number): Date {
  switch (freq) {
    case "D":
      return new Date(base.getTime() + n * 86400000);
    case "H":
      return new Date(base.getTime() + n * 3600000);
    case "min":
      return new Date(base.getTime() + n * 60000);
    case "S":
      return new Date(base.getTime() + n * 1000);
    case "MS": {
      const d = new Date(base.getTime());
      d.setMonth(d.getMonth() + n);
      return d;
    }
    case "YS": {
      const d = new Date(base.getTime());
      d.setFullYear(d.getFullYear() + n);
      return d;
    }
  }
}

/**
 * Compute the difference between two Date Series in the given unit.
 *
 * @param left - Left operand Series of Dates
 * @param right - Right operand Series of Dates
 * @param unit - Unit for result: 'ms', 'S', 'min', 'H', 'D'
 * @returns Series<number | null> of differences (left - right) in the given unit
 */
export function timedelta(
  left: Series<unknown>,
  right: Series<unknown>,
  unit: "ms" | "S" | "min" | "H" | "D" = "ms"
): Series<number | null> {
  const leftData = left.data;
  const rightData = right.data;
  if (leftData.length !== rightData.length) {
    throw new DataValidationError(
      `timedelta: Series lengths must match (${leftData.length} vs ${rightData.length})`
    );
  }

  const divisor = unitToMs(unit);
  const result: Array<number | null> = new Array(leftData.length);
  for (let i = 0; i < leftData.length; i++) {
    const l = toDateOrNull(leftData[i], i);
    const r = toDateOrNull(rightData[i], i);
    if (l === null || r === null) {
      result[i] = null;
    } else {
      result[i] = (l.getTime() - r.getTime()) / divisor;
    }
  }

  return new Series(result, { index: [...left.index] });
}

function unitToMs(unit: "ms" | "S" | "min" | "H" | "D"): number {
  switch (unit) {
    case "ms":
      return 1;
    case "S":
      return 1000;
    case "min":
      return 60000;
    case "H":
      return 3600000;
    case "D":
      return 86400000;
  }
}
