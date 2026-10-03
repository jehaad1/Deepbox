/**
 * @see {@link https://deepbox.dev/docs/dataframe-overview | Deepbox documentation}
 */

import { DataValidationError, InvalidParameterError } from "../core/errors/index";
import { Series } from "./Series";
import type { SeriesOptions } from "./types";

// Date handling in this module follows pandas' naive-datetime model: every
// component (year, month, day, hour, ...) is read in the local time zone, and
// floor/ceil/round and calendar stepping work on wall-clock values, so a
// daylight saving change never shifts a midnight to 01:00 or 23:00.

const MS_PER_SECOND = 1000;
const MS_PER_MINUTE = 60000;
const MS_PER_HOUR = 3600000;
const MS_PER_DAY = 86400000;

const DAY_NAMES = [
  "Sunday",
  "Monday",
  "Tuesday",
  "Wednesday",
  "Thursday",
  "Friday",
  "Saturday",
] as const;
const MONTH_NAMES = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
] as const;

/**
 * Build a local-time Date from components. Unlike `new Date(y, ...)`, years 0-99
 * are taken literally instead of being mapped to 1900-1999.
 */
function localDate(
  year: number,
  month: number,
  day: number,
  hour = 0,
  minute = 0,
  second = 0,
  ms = 0
): Date {
  const d = new Date(year, month, day, hour, minute, second, ms);
  if (year >= 0 && year < 100) {
    d.setFullYear(year, month, day);
  }
  return d;
}

/**
 * Milliseconds since the epoch of a Date's local wall-clock components,
 * as if they were UTC. Arithmetic on this value is free of DST jumps.
 */
function naiveMs(d: Date): number {
  const t = new Date(0);
  t.setUTCFullYear(d.getFullYear(), d.getMonth(), d.getDate());
  t.setUTCHours(d.getHours(), d.getMinutes(), d.getSeconds(), d.getMilliseconds());
  return t.getTime();
}

/** Inverse of {@link naiveMs}: wall-clock value back to a local Date. */
function fromNaiveMs(ms: number): Date {
  const u = new Date(ms);
  return localDate(
    u.getUTCFullYear(),
    u.getUTCMonth(),
    u.getUTCDate(),
    u.getUTCHours(),
    u.getUTCMinutes(),
    u.getUTCSeconds(),
    u.getUTCMilliseconds()
  );
}

/** Day of the year (1-366) of a calendar date, computed without DST effects. */
function ordinalDay(year: number, month: number, day: number): number {
  const a = new Date(0);
  a.setUTCFullYear(year, month, day);
  const b = new Date(0);
  b.setUTCFullYear(year, 0, 1);
  return Math.round((a.getTime() - b.getTime()) / MS_PER_DAY) + 1;
}

function isLeapYear(y: number): boolean {
  return (y % 4 === 0 && y % 100 !== 0) || y % 400 === 0;
}

function daysInMonth(year: number, month: number): number {
  const d = new Date(0);
  d.setUTCFullYear(year, month + 1, 0);
  return d.getUTCDate();
}

/**
 * Parse a single value to a Date, returning null for missing values.
 *
 * null, undefined, NaN and empty strings count as missing (pandas' NaT).
 * The result is always a fresh Date, never the caller's instance.
 */
function toDateOrNull(value: unknown, index: number): Date | null {
  if (value === null || value === undefined) return null;
  if (value instanceof Date) {
    if (Number.isNaN(value.getTime())) {
      throw new DataValidationError(
        `DateTimeAccessor: element at index ${index} is an invalid Date`
      );
    }
    return new Date(value.getTime());
  }
  if (typeof value === "string") {
    const str = value.trim();
    if (str === "") return null;
    // A bare ISO date ('2024-01-01') is parsed by JS as UTC midnight, but the
    // component getters below read LOCAL time, so west of UTC the day/month
    // would come out off by one. Parse date-only and naive-datetime strings as
    // LOCAL time to keep parsing and extraction consistent (pandas' naive
    // semantics). Strings with an explicit Z or offset keep their time zone
    // (handled by the fallback Date parse).
    const dateOnly = /^(\d{4})-(\d{2})-(\d{2})$/.exec(str);
    if (dateOnly) {
      return checkedLocalDate(value, index, dateOnly, false);
    }
    const naive = /^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::(\d{2})(?:\.(\d+))?)?$/.exec(str);
    if (naive) {
      return checkedLocalDate(value, index, naive, true);
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
    if (Number.isNaN(value)) return null;
    if (!Number.isFinite(value)) {
      throw new DataValidationError(
        `DateTimeAccessor: element at index ${index} is not a finite number`
      );
    }
    const d = new Date(value);
    if (Number.isNaN(d.getTime())) {
      throw new DataValidationError(
        `DateTimeAccessor: epoch value ${value} at index ${index} is outside the supported date range`
      );
    }
    return d;
  }
  throw new DataValidationError(
    `DateTimeAccessor: element at index ${index} has unsupported type ${typeof value}`
  );
}

/**
 * Build a local Date from regex captures and reject out-of-range components
 * (month 13, February 30, hour 24, ...) that `Date` would silently roll over.
 */
function checkedLocalDate(
  source: string,
  index: number,
  m: RegExpExecArray,
  withTime: boolean
): Date {
  const year = Number(m[1]);
  const month = Number(m[2]) - 1;
  const day = Number(m[3]);
  const hour = withTime ? Number(m[4]) : 0;
  const minute = withTime ? Number(m[5]) : 0;
  const second = withTime && m[6] ? Number(m[6]) : 0;
  // Keep the first three fractional digits (truncate, like Date's own ISO parser).
  const ms = withTime && m[7] ? Number(`${m[7]}00`.slice(0, 3)) : 0;
  const d = localDate(year, month, day, hour, minute, second, ms);
  const valid =
    month >= 0 &&
    month < 12 &&
    day >= 1 &&
    day <= daysInMonth(year, month) &&
    hour < 24 &&
    minute < 60 &&
    second < 60 &&
    d.getFullYear() === year &&
    d.getMonth() === month &&
    d.getDate() === day;
  if (!valid) {
    throw new DataValidationError(
      `DateTimeAccessor: cannot parse '${source}' at index ${index} as a date`
    );
  }
  return d;
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

/** Frequencies accepted by `floor`, `ceil` and `round`. Lowercase `h` and `s` are pandas' current spellings. */
export type DateRoundFreq = "D" | "H" | "h" | "min" | "S" | "s";

function freqToMs(freq: DateRoundFreq): number {
  switch (freq) {
    case "D":
      return MS_PER_DAY;
    case "H":
    case "h":
      return MS_PER_HOUR;
    case "min":
      return MS_PER_MINUTE;
    case "S":
    case "s":
      return MS_PER_SECOND;
    default:
      throw new InvalidParameterError(
        `freq must be one of 'D', 'H', 'min', 'S'; received ${String(freq)}`,
        "freq",
        freq
      );
  }
}

/**
 * Apply a rounding rule to a Date's wall-clock value in units of `freq`.
 * `mode` picks the integer multiple: floor, ceil, or round half to even
 * (pandas' tie-breaking rule for `Timestamp.round`).
 */
function roundDate(d: Date, freq: DateRoundFreq, mode: "floor" | "ceil" | "round"): Date {
  const unit = freqToMs(freq);
  const t = naiveMs(d);
  const floored = Math.floor(t / unit) * unit;
  const rem = t - floored;
  let result = floored;
  if (mode === "ceil") {
    result = rem === 0 ? floored : floored + unit;
  } else if (mode === "round") {
    const twice = rem * 2;
    if (twice > unit) {
      result = floored + unit;
    } else if (twice === unit) {
      // Tie: pick the even multiple.
      result = (floored / unit) % 2 === 0 ? floored : floored + unit;
    }
  }
  return fromNaiveMs(result);
}

const pad2 = (n: number): string => String(n).padStart(2, "0");

/**
 * DateTime accessor for Series, providing vectorized date/time operations.
 *
 * Accessed via `series.dt`. Operates element-wise on Series containing
 * Date objects, ISO date strings, or epoch milliseconds.
 * Null, undefined, NaN and empty-string elements propagate as null in the output.
 *
 * Date-only and zone-less datetime strings are read as local time, and all
 * component getters (`year()`, `hour()`, ...) report local time, like
 * pandas' naive datetimes. Strings with an explicit `Z` or UTC offset denote
 * an exact instant and are shown in local time.
 *
 * @example
 * ```ts
 * const s = new Series([new Date('2024-01-15'), new Date('2024-06-20')]);
 * s.dt.year();       // Series([2024, 2024])
 * s.dt.month();      // Series([1, 6])
 * s.dt.dayOfWeek();  // Series([0, 3])  (Mon=0 ... Sun=6)
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

  private map<T>(fn: (date: Date) => T): Series<T | null> {
    return mapDateSeries(this._data, this._index, this._name, fn);
  }

  // ─── Component extraction ──────────────────────────────────────

  /** Calendar year of each date. */
  year(): Series<number | null> {
    return this.map((d) => d.getFullYear());
  }

  /** Month of each date, 1 (January) to 12 (December). */
  month(): Series<number | null> {
    return this.map((d) => d.getMonth() + 1);
  }

  /** Day of the month, 1-31. */
  day(): Series<number | null> {
    return this.map((d) => d.getDate());
  }

  /** Hour of the day, 0-23. */
  hour(): Series<number | null> {
    return this.map((d) => d.getHours());
  }

  /** Minute of the hour, 0-59. */
  minute(): Series<number | null> {
    return this.map((d) => d.getMinutes());
  }

  /** Second of the minute, 0-59. */
  second(): Series<number | null> {
    return this.map((d) => d.getSeconds());
  }

  /** Millisecond of the second, 0-999. */
  millisecond(): Series<number | null> {
    return this.map((d) => d.getMilliseconds());
  }

  /**
   * Day of week: Monday=0, Tuesday=1, ..., Sunday=6 (ISO convention).
   */
  dayOfWeek(): Series<number | null> {
    return this.map((d) => {
      const jsDay = d.getDay(); // 0=Sun, 1=Mon, ... 6=Sat
      return jsDay === 0 ? 6 : jsDay - 1; // Convert to Mon=0 ... Sun=6
    });
  }

  /**
   * Same as {@link DateTimeAccessor.dayOfWeek}.
   *
   * @deprecated Prefer {@link DateTimeAccessor.dayOfWeek}.
   */
  dayofweek(): Series<number | null> {
    return this.dayOfWeek();
  }

  /** Alias of {@link DateTimeAccessor.dayOfWeek}. */
  weekday(): Series<number | null> {
    return this.dayOfWeek();
  }

  /**
   * Day of year: 1-366.
   */
  dayOfYear(): Series<number | null> {
    return this.map((d) => ordinalDay(d.getFullYear(), d.getMonth(), d.getDate()));
  }

  /**
   * Same as {@link DateTimeAccessor.dayOfYear}.
   *
   * @deprecated Prefer {@link DateTimeAccessor.dayOfYear}.
   */
  dayofyear(): Series<number | null> {
    return this.dayOfYear();
  }

  /**
   * ISO 8601 week number (1-53). Weeks start on Monday and week 1 is the week
   * that contains the year's first Thursday.
   */
  weekOfYear(): Series<number | null> {
    return this.map((d) => {
      // The ISO week belongs to the year of its Thursday.
      const jsDay = d.getDay();
      const shiftToThursday = 3 - ((jsDay + 6) % 7);
      const thursday = localDate(d.getFullYear(), d.getMonth(), d.getDate() + shiftToThursday);
      return Math.ceil(
        ordinalDay(thursday.getFullYear(), thursday.getMonth(), thursday.getDate()) / 7
      );
    });
  }

  /**
   * Same as {@link DateTimeAccessor.weekOfYear}.
   *
   * @deprecated Prefer {@link DateTimeAccessor.weekOfYear}.
   */
  weekofyear(): Series<number | null> {
    return this.weekOfYear();
  }

  /** Alias of {@link DateTimeAccessor.weekOfYear}. */
  week(): Series<number | null> {
    return this.weekOfYear();
  }

  /**
   * Quarter: 1-4.
   */
  quarter(): Series<number | null> {
    return this.map((d) => Math.floor(d.getMonth() / 3) + 1);
  }

  /**
   * Whether the year is a leap year.
   */
  isLeapYear(): Series<boolean | null> {
    return this.map((d) => isLeapYear(d.getFullYear()));
  }

  /**
   * Same as {@link DateTimeAccessor.isLeapYear}.
   *
   * @deprecated Prefer {@link DateTimeAccessor.isLeapYear}.
   */
  is_leap_year(): Series<boolean | null> {
    return this.isLeapYear();
  }

  /**
   * Days in the month of each date.
   */
  daysInMonth(): Series<number | null> {
    return this.map((d) => daysInMonth(d.getFullYear(), d.getMonth()));
  }

  /**
   * Same as {@link DateTimeAccessor.daysInMonth}.
   *
   * @deprecated Prefer {@link DateTimeAccessor.daysInMonth}.
   */
  days_in_month(): Series<number | null> {
    return this.daysInMonth();
  }

  /**
   * Same as {@link DateTimeAccessor.daysInMonth}.
   *
   * @deprecated Prefer {@link DateTimeAccessor.daysInMonth}.
   */
  daysinmonth(): Series<number | null> {
    return this.daysInMonth();
  }

  /** Whether each date is the first day of its month. */
  isMonthStart(): Series<boolean | null> {
    return this.map((d) => d.getDate() === 1);
  }

  /**
   * Same as {@link DateTimeAccessor.isMonthStart}.
   *
   * @deprecated Prefer {@link DateTimeAccessor.isMonthStart}.
   */
  is_month_start(): Series<boolean | null> {
    return this.isMonthStart();
  }

  /** Whether each date is the last day of its month. */
  isMonthEnd(): Series<boolean | null> {
    return this.map((d) => d.getDate() === daysInMonth(d.getFullYear(), d.getMonth()));
  }

  /**
   * Same as {@link DateTimeAccessor.isMonthEnd}.
   *
   * @deprecated Prefer {@link DateTimeAccessor.isMonthEnd}.
   */
  is_month_end(): Series<boolean | null> {
    return this.isMonthEnd();
  }

  /** Whether each date is the first day of a quarter. */
  isQuarterStart(): Series<boolean | null> {
    return this.map((d) => d.getDate() === 1 && d.getMonth() % 3 === 0);
  }

  /**
   * Same as {@link DateTimeAccessor.isQuarterStart}.
   *
   * @deprecated Prefer {@link DateTimeAccessor.isQuarterStart}.
   */
  is_quarter_start(): Series<boolean | null> {
    return this.isQuarterStart();
  }

  /** Whether each date is the last day of a quarter. */
  isQuarterEnd(): Series<boolean | null> {
    return this.map(
      (d) => d.getMonth() % 3 === 2 && d.getDate() === daysInMonth(d.getFullYear(), d.getMonth())
    );
  }

  /**
   * Same as {@link DateTimeAccessor.isQuarterEnd}.
   *
   * @deprecated Prefer {@link DateTimeAccessor.isQuarterEnd}.
   */
  is_quarter_end(): Series<boolean | null> {
    return this.isQuarterEnd();
  }

  /** Whether each date is January 1. */
  isYearStart(): Series<boolean | null> {
    return this.map((d) => d.getMonth() === 0 && d.getDate() === 1);
  }

  /**
   * Same as {@link DateTimeAccessor.isYearStart}.
   *
   * @deprecated Prefer {@link DateTimeAccessor.isYearStart}.
   */
  is_year_start(): Series<boolean | null> {
    return this.isYearStart();
  }

  /** Whether each date is December 31. */
  isYearEnd(): Series<boolean | null> {
    return this.map((d) => d.getMonth() === 11 && d.getDate() === 31);
  }

  /**
   * Same as {@link DateTimeAccessor.isYearEnd}.
   *
   * @deprecated Prefer {@link DateTimeAccessor.isYearEnd}.
   */
  is_year_end(): Series<boolean | null> {
    return this.isYearEnd();
  }

  /** English weekday name of each date ("Monday", ...). */
  dayName(): Series<string | null> {
    return this.map((d) => DAY_NAMES[d.getDay()] as string);
  }

  /**
   * Same as {@link DateTimeAccessor.dayName}.
   *
   * @deprecated Prefer {@link DateTimeAccessor.dayName}.
   */
  day_name(): Series<string | null> {
    return this.dayName();
  }

  /** English month name of each date ("January", ...). */
  monthName(): Series<string | null> {
    return this.map((d) => MONTH_NAMES[d.getMonth()] as string);
  }

  /**
   * Same as {@link DateTimeAccessor.monthName}.
   *
   * @deprecated Prefer {@link DateTimeAccessor.monthName}.
   */
  month_name(): Series<string | null> {
    return this.monthName();
  }

  // ─── Conversion ────────────────────────────────────────────────

  /**
   * Convert each date to epoch milliseconds (UTC based, independent of the local zone).
   */
  timestamp(): Series<number | null> {
    return this.map((d) => d.getTime());
  }

  /**
   * Convert each date to an ISO 8601 string in UTC, such as
   * `"2024-01-01T12:00:00.000Z"`. Unlike the component getters, this
   * is not local time: the trailing `Z` makes the string an exact instant that
   * `toDatetime` parses back to the same Date.
   */
  isoformat(): Series<string | null> {
    return this.map((d) => d.toISOString());
  }

  /**
   * Format each date using a strftime-style pattern, in local time.
   *
   * Supported directives: `%Y` `%y` `%m` `%d` `%H` `%I` `%M` `%S` `%f`
   * (microseconds, 6 digits) `%p` (AM/PM) `%a` `%A` `%b` `%B` `%j` `%w`
   * (Sunday=0) `%u` (Monday=1) `%F` (`%Y-%m-%d`) `%T` (`%H:%M:%S`) and `%%`.
   * Unknown directives are left unchanged.
   *
   * @param fmt - Format string
   * @throws {InvalidParameterError} If fmt is not a string
   *
   * @example
   * ```ts
   * toDatetime(['2024-01-15']).dt.strftime('%d %B %Y');  // ['15 January 2024']
   * ```
   */
  strftime(fmt: string): Series<string | null> {
    if (typeof fmt !== "string") {
      throw new InvalidParameterError("fmt must be a string", "fmt", fmt);
    }
    return this.map((d) => formatDate(d, fmt));
  }

  // ─── Date-only (truncation) ────────────────────────────────────

  /**
   * Truncate to date only (midnight).
   */
  date(): Series<Date | null> {
    return this.map((d) => localDate(d.getFullYear(), d.getMonth(), d.getDate()));
  }

  /** Alias of {@link DateTimeAccessor.date}, like pandas' `normalize()`. */
  normalize(): Series<Date | null> {
    return this.date();
  }

  /**
   * Extract the time portion as "HH:MM:SS".
   */
  time(): Series<string | null> {
    return this.map((d) => `${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`);
  }

  // ─── Floor / Ceil / Round ──────────────────────────────────────

  /**
   * Floor each date to the start of the given frequency.
   * @param freq - 'D' (day), 'H' (hour), 'min' (minute), 'S' (second)
   * @throws {InvalidParameterError} If freq is not one of the supported values
   */
  floor(freq: DateRoundFreq): Series<Date | null> {
    freqToMs(freq);
    return this.map((d) => roundDate(d, freq, "floor"));
  }

  /**
   * Ceil each date to the next boundary of the given frequency.
   * Dates already on a boundary are unchanged.
   * @throws {InvalidParameterError} If freq is not one of the supported values
   */
  ceil(freq: DateRoundFreq): Series<Date | null> {
    freqToMs(freq);
    return this.map((d) => roundDate(d, freq, "ceil"));
  }

  /**
   * Round each date to the nearest boundary of the given frequency. Exact ties
   * go to the even multiple of the unit, as in pandas (`00:00:30` rounds to
   * `00:00` and `00:01:30` to `00:02` at 'min').
   * @throws {InvalidParameterError} If freq is not one of the supported values
   */
  round(freq: DateRoundFreq): Series<Date | null> {
    freqToMs(freq);
    return this.map((d) => roundDate(d, freq, "round"));
  }
}

/** Format one Date for {@link DateTimeAccessor.strftime}. */
function formatDate(d: Date, fmt: string): string {
  return fmt.replace(/%([a-zA-Z%])/g, (match, code: string) => {
    switch (code) {
      case "Y":
        return String(d.getFullYear()).padStart(4, "0");
      case "y":
        return pad2(((d.getFullYear() % 100) + 100) % 100);
      case "m":
        return pad2(d.getMonth() + 1);
      case "d":
        return pad2(d.getDate());
      case "H":
        return pad2(d.getHours());
      case "I":
        return pad2(d.getHours() % 12 || 12);
      case "M":
        return pad2(d.getMinutes());
      case "S":
        return pad2(d.getSeconds());
      case "f":
        return String(d.getMilliseconds() * 1000).padStart(6, "0");
      case "p":
        return d.getHours() < 12 ? "AM" : "PM";
      case "a":
        return (DAY_NAMES[d.getDay()] as string).slice(0, 3);
      case "A":
        return DAY_NAMES[d.getDay()] as string;
      case "b":
        return (MONTH_NAMES[d.getMonth()] as string).slice(0, 3);
      case "B":
        return MONTH_NAMES[d.getMonth()] as string;
      case "j":
        return String(ordinalDay(d.getFullYear(), d.getMonth(), d.getDate())).padStart(3, "0");
      case "w":
        return String(d.getDay());
      case "u":
        return String(d.getDay() === 0 ? 7 : d.getDay());
      case "F":
        return `${String(d.getFullYear()).padStart(4, "0")}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}`;
      case "T":
        return `${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`;
      case "%":
        return "%";
      default:
        return match;
    }
  });
}

// ─── Standalone functions ──────────────────────────────────────────

/**
 * Frequencies accepted by {@link date_range}: day, hour, minute, second, month
 * start (`MS`) and year start (`YS`). Lowercase `h` and `s` are pandas' current
 * spellings of `H` and `S`.
 */
export type DateFreq = "D" | "H" | "h" | "min" | "S" | "s" | "MS" | "YS";

/**
 * Parse an array of values to Date Series.
 *
 * Accepts strings (ISO 8601), epoch milliseconds (numbers), or Date objects.
 * null, undefined, NaN and empty strings become null. Date-only and zone-less
 * datetime strings are read as local time; strings with `Z` or an offset keep
 * their exact instant. Out-of-range components such as `2024-02-30` throw.
 * The returned Dates are new objects, never the input instances.
 *
 * @param data - Array of date-like values
 * @param options - Series options (name, index)
 * @returns Series of Date objects
 * @throws {DataValidationError} If an element cannot be parsed as a date
 *
 * @example
 * ```ts
 * import { toDatetime } from 'deepbox/dataframe';
 * const dates = toDatetime(['2024-01-01', '2024-06-15', '2024-12-31']);
 * dates.dt.month();  // Series([1, 6, 12])
 * ```
 *
 * @deprecated Prefer {@link toDatetime}.
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
 * `D` steps by calendar days (the time of day is kept across DST changes);
 * `H`, `min` and `S` step by elapsed time. `MS` and `YS` follow pandas: the
 * first value is the first month (or year) start on or after `start`, keeping
 * the time of day, and later values step one month (or year) at a time.
 *
 * @param start - Start date (string, Date, or epoch ms)
 * @param periods - Number of periods to generate
 * @param freq - Frequency: 'D' (day), 'H' (hour), 'min' (minute), 'S' (second), 'MS' (month start), 'YS' (year start)
 * @param options - Series options (name, index) for the result
 * @returns Series of Dates
 * @throws {InvalidParameterError} If periods is not a non-negative integer, start is missing, or freq is unknown
 *
 * @example
 * ```ts
 * import { dateRange } from 'deepbox/dataframe';
 * const dates = dateRange('2024-01-01', 5, 'D');
 * // Series([Date(Jan 1), Date(Jan 2), ..., Date(Jan 5)])
 * ```
 *
 * @deprecated Prefer {@link dateRange}.
 */
export function date_range(
  start: string | Date | number,
  periods: number,
  freq: DateFreq = "D",
  options: SeriesOptions = {}
): Series<Date> {
  if (!Number.isFinite(periods) || !Number.isInteger(periods) || periods < 0) {
    throw new InvalidParameterError("periods must be a non-negative integer", "periods", periods);
  }

  const startDate = toDateOrNull(start, 0);
  if (startDate === null) {
    throw new InvalidParameterError("start must be a valid date value", "start", start);
  }

  const dates: Date[] = new Array(periods);
  const step = makeStepper(startDate, freq);
  for (let i = 0; i < periods; i++) {
    dates[i] = step(i);
  }

  return new Series(dates, options);
}

/** Returns a function mapping the period number to its Date. */
function makeStepper(base: Date, freq: DateFreq): (n: number) => Date {
  const y = base.getFullYear();
  const mo = base.getMonth();
  const dd = base.getDate();
  const h = base.getHours();
  const mi = base.getMinutes();
  const sec = base.getSeconds();
  const ms = base.getMilliseconds();
  switch (freq) {
    case "D":
      return (n) => localDate(y, mo, dd + n, h, mi, sec, ms);
    case "H":
    case "h":
      return (n) => new Date(base.getTime() + n * MS_PER_HOUR);
    case "min":
      return (n) => new Date(base.getTime() + n * MS_PER_MINUTE);
    case "S":
    case "s":
      return (n) => new Date(base.getTime() + n * MS_PER_SECOND);
    case "MS": {
      const first = dd === 1 ? mo : mo + 1;
      return (n) => localDate(y, first + n, 1, h, mi, sec, ms);
    }
    case "YS": {
      const first = mo === 0 && dd === 1 ? y : y + 1;
      return (n) => localDate(first + n, 0, 1, h, mi, sec, ms);
    }
    default:
      throw new InvalidParameterError(
        `freq must be one of 'D', 'H', 'min', 'S', 'MS', 'YS'; received ${String(freq)}`,
        "freq",
        freq
      );
  }
}

/**
 * Compute the difference between two Date Series in the given unit.
 *
 * @param left - Left operand Series of Dates
 * @param right - Right operand Series of Dates
 * @param unit - Unit for result: 'ms', 'S', 'min', 'H', 'D'
 * @returns Series<number | null> of differences (left - right) in the given unit,
 *   indexed like `left`. A null on either side gives null.
 * @throws {DataValidationError} If the Series lengths differ
 * @throws {InvalidParameterError} If unit is unknown
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
      return MS_PER_SECOND;
    case "min":
      return MS_PER_MINUTE;
    case "H":
      return MS_PER_HOUR;
    case "D":
      return MS_PER_DAY;
    default:
      throw new InvalidParameterError(
        `unit must be one of 'ms', 'S', 'min', 'H', 'D'; received ${String(unit)}`,
        "unit",
        unit
      );
  }
}

/**
 * Generate a range of dates at a fixed frequency.
 *
 * `D` steps by calendar days (the time of day is kept across DST changes);
 * `H`, `min` and `S` step by elapsed time. `MS` and `YS` follow pandas: the
 * first value is the first month (or year) start on or after `start`, keeping
 * the time of day, and later values step one month (or year) at a time.
 *
 * @param start - Start date (string, Date, or epoch ms)
 * @param periods - Number of periods to generate
 * @param freq - Frequency: 'D' (day), 'H' (hour), 'min' (minute), 'S' (second), 'MS' (month start), 'YS' (year start)
 * @param options - Series options (name, index) for the result
 * @returns Series of Dates
 * @throws {InvalidParameterError} If periods is not a non-negative integer, start is missing, or freq is unknown
 *
 * @example
 * ```ts
 * import { dateRange } from 'deepbox/dataframe';
 * const dates = dateRange('2024-01-01', 5, 'D');
 * // Series([Date(Jan 1), Date(Jan 2), ..., Date(Jan 5)])
 * ```
 */
export const dateRange = date_range;

/**
 * Parse an array of values to Date Series.
 *
 * Accepts strings (ISO 8601), epoch milliseconds (numbers), or Date objects.
 * null, undefined, NaN and empty strings become null. Date-only and zone-less
 * datetime strings are read as local time; strings with `Z` or an offset keep
 * their exact instant. Out-of-range components such as `2024-02-30` throw.
 * The returned Dates are new objects, never the input instances.
 *
 * @param data - Array of date-like values
 * @param options - Series options (name, index)
 * @returns Series of Date objects
 * @throws {DataValidationError} If an element cannot be parsed as a date
 *
 * @example
 * ```ts
 * import { toDatetime } from 'deepbox/dataframe';
 * const dates = toDatetime(['2024-01-01', '2024-06-15', '2024-12-31']);
 * dates.dt.month();  // Series([1, 6, 12])
 * ```
 */
export const toDatetime = to_datetime;
