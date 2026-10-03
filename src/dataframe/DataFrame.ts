import { type Axis, normalizeAxis } from "../core";
import { DataValidationError, IndexError, InvalidParameterError } from "../core/errors/index";
import { reshape, type Tensor, tensor } from "../ndarray/index";
import { Generator } from "../random/Generator";
import { __random, __randomBelow } from "../random/random";
import { correlate, averageRanks as correlationRanks } from "./correlation";
import {
  type ParquetReadOptions,
  type ParquetWriteOptions,
  readParquet,
  readXlsx,
  writeParquet,
  writeXlsx,
  type XlsxReadOptions,
  type XlsxWriteOptions,
} from "./io/index";
import { PlotAccessor } from "./PlotAccessor";
import { Series } from "./Series";
import { StyleAccessor } from "./StyleAccessor";
import type {
  AggregateFunction,
  ConcatOptions,
  CorrelationMethod,
  CorrOptions,
  DataFrameData,
  DataFrameOptions,
  FillMethod,
  FillnaMethodOptions,
  FillOptions,
  GroupByOptions,
  NamedAggregation,
  RollingOptions,
  SampleOptions,
  ValueCountsOptions,
} from "./types";
import {
  cellText,
  checkFillLimit,
  checkFillMethod,
  compareStrings,
  compensatedSum,
  createKey,
  isPlainObject,
  isRecord,
  isValidNumber,
  propagateFill,
  sumSquaredDeviations,
} from "./utils";

const isIndexLabel = (value: unknown): value is string | number =>
  typeof value === "string" || typeof value === "number";

const isStringArray = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every((entry) => typeof entry === "string");

const isIndexLabelArray = (value: unknown): value is (string | number)[] =>
  Array.isArray(value) && value.every(isIndexLabel);

const AGGREGATE_FUNCTIONS: readonly string[] = [
  "sum",
  "mean",
  "median",
  "min",
  "max",
  "std",
  "var",
  "count",
  "first",
  "last",
  "nunique",
];

const FILL_METHOD_NAMES: readonly string[] = ["ffill", "bfill", "pad", "backfill"];

const JOIN_KINDS: readonly string[] = ["inner", "left", "right", "outer"];

const assertCsvCharacters = (delimiter: string, quoteChar: string): void => {
  if (
    typeof delimiter !== "string" ||
    delimiter.length !== 1 ||
    delimiter === "\n" ||
    delimiter === "\r"
  ) {
    throw new InvalidParameterError(
      "delimiter must be a single character other than a line break",
      "delimiter",
      delimiter
    );
  }
  if (
    typeof quoteChar !== "string" ||
    quoteChar.length !== 1 ||
    quoteChar === "\n" ||
    quoteChar === "\r"
  ) {
    throw new InvalidParameterError(
      "quoteChar must be a single character other than a line break",
      "quoteChar",
      quoteChar
    );
  }
  if (delimiter === quoteChar) {
    throw new InvalidParameterError(
      "delimiter and quoteChar must be different",
      "quoteChar",
      quoteChar
    );
  }
};

const ensureUniqueLabels = (labels: readonly string[], labelName: string): void => {
  const seen = new Set<string>();
  for (const label of labels) {
    if (seen.has(label)) {
      throw new DataValidationError(`Duplicate ${labelName} '${label}' is not supported`);
    }
    seen.add(label);
  }
};

/** True for null, undefined and NaN (the values treated as "missing"). */
const isMissing = (value: unknown): boolean =>
  value === null || value === undefined || (typeof value === "number" && Number.isNaN(value));

/** A number that is not NaN. Unlike `isValidNumber`, infinities are accepted. */
const isOrderedNumber = (value: unknown): value is number =>
  typeof value === "number" && !Number.isNaN(value);

const compareNumbers = (a: number, b: number): number => (a < b ? -1 : a > b ? 1 : 0);

/**
 * Linear-interpolated quantile of an ascending, NaN-free array
 * (NumPy's "linear" method, including its overflow-safe lerp).
 */
const quantileSorted = (sorted: ArrayLike<number>, q: number): number => {
  const idx = q * (sorted.length - 1);
  const lower = Math.floor(idx);
  const weight = idx - lower;
  const a = sorted[lower] as number;
  if (weight === 0) return a;
  const b = sorted[Math.ceil(idx)] as number;
  const diff = b - a;
  return weight >= 0.5 ? b - diff * (1 - weight) : a + diff * weight;
};

/** Rounding can push a correlation a hair past +-1; keep it inside the valid range. */
const clampCorrelation = (r: number): number => (r > 1 ? 1 : r < -1 ? -1 : r);

/** Round half to even (IEEE "banker's" rounding), as NumPy and pandas do. */
const roundHalfEven = (x: number): number => {
  const floor = Math.floor(x);
  const diff = x - floor;
  if (diff < 0.5) return floor;
  if (diff > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
};

/**
 * Draws `count` row positions out of `total`. Without weights, distinct positions come from a
 * partial Fisher-Yates shuffle (or independent uniform draws with replacement). With weights,
 * each draw is proportional to the weights; without replacement a drawn row's weight is set to
 * zero in a Fenwick tree, so a draw costs O(log total).
 */
const drawRows = (
  total: number,
  count: number,
  replace: boolean,
  weights: Float64Array | undefined,
  rng: () => number
): number[] => {
  const picked: number[] = new Array<number>(count);
  if (count === 0) return picked;
  if (weights === undefined) {
    if (replace) {
      for (let k = 0; k < count; k++) picked[k] = __randomBelow(rng, total);
      return picked;
    }
    const positions = Array.from({ length: total }, (_, i) => i);
    for (let k = 0; k < count; k++) {
      const j = k + __randomBelow(rng, total - k);
      const chosen = positions[j] as number;
      positions[j] = positions[k] as number;
      positions[k] = chosen;
      picked[k] = chosen;
    }
    return picked;
  }
  const tree = new Float64Array(total + 1);
  for (let i = 1; i <= total; i++) {
    tree[i] = (tree[i] as number) + (weights[i - 1] as number);
    const parent = i + (i & -i);
    if (parent <= total) tree[parent] = (tree[parent] as number) + (tree[i] as number);
  }
  let top = 1;
  while (top * 2 <= total) top *= 2;
  let remaining = 0;
  for (let i = 0; i < total; i++) remaining += weights[i] as number;
  const find = (target: number): number => {
    let pos = 0;
    let rest = target;
    for (let step = top; step > 0; step >>= 1) {
      const next = pos + step;
      if (next <= total && (tree[next] as number) <= rest) {
        pos = next;
        rest -= tree[next] as number;
      }
    }
    return pos;
  };
  const lastPositive = (): number => {
    for (let i = total - 1; i >= 0; i--) if ((weights[i] as number) > 0) return i;
    return total - 1;
  };
  for (let k = 0; k < count; k++) {
    let row = find(rng() * remaining);
    // Rounding can leave the target at or past the end, or on a row already removed.
    if (row >= total || (weights[row] as number) <= 0) row = lastPositive();
    picked[k] = row;
    if (!replace) {
      const w = weights[row] as number;
      for (let i = row + 1; i <= total; i += i & -i) tree[i] = (tree[i] as number) - w;
      weights[row] = 0;
      remaining -= w;
    }
  }
  return picked;
};

/**
 * Total order used by sort(): numbers and bigints numerically, strings by
 * UTF-16 code unit (locale independent, like pandas), dates by timestamp,
 * booleans false < true. Mixed types fall back to comparing their string forms.
 * Callers handle missing values first.
 */
const compareValues = (a: unknown, b: unknown): number => {
  if (
    (typeof a === "number" || typeof a === "bigint") &&
    (typeof b === "number" || typeof b === "bigint")
  ) {
    return a < b ? -1 : a > b ? 1 : 0;
  }
  if (typeof a === "string" && typeof b === "string") return compareStrings(a, b);
  if (a instanceof Date && b instanceof Date) return compareNumbers(a.getTime(), b.getTime());
  if (typeof a === "boolean" && typeof b === "boolean") return Number(a) - Number(b);
  return compareStrings(String(a), String(b));
};

/** Sort order for labels in reshaped output: numbers ascending, otherwise by string form. */
const compareLabels = (a: unknown, b: unknown): number => {
  if (typeof a === "number" && typeof b === "number") return compareNumbers(a, b);
  const sa = String(a);
  const sb = String(b);
  return sa < sb ? -1 : sa > sb ? 1 : 0;
};

/**
 * Empty column map for building a DataFrame. It has no prototype, so a column
 * called "__proto__" or "toString" is stored as an ordinary entry.
 */
const newColumnMap = (): DataFrameData => Object.create(null) as DataFrameData;

/** Sets `target[key]`; a key named "__proto__" becomes an own property, not a prototype change. */
const setField = (target: Record<string, unknown>, key: string, value: unknown): void => {
  if (key === "__proto__") {
    Object.defineProperty(target, key, {
      value,
      writable: true,
      enumerable: true,
      configurable: true,
    });
  } else {
    target[key] = value;
  }
};

/** Numbers of one group, skipping missing values and rejecting other types. */
const numbersOf = (
  data: readonly unknown[],
  indices: readonly number[],
  func: string
): number[] => {
  const nums: number[] = [];
  for (const idx of indices) {
    const val = data[idx];
    if (val === null || val === undefined) continue;
    if (typeof val !== "number") {
      throw new DataValidationError(`${func}() only works on numbers`);
    }
    if (!Number.isNaN(val)) nums.push(val);
  }
  return nums;
};

/** Sample variance (ddof = 1) with the corrected two-pass formula; NaN for fewer than two values. */
const sampleVariance = (nums: number[]): number => {
  if (nums.length < 2) return NaN;
  // Same corrected two-pass formula as Series.var, so both give the same bits.
  return Math.max(0, sumSquaredDeviations(nums)) / (nums.length - 1);
};

/** Applies one named aggregation to the rows `indices` of a column. */
const aggregateRows = (
  func: AggregateFunction,
  data: readonly unknown[],
  indices: readonly number[]
): unknown => {
  switch (func) {
    case "count": {
      let count = 0;
      for (const idx of indices) {
        if (!isMissing(data[idx])) count++;
      }
      return count;
    }
    case "nunique": {
      const seen = new Set<string>();
      for (const idx of indices) {
        const val = data[idx];
        if (!isMissing(val)) seen.add(createKey(val));
      }
      return seen.size;
    }
    case "first": {
      const firstIdx = indices[0];
      return firstIdx !== undefined ? data[firstIdx] : undefined;
    }
    case "last": {
      const lastIdx = indices[indices.length - 1];
      return lastIdx !== undefined ? data[lastIdx] : undefined;
    }
    case "sum":
      return compensatedSum(numbersOf(data, indices, "sum"));
    case "mean": {
      const nums = numbersOf(data, indices, "mean");
      return nums.length > 0 ? compensatedSum(nums) / nums.length : NaN;
    }
    case "median": {
      const nums = numbersOf(data, indices, "median");
      return nums.length > 0 ? quantileSorted(Float64Array.from(nums).sort(), 0.5) : NaN;
    }
    case "min": {
      const nums = numbersOf(data, indices, "min");
      let min = NaN;
      for (const val of nums) if (Number.isNaN(min) || val < min) min = val;
      return min;
    }
    case "max": {
      const nums = numbersOf(data, indices, "max");
      let max = NaN;
      for (const val of nums) if (Number.isNaN(max) || val > max) max = val;
      return max;
    }
    case "std":
      return Math.sqrt(sampleVariance(numbersOf(data, indices, "std")));
    case "var":
      return sampleVariance(numbersOf(data, indices, "var"));
    default:
      throw new DataValidationError(`Unsupported aggregation function: ${String(func)}`);
  }
};

/** One output column of a group aggregation: how to reduce the rows `indices` of `data`. */
type GroupPlanEntry = {
  readonly outCol: string;
  readonly data: readonly unknown[];
  readonly run: (data: readonly unknown[], indices: readonly number[]) => unknown;
};

/** Row-wise transformations of `DataFrameGroupBy.transform` that keep the group's length. */
const TRANSFORM_FUNCTIONS: readonly string[] = [
  "cumsum",
  "cumprod",
  "cummax",
  "cummin",
  "ffill",
  "bfill",
];

/** Converts a cell value into a legal row label. */
const toIndexLabel = (value: unknown): string | number =>
  typeof value === "string" || typeof value === "number" ? value : String(value);

/**
 * Two-dimensional, size-mutable, potentially heterogeneous tabular data.
 *
 * A DataFrame is like a spreadsheet or SQL table. It consists of:
 * - Rows (observations) identified by an index
 * - Columns (variables) identified by column names
 * - Data stored in a columnar format for efficient access
 *
 * A tabular data structure with labeled columns. @see { https://deepbox.dev/docs/dataframe-overview | Deepbox DataFrame}
 *
 * @example
 * ```ts
 * import { DataFrame } from 'deepbox/dataframe';
 *
 * const df = new DataFrame({
 *   name: ['Alice', 'Bob', 'Charlie'],
 *   age: [25, 30, 35],
 *   score: [85.5, 92.0, 78.5]
 * });
 *
 * console.log(df.shape);  // [3, 3]
 * console.log(df.columns);  // ['name', 'age', 'score']
 * ```
 *
 * @see {@link https://deepbox.dev/docs/dataframe-overview | Deepbox DataFrame}
 */
export class DataFrame {
  // Internal storage: Map of column names to data arrays
  private _data: Map<string, unknown[]>;
  // Row labels (can be strings or numbers)
  private _index: (string | number)[];
  // Fast label -> position lookup for O(1) loc() access
  private _indexPos: Map<string | number, number>;
  // Column names
  private _columns: string[];

  /**
   * Creates a new DataFrame instance.
   *
   * @param data - Object mapping column names to arrays of values.
   *               All arrays must have the same length.
   * @param options - Configuration options
   * @param options.columns - Custom column order (defaults to Object.keys(data))
   * @param options.index - Custom row labels (defaults to 0, 1, 2, ...)
   *
   * @example
   * ```ts
   * const df = new DataFrame({
   *   col1: [1, 2, 3],
   *   col2: ['a', 'b', 'c']
   * }, {
   *   index: ['row1', 'row2', 'row3']
   * });
   * ```
   */
  constructor(data: DataFrameData, options: DataFrameOptions = {}) {
    // Determine column order (use provided order or infer from data object keys)
    this._columns = options.columns ? [...options.columns] : Object.keys(data);
    ensureUniqueLabels(this._columns, "column name");

    // If user provided columns, enforce that each requested column exists in data.
    for (const col of this._columns) {
      if (!Object.hasOwn(data, col)) {
        throw new DataValidationError(`Column '${col}' not found in DataFrame data`);
      }
    }

    // Determine number of rows from first column
    let firstColumnLength = 0;
    if (this._columns.length > 0) {
      const firstCol = this._columns[0];
      if (firstCol === undefined) {
        throw new DataValidationError("First column is undefined");
      }
      const firstColData = data[firstCol];
      if (!Array.isArray(firstColData)) {
        throw new DataValidationError(`Column '${firstCol}' must be an array`);
      }
      firstColumnLength = firstColData.length;
    }

    // Create row index (use provided labels or generate 0, 1, 2, ...)
    this._index = options.index
      ? options.copy === false
        ? options.index
        : [...options.index]
      : Array.from({ length: firstColumnLength }, (_, i) => i);

    // Validate index length matches the inferred row count.
    // If we have columns, row count is defined by column length.
    // If we have no columns (index-only DataFrame), row count is defined by index length.
    if (this._columns.length > 0 && this._index.length !== firstColumnLength) {
      throw new DataValidationError(
        `Index length (${this._index.length}) must match row count (${firstColumnLength})`
      );
    }

    // Build index lookup map and enforce unique labels (required for unambiguous O(1) loc()).
    this._indexPos = new Map();
    for (let i = 0; i < this._index.length; i++) {
      const label = this._index[i];
      if (!isIndexLabel(label)) {
        throw new DataValidationError(
          `Index label at position ${i} must be a string or number, received ${
            label === null ? "null" : typeof label
          }`
        );
      }
      if (this._indexPos.has(label)) {
        throw new DataValidationError(`Duplicate index label '${String(label)}' is not supported`);
      }
      this._indexPos.set(label, i);
    }

    // Store data in a Map for efficient column access
    this._data = new Map();
    for (const col of this._columns) {
      // Enforce column exists (validated above) and is aligned with row count.
      const colData = data[col];
      if (!Array.isArray(colData)) {
        throw new DataValidationError(`Column '${col}' must be an array`);
      }
      if (colData.length !== firstColumnLength) {
        throw new DataValidationError(
          `Column '${col}' length (${colData.length}) must match row count (${firstColumnLength})`
        );
      }
      // Store a copy to avoid external mutation, unless copy=false.
      this._data.set(col, options.copy === false ? colData : [...colData]);
    }
  }

  /**
   * Get the dimensions of the DataFrame.
   *
   * @returns Tuple of [rows, columns]
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, 2, 3], b: [4, 5, 6] });
   * df.shape;  // [3, 2]
   * ```
   */
  get shape(): [number, number] {
    return [this._index.length, this._columns.length];
  }

  /**
   * Plotting accessor. Provides pandas-like `df.plot.line()`, `df.plot.bar()`, etc.
   *
   * @example
   * ```ts
   * const fig = df.plot.line({ x: 'date', y: 'price' });
   * const fig2 = df.plot.scatter({ x: 'x', y: 'y' });
   * ```
   */
  get plot(): PlotAccessor {
    return new PlotAccessor(() => ({
      columns: this._columns.slice(),
      getColumn: (name: string) => this.getColumnData(name).slice(),
      nRows: this._index.length,
    }));
  }

  /**
   * Style accessor. Provides pandas-like conditional formatting with
   * `df.style.highlightMax()`, `df.style.backgroundGradient()`, etc.
   *
   * @example
   * ```ts
   * const html = df.style.highlightMax().toHTML();
   * ```
   */
  get style(): StyleAccessor {
    return new StyleAccessor(() => ({
      columns: this._columns.slice(),
      getColumn: (name: string) => this.getColumnData(name).slice(),
      index: this._index.slice(),
      nRows: this._index.length,
    }));
  }

  /**
   * Get the column names.
   *
   * @returns Array of column names (copy)
   */
  get columns(): string[] {
    return [...this._columns];
  }

  /**
   * Get the row index labels.
   *
   * @returns Array of index labels (copy)
   */
  get index(): (string | number)[] {
    return [...this._index];
  }

  /**
   * Get a column as a Series.
   *
   * @param column - Column name to retrieve
   * @returns Series containing the column data
   * @throws {InvalidParameterError} If column doesn't exist
   *
   * @example
   * ```ts
   * const df = new DataFrame({ age: [25, 30, 35], name: ['Alice', 'Bob', 'Carol'] });
   * const ageSeries = df.get('age');  // Series([25, 30, 35])
   * ```
   */
  get(column: string): Series<unknown>;
  get<T>(column: string, guard: (value: unknown) => value is T): Series<T>;
  get<T>(column: string, guard?: (value: unknown) => value is T): Series<unknown> | Series<T> {
    // Check if column exists
    const data = this._data.get(column);
    if (data === undefined) {
      throw new InvalidParameterError(
        `Column '${column}' not found in DataFrame`,
        "column",
        column
      );
    }

    if (guard) {
      const validated: T[] = [];
      for (const value of data) {
        if (!guard(value)) {
          throw new DataValidationError(
            `Column '${column}' contains values that do not match the requested type`
          );
        }
        validated.push(value);
      }
      return new Series(validated, {
        index: this._index,
        name: column,
        copy: false,
      });
    }

    return new Series(data, {
      index: this._index,
      name: column,
      copy: false,
    });
  }

  /**
   * Raw column data without constructing a Series (no copies, no index map).
   *
   * @internal
   */
  getColumnData(column: string): readonly unknown[] {
    const data = this._data.get(column);
    if (!data) {
      throw new InvalidParameterError(
        `Column '${column}' not found in DataFrame`,
        "column",
        column
      );
    }
    return data;
  }

  /**
   * Access a row by label (label-based indexing).
   *
   * @param row - The index label of the row
   * @returns Object mapping column names to values for that row
   * @throws {IndexError} If row label not found
   *
   * @example
   * ```ts
   * const df = new DataFrame(
   *   { age: [25, 30], name: ['Alice', 'Bob'] },
   *   { index: ['row1', 'row2'] }
   * );
   * df.loc('row1');  // { age: 25, name: 'Alice' }
   * ```
   */
  loc(row: string | number): Record<string, unknown> {
    // Find position of this label in the index (O(1) via lookup map)
    const position = this._indexPos.get(row) ?? -1;

    if (position === -1) {
      throw new IndexError(`Row label '${row}' not found in index`);
    }

    // Build object with all column values for this row
    const result: Record<string, unknown> = {};
    for (const col of this._columns) {
      const colData = this._data.get(col);
      if (colData) {
        setField(result, col, colData[position]);
      }
    }

    return result;
  }

  /**
   * Access a row by integer position (position-based indexing).
   *
   * @param position - The integer position (0-based)
   * @returns Object mapping column names to values for that row
   * @throws {IndexError} If position is out of bounds
   *
   * @example
   * ```ts
   * const df = new DataFrame({ age: [25, 30], name: ['Alice', 'Bob'] });
   * df.iloc(0);  // { age: 25, name: 'Alice' }
   * df.iloc(1);  // { age: 30, name: 'Bob' }
   * ```
   */
  iloc(position: number): Record<string, unknown> {
    if (!Number.isInteger(position)) {
      throw new InvalidParameterError("position must be an integer", "position", position);
    }
    // Validate position is within bounds
    if (this._index.length === 0) {
      throw new IndexError(`DataFrame is empty`, {
        index: position,
        validRange: [0, 0],
      });
    }
    if (position < 0 || position >= this._index.length) {
      throw new IndexError(`Position ${position} is out of bounds (0-${this._index.length - 1})`, {
        index: position,
        validRange: [0, this._index.length - 1],
      });
    }

    // Build object with all column values at this position
    const result: Record<string, unknown> = {};
    for (const col of this._columns) {
      const colData = this._data.get(col);
      if (colData) {
        setField(result, col, colData[position]);
      }
    }

    return result;
  }

  /**
   * Return the first n rows.
   *
   * @param n - Number of rows to return (default: 5)
   * @returns New DataFrame with first n rows
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, 2, 3, 4, 5], b: [6, 7, 8, 9, 10] });
   * df.head(3);  // DataFrame with rows 0-2
   * ```
   */
  head(n: number = 5): DataFrame {
    if (!Number.isFinite(n) || !Number.isInteger(n) || n < 0) {
      throw new InvalidParameterError("n must be a non-negative integer", "n", n);
    }
    // Slice each column's data to first n rows
    const newData: DataFrameData = newColumnMap();
    for (const col of this._columns) {
      const colData = this._data.get(col);
      newData[col] = colData ? colData.slice(0, n) : [];
    }

    // Create new DataFrame with sliced data and index
    return new DataFrame(newData, {
      columns: this._columns,
      index: this._index.slice(0, n),
      copy: false,
    });
  }

  /**
   * Return the last n rows.
   *
   * @param n - Number of rows to return (default: 5)
   * @returns New DataFrame with last n rows
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, 2, 3, 4, 5], b: [6, 7, 8, 9, 10] });
   * df.tail(3);  // DataFrame with rows 2-4
   * ```
   */
  tail(n: number = 5): DataFrame {
    if (!Number.isFinite(n) || !Number.isInteger(n) || n < 0) {
      throw new InvalidParameterError("n must be a non-negative integer", "n", n);
    }
    // Clamp so that n > rowCount returns every row (a negative start would
    // otherwise be read by slice() as an offset from the end).
    const sliceStart = Math.max(0, this._index.length - n);
    const newData: DataFrameData = newColumnMap();
    for (const col of this._columns) {
      const colData = this._data.get(col);
      newData[col] = colData ? colData.slice(sliceStart) : [];
    }

    // Create new DataFrame with sliced data and index
    return new DataFrame(newData, {
      columns: this._columns,
      index: this._index.slice(sliceStart),
      copy: false,
    });
  }

  /**
   * Filter rows based on a boolean predicate function.
   *
   * The row object passed to `predicate` is reused between calls to avoid one
   * allocation per row, so copy it (`{ ...row }`) if you need to keep it.
   *
   * @param predicate - Function that returns true for rows to keep
   * @returns New DataFrame with filtered rows
   *
   * @example
   * ```ts
   * const df = new DataFrame({ age: [25, 30, 35], name: ['Alice', 'Bob', 'Carol'] });
   * const filtered = df.filter(row => row.age > 28);
   * // DataFrame with Bob and Carol
   * ```
   */
  // biome-ignore lint/suspicious/noExplicitAny: DataFrame rows are dynamically typed
  filter(predicate: (row: Record<string, any>) => boolean): DataFrame {
    const nCols = this._columns.length;
    const nRows = this._index.length;

    // Pre-fetch column arrays into a flat array for direct indexed access
    const colArrays: unknown[][] = new Array(nCols);
    for (let c = 0; c < nCols; c++) {
      colArrays[c] = this._data.get(this._columns[c] as string) ?? [];
    }

    // First pass: find matching row indices using a reusable row object
    const matchIndices: number[] = [];
    // biome-ignore lint/suspicious/noExplicitAny: DataFrame rows are dynamically typed
    const row: Record<string, any> = {};
    for (let i = 0; i < nRows; i++) {
      for (let c = 0; c < nCols; c++) {
        row[this._columns[c] as string] = (colArrays[c] as unknown[])[i];
      }
      if (predicate(row)) {
        matchIndices.push(i);
      }
    }

    // Second pass: build output columns from matched indices
    const matchCount = matchIndices.length;
    const filteredData: DataFrameData = newColumnMap();
    for (let c = 0; c < nCols; c++) {
      const src = colArrays[c] as unknown[];
      const dst = new Array(matchCount);
      for (let m = 0; m < matchCount; m++) {
        dst[m] = src[matchIndices[m] as number];
      }
      filteredData[this._columns[c] as string] = dst;
    }

    const filteredIndex = new Array<string | number>(matchCount);
    for (let m = 0; m < matchCount; m++) {
      filteredIndex[m] = this._index[matchIndices[m] as number] as string | number;
    }

    return new DataFrame(filteredData, {
      columns: this._columns,
      index: filteredIndex,
      copy: false,
    });
  }

  /**
   * Select a subset of columns.
   *
   * @param columns - Array of column names to select
   * @returns New DataFrame with only specified columns
   * @throws {InvalidParameterError} If any column doesn't exist
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, 2], b: [3, 4], c: [5, 6] });
   * df.select(['a', 'c']);  // DataFrame with only columns a and c
   * ```
   */
  select(columns: string[]): DataFrame {
    if (!isStringArray(columns)) {
      throw new InvalidParameterError("columns must be an array of strings", "columns", columns);
    }
    ensureUniqueLabels(columns, "column name");
    // Validate all columns exist
    for (const col of columns) {
      if (!this._data.has(col)) {
        throw new InvalidParameterError(`Column '${col}' not found in DataFrame`, "columns", col);
      }
    }

    // Build new data with only selected columns (slice to avoid shared mutation)
    const newData: DataFrameData = newColumnMap();
    for (const col of columns) {
      const colData = this._data.get(col);
      newData[col] = colData ? colData.slice() : [];
    }

    return new DataFrame(newData, {
      columns: columns,
      index: this._index,
      copy: false,
    });
  }

  /**
   * Drop (remove) specified columns.
   *
   * @param columns - Array of column names to drop
   * @returns New DataFrame without the dropped columns
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, 2], b: [3, 4], c: [5, 6] });
   * df.drop(['b']);  // DataFrame with only columns a and c
   * ```
   */
  drop(columns: string[]): DataFrame {
    if (!isStringArray(columns)) {
      throw new InvalidParameterError("columns must be an array of strings", "columns", columns);
    }
    ensureUniqueLabels(columns, "column name");
    for (const col of columns) {
      if (!this._data.has(col)) {
        throw new InvalidParameterError(`Column '${col}' not found in DataFrame`, "columns", col);
      }
    }

    // Get columns to keep (all columns except the ones to drop)
    const dropSet = new Set(columns);
    const columnsToKeep = this._columns.filter((col) => !dropSet.has(col));

    // Build new data with remaining columns
    const newData: DataFrameData = newColumnMap();
    for (const col of columnsToKeep) {
      const colData = this._data.get(col);
      newData[col] = colData ? [...colData] : [];
    }

    return new DataFrame(newData, {
      columns: columnsToKeep,
      index: this._index,
      copy: false,
    });
  }

  /**
   * Sort DataFrame by one or more columns.
   *
   * The sort is stable. Missing values (null, undefined, NaN) always go last,
   * for both sort directions. Numbers and bigints compare numerically, strings
   * by UTF-16 code unit (the same locale independent order as `Series.sort` and
   * pandas, so "B" sorts before "a"), dates by timestamp.
   *
   * @param by - Column name or array of column names to sort by
   * @param ascending - Sort in ascending order (default: true). Pass an array with
   *   one flag per sort column to mix directions.
   * @returns New sorted DataFrame
   *
   * @example
   * ```ts
   * const df = new DataFrame({ age: [30, 25, 35], name: ['Bob', 'Alice', 'Carol'] });
   * df.sort('age');  // Sorted by age ascending
   * df.sort(['age'], false);  // Sorted by age descending
   * df.sort(['name', 'age'], [true, false]);  // name ascending, then age descending
   * ```
   */
  sort(by: string | string[], ascending: boolean | boolean[] = true): DataFrame {
    const sortCols = Array.isArray(by) ? by : [by];

    // Validate sort columns exist
    for (const col of sortCols) {
      if (!this._data.has(col)) {
        throw new InvalidParameterError(`Column '${col}' not found in DataFrame`, "by", col);
      }
    }

    let directions: boolean[];
    if (Array.isArray(ascending)) {
      if (ascending.length !== sortCols.length) {
        throw new InvalidParameterError(
          `ascending has ${ascending.length} entries but ${sortCols.length} sort columns were given`,
          "ascending",
          ascending
        );
      }
      directions = ascending;
    } else {
      directions = sortCols.map(() => ascending);
    }

    const nRows = this._index.length;

    // Pre-fetch sort column arrays for direct indexed access
    const sortColArrays: unknown[][] = new Array(sortCols.length);
    for (let c = 0; c < sortCols.length; c++) {
      sortColArrays[c] = this._data.get(sortCols[c] as string) ?? [];
    }

    // Sort row indices instead of full row objects
    const indices = new Array<number>(nRows);
    for (let i = 0; i < nRows; i++) indices[i] = i;

    indices.sort((ai, bi) => {
      for (let c = 0; c < sortColArrays.length; c++) {
        const colArr = sortColArrays[c] as unknown[];
        const aVal = colArr[ai];
        const bVal = colArr[bi];

        // Missing values (null/undefined/NaN) always sort LAST, regardless of
        // ascending/descending (pandas na_position='last' default).
        const aMissing = isMissing(aVal);
        const bMissing = isMissing(bVal);
        if (aMissing || bMissing) {
          if (aMissing && bMissing) continue;
          return aMissing ? 1 : -1;
        }

        const cmp = compareValues(aVal, bVal);
        if (cmp !== 0) return directions[c] ? cmp : -cmp;
      }
      return 0;
    });

    // Build sorted data by gathering from original columns using sorted indices
    const sortedData: DataFrameData = newColumnMap();
    for (const col of this._columns) {
      const src = this._data.get(col) ?? [];
      const dst = new Array(nRows);
      for (let i = 0; i < nRows; i++) {
        dst[i] = src[indices[i] as number];
      }
      sortedData[col] = dst;
    }

    const sortedIndex = new Array<string | number>(nRows);
    for (let i = 0; i < nRows; i++) {
      sortedIndex[i] = this._index[indices[i] as number] as string | number;
    }

    return new DataFrame(sortedData, {
      columns: this._columns,
      index: sortedIndex,
      copy: false,
    });
  }

  /**
   * Group DataFrame by one or more columns.
   *
   * Returns a DataFrameGroupBy object for performing aggregations.
   *
   * By default groups are listed in the order their keys first appear and rows with
   * a null, undefined or NaN key form groups of their own. Unlike pandas, which
   * sorts the keys and drops missing keys by default, this is kept for backward
   * compatibility; pass `{ sort: true, dropna: true }` for the pandas behavior.
   *
   * @param by - Column name or array of column names to group by
   * @param options - Optional grouping settings
   * @param options.sort - List groups sorted by key (ascending, column by column,
   *   strings by UTF-16 code unit) instead of by first appearance (default: false)
   * @param options.dropna - Drop groups whose key has a null, undefined or NaN part
   *   (default: false)
   * @returns DataFrameGroupBy object for aggregation operations
   *
   * @example
   * ```ts
   * const df = new DataFrame({
   *   category: ['A', 'B', 'A', 'B'],
   *   value: [10, 20, 30, 40]
   * });
   * const grouped = df.groupBy('category');
   * grouped.sum();  // Sum values by category
   * df.groupBy('category', { sort: true, dropna: true }).sum();  // pandas ordering
   * ```
   */
  groupBy(by: string | string[], options: GroupByOptions = {}): DataFrameGroupBy {
    return new DataFrameGroupBy(this, by, options);
  }

  /**
   * Join with another DataFrame using SQL-style join.
   *
   * Uses a hash join, O(n + m) in the number of rows. Rows with a null or
   * undefined key never match; NaN keys match each other. The result has a fresh
   * 0..n-1 index. Non-key columns that exist on both sides are renamed with the
   * suffixes `_left` and `_right`.
   *
   * Row order: inner and left joins follow the left rows, right joins follow the
   * right rows, and outer joins list the left-driven rows first followed by the
   * unmatched right rows.
   *
   * @param other - DataFrame to join with
   * @param on - Column name to join on (must exist in both DataFrames)
   * @param how - Type of join operation
   *   - 'inner': Only rows with matching keys in both DataFrames
   *   - 'left': All rows from left, matched rows from right (nulls for non-matches)
   *   - 'right': All rows from right, matched rows from left (nulls for non-matches)
   *   - 'outer': All rows from both DataFrames (nulls for non-matches)
   * @returns New DataFrame with joined data
   *
   * @throws {InvalidParameterError} If join column doesn't exist in either DataFrame
   *
   * @example
   * ```ts
   * const customers = new DataFrame({
   *   id: [1, 2, 3],
   *   name: ['Alice', 'Bob', 'Charlie']
   * });
   * const orders = new DataFrame({
   *   id: [1, 1, 2, 4],
   *   product: ['Laptop', 'Mouse', 'Keyboard', 'Monitor']
   * });
   *
   * // Inner join - only customers with orders
   * const inner = customers.join(orders, 'id', 'inner');
   * // Result: Alice with 2 orders, Bob with 1 order
   *
   * // Left join - all customers, with/without orders
   * const left = customers.join(orders, 'id', 'left');
   * // Result: Alice, Bob, Charlie (Charlie has null for product)
   * ```
   *
   * @see {@link https://deepbox.dev/docs/dataframe-overview | Deepbox DataFrame}
   */
  join(
    other: DataFrame,
    on: string,
    how: "inner" | "left" | "right" | "outer" = "inner"
  ): DataFrame {
    if (!JOIN_KINDS.includes(how)) {
      throw new InvalidParameterError(
        'how must be one of "inner", "left", "right", or "outer"',
        "how",
        how
      );
    }
    // Validate join column exists in both DataFrames
    if (!this._data.has(on)) {
      throw new InvalidParameterError(`Join column '${on}' not found in left DataFrame`, "on", on);
    }
    if (!other._data.has(on)) {
      throw new InvalidParameterError(`Join column '${on}' not found in right DataFrame`, "on", on);
    }

    return this.hashJoin(other, on, on, how, ["_left", "_right"]);
  }

  /**
   * Hash join shared by join() and merge().
   *
   * Row order follows pandas: inner and left joins keep the left row order
   * (matches for one left row appear in right order), right joins keep the
   * right row order, outer joins list left-driven rows first and then the
   * unmatched right rows. Rows with a null/undefined key never match.
   *
   * @private
   */
  private hashJoin(
    other: DataFrame,
    leftOn: string,
    rightOn: string,
    how: "inner" | "left" | "right" | "outer",
    suffixes: readonly [string, string]
  ): DataFrame {
    const leftKeys = this._data.get(leftOn) ?? [];
    const rightKeys = other._data.get(rightOn) ?? [];
    const sameKey = leftOn === rightOn;

    const buildHash = (keys: readonly unknown[]): Map<string, number[]> => {
      const hash = new Map<string, number[]>();
      for (let i = 0; i < keys.length; i++) {
        const val = keys[i];
        if (val === null || val === undefined) continue;
        const key = createKey(val);
        const bucket = hash.get(key);
        if (bucket === undefined) hash.set(key, [i]);
        else bucket.push(i);
      }
      return hash;
    };

    // Row pairs of the result; -1 marks "no row on this side".
    const leftRows: number[] = [];
    const rightRows: number[] = [];

    if (how === "right") {
      const leftHash = buildHash(leftKeys);
      for (let r = 0; r < rightKeys.length; r++) {
        const val = rightKeys[r];
        const matches =
          val === null || val === undefined ? undefined : leftHash.get(createKey(val));
        if (matches === undefined) {
          leftRows.push(-1);
          rightRows.push(r);
        } else {
          for (const l of matches) {
            leftRows.push(l);
            rightRows.push(r);
          }
        }
      }
    } else {
      const rightHash = buildHash(rightKeys);
      const matchedRight = how === "outer" ? new Uint8Array(rightKeys.length) : null;
      for (let l = 0; l < leftKeys.length; l++) {
        const val = leftKeys[l];
        const matches =
          val === null || val === undefined ? undefined : rightHash.get(createKey(val));
        if (matches === undefined) {
          if (how === "left" || how === "outer") {
            leftRows.push(l);
            rightRows.push(-1);
          }
        } else {
          for (const r of matches) {
            leftRows.push(l);
            rightRows.push(r);
            if (matchedRight) matchedRight[r] = 1;
          }
        }
      }
      if (matchedRight) {
        for (let r = 0; r < rightKeys.length; r++) {
          if (matchedRight[r] === 0) {
            leftRows.push(-1);
            rightRows.push(r);
          }
        }
      }
    }

    // Output column names. A column present on both sides gets a suffix on each
    // side (the shared join key itself is emitted once, unsuffixed).
    const rightSource = other._columns.filter((col) => !(sameKey && col === rightOn));
    const leftColumnSet = new Set(this._columns);
    const overlapping = new Set(rightSource.filter((col) => leftColumnSet.has(col)));
    const usedNames = new Set<string>();
    const uniqueName = (base: string): string => {
      let name = base;
      for (let n = 1; usedNames.has(name); n++) name = `${base}_${n}`;
      usedNames.add(name);
      return name;
    };
    const leftNames = this._columns.map((col) =>
      uniqueName(overlapping.has(col) ? `${col}${suffixes[0]}` : col)
    );
    const rightNames = rightSource.map((col) =>
      uniqueName(overlapping.has(col) ? `${col}${suffixes[1]}` : col)
    );

    const nOut = leftRows.length;
    const resultData: DataFrameData = newColumnMap();

    for (let j = 0; j < this._columns.length; j++) {
      const col = this._columns[j] as string;
      const src = this._data.get(col) ?? [];
      // Unmatched right rows take the join key from the right frame.
      const fallback = sameKey && col === leftOn ? rightKeys : null;
      const out = new Array<unknown>(nOut);
      for (let k = 0; k < nOut; k++) {
        const l = leftRows[k] as number;
        const v = l >= 0 ? src[l] : fallback ? fallback[rightRows[k] as number] : null;
        out[k] = v === undefined ? null : v;
      }
      resultData[leftNames[j] as string] = out;
    }

    for (let j = 0; j < rightSource.length; j++) {
      const src = other._data.get(rightSource[j] as string) ?? [];
      const out = new Array<unknown>(nOut);
      for (let k = 0; k < nOut; k++) {
        const r = rightRows[k] as number;
        const v = r >= 0 ? src[r] : null;
        out[k] = v === undefined ? null : v;
      }
      resultData[rightNames[j] as string] = out;
    }

    return new DataFrame(resultData, { columns: [...leftNames, ...rightNames], copy: false });
  }

  /**
   * Merge with another DataFrame using SQL-style merge.
   *
   * More flexible than join() - supports different column names for join keys.
   * Uses a hash join, O(n + m) in the number of rows, with the same matching and
   * row order rules as {@link DataFrame.join}. When the key names differ, both key
   * columns are kept. Columns that exist on both sides (other than a shared key)
   * get the `suffixes` on the left and right copy respectively.
   *
   * @param other - DataFrame to merge with
   * @param options - Merge configuration
   *   - on: Column name to join on (must exist in both DataFrames)
   *   - leftOn: Column name in left DataFrame
   *   - rightOn: Column name in right DataFrame
   *   - left_on, right_on: Deprecated spellings of `leftOn` and `rightOn`. `leftOn` and
   *     `rightOn` win when both spellings are given.
   *   - how: Join type ('inner', 'left', 'right', 'outer')
   *   - suffixes: Suffix for duplicate column names ['_x', '_y']
   * @returns New DataFrame with merged data
   *
   * @throws {InvalidParameterError} If merge columns don't exist or conflicting options provided
   *
   * @example
   * ```ts
   * const employees = new DataFrame({
   *   emp_id: [1, 2, 3],
   *   name: ['Alice', 'Bob', 'Charlie']
   * });
   * const salaries = new DataFrame({
   *   employee_id: [1, 2, 4],
   *   salary: [50000, 60000, 55000]
   * });
   *
   * // Merge on different column names
   * const result = employees.merge(salaries, {
   *   leftOn: 'emp_id',
   *   rightOn: 'employee_id',
   *   how: 'left'
   * });
   * ```
   *
   * @see {@link https://deepbox.dev/docs/dataframe-overview | Deepbox DataFrame}
   */
  merge(
    other: DataFrame,
    options: {
      readonly on?: string;
      readonly leftOn?: string;
      readonly rightOn?: string;
      /** @deprecated Prefer `leftOn`. */
      readonly left_on?: string;
      /** @deprecated Prefer `rightOn`. */
      readonly right_on?: string;
      readonly how?: "inner" | "left" | "right" | "outer";
      readonly suffixes?: readonly [string, string];
    } = {}
  ): DataFrame {
    const how = options.how ?? "inner";
    if (!JOIN_KINDS.includes(how)) {
      throw new InvalidParameterError(
        'how must be one of "inner", "left", "right", or "outer"',
        "how",
        how
      );
    }

    if (options.suffixes !== undefined) {
      if (
        !Array.isArray(options.suffixes) ||
        options.suffixes.length !== 2 ||
        typeof options.suffixes[0] !== "string" ||
        typeof options.suffixes[1] !== "string"
      ) {
        throw new InvalidParameterError(
          "suffixes must be a tuple of two strings",
          "suffixes",
          options.suffixes
        );
      }
    }
    const suffixes: readonly [string, string] = options.suffixes ?? ["_x", "_y"];

    // Determine join columns. The camelCase keys win over the deprecated snake_case ones.
    const leftOnOption = options.leftOn ?? options.left_on;
    const rightOnOption = options.rightOn ?? options.right_on;
    let leftOn: string;
    let rightOn: string;

    if (options.on) {
      if (typeof options.on !== "string") {
        throw new InvalidParameterError("on must be a string", "on", options.on);
      }
      // Same column name in both DataFrames
      if (leftOnOption || rightOnOption) {
        throw new InvalidParameterError(
          'Cannot specify both "on" and "left_on"/"right_on" (or "leftOn"/"rightOn")'
        );
      }
      leftOn = options.on;
      rightOn = options.on;
    } else if (leftOnOption && rightOnOption) {
      if (typeof leftOnOption !== "string") {
        throw new InvalidParameterError("leftOn must be a string", "leftOn", leftOnOption);
      }
      if (typeof rightOnOption !== "string") {
        throw new InvalidParameterError("rightOn must be a string", "rightOn", rightOnOption);
      }
      // Different column names
      leftOn = leftOnOption;
      rightOn = rightOnOption;
    } else {
      throw new InvalidParameterError(
        'Must specify either "on" or both "left_on" and "right_on" (or "leftOn" and "rightOn")'
      );
    }

    // Validate columns exist
    if (!this._data.has(leftOn)) {
      throw new InvalidParameterError(
        `Column '${leftOn}' not found in left DataFrame`,
        "leftOn",
        leftOn
      );
    }
    if (!other._data.has(rightOn)) {
      throw new InvalidParameterError(
        `Column '${rightOn}' not found in right DataFrame`,
        "rightOn",
        rightOn
      );
    }

    return this.hashJoin(other, leftOn, rightOn, how, suffixes);
  }

  /**
   * Concatenate with another DataFrame.
   *
   * With `axis=0` the rows of `other` are appended. Without `join` both frames must have the
   * same set of columns (the column order of `this` is kept), as in Deepbox 1.0. With
   * `join: "outer"` the columns of both frames are kept (the columns only `other` has come last)
   * and cells a frame has no column for become null; with `join: "inner"` only the shared columns
   * are kept. The row labels are reset to 0..n-1 unless `ignoreIndex` is `false`, which keeps the
   * labels of both frames and throws if they collide, because labels must stay unique.
   *
   * With `axis=1` rows are aligned on their labels and columns that exist on both sides are
   * suffixed with `_left` and `_right`. The row labels are the union of both indexes (missing cells
   * become null), or only the shared labels with `join: "inner"`. `ignoreIndex: true` names the
   * resulting columns "0", "1", ... instead.
   *
   * @param other - DataFrame to concatenate
   * @param axis - Axis to concatenate along, or the options object.
   *               - 0 or "rows" or "index": Stack vertically (append rows)
   *               - 1 or "columns": Stack horizontally (append columns)
   * @param options - `join` and `ignoreIndex` (see {@link ConcatOptions})
   * @returns Concatenated DataFrame
   * @throws {DataValidationError} If the columns differ and no `join` is given, or labels collide
   *   with `ignoreIndex: false`
   *
   * @example
   * ```ts
   * const df1 = new DataFrame({ a: [1, 2], b: [3, 4] });
   * const df2 = new DataFrame({ a: [5, 6], c: [7, 8] });
   * df1.concat(df2, "columns");                         // Stack horizontally
   * df1.concat(df2, 0, { join: "outer" });              // columns a, b, c with nulls
   * df1.concat(df2, { join: "inner" });                 // only column a
   * ```
   */
  concat(other: DataFrame, axis: Axis | ConcatOptions = 0, options: ConcatOptions = {}): DataFrame {
    const opts: ConcatOptions = typeof axis === "object" ? { ...axis, ...options } : options;
    const ax = normalizeAxis(typeof axis === "object" ? 0 : axis, 2);
    const join = opts.join;
    if (join !== undefined && join !== "outer" && join !== "inner") {
      throw new InvalidParameterError('join must be "outer" or "inner"', "join", join);
    }
    if (opts.ignoreIndex !== undefined && typeof opts.ignoreIndex !== "boolean") {
      throw new InvalidParameterError(
        "ignoreIndex must be a boolean",
        "ignoreIndex",
        opts.ignoreIndex
      );
    }

    if (ax === 0) {
      let outColumns: string[];
      if (join === undefined) {
        // Columns must match
        for (const col of this._columns) {
          if (!other._columns.includes(col)) {
            throw new DataValidationError(
              `Cannot concat on axis=0: missing column '${col}' in other DataFrame`
            );
          }
        }
        for (const col of other._columns) {
          if (!this._columns.includes(col)) {
            throw new DataValidationError(
              `Cannot concat on axis=0: extra column '${col}' in other DataFrame`
            );
          }
        }
        outColumns = this._columns;
      } else if (join === "inner") {
        outColumns = this._columns.filter((col) => other._data.has(col));
      } else {
        outColumns = [...this._columns, ...other._columns.filter((col) => !this._data.has(col))];
      }
      const newData: DataFrameData = newColumnMap();
      const leftRows = this._index.length;
      const rightRows = other._index.length;

      // Copy data from both DataFrames for each column; a missing column becomes nulls
      for (const col of outColumns) {
        const left = this._data.get(col) ?? new Array<unknown>(leftRows).fill(null);
        const right = other._data.get(col) ?? new Array<unknown>(rightRows).fill(null);
        newData[col] = [...left, ...right];
      }

      let newIndex: (string | number)[];
      if (opts.ignoreIndex === false) {
        newIndex = [...this._index, ...other._index];
        const seen = new Set<string | number>();
        for (const label of newIndex) {
          if (seen.has(label)) {
            throw new DataValidationError(
              `Cannot concat with ignoreIndex=false: duplicate index label '${String(label)}'. ` +
                "Use ignoreIndex: true to renumber the rows"
            );
          }
          seen.add(label);
        }
      } else {
        // Reset index to sequential integers to avoid duplicate index errors
        newIndex = Array.from({ length: leftRows + rightRows }, (_, i) => i);
      }

      return new DataFrame(newData, {
        columns: outColumns,
        index: newIndex,
        copy: false,
      });
    } else {
      // Concatenate columns (stack horizontally) with index alignment
      // 1. Determine new index (union of indices, or the shared ones for an inner join)
      const newIndex: (string | number)[] = [];
      if (join === "inner") {
        for (const idx of this._index) if (other._indexPos.has(idx)) newIndex.push(idx);
      } else {
        newIndex.push(...this._index);
        const seenIndices = new Set(this._index);
        for (const idx of other._index) {
          if (!seenIndices.has(idx)) {
            newIndex.push(idx);
            seenIndices.add(idx);
          }
        }
      }

      // 2. Build new data
      const newData: DataFrameData = newColumnMap();
      const newColumns: string[] = [];

      // Helper to align data
      const alignColumn = (
        df: DataFrame,
        col: string,
        targetIndex: (string | number)[]
      ): unknown[] => {
        const sourceData = df._data.get(col);
        if (!sourceData) return [];

        // Use _indexPos for O(1) lookup
        const indexPos = df._indexPos;

        return targetIndex.map((label) => {
          const pos = indexPos.get(label);
          if (pos !== undefined) {
            return sourceData[pos];
          }
          return null;
        });
      };

      // Detect overlapping column names
      const rightColSet = new Set(other._columns);
      const overlapping = new Set<string>();
      for (const col of this._columns) {
        if (rightColSet.has(col)) {
          overlapping.add(col);
        }
      }

      const renumber = opts.ignoreIndex === true;
      // Copy all columns from this DataFrame, suffixing overlapping ones
      for (const col of this._columns) {
        const outputName = renumber
          ? String(newColumns.length)
          : overlapping.has(col)
            ? `${col}_left`
            : col;
        newData[outputName] = alignColumn(this, col, newIndex);
        newColumns.push(outputName);
      }

      // Add columns from other DataFrame, suffixing overlapping ones
      for (const col of other._columns) {
        const outputName = renumber
          ? String(newColumns.length)
          : overlapping.has(col)
            ? `${col}_right`
            : col;
        newData[outputName] = alignColumn(other, col, newIndex);
        newColumns.push(outputName);
      }

      return new DataFrame(newData, {
        columns: newColumns,
        index: newIndex,
      });
    }
  }

  /**
   * Fill missing values (null, undefined or NaN).
   *
   * The argument selects how:
   * - a plain value fills every missing cell with it;
   * - a plain object maps column names to fill values, like pandas' `fillna({ a: 0 })`.
   *   Columns that are not listed stay unchanged, and names that are not columns are ignored;
   * - `{ method: "ffill" | "bfill", limit? }` propagates the previous or next valid value
   *   (see {@link DataFrame.ffill} and {@link DataFrame.bfill}). `"pad"` and `"backfill"` are accepted
   *   as aliases.
   *
   * An object whose only keys are `method`, `limit` and `axis` is read as the method form; any other
   * plain object is read as a per-column map. (If the frame has a column called `method`, such an
   * object is a per-column map unless its `method` is a fill method name.) Dates, arrays and class instances are always plain fill
   * values. Before 1.5 a plain object was used as the fill value itself.
   *
   * @param value - Fill value, per-column object or `{ method, limit? }`
   * @returns New DataFrame with missing values filled
   * @throws {InvalidParameterError} If `method` or `limit` is invalid
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, null, 3], b: [null, 5, undefined] });
   * df.fillna(0);                          // a: [1, 0, 3], b: [0, 5, 0]
   * df.fillna({ a: -1 });                  // only column a is filled
   * df.fillna({ method: "ffill" });        // a: [1, 1, 3], b: [null, 5, 5]
   * df.fillna({ method: "bfill", limit: 1 });
   * ```
   */
  fillna(value: unknown): DataFrame {
    if (isPlainObject(value)) {
      const keys = Object.keys(value);
      // A column that is itself called "method" wins unless the value names a fill method.
      const isMethodForm =
        Object.hasOwn(value, "method") &&
        keys.every((k) => k === "method" || k === "limit" || k === "axis") &&
        (!this._data.has("method") || FILL_METHOD_NAMES.includes(value["method"] as string));
      if (isMethodForm) {
        const options = value as unknown as FillnaMethodOptions;
        const direction = checkFillMethod(options.method);
        const opts: { limit?: number; axis?: number | "index" | "rows" | "columns" } = {};
        if (options.limit !== undefined) opts.limit = options.limit;
        if (options.axis !== undefined) opts.axis = options.axis;
        return this.propagate(direction, opts);
      }
      const newData: DataFrameData = newColumnMap();
      for (const col of this._columns) {
        const colData = this._data.get(col) ?? [];
        if (Object.hasOwn(value, col)) {
          const fill = value[col];
          newData[col] = colData.map((v) => (isMissing(v) ? fill : v));
        } else {
          newData[col] = colData.slice();
        }
      }
      return new DataFrame(newData, { columns: this._columns, index: this._index, copy: false });
    }

    const newData: DataFrameData = newColumnMap();

    // Replace null/undefined/NaN in each column
    for (const col of this._columns) {
      const colData = this._data.get(col);
      if (colData) {
        newData[col] = colData.map((v) => (isMissing(v) ? value : v));
      }
    }

    return new DataFrame(newData, {
      columns: this._columns,
      index: this._index,
      copy: false,
    });
  }

  /**
   * Fill missing values with the previous valid value (forward fill).
   *
   * Leading missing values have nothing to copy and stay missing. With `limit`, at most that many
   * consecutive missing values are filled in each gap, as in pandas.
   *
   * @param options - `limit`: longest run to fill; `axis`: 0 fills down columns (default), 1 across rows
   * @returns New DataFrame
   * @throws {InvalidParameterError} If `limit` is not a positive integer
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, null, null, 4] });
   * df.ffill();              // a: [1, 1, 1, 4]
   * df.ffill({ limit: 1 });  // a: [1, 1, null, 4]
   * ```
   */
  ffill(options: FillOptions = {}): DataFrame {
    return this.propagate("ffill", options);
  }

  /**
   * Fill missing values with the next valid value (backward fill).
   *
   * Trailing missing values have nothing to copy and stay missing. With `limit`, at most that many
   * consecutive missing values are filled in each gap, counted back from the next valid value.
   *
   * @param options - `limit`: longest run to fill; `axis`: 0 fills up columns (default), 1 across rows
   * @returns New DataFrame
   * @throws {InvalidParameterError} If `limit` is not a positive integer
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [null, null, 3, null] });
   * df.bfill();  // a: [3, 3, 3, null]
   * ```
   */
  bfill(options: FillOptions = {}): DataFrame {
    return this.propagate("bfill", options);
  }

  /** Shared implementation of ffill, bfill and fillna with a method. */
  private propagate(direction: FillMethod, options: FillOptions): DataFrame {
    const limit = checkFillLimit(options.limit);
    const axis = normalizeAxis(options.axis ?? 0, 2);
    const newData: DataFrameData = newColumnMap();
    if (axis === 0) {
      for (const col of this._columns) {
        newData[col] = propagateFill(this._data.get(col) ?? [], direction, limit, isMissing);
      }
    } else {
      const arrays = this._columns.map((c) => this._data.get(c) ?? []);
      const filled: unknown[][] = arrays.map(() => new Array<unknown>(this._index.length));
      for (let i = 0; i < this._index.length; i++) {
        const row = propagateFill(
          arrays.map((arr) => arr[i]),
          direction,
          limit,
          isMissing
        );
        for (let c = 0; c < row.length; c++) (filled[c] as unknown[])[i] = row[c];
      }
      this._columns.forEach((col, c) => {
        newData[col] = filled[c] as unknown[];
      });
    }
    return new DataFrame(newData, { columns: this._columns, index: this._index, copy: false });
  }

  /**
   * Drop rows that contain missing values (null, undefined or NaN).
   *
   * @param options - Optional settings
   * @param options.how - `"any"` (default) drops a row if any checked value is
   *   missing, `"all"` only if every checked value is missing
   * @param options.subset - Columns to check (default: all columns)
   * @param options.thresh - Keep rows that have at least this many non-missing
   *   values among the checked columns. Overrides `how`.
   * @returns New DataFrame with the matching rows removed
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, null, 3], b: [4, 5, 6] });
   * df.dropna();  // Only keeps rows 0 and 2
   * df.dropna({ subset: ['b'] });  // Keeps every row, b has no gaps
   * ```
   */
  dropna(
    options: {
      readonly how?: "any" | "all";
      readonly subset?: readonly string[];
      readonly thresh?: number;
    } = {}
  ): DataFrame {
    const how = options.how ?? "any";
    if (how !== "any" && how !== "all") {
      throw new InvalidParameterError('how must be "any" or "all"', "how", how);
    }
    const checkCols = options.subset ?? this._columns;
    const checkArrays: unknown[][] = [];
    for (const col of checkCols) {
      const arr = this._data.get(col);
      if (arr === undefined) {
        throw new InvalidParameterError(`Column '${col}' not found in DataFrame`, "subset", col);
      }
      checkArrays.push(arr);
    }
    const thresh = options.thresh;
    if (thresh !== undefined && (!Number.isInteger(thresh) || thresh < 0)) {
      throw new InvalidParameterError("thresh must be a non-negative integer", "thresh", thresh);
    }

    const nRows = this._index.length;
    const keep: number[] = [];
    for (let i = 0; i < nRows; i++) {
      let present = 0;
      for (const arr of checkArrays) {
        if (!isMissing(arr[i])) present++;
      }
      const total = checkArrays.length;
      let keepRow: boolean;
      if (thresh !== undefined) keepRow = present >= thresh;
      else if (how === "any") keepRow = present === total;
      else keepRow = total === 0 || present > 0;
      if (keepRow) keep.push(i);
    }

    const newData: DataFrameData = newColumnMap();
    for (const col of this._columns) {
      const src = this._data.get(col) ?? [];
      const dst = new Array<unknown>(keep.length);
      for (let k = 0; k < keep.length; k++) dst[k] = src[keep[k] as number];
      newData[col] = dst;
    }
    const newIndex = keep.map((i) => this._index[i] as string | number);

    return new DataFrame(newData, {
      columns: this._columns,
      index: newIndex,
      copy: false,
    });
  }

  /**
   * Generate descriptive statistics.
   *
   * Computes count, mean, std, min, 25%, 50%, 75%, max for numeric columns.
   * NaN, null and non-numeric values are ignored; infinite values are kept.
   * Columns without any numeric value are left out.
   *
   * @returns DataFrame with statistics
   */
  describe(): DataFrame {
    const stats: DataFrameData = newColumnMap();
    const metrics = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"];

    // Handle empty DataFrame - return DataFrame with metrics as index and no data columns
    if (this._columns.length === 0 || this._index.length === 0) {
      return new DataFrame({}, { columns: [], index: metrics });
    }

    for (const col of this._columns) {
      const colData = this._data.get(col);
      if (!colData) continue;

      // Collect valid numbers into a typed buffer: the comparator-free typed
      // sort and plain loops are ~5x faster than filter/spread/sort/reduce.
      let validCount = 0;
      for (const v of colData) {
        if (isOrderedNumber(v)) validCount++;
      }
      if (validCount === 0) continue;
      const numericData = new Float64Array(validCount);
      {
        let w = 0;
        for (const v of colData) {
          if (isOrderedNumber(v)) numericData[w++] = v;
        }
      }

      const sorted = numericData.slice();
      sorted.sort();
      const mean = compensatedSum(numericData) / validCount;
      let variance: number;
      let std: number;

      if (validCount > 1) {
        variance = Math.max(0, sumSquaredDeviations(numericData)) / (validCount - 1);
        std = Math.sqrt(variance);
      } else {
        variance = NaN;
        std = NaN;
      }

      const minVal = sorted[0];
      const maxVal = sorted[sorted.length - 1];
      if (minVal === undefined || maxVal === undefined) {
        throw new DataValidationError(`Unable to compute min/max for column '${col}'`);
      }

      stats[col] = [
        numericData.length,
        mean,
        std,
        minVal,
        quantileSorted(sorted, 0.25),
        quantileSorted(sorted, 0.5),
        quantileSorted(sorted, 0.75),
        maxVal,
      ];
    }

    // If no numeric columns were found, return DataFrame with metrics as index and no data columns
    if (Object.keys(stats).length === 0) {
      return new DataFrame({}, { columns: [], index: metrics });
    }

    return new DataFrame(stats, { index: metrics });
  }

  /**
   * Compute the pairwise correlation matrix of the numeric columns.
   *
   * Each cell uses the rows where both columns hold a finite number (pairwise complete
   * observations). Three coefficients are available, as in pandas:
   * - `"pearson"` (default): linear correlation;
   * - `"spearman"`: Pearson correlation of the average ranks, ranked within each pair;
   * - `"kendall"`: Kendall's tau-b, which corrects for ties.
   *
   * A cell is NaN when fewer than `minPeriods` complete pairs exist (default 1), when fewer than
   * two pairs exist, or when one of the two samples is constant. The diagonal is exactly 1 for
   * every column that has a defined correlation with itself, and NaN otherwise. As in pandas,
   * the Kendall diagonal is 1 for every column with at least `minPeriods` values, even a
   * constant one.
   *
   * @param method - Coefficient name, or an options object `{ method?, minPeriods? }`
   * @param minPeriods - Minimum number of complete pairs per cell (default: 1)
   * @returns Square DataFrame labelled by the numeric column names
   * @throws {InvalidParameterError} If `method` is unknown or `minPeriods` is not a non-negative integer
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, 2, 3, 4], b: [1, 4, 9, 16] });
   * df.corr();                     // Pearson, about 0.984 off the diagonal
   * df.corr("spearman");           // 1, the relation is monotonic
   * df.corr({ method: "kendall", minPeriods: 3 });
   * ```
   */
  corr(method: CorrelationMethod | CorrOptions = {}, minPeriods?: number): DataFrame {
    const options: CorrOptions = typeof method === "string" ? { method } : method;
    const kind: CorrelationMethod = options.method ?? "pearson";
    if (kind !== "pearson" && kind !== "spearman" && kind !== "kendall") {
      throw new InvalidParameterError(
        'method must be "pearson", "spearman" or "kendall"',
        "method",
        kind
      );
    }
    const minObs = options.minPeriods ?? minPeriods ?? 1;
    if (typeof minObs !== "number" || !Number.isInteger(minObs) || minObs < 0) {
      throw new InvalidParameterError(
        "minPeriods must be a non-negative integer",
        "minPeriods",
        minObs
      );
    }
    // Fewer than two pairs never define a correlation.
    const need = Math.max(minObs, 2);

    const numericCols: string[] = [];
    for (const col of this._columns) {
      const colData = this._data.get(col);
      if (colData?.some(isValidNumber)) numericCols.push(col);
    }

    const nRows = this._index.length;
    const kCols = numericCols.length;
    const columns: Float64Array[] = [];
    const valid: Uint8Array[] = [];
    const counts: number[] = [];
    let allValid = true;
    for (const name of numericCols) {
      const data = this._data.get(name) ?? [];
      const values = new Float64Array(nRows);
      const mask = new Uint8Array(nRows);
      let count = 0;
      for (let i = 0; i < nRows; i++) {
        const v = data[i];
        if (isValidNumber(v)) {
          values[i] = v;
          mask[i] = 1;
          count++;
        }
      }
      if (count < nRows) allValid = false;
      columns.push(values);
      valid.push(mask);
      counts.push(count);
    }

    const isConstant = (values: ArrayLike<number>): boolean => {
      for (let i = 1; i < values.length; i++) if (values[i] !== values[0]) return false;
      return true;
    };

    const out: number[][] = [];
    for (let a = 0; a < kCols; a++) out.push(new Array<number>(kCols).fill(Number.NaN));

    // With no gaps every pair uses all rows: rank-based methods rank each column once, and
    // Pearson centers each column once and reuses its norm.
    const ranked =
      allValid && kind === "spearman" ? columns.map((col) => correlationRanks(col)) : columns;
    const pairKind: CorrelationMethod = allValid && kind === "spearman" ? "pearson" : kind;
    const centered: Float64Array[] = [];
    const norms: number[] = [];
    if (allValid && pairKind === "pearson" && nRows >= need) {
      for (const col of ranked) {
        const shifted = new Float64Array(nRows);
        let mean = 0;
        for (let i = 0; i < nRows; i++) mean += col[i] as number;
        mean /= nRows;
        let ss = 0;
        for (let i = 0; i < nRows; i++) {
          const d = (col[i] as number) - mean;
          shifted[i] = d;
          ss += d * d;
        }
        centered.push(shifted);
        norms.push(Math.sqrt(ss));
      }
    }

    for (let a = 0; a < kCols; a++) {
      const colA = columns[a] as Float64Array;
      const maskA = valid[a] as Uint8Array;
      const countA = counts[a] as number;
      // pandas reports 1 for every Kendall diagonal cell that reaches minPeriods, even for a
      // constant column or a single value.
      if (kind === "kendall" && countA >= minObs) {
        (out[a] as number[])[a] = 1;
      } else if (countA >= need) {
        let first = Number.NaN;
        let constant = true;
        for (let i = 0; i < nRows; i++) {
          if (maskA[i] === 0) continue;
          if (Number.isNaN(first)) first = colA[i] as number;
          else if (colA[i] !== first) {
            constant = false;
            break;
          }
        }
        (out[a] as number[])[a] = constant ? Number.NaN : 1;
      }
      for (let b = a + 1; b < kCols; b++) {
        const maskB = valid[b] as Uint8Array;
        let r = Number.NaN;
        if (allValid) {
          if (nRows >= need) {
            if (pairKind === "pearson") {
              const normA = norms[a] as number;
              const normB = norms[b] as number;
              if (normA !== 0 && normB !== 0) {
                const ca = centered[a] as Float64Array;
                const cb = centered[b] as Float64Array;
                let dot = 0;
                for (let i = 0; i < nRows; i++) dot += (ca[i] as number) * (cb[i] as number);
                r = clampCorrelation(dot / (normA * normB));
              }
            } else {
              r = correlate(pairKind, ranked[a] as Float64Array, ranked[b] as Float64Array);
            }
          }
        } else {
          const colB = columns[b] as Float64Array;
          const xs: number[] = [];
          const ys: number[] = [];
          for (let i = 0; i < nRows; i++) {
            if (maskA[i] === 1 && maskB[i] === 1) {
              xs.push(colA[i] as number);
              ys.push(colB[i] as number);
            }
          }
          if (xs.length >= need && !isConstant(xs) && !isConstant(ys)) r = correlate(kind, xs, ys);
        }
        (out[a] as number[])[b] = r;
        (out[b] as number[])[a] = r;
      }
    }

    const corrMatrix: DataFrameData = newColumnMap();
    for (let a = 0; a < kCols; a++) corrMatrix[numericCols[a] as string] = out[a] as number[];
    return new DataFrame(corrMatrix, { index: numericCols, columns: numericCols });
  }

  /**
   * Compute covariance matrix.
   *
   * Uses pairwise complete observations.
   *
   * @returns DataFrame containing pairwise covariances
   */
  cov(): DataFrame {
    const numericCols: string[] = [];

    // Identify numeric columns
    for (const col of this._columns) {
      const colData = this._data.get(col);
      if (!colData) continue;
      if (colData.some(isValidNumber)) {
        numericCols.push(col);
      }
    }

    const covMatrix: DataFrameData = newColumnMap();

    for (const col1 of numericCols) {
      covMatrix[col1] = [];
      const data1 = this._data.get(col1);

      for (const col2 of numericCols) {
        const data2 = this._data.get(col2);

        if (!data1 || !data2) {
          covMatrix[col1]?.push(NaN);
          continue;
        }

        // Collect pairwise valid observations
        const valid1: number[] = [];
        const valid2: number[] = [];

        for (let i = 0; i < this._index.length; i++) {
          const v1 = data1[i];
          const v2 = data2[i];

          if (isValidNumber(v1) && isValidNumber(v2)) {
            valid1.push(v1);
            valid2.push(v2);
          }
        }

        if (valid1.length < 2) {
          covMatrix[col1]?.push(NaN);
          continue;
        }

        // Compute covariance
        const mean1 = valid1.reduce((a, b) => a + b, 0) / valid1.length;
        const mean2 = valid2.reduce((a, b) => a + b, 0) / valid2.length;

        let cov = 0;
        for (let k = 0; k < valid1.length; k++) {
          const val1 = valid1[k];
          const val2 = valid2[k];
          // val1 and val2 are guaranteed to be numbers from valid1/valid2 construction
          // However, we check for undefined to satisfy strict null checks (noUncheckedIndexedAccess)
          if (val1 === undefined || val2 === undefined) continue;

          cov += (val1 - mean1) * (val2 - mean2);
        }
        cov /= valid1.length - 1;

        covMatrix[col1]?.push(cov);
      }
    }

    return new DataFrame(covMatrix, {
      index: numericCols,
      columns: numericCols,
    });
  }

  /**
   * Apply a function along an axis of the DataFrame.
   *
   * When `axis=1`, the provided Series is indexed by column names.
   *
   * @param fn - Function to apply to each Series
   * @param axis - Axis to apply along (0=columns, 1=rows)
   * @returns New DataFrame with function applied
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, 2, 3], b: [4, 5, 6] });
   * // Apply function to each column
   * df.apply(series => series.map(x => Number(x) * 2), 0);
   * ```
   */
  apply<U = unknown>(fn: (series: Series<unknown>) => Series<U>, axis: Axis = 0): DataFrame {
    const ax = normalizeAxis(axis, 2);
    if (ax === 0) {
      // Apply function to each column
      const newData: DataFrameData = newColumnMap();

      for (const col of this._columns) {
        const series = this.get(col);
        const result = fn(series);
        if (!(result instanceof Series)) {
          throw new DataValidationError("Function must return a Series when axis=0");
        }
        newData[col] = [...result.data];
      }

      return new DataFrame(newData, {
        columns: this._columns,
        index: this._index,
        copy: false,
      });
    } else {
      // Apply function to each row.
      const results: Series<U>[] = [];
      const columnLabelMap = new Map<string, string | number>();
      const newColumns: string[] = [];

      // First pass: Apply function and collect results + columns
      for (let i = 0; i < this._index.length; i++) {
        const rowValues: unknown[] = [];
        for (const col of this._columns) {
          rowValues.push(this._data.get(col)?.[i]);
        }

        const rowSeries = new Series(rowValues, {
          name: "row",
          index: this._columns,
          copy: false,
        });
        const result = fn(rowSeries);

        if (!(result instanceof Series)) {
          throw new DataValidationError("Function must return a Series when axis=1");
        }

        results.push(result);

        for (const label of result.index) {
          const columnName = String(label);
          const existing = columnLabelMap.get(columnName);
          if (existing !== undefined && existing !== label) {
            throw new DataValidationError(
              `Column label '${columnName}' is ambiguous between '${String(
                existing
              )}' and '${String(label)}'`
            );
          }
          if (!columnLabelMap.has(columnName)) {
            newColumns.push(columnName);
            columnLabelMap.set(columnName, label);
          }
        }
      }

      const newData: DataFrameData = newColumnMap();

      for (const col of newColumns) {
        newData[col] = [];
      }

      // Second pass: Populate new data
      for (const result of results) {
        for (const col of newColumns) {
          const label = columnLabelMap.get(col);
          if (label === undefined) {
            throw new DataValidationError(`Missing label mapping for column '${col}'`);
          }
          // Use get() to handle missing values (returns undefined/null)
          // We convert undefined to null for consistency
          const val = result.get(label);
          newData[col]?.push(val === undefined ? null : val);
        }
      }

      return new DataFrame(newData, {
        columns: newColumns,
        index: this._index,
      });
    }
  }

  /**
   * Convert DataFrame to a 2D Tensor.
   *
   * All columns must contain numeric data. Null and undefined become NaN.
   * The tensor is float32 unless `options.dtype` says otherwise; float32 only
   * keeps 24 bits of precision, so integers above 2^24 and values that need more
   * than about 7 significant digits are rounded. Pass `{ dtype: "float64" }` to
   * keep the stored doubles exactly.
   *
   * @param options - Optional settings
   * @param options.dtype - Element type of the tensor (default: "float32")
   * @returns 2D Tensor with shape [rows, columns]
   * @throws {DataValidationError} If data is non-numeric
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, 2, 3], b: [4, 5, 6] });
   * const t = df.toTensor();  // 2D tensor [[1,4], [2,5], [3,6]]
   * const t64 = df.toTensor({ dtype: "float64" });
   * ```
   */
  toTensor(options: { readonly dtype?: "float32" | "float64" } = {}): Tensor {
    const dtype = options.dtype ?? "float32";
    if (dtype !== "float32" && dtype !== "float64") {
      throw new InvalidParameterError('dtype must be "float32" or "float64"', "dtype", dtype);
    }
    const rows = this._index.length;
    const cols = this._columns.length;

    // Row-major flattening straight from the column arrays.
    const flat = new Array<number>(rows * cols);
    for (let c = 0; c < cols; c++) {
      const colData = this._data.get(this._columns[c] as string) ?? [];
      for (let r = 0; r < rows; r++) {
        const val = colData[r];
        if (typeof val === "number") {
          flat[r * cols + c] = val;
        } else if (val === null || val === undefined) {
          flat[r * cols + c] = NaN;
        } else {
          throw new DataValidationError(
            `Non-numeric value found: ${String(val)}. All data must be numeric (or null/undefined) for tensor conversion.`
          );
        }
      }
    }

    return reshape(tensor(flat, { dtype }), [rows, cols]);
  }

  /**
   * Convert DataFrame to a 2D JavaScript array.
   *
   * Each inner array represents a row.
   *
   * @returns 2D array of values
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, 2], b: [3, 4] });
   * df.toArray();  // [[1, 3], [2, 4]]
   * ```
   */
  toArray(): unknown[][] {
    const result: unknown[][] = [];

    // Build each row
    for (let i = 0; i < this._index.length; i++) {
      const row: unknown[] = [];
      for (const col of this._columns) {
        const colData = this._data.get(col);
        row.push(colData ? colData[i] : undefined);
      }
      result.push(row);
    }

    return result;
  }

  /**
   * Parse a CSV string into a DataFrame, inferring cell types and handling quoted fields.
   *
   * Cell types are inferred per cell: numbers (including `Infinity` and `NaN`),
   * `true`/`false`, and strings. Empty cells and the words `null` and `undefined`
   * become null. Strings with a leading zero such as `007` stay strings. Header
   * cells that are empty are named `Unnamed: <position>`. Quoted fields may
   * contain delimiters, line breaks and doubled quote characters.
   *
   * Time complexity: O(n) where n is number of characters.
   *
   * @param csvString - CSV text
   * @param options - Parsing options
   * @param options.delimiter - Single-character field separator (default: ",")
   * @param options.quoteChar - Single-character quote (default: '"')
   * @param options.hasHeader - Whether the first row holds column names (default: true).
   *   Without a header the columns are named col0, col1, ...
   * @param options.skipRows - Number of leading non-blank rows to skip (default: 0)
   * @throws {InvalidParameterError} If an option is invalid
   * @throws {DataValidationError} If the CSV is malformed
   */
  static fromCsvString(
    csvString: string,
    options: {
      readonly delimiter?: string;
      readonly quoteChar?: string;
      readonly hasHeader?: boolean;
      readonly skipRows?: number;
    } = {}
  ): DataFrame {
    const delimiter = options.delimiter ?? ",";
    const quoteChar = options.quoteChar ?? '"';
    const hasHeader = options.hasHeader ?? true;
    const skipRows = options.skipRows ?? 0;
    assertCsvCharacters(delimiter, quoteChar);
    if (!Number.isInteger(skipRows) || skipRows < 0) {
      throw new InvalidParameterError(
        "skipRows must be a non-negative integer",
        "skipRows",
        skipRows
      );
    }

    // Strip a leading UTF-8 BOM (common in Excel-exported CSVs); otherwise the
    // first column name becomes "﻿<name>" and lookups fail.
    if (csvString.charCodeAt(0) === 0xfeff) {
      csvString = csvString.slice(1);
    }

    const rows: string[][] = [];
    let fields: string[] = [];
    let currentField = "";
    let inQuotes = false;
    // True once the current row contained a quote character, so a row made of a
    // single quoted empty field ("") is data, not a blank line.
    let rowHadQuote = false;
    let rowCount = 0;

    // A row is a blank line (skippable) only when it is a single empty field.
    // A multi-field all-empty row like ",," is a real all-null record.
    const isBlankLine = (fs: string[]): boolean =>
      !rowHadQuote && fs.length === 1 && (fs[0] ?? "").trim() === "";

    // Parse character by character to handle newlines in quoted fields
    for (let i = 0; i < csvString.length; i++) {
      const char = csvString[i];
      const nextChar = csvString[i + 1];

      if (char === quoteChar) {
        rowHadQuote = true;
        if (inQuotes && nextChar === quoteChar) {
          // Escaped quote
          currentField += quoteChar;
          i++;
        } else {
          // Toggle quote state
          inQuotes = !inQuotes;
        }
      } else if (char === delimiter && !inQuotes) {
        // Field separator
        fields.push(currentField);
        currentField = "";
      } else if ((char === "\n" || char === "\r") && !inQuotes) {
        // End of row (but not if inside quotes)
        if (char === "\r" && nextChar === "\n") {
          i++; // Skip \n in \r\n
        }
        fields.push(currentField);
        currentField = "";

        // Skip only truly blank lines and rows before skipRows; keep all-empty
        // multi-field rows (",," gives an all-null record, like pandas).
        if (!isBlankLine(fields)) {
          if (rowCount >= skipRows) {
            rows.push(fields);
          }
          rowCount++;
        }
        fields = [];
        rowHadQuote = false;
      } else {
        // Regular character (including newlines inside quotes)
        currentField += char;
      }
    }

    if (inQuotes) {
      throw new DataValidationError("CSV contains an unmatched quote");
    }

    // Handle last row if no trailing newline
    if (currentField !== "" || fields.length > 0 || rowHadQuote) {
      fields.push(currentField);
      if (!isBlankLine(fields) && rowCount >= skipRows) {
        rows.push(fields);
      }
    }

    if (rows.length === 0) {
      throw new DataValidationError("CSV contains no data rows");
    }

    let columns: string[];
    let dataRows: string[][];

    if (hasHeader) {
      const firstRow = rows[0];
      if (!firstRow) throw new DataValidationError("CSV has no header row");
      columns = firstRow.map((name, i) => (name === "" ? `Unnamed: ${i}` : name));
      ensureUniqueLabels(columns, "column name");
      dataRows = rows.slice(1);
    } else {
      const numCols = rows[0]?.length ?? 0;
      columns = Array.from({ length: numCols }, (_, i) => `col${i}`);
      dataRows = rows;
    }

    for (let i = 0; i < dataRows.length; i++) {
      const row = dataRows[i];
      if (row && row.length !== columns.length) {
        throw new DataValidationError(
          `Row ${i + (hasHeader ? 2 : 1)} has ${row.length} fields, expected ${columns.length}`
        );
      }
    }

    const data: DataFrameData = newColumnMap();
    for (let colIdx = 0; colIdx < columns.length; colIdx++) {
      const colName = columns[colIdx] as string;
      const colData: unknown[] = [];

      for (const row of dataRows) {
        const value = row[colIdx];
        const trimmed = value === undefined ? "" : value.trim();
        if (trimmed === "" || trimmed === "null" || trimmed === "undefined") {
          // Empty or whitespace-only cells are missing (null), NOT numeric 0.
          colData.push(null);
        } else if (trimmed === "NaN" || trimmed === "nan") {
          colData.push(NaN);
        } else if (
          !Number.isNaN(Number(trimmed)) &&
          // Preserve leading-zero strings like "007"/"01" (but keep "0", "0.5").
          (trimmed === "0" || !trimmed.startsWith("0") || trimmed.startsWith("0."))
        ) {
          colData.push(Number(trimmed));
        } else if (trimmed === "true" || trimmed === "false") {
          colData.push(trimmed === "true");
        } else {
          colData.push(value);
        }
      }

      data[colName] = colData;
    }

    return new DataFrame(data, { columns, copy: false });
  }

  /**
   * Read CSV file - environment-aware (Node.js fs or browser fetch).
   * Time complexity: O(n) for file read + O(m) for parsing.
   */
  static async readCsv(
    path: string,
    options: {
      readonly delimiter?: string;
      readonly quoteChar?: string;
      readonly hasHeader?: boolean;
      readonly skipRows?: number;
    } = {}
  ): Promise<DataFrame> {
    let csvString: string;

    if (typeof process !== "undefined" && process.versions?.node) {
      try {
        const fs = await import("node:fs/promises");
        csvString = await fs.readFile(path, "utf-8");
      } catch (error) {
        throw new DataValidationError(
          `Failed to read CSV file: ${error instanceof Error ? error.message : String(error)}`
        );
      }
    } else if (typeof fetch !== "undefined") {
      try {
        const response = await fetch(path);
        if (!response.ok) {
          throw new DataValidationError(`HTTP ${response.status}: ${response.statusText}`);
        }
        csvString = await response.text();
      } catch (error) {
        throw new DataValidationError(
          `Failed to fetch CSV: ${error instanceof Error ? error.message : String(error)}`
        );
      }
    } else {
      throw new DataValidationError("Environment not supported");
    }

    return DataFrame.fromCsvString(csvString, options);
  }

  /**
   * Convert DataFrame to CSV string with proper quoting and escaping.
   *
   * Null and undefined are written as empty cells, dates as ISO 8601 strings,
   * arrays and plain objects as JSON. Lines are separated by `\n` with no
   * trailing newline.
   *
   * Time complexity: O(n × m) where n is rows, m is columns.
   *
   * @param options - Output options
   * @param options.delimiter - Single-character field separator (default: ",")
   * @param options.quoteChar - Single-character quote (default: '"')
   * @param options.includeIndex - Write the row labels as a leading "index" column (default: false)
   * @param options.header - Write a header row (default: true)
   */
  toCsvString(
    options: {
      readonly delimiter?: string;
      readonly quoteChar?: string;
      readonly includeIndex?: boolean;
      readonly header?: boolean;
    } = {}
  ): string {
    const delimiter = options.delimiter ?? ",";
    const quoteChar = options.quoteChar ?? '"';
    const includeIndex = options.includeIndex ?? false;
    const header = options.header ?? true;
    assertCsvCharacters(delimiter, quoteChar);

    const lines: string[] = [];

    const stringify = (value: unknown): string => {
      if (value === null || value === undefined) return "";
      if (value instanceof Date) {
        return Number.isNaN(value.getTime()) ? "" : value.toISOString();
      }
      if (typeof value === "object") return JSON.stringify(value);
      return String(value);
    };

    const escapeField = (value: unknown): string => {
      const str = stringify(value);
      if (
        str.includes(delimiter) ||
        str.includes(quoteChar) ||
        str.includes("\n") ||
        str.includes("\r")
      ) {
        return quoteChar + str.split(quoteChar).join(quoteChar + quoteChar) + quoteChar;
      }
      return str;
    };

    const nCols = this._columns.length + (includeIndex ? 1 : 0);
    // A line holding a single empty cell would read back as a blank line, so it is quoted.
    const finishLine = (fields: string[]): string => {
      if (nCols === 1 && fields[0] === "") return quoteChar + quoteChar;
      return fields.join(delimiter);
    };

    if (header) {
      const headerFields = includeIndex ? ["index", ...this._columns] : [...this._columns];
      lines.push(finishLine(headerFields.map(escapeField)));
    }

    const columnArrays = this._columns.map((col) => this._data.get(col) ?? []);
    for (let i = 0; i < this._index.length; i++) {
      const rowFields: string[] = [];

      if (includeIndex) {
        rowFields.push(escapeField(this._index[i]));
      }

      for (const colData of columnArrays) {
        rowFields.push(escapeField(colData[i]));
      }

      lines.push(finishLine(rowFields));
    }

    return lines.join("\n");
  }

  /**
   * Write DataFrame to CSV file - environment-aware.
   * Time complexity: O(n × m) for generation + O(k) for write.
   */
  async toCsv(
    path: string,
    options: {
      readonly delimiter?: string;
      readonly quoteChar?: string;
      readonly includeIndex?: boolean;
      readonly header?: boolean;
    } = {}
  ): Promise<void> {
    const csvString = this.toCsvString(options);

    if (typeof process !== "undefined" && process.versions?.node) {
      try {
        const fs = await import("node:fs/promises");
        await fs.writeFile(path, csvString, "utf-8");
      } catch (error) {
        throw new DataValidationError(
          `Failed to write CSV file: ${error instanceof Error ? error.message : String(error)}`
        );
      }
    } else if (typeof document !== "undefined" && typeof URL !== "undefined") {
      const blob = new Blob([csvString], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = path;
      link.style.display = "none";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } else {
      throw new DataValidationError("Environment not supported");
    }
  }

  /**
   * Serialize DataFrame to a JSON string with `columns`, `index` and `data` fields.
   *
   * JSON has no NaN or Infinity, so those numbers (and undefined) are written as
   * null and read back as null; dates are written as ISO strings. A `bigint` inside
   * the safe integer range is written as a number, a larger one as a decimal string.
   *
   * Time complexity: O(n × m).
   */
  toJsonString(): string {
    return JSON.stringify(
      {
        columns: this._columns,
        index: this._index,
        data: Object.fromEntries(this._data),
      },
      (_key, value: unknown) => {
        if (typeof value !== "bigint") return value;
        const asNumber = Number(value);
        return Number.isSafeInteger(asNumber) ? asNumber : value.toString();
      },
      2
    );
  }

  /**
   * Create DataFrame from JSON string.
   * Time complexity: O(n × m).
   */
  static fromJsonString(jsonStr: string): DataFrame {
    let parsed: unknown;
    try {
      parsed = JSON.parse(jsonStr);
    } catch (error) {
      throw new DataValidationError(
        `Failed to parse JSON: ${error instanceof Error ? error.message : String(error)}`
      );
    }

    if (!isRecord(parsed)) {
      throw new DataValidationError("Invalid JSON: expected object (not array)");
    }

    const obj = parsed;

    if (!isStringArray(obj["columns"])) {
      throw new DataValidationError(
        'Invalid JSON: missing or invalid "columns" field (expected array)'
      );
    }

    if (!isIndexLabelArray(obj["index"])) {
      throw new DataValidationError(
        'Invalid JSON: missing or invalid "index" field (expected array)'
      );
    }

    if (!isRecord(obj["data"])) {
      throw new DataValidationError(
        'Invalid JSON: missing or invalid "data" field (expected object)'
      );
    }

    const columns = obj["columns"];
    const index = obj["index"];
    const rawData = obj["data"];

    ensureUniqueLabels(columns, "column name");

    const dataKeys = Object.keys(rawData);
    for (const col of columns) {
      if (!Object.hasOwn(rawData, col)) {
        throw new DataValidationError(`Missing data for column '${col}'`);
      }
    }
    const columnSet = new Set(columns);
    for (const key of dataKeys) {
      if (!columnSet.has(key)) {
        throw new DataValidationError(`Unexpected data column '${key}' not listed in columns`);
      }
    }

    // A null-prototype object keeps a column called "__proto__" a plain entry.
    const data: DataFrameData = Object.create(null) as DataFrameData;
    for (const [key, value] of Object.entries(rawData)) {
      if (!Array.isArray(value)) {
        throw new DataValidationError(`Invalid data for column '${key}': expected array`);
      }
      data[key] = value;
    }

    return new DataFrame(data, {
      columns,
      index,
    });
  }

  /**
   * Read JSON file - environment-aware.
   * Time complexity: O(n) for file read + O(m) for parsing.
   */
  static async readJson(path: string): Promise<DataFrame> {
    let jsonString: string;

    if (typeof process !== "undefined" && process.versions?.node) {
      try {
        const fs = await import("node:fs/promises");
        jsonString = await fs.readFile(path, "utf-8");
      } catch (error) {
        throw new DataValidationError(
          `Failed to read JSON file: ${error instanceof Error ? error.message : String(error)}`
        );
      }
    } else if (typeof fetch !== "undefined") {
      try {
        const response = await fetch(path);
        if (!response.ok) {
          throw new DataValidationError(`HTTP ${response.status}: ${response.statusText}`);
        }
        jsonString = await response.text();
      } catch (error) {
        throw new DataValidationError(
          `Failed to fetch JSON: ${error instanceof Error ? error.message : String(error)}`
        );
      }
    } else {
      throw new DataValidationError("Environment not supported");
    }

    return DataFrame.fromJsonString(jsonString);
  }

  /**
   * Write DataFrame to JSON file - environment-aware.
   * Time complexity: O(n × m) for generation + O(k) for write.
   */
  async toJson(path: string): Promise<void> {
    const jsonString = this.toJsonString();

    if (typeof process !== "undefined" && process.versions?.node) {
      try {
        const fs = await import("node:fs/promises");
        await fs.writeFile(path, jsonString, "utf-8");
      } catch (error) {
        throw new DataValidationError(
          `Failed to write JSON file: ${error instanceof Error ? error.message : String(error)}`
        );
      }
    } else if (typeof document !== "undefined" && typeof URL !== "undefined") {
      const blob = new Blob([jsonString], {
        type: "application/json;charset=utf-8;",
      });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = path;
      link.style.display = "none";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } else {
      throw new DataValidationError("Environment not supported");
    }
  }

  /** One record per row, keyed by column name (the index is not included). */
  private toRowRecords(): Record<string, unknown>[] {
    const records: Record<string, unknown>[] = new Array(this._index.length);
    for (let r = 0; r < records.length; r++) {
      const row: Record<string, unknown> = Object.create(null) as Record<string, unknown>;
      for (const col of this._columns) setField(row, col, (this._data.get(col) ?? [])[r]);
      records[r] = row;
    }
    return records;
  }

  /** Build a DataFrame from the `{ columns, data }` result of an IO reader. */
  private static fromRowRecords(
    columns: readonly string[],
    rows: readonly Record<string, unknown>[]
  ): DataFrame {
    const data: DataFrameData = newColumnMap();
    for (const col of columns) {
      const values: unknown[] = new Array(rows.length);
      for (let r = 0; r < rows.length; r++) values[r] = rows[r]?.[col];
      data[col] = values;
    }
    return new DataFrame(data, { columns: [...columns], copy: false });
  }

  /**
   * Serialize the DataFrame to Parquet and return the file bytes.
   *
   * See {@link writeParquet} for the type mapping. The row index is not stored, and
   * reading the file back gives the default 0..n-1 index.
   *
   * @param options - Parquet write options
   * @returns Uint8Array containing the .parquet file
   * @throws {DataValidationError} If a column cannot be written (see {@link writeParquet})
   *
   * @example
   * ```ts
   * const bytes = df.toParquet();
   * const back = DataFrame.fromParquet(bytes);
   * ```
   */
  toParquet(options: ParquetWriteOptions = {}): Uint8Array {
    return writeParquet(this._columns, this.toRowRecords(), options);
  }

  /**
   * Create a DataFrame from Parquet file bytes.
   *
   * See {@link readParquet} for the value mapping: dates and timestamps become `Date`,
   * INT64 values beyond 2^53 become `bigint`, unannotated binary columns become
   * `Uint8Array`, and nulls become `null`. Those values are stored as they are;
   * numeric statistics skip columns that do not hold plain numbers.
   *
   * @param buffer - The raw .parquet file
   * @param options - Parquet read options (`columns` selects and orders a subset)
   * @returns DataFrame with a default 0..n-1 index
   * @throws {DataValidationError} If the buffer is not a valid or supported Parquet file
   */
  static fromParquet(buffer: Uint8Array, options: ParquetReadOptions = {}): DataFrame {
    const { columns, data } = readParquet(buffer, options);
    return DataFrame.fromRowRecords(columns, data);
  }

  /**
   * Serialize the DataFrame to an XLSX workbook and return the file bytes.
   *
   * See {@link writeXlsx} for the value mapping. The row index is not stored.
   *
   * @param options - XLSX write options
   * @returns Uint8Array containing the .xlsx file
   * @throws {DataValidationError} If the data exceeds Excel's sheet limits
   * @throws {InvalidParameterError} If the sheet name is not valid
   */
  toXlsx(options: XlsxWriteOptions = {}): Uint8Array {
    return writeXlsx(this._columns, this.toRowRecords(), options);
  }

  /**
   * Create a DataFrame from XLSX file bytes.
   *
   * See {@link readXlsx}: empty cells are `null` and date cells are Excel serial
   * numbers (the reader does not look at cell number formats).
   *
   * @param buffer - The raw .xlsx file
   * @param options - XLSX read options (sheet, header row, size limit)
   * @returns DataFrame with a default 0..n-1 index
   * @throws {DataValidationError} If the buffer is not a valid .xlsx file or the sheet does not exist
   */
  static fromXlsx(buffer: Uint8Array, options: XlsxReadOptions = {}): DataFrame {
    const { columns, data } = readXlsx(buffer, options);
    return DataFrame.fromRowRecords(columns, data);
  }

  /**
   * Create DataFrame from a Tensor.
   *
   * Strided views (transposes, slices) are read in logical order. Int64 values are
   * converted to numbers, so integers beyond 2^53 lose precision.
   *
   * @param source - Tensor to convert (must be 1D or 2D)
   * @param columns - Column names (optional). If provided, length must match tensor columns.
   * @returns DataFrame
   *
   * @example
   * ```ts
   * import { tensor } from 'deepbox/ndarray';
   *
   * const t = tensor([[1, 2], [3, 4], [5, 6]]);
   * const df = DataFrame.fromTensor(t, ['col1', 'col2']);
   * ```
   */
  static fromTensor(source: Tensor, columns?: string[]): DataFrame {
    const storage = source.data as ArrayLike<unknown>;
    const base = source.offset;

    const read = (flatIndex: number): unknown => {
      const v = storage[flatIndex];
      if (v === undefined) {
        throw new DataValidationError("Tensor storage is smaller than its shape and strides imply");
      }
      return typeof v === "bigint" ? Number(v) : v;
    };

    if (source.ndim === 1) {
      if (columns && columns.length !== 1) {
        throw new DataValidationError(
          `Expected exactly 1 column name for 1D tensor, received ${columns.length}`
        );
      }
      const n = source.shape[0] ?? 0;
      const stride = source.strides[0] ?? 1;
      const data: unknown[] = new Array(n);
      for (let i = 0; i < n; i++) data[i] = read(base + i * stride);
      const colName = columns?.[0] ?? "col0";
      return new DataFrame({ [colName]: data }, { copy: false });
    }

    if (source.ndim === 2) {
      const rows = source.shape[0];
      const cols = source.shape[1];

      if (rows === undefined || cols === undefined) {
        throw new DataValidationError("Invalid tensor shape");
      }

      if (columns && columns.length !== cols) {
        throw new DataValidationError(
          `Column count (${columns.length}) must match tensor columns (${cols})`
        );
      }

      const rowStride = source.strides[0] ?? cols;
      const colStride = source.strides[1] ?? 1;
      const names = columns ?? Array.from({ length: cols }, (_, i) => `col${i}`);
      const dfData: DataFrameData = newColumnMap();

      for (let c = 0; c < cols; c++) {
        const colData: unknown[] = new Array(rows);
        for (let r = 0; r < rows; r++) {
          colData[r] = read(base + r * rowStride + c * colStride);
        }
        dfData[names[c] as string] = colData;
      }

      return new DataFrame(dfData, { columns: names, copy: false });
    }

    throw new DataValidationError(
      `Cannot create DataFrame from ${source.ndim}D tensor. Only 1D and 2D tensors supported.`
    );
  }

  /**
   * Flags duplicated rows for dropDuplicates() and duplicated().
   *
   * @private
   */
  private duplicateMask(
    subset: readonly string[] | undefined,
    keep: "first" | "last" | false
  ): boolean[] {
    if (keep !== "first" && keep !== "last" && keep !== false) {
      throw new InvalidParameterError('keep must be "first", "last" or false', "keep", keep);
    }
    const checkCols = subset ?? this._columns;
    const arrays: (readonly unknown[])[] = [];
    for (const col of checkCols) {
      const arr = this._data.get(col);
      if (arr === undefined) {
        throw new DataValidationError(`Column '${col}' not found in DataFrame`);
      }
      arrays.push(arr);
    }

    const nRows = this._index.length;
    // Per distinct row: first position, last position and how often it occurs.
    const groups = new Map<string, { first: number; last: number; count: number }>();
    const rowKeys = new Array<string>(nRows);
    for (let i = 0; i < nRows; i++) {
      const signature: unknown[] = new Array(arrays.length);
      for (let c = 0; c < arrays.length; c++) signature[c] = (arrays[c] as readonly unknown[])[i];
      const key = createKey(signature);
      rowKeys[i] = key;
      const group = groups.get(key);
      if (group === undefined) {
        groups.set(key, { first: i, last: i, count: 1 });
      } else {
        group.last = i;
        group.count++;
      }
    }

    const isDuplicate = new Array<boolean>(nRows);
    for (let i = 0; i < nRows; i++) {
      const group = groups.get(rowKeys[i] as string) as {
        first: number;
        last: number;
        count: number;
      };
      if (keep === "first") isDuplicate[i] = group.first !== i;
      else if (keep === "last") isDuplicate[i] = group.last !== i;
      else isDuplicate[i] = group.count > 1;
    }
    return isDuplicate;
  }

  /**
   * Remove duplicate rows from DataFrame.
   * Time complexity: O(n × m) where n is rows, m is columns.
   *
   * @param subset - Columns to consider for identifying duplicates (default: all columns)
   * @param keep - Which duplicates to keep: 'first', 'last', or false (remove all)
   * @returns New DataFrame with duplicates removed
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, 1, 2], b: [3, 3, 4] });
   * df.dropDuplicates();  // Keeps first occurrence: [[1, 3], [2, 4]]
   * df.dropDuplicates(undefined, 'last');  // Keeps last occurrence
   * ```
   */
  dropDuplicates(subset?: string[], keep: "first" | "last" | false = "first"): DataFrame {
    const isDuplicate = this.duplicateMask(subset, keep);

    const keepIndices: number[] = [];
    for (let i = 0; i < isDuplicate.length; i++) {
      if (!isDuplicate[i]) keepIndices.push(i);
    }

    // Build result DataFrame from kept indices
    const newData: DataFrameData = newColumnMap();
    for (const col of this._columns) {
      const src = this._data.get(col) ?? [];
      const dst = new Array<unknown>(keepIndices.length);
      for (let k = 0; k < keepIndices.length; k++) dst[k] = src[keepIndices[k] as number];
      newData[col] = dst;
    }
    const newIndex = keepIndices.map((i) => this._index[i] as string | number);

    return new DataFrame(newData, {
      columns: this._columns,
      index: newIndex,
      copy: false,
    });
  }

  /**
   * Same as {@link DataFrame.dropDuplicates}.
   *
   * @deprecated Prefer {@link DataFrame.dropDuplicates}.
   */
  drop_duplicates(subset?: string[], keep: "first" | "last" | false = "first"): DataFrame {
    return this.dropDuplicates(subset, keep);
  }

  /**
   * Return boolean Series indicating duplicate rows.
   * Time complexity: O(n × m).
   *
   * @param subset - Columns to consider for identifying duplicates
   * @param keep - Which duplicates to mark as False: 'first', 'last', or false (mark all)
   * @returns Series of booleans (true = duplicate, false = unique)
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, 1, 2], b: [3, 3, 4] });
   * df.duplicated();  // Series([false, true, false])
   * ```
   */
  duplicated(subset?: string[], keep: "first" | "last" | false = "first"): Series<boolean> {
    return new Series(this.duplicateMask(subset, keep), { index: this._index });
  }

  /**
   * Rename columns or index labels.
   * Time complexity: O(m) for columns, O(n) for index.
   *
   * Labels missing from an object mapper keep their name. When renaming the
   * index with a function, the function receives each label as a string and the
   * result is used as the new label. Renames that produce duplicate labels throw.
   *
   * @param mapper - Object mapping old names to new names, or function to transform names
   * @param axis - 0 or "index" for row labels, 1 or "columns" for columns (default: 1)
   * @returns New DataFrame with renamed labels
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, 2], b: [3, 4] });
   * df.rename({ a: 'x', b: 'y' }, 1);  // Rename columns a->x, b->y
   * df.rename((name) => name.toUpperCase(), 1);  // Uppercase all column names
   * ```
   */
  rename(mapper: Record<string, string> | ((name: string) => string), axis: Axis = 1): DataFrame {
    const ax = normalizeAxis(axis, 2);
    const mapLabel = (label: string): string | undefined => {
      if (typeof mapper === "function") return mapper(label);
      return Object.hasOwn(mapper, label) ? mapper[label] : undefined;
    };

    if (ax === 1) {
      // Rename columns
      const newColumns = this._columns.map((col) => {
        const mapped = mapLabel(col);
        return mapped === undefined ? col : mapped;
      });

      const newData: DataFrameData = newColumnMap();
      for (let i = 0; i < this._columns.length; i++) {
        const oldCol = this._columns[i] as string;
        const newCol = newColumns[i] as string;
        newData[newCol] = [...(this._data.get(oldCol) ?? [])];
      }

      return new DataFrame(newData, {
        columns: newColumns,
        index: this._index,
        copy: false,
      });
    }

    // Rename index
    const newIndex = this._index.map((label) => {
      const mapped = mapLabel(String(label));
      return mapped === undefined ? label : mapped;
    });

    const newData: DataFrameData = newColumnMap();
    for (const col of this._columns) {
      newData[col] = [...(this._data.get(col) ?? [])];
    }

    return new DataFrame(newData, {
      columns: this._columns,
      index: newIndex,
      copy: false,
    });
  }

  /**
   * Reset index to default integer index.
   * Time complexity: O(n).
   *
   * @param drop - If true, don't add old index as column.
   *              If a column named "index" already exists, the new column will be
   *              named "index_1", "index_2", etc.
   * @returns New DataFrame with reset index
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, 2] }, { index: ['x', 'y'] });
   * df.resetIndex();  // Index becomes [0, 1], adds 'index' column with ['x', 'y']
   * df.resetIndex(true);  // Index becomes [0, 1], no new column
   * ```
   */
  resetIndex(drop: boolean = false): DataFrame {
    const newData: DataFrameData = newColumnMap();

    let indexName = "index";
    if (!drop) {
      if (this._columns.includes(indexName)) {
        let suffix = 1;
        while (this._columns.includes(`${indexName}_${suffix}`)) {
          suffix++;
        }
        indexName = `${indexName}_${suffix}`;
      }
      newData[indexName] = [...this._index];
    }

    for (const col of this._columns) {
      const colData = this._data.get(col);
      if (colData) {
        newData[col] = [...colData];
      }
    }

    const newColumns = drop ? this._columns : [indexName, ...this._columns];

    return new DataFrame(newData, {
      columns: newColumns,
      index: Array.from({ length: this._index.length }, (_, i) => i),
      copy: false,
    });
  }

  /**
   * Same as {@link DataFrame.resetIndex}.
   *
   * @deprecated Prefer {@link DataFrame.resetIndex}.
   */
  reset_index(drop: boolean = false): DataFrame {
    return this.resetIndex(drop);
  }

  /**
   * Set a column as the index.
   * Time complexity: O(n).
   *
   * @param column - Column name to use as index
   * @param drop - If true, remove the column after setting it as index
   * @returns New DataFrame with new index
   *
   * @example
   * ```ts
   * const df = new DataFrame({ id: ['a', 'b', 'c'], value: [1, 2, 3] });
   * df.setIndex('id');  // Index becomes ['a', 'b', 'c']
   * ```
   */
  setIndex(column: string, drop: boolean = true): DataFrame {
    if (!this._columns.includes(column)) {
      throw new InvalidParameterError(
        `Column '${column}' not found in DataFrame`,
        "column",
        column
      );
    }

    const newIndexData = this._data.get(column);
    if (!newIndexData) {
      throw new DataValidationError(`Column '${column}' has no data`);
    }

    const newIndex = newIndexData.map((v) =>
      typeof v === "string" || typeof v === "number" ? v : String(v)
    );

    const newData: DataFrameData = newColumnMap();
    const newColumns: string[] = [];

    for (const col of this._columns) {
      if (col === column && drop) continue;
      const colData = this._data.get(col);
      if (colData) {
        newData[col] = [...colData];
        newColumns.push(col);
      }
    }

    return new DataFrame(newData, {
      columns: newColumns,
      index: newIndex,
      copy: false,
    });
  }

  /**
   * Same as {@link DataFrame.setIndex}.
   *
   * @deprecated Prefer {@link DataFrame.setIndex}.
   */
  set_index(column: string, drop: boolean = true): DataFrame {
    return this.setIndex(column, drop);
  }

  /**
   * Return boolean DataFrame showing missing values (null, undefined and NaN).
   * Time complexity: O(n × m).
   *
   * @returns DataFrame of booleans (true = null, undefined or NaN)
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, null, 3], b: [4, 5, undefined] });
   * df.isnull();  // [[false, false], [true, false], [false, true]]
   * ```
   */
  isnull(): DataFrame {
    const newData: DataFrameData = newColumnMap();

    for (const col of this._columns) {
      const colData = this._data.get(col);
      if (colData) {
        newData[col] = colData.map((v) => isMissing(v));
      }
    }

    return new DataFrame(newData, {
      columns: this._columns,
      index: this._index,
      copy: false,
    });
  }

  /**
   * Return boolean DataFrame showing present values (not null, undefined or NaN).
   * Time complexity: O(n × m).
   *
   * @returns DataFrame of booleans (false = null, undefined or NaN)
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, null, 3], b: [4, 5, undefined] });
   * df.notnull();  // [[true, true], [false, true], [true, false]]
   * ```
   */
  notnull(): DataFrame {
    const newData: DataFrameData = newColumnMap();

    for (const col of this._columns) {
      const colData = this._data.get(col);
      if (colData) {
        newData[col] = colData.map((v) => !isMissing(v));
      }
    }

    return new DataFrame(newData, {
      columns: this._columns,
      index: this._index,
      copy: false,
    });
  }

  /**
   * Replace values in DataFrame.
   * Time complexity: O(n × m).
   *
   * @param toReplace - Value or array of values to replace
   * @param value - Replacement value
   * @returns New DataFrame with replaced values
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, 2, 3], b: [4, 5, 6] });
   * df.replace(2, 99);  // Replace all 2s with 99
   * df.replace([1, 2], 0);  // Replace 1s and 2s with 0
   * ```
   */
  replace(toReplace: unknown | unknown[], value: unknown): DataFrame {
    const replaceSet = new Set(Array.isArray(toReplace) ? toReplace : [toReplace]);

    const newData: DataFrameData = newColumnMap();

    for (const col of this._columns) {
      const colData = this._data.get(col);
      if (colData) {
        newData[col] = colData.map((v) => (replaceSet.has(v) ? value : v));
      }
    }

    return new DataFrame(newData, {
      columns: this._columns,
      index: this._index,
      copy: false,
    });
  }

  /**
   * Clip (limit) values in a range.
   * Time complexity: O(n × m).
   *
   * @param lower - Minimum value (values below are set to this)
   * @param upper - Maximum value (values above are set to this)
   * @returns New DataFrame with clipped values
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, 5, 10], b: [2, 8, 15] });
   * df.clip(3, 9);  // [[3, 3], [5, 8], [9, 9]]
   * ```
   */
  clip(lower?: number, upper?: number): DataFrame {
    if (lower !== undefined && Number.isNaN(lower)) {
      throw new InvalidParameterError("lower must not be NaN", "lower", lower);
    }
    if (upper !== undefined && Number.isNaN(upper)) {
      throw new InvalidParameterError("upper must not be NaN", "upper", upper);
    }
    const newData: DataFrameData = newColumnMap();

    for (const col of this._columns) {
      const colData = this._data.get(col);
      if (colData) {
        newData[col] = colData.map((v) => {
          if (typeof v !== "number") return v;
          let result = v;
          if (lower !== undefined && result < lower) result = lower;
          if (upper !== undefined && result > upper) result = upper;
          return result;
        });
      }
    }

    return new DataFrame(newData, {
      columns: this._columns,
      index: this._index,
      copy: false,
    });
  }

  /**
   * Return a random sample of rows.
   *
   * Call it as `sample(n, randomState?)` or with an options object. Exactly one of `n` and `frac`
   * may be given; with neither, one row is drawn. `frac` is turned into a row count with
   * round-half-to-even, as pandas does.
   *
   * Randomness comes from Deepbox's own generators: `randomState` creates a `Generator` seeded
   * with that integer, and without it the global generator is used, so `setSeed` makes the call
   * reproducible. The same seed always gives the same rows, but the rows differ from pandas.
   *
   * Without `replace`, rows are distinct and `n` may not exceed the number of rows. With
   * `weights`, each draw picks among the remaining rows in proportion to their weights. With
   * `replace: true` a row can appear several times; because row labels must stay unique, the
   * result is then labelled 0..k-1.
   *
   * @param n - Number of rows, or an options object (see {@link SampleOptions})
   * @param randomState - Integer seed (positional form only)
   * @returns New DataFrame with the sampled rows, in draw order
   * @throws {InvalidParameterError} If `n`, `frac`, the seed or the weights are invalid
   * @throws {DataValidationError} If more rows are requested than exist without `replace`
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, 2, 3, 4, 5], w: [0, 0, 1, 1, 8] });
   * df.sample(3);                                     // 3 random rows
   * df.sample(3, 42);                                 // reproducible
   * df.sample({ frac: 0.4, randomState: 1 });         // 2 rows
   * df.sample({ n: 10, replace: true, weights: "w" });
   * ```
   */
  sample(n: number | SampleOptions = {}, randomState?: number): DataFrame {
    const positional = typeof n === "number" || n === undefined;
    const options: SampleOptions = positional
      ? {
          ...(n !== undefined ? { n } : {}),
          ...(randomState !== undefined ? { randomState } : {}),
        }
      : n;
    if (!positional && !isPlainObject(n)) {
      throw new InvalidParameterError("n must be an integer or an options object", "n", n);
    }
    const seedName = positional ? "random_state" : "randomState";
    const total = this._index.length;

    if (options.n !== undefined && options.frac !== undefined) {
      throw new InvalidParameterError("Specify either n or frac, not both", "frac", options.frac);
    }
    if (options.n !== undefined && (!Number.isFinite(options.n) || !Number.isInteger(options.n))) {
      throw new InvalidParameterError("n must be a finite integer", "n", options.n);
    }
    if (options.frac !== undefined && (!Number.isFinite(options.frac) || options.frac < 0)) {
      throw new InvalidParameterError(
        "frac must be a finite non-negative number",
        "frac",
        options.frac
      );
    }
    const seed = options.randomState;
    if (seed !== undefined && (!Number.isFinite(seed) || !Number.isInteger(seed))) {
      throw new InvalidParameterError(`${seedName} must be a finite integer`, seedName, seed);
    }
    if (options.replace !== undefined && typeof options.replace !== "boolean") {
      throw new InvalidParameterError("replace must be a boolean", "replace", options.replace);
    }
    const replace = options.replace === true;
    const count =
      options.n ?? (options.frac !== undefined ? roundHalfEven(options.frac * total) : 1);
    if (count < 0 || (!replace && count > total) || (total === 0 && count > 0)) {
      throw new DataValidationError(`Sample size ${count} must be between 0 and ${total}`);
    }

    let weights: Float64Array | undefined;
    if (options.weights !== undefined) {
      const raw =
        typeof options.weights === "string"
          ? this.getColumnDataOrThrow(options.weights)
          : options.weights;
      if (raw.length !== total) {
        throw new InvalidParameterError(
          `weights must have one entry per row (${total}), got ${raw.length}`,
          "weights",
          raw.length
        );
      }
      weights = new Float64Array(total);
      let sum = 0;
      let positive = 0;
      for (let i = 0; i < total; i++) {
        const w = raw[i];
        if (isMissing(w)) continue;
        if (typeof w !== "number" || w === Infinity || w === -Infinity) {
          throw new InvalidParameterError("weights must be finite numbers", "weights", w);
        }
        if (w < 0) {
          throw new InvalidParameterError("weights must not be negative", "weights", w);
        }
        weights[i] = w;
        sum += w;
        if (w > 0) positive++;
      }
      if (count > 0 && !(sum > 0)) {
        throw new InvalidParameterError("weights must not sum to zero", "weights", sum);
      }
      if (count > 0 && !replace && positive < count) {
        throw new InvalidParameterError(
          `Cannot take ${count} rows without replacement: only ${positive} have a positive weight`,
          "weights",
          positive
        );
      }
    }

    const generator = seed !== undefined ? new Generator(seed) : undefined;
    const rng = generator !== undefined ? () => generator.random() : __random;
    const picked = drawRows(total, count, replace, weights, rng);

    const newData: DataFrameData = newColumnMap();
    for (const col of this._columns) {
      const source = this._data.get(col) ?? [];
      const column = new Array<unknown>(picked.length);
      for (let k = 0; k < picked.length; k++) column[k] = source[picked[k] as number];
      newData[col] = column;
    }
    const newIndex = replace
      ? picked.map((_, k) => k)
      : picked.map((row) => this._index[row] as string | number);

    return new DataFrame(newData, {
      columns: this._columns,
      index: newIndex,
      copy: false,
    });
  }

  /** Column values by name; throws InvalidParameterError for an unknown column. */
  private getColumnDataOrThrow(column: string): readonly unknown[] {
    const data = this._data.get(column);
    if (data === undefined) {
      throw new InvalidParameterError(
        `Column '${column}' not found in DataFrame`,
        "weights",
        column
      );
    }
    return data;
  }

  /**
   * Return values at the given quantile, using linear interpolation between
   * the two nearest ranks (the NumPy and pandas default).
   * NaN, null and non-numeric values are ignored. A column without any numeric
   * value gives NaN.
   * Time complexity: O(n log n) per column due to sorting.
   *
   * @param q - Quantile to compute (0 to 1)
   * @returns Series with quantile values for each numeric column
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, 2, 3, 4, 5], b: [10, 20, 30, 40, 50] });
   * df.quantile(0.5);  // Median: Series({ a: 3, b: 30 })
   * df.quantile(0.25);  // 25th percentile
   * ```
   */
  quantile(q: number): Series<number> {
    if (!Number.isFinite(q) || q < 0 || q > 1) {
      throw new InvalidParameterError("q must be a finite number between 0 and 1", "q", q);
    }

    const result: number[] = [];
    const resultIndex: string[] = [];

    for (const col of this._columns) {
      const colData = this._data.get(col);
      if (!colData) continue;

      const numericData = Float64Array.from(colData.filter(isOrderedNumber));
      resultIndex.push(col);
      if (numericData.length === 0) {
        result.push(NaN);
        continue;
      }

      numericData.sort();
      result.push(quantileSorted(numericData, q));
    }

    return new Series(result, { index: resultIndex });
  }

  /**
   * Compute numerical rank of values (1 through n) for each column.
   * Only numbers are ranked; NaN, null and other values get a null rank.
   * Time complexity: O(n log n) per column.
   *
   * @param method - How to rank ties: 'average', 'min', 'max', 'first', 'dense'
   * @param ascending - Rank in ascending order
   * @returns New DataFrame with ranks
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [3, 1, 2, 1] });
   * df.rank();  // [[4], [1.5], [3], [1.5]] (average method)
   * df.rank('min');  // [[4], [1], [3], [1]]
   * ```
   */
  rank(
    method: "average" | "min" | "max" | "first" | "dense" = "average",
    ascending: boolean = true
  ): DataFrame {
    if (!["average", "min", "max", "first", "dense"].includes(method)) {
      throw new InvalidParameterError(
        'method must be one of "average", "min", "max", "first" or "dense"',
        "method",
        method
      );
    }
    const newData: DataFrameData = newColumnMap();

    for (const col of this._columns) {
      const colData = this._data.get(col);
      if (!colData) continue;

      // Positions of rankable values; NaN, null and non-numbers get no rank.
      const positions: number[] = [];
      for (let i = 0; i < colData.length; i++) {
        if (isOrderedNumber(colData[i])) positions.push(i);
      }
      if (positions.length === 0) {
        newData[col] = colData.map(() => null);
        continue;
      }

      // Stable sort, so equal values keep their original order ("first" method).
      positions.sort((a, b) => {
        const cmp = compareNumbers(colData[a] as number, colData[b] as number);
        return ascending ? cmp : -cmp;
      });

      const ranks: (number | null)[] = new Array(colData.length).fill(null);

      let i = 0;
      let denseRank = 0;
      while (i < positions.length) {
        const tieStart = i;
        const tieValue = colData[positions[i] as number] as number;
        while (i < positions.length && colData[positions[i] as number] === tieValue) {
          i++;
        }
        const tieEnd = i;
        denseRank++;

        for (let j = tieStart; j < tieEnd; j++) {
          let rank: number;
          if (method === "average") {
            rank = (tieStart + tieEnd + 1) / 2;
          } else if (method === "min") {
            rank = tieStart + 1;
          } else if (method === "max") {
            rank = tieEnd;
          } else if (method === "first") {
            rank = j + 1;
          } else {
            rank = denseRank;
          }
          ranks[positions[j] as number] = rank;
        }
      }

      newData[col] = ranks;
    }

    return new DataFrame(newData, {
      columns: this._columns,
      index: this._index,
      copy: false,
    });
  }

  /**
   * Calculate the difference between consecutive rows.
   * Time complexity: O(n × m).
   *
   * @param periods - Number of periods to shift (default: 1)
   * @returns New DataFrame with differences
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, 3, 6, 10] });
   * df.diff();  // [[null], [2], [3], [4]]
   * df.diff(2);  // [[null], [null], [5], [7]]
   * ```
   */
  diff(periods: number = 1): DataFrame {
    if (!Number.isFinite(periods) || !Number.isInteger(periods) || periods < 0) {
      throw new InvalidParameterError("periods must be a non-negative integer", "periods", periods);
    }
    const newData: DataFrameData = newColumnMap();

    for (const col of this._columns) {
      const colData = this._data.get(col);
      if (!colData) continue;

      const diffData: unknown[] = [];

      for (let i = 0; i < colData.length; i++) {
        if (i < periods) {
          diffData.push(null);
        } else {
          const current = colData[i];
          const previous = colData[i - periods];

          if (typeof current === "number" && typeof previous === "number") {
            diffData.push(current - previous);
          } else {
            diffData.push(null);
          }
        }
      }

      newData[col] = diffData;
    }

    return new DataFrame(newData, {
      columns: this._columns,
      index: this._index,
      copy: false,
    });
  }

  /**
   * Calculate percentage change between consecutive rows.
   * Time complexity: O(n × m).
   *
   * @param periods - Number of periods to shift (default: 1)
   * @returns New DataFrame with percentage changes
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [100, 110, 121] });
   * df.pctChange();  // [[null], [0.1], [0.1]] (10% increase each time)
   * ```
   */
  pctChange(periods: number = 1): DataFrame {
    if (!Number.isFinite(periods) || !Number.isInteger(periods) || periods < 0) {
      throw new InvalidParameterError("periods must be a non-negative integer", "periods", periods);
    }
    const newData: DataFrameData = newColumnMap();

    for (const col of this._columns) {
      const colData = this._data.get(col);
      if (!colData) continue;

      const pctData: unknown[] = [];

      for (let i = 0; i < colData.length; i++) {
        if (i < periods) {
          pctData.push(null);
        } else {
          const current = colData[i];
          const previous = colData[i - periods];

          if (typeof current === "number" && typeof previous === "number" && previous !== 0) {
            pctData.push((current - previous) / previous);
          } else {
            pctData.push(null);
          }
        }
      }

      newData[col] = pctData;
    }

    return new DataFrame(newData, {
      columns: this._columns,
      index: this._index,
      copy: false,
    });
  }

  /**
   * Same as {@link DataFrame.pctChange}.
   *
   * @deprecated Prefer {@link DataFrame.pctChange}.
   */
  pct_change(periods: number = 1): DataFrame {
    return this.pctChange(periods);
  }

  /**
   * Return the cumulative sum of each column.
   *
   * NaN entries stay NaN and are skipped by the running total; null and other
   * non-numbers become null.
   * Time complexity: O(n × m).
   *
   * @returns New DataFrame with cumulative sums
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, 2, 3], b: [4, 5, 6] });
   * df.cumsum();  // [[1, 4], [3, 9], [6, 15]]
   * ```
   */
  cumsum(): DataFrame {
    const newData: DataFrameData = newColumnMap();

    for (const col of this._columns) {
      const colData = this._data.get(col);
      if (!colData) continue;

      const cumData: unknown[] = [];
      let cumSum = 0;

      for (const value of colData) {
        if (typeof value !== "number") {
          cumData.push(null);
        } else if (Number.isNaN(value)) {
          // NaN stays NaN in place but does not poison the running total.
          cumData.push(NaN);
        } else {
          cumSum += value;
          cumData.push(cumSum);
        }
      }

      newData[col] = cumData;
    }

    return new DataFrame(newData, {
      columns: this._columns,
      index: this._index,
      copy: false,
    });
  }

  /**
   * Return the cumulative product of each column.
   *
   * NaN entries stay NaN and are skipped by the running product; null and other
   * non-numbers become null.
   * Time complexity: O(n × m).
   *
   * @returns New DataFrame with cumulative products
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [2, 3, 4] });
   * df.cumprod();  // [[2], [6], [24]]
   * ```
   */
  cumprod(): DataFrame {
    const newData: DataFrameData = newColumnMap();

    for (const col of this._columns) {
      const colData = this._data.get(col);
      if (!colData) continue;

      const cumData: unknown[] = [];
      let cumProd = 1;

      for (const value of colData) {
        if (typeof value !== "number") {
          cumData.push(null);
        } else if (Number.isNaN(value)) {
          cumData.push(NaN);
        } else {
          cumProd *= value;
          cumData.push(cumProd);
        }
      }

      newData[col] = cumData;
    }

    return new DataFrame(newData, {
      columns: this._columns,
      index: this._index,
      copy: false,
    });
  }

  /**
   * Return the cumulative maximum of each column.
   *
   * NaN entries stay NaN and are skipped; null and other non-numbers become null.
   * Time complexity: O(n × m).
   *
   * @returns New DataFrame with cumulative maximums
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [3, 1, 5, 2] });
   * df.cummax();  // [[3], [3], [5], [5]]
   * ```
   */
  cummax(): DataFrame {
    const newData: DataFrameData = newColumnMap();

    for (const col of this._columns) {
      const colData = this._data.get(col);
      if (!colData) continue;

      const cumData: unknown[] = [];
      let cumMax = -Infinity;

      for (const value of colData) {
        if (typeof value !== "number") {
          cumData.push(null);
        } else if (Number.isNaN(value)) {
          cumData.push(NaN);
        } else {
          if (value > cumMax) cumMax = value;
          cumData.push(cumMax);
        }
      }

      newData[col] = cumData;
    }

    return new DataFrame(newData, {
      columns: this._columns,
      index: this._index,
      copy: false,
    });
  }

  /**
   * Return the cumulative minimum of each column.
   *
   * NaN entries stay NaN and are skipped; null and other non-numbers become null.
   * Time complexity: O(n × m).
   *
   * @returns New DataFrame with cumulative minimums
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [3, 1, 5, 2] });
   * df.cummin();  // [[3], [1], [1], [1]]
   * ```
   */
  cummin(): DataFrame {
    const newData: DataFrameData = newColumnMap();

    for (const col of this._columns) {
      const colData = this._data.get(col);
      if (!colData) continue;

      const cumData: unknown[] = [];
      let cumMin = Infinity;

      for (const value of colData) {
        if (typeof value !== "number") {
          cumData.push(null);
        } else if (Number.isNaN(value)) {
          cumData.push(NaN);
        } else {
          if (value < cumMin) cumMin = value;
          cumData.push(cumMin);
        }
      }

      newData[col] = cumData;
    }

    return new DataFrame(newData, {
      columns: this._columns,
      index: this._index,
      copy: false,
    });
  }

  /**
   * Shift index by desired number of periods.
   * Time complexity: O(n × m).
   *
   * @param periods - Number of periods to shift (positive = down, negative = up)
   * @param fillValue - Value to use for newly introduced missing values
   * @returns New DataFrame with shifted data
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, 2, 3, 4] });
   * df.shift(1);  // [[null], [1], [2], [3]]
   * df.shift(-1);  // [[2], [3], [4], [null]]
   * df.shift(1, 0);  // [[0], [1], [2], [3]]
   * ```
   */
  shift(periods: number = 1, fillValue: unknown = null): DataFrame {
    if (!Number.isFinite(periods) || !Number.isInteger(periods)) {
      throw new InvalidParameterError("periods must be a finite integer", "periods", periods);
    }

    const newData: DataFrameData = newColumnMap();

    for (const col of this._columns) {
      const colData = this._data.get(col);
      if (!colData) continue;

      const shiftedData: unknown[] = [];
      const rowCount = colData.length;

      if (periods > 0) {
        const shift = Math.min(periods, rowCount);
        for (let i = 0; i < shift; i++) {
          shiftedData.push(fillValue);
        }
        for (let i = 0; i < rowCount - shift; i++) {
          shiftedData.push(colData[i]);
        }
      } else if (periods < 0) {
        const absPeriods = Math.min(Math.abs(periods), rowCount);
        for (let i = absPeriods; i < rowCount; i++) {
          shiftedData.push(colData[i]);
        }
        for (let i = 0; i < absPeriods; i++) {
          shiftedData.push(fillValue);
        }
      } else {
        for (let i = 0; i < rowCount; i++) shiftedData.push(colData[i]);
      }

      newData[col] = shiftedData;
    }

    return new DataFrame(newData, {
      columns: this._columns,
      index: this._index,
      copy: false,
    });
  }

  /**
   * Pivot DataFrame.
   * Time complexity: O(n × m).
   *
   * @param index - Column to use as index
   * @param columns - Column to use as column headers
   * @param values - Column to use as values
   * @returns New DataFrame with pivoted data
   *
   * @example
   * ```ts
   * const df = new DataFrame({
   *   country: ['USA', 'USA', 'Canada', 'Canada'],
   *   year: [2010, 2011, 2010, 2011],
   *   value: [100, 200, 300, 400]
   * });
   * df.pivot('country', 'year', 'value');
   * // country | 2010 | 2011
   * //   USA   | 100  | 200
   * //  Canada | 300  | 400
   * ```
   */
  pivot(index: string, columns: string, values: string): DataFrame {
    if (!this._columns.includes(index)) {
      throw new DataValidationError(`Column '${index}' not found in DataFrame`);
    }

    if (!this._columns.includes(columns)) {
      throw new DataValidationError(`Column '${columns}' not found in DataFrame`);
    }

    if (!this._columns.includes(values)) {
      throw new DataValidationError(`Column '${values}' not found in DataFrame`);
    }

    const indexData = this._data.get(index);
    const columnData = this._data.get(columns);
    const valueData = this._data.get(values);

    if (!indexData || !columnData || !valueData) {
      throw new DataValidationError("Pivot columns have no data");
    }

    const pivotData: DataFrameData = newColumnMap();
    const pivotIndex: (string | number)[] = [];
    const uniqueIndices = new Set<string | number>();
    const uniqueColumns: string[] = [];
    const seenColumns = new Set<string>();

    for (const idx of indexData) {
      if (idx === null || idx === undefined) {
        continue;
      }
      const key = typeof idx === "string" || typeof idx === "number" ? idx : String(idx);
      if (!uniqueIndices.has(key)) {
        uniqueIndices.add(key);
        pivotIndex.push(key);
      }
    }

    for (const col of columnData) {
      if (col === null || col === undefined) {
        continue;
      }
      const colKey = String(col);
      if (!seenColumns.has(colKey)) {
        seenColumns.add(colKey);
        uniqueColumns.push(colKey);
      }
    }

    const rowPositionByIndex = new Map<string | number, number>();
    for (let i = 0; i < pivotIndex.length; i++) {
      const key = pivotIndex[i];
      if (key !== undefined) {
        rowPositionByIndex.set(key, i);
      }
    }

    for (const colKey of uniqueColumns) {
      pivotData[colKey] = new Array<unknown>(pivotIndex.length).fill(null);
    }

    // Track visited cells to detect duplicates even when values are null
    const visited = new Set<string>();

    for (let i = 0; i < indexData.length; i++) {
      const idx = indexData[i];
      const col = columnData[i];
      const value = valueData[i];

      if (idx !== null && idx !== undefined && col !== null && col !== undefined) {
        const indexKey = typeof idx === "string" || typeof idx === "number" ? idx : String(idx);
        const colKey = String(col);
        const rowPos = rowPositionByIndex.get(indexKey);

        if (rowPos === undefined) {
          continue;
        }

        const cellKey = `${rowPos}:${colKey}`;
        if (visited.has(cellKey)) {
          throw new DataValidationError(
            `Duplicate pivot entry for index '${String(indexKey)}' and column '${colKey}'`
          );
        }
        visited.add(cellKey);

        const targetColumn = pivotData[colKey];
        if (targetColumn) {
          targetColumn[rowPos] = value;
        }
      }
    }

    return new DataFrame(pivotData, {
      columns: uniqueColumns,
      index: pivotIndex,
    });
  }

  /**
   * Melt DataFrame.
   * Time complexity: O(n × m).
   *
   * @param idVars - Columns to keep as is
   * @param valueVars - Columns to melt (default: every column that is not in `idVars`)
   * @param varName - Name for new column with melted variable names
   * @param valueName - Name for new column with melted values.
   *                    Must not conflict with existing columns or varName.
   * @returns New DataFrame with melted data
   *
   * @example
   * ```ts
   * const df = new DataFrame({
   *   id: ['a', 'b'],
   *   x: [1, 2],
   *   y: [3, 4]
   * });
   * df.melt(['id'], ['x', 'y'], 'variable', 'value');
   * // id | variable | value
   * //  a |       x |     1
   * //  a |       y |     3
   * //  b |       x |     2
   * //  b |       y |     4
   * ```
   */
  melt(
    idVars: string[],
    valueVars?: string[],
    varName: string = "variable",
    valueName: string = "value"
  ): DataFrame {
    const ids = [...idVars];
    const idSet = new Set(ids);
    const values = valueVars ? [...valueVars] : this._columns.filter((c) => !idSet.has(c));

    ensureUniqueLabels(ids, "idVars element");
    ensureUniqueLabels(values, "valueVars element");

    for (const idVar of ids) {
      if (!this._columns.includes(idVar)) {
        throw new DataValidationError(`Column '${idVar}' not found in DataFrame`);
      }
    }

    for (const valueVar of values) {
      if (!this._columns.includes(valueVar)) {
        throw new DataValidationError(`Column '${valueVar}' not found in DataFrame`);
      }
    }

    if (varName === valueName) {
      throw new DataValidationError("varName and valueName must be different");
    }

    const reservedNames = new Set([...ids, ...values]);
    if (reservedNames.has(varName) || reservedNames.has(valueName)) {
      throw new DataValidationError(
        "varName and valueName must not conflict with existing columns"
      );
    }

    const nRows = this._index.length;
    const nOut = nRows * values.length;
    const newData: DataFrameData = newColumnMap();
    for (const idVar of ids) {
      const src = this._data.get(idVar) ?? [];
      const out = new Array<unknown>(nOut);
      let w = 0;
      for (let i = 0; i < nRows; i++) {
        for (let v = 0; v < values.length; v++) out[w++] = src[i];
      }
      newData[idVar] = out;
    }

    const varColumn = new Array<unknown>(nOut);
    const valueColumn = new Array<unknown>(nOut);
    {
      let w = 0;
      for (let i = 0; i < nRows; i++) {
        for (const valueVar of values) {
          varColumn[w] = valueVar;
          valueColumn[w] = this._data.get(valueVar)?.[i];
          w++;
        }
      }
    }
    newData[varName] = varColumn;
    newData[valueName] = valueColumn;

    return new DataFrame(newData, {
      columns: [...ids, varName, valueName],
      copy: false,
    });
  }

  /**
   * Stack (pivot) specified value columns into rows.
   *
   * Converts a wide-format DataFrame into a long-format one by taking
   * the specified columns and stacking their values into two new columns:
   * one for the variable name and one for the value.
   *
   * Non-stacked columns are repeated for each stacked variable.
   * Null/undefined values are preserved in the output.
   *
   * @param options - Configuration options
   * @param options.columns - Columns to stack (default: all columns)
   * @param options.varName - Name for the new variable column (default: 'variable')
   * @param options.valueName - Name for the new value column (default: 'value')
   * @returns New DataFrame in long format
   *
   * @example
   * ```ts
   * const df = new DataFrame({
   *   city: ['NYC', 'LA'],
   *   pop_2020: [8.3, 3.9],
   *   pop_2021: [8.4, 4.0]
   * });
   * df.stack({ columns: ['pop_2020', 'pop_2021'], varName: 'year', valueName: 'population' });
   * // city | year     | population
   * // NYC  | pop_2020 | 8.3
   * // NYC  | pop_2021 | 8.4
   * // LA   | pop_2020 | 3.9
   * // LA   | pop_2021 | 4.0
   * ```
   */
  stack(
    options: {
      readonly columns?: string[];
      readonly varName?: string;
      readonly valueName?: string;
    } = {}
  ): DataFrame {
    const varName = options.varName ?? "variable";
    const valueName = options.valueName ?? "value";

    if (varName === valueName) {
      throw new DataValidationError("varName and valueName must be different");
    }

    const stackCols = options.columns ?? [...this._columns];
    for (const col of stackCols) {
      if (!this._columns.includes(col)) {
        throw new DataValidationError(`Column '${col}' not found in DataFrame`);
      }
    }
    ensureUniqueLabels(stackCols, "stack column");

    const stackSet = new Set(stackCols);
    const idCols = this._columns.filter((c) => !stackSet.has(c));
    if (idCols.includes(varName) || idCols.includes(valueName)) {
      throw new DataValidationError(
        "varName and valueName must not conflict with the columns that are not stacked"
      );
    }

    // Build output
    const nRows = this._index.length;
    const nOut = nRows * stackCols.length;
    const newData: DataFrameData = newColumnMap();
    for (const id of idCols) {
      const src = this._data.get(id) ?? [];
      const out = new Array<unknown>(nOut);
      let w = 0;
      for (let i = 0; i < nRows; i++) {
        for (let k = 0; k < stackCols.length; k++) out[w++] = src[i];
      }
      newData[id] = out;
    }
    const varColumn = new Array<unknown>(nOut);
    const valueColumn = new Array<unknown>(nOut);
    {
      let w = 0;
      for (let i = 0; i < nRows; i++) {
        for (const sc of stackCols) {
          varColumn[w] = sc;
          valueColumn[w] = this._data.get(sc)?.[i];
          w++;
        }
      }
    }
    newData[varName] = varColumn;
    newData[valueName] = valueColumn;

    return new DataFrame(newData, {
      columns: [...idCols, varName, valueName],
      copy: false,
    });
  }

  /**
   * Unstack (pivot) a column's values into new columns.
   *
   * Converts a long-format DataFrame into a wide-format one by taking
   * the unique values of a specified column and creating a new column
   * for each, filled with the corresponding values from a value column.
   *
   * Requires an index column whose values (combined with the unstacked
   * column values) uniquely identify each row.
   *
   * @param options - Configuration options
   * @param options.index - Column to use as the row index
   * @param options.column - Column whose unique values become new columns
   * @param options.value - Column containing the values to fill
   * @returns New DataFrame in wide format
   *
   * @example
   * ```ts
   * const df = new DataFrame({
   *   city: ['NYC', 'NYC', 'LA', 'LA'],
   *   year: ['2020', '2021', '2020', '2021'],
   *   pop: [8.3, 8.4, 3.9, 4.0]
   * });
   * df.unstack({ index: 'city', column: 'year', value: 'pop' });
   * // city | 2020 | 2021
   * // NYC  | 8.3  | 8.4
   * // LA   | 3.9  | 4.0
   * ```
   */
  unstack(options: {
    readonly index: string;
    readonly column: string;
    readonly value: string;
  }): DataFrame {
    const { index: indexCol, column: pivotCol, value: valueCol } = options;

    for (const col of [indexCol, pivotCol, valueCol]) {
      if (!this._columns.includes(col)) {
        throw new DataValidationError(`Column '${col}' not found in DataFrame`);
      }
    }

    const indexData = this._data.get(indexCol);
    const pivotData = this._data.get(pivotCol);
    const valueData = this._data.get(valueCol);

    if (!indexData || !pivotData || !valueData) {
      throw new DataValidationError("Unstack columns have no data");
    }

    // Collect unique index values and unique pivot column values
    const uniqueIdx: (string | number)[] = [];
    const idxSet = new Set<string | number>();
    const uniquePivot: string[] = [];
    const pivotSet = new Set<string>();

    for (const val of indexData) {
      if (val === null || val === undefined) continue;
      const key = typeof val === "string" || typeof val === "number" ? val : String(val);
      if (!idxSet.has(key)) {
        idxSet.add(key);
        uniqueIdx.push(key);
      }
    }

    for (const val of pivotData) {
      if (val === null || val === undefined) continue;
      const key = String(val);
      if (!pivotSet.has(key)) {
        pivotSet.add(key);
        uniquePivot.push(key);
      }
    }

    // Build row position lookup
    const rowPos = new Map<string | number, number>();
    for (let i = 0; i < uniqueIdx.length; i++) {
      const key = uniqueIdx[i];
      if (key !== undefined) rowPos.set(key, i);
    }

    // Initialize output columns
    const newData: DataFrameData = newColumnMap();
    for (const pv of uniquePivot) {
      newData[pv] = new Array<unknown>(uniqueIdx.length).fill(null);
    }

    // Fill values; a repeated (index, column) pair would silently overwrite
    // the earlier value, so it is rejected like pandas does.
    const nRows = indexData.length;
    const filled = new Set<string>();
    for (let i = 0; i < nRows; i++) {
      const idx = indexData[i];
      const pv = pivotData[i];
      if (idx === null || idx === undefined || pv === null || pv === undefined) {
        continue;
      }
      const idxKey = toIndexLabel(idx);
      const pvKey = String(pv);
      const rp = rowPos.get(idxKey);
      if (rp !== undefined) {
        const cellKey = `${rp}:${pvKey}`;
        if (filled.has(cellKey)) {
          throw new DataValidationError(
            `Duplicate unstack entry for index '${String(idxKey)}' and column '${pvKey}'`
          );
        }
        filled.add(cellKey);
        const col = newData[pvKey];
        if (col) {
          col[rp] = valueData[i];
        }
      }
    }

    return new DataFrame(newData, {
      columns: uniquePivot,
      index: uniqueIdx,
    });
  }

  /**
   * Create a Rolling object for window-based calculations.
   *
   * By default a window ends at the current row, covers `window` rows, and needs `window` valid
   * numbers to produce a result, so the first `window - 1` rows and any window holding a missing
   * value give null. `minPeriods` lowers that requirement: windows cut off by the start of the data
   * (or containing missing values) then compute from the valid values they have, provided there
   * are at least `minPeriods` of them. With `center: true` the window is placed around the current
   * row instead of ending at it: it spans `window >> 1` rows before and `(window - 1) >> 1` rows
   * after, exactly as in pandas.
   *
   * @param window - Size of the rolling window
   * @param on - Column to apply rolling calculation to (if omitted, applies to all columns), or an
   *   options object `{ on?, minPeriods?, center? }`
   * @returns Rolling object with mean(), sum(), std(), var(), min(), max(), apply() methods
   * @throws {InvalidParameterError} If `window` is not a positive integer or `minPeriods` is not an
   *   integer between 0 and `window`
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, 2, 3, 4, 5] });
   * df.rolling(3).mean();                          // [null, null, 2, 3, 4]
   * df.rolling(3, { minPeriods: 1 }).sum();        // [1, 3, 6, 9, 12]
   * df.rolling(3, { center: true }).mean();        // [null, 2, 3, 4, null]
   * ```
   */
  rolling(window: number, on?: string | RollingOptions): Rolling {
    if (!Number.isFinite(window) || !Number.isInteger(window) || window <= 0) {
      throw new InvalidParameterError("window must be a positive integer", "window", window);
    }
    const options: RollingOptions = typeof on === "object" && on !== null ? on : {};
    const column = typeof on === "string" ? on : options.on;
    if (column && !this._columns.includes(column)) {
      throw new DataValidationError(`Column '${column}' not found in DataFrame`);
    }
    const minPeriods = options.minPeriods;
    if (
      minPeriods !== undefined &&
      (!Number.isInteger(minPeriods) || minPeriods < 0 || minPeriods > window)
    ) {
      throw new InvalidParameterError(
        "minPeriods must be an integer between 0 and window",
        "minPeriods",
        minPeriods
      );
    }
    if (options.center !== undefined && typeof options.center !== "boolean") {
      throw new InvalidParameterError("center must be a boolean", "center", options.center);
    }
    const extra: { minPeriods?: number; center?: boolean } = {};
    if (minPeriods !== undefined) extra.minPeriods = minPeriods;
    if (options.center !== undefined) extra.center = options.center;
    return new Rolling(this, window, column, extra);
  }

  /**
   * Apply a function element-wise to the DataFrame.
   *
   * @param fn - Function to apply to each element. It receives only the value.
   * @returns New DataFrame with transformed values
   */
  applymap(fn: (value: unknown) => unknown): DataFrame {
    const newData: DataFrameData = newColumnMap();
    for (const col of this._columns) {
      const colData = this._data.get(col);
      if (!colData) continue;
      // Wrapped so Array.map's (value, index, array) arguments never reach `fn`
      // (otherwise applymap(parseInt) would read the index as a radix).
      newData[col] = colData.map((v) => fn(v));
    }
    return new DataFrame(newData, {
      columns: this._columns,
      index: this._index,
      copy: false,
    });
  }

  /**
   * Pass the DataFrame through a function chain.
   *
   * Useful for method chaining with custom functions.
   *
   * @param fn - Function that takes a DataFrame and returns a value
   * @param args - Additional arguments to pass to the function
   * @returns The result of fn(this, ...args)
   */
  pipe<T>(fn: (df: DataFrame, ...args: unknown[]) => T, ...args: unknown[]): T {
    return fn(this, ...args);
  }

  /**
   * Explode a list-like column into separate rows.
   *
   * Each element in the specified column that is an array will produce
   * one row per element. Non-array values are kept as-is. An empty array
   * produces a single row with null, as in pandas. The result gets a fresh
   * 0..n-1 index because the original labels would repeat.
   *
   * @param column - Column name containing array values
   * @returns New DataFrame with exploded rows
   */
  explode(column: string): DataFrame {
    if (!this._columns.includes(column)) {
      throw new DataValidationError(`Column '${column}' not found in DataFrame`);
    }
    const colData = this._data.get(column);
    if (!colData) {
      throw new DataValidationError(`Column '${column}' has no data`);
    }

    // For every output row remember which input row it came from.
    const sourceRows: number[] = [];
    const exploded: unknown[] = [];
    for (let i = 0; i < this._index.length; i++) {
      const val = colData[i];
      if (Array.isArray(val)) {
        if (val.length === 0) {
          sourceRows.push(i);
          exploded.push(null);
        } else {
          for (const item of val) {
            sourceRows.push(i);
            exploded.push(item);
          }
        }
      } else {
        sourceRows.push(i);
        exploded.push(val);
      }
    }

    const obj: DataFrameData = newColumnMap();
    for (const c of this._columns) {
      if (c === column) {
        obj[c] = exploded;
        continue;
      }
      const src = this._data.get(c) ?? [];
      obj[c] = sourceRows.map((i) => src[i]);
    }
    // Use numeric index to avoid duplicate label errors after explosion
    const numericIndex = Array.from({ length: sourceRows.length }, (_, i) => i);
    return new DataFrame(obj, { columns: this._columns, index: numericIndex, copy: false });
  }

  /**
   * Convert categorical column(s) into dummy/indicator variables.
   *
   * Each distinct non-missing value becomes a 0/1 column named `<prefix>_<value>`.
   * Categories are ordered numerically when they are numbers and by string
   * otherwise. Null, undefined and NaN get all-zero rows (unless `dummyNa` is
   * set). The new columns are appended after the columns that were not encoded.
   *
   * @param columns - Column name(s) to encode. If omitted, encodes all columns that hold strings.
   * @param options - Options: prefix (column name prefix, default: the column name),
   *   dropFirst (drop the first category of each column), dummyNa (add a `<prefix>_nan`
   *   column that flags missing values)
   * @returns New DataFrame with dummy columns
   * @throws {InvalidParameterError} If a requested column does not exist
   */
  getDummies(
    columns?: string | string[],
    options: { prefix?: string; dropFirst?: boolean; dummyNa?: boolean } = {}
  ): DataFrame {
    const cols = columns
      ? Array.isArray(columns)
        ? columns
        : [columns]
      : this._columns.filter((c) => {
          const d = this._data.get(c);
          return d?.some((v) => typeof v === "string") ?? false;
        });
    for (const c of cols) {
      if (!this._data.has(c)) {
        throw new InvalidParameterError(`Column '${c}' not found in DataFrame`, "columns", c);
      }
    }
    ensureUniqueLabels(cols, "column name");

    const encoded = new Set(cols);
    const newData: DataFrameData = newColumnMap();
    const newColumns: string[] = [];

    // Copy non-dummy columns
    for (const c of this._columns) {
      if (!encoded.has(c)) {
        newData[c] = [...(this._data.get(c) ?? [])];
        newColumns.push(c);
      }
    }

    const nRows = this._index.length;
    for (const col of cols) {
      const colData = this._data.get(col) ?? [];
      const prefix = options.prefix ?? col;

      // Distinct values by their string form, remembering one raw value for ordering.
      const categories = new Map<string, unknown>();
      for (const v of colData) {
        if (isMissing(v)) continue;
        const key = String(v);
        if (!categories.has(key)) categories.set(key, v);
      }
      const ordered = [...categories.entries()].sort((a, b) => compareLabels(a[1], b[1]));
      const kept = options.dropFirst ? ordered.slice(1) : ordered;

      const columnFor = new Map<string, number[]>();
      for (const [key] of kept) {
        const arr = new Array<number>(nRows).fill(0);
        columnFor.set(key, arr);
        const dummyName = `${prefix}_${key}`;
        newData[dummyName] = arr;
        newColumns.push(dummyName);
      }
      let naColumn: number[] | undefined;
      if (options.dummyNa) {
        naColumn = new Array<number>(nRows).fill(0);
        newData[`${prefix}_nan`] = naColumn;
        newColumns.push(`${prefix}_nan`);
      }

      for (let i = 0; i < nRows; i++) {
        const v = colData[i];
        if (isMissing(v)) {
          if (naColumn) naColumn[i] = 1;
          continue;
        }
        const arr = columnFor.get(String(v));
        if (arr) arr[i] = 1;
      }
    }

    return new DataFrame(newData, { columns: newColumns, index: this._index, copy: false });
  }

  /**
   * Return a human-readable tabular string representation.
   *
   * Columns are right-aligned and padded so that rows line up.
   * Large DataFrames are truncated with an ellipsis row.
   *
   * @param maxRows - Maximum rows to display before summarizing (default: 20).
   * @returns Formatted table string
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, 2], b: [3, 4] });
   * df.toString();
   * // "   a  b\n0  1  3\n1  2  4"
   * ```
   */
  toString(maxRows = 20): string {
    if (!Number.isInteger(maxRows) || maxRows < 0) {
      throw new InvalidParameterError("maxRows must be a non-negative integer", "maxRows", maxRows);
    }
    const nRows = this._index.length;
    const cols = this._columns;

    // Determine which rows to show
    const half = Math.floor(maxRows / 2);
    const showAll = nRows <= maxRows;
    const topCount = showAll ? nRows : half;
    const bottomCount = showAll ? 0 : half;

    const formatRow = (i: number): string[] => {
      const row: string[] = [String(this._index[i] ?? i)];
      for (const col of cols) {
        const val = this._data.get(col)?.[i];
        row.push(cellText(val));
      }
      return row;
    };

    // Build header + data rows
    const allRows: string[][] = [["", ...cols]];
    for (let i = 0; i < topCount; i++) allRows.push(formatRow(i));
    if (!showAll) {
      allRows.push(["...", ...cols.map(() => "...")]);
      for (let i = nRows - bottomCount; i < nRows; i++) allRows.push(formatRow(i));
    }

    // Calculate column widths
    const numCols = cols.length + 1;
    const widths = new Array<number>(numCols).fill(0);
    for (const row of allRows) {
      for (let c = 0; c < numCols; c++) {
        const cell = row[c] ?? "";
        if (cell.length > (widths[c] ?? 0)) {
          widths[c] = cell.length;
        }
      }
    }

    // Format each row
    const lines: string[] = [];
    for (const row of allRows) {
      const cells: string[] = [];
      for (let c = 0; c < numCols; c++) {
        cells.push((row[c] ?? "").padStart(widths[c] ?? 0));
      }
      lines.push(cells.join("  "));
    }

    return lines.join("\n");
  }

  /**
   * Element-wise absolute value for all numeric columns.
   * Non-numeric values are kept unchanged.
   *
   * @returns New DataFrame
   */
  abs(): DataFrame {
    const data: DataFrameData = newColumnMap();
    for (const col of this._columns) {
      const colData = this._data.get(col) ?? [];
      data[col] = colData.map((v) => (typeof v === "number" ? Math.abs(v) : v));
    }
    return new DataFrame(data, {
      columns: this._columns,
      index: this._index,
      copy: false,
    });
  }

  /**
   * Round numeric columns to the given number of decimal places.
   *
   * Ties round to the nearest even value (`round(0.5)` is 0, `round(2.5)` is 2),
   * as in NumPy and pandas. A negative `decimals` rounds to tens, hundreds, and so on.
   * Non-numeric values are kept unchanged.
   *
   * @param decimals - Number of decimal places, may be negative (default: 0)
   * @returns New DataFrame
   * @throws {InvalidParameterError} If `decimals` is not an integer
   */
  round(decimals = 0): DataFrame {
    if (!Number.isInteger(decimals)) {
      throw new InvalidParameterError("decimals must be an integer", "decimals", decimals);
    }
    const factor = 10 ** Math.abs(decimals);
    const roundValue = (v: number): number => {
      if (!Number.isFinite(v)) return v;
      if (decimals >= 0) {
        const scaled = v * factor;
        // Overflowing here means the value has no fractional digits left to round.
        return Number.isFinite(scaled) ? roundHalfEven(scaled) / factor : v;
      }
      // A factor that overflows is far above any double, so everything rounds to zero.
      return Number.isFinite(factor) ? roundHalfEven(v / factor) * factor : 0 * v;
    };
    const data: DataFrameData = newColumnMap();
    for (const col of this._columns) {
      const colData = this._data.get(col) ?? [];
      data[col] = colData.map((v) => (typeof v === "number" ? roundValue(v) : v));
    }
    return new DataFrame(data, {
      columns: this._columns,
      index: this._index,
      copy: false,
    });
  }

  /**
   * Count distinct values per column. Null, undefined and NaN are not counted,
   * as in pandas.
   *
   * @returns Series of counts indexed by column name
   */
  nunique(): Series {
    const counts: number[] = [];
    for (const col of this._columns) {
      const colData = this._data.get(col) ?? [];
      // Primitives are distinct as themselves; objects (dates, arrays) by content.
      const primitives = new Set<unknown>();
      const objects = new Set<string>();
      for (const v of colData) {
        if (isMissing(v)) continue;
        if (typeof v === "object") objects.add(createKey(v));
        else primitives.add(v);
      }
      counts.push(primitives.size + objects.size);
    }
    return new Series(counts, { index: this._columns });
  }

  /**
   * Rows with the n largest or smallest values of a column. Ties keep their
   * original order. NaN and null values are never selected.
   *
   * @private
   */
  private nExtreme(n: number, column: string, largest: boolean): DataFrame {
    const colData = this._data.get(column);
    if (!colData) throw new IndexError(`Column '${column}' not found`);
    if (!Number.isInteger(n) || n < 0) {
      throw new InvalidParameterError("n must be a non-negative integer", "n", n);
    }
    const positions: number[] = [];
    for (let i = 0; i < colData.length; i++) {
      const v = colData[i];
      if (isOrderedNumber(v)) {
        positions.push(i);
      } else if (!isMissing(v)) {
        throw new DataValidationError(`Column '${column}' must be numeric`);
      }
    }
    positions.sort((a, b) => {
      const cmp = compareNumbers(colData[a] as number, colData[b] as number);
      return largest ? -cmp : cmp;
    });
    const selected = positions.slice(0, n);
    const data: DataFrameData = newColumnMap();
    for (const col of this._columns) {
      const cd = this._data.get(col) ?? [];
      data[col] = selected.map((i) => cd[i]);
    }
    return new DataFrame(data, {
      columns: this._columns,
      index: selected.map((i) => this._index[i] as string | number),
      copy: false,
    });
  }

  /**
   * Return the n rows with the largest values in a column, largest first.
   *
   * NaN and null values are skipped; ties keep their original order.
   *
   * @param n - Number of rows to return
   * @param column - Numeric column to rank by
   * @returns New DataFrame with at most n rows
   * @throws {IndexError} If the column does not exist
   * @throws {DataValidationError} If the column holds non-numeric values
   */
  nlargest(n: number, column: string): DataFrame {
    return this.nExtreme(n, column, true);
  }

  /**
   * Return the n rows with the smallest values in a column, smallest first.
   *
   * NaN and null values are skipped; ties keep their original order.
   *
   * @param n - Number of rows to return
   * @param column - Numeric column to rank by
   * @returns New DataFrame with at most n rows
   * @throws {IndexError} If the column does not exist
   * @throws {DataValidationError} If the column holds non-numeric values
   */
  nsmallest(n: number, column: string): DataFrame {
    return this.nExtreme(n, column, false);
  }

  /**
   * Index label of the first minimum value in each column. NaN and non-numeric
   * values are skipped; a column without any number gives null.
   *
   * @returns Series of row labels indexed by column name
   */
  idxmin(): Series {
    const result: (string | number | null)[] = [];
    for (const col of this._columns) {
      const colData = this._data.get(col) ?? [];
      let minVal = 0;
      let minIdx: string | number | null = null;
      for (let i = 0; i < colData.length; i++) {
        const v = colData[i];
        if (isOrderedNumber(v) && (minIdx === null || v < minVal)) {
          minVal = v;
          minIdx = this._index[i] as string | number;
        }
      }
      result.push(minIdx);
    }
    return new Series(result, { index: this._columns });
  }

  /**
   * Index label of the first maximum value in each column. NaN and non-numeric
   * values are skipped; a column without any number gives null.
   *
   * @returns Series of row labels indexed by column name
   */
  idxmax(): Series {
    const result: (string | number | null)[] = [];
    for (const col of this._columns) {
      const colData = this._data.get(col) ?? [];
      let maxVal = 0;
      let maxIdx: string | number | null = null;
      for (let i = 0; i < colData.length; i++) {
        const v = colData[i];
        if (isOrderedNumber(v) && (maxIdx === null || v > maxVal)) {
          maxVal = v;
          maxIdx = this._index[i] as string | number;
        }
      }
      result.push(maxIdx);
    }
    return new Series(result, { index: this._columns });
  }

  /**
   * Keep the rows whose value in `column` lies between `left` and `right`.
   *
   * Non-numeric and NaN values are never kept.
   *
   * @param column - Column to test
   * @param left - Lower bound
   * @param right - Upper bound
   * @param inclusive - Which bounds are included: `true`/`"both"` (default),
   *   `false`/`"neither"`, `"left"` or `"right"`
   * @returns New DataFrame with the matching rows
   * @throws {IndexError} If the column does not exist
   */
  between(
    column: string,
    left: number,
    right: number,
    inclusive: boolean | "both" | "neither" | "left" | "right" = true
  ): DataFrame {
    const colData = this._data.get(column);
    if (!colData) throw new IndexError(`Column '${column}' not found`);
    const mode = inclusive === true ? "both" : inclusive === false ? "neither" : inclusive;
    if (mode !== "both" && mode !== "neither" && mode !== "left" && mode !== "right") {
      throw new InvalidParameterError(
        'inclusive must be a boolean or one of "both", "neither", "left", "right"',
        "inclusive",
        inclusive
      );
    }
    const includeLeft = mode === "both" || mode === "left";
    const includeRight = mode === "both" || mode === "right";
    const selected: number[] = [];
    for (let i = 0; i < colData.length; i++) {
      const v = colData[i];
      if (typeof v !== "number") continue;
      const aboveLeft = includeLeft ? v >= left : v > left;
      const belowRight = includeRight ? v <= right : v < right;
      if (aboveLeft && belowRight) selected.push(i);
    }
    const data: DataFrameData = newColumnMap();
    for (const col of this._columns) {
      const cd = this._data.get(col) ?? [];
      data[col] = selected.map((i) => cd[i]);
    }
    return new DataFrame(data, {
      columns: this._columns,
      index: selected.map((i) => this._index[i] as string | number),
      copy: false,
    });
  }

  /**
   * Add or modify columns functionally.
   *
   * Each value can be an array (one entry per row), a function that receives a
   * row object and the row position and returns the cell value, or a scalar that
   * is repeated for every row. Functions see the columns of the original
   * DataFrame, not columns assigned earlier in the same call.
   *
   * @param columns - Map of column name to array, row function or scalar
   * @returns New DataFrame with the columns added or replaced
   * @throws {DataValidationError} If an array has the wrong length
   */
  assign(
    columns: Record<
      string,
      | unknown[]
      | ((row: Record<string, unknown>, i: number) => unknown)
      | string
      | number
      | boolean
      | bigint
      | Date
      | null
    >
  ): DataFrame {
    const nRows = this._index.length;
    const data: DataFrameData = newColumnMap();
    for (const col of this._columns) {
      data[col] = [...(this._data.get(col) ?? [])];
    }
    const colNames = [...this._columns];
    const sources = this._columns.map((col) => this._data.get(col) ?? []);
    for (const [name, valueOrFn] of Object.entries(columns)) {
      if (typeof valueOrFn === "function") {
        const arr: unknown[] = new Array(nRows);
        for (let i = 0; i < nRows; i++) {
          const row: Record<string, unknown> = {};
          for (let c = 0; c < this._columns.length; c++) {
            row[this._columns[c] as string] = (sources[c] as unknown[])[i];
          }
          arr[i] = valueOrFn(row, i);
        }
        data[name] = arr;
      } else if (Array.isArray(valueOrFn)) {
        if (valueOrFn.length !== nRows) {
          throw new DataValidationError(
            `Column '${name}' length (${valueOrFn.length}) must match row count (${nRows})`
          );
        }
        data[name] = [...valueOrFn];
      } else {
        data[name] = new Array(nRows).fill(valueOrFn);
      }
      if (!colNames.includes(name)) colNames.push(name);
    }
    return new DataFrame(data, { columns: colNames, index: this._index, copy: false });
  }

  /**
   * Resolves the where()/mask() condition into one boolean per cell lookup.
   *
   * @private
   */
  private conditionFn(
    cond: boolean[] | ((val: unknown, i: number) => boolean)
  ): (val: unknown, i: number) => boolean {
    if (typeof cond === "function") return cond;
    if (!Array.isArray(cond) || cond.length !== this._index.length) {
      throw new InvalidParameterError(
        `cond must be a function or an array with one entry per row (${this._index.length})`,
        "cond",
        cond
      );
    }
    return (_val, i) => cond[i] === true;
  }

  /**
   * Replace values where condition is false with other.
   *
   * @param cond - Function called with each value and its row position, or an
   *   array with one boolean per row (applied to every column)
   * @param other - Replacement value (default: null)
   * @returns New DataFrame
   * @throws {InvalidParameterError} If an array condition does not have one entry per row
   */
  where(
    cond: boolean[] | ((val: unknown, i: number) => boolean),
    other: unknown = null
  ): DataFrame {
    const test = this.conditionFn(cond);
    const data: DataFrameData = newColumnMap();
    for (const col of this._columns) {
      const colData = this._data.get(col) ?? [];
      data[col] = colData.map((v, i) => (test(v, i) ? v : other));
    }
    return new DataFrame(data, {
      columns: this._columns,
      index: this._index,
      copy: false,
    });
  }

  /**
   * Replace values where condition is true with other. Inverse of where().
   *
   * @param cond - Function called with each value and its row position, or an
   *   array with one boolean per row (applied to every column)
   * @param other - Replacement value (default: null)
   * @returns New DataFrame
   * @throws {InvalidParameterError} If an array condition does not have one entry per row
   */
  mask(cond: boolean[] | ((val: unknown, i: number) => boolean), other: unknown = null): DataFrame {
    const test = this.conditionFn(cond);
    const data: DataFrameData = newColumnMap();
    for (const col of this._columns) {
      const colData = this._data.get(col) ?? [];
      data[col] = colData.map((v, i) => (test(v, i) ? other : v));
    }
    return new DataFrame(data, {
      columns: this._columns,
      index: this._index,
      copy: false,
    });
  }

  /**
   * Cast columns to a target type.
   *
   * Missing values (null, undefined, NaN) stay missing: they become NaN for
   * `"number"` and null for `"string"` and `"boolean"`. Strings convert to
   * numbers like `Number()` does, except that empty or blank strings give NaN.
   * The strings `"true"` and `"false"` (any case) convert to booleans.
   *
   * @param column - Column to cast, or an object mapping column names to types
   * @param dtype - Target type when `column` is a name
   * @returns New DataFrame
   * @throws {IndexError} If a column does not exist
   * @throws {InvalidParameterError} If the target type is unknown
   *
   * @example
   * ```ts
   * df.astype("age", "number");
   * df.astype({ age: "number", name: "string" });
   * ```
   */
  astype(column: string, dtype: "number" | "string" | "boolean"): DataFrame;
  astype(dtypes: Record<string, "number" | "string" | "boolean">): DataFrame;
  astype(
    columnOrMap: string | Record<string, "number" | "string" | "boolean">,
    dtype?: "number" | "string" | "boolean"
  ): DataFrame {
    const plan: [string, "number" | "string" | "boolean"][] =
      typeof columnOrMap === "string"
        ? [[columnOrMap, dtype as "number" | "string" | "boolean"]]
        : Object.entries(columnOrMap);

    for (const [col, target] of plan) {
      if (!this._data.has(col)) throw new IndexError(`Column '${col}' not found`);
      if (target !== "number" && target !== "string" && target !== "boolean") {
        throw new InvalidParameterError(
          'dtype must be "number", "string" or "boolean"',
          "dtype",
          target
        );
      }
    }

    const toNumber = (v: unknown): number => {
      if (isMissing(v)) return NaN;
      if (typeof v === "number") return v;
      if (typeof v === "string") return v.trim() === "" ? NaN : Number(v);
      if (typeof v === "boolean") return v ? 1 : 0;
      if (typeof v === "bigint") return Number(v);
      if (v instanceof Date) return v.getTime();
      return Number(v);
    };
    const toText = (v: unknown): string | null => (isMissing(v) ? null : String(v));
    const toBool = (v: unknown): boolean | null => {
      if (isMissing(v)) return null;
      if (typeof v === "string") {
        const t = v.trim().toLowerCase();
        if (t === "true") return true;
        if (t === "false") return false;
      }
      return Boolean(v);
    };

    const targets = new Map(plan);
    const data: DataFrameData = newColumnMap();
    for (const col of this._columns) {
      const colData = this._data.get(col) ?? [];
      const target = targets.get(col);
      if (target === "number") data[col] = colData.map(toNumber);
      else if (target === "string") data[col] = colData.map(toText);
      else if (target === "boolean") data[col] = colData.map(toBool);
      else data[col] = [...colData];
    }
    return new DataFrame(data, {
      columns: this._columns,
      index: this._index,
      copy: false,
    });
  }

  /**
   * Summarize the DataFrame as text: row count, and for every column the number
   * of non-null values and a dtype guess taken from its first non-null value.
   *
   * @returns Multi-line summary string
   */
  info(): string {
    const lines: string[] = [];
    lines.push(`<DataFrame>`);
    lines.push(`RangeIndex: ${this._index.length} entries`);
    lines.push(`Data columns (total ${this._columns.length} columns):`);
    lines.push(` #   Column  Non-Null Count  Dtype`);
    lines.push(`---  ------  --------------  -----`);
    for (let i = 0; i < this._columns.length; i++) {
      const col = this._columns[i] as string;
      const colData = this._data.get(col) ?? [];
      const nonNull = colData.filter((v) => v !== null && v !== undefined).length;
      let dtype = "object";
      const first = colData.find((v) => v !== null && v !== undefined);
      if (typeof first === "number") dtype = "float64";
      else if (typeof first === "boolean") dtype = "bool";
      else if (typeof first === "string") dtype = "string";
      lines.push(` ${i}   ${col.padEnd(8)}${String(nonNull).padStart(4)} non-null    ${dtype}`);
    }
    return lines.join("\n");
  }

  /**
   * Whether any value is truthy.
   *
   * @param axis - 0 or "index" tests each column and returns a Series indexed by
   *   column name; 1 or "columns" tests each row and returns one boolean per row
   * @returns Series (axis 0) or array of booleans (axis 1)
   */
  any(axis: Axis = 0): Series | boolean[] {
    const ax = normalizeAxis(axis, 2);
    if (ax === 0) {
      const result: boolean[] = [];
      for (const col of this._columns) {
        const colData = this._data.get(col) ?? [];
        result.push(colData.some((v) => Boolean(v)));
      }
      return new Series(result, { index: this._columns });
    }
    const arrays = this._columns.map((col) => this._data.get(col) ?? []);
    return this._index.map((_, i) => arrays.some((arr) => Boolean(arr[i])));
  }

  /**
   * Whether all values are truthy.
   *
   * @param axis - 0 or "index" tests each column and returns a Series indexed by
   *   column name; 1 or "columns" tests each row and returns one boolean per row
   * @returns Series (axis 0) or array of booleans (axis 1)
   */
  all(axis: Axis = 0): Series | boolean[] {
    const ax = normalizeAxis(axis, 2);
    if (ax === 0) {
      const result: boolean[] = [];
      for (const col of this._columns) {
        const colData = this._data.get(col) ?? [];
        result.push(colData.every((v) => Boolean(v)));
      }
      return new Series(result, { index: this._columns });
    }
    const arrays = this._columns.map((col) => this._data.get(col) ?? []);
    return this._index.map((_, i) => arrays.every((arr) => Boolean(arr[i])));
  }

  /**
   * Count occurrences of unique value combinations across specified columns.
   *
   * Groups are listed by descending count; equal counts keep the order of first appearance.
   * Unlike pandas, missing values are counted as their own group by default (Deepbox 1.0
   * behavior); pass `{ dropna: true }` to leave out every row that has a missing value in one of
   * the counted columns. Pass `{ normalize: true }` for relative frequencies: the count column
   * is then called `proportion`. `sort: false` keeps the order of first appearance and
   * `ascending: true` lists the rarest combinations first.
   *
   * @param args - Columns to combine (default: all columns), optionally followed by one
   *   options object (see {@link ValueCountsOptions})
   * @returns DataFrame with the combination values and a `count` (or `proportion`) column
   * @throws {IndexError} If a column does not exist
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: ["x", "x", "y", null] });
   * df.valueCounts("a");                                 // x: 2, y: 1, null: 1
   * df.valueCounts("a", { dropna: true, normalize: true }); // x: 2/3, y: 1/3
   * ```
   */
  valueCounts(...args: (string | ValueCountsOptions)[]): DataFrame {
    const last = args[args.length - 1];
    const options: ValueCountsOptions = typeof last === "object" && last !== null ? last : {};
    const columns = args.filter((a): a is string => typeof a === "string");
    if (columns.length !== args.length - (options === last ? 1 : 0)) {
      throw new InvalidParameterError(
        "valueCounts takes column names followed by at most one options object",
        "columns",
        args
      );
    }
    const cols = columns.length > 0 ? columns : this._columns;
    for (const c of cols) {
      if (!this._data.has(c)) throw new IndexError(`Column '${c}' not found`);
    }
    const dropna = options.dropna ?? false;
    const normalize = options.normalize ?? false;
    const sort = options.sort ?? true;
    const ascending = options.ascending ?? false;
    for (const [name, flag] of [
      ["dropna", dropna],
      ["normalize", normalize],
      ["sort", sort],
      ["ascending", ascending],
    ] as const) {
      if (typeof flag !== "boolean") {
        throw new InvalidParameterError(`${name} must be a boolean`, name, flag);
      }
    }
    const arrays = cols.map((c) => this._data.get(c) ?? []);
    const groups = new Map<string, { values: unknown[]; count: number }>();
    let counted = 0;
    for (let i = 0; i < this._index.length; i++) {
      const vals = arrays.map((arr) => arr[i]);
      if (dropna && vals.some(isMissing)) continue;
      counted++;
      const key = createKey(vals);
      const group = groups.get(key);
      if (group === undefined) groups.set(key, { values: vals, count: 1 });
      else group.count++;
    }
    const listed = [...groups.values()];
    // Array.prototype.sort is stable, so ties keep first-appearance order.
    if (sort) listed.sort((a, b) => (ascending ? a.count - b.count : b.count - a.count));
    const data: DataFrameData = newColumnMap();
    for (let j = 0; j < cols.length; j++) {
      data[cols[j] as string] = listed.map((g) => g.values[j]);
    }
    const valueName = normalize ? "proportion" : "count";
    data[valueName] = listed.map((g) => (normalize ? g.count / counted : g.count));
    return new DataFrame(data, { columns: [...cols, valueName], copy: false });
  }

  /**
   * Same as {@link DataFrame.valueCounts}.
   *
   * @deprecated Prefer {@link DataFrame.valueCounts}.
   */
  value_counts(...args: (string | ValueCountsOptions)[]): DataFrame {
    return this.valueCounts(...args);
  }

  /**
   * Filter rows using a simple expression string.
   *
   * Supports comparisons of the form `column op value`, joined with `and` / `or`
   * (also `&`, `&&`, `|`, `||`). `and` binds tighter than `or`, as in Python.
   * Operators: `==`, `!=`, `>`, `>=`, `<`, `<=`. The value is a number, a quoted
   * string, `true`, `false`, `null` / `NaN` (matches missing values with `==` and
   * `!=`), or another column name. Column names with spaces or symbols can be
   * written in backticks. Ordering comparisons (`>`, `<`, ...) hold only for two
   * numbers or two strings. Parentheses are not supported.
   *
   * @param expr - Query expression string, e.g. `"age > 30 and salary > 50000"`
   * @returns New DataFrame with the rows that match
   * @throws {InvalidParameterError} If the expression cannot be parsed
   * @throws {IndexError} If a referenced column does not exist
   */
  query(expr: string): DataFrame {
    if (typeof expr !== "string") {
      throw new InvalidParameterError("expr must be a string", "expr", expr);
    }

    // Split on top-level connectors, ignoring any inside quotes.
    const parts: string[] = [];
    const connectors: ("and" | "or")[] = [];
    const connectorRe = /\s+(and|or)(?=\s|$)\s*|\s*(&&|\|\||&|\|)\s*/iy;
    let start = 0;
    let quote: string | null = null;
    for (let i = 0; i < expr.length; i++) {
      const ch = expr[i] as string;
      if (quote !== null) {
        if (ch === quote) quote = null;
        continue;
      }
      if (ch === "'" || ch === '"' || ch === "`") {
        quote = ch;
        continue;
      }
      connectorRe.lastIndex = i;
      const m = connectorRe.exec(expr);
      if (m) {
        parts.push(expr.slice(start, i));
        const word = (m[1] ?? m[2] ?? "").toLowerCase();
        connectors.push(word === "and" || word === "&" || word === "&&" ? "and" : "or");
        start = connectorRe.lastIndex;
        i = start - 1;
      }
    }
    if (quote !== null) {
      throw new InvalidParameterError(`Unterminated quote in query expression`, "expr", expr);
    }
    parts.push(expr.slice(start));

    const columnOf = (name: string): unknown[] => {
      const arr = this._data.get(name);
      if (arr === undefined) throw new IndexError(`Column '${name}' not found`);
      return arr;
    };
    const unquoteName = (name: string): string =>
      name.startsWith("`") && name.endsWith("`") ? name.slice(1, -1) : name;

    type Test = (i: number) => boolean;
    const compare = (op: string, cell: unknown, rhs: unknown): boolean => {
      switch (op) {
        case "==":
          return cell === rhs;
        case "!=":
          return cell !== rhs;
        case ">":
        case ">=":
        case "<":
        case "<=": {
          const comparable =
            (typeof cell === "number" && typeof rhs === "number") ||
            (typeof cell === "string" && typeof rhs === "string");
          if (!comparable) return false;
          const a = cell as number | string;
          const b = rhs as number | string;
          if (op === ">") return a > b;
          if (op === ">=") return a >= b;
          if (op === "<") return a < b;
          return a <= b;
        }
        default:
          return false;
      }
    };

    const tests: Test[] = parts.map((part) => {
      const t = part.trim();
      const match = t.match(/^(`[^`]+`|\w+)\s*(==|!=|>=|<=|>|<)\s*(.+)$/);
      if (!match) {
        throw new InvalidParameterError(`Invalid query expression: '${t}'`, "expr", expr);
      }
      const colArr = columnOf(unquoteName(match[1] as string));
      const op = match[2] as string;
      const valStr = (match[3] as string).trim();

      if (
        valStr.length >= 2 &&
        ((valStr.startsWith("'") && valStr.endsWith("'")) ||
          (valStr.startsWith('"') && valStr.endsWith('"')))
      ) {
        const text = valStr.slice(1, -1);
        return (i) => compare(op, colArr[i], text);
      }
      if (valStr === "true" || valStr === "false") {
        const flag = valStr === "true";
        return (i) => compare(op, colArr[i], flag);
      }
      if (valStr === "null" || valStr === "NaN") {
        if (op !== "==" && op !== "!=") return () => false;
        return op === "==" ? (i) => isMissing(colArr[i]) : (i) => !isMissing(colArr[i]);
      }
      const refName = unquoteName(valStr);
      if ((/^[a-zA-Z_]\w*$/.test(valStr) || /^`[^`]+`$/.test(valStr)) && this._data.has(refName)) {
        // Bare identifier matching a column name: column-vs-column comparison
        // (e.g. `a > b`), not the string literal "b".
        const rhsArr = columnOf(refName);
        return (i) => compare(op, colArr[i], rhsArr[i]);
      }
      const num = Number(valStr);
      const rhs: unknown = Number.isNaN(num) ? valStr : num;
      return (i) => compare(op, colArr[i], rhs);
    });

    // OR of AND-groups: connectors[k] joins tests[k] and tests[k + 1].
    const groups: Test[][] = [[tests[0] as Test]];
    for (let k = 1; k < tests.length; k++) {
      if (connectors[k - 1] === "and") (groups[groups.length - 1] as Test[]).push(tests[k] as Test);
      else groups.push([tests[k] as Test]);
    }

    const keep: number[] = [];
    for (let i = 0; i < this._index.length; i++) {
      if (groups.some((group) => group.every((test) => test(i)))) keep.push(i);
    }
    const newData: DataFrameData = newColumnMap();
    for (const col of this._columns) {
      const src = this._data.get(col) ?? [];
      newData[col] = keep.map((i) => src[i]);
    }
    return new DataFrame(newData, {
      columns: this._columns,
      index: keep.map((i) => this._index[i] as string | number),
      copy: false,
    });
  }

  /**
   * Return approximate memory usage in bytes per column.
   *
   * The figure is an estimate (8 bytes per number or other value, 2 bytes per
   * string character, 4 bytes per boolean), not a measurement of the JS heap.
   *
   * @returns DataFrame with `column` and `bytes` columns
   */
  memoryUsage(): DataFrame {
    const cols: string[] = [];
    const bytes: number[] = [];
    for (const col of this._columns) {
      const data = this._data.get(col) ?? [];
      let size = 0;
      for (const v of data) {
        if (typeof v === "number") size += 8;
        else if (typeof v === "string") size += v.length * 2;
        else if (typeof v === "boolean") size += 4;
        else size += 8;
      }
      cols.push(col);
      bytes.push(size);
    }
    return new DataFrame({ column: cols, bytes }, { columns: ["column", "bytes"] });
  }

  /**
   * Same as {@link DataFrame.memoryUsage}.
   *
   * @deprecated Prefer {@link DataFrame.memoryUsage}.
   */
  memory_usage(): DataFrame {
    return this.memoryUsage();
  }

  /**
   * Fill NaN/null values by interpolation, like pandas' default settings.
   *
   * - `"linear"`: gaps between two numbers are filled on a straight line over
   *   the row positions; a gap after the last number repeats that number;
   *   a gap before the first number stays missing.
   * - `"nearest"`: gaps between two numbers take the closer neighbour (the
   *   earlier one on a tie); gaps before the first or after the last number stay
   *   missing.
   *
   * Only finite numbers act as neighbours; other values are left untouched.
   *
   * @param method - Interpolation method: 'linear' (default) or 'nearest'
   * @returns New DataFrame
   */
  interpolate(method: "linear" | "nearest" = "linear"): DataFrame {
    if (method !== "linear" && method !== "nearest") {
      throw new InvalidParameterError('method must be "linear" or "nearest"', "method", method);
    }
    const newData: DataFrameData = newColumnMap();
    for (const col of this._columns) {
      const data = this._data.get(col) ?? [];
      const result = [...data];

      const anchors: number[] = [];
      for (let i = 0; i < data.length; i++) {
        if (isValidNumber(data[i])) anchors.push(i);
      }

      // `next` walks the anchors in step with i, so the pass is O(n).
      let next = 0;
      for (let i = 0; i < data.length; i++) {
        if (!isMissing(data[i])) continue;
        while (next < anchors.length && (anchors[next] as number) < i) next++;
        const prevPos = next > 0 ? (anchors[next - 1] as number) : -1;
        const nextPos = next < anchors.length ? (anchors[next] as number) : -1;
        if (prevPos >= 0 && nextPos >= 0) {
          const prevVal = data[prevPos] as number;
          const nextVal = data[nextPos] as number;
          if (method === "linear") {
            const frac = (i - prevPos) / (nextPos - prevPos);
            result[i] = prevVal + frac * (nextVal - prevVal);
          } else {
            result[i] = i - prevPos <= nextPos - i ? prevVal : nextVal;
          }
        } else if (prevPos >= 0 && method === "linear") {
          result[i] = data[prevPos];
        }
      }
      newData[col] = result;
    }
    return new DataFrame(newData, {
      columns: this._columns,
      index: this._index,
      copy: false,
    });
  }

  /**
   * Pivot with aggregation (unlike pivot() which requires unique index/column pairs).
   *
   * Rows whose index or column value is missing are skipped. Row labels and
   * column names are sorted (numbers numerically, otherwise by string). Only
   * numeric values enter the aggregation; a cell without any gives null.
   *
   * @param options.index - Column to use as new index
   * @param options.columns - Column whose values become new columns
   * @param options.values - Column to aggregate
   * @param options.aggFunc - Aggregation function: 'mean', 'sum', 'count', 'min', 'max', 'median', 'first', 'last' (default: 'mean')
   * @returns New DataFrame
   * @throws {IndexError} If a column does not exist
   */
  pivotTable(options: {
    index: string;
    columns: string;
    values: string;
    aggFunc?: "mean" | "sum" | "count" | "min" | "max" | "median" | "first" | "last";
  }): DataFrame {
    const { index: idxCol, columns: colCol, values: valCol } = options;
    const aggFunc = options.aggFunc ?? "mean";
    for (const c of [idxCol, colCol, valCol]) {
      if (!this._data.has(c)) throw new IndexError(`Column '${c}' not found`);
    }
    if (!["mean", "sum", "count", "min", "max", "median", "first", "last"].includes(aggFunc)) {
      throw new InvalidParameterError(
        'aggFunc must be one of "mean", "sum", "count", "min", "max", "median", "first" or "last"',
        "aggFunc",
        aggFunc
      );
    }
    const idxData = this._data.get(idxCol) ?? [];
    const colData = this._data.get(colCol) ?? [];
    const valData = this._data.get(valCol) ?? [];

    // Distinct row labels / column names in order of appearance, with the
    // numeric values that fall into each (row, column) cell.
    const rowLabels: (string | number)[] = [];
    const colNames: string[] = [];
    const rowOf = new Map<string | number, number>();
    const colOf = new Map<string, number>();
    const cellsByRow: Map<number, number[]>[] = [];
    let nColsSeen = 0;

    for (let i = 0; i < this._index.length; i++) {
      const rk = idxData[i];
      const ck = colData[i];
      if (isMissing(rk) || isMissing(ck)) continue;
      const label = toIndexLabel(rk);
      const name = String(ck);
      let r = rowOf.get(label);
      if (r === undefined) {
        r = rowLabels.length;
        rowOf.set(label, r);
        rowLabels.push(label);
        cellsByRow.push(new Map());
      }
      let c = colOf.get(name);
      if (c === undefined) {
        c = nColsSeen++;
        colOf.set(name, c);
        colNames.push(name);
      }
      const rowCells = cellsByRow[r] as Map<number, number[]>;
      let cell = rowCells.get(c);
      if (cell === undefined) {
        cell = [];
        rowCells.set(c, cell);
      }
      const v = valData[i];
      if (isOrderedNumber(v)) cell.push(v);
    }

    const aggregate = (vals: number[]): unknown => {
      if (vals.length === 0) return null;
      switch (aggFunc) {
        case "sum":
          return compensatedSum(vals);
        case "count":
          return vals.length;
        case "min": {
          let m = vals[0] as number;
          for (const v of vals) if (v < m) m = v;
          return m;
        }
        case "max": {
          let m = vals[0] as number;
          for (const v of vals) if (v > m) m = v;
          return m;
        }
        case "median": {
          const sorted = Float64Array.from(vals).sort();
          return quantileSorted(sorted, 0.5);
        }
        case "first":
          return vals[0];
        case "last":
          return vals[vals.length - 1];
        default:
          return compensatedSum(vals) / vals.length;
      }
    };

    const rowOrder = rowLabels
      .map((_, r) => r)
      .sort((a, b) => compareLabels(rowLabels[a], rowLabels[b]));
    const colOrder = colNames
      .map((_, c) => c)
      .sort((a, b) => compareLabels(colNames[a], colNames[b]));

    const result: DataFrameData = newColumnMap();
    for (const c of colOrder) {
      result[colNames[c] as string] = rowOrder.map((r) => {
        const cell = (cellsByRow[r] as Map<number, number[]>).get(c);
        return cell === undefined ? null : aggregate(cell);
      });
    }

    return new DataFrame(result, {
      columns: colOrder.map((c) => colNames[c] as string),
      index: rowOrder.map((r) => rowLabels[r] as string | number),
      copy: false,
    });
  }

  /**
   * Same as {@link DataFrame.pivotTable}.
   *
   * @deprecated Prefer {@link DataFrame.pivotTable}.
   */
  pivot_table(options: {
    index: string;
    columns: string;
    values: string;
    aggFunc?: "mean" | "sum" | "count" | "min" | "max" | "median" | "first" | "last";
  }): DataFrame {
    return this.pivotTable(options);
  }

  /**
   * Expanding (cumulative) window calculations.
   *
   * @param minPeriods - Minimum number of observations to produce a result (default: 1)
   * @returns Expanding object with mean(), sum(), std(), var(), min() and max()
   * @throws {InvalidParameterError} If minPeriods is not a non-negative integer
   */
  expanding(minPeriods = 1): Expanding {
    if (!Number.isInteger(minPeriods) || minPeriods < 0) {
      throw new InvalidParameterError(
        "minPeriods must be a non-negative integer",
        "minPeriods",
        minPeriods
      );
    }
    return new Expanding(this, minPeriods);
  }

  /**
   * Exponentially weighted moving calculations.
   *
   * The decay is given by one of `alpha`, `span`, `com` or `halflife`
   * (checked in that order): alpha = 2 / (span + 1) = 1 / (1 + com) =
   * 1 - exp(-ln(2) / halflife).
   *
   * Unlike pandas, `adjust` defaults to `false` and `bias` to `true`, so
   * `mean()` follows the recursion `y[t] = alpha * x[t] + (1 - alpha) * y[t-1]`.
   * Pass `{ adjust: true, bias: false }` for pandas' defaults.
   *
   * With `adjust: false` and `ignoreNa: false`, pandas 3.0 returns a different mean
   * after a gap of missing values when `alpha` is exactly 0.5 (for `[1, NaN, 7]` it
   * gives 5.5). That result contradicts pandas' own documented weights; Deepbox follows
   * the documented weights and gives 5, as pandas does for every other alpha tested.
   *
   * @param options.span - Decay span, at least 1
   * @param options.alpha - Smoothing factor in (0, 1]
   * @param options.com - Center of mass, at least 0
   * @param options.halflife - Half-life, greater than 0
   * @param options.adjust - Divide by the decaying weight sum so early values are not
   *   over-weighted (default: false)
   * @param options.bias - Use the biased variance estimate for std() and var() (default: true)
   * @param options.ignoreNa - Ignore missing values when computing weights, so a gap does
   *   not age earlier values (default: false)
   * @returns EWM object with mean(), std() and var()
   * @throws {InvalidParameterError} If no decay is given or a value is out of range
   */
  ewm(options: {
    span?: number;
    alpha?: number;
    com?: number;
    halflife?: number;
    adjust?: boolean;
    bias?: boolean;
    ignoreNa?: boolean;
  }): EWM {
    let alpha: number;
    if (options.alpha !== undefined) {
      alpha = options.alpha;
    } else if (options.span !== undefined) {
      if (!Number.isFinite(options.span) || options.span < 1) {
        throw new InvalidParameterError("span must be a number >= 1", "span", options.span);
      }
      alpha = 2 / (options.span + 1);
    } else if (options.com !== undefined) {
      if (!Number.isFinite(options.com) || options.com < 0) {
        throw new InvalidParameterError("com must be a number >= 0", "com", options.com);
      }
      alpha = 1 / (1 + options.com);
    } else if (options.halflife !== undefined) {
      if (!Number.isFinite(options.halflife) || options.halflife <= 0) {
        throw new InvalidParameterError(
          "halflife must be a number > 0",
          "halflife",
          options.halflife
        );
      }
      alpha = 1 - Math.exp(-Math.LN2 / options.halflife);
    } else {
      throw new InvalidParameterError(
        "ewm requires span, alpha, com or halflife",
        "options",
        options
      );
    }
    if (!(alpha > 0 && alpha <= 1)) {
      throw new InvalidParameterError("alpha must be in (0, 1]", "alpha", alpha);
    }
    return new EWM(this, alpha, {
      adjust: options.adjust ?? false,
      bias: options.bias ?? true,
      ignoreNa: options.ignoreNa ?? false,
    });
  }

  /**
   * Return a deep copy of the DataFrame.
   *
   * All column data and index labels are copied so that mutations to the
   * copy do not affect the original and vice-versa.
   *
   * @returns A new DataFrame that is a deep copy of this one
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, 2], b: [3, 4] });
   * const df2 = df.copy();
   * ```
   */
  copy(): DataFrame {
    const data: DataFrameData = newColumnMap();
    for (const col of this._columns) {
      const colData = this._data.get(col);
      data[col] = colData ? [...colData] : [];
    }
    return new DataFrame(data, {
      columns: this._columns,
      index: [...this._index],
      copy: false,
    });
  }

  /**
   * Test whether each element is contained in the given values.
   *
   * Returns a DataFrame of booleans indicating whether each element
   * is found in the provided iterable of values.
   *
   * @param values - Values to test for membership
   * @returns DataFrame of booleans with same shape
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, 2, 3], b: [4, 5, 6] });
   * df.isin([2, 4]);
   * // DataFrame({ a: [false, true, false], b: [true, false, false] })
   * ```
   */
  isin(values: readonly unknown[]): DataFrame {
    const valueSet = new Set(values);
    const data: DataFrameData = newColumnMap();
    for (const col of this._columns) {
      const colData = this._data.get(col);
      data[col] = colData ? colData.map((v) => valueSet.has(v)) : [];
    }
    return new DataFrame(data, {
      columns: this._columns,
      index: this._index,
      copy: false,
    });
  }

  /**
   * Apply a function to each column (axis=0) or each row (axis=1),
   * producing a result with the same shape as the input.
   *
   * Unlike {@link apply}, `transform` enforces that the output has
   * the same number of elements as the input along the given axis.
   *
   * @param fn - Function to apply to each Series
   * @param axis - 0 = columns, 1 = rows
   * @returns New DataFrame with same shape
   * @throws {DataValidationError} If the result shape does not match the input
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, 2, 3], b: [4, 5, 6] });
   * df.transform(series => series.map(x => Number(x) * 2));
   * ```
   */
  transform(fn: (series: Series<unknown>) => Series<unknown>, axis: Axis = 0): DataFrame {
    const ax = normalizeAxis(axis, 2);
    if (ax === 0) {
      const newData: DataFrameData = newColumnMap();
      for (const col of this._columns) {
        const series = this.get(col);
        const result = fn(series);
        if (!(result instanceof Series)) {
          throw new DataValidationError("transform function must return a Series when axis=0");
        }
        if (result.data.length !== this._index.length) {
          throw new DataValidationError(
            `transform: result length (${result.data.length}) must match input length (${this._index.length})`
          );
        }
        newData[col] = [...result.data];
      }
      return new DataFrame(newData, {
        columns: [...this._columns],
        index: [...this._index],
      });
    }
    // axis === 1: apply to each row
    const newData: DataFrameData = newColumnMap();
    for (const col of this._columns) {
      newData[col] = [];
    }
    for (let i = 0; i < this._index.length; i++) {
      const rowValues: unknown[] = [];
      for (const col of this._columns) {
        rowValues.push(this._data.get(col)?.[i]);
      }
      const rowSeries = new Series(rowValues, {
        name: "row",
        index: this._columns,
        copy: false,
      });
      const result = fn(rowSeries);
      if (!(result instanceof Series)) {
        throw new DataValidationError("transform function must return a Series when axis=1");
      }
      if (result.data.length !== this._columns.length) {
        throw new DataValidationError(
          `transform: result length (${result.data.length}) must match column count (${this._columns.length})`
        );
      }
      for (let c = 0; c < this._columns.length; c++) {
        const col = this._columns[c];
        if (col !== undefined) {
          (newData[col] as unknown[]).push(result.data[c]);
        }
      }
    }
    return new DataFrame(newData, {
      columns: [...this._columns],
      index: [...this._index],
    });
  }

  /**
   * Evaluate a string expression on the DataFrame, producing a new column or filtering rows.
   *
   * Supports arithmetic (`+`, `-`, `*`, `/`, `%`, `**`), comparisons (`==`, `!=`,
   * `<`, `<=`, `>`, `>=`), logic (`and`, `or`, `not`, `&`, `|`, `!`) and
   * parentheses over column names and numbers; see the precedence rules in the
   * pandas `eval` documentation, which this follows. Column values are read as
   * numbers (booleans as 1 and 0, null and undefined as NaN). Names that are not
   * columns raise an error.
   *
   * An expression of the form `"c = a + b"` creates (or replaces) a column; any
   * other expression filters the rows where it is truthy.
   *
   * @param expr - Expression string
   * @returns New DataFrame with the result
   * @throws {InvalidParameterError} If the expression cannot be parsed or names an unknown column
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, 2, 3], b: [4, 5, 6] });
   * df.eval('c = a + b');  // adds column c = [5, 7, 9]
   * df.eval('a > 1');      // filters to rows where a > 1
   * df.eval('a > 1 and b < 6');  // filters to the middle row
   * ```
   */
  eval(expr: string): DataFrame {
    const trimmed = expr.trim();

    // Check if it's an assignment: "newCol = expression".
    // The `=` must be a lone assignment, NOT part of ==, >=, <=, or !=, the
    // negative lookahead/lookbehind prevents `eval("a == 3")` from being
    // parsed as `a = (= 3)` and silently overwriting column `a`.
    const assignMatch = trimmed.match(/^([a-zA-Z_]\w*)\s*(?<![=<>!])=(?!=)\s*(.+)$/);
    if (assignMatch) {
      const newColName = assignMatch[1]!;
      const rhsExpr = assignMatch[2]!;
      const values = this.evalExprPerRow(rhsExpr);
      const data: DataFrameData = newColumnMap();
      for (const col of this._columns) {
        data[col] = [...(this._data.get(col) ?? [])];
      }
      data[newColName] = values;
      const cols = this._columns.includes(newColName)
        ? [...this._columns]
        : [...this._columns, newColName];
      return new DataFrame(data, { columns: cols, index: [...this._index] });
    }

    // Otherwise treat as a boolean filter expression
    const mask = this.evalExprPerRow(trimmed);
    const data: DataFrameData = newColumnMap();
    for (const col of this._columns) data[col] = [];
    const newIndex: (string | number)[] = [];
    for (let i = 0; i < this._index.length; i++) {
      if (mask[i]) {
        for (const col of this._columns) {
          (data[col] as unknown[]).push(this._data.get(col)?.[i]);
        }
        newIndex.push(this._index[i]!);
      }
    }
    return new DataFrame(data, {
      columns: [...this._columns],
      index: newIndex,
    });
  }

  /**
   * Compiles an eval() expression once and evaluates it for every row.
   *
   * Grammar, loosest to tightest: `or` / `|` / `||`, `and` / `&` / `&&`,
   * `not` / `!`, one comparison (`==`, `!=`, `<`, `<=`, `>`, `>=`), `+ -`,
   * `* / %`, unary `-` / `+`, `**` (right-associative, binds tighter than a
   * unary minus on its left, so `-a ** 2` is `-(a ** 2)`), then numbers, column
   * names (or backticked names), `true`, `false`, `NaN`, `Infinity` and
   * parentheses. Booleans count as 1 and 0 in arithmetic, and any name that is
   * neither a column nor one of those constants is an error.
   *
   * @private
   */
  private evalExprPerRow(expr: string): unknown[] {
    const tokenRe =
      /\s*(?:(`[^`]+`|[a-zA-Z_]\w*)|(\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)|(\*\*|>=|<=|==|!=|&&|\|\||[+\-*/%()<>!&|]))/y;
    const tokens: { kind: "name" | "number" | "op"; text: string }[] = [];
    let at = 0;
    const trimmedEnd = expr.trimEnd().length;
    while (at < trimmedEnd) {
      tokenRe.lastIndex = at;
      const m = tokenRe.exec(expr);
      if (!m) {
        const bad = expr.slice(at).trimStart().charAt(0);
        throw new InvalidParameterError(
          `eval: unexpected character '${bad}' in expression '${expr}'`,
          "expr",
          expr
        );
      }
      if (m[1] !== undefined) tokens.push({ kind: "name", text: m[1] });
      else if (m[2] !== undefined) tokens.push({ kind: "number", text: m[2] });
      else tokens.push({ kind: "op", text: m[3] as string });
      at = tokenRe.lastIndex;
    }
    if (tokens.length === 0) {
      throw new InvalidParameterError(`eval: could not parse expression '${expr}'`, "expr", expr);
    }

    type Value = number | boolean;
    type Node = () => Value;
    const num = (v: Value): number => (typeof v === "boolean" ? (v ? 1 : 0) : v);
    const truthy = (v: Value): boolean =>
      typeof v === "boolean" ? v : v !== 0 && !Number.isNaN(v);

    let row = 0;
    let pos = 0;
    const fail = (message: string): never => {
      throw new InvalidParameterError(`eval: ${message} in expression '${expr}'`, "expr", expr);
    };
    const peek = (): string | undefined => tokens[pos]?.text;
    const peekOp = (...ops: string[]): string | undefined => {
      const t = tokens[pos];
      return t !== undefined && t.kind === "op" && ops.includes(t.text) ? t.text : undefined;
    };
    const peekWord = (...words: string[]): string | undefined => {
      const t = tokens[pos];
      return t !== undefined && t.kind === "name" && words.includes(t.text) ? t.text : undefined;
    };

    const parseOr = (): Node => {
      let left = parseAnd();
      while (peekWord("or") || peekOp("|", "||")) {
        pos++;
        const l = left;
        const r = parseAnd();
        left = () => truthy(l()) || truthy(r());
      }
      return left;
    };
    const parseAnd = (): Node => {
      let left = parseNot();
      while (peekWord("and") || peekOp("&", "&&")) {
        pos++;
        const l = left;
        const r = parseNot();
        left = () => truthy(l()) && truthy(r());
      }
      return left;
    };
    const parseNot = (): Node => {
      if (peekWord("not") || peekOp("!")) {
        pos++;
        const inner = parseNot();
        return () => !truthy(inner());
      }
      return parseComparison();
    };
    const parseComparison = (): Node => {
      const left = parseAdd();
      const cmp = peekOp(">=", "<=", "==", "!=", ">", "<");
      if (cmp === undefined) return left;
      pos++;
      const right = parseAdd();
      switch (cmp) {
        case ">=":
          return () => num(left()) >= num(right());
        case "<=":
          return () => num(left()) <= num(right());
        case "==":
          return () => num(left()) === num(right());
        case "!=":
          return () => num(left()) !== num(right());
        case ">":
          return () => num(left()) > num(right());
        default:
          return () => num(left()) < num(right());
      }
    };
    const parseAdd = (): Node => {
      let left = parseMul();
      for (let op = peekOp("+", "-"); op !== undefined; op = peekOp("+", "-")) {
        pos++;
        const l = left;
        const r = parseMul();
        left = op === "+" ? () => num(l()) + num(r()) : () => num(l()) - num(r());
      }
      return left;
    };
    const parseMul = (): Node => {
      let left = parseUnary();
      for (let op = peekOp("*", "/", "%"); op !== undefined; op = peekOp("*", "/", "%")) {
        pos++;
        const l = left;
        const r = parseUnary();
        if (op === "*") left = () => num(l()) * num(r());
        else if (op === "/") left = () => num(l()) / num(r());
        else {
          // Python semantics: a non-zero result takes the sign of the divisor.
          left = () => {
            const a = num(l());
            const b = num(r());
            const rem = a % b;
            return rem !== 0 && rem < 0 !== b < 0 ? rem + b : rem;
          };
        }
      }
      return left;
    };
    const parseUnary = (): Node => {
      const op = peekOp("-", "+");
      if (op !== undefined) {
        pos++;
        const inner = parseUnary();
        return op === "-" ? () => -num(inner()) : () => num(inner());
      }
      return parsePower();
    };
    const parsePower = (): Node => {
      const base = parseAtom();
      if (peekOp("**")) {
        pos++;
        // The exponent may carry its own sign: 2 ** -1.
        const exponent = parseUnary();
        return () => num(base()) ** num(exponent());
      }
      return base;
    };
    const parseAtom = (): Node => {
      const t = tokens[pos];
      if (t === undefined) return fail("unexpected end of expression");
      pos++;
      if (t.kind === "op") {
        if (t.text !== "(") return fail(`unexpected '${t.text}'`);
        const inner = parseOr();
        if (peek() !== ")") return fail("missing closing parenthesis");
        pos++;
        return inner;
      }
      if (t.kind === "number") {
        const value = Number(t.text);
        return () => value;
      }
      const name = t.text.startsWith("`") ? t.text.slice(1, -1) : t.text;
      const column = this._data.get(name);
      if (column !== undefined) {
        return () => {
          const val = column[row];
          if (typeof val === "number") return val;
          if (typeof val === "boolean") return val ? 1 : 0;
          if (val === null || val === undefined) return NaN;
          return Number(String(val));
        };
      }
      switch (name) {
        case "true":
          return () => true;
        case "false":
          return () => false;
        case "NaN":
          return () => NaN;
        case "Infinity":
          return () => Infinity;
        default:
          return fail(`unknown name '${name}'`);
      }
    };

    const root = parseOr();
    if (pos < tokens.length) fail(`unexpected '${peek()}'`);

    const n = this._index.length;
    const results: unknown[] = new Array(n);
    for (let i = 0; i < n; i++) {
      row = i;
      results[i] = root();
    }
    return results;
  }

  /**
   * Compute a cross-tabulation of two columns.
   *
   * Returns a DataFrame where rows correspond to unique values of `rowCol`,
   * columns correspond to unique values of `colCol`, and cell values are counts.
   * Row labels and column names are sorted (numbers numerically, otherwise by
   * string). Rows where either value is missing are not counted.
   *
   * @param rowCol - Column name for row grouping
   * @param colCol - Column name for column grouping
   * @returns Cross-tabulation DataFrame
   * @throws {IndexError} If a column does not exist
   *
   * @example
   * ```ts
   * const df = new DataFrame({
   *   gender: ['M', 'F', 'M', 'F', 'M'],
   *   handed: ['R', 'R', 'L', 'R', 'R'],
   * });
   * df.crosstab('gender', 'handed');
   * // DataFrame with rows F/M, columns L/R, values are counts
   * ```
   */
  crosstab(rowCol: string, colCol: string): DataFrame {
    if (!this._data.has(rowCol)) throw new IndexError(`Column '${rowCol}' not found`);
    if (!this._data.has(colCol)) throw new IndexError(`Column '${colCol}' not found`);

    const rowData = this._data.get(rowCol) ?? [];
    const colData = this._data.get(colCol) ?? [];

    const rowLabels: (string | number)[] = [];
    const colNames: string[] = [];
    const rowOf = new Map<string | number, number>();
    const colOf = new Map<string, number>();
    const countsByRow: Map<number, number>[] = [];

    for (let i = 0; i < this._index.length; i++) {
      const rk = rowData[i];
      const ck = colData[i];
      if (isMissing(rk) || isMissing(ck)) continue;
      const label = toIndexLabel(rk);
      const name = String(ck);
      let r = rowOf.get(label);
      if (r === undefined) {
        r = rowLabels.length;
        rowOf.set(label, r);
        rowLabels.push(label);
        countsByRow.push(new Map());
      }
      let c = colOf.get(name);
      if (c === undefined) {
        c = colNames.length;
        colOf.set(name, c);
        colNames.push(name);
      }
      const rowCounts = countsByRow[r] as Map<number, number>;
      rowCounts.set(c, (rowCounts.get(c) ?? 0) + 1);
    }

    const rowOrder = rowLabels
      .map((_, r) => r)
      .sort((a, b) => compareLabels(rowLabels[a], rowLabels[b]));
    const colOrder = colNames
      .map((_, c) => c)
      .sort((a, b) => compareLabels(colNames[a], colNames[b]));

    const result: DataFrameData = newColumnMap();
    for (const c of colOrder) {
      result[colNames[c] as string] = rowOrder.map(
        (r) => (countsByRow[r] as Map<number, number>).get(c) ?? 0
      );
    }

    return new DataFrame(result, {
      columns: colOrder.map((c) => colNames[c] as string),
      index: rowOrder.map((r) => rowLabels[r] as string | number),
      copy: false,
    });
  }
}

/**
 * Expanding (cumulative) window calculations on DataFrame columns.
 *
 * Created by DataFrame.expanding(). NaN, null, infinite and non-numeric values are not
 * observations (pandas treats infinities the same way in window functions); a row's
 * result needs at least `minPeriods` observations so far.
 */
export class Expanding {
  private df: DataFrame;
  private minPeriods: number;

  constructor(df: DataFrame, minPeriods: number) {
    this.df = df;
    this.minPeriods = minPeriods;
  }

  /**
   * Runs `step` over every observation of every column. `step` receives the new
   * value and returns the current statistic (or null while it is undefined).
   */
  private stream(makeStep: () => (value: number, count: number) => number | null): DataFrame {
    const newData: DataFrameData = newColumnMap();
    for (const col of this.df.columns) {
      const colData = this.df.getColumnData(col);
      const step = makeStep();
      const result: unknown[] = new Array(colData.length);
      let count = 0;
      let current: number | null = null;
      for (let i = 0; i < colData.length; i++) {
        const v = colData[i];
        if (isValidNumber(v)) {
          count++;
          current = step(v, count);
        }
        result[i] = count < this.minPeriods ? null : current;
      }
      newData[col] = result;
    }
    return new DataFrame(newData, {
      columns: this.df.columns,
      index: this.df.index,
      copy: false,
    });
  }

  /** Expanding mean (compensated running sum). */
  mean(): DataFrame {
    return this.stream(() => {
      let sum = 0;
      let comp = 0;
      return (v, count) => {
        const t = sum + v;
        comp += Math.abs(sum) >= Math.abs(v) ? sum - t + v : v - t + sum;
        sum = t;
        return (sum + comp) / count;
      };
    });
  }

  /** Expanding sum (compensated). */
  sum(): DataFrame {
    return this.stream(() => {
      let sum = 0;
      let comp = 0;
      return (v) => {
        const t = sum + v;
        comp += Math.abs(sum) >= Math.abs(v) ? sum - t + v : v - t + sum;
        sum = t;
        return sum + comp;
      };
    });
  }

  /** Welford update shared by var() and std(); null until two observations exist. */
  private variance(root: boolean): DataFrame {
    return this.stream(() => {
      let mean = 0;
      let m2 = 0;
      return (v, count) => {
        const delta = v - mean;
        mean += delta / count;
        m2 += delta * (v - mean);
        if (count < 2) return null;
        const variance = m2 / (count - 1);
        return root ? Math.sqrt(variance) : variance;
      };
    });
  }

  /** Expanding sample standard deviation (ddof = 1). */
  std(): DataFrame {
    return this.variance(true);
  }

  /** Expanding sample variance (ddof = 1). */
  var(): DataFrame {
    return this.variance(false);
  }

  /** Expanding minimum. */
  min(): DataFrame {
    return this.stream(() => {
      let best = Infinity;
      return (v) => {
        if (v < best) best = v;
        return best;
      };
    });
  }

  /** Expanding maximum. */
  max(): DataFrame {
    return this.stream(() => {
      let best = -Infinity;
      return (v) => {
        if (v > best) best = v;
        return best;
      };
    });
  }
}

/**
 * Exponentially weighted moving calculations on DataFrame columns.
 *
 * Created by DataFrame.ewm(). The weights follow pandas' algorithm, so results
 * can be compared against `Series.ewm(...)` with the same `adjust`, `bias` and
 * `ignoreNa` settings. NaN, null, infinite and non-numeric values are treated as
 * missing observations.
 */
export class EWM {
  private df: DataFrame;
  private alpha: number;
  private adjust: boolean;
  private bias: boolean;
  private ignoreNa: boolean;

  constructor(
    df: DataFrame,
    alpha: number,
    settings: { adjust: boolean; bias: boolean; ignoreNa: boolean } = {
      adjust: false,
      bias: true,
      ignoreNa: false,
    }
  ) {
    this.df = df;
    this.alpha = alpha;
    this.adjust = settings.adjust;
    this.bias = settings.bias;
    this.ignoreNa = settings.ignoreNa;
  }

  /** Exponentially weighted mean. Positions before the first observation are null. */
  mean(): DataFrame {
    const decay = 1 - this.alpha;
    const newWt = this.adjust ? 1 : this.alpha;
    const newData: DataFrameData = newColumnMap();
    for (const col of this.df.columns) {
      const colData = this.df.getColumnData(col);
      const result: unknown[] = new Array(colData.length);
      let started = false;
      let weighted = 0;
      let oldWt = 1;
      for (let i = 0; i < colData.length; i++) {
        const v = colData[i];
        const observed = isValidNumber(v);
        if (!started) {
          if (observed) {
            started = true;
            weighted = v;
            oldWt = 1;
          }
        } else if (observed || !this.ignoreNa) {
          oldWt *= decay;
          if (observed) {
            // The old value has been decayed once per row since the last observation
            // (a gap of missing rows ages it further), the new value has weight newWt.
            if (weighted !== v) weighted = (oldWt * weighted + newWt * v) / (oldWt + newWt);
            oldWt = this.adjust ? oldWt + newWt : 1;
          }
        }
        result[i] = started ? weighted : null;
      }
      newData[col] = result;
    }
    return new DataFrame(newData, {
      columns: this.df.columns,
      index: this.df.index,
      copy: false,
    });
  }

  /** Weighted variance (pandas' ewmcov with x = y); null until two observations exist. */
  private variance(root: boolean): DataFrame {
    const decay = 1 - this.alpha;
    const newWt = this.adjust ? 1 : this.alpha;
    const newData: DataFrameData = newColumnMap();
    for (const col of this.df.columns) {
      const colData = this.df.getColumnData(col);
      const result: unknown[] = new Array(colData.length);
      let started = false;
      let mean = 0;
      let cov = 0;
      let sumWt = 1;
      let sumWt2 = 1;
      let oldWt = 1;
      let nobs = 0;
      for (let i = 0; i < colData.length; i++) {
        const v = colData[i];
        const observed = isValidNumber(v);
        if (observed) nobs++;
        if (!started) {
          if (observed) {
            started = true;
            mean = v;
            cov = 0;
            sumWt = 1;
            sumWt2 = 1;
            oldWt = 1;
          }
        } else if (observed || !this.ignoreNa) {
          sumWt *= decay;
          sumWt2 *= decay * decay;
          oldWt *= decay;
          if (observed) {
            const oldMean = mean;
            if (mean !== v) mean = (oldWt * oldMean + newWt * v) / (oldWt + newWt);
            cov =
              (oldWt * (cov + (oldMean - mean) * (oldMean - mean)) +
                newWt * ((v - mean) * (v - mean))) /
              (oldWt + newWt);
            sumWt += newWt;
            sumWt2 += newWt * newWt;
            oldWt += newWt;
            if (!this.adjust) {
              sumWt /= oldWt;
              sumWt2 /= oldWt * oldWt;
              oldWt = 1;
            }
          }
        }

        if (!started || nobs < 2) {
          result[i] = null;
          continue;
        }
        let variance = cov;
        if (!this.bias) {
          const numerator = sumWt * sumWt;
          const denominator = numerator - sumWt2;
          variance = denominator > 0 ? (numerator / denominator) * cov : NaN;
        }
        result[i] = root ? Math.sqrt(variance) : variance;
      }
      newData[col] = result;
    }
    return new DataFrame(newData, {
      columns: this.df.columns,
      index: this.df.index,
      copy: false,
    });
  }

  /** Exponentially weighted standard deviation. */
  std(): DataFrame {
    return this.variance(true);
  }

  /** Exponentially weighted variance. */
  var(): DataFrame {
    return this.variance(false);
  }
}

/**
 * GroupBy object for aggregation operations.
 *
 * Created by DataFrame.groupBy(). Used to perform aggregations on grouped data.
 *
 * @example
 * ```ts
 * const df = new DataFrame({
 *   category: ['A', 'B', 'A', 'B'],
 *   value: [10, 20, 30, 40]
 * });
 * const grouped = df.groupBy('category');
 * grouped.sum();   // Sum by category
 * grouped.mean();  // Mean by category
 * ```
 */
export class DataFrameGroupBy {
  // Store the group mapping (computed once)
  private groupMap: Map<string, number[]>;
  // Store the original key values for each group key (to avoid parsing)
  private keyValuesMap: Map<string, unknown[]>;
  private df: DataFrame;
  private by: string | string[];

  constructor(df: DataFrame, by: string | string[], options: GroupByOptions = {}) {
    const byCols = Array.isArray(by) ? by : [by];
    if (byCols.length === 0 || !isStringArray(byCols)) {
      throw new InvalidParameterError(
        "by must be a column name or a non-empty array of column names",
        "by",
        by
      );
    }
    ensureUniqueLabels(byCols, "group column");
    for (const name of ["sort", "dropna"] as const) {
      const flag = options[name];
      if (flag !== undefined && typeof flag !== "boolean") {
        throw new InvalidParameterError(`${name} must be a boolean`, name, flag);
      }
    }
    const knownColumns = new Set(df.columns);
    for (const col of byCols) {
      if (!knownColumns.has(col)) {
        throw new InvalidParameterError(`Column '${col}' not found in DataFrame`, "by", col);
      }
    }
    this.df = df;
    this.by = by;
    // Build group map once during construction
    const buildResult = this.buildGroupMap();
    this.groupMap = buildResult.groupMap;
    this.keyValuesMap = buildResult.keyValuesMap;
    if (options.dropna === true) this.dropMissingKeys();
    if (options.sort === true) this.sortGroups();
  }

  /** Remove the groups whose key has a null, undefined or NaN part. */
  private dropMissingKeys(): void {
    for (const [key, parts] of this.keyValuesMap) {
      if (parts.some(isMissing)) {
        this.groupMap.delete(key);
        this.keyValuesMap.delete(key);
      }
    }
  }

  /**
   * Reorder the groups by key: column by column ascending, missing parts last.
   * The sort is stable, so equal keys (which cannot occur) would keep their order.
   */
  private sortGroups(): void {
    const entries = [...this.groupMap.entries()].map(([key, rows]) => ({
      key,
      rows,
      parts: this.keyValuesMap.get(key) ?? [],
    }));
    entries.sort((a, b) => {
      for (let c = 0; c < a.parts.length; c++) {
        const x = a.parts[c];
        const y = b.parts[c];
        const xMissing = isMissing(x);
        const yMissing = isMissing(y);
        if (xMissing || yMissing) {
          if (xMissing && yMissing) continue;
          return xMissing ? 1 : -1;
        }
        const cmp = compareValues(x, y);
        if (cmp !== 0) return cmp;
      }
      return 0;
    });
    this.groupMap = new Map(entries.map((e) => [e.key, e.rows]));
    this.keyValuesMap = new Map(entries.map((e) => [e.key, e.parts]));
  }

  /**
   * Build the grouping map: group key -> array of row indices.
   *
   * @private
   */
  private buildGroupMap(): {
    groupMap: Map<string, number[]>;
    keyValuesMap: Map<string, unknown[]>;
  } {
    const groupByCols = Array.isArray(this.by) ? this.by : [this.by];
    const groupMap = new Map<string, number[]>();
    const keyValuesMap = new Map<string, unknown[]>();

    const numRows = this.df.shape[0];

    // Fast path: single column groupBy, avoid array allocation and composite key
    if (groupByCols.length === 1) {
      const colData = this.df.getColumnData(groupByCols[0] as string);
      // When every value is a primitive, the value itself is a collision-free
      // Map key (SameValueZero matches createKey semantics for numbers,
      // strings, booleans, null and NaN), skipping per-row key-string
      // construction. Any object value falls back to createKey for all rows.
      let allPrimitive = true;
      for (let i = 0; i < numRows; i++) {
        const v = colData[i];
        if (v !== null && (typeof v === "object" || typeof v === "function")) {
          allPrimitive = false;
          break;
        }
      }
      if (allPrimitive) {
        const rawMap = new Map<unknown, number[]>();
        for (let i = 0; i < numRows; i++) {
          const val = colData[i];
          let bucket = rawMap.get(val);
          if (bucket === undefined) {
            bucket = [];
            rawMap.set(val, bucket);
          }
          bucket.push(i);
        }
        for (const [val, bucket] of rawMap) {
          const key = createKey(val);
          groupMap.set(key, bucket);
          keyValuesMap.set(key, [val]);
        }
      } else {
        for (let i = 0; i < numRows; i++) {
          const val = colData[i];
          const key = createKey(val);

          let bucket = groupMap.get(key);
          if (bucket === undefined) {
            bucket = [];
            groupMap.set(key, bucket);
            keyValuesMap.set(key, [val]);
          }
          bucket.push(i);
        }
      }
    } else {
      // Multi-column: pre-fetch all column data arrays
      const colDataArrays: (readonly unknown[])[] = [];
      for (let c = 0; c < groupByCols.length; c++) {
        colDataArrays.push(this.df.getColumnData(groupByCols[c] as string));
      }

      for (let i = 0; i < numRows; i++) {
        const keyParts: unknown[] = new Array(groupByCols.length);
        for (let c = 0; c < groupByCols.length; c++) {
          const colArr = colDataArrays[c];
          keyParts[c] = colArr !== undefined ? colArr[i] : undefined;
        }

        const key = createKey(keyParts);

        let bucket = groupMap.get(key);
        if (bucket === undefined) {
          bucket = [];
          groupMap.set(key, bucket);
          keyValuesMap.set(key, keyParts);
        }
        bucket.push(i);
      }
    }

    return { groupMap, keyValuesMap };
  }

  /**
   * Aggregate grouped data.
   *
   * Groups appear in the order their keys first occur (or sorted by key when the
   * GroupBy was created with `sort: true`). Numeric aggregations
   * (`sum`, `mean`, `median`, `min`, `max`, `std`, `var`) skip null, undefined
   * and NaN, keep infinities, and throw on other non-numeric values. `std` and
   * `var` use ddof = 1 and give NaN for fewer than two values; `sum` of nothing is 0
   * and the other numeric aggregations of nothing are NaN. `count` counts values that
   * are not null, undefined or NaN. `first` and `last` return the first and last
   * value of the group as stored, including missing ones.
   *
   * `nunique` counts the distinct values that are not missing.
   *
   * Besides `{ column: func }` and `{ column: [func, ...] }`, a value can be a pair
   * `[column, aggregation]` stored under the output name used as the key, like pandas' named
   * aggregation. The aggregation is a function name or a function that receives the group's
   * values for that column (missing values included).
   *
   * @param operations - Dictionary of column name to aggregation function(s), or of output name to
   *   `[column, aggregation]`. With an array of functions the output columns are named
   *   `<column>_<function>`.
   * @returns New DataFrame with aggregated data
   * @throws {InvalidParameterError} If a column does not exist
   * @throws {DataValidationError} If an aggregation function is unknown or the data is not numeric
   *
   * @example
   * ```ts
   * const grouped = df.groupBy('category');
   * const result = grouped.agg({ value: 'sum', count: 'count' });
   * grouped.agg({ avgValue: ['value', 'mean'], spread: ['value', (v) => Math.max(...(v as number[])) - Math.min(...(v as number[]))] });
   * ```
   */
  agg(
    operations: Record<string, AggregateFunction | AggregateFunction[] | NamedAggregation>
  ): DataFrame {
    const groupByCols = Array.isArray(this.by) ? this.by : [this.by];
    const resultData: DataFrameData = newColumnMap();
    const outputColumns: string[] = [];

    // Initialize result columns (groupby columns + aggregated columns)
    for (const col of groupByCols) {
      resultData[col] = [];
      outputColumns.push(col);
    }

    // Validate the whole request before touching any group, so an empty
    // DataFrame reports the same errors as a populated one.
    const plan: GroupPlanEntry[] = [];
    const register = (
      outCol: string,
      func: AggregateFunction | ((values: unknown[]) => unknown),
      data: readonly unknown[]
    ): void => {
      if (typeof func !== "function" && !AGGREGATE_FUNCTIONS.includes(func)) {
        throw new DataValidationError(`Unsupported aggregation function: ${String(func)}`);
      }
      if (outputColumns.includes(outCol)) {
        throw new DataValidationError(`Duplicate output column '${outCol}' in aggregation`);
      }
      resultData[outCol] = [];
      outputColumns.push(outCol);
      plan.push({
        outCol,
        data,
        run:
          typeof func === "function"
            ? (values, indices) => func(indices.map((idx) => values[idx]))
            : (values, indices) => aggregateRows(func, values, indices),
      });
    };
    const columnNames = new Set(this.df.columns);
    for (const [name, spec] of Object.entries(operations)) {
      // [column, aggregation] under an output name: named aggregation. A pair of function
      // names under a column name stays the list form of Deepbox 1.0.
      if (
        Array.isArray(spec) &&
        spec.length === 2 &&
        typeof spec[0] === "string" &&
        columnNames.has(spec[0]) &&
        (typeof spec[1] === "function" || AGGREGATE_FUNCTIONS.includes(spec[1] as string)) &&
        !(columnNames.has(name) && AGGREGATE_FUNCTIONS.includes(spec[0]))
      ) {
        register(name, spec[1] as AggregateFunction, this.df.getColumnData(spec[0]));
        continue;
      }
      const data = this.df.getColumnData(name);
      const funcs = Array.isArray(spec) ? spec : [spec];
      for (const func of funcs) {
        register(
          Array.isArray(spec) ? `${name}_${String(func)}` : name,
          func as AggregateFunction,
          data
        );
      }
    }

    return this.runPlan(plan, outputColumns, resultData);
  }

  /**
   * Evaluates an aggregation plan for every group and builds the result frame
   * (group key columns followed by one column per plan entry).
   */
  private runPlan(
    plan: readonly GroupPlanEntry[],
    outputColumns: string[],
    resultData: DataFrameData
  ): DataFrame {
    const groupByCols = Array.isArray(this.by) ? this.by : [this.by];
    for (const [keyStr, indices] of this.groupMap.entries()) {
      const keyParts = this.keyValuesMap.get(keyStr);
      if (!keyParts) {
        throw new DataValidationError(`Missing key values for group: ${keyStr}`);
      }
      for (let i = 0; i < groupByCols.length; i++) {
        const groupCol = groupByCols[i];
        if (groupCol !== undefined) resultData[groupCol]?.push(keyParts[i]);
      }
      for (const { outCol, data, run } of plan) {
        resultData[outCol]?.push(run(data, indices));
      }
    }
    return new DataFrame(resultData, { columns: outputColumns, copy: false });
  }

  /**
   * Helper to identify numeric columns (excluding grouping columns).
   * @private
   */
  private getNumericColumns(): string[] {
    const groupByCols = Array.isArray(this.by) ? this.by : [this.by];
    const otherCols = this.df.columns.filter((c) => !groupByCols.includes(c));
    return otherCols.filter((col) => {
      // efficient check: look for at least one valid number
      return this.df.getColumnData(col).some(isOrderedNumber);
    });
  }

  /**
   * Helper method to perform same aggregation on all numeric non-grouping columns.
   * @private
   */
  private aggNumeric(operation: AggregateFunction): DataFrame {
    const numericCols = this.getNumericColumns();
    const operations = Object.create(null) as Record<string, AggregateFunction>;
    for (const col of numericCols) {
      operations[col] = operation;
    }
    return this.agg(operations);
  }

  /**
   * Helper method to perform same aggregation on all non-grouping columns.
   *
   * @private
   */
  private aggAll(operation: AggregateFunction): DataFrame {
    const groupByCols = Array.isArray(this.by) ? this.by : [this.by];
    const otherCols = this.df.columns.filter((c) => !groupByCols.includes(c));

    const operations = Object.create(null) as Record<string, AggregateFunction>;
    for (const col of otherCols) {
      operations[col] = operation;
    }

    return this.agg(operations);
  }

  /**
   * Compute sum for each group.
   *
   * @returns DataFrame with summed values by group
   *
   * @example
   * ```ts
   * const df = new DataFrame({
   *   category: ['A', 'A', 'B', 'B'],
   *   value: [1, 2, 3, 4]
   * });
   * df.groupBy('category').sum();
   * // category | value
   * //    A     |   3
   * //    B     |   7
   * ```
   */
  sum(): DataFrame {
    return this.aggNumeric("sum");
  }

  /**
   * Compute mean (average) for each group.
   *
   * @returns DataFrame with mean values by group
   */
  mean(): DataFrame {
    return this.aggNumeric("mean");
  }

  /**
   * Count non-null values in each non-grouping column for every group.
   *
   * @returns DataFrame with per-column non-null counts by group
   */
  count(): DataFrame {
    return this.aggAll("count");
  }

  /**
   * Compute minimum value for each group.
   *
   * @returns DataFrame with minimum values by group
   */
  min(): DataFrame {
    return this.aggNumeric("min");
  }

  /**
   * Compute maximum value for each group.
   *
   * @returns DataFrame with maximum values by group
   */
  max(): DataFrame {
    return this.aggNumeric("max");
  }

  /**
   * Compute standard deviation for each group.
   *
   * @returns DataFrame with standard deviation values by group
   */
  std(): DataFrame {
    return this.aggNumeric("std");
  }

  /**
   * Compute variance for each group.
   *
   * @returns DataFrame with variance values by group
   */
  var(): DataFrame {
    return this.aggNumeric("var");
  }

  /**
   * Compute median for each group.
   *
   * @returns DataFrame with median values by group
   */
  median(): DataFrame {
    return this.aggNumeric("median");
  }

  /**
   * First value of every non-grouping column in each group (missing values included).
   *
   * @returns DataFrame with one row per group
   */
  first(): DataFrame {
    return this.aggAll("first");
  }

  /**
   * Last value of every non-grouping column in each group (missing values included).
   *
   * @returns DataFrame with one row per group
   */
  last(): DataFrame {
    return this.aggAll("last");
  }

  /**
   * Number of rows in each group, missing values included.
   *
   * @returns DataFrame with the group columns and a `size` column
   */
  size(): DataFrame {
    const groupByCols = Array.isArray(this.by) ? this.by : [this.by];
    const data: DataFrameData = newColumnMap();
    for (const col of groupByCols) data[col] = [];
    const sizes: number[] = [];
    for (const [key, indices] of this.groupMap) {
      const keyParts = this.keyValuesMap.get(key) ?? [];
      for (let i = 0; i < groupByCols.length; i++) {
        data[groupByCols[i] as string]?.push(keyParts[i]);
      }
      sizes.push(indices.length);
    }
    data["size"] = sizes;
    return new DataFrame(data, { columns: [...groupByCols, "size"], copy: false });
  }

  /**
   * Rows of one group, with all columns and the original row labels.
   *
   * @param key - The group's key: the value for one grouping column, or an array with one value per
   *   grouping column (in the order given to `groupBy`). `NaN`, `null` and `undefined` keys are
   *   looked up like any other value.
   * @returns DataFrame with the rows of the group in their original order
   * @throws {IndexError} If no group has that key
   * @throws {InvalidParameterError} If an array key has the wrong length
   *
   * @example
   * ```ts
   * const df = new DataFrame({ team: ["a", "b", "a"], x: [1, 2, 3] });
   * df.groupBy("team").getGroup("a");  // rows 0 and 2
   * df.groupBy(["team", "x"]).getGroup(["b", 2]);
   * ```
   */
  getGroup(key: unknown): DataFrame {
    const byCols = Array.isArray(this.by) ? this.by : [this.by];
    if (byCols.length > 1 && (!Array.isArray(key) || key.length !== byCols.length)) {
      throw new InvalidParameterError(
        `key must be an array with one value per grouping column (${byCols.length})`,
        "key",
        key
      );
    }
    const rows = this.groupMap.get(createKey(key));
    if (rows === undefined) {
      throw new IndexError(`Group not found: ${cellText(key)}`);
    }
    const labels = this.df.index;
    const columns = this.df.columns;
    const data: DataFrameData = newColumnMap();
    for (const col of columns) {
      const source = this.df.getColumnData(col);
      data[col] = rows.map((row) => source[row]);
    }
    return new DataFrame(data, {
      columns,
      index: rows.map((row) => labels[row] as string | number),
      copy: false,
    });
  }

  /**
   * Number of distinct values, not counting missing ones, of every non-grouping column in each group.
   *
   * @returns DataFrame with the group columns followed by one count column per other column
   *
   * @example
   * ```ts
   * const df = new DataFrame({ team: ["a", "a", "b"], x: [1, 1, 2], y: ["p", "q", null] });
   * df.groupBy("team").nunique();  // a: x=1, y=2; b: x=1, y=0
   * ```
   */
  nunique(): DataFrame {
    return this.aggAll("nunique");
  }

  /**
   * Quantile of every numeric column in each group, by linear interpolation between ranks
   * (the NumPy and pandas default). NaN, null and undefined are skipped; a group without values
   * gives NaN.
   *
   * @param q - Quantile in [0, 1]
   * @returns DataFrame with the group columns followed by the quantile of each numeric column
   * @throws {InvalidParameterError} If `q` is not a finite number in [0, 1]
   *
   * @example
   * ```ts
   * const df = new DataFrame({ team: ["a", "a", "a", "b"], x: [1, 2, 4, 9] });
   * df.groupBy("team").quantile(0.5);  // a: 2, b: 9
   * ```
   */
  quantile(q: number): DataFrame {
    if (typeof q !== "number" || !Number.isFinite(q) || q < 0 || q > 1) {
      throw new InvalidParameterError("q must be a finite number between 0 and 1", "q", q);
    }
    const byCols = Array.isArray(this.by) ? this.by : [this.by];
    const resultData: DataFrameData = newColumnMap();
    const outputColumns: string[] = [...byCols];
    for (const col of byCols) resultData[col] = [];
    const plan: GroupPlanEntry[] = [];
    for (const col of this.getNumericColumns()) {
      resultData[col] = [];
      outputColumns.push(col);
      plan.push({
        outCol: col,
        data: this.df.getColumnData(col),
        run: (values, indices) => {
          const nums = numbersOf(values, indices, "quantile");
          return nums.length > 0 ? quantileSorted(Float64Array.from(nums).sort(), q) : NaN;
        },
      });
    }
    return this.runPlan(plan, outputColumns, resultData);
  }

  /**
   * Transform every non-grouping column group by group, returning a frame with the same rows (and
   * row labels) as the original, so the result lines up with the source data.
   *
   * `how` is either
   * - an aggregation name (`sum`, `mean`, `median`, `min`, `max`, `std`, `var`, `count`, `first`,
   *   `last`, `nunique`): the group's value is repeated on each of its rows. The numeric ones apply
   *   to the numeric columns only, like the other group aggregations;
   * - `cumsum`, `cumprod`, `cummax`, `cummin` (numeric columns), `ffill` or `bfill`: computed within
   *   each group, with the semantics of the same-named DataFrame methods;
   * - a function that receives the group's values of one column as a Series (labelled with the row
   *   labels) and returns a Series or array of the same length, or one value that is repeated.
   *
   * Rows that belong to no group (a dropped missing key) get null.
   *
   * @param how - Name or function
   * @returns DataFrame with the columns that were transformed
   * @throws {InvalidParameterError} If a name is unknown
   * @throws {DataValidationError} If a function returns an array of the wrong length
   *
   * @example
   * ```ts
   * const df = new DataFrame({ team: ["a", "b", "a"], x: [1, 5, 3] });
   * df.groupBy("team").transform("mean");                       // x: [2, 5, 2]
   * df.groupBy("team").transform((s) => s.map((v) => Number(v) - 2));  // x: [-1, 3, 1]
   * ```
   */
  transform(
    how:
      | AggregateFunction
      | "cumsum"
      | "cumprod"
      | "cummax"
      | "cummin"
      | "ffill"
      | "bfill"
      | ((values: Series<unknown>) => unknown)
  ): DataFrame {
    const byCols = Array.isArray(this.by) ? this.by : [this.by];
    const otherCols = this.df.columns.filter((c) => !byCols.includes(c));
    const nRows = this.df.shape[0];
    const labels = this.df.index;
    const out: DataFrameData = newColumnMap();
    const fresh = (cols: readonly string[]): void => {
      for (const col of cols) out[col] = new Array<unknown>(nRows).fill(null);
    };

    if (typeof how === "function") {
      fresh(otherCols);
      for (const rows of this.groupMap.values()) {
        const rowLabels = rows.map((r) => labels[r] as string | number);
        for (const col of otherCols) {
          const source = this.df.getColumnData(col);
          const series = new Series<unknown>(
            rows.map((r) => source[r]),
            { index: rowLabels, copy: false }
          );
          const result = how(series);
          const values: readonly unknown[] | undefined =
            result instanceof Series ? result.data : Array.isArray(result) ? result : undefined;
          if (values !== undefined && values.length !== rows.length) {
            throw new DataValidationError(
              `transform function returned ${values.length} values for a group of ${rows.length} rows`
            );
          }
          const target = out[col] as unknown[];
          rows.forEach((r, k) => {
            target[r] = values !== undefined ? values[k] : result;
          });
        }
      }
      return new DataFrame(out, { columns: otherCols, index: labels, copy: false });
    }

    if (AGGREGATE_FUNCTIONS.includes(how)) {
      const numericOnly = !["count", "first", "last", "nunique"].includes(how);
      const cols = numericOnly ? this.getNumericColumns() : otherCols;
      fresh(cols);
      for (const rows of this.groupMap.values()) {
        for (const col of cols) {
          const value = aggregateRows(how as AggregateFunction, this.df.getColumnData(col), rows);
          const target = out[col] as unknown[];
          for (const r of rows) target[r] = value;
        }
      }
      return new DataFrame(out, { columns: cols, index: labels, copy: false });
    }

    if (TRANSFORM_FUNCTIONS.includes(how)) {
      const cols = how === "ffill" || how === "bfill" ? otherCols : this.getNumericColumns();
      fresh(cols);
      for (const rows of this.groupMap.values()) {
        const part: DataFrameData = newColumnMap();
        for (const col of cols) {
          const source = this.df.getColumnData(col);
          part[col] = rows.map((r) => source[r]);
        }
        const sub = new DataFrame(part, { columns: cols, copy: false });
        const done =
          how === "cumsum"
            ? sub.cumsum()
            : how === "cumprod"
              ? sub.cumprod()
              : how === "cummax"
                ? sub.cummax()
                : how === "cummin"
                  ? sub.cummin()
                  : how === "ffill"
                    ? sub.ffill()
                    : sub.bfill();
        for (const col of cols) {
          const values = done.getColumnData(col);
          const target = out[col] as unknown[];
          rows.forEach((r, k) => {
            target[r] = values[k];
          });
        }
      }
      return new DataFrame(out, { columns: cols, index: labels, copy: false });
    }

    throw new InvalidParameterError(
      `Unknown transform '${String(how)}'; use an aggregation name, one of ` +
        `${TRANSFORM_FUNCTIONS.join(", ")}, or a function`,
      "how",
      how
    );
  }

  /** Number of distinct groups. */
  get ngroups(): number {
    return this.groupMap.size;
  }
}

/**
 * Rolling window calculations on DataFrame columns.
 *
 * Created by DataFrame.rolling(). Provides mean, sum, std, var, min, max, median,
 * apply, corr and cov. Like pandas with its default `min_periods`, the single-column
 * statistics need a full window of valid numbers: windows that reach before the
 * first row or contain a NaN, null or non-numeric value give null. With `minPeriods`
 * they work from the valid values of a shorter window, and with `center` the window
 * is placed around each row. `corr` and `cov` use the valid pairs inside each window
 * and need at least two (or `minPeriods`, when given; the window must then be full only
 * if `minPeriods` is not given).
 */
export class Rolling {
  private df: DataFrame;
  private window: number;
  private on: string | undefined;
  private minPeriods: number;
  private explicitMinPeriods: boolean;
  private center: boolean;

  /**
   * @param df - Source DataFrame
   * @param window - Window size, a positive integer
   * @param on - Column to roll over (default: every column)
   * @param options - `minPeriods` (default: `window`) and `center` (default: false)
   */
  constructor(
    df: DataFrame,
    window: number,
    on?: string,
    options: { readonly minPeriods?: number; readonly center?: boolean } = {}
  ) {
    this.df = df;
    this.window = window;
    this.on = on;
    this.minPeriods = options.minPeriods ?? window;
    this.explicitMinPeriods = options.minPeriods !== undefined;
    this.center = options.center === true;
  }

  /** First and last row (inclusive, clipped to the data) of the window at row i. */
  private bounds(i: number, n: number): [number, number] {
    const right = this.center ? (this.window - 1) >> 1 : 0;
    return [Math.max(0, i + right - this.window + 1), Math.min(n - 1, i + right)];
  }

  private getWindowValues(colData: readonly unknown[], i: number): number[] {
    const [lo, hi] = this.bounds(i, colData.length);
    const vals: number[] = [];
    for (let j = lo; j <= hi; j++) {
      const v = colData[j];
      if (isValidNumber(v)) vals.push(v);
    }
    return vals;
  }

  private compute(fn: (vals: number[]) => number | null): DataFrame {
    const newData: DataFrameData = newColumnMap();
    const cols = this.on ? [this.on] : this.df.columns;
    for (const col of cols) {
      const colData = this.df.getColumnData(col);
      const result: unknown[] = new Array(colData.length);
      for (let i = 0; i < colData.length; i++) {
        // A window with fewer rows than minPeriods cannot qualify; skip scanning it.
        const [lo, hi] = this.bounds(i, colData.length);
        if (hi - lo + 1 < this.minPeriods) {
          result[i] = null;
          continue;
        }
        const vals = this.getWindowValues(colData, i);
        result[i] = vals.length === 0 || vals.length < this.minPeriods ? null : fn(vals);
      }
      newData[col] = result;
    }
    return new DataFrame(newData, { columns: cols, index: this.df.index, copy: false });
  }

  /**
   * Sliding window sum in O(n). The window total is kept as a compensated
   * (Neumaier) running sum, so adding and dropping values does not accumulate
   * rounding error the way a plain running sum does. `divide` turns the sum into the
   * statistic (the mean divides by the number of valid values in the window).
   */
  private slidingSum(divide: boolean): DataFrame {
    const cols = this.on ? [this.on] : this.df.columns;
    const newData: DataFrameData = newColumnMap();
    for (const col of cols) {
      const colData = this.df.getColumnData(col);
      const n = colData.length;
      const result: unknown[] = new Array(n);
      let sum = 0;
      let comp = 0;
      let validCount = 0;
      let added = 0;
      let removed = 0;
      const add = (x: number): void => {
        const t = sum + x;
        comp += Math.abs(sum) >= Math.abs(x) ? sum - t + x : x - t + sum;
        sum = t;
      };
      for (let i = 0; i < n; i++) {
        const [lo, hi] = this.bounds(i, n);
        while (added <= hi) {
          const v = colData[added];
          if (isValidNumber(v)) {
            add(v);
            validCount++;
          }
          added++;
        }
        while (removed < lo) {
          const old = colData[removed];
          if (isValidNumber(old)) {
            add(-old);
            validCount--;
          }
          removed++;
        }
        if (validCount < this.minPeriods || (divide && validCount === 0)) {
          result[i] = null;
        } else if (validCount === 0) {
          result[i] = 0;
        } else {
          result[i] = divide ? (sum + comp) / validCount : sum + comp;
        }
      }
      newData[col] = result;
    }
    return new DataFrame(newData, { columns: cols, index: this.df.index, copy: false });
  }

  /** Rolling mean. */
  mean(): DataFrame {
    return this.slidingSum(true);
  }

  /** Rolling sum. */
  sum(): DataFrame {
    return this.slidingSum(false);
  }

  /** Rolling sample variance (ddof = 1); null for a window of one value. */
  var(): DataFrame {
    return this.compute((vals) => {
      if (vals.length < 2) return null;
      const m = compensatedSum(vals) / vals.length;
      let sq = 0;
      for (const v of vals) sq += (v - m) ** 2;
      return sq / (vals.length - 1);
    });
  }

  /** Rolling sample standard deviation (ddof = 1); null for a window of one value. */
  std(): DataFrame {
    return this.compute((vals) => {
      if (vals.length < 2) return null;
      const m = compensatedSum(vals) / vals.length;
      let sq = 0;
      for (const v of vals) sq += (v - m) ** 2;
      return Math.sqrt(sq / (vals.length - 1));
    });
  }

  /** Rolling minimum. */
  min(): DataFrame {
    return this.compute((vals) => {
      let best = vals[0] as number;
      for (const v of vals) if (v < best) best = v;
      return best;
    });
  }

  /** Rolling maximum. */
  max(): DataFrame {
    return this.compute((vals) => {
      let best = vals[0] as number;
      for (const v of vals) if (v > best) best = v;
      return best;
    });
  }

  /** Rolling median. */
  median(): DataFrame {
    return this.compute((vals) => quantileSorted(Float64Array.from(vals).sort(), 0.5));
  }

  /**
   * Apply a function to every full window.
   *
   * @param fn - Receives the window's values (oldest first) and returns a number or null
   */
  apply(fn: (vals: number[]) => number | null): DataFrame {
    return this.compute(fn);
  }

  /**
   * Values of two columns over the window at row i, restricted to rows where both are
   * valid numbers. Unlike the single-column statistics, pair statistics only need two
   * valid pairs inside the window (not a full window of them). Returns null for windows holding
   * fewer pairs than needed and, when `minPeriods` was not given, for windows that are
   * cut off by the start or end of the data.
   */
  private pairedWindow(
    data1: readonly unknown[],
    data2: readonly unknown[],
    i: number
  ): [number[], number[]] | null {
    const [lo, hi] = this.bounds(i, data1.length);
    // Without an explicit minPeriods the window must lie fully inside the data.
    if (!this.explicitMinPeriods && hi - lo + 1 < this.window) return null;
    const v1: number[] = [];
    const v2: number[] = [];
    for (let j = lo; j <= hi; j++) {
      const a = data1[j];
      const b = data2[j];
      if (isValidNumber(a) && isValidNumber(b)) {
        v1.push(a);
        v2.push(b);
      }
    }
    const need = this.explicitMinPeriods ? Math.max(this.minPeriods, 2) : 2;
    return v1.length < need ? null : [v1, v2];
  }

  /**
   * Compute rolling pairwise correlation between two columns.
   *
   * @param col1 - First column name
   * @param col2 - Second column name
   * @returns DataFrame with a single column "{col1}_{col2}" containing rolling Pearson correlation
   */
  corr(col1: string, col2: string): DataFrame {
    const data1 = this.df.getColumnData(col1);
    const data2 = this.df.getColumnData(col2);
    const n = data1.length;
    const result: unknown[] = new Array(n);
    const label = `${col1}_${col2}`;

    for (let i = 0; i < n; i++) {
      const pair = this.pairedWindow(data1, data2, i);
      if (pair === null) {
        result[i] = null;
        continue;
      }
      const [v1, v2] = pair;
      const m1 = compensatedSum(v1) / v1.length;
      const m2 = compensatedSum(v2) / v2.length;
      let num = 0;
      let d1 = 0;
      let d2 = 0;
      for (let k = 0; k < v1.length; k++) {
        const diff1 = (v1[k] as number) - m1;
        const diff2 = (v2[k] as number) - m2;
        num += diff1 * diff2;
        d1 += diff1 * diff1;
        d2 += diff2 * diff2;
      }
      result[i] =
        d1 === 0 || d2 === 0 ? NaN : clampCorrelation(num / (Math.sqrt(d1) * Math.sqrt(d2)));
    }

    return new DataFrame({ [label]: result }, { index: this.df.index, copy: false });
  }

  /**
   * Compute rolling pairwise covariance between two columns.
   *
   * Uses sample covariance (ddof=1).
   *
   * @param col1 - First column name
   * @param col2 - Second column name
   * @returns DataFrame with a single column "{col1}_{col2}" containing rolling covariance
   */
  cov(col1: string, col2: string): DataFrame {
    const data1 = this.df.getColumnData(col1);
    const data2 = this.df.getColumnData(col2);
    const n = data1.length;
    const result: unknown[] = new Array(n);
    const label = `${col1}_${col2}`;

    for (let i = 0; i < n; i++) {
      const pair = this.pairedWindow(data1, data2, i);
      if (pair === null) {
        result[i] = null;
        continue;
      }
      const [v1, v2] = pair;
      const m1 = compensatedSum(v1) / v1.length;
      const m2 = compensatedSum(v2) / v2.length;
      let covSum = 0;
      for (let k = 0; k < v1.length; k++) {
        covSum += ((v1[k] as number) - m1) * ((v2[k] as number) - m2);
      }
      result[i] = covSum / (v1.length - 1);
    }

    return new DataFrame({ [label]: result }, { index: this.df.index, copy: false });
  }
}
