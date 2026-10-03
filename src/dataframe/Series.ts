import { DataValidationError, IndexError, InvalidParameterError } from "../core/errors/index";
import { type Tensor, tensor } from "../ndarray/index";
import { DateTimeAccessor } from "./DateTimeAccessor";
import { StringAccessor } from "./StringAccessor";
import type { FillnaMethodOptions, SeriesOptions, ValueCountsOptions } from "./types";
import {
  cellText,
  checkFillLimit,
  checkFillMethod,
  compareStrings,
  compensatedSum,
  createKey,
  isPlainObject,
  propagateFill,
  sumSquaredDeviations,
} from "./utils";

/**
 * Collect the non-missing numbers of a Series. null, undefined and NaN are
 * skipped; any other type raises a DataValidationError naming the method.
 */
function collectNumeric(data: readonly unknown[], method: string): number[] {
  const out: number[] = [];
  for (const value of data) {
    if (value === null || value === undefined) continue;
    if (typeof value !== "number") {
      throw new DataValidationError(`Series.${method}() only works on numeric data`);
    }
    if (!Number.isNaN(value)) out.push(value);
  }
  return out;
}

function validateDdof(ddof: number): void {
  if (!Number.isInteger(ddof) || ddof < 0) {
    throw new InvalidParameterError("ddof must be a non-negative integer", "ddof", ddof);
  }
}

/** True for null, undefined, NaN and invalid Dates, which sort after every other value. */
function isMissing(value: unknown): boolean {
  return (
    value === null ||
    value === undefined ||
    (typeof value === "number" && Number.isNaN(value)) ||
    (value instanceof Date && Number.isNaN(value.getTime()))
  );
}

/** Ascending comparison of two non-missing values. */
function compareValues(a: unknown, b: unknown): number {
  if (typeof a === "number" && typeof b === "number") return a < b ? -1 : a > b ? 1 : 0;
  if (typeof a === "string" && typeof b === "string") return compareStrings(a, b);
  if (typeof a === "bigint" && typeof b === "bigint") return a < b ? -1 : a > b ? 1 : 0;
  if (typeof a === "boolean" && typeof b === "boolean") return Number(a) - Number(b);
  if (a instanceof Date && b instanceof Date) {
    const ta = a.getTime();
    const tb = b.getTime();
    return ta < tb ? -1 : ta > tb ? 1 : 0;
  }
  return compareStrings(String(a), String(b));
}

/**
 * One-dimensional labeled array capable of holding any data type.
 *
 * A Series is like a column in a spreadsheet or database table. It combines:
 * - An array of data values
 * - An array of index labels (can be strings or numbers)
 * - An optional name
 *
 * @template T - The type of data stored in the Series
 *
 * @example
 * ```ts
 * // Create a numeric series
 * const s = new Series([1, 2, 3, 4], { name: 'numbers' });
 *
 * // Create a series with custom index
 * const s2 = new Series(['a', 'b', 'c'], {
 *   index: ['row1', 'row2', 'row3'],
 *   name: 'letters'
 * });
 * ```
 *
 * @see {@link https://deepbox.dev/docs/dataframe-series | Deepbox Series}
 */
export class Series<T = unknown> {
  // Internal storage for the actual data values
  private _data: T[];
  // Internal storage for index labels (can be strings or numbers)
  private _index: (string | number)[];
  // Fast label -> position lookup for O(1) label-based access
  private _indexPos: Map<string | number, number>;
  // Optional name for this Series
  private _name: string | undefined;

  /**
   * Creates a new Series instance.
   *
   * @param data - Array of values to store in the Series
   * @param options - Configuration options
   * @param options.index - Custom index labels (defaults to 0, 1, 2, ...)
   * @param options.name - Optional name for the Series
   *
   * @example
   * ```ts
   * const s = new Series([10, 20, 30], {
   *   index: ['a', 'b', 'c'],
   *   name: 'values'
   * });
   * ```
   */
  constructor(data: T[], options: SeriesOptions = {}) {
    // Store a shallow copy to prevent external mutation of internal state, unless copy=false
    this._data = options.copy === false ? data : [...data];

    // Use provided index or generate default numeric index [0, 1, 2, ...]
    this._index = options.index
      ? options.copy === false
        ? options.index
        : [...options.index]
      : Array.from({ length: this._data.length }, (_, i) => i);

    if (this._index.length !== this._data.length) {
      throw new DataValidationError(
        `Index length (${this._index.length}) must match data length (${this._data.length})`
      );
    }

    // Build index lookup map and enforce unique labels (required for unambiguous label-based access)
    this._indexPos = new Map();
    for (let i = 0; i < this._index.length; i++) {
      const label = this._index[i];
      if (label === undefined) {
        throw new DataValidationError("Index labels cannot be undefined");
      }
      if (this._indexPos.has(label)) {
        throw new DataValidationError(`Duplicate index label '${String(label)}' is not supported`);
      }
      this._indexPos.set(label, i);
    }

    // Store the optional name
    this._name = options.name;
  }

  /**
   * Get the underlying data array.
   *
   * @returns Read-only view of the data array
   */
  get data(): readonly T[] {
    return this._data;
  }

  /**
   * Get the index labels.
   *
   * @returns Read-only view of the index array
   */
  get index(): readonly (string | number)[] {
    return this._index;
  }

  /**
   * Get the Series name.
   *
   * @returns The name of this Series, or undefined if not set
   */
  get name(): string | undefined {
    return this._name;
  }

  /**
   * Access string methods on this Series.
   *
   * Returns a {@link StringAccessor} that provides vectorized string
   * operations. Each method operates element-wise, propagating null
   * values as null.
   *
   * @returns StringAccessor for this Series
   *
   * @example
   * ```ts
   * const s = new Series(['hello', 'world', null]);
   * s.str.upper();        // Series(['HELLO', 'WORLD', null])
   * s.str.contains('lo'); // Series([true, false, null])
   * ```
   */
  get str(): StringAccessor {
    return new StringAccessor(this);
  }

  /**
   * Access datetime methods on this Series.
   *
   * Returns a {@link DateTimeAccessor} that provides vectorized date/time
   * operations. Each method operates element-wise, propagating null
   * values as null.
   *
   * @returns DateTimeAccessor for this Series
   *
   * @example
   * ```ts
   * const s = new Series([new Date('2024-01-15'), new Date('2024-06-20')]);
   * s.dt.year();       // Series([2024, 2024])
   * s.dt.month();      // Series([1, 6])
   * s.dt.dayOfWeek();  // Series([0, 3])  (Monday is 0)
   * ```
   */
  get dt(): DateTimeAccessor {
    return new DateTimeAccessor(this);
  }

  /**
   * Get the number of elements in the Series.
   *
   * @returns Length of the Series
   */
  get length(): number {
    return this._data.length;
  }

  /**
   * Get a value by label.
   *
   * This method is an alias for `loc()`. It performs strict label-based lookup.
   * For positional access, use `iloc()`.
   *
   * @param label - The index label to look up
   * @returns The value at that label, or undefined if not found
   *
   * @example
   * ```ts
   * const s = new Series([10, 20, 30], { index: ['a', 'b', 'c'] });
   * s.get('a');  // 10
   * s.get('z');  // undefined
   * ```
   */
  get(label: number | string): T | undefined {
    const position = this._indexPos.get(label);
    return position === undefined ? undefined : this._data[position];
  }

  /**
   * Access a value by label (label-based indexing).
   *
   * @param label - The index label to look up
   * @returns The value at that label, or undefined if not found
   *
   * @example
   * ```ts
   * const s = new Series([10, 20], { index: ['a', 'b'] });
   * s.loc('a');  // 10
   * ```
   */
  loc(label: string | number): T | undefined {
    const position = this._indexPos.get(label);
    return position === undefined ? undefined : this._data[position];
  }

  /**
   * Access a value by integer position (position-based indexing).
   *
   * @param position - The integer position (0-based)
   * @returns The value at that position
   * @throws {InvalidParameterError} If position is not an integer
   * @throws {IndexError} If position is out of bounds
   *
   * @example
   * ```ts
   * const s = new Series([10, 20, 30]);
   * s.iloc(0);  // 10
   * s.iloc(2);  // 30
   * ```
   */
  iloc(position: number): T | undefined {
    if (!Number.isInteger(position)) {
      throw new InvalidParameterError("position must be an integer", "position", position);
    }
    if (this._data.length === 0) {
      throw new IndexError(`Series is empty`, {
        index: position,
        validRange: [0, 0],
      });
    }
    if (position < 0 || position >= this._data.length) {
      throw new IndexError(`Position ${position} is out of bounds (0-${this._data.length - 1})`, {
        index: position,
        validRange: [0, this._data.length - 1],
      });
    }
    // Direct array access by position
    return this._data[position];
  }

  /**
   * Return the first n elements.
   *
   * @param n - Number of elements to return (default: 5)
   * @returns New Series with the first n elements
   *
   * @example
   * ```ts
   * const s = new Series([1, 2, 3, 4, 5, 6]);
   * s.head(3);  // Series([1, 2, 3])
   * ```
   */
  head(n: number = 5): Series<T> {
    if (!Number.isFinite(n) || !Number.isInteger(n) || n < 0) {
      throw new InvalidParameterError("n must be a non-negative integer", "n", n);
    }
    // Slice both data and index from start to n
    const options: SeriesOptions = {
      index: this._index.slice(0, n),
    };
    if (this._name !== undefined) {
      options.name = this._name;
    }
    return new Series(this._data.slice(0, n), options);
  }

  /**
   * Return the last n elements.
   *
   * @param n - Number of elements to return (default: 5)
   * @returns New Series with the last n elements
   *
   * @example
   * ```ts
   * const s = new Series([1, 2, 3, 4, 5, 6]);
   * s.tail(3);  // Series([4, 5, 6])
   * ```
   */
  tail(n: number = 5): Series<T> {
    if (!Number.isFinite(n) || !Number.isInteger(n) || n < 0) {
      throw new InvalidParameterError("n must be a non-negative integer", "n", n);
    }
    // Clamp at 0: slice() with a negative start would count from the end.
    const sliceStart = Math.max(0, this._data.length - n);
    const options: SeriesOptions = {
      index: this._index.slice(sliceStart),
    };
    if (this._name !== undefined) {
      options.name = this._name;
    }
    return new Series(this._data.slice(sliceStart), options);
  }

  /**
   * Filter Series by a boolean predicate function.
   *
   * Filters both data AND index to maintain alignment.
   *
   * @param predicate - Function that returns true for elements to keep
   * @returns New Series with only elements that passed the predicate
   *
   * @example
   * ```ts
   * const s = new Series([1, 2, 3, 4, 5]);
   * s.filter(x => x > 2);  // Series([3, 4, 5])
   * ```
   */
  filter(predicate: (value: T, index: number) => boolean): Series<T> {
    // Filter data and collect corresponding indices
    const filteredData: T[] = [];
    const filteredIndex: (string | number)[] = [];

    // Iterate through data and keep matching elements + their indices
    let dataIndex = 0;
    for (const dataItem of this._data) {
      const indexItem = this._index[dataIndex];
      if (indexItem === undefined) {
        throw new DataValidationError("Index labels cannot be undefined");
      }

      if (predicate(dataItem, dataIndex)) {
        filteredData.push(dataItem);
        filteredIndex.push(indexItem);
      }
      dataIndex++;
    }

    // Create new Series with aligned data and index
    const options: SeriesOptions = {
      index: filteredIndex,
    };
    if (this._name !== undefined) {
      options.name = this._name;
    }
    return new Series(filteredData, options);
  }

  /**
   * Transform each element using a mapping function.
   *
   * @template U - The type of the transformed values
   * @param fn - Function to apply to each element
   * @returns New Series with transformed values
   *
   * @example
   * ```ts
   * const s = new Series([1, 2, 3]);
   * s.map(x => x * 2);  // Series([2, 4, 6])
   * ```
   */
  map<U>(fn: (value: T, index: number) => U): Series<U> {
    // Map over data, preserving index and name
    const options: SeriesOptions = {
      index: this._index,
    };
    if (this._name !== undefined) {
      options.name = this._name;
    }
    return new Series(this._data.map(fn), options);
  }

  /**
   * Sort the Series values.
   *
   * Preserves index-value mapping. The sort is stable. null, undefined, NaN and
   * invalid Dates always go last, in both directions. Numbers, bigints, booleans and Dates
   * compare by value; strings compare by UTF-16 code unit (locale
   * independent, like pandas, so "B" sorts before "a").
   *
   * @param ascending - Sort in ascending order (default: true)
   * @returns New sorted Series with index reordered to match
   *
   * @example
   * ```ts
   * const s = new Series([3, 1, 2], { index: ['a', 'b', 'c'] });
   * s.sort();  // Series([1, 2, 3]) with index ['b', 'c', 'a']
   * ```
   */
  sort(ascending: boolean = true): Series<T> {
    const order = Array.from({ length: this._data.length }, (_, i) => i);
    const data = this._data;
    order.sort((i, j) => {
      const a = data[i];
      const b = data[j];
      const aMissing = isMissing(a);
      const bMissing = isMissing(b);
      if (aMissing || bMissing) {
        if (aMissing && bMissing) return 0;
        return aMissing ? 1 : -1;
      }
      const c = compareValues(a, b);
      return ascending ? c : -c;
    });

    const sortedData = order.map((i) => data[i] as T);
    const sortedIndex = order.map((i) => this._index[i] as string | number);

    const options: SeriesOptions = {
      index: sortedIndex,
    };
    if (this._name !== undefined) {
      options.name = this._name;
    }
    return new Series(sortedData, options);
  }

  /**
   * Get unique values in the Series.
   *
   * @returns Array of unique values (order preserved)
   *
   * @example
   * ```ts
   * const s = new Series([1, 2, 2, 3, 1]);
   * s.unique();  // [1, 2, 3]
   * ```
   */
  unique(): T[] {
    // Use Set to remove duplicates, then convert back to array
    return [...new Set(this._data)];
  }

  /**
   * Count occurrences of unique values.
   *
   * Returns a Series where the index is the unique values and the data is
   * their counts, sorted by count in descending order. Values with equal counts
   * keep the order of their first appearance.
   *
   * Like pandas, null, undefined and NaN are left out unless `dropna` is false.
   * When they are kept, they appear in the index as the labels `"null"`,
   * `"undefined"` and `NaN`.
   *
   * With `normalize: true` the data are relative frequencies (they sum to 1 over the values that
   * are counted) and the Series is named `proportion`. `sort: false` keeps the order of first
   * appearance, and `ascending: true` lists the rarest values first.
   *
   * @param dropna - Exclude null, undefined and NaN (default: true), or an options object
   *   `{ normalize?, dropna?, sort?, ascending? }` (see {@link ValueCountsOptions})
   * @returns Series where index is unique values and data is their counts
   * @throws {DataValidationError} If the Series holds values other than strings and numbers
   *
   * @example
   * ```ts
   * const s = new Series(['a', 'b', 'a', 'c', 'a']);
   * s.valueCounts();  // Series([3, 1, 1]) with index ['a', 'b', 'c']
   * s.valueCounts({ normalize: true });  // Series([0.6, 0.2, 0.2])
   * ```
   */
  valueCounts(dropna: boolean | ValueCountsOptions = true): Series<number> {
    const options: ValueCountsOptions = typeof dropna === "object" ? dropna : { dropna };
    const skipMissing = options.dropna ?? true;
    const normalize = options.normalize ?? false;
    const sort = options.sort ?? true;
    const ascending = options.ascending ?? false;
    for (const [name, flag] of [
      ["dropna", skipMissing],
      ["normalize", normalize],
      ["sort", sort],
      ["ascending", ascending],
    ] as const) {
      if (typeof flag !== "boolean") {
        throw new InvalidParameterError(`${name} must be a boolean`, name, flag);
      }
    }
    // Validate types: must be string or number
    for (const v of this._data) {
      if (typeof v !== "string" && typeof v !== "number" && v !== null && v !== undefined) {
        throw new DataValidationError("Series.valueCounts() only supports Series<string | number>");
      }
    }

    const counts = new Map<string, number>();
    const keyToValue = new Map<string, T>();
    let counted = 0;

    for (const v of this._data) {
      if (skipMissing && isMissing(v)) continue;
      counted++;
      const key = createKey(v);
      counts.set(key, (counts.get(key) ?? 0) + 1);
      if (!keyToValue.has(key)) {
        keyToValue.set(key, v);
      }
    }

    // Array.prototype.sort is stable, so ties keep first-appearance order.
    const sortedKeys = [...counts.keys()];
    if (sort) {
      sortedKeys.sort((a, b) =>
        ascending
          ? (counts.get(a) ?? 0) - (counts.get(b) ?? 0)
          : (counts.get(b) ?? 0) - (counts.get(a) ?? 0)
      );
    }

    const values = sortedKeys.map((k) => {
      const count = counts.get(k) ?? 0;
      return normalize ? count / counted : count;
    });
    const used = new Set<string | number>();
    const index = sortedKeys.map((k) => {
      const val = keyToValue.get(k);
      if (typeof val === "string" || typeof val === "number") {
        used.add(val);
        return val;
      }
      return String(val);
    });
    // A missing-value label such as "null" must not collide with a real string label.
    for (let i = 0; i < index.length; i++) {
      const label = index[i] as string | number;
      const val = keyToValue.get(sortedKeys[i] as string);
      if (val === null || val === undefined) {
        let candidate: string = String(label);
        while (used.has(candidate)) candidate = `${candidate} (missing)`;
        used.add(candidate);
        index[i] = candidate;
      }
    }

    const suffix = normalize ? "proportion" : "counts";
    return new Series(values, {
      index: index,
      name: this._name ? `${this._name}_${suffix}` : suffix,
    });
  }

  /**
   * Fill missing values (null, undefined, NaN and invalid Dates).
   *
   * Pass a value to replace every missing entry with it, or `{ method: "ffill" | "bfill", limit? }`
   * to copy the previous or next valid value (`"pad"` and `"backfill"` are accepted aliases). An
   * object whose only keys are `method` and `limit` is read as the method form, so such an object
   * cannot be used as a fill value.
   *
   * @param value - Fill value, or `{ method, limit? }`
   * @returns New Series with the same index and name
   * @throws {InvalidParameterError} If `method` or `limit` is invalid
   *
   * @example
   * ```ts
   * const s = new Series([1, null, null, 4]);
   * s.fillna(0);                                 // [1, 0, 0, 4]
   * s.fillna({ method: "ffill" });               // [1, 1, 1, 4]
   * s.fillna({ method: "bfill", limit: 1 });     // [1, null, 4, 4]
   * ```
   */
  fillna(options: FillnaMethodOptions): Series<T>;
  fillna<U>(value: U): Series<T | U>;
  fillna(value: unknown): Series<unknown> {
    if (
      isPlainObject(value) &&
      Object.hasOwn(value, "method") &&
      Object.keys(value).every((k) => k === "method" || k === "limit")
    ) {
      const options = value as unknown as FillnaMethodOptions;
      return this.propagate(checkFillMethod(options.method), options.limit);
    }
    return new Series<unknown>(
      this._data.map((v) => (isMissing(v) ? value : v)),
      { index: this._index, ...(this._name !== undefined ? { name: this._name } : {}) }
    );
  }

  /**
   * Fill missing values with the previous valid value (forward fill).
   *
   * Leading missing values stay missing. With `limit`, at most that many consecutive missing values
   * are filled in each gap.
   *
   * @param limit - Longest run of missing values to fill (default: no limit)
   * @returns New Series with the same index and name
   * @throws {InvalidParameterError} If `limit` is not a positive integer
   *
   * @example
   * ```ts
   * new Series([null, 1, null, null]).ffill();  // [null, 1, 1, 1]
   * ```
   */
  ffill(limit?: number): Series<T> {
    return this.propagate("ffill", limit);
  }

  /**
   * Fill missing values with the next valid value (backward fill).
   *
   * Trailing missing values stay missing. With `limit`, at most that many consecutive missing values
   * are filled in each gap, counted back from the next valid value.
   *
   * @param limit - Longest run of missing values to fill (default: no limit)
   * @returns New Series with the same index and name
   * @throws {InvalidParameterError} If `limit` is not a positive integer
   *
   * @example
   * ```ts
   * new Series([null, null, 3, null]).bfill();  // [3, 3, 3, null]
   * ```
   */
  bfill(limit?: number): Series<T> {
    return this.propagate("bfill", limit);
  }

  private propagate(direction: "ffill" | "bfill", limit: number | undefined): Series<T> {
    const filled = propagateFill(this._data, direction, checkFillLimit(limit), isMissing) as T[];
    return new Series<T>(filled, {
      index: this._index,
      ...(this._name !== undefined ? { name: this._name } : {}),
    });
  }

  /**
   * Calculate the sum of all values.
   *
   * Skips null, undefined, and NaN values. Uses compensated summation, so the
   * result does not drift on long or badly scaled inputs. A Series that holds
   * only missing values sums to 0.
   *
   * @returns Sum of all numeric values.
   * @throws {DataValidationError} If Series is empty or contains non-numeric data
   *
   * @example
   * ```ts
   * const s = new Series([1, 2, null, 3, 4]);
   * s.sum();  // 10
   * ```
   */
  sum(): number {
    if (this._data.length === 0) {
      throw new DataValidationError("Cannot get sum of empty Series");
    }
    return compensatedSum(collectNumeric(this._data, "sum"));
  }

  /**
   * Calculate the arithmetic mean (average) of all values.
   *
   * Skips null, undefined, and NaN values. Returns NaN when no numeric value
   * remains.
   *
   * @returns Mean of all numeric values.
   * @throws {DataValidationError} If Series is empty or contains non-numeric data
   *
   * @example
   * ```ts
   * const s = new Series([1, 2, null, 3, 4]);
   * s.mean();  // 2.5
   * ```
   */
  mean(): number {
    if (this._data.length === 0) {
      throw new DataValidationError("Cannot get mean of empty Series");
    }
    const numericData = collectNumeric(this._data, "mean");
    return numericData.length > 0 ? compensatedSum(numericData) / numericData.length : NaN;
  }

  /**
   * Calculate the median (middle value) of all values.
   *
   * Skips null, undefined, and NaN values.
   * For even-length Series, returns the average of the two middle values.
   *
   * @returns Median value, or NaN when no numeric value remains.
   * @throws {DataValidationError} If Series is empty or contains non-numeric data
   *
   * @example
   * ```ts
   * const s = new Series([1, 2, 3, 4, 5]);
   * s.median();  // 3
   * ```
   */
  median(): number {
    if (this._data.length === 0) {
      throw new DataValidationError("Cannot get median of empty Series");
    }

    const sorted = Float64Array.from(collectNumeric(this._data, "median")).sort();
    if (sorted.length === 0) {
      return NaN;
    }

    const middle = Math.floor(sorted.length / 2);
    if (sorted.length % 2 === 0) {
      const lo = sorted[middle - 1] as number;
      const hi = sorted[middle] as number;
      if (lo === hi) return lo;
      const mid = (lo + hi) / 2;
      // Halve before adding when the plain sum overflows to Infinity.
      return Number.isFinite(mid) || !Number.isFinite(lo) || !Number.isFinite(hi)
        ? mid
        : lo / 2 + hi / 2;
    }
    return sorted[middle] as number;
  }

  /**
   * Calculate the standard deviation of all values.
   *
   * Skips null, undefined, and NaN values.
   * Uses the sample standard deviation (divides by n - ddof, default n - 1).
   *
   * @param ddof - Delta degrees of freedom (default: 1)
   * @returns Standard deviation, or NaN when n <= ddof.
   * @throws {DataValidationError} If Series is empty or contains non-numeric data
   * @throws {InvalidParameterError} If ddof is not a non-negative integer
   *
   * @example
   * ```ts
   * const s = new Series([2, 4, 6, 8]);
   * s.std();  // ~2.58
   * ```
   */
  std(ddof: number = 1): number {
    validateDdof(ddof);
    return Math.sqrt(this.variance("std", "Cannot get std of empty Series", ddof));
  }

  /**
   * Calculate the variance of all values.
   *
   * Skips null, undefined, and NaN values.
   * Uses the sample variance (divides by n - ddof, default n - 1).
   *
   * @param ddof - Delta degrees of freedom (default: 1)
   * @returns Variance, or NaN when n <= ddof.
   * @throws {DataValidationError} If Series is empty or contains non-numeric data
   * @throws {InvalidParameterError} If ddof is not a non-negative integer
   *
   * @example
   * ```ts
   * const s = new Series([2, 4, 6, 8]);
   * s.var();  // ~6.67
   * ```
   */
  var(ddof: number = 1): number {
    validateDdof(ddof);
    return this.variance("var", "Cannot get variance of empty Series", ddof);
  }

  private variance(method: string, emptyMessage: string, ddof: number): number {
    if (this._data.length === 0) {
      throw new DataValidationError(emptyMessage);
    }
    const numericData = collectNumeric(this._data, method);
    if (numericData.length <= ddof) {
      return NaN;
    }
    // The corrected two-pass formula can land a hair below zero for constant data.
    return Math.max(0, sumSquaredDeviations(numericData)) / (numericData.length - ddof);
  }

  /**
   * Find the minimum value in the Series.
   *
   * Skips null, undefined, and NaN values.
   *
   * @returns Minimum value, or NaN when no numeric value remains.
   * @throws {DataValidationError} If Series is empty or contains non-numeric data
   *
   * @example
   * ```ts
   * const s = new Series([5, 2, 8, 1, 9]);
   * s.min();  // 1
   * ```
   */
  min(): number {
    if (this._data.length === 0) {
      throw new DataValidationError("Cannot get min of empty Series");
    }
    const numericData = collectNumeric(this._data, "min");
    if (numericData.length === 0) return NaN;
    let minVal = Infinity;
    for (const v of numericData) {
      if (v < minVal) minVal = v;
    }
    return minVal;
  }

  /**
   * Find the maximum value in the Series.
   *
   * Skips null, undefined, and NaN values.
   *
   * @returns Maximum value, or NaN when no numeric value remains.
   * @throws {DataValidationError} If Series is empty or contains non-numeric data
   *
   * @example
   * ```ts
   * const s = new Series([5, 2, 8, 1, 9]);
   * s.max();  // 9
   * ```
   */
  max(): number {
    if (this._data.length === 0) {
      throw new DataValidationError("Cannot get max of empty Series");
    }
    const numericData = collectNumeric(this._data, "max");
    if (numericData.length === 0) return NaN;
    let maxVal = -Infinity;
    for (const v of numericData) {
      if (v > maxVal) maxVal = v;
    }
    return maxVal;
  }

  /**
   * Boolean Series marking missing values (null, undefined, NaN and invalid Dates).
   *
   * This is the same notion of "missing" that sort, valueCounts, sum, mean and the
   * other statistics use. The index and name are kept.
   *
   * @returns Series of booleans (true = missing)
   *
   * @example
   * ```ts
   * new Series([1, null, NaN, 4]).isnull().toArray();  // [false, true, true, false]
   * ```
   */
  isnull(): Series<boolean> {
    return this.flagMissing(true);
  }

  /**
   * Boolean Series marking present values, the complement of {@link Series.isnull}.
   *
   * @returns Series of booleans (true = present)
   *
   * @example
   * ```ts
   * new Series([1, null, NaN, 4]).notnull().toArray();  // [true, false, false, true]
   * ```
   */
  notnull(): Series<boolean> {
    return this.flagMissing(false);
  }

  private flagMissing(missing: boolean): Series<boolean> {
    const options: SeriesOptions = { index: this._index };
    if (this._name !== undefined) options.name = this._name;
    return new Series(
      this._data.map((v) => isMissing(v) === missing),
      options
    );
  }

  /**
   * Convert the Series to a plain JavaScript array.
   *
   * Returns a shallow copy of the data.
   *
   * @returns Array copy of the data
   *
   * @example
   * ```ts
   * const s = new Series([1, 2, 3]);
   * const arr = s.toArray();  // [1, 2, 3]
   * ```
   */
  toArray(): T[] {
    // Return a copy to prevent external mutation
    return [...this._data];
  }

  /**
   * Convert the Series to a 1D ndarray Tensor.
   *
   * Null and undefined become NaN. The tensor is float32 unless `options.dtype`
   * says otherwise; float32 only keeps 24 bits of precision, so integers above
   * 2^24 and values that need more than about 7 significant digits are rounded.
   * Pass `{ dtype: "float64" }` to keep the stored doubles exactly (this matches
   * `DataFrame.toTensor`).
   *
   * @param options - Optional settings
   * @param options.dtype - Element type of the tensor (default: "float32")
   * @returns Tensor containing the Series data
   * @throws {DataValidationError} If data cannot be converted to Tensor
   * @throws {InvalidParameterError} If dtype is not "float32" or "float64"
   *
   * @example
   * ```ts
   * import { Series } from 'deepbox/dataframe';
   *
   * const s = new Series([1, 2, 3, 4]);
   * const t = s.toTensor();  // Tensor([1, 2, 3, 4])
   * const t64 = s.toTensor({ dtype: "float64" });
   * ```
   */
  toTensor(options: { readonly dtype?: "float32" | "float64" } = {}): Tensor {
    const dtype = options.dtype ?? "float32";
    if (dtype !== "float32" && dtype !== "float64") {
      throw new InvalidParameterError('dtype must be "float32" or "float64"', "dtype", dtype);
    }
    const numeric: number[] = [];
    for (const v of this._data) {
      if (typeof v === "number") {
        numeric.push(v);
      } else if (v === null || v === undefined) {
        numeric.push(NaN);
      } else {
        throw new DataValidationError(
          "Series.toTensor() only works on numeric data (or null/undefined)"
        );
      }
    }
    return tensor(numeric, { dtype });
  }

  /**
   * Return a human-readable string representation of this Series.
   *
   * Each row is printed as `index  value`, followed by a footer with the
   * optional name and the length. Series longer than `maxRows` show the first
   * and last `floor(maxRows / 2)` rows with an ellipsis row in between.
   *
   * @param maxRows - Maximum rows to display before summarizing (default: 20).
   *   Pass `Infinity` to show every row.
   * @returns Formatted string representation
   * @throws {InvalidParameterError} If maxRows is negative or not an integer
   *
   * @example
   * ```ts
   * const s = new Series([10, 20, 30], { name: 'values' });
   * s.toString();
   * // "0  10\n1  20\n2  30\nName: values, Length: 3"
   * ```
   */
  toString(maxRows = 20): string {
    if (maxRows !== Number.POSITIVE_INFINITY && (!Number.isInteger(maxRows) || maxRows < 0)) {
      throw new InvalidParameterError("maxRows must be a non-negative integer", "maxRows", maxRows);
    }
    const n = this._data.length;
    const half = Math.floor(maxRows / 2);
    const showAll = n <= maxRows;

    const rows: string[][] = [];

    const topCount = showAll ? n : half;
    const bottomCount = showAll ? 0 : half;

    for (let i = 0; i < topCount; i++) {
      const idx = this._index[i];
      const val = this._data[i];
      rows.push([String(idx ?? i), cellText(val)]);
    }

    if (!showAll) {
      rows.push(["...", "..."]);
      for (let i = n - bottomCount; i < n; i++) {
        const idx = this._index[i];
        const val = this._data[i];
        rows.push([String(idx ?? i), cellText(val)]);
      }
    }

    // Calculate column widths
    let idxWidth = 0;
    let valWidth = 0;
    for (const [idx, val] of rows) {
      if ((idx ?? "").length > idxWidth) idxWidth = (idx ?? "").length;
      if ((val ?? "").length > valWidth) valWidth = (val ?? "").length;
    }

    const lines: string[] = [];
    for (const [idx, val] of rows) {
      lines.push(`${(idx ?? "").padStart(idxWidth)}  ${val ?? ""}`);
    }

    // Footer
    const parts: string[] = [];
    if (this._name !== undefined) parts.push(`Name: ${this._name}`);
    parts.push(`Length: ${n}`);
    lines.push(parts.join(", "));

    return lines.join("\n");
  }
}
