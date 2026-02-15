import { DataValidationError, IndexError, InvalidParameterError } from "../core/errors/index.js";
import { type Tensor, tensor } from "../ndarray/index.js";
import type { SeriesOptions } from "./types.js";
import { createKey } from "./utils.js";

/**
 * One-dimensional labeled array capable of holding any data type.
 *
 * A Series is like a column in a spreadsheet or database table. It combines:
 * - An array of data values
 * - An array of index labels (can be strings or numbers)
 * - An optional name
 *
 * A one-dimensional labeled array. @see { https://deepbox.dev/docs/dataframe-series | Deepbox Series}
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
   * @returns The value at that position, or undefined if out of bounds
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
    const sliceStart = this._data.length - n;
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
   * Preserves index-value mapping by sorting `[value, index]` pairs.
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
    // Create array of [value, index] pairs to maintain association
    const paired: Array<[T, string | number]> = [];
    let pairIndex = 0;
    for (const value of this._data) {
      const idx = this._index[pairIndex];
      if (idx === undefined) {
        throw new DataValidationError("Index labels cannot be undefined");
      }
      paired.push([value, idx]);
      pairIndex++;
    }

    // Sort the pairs by value
    paired.sort((a, b) => {
      const aVal = a[0];
      const bVal = b[0];

      // Handle numeric comparison (NaN sorts to end)
      if (typeof aVal === "number" && typeof bVal === "number") {
        const aIsNaN = Number.isNaN(aVal);
        const bIsNaN = Number.isNaN(bVal);
        if (aIsNaN && bIsNaN) return 0;
        if (aIsNaN) return 1;
        if (bIsNaN) return -1;
        return ascending ? aVal - bVal : bVal - aVal;
      }

      // Handle string comparison
      if (typeof aVal === "string" && typeof bVal === "string") {
        return ascending ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
      }

      // Fallback: convert to string and compare
      const aStr = String(aVal);
      const bStr = String(bVal);
      return ascending ? aStr.localeCompare(bStr) : bStr.localeCompare(aStr);
    });

    // Separate back into data and index arrays
    const sortedData = paired.map((p) => p[0]);
    const sortedIndex = paired.map((p) => p[1]);

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
   * Returns a Series where index is the unique values and data is their counts.
   *
   * @returns Series where index is unique values and data is their counts
   *
   * @example
   * ```ts
   * const s = new Series(['a', 'b', 'a', 'c', 'a']);
   * s.valueCounts();  // Series([3, 1, 1]) with index ['a', 'b', 'c']
   * ```
   */
  valueCounts(): Series<number> {
    // Validate types: must be string or number
    for (const v of this._data) {
      if (typeof v !== "string" && typeof v !== "number" && v !== null && v !== undefined) {
        throw new DataValidationError("Series.valueCounts() only supports Series<string | number>");
      }
    }

    const counts = new Map<string, number>();
    const keyToValue = new Map<string, T>();

    for (const v of this._data) {
      const key = createKey(v);
      counts.set(key, (counts.get(key) ?? 0) + 1);
      if (!keyToValue.has(key)) {
        keyToValue.set(key, v);
      }
    }

    // Sort keys by count (descending)
    const sortedKeys = [...counts.keys()].sort((a, b) => {
      const countA = counts.get(a) ?? 0;
      const countB = counts.get(b) ?? 0;
      return countB - countA;
    });

    const values = sortedKeys.map((k) => counts.get(k) ?? 0);
    // Use the original values as index labels.
    const index = sortedKeys.map((k) => {
      const val = keyToValue.get(k);
      if (typeof val === "string" || typeof val === "number") {
        return val;
      }
      return String(val);
    });

    return new Series(values, {
      index: index,
      name: this._name ? `${this._name}_counts` : "counts",
    });
  }

  /**
   * Calculate the sum of all values.
   *
   * Skips null, undefined, and NaN values.
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

    let total = 0;
    for (const val of this._data) {
      if (val === null || val === undefined) continue;
      if (typeof val !== "number") {
        throw new DataValidationError("Series.sum() only works on numeric data");
      }
      if (Number.isNaN(val)) continue;

      total += val;
    }

    return total;
  }

  /**
   * Calculate the arithmetic mean (average) of all values.
   *
   * Skips null, undefined, and NaN values.
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

    let total = 0;
    let count = 0;

    for (const val of this._data) {
      if (val === null || val === undefined) continue;
      if (typeof val !== "number") {
        throw new DataValidationError("Series.mean() only works on numeric data");
      }
      if (Number.isNaN(val)) continue;

      total += val;
      count++;
    }

    return count > 0 ? total / count : NaN;
  }

  /**
   * Calculate the median (middle value) of all values.
   *
   * Skips null, undefined, and NaN values.
   * For even-length Series, returns the average of the two middle values.
   *
   * @returns Median value.
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

    const numericData: number[] = [];
    for (const value of this._data) {
      if (value === null || value === undefined) continue;
      if (typeof value !== "number") {
        throw new DataValidationError("Series.median() only works on numeric data");
      }
      if (!Number.isNaN(value)) {
        numericData.push(value);
      }
    }

    if (numericData.length === 0) {
      return NaN;
    }

    // Create a sorted copy (don't mutate numericData)
    const sorted = [...numericData].sort((a, b) => a - b);

    // Find the middle index
    const middle = Math.floor(sorted.length / 2);

    // If even length, average the two middle values
    // If odd length, return the single middle value
    if (sorted.length % 2 === 0) {
      const val1 = sorted[middle - 1];
      const val2 = sorted[middle];
      if (val1 === undefined || val2 === undefined) {
        return NaN;
      }
      return (val1 + val2) / 2;
    }
    const val = sorted[middle];
    return val !== undefined ? val : NaN;
  }

  /**
   * Calculate the standard deviation of all values.
   *
   * Skips null, undefined, and NaN values.
   * Uses sample standard deviation (divides by n-1).
   *
   * @returns Standard deviation.
   * @throws {DataValidationError} If Series is empty or contains non-numeric data
   *
   * @example
   * ```ts
   * const s = new Series([2, 4, 6, 8]);
   * s.std();  // ~2.58
   * ```
   */
  std(): number {
    if (this._data.length === 0) {
      throw new DataValidationError("Cannot get std of empty Series");
    }

    const numericData: number[] = [];
    for (const value of this._data) {
      if (value === null || value === undefined) continue;
      if (typeof value !== "number") {
        throw new DataValidationError("Series.std() only works on numeric data");
      }
      if (!Number.isNaN(value)) {
        numericData.push(value);
      }
    }

    // Need at least 2 values for sample std
    if (numericData.length < 2) {
      return NaN;
    }

    // Calculate mean first
    const sum = numericData.reduce((acc, val) => acc + val, 0);
    const meanVal = sum / numericData.length;

    // Sum of squared differences from mean
    let sumSquaredDiff = 0;
    for (const val of numericData) {
      const diff = val - meanVal;
      sumSquaredDiff += diff * diff;
    }

    // Sample standard deviation: divide by (n-1) then sqrt
    return Math.sqrt(sumSquaredDiff / (numericData.length - 1));
  }

  /**
   * Calculate the variance of all values.
   *
   * Skips null, undefined, and NaN values.
   * Uses sample variance (divides by n-1).
   *
   * @returns Variance.
   * @throws {DataValidationError} If Series is empty or contains non-numeric data
   *
   * @example
   * ```ts
   * const s = new Series([2, 4, 6, 8]);
   * s.var();  // ~6.67
   * ```
   */
  var(): number {
    if (this._data.length === 0) {
      throw new DataValidationError("Cannot get variance of empty Series");
    }

    const numericData: number[] = [];
    for (const value of this._data) {
      if (value === null || value === undefined) continue;
      if (typeof value !== "number") {
        throw new DataValidationError("Series.var() only works on numeric data");
      }
      if (!Number.isNaN(value)) {
        numericData.push(value);
      }
    }

    // Need at least 2 values for sample variance
    if (numericData.length < 2) {
      return NaN;
    }

    // Calculate mean first
    const sum = numericData.reduce((acc, val) => acc + val, 0);
    const meanVal = sum / numericData.length;

    // Sum of squared differences from mean
    let sumSquaredDiff = 0;
    for (const val of numericData) {
      const diff = val - meanVal;
      sumSquaredDiff += diff * diff;
    }

    // Sample variance: divide by (n-1)
    return sumSquaredDiff / (numericData.length - 1);
  }

  /**
   * Find the minimum value in the Series.
   *
   * Skips null, undefined, and NaN values.
   *
   * @returns Minimum value.
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

    let minVal = Infinity;
    let hasNumeric = false;

    for (const val of this._data) {
      if (val === null || val === undefined) continue;
      if (typeof val !== "number") {
        throw new DataValidationError("Series.min() only works on numeric data");
      }
      if (!Number.isNaN(val)) {
        if (val < minVal) {
          minVal = val;
        }
        hasNumeric = true;
      }
    }

    return hasNumeric ? minVal : NaN;
  }

  /**
   * Find the maximum value in the Series.
   *
   * Skips null, undefined, and NaN values.
   *
   * @returns Maximum value.
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

    let maxVal = -Infinity;
    let hasNumeric = false;

    for (const val of this._data) {
      if (val === null || val === undefined) continue;
      if (typeof val !== "number") {
        throw new DataValidationError("Series.max() only works on numeric data");
      }
      if (!Number.isNaN(val)) {
        if (val > maxVal) {
          maxVal = val;
        }
        hasNumeric = true;
      }
    }

    return hasNumeric ? maxVal : NaN;
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
   * Convert the Series to an ndarray Tensor.
   *
   * Uses the `tensor()` factory function.
   *
   * @returns Tensor containing the Series data
   * @throws {DataValidationError} If data cannot be converted to Tensor
   *
   * @example
   * ```ts
   * import { Series } from 'deepbox/dataframe';
   *
   * const s = new Series([1, 2, 3, 4]);
   * const t = s.toTensor();  // Tensor([1, 2, 3, 4])
   * ```
   */
  toTensor(): Tensor {
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
    return tensor(numeric);
  }

  /**
   * Return a human-readable string representation of this Series.
   *
   * Each row is printed as `index  value`, with an optional name/dtype
   * footer.  Large Series are truncated with an ellipsis.
   *
   * @param maxRows - Maximum rows to display before summarizing (default: 20).
   * @returns Formatted string representation
   *
   * @example
   * ```ts
   * const s = new Series([10, 20, 30], { name: 'values' });
   * s.toString();
   * // "0  10\n1  20\n2  30\nName: values, Length: 3"
   * ```
   */
  toString(maxRows = 20): string {
    const n = this._data.length;
    const half = Math.floor(maxRows / 2);
    const showAll = n <= maxRows;

    const rows: string[][] = [];

    const topCount = showAll ? n : half;
    const bottomCount = showAll ? 0 : half;

    for (let i = 0; i < topCount; i++) {
      const idx = this._index[i];
      const val = this._data[i];
      rows.push([String(idx ?? i), val === null || val === undefined ? "null" : String(val)]);
    }

    if (!showAll) {
      rows.push(["...", "..."]);
      for (let i = n - bottomCount; i < n; i++) {
        const idx = this._index[i];
        const val = this._data[i];
        rows.push([String(idx ?? i), val === null || val === undefined ? "null" : String(val)]);
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
