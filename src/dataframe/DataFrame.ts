import { type Axis, normalizeAxis } from "../core";
import { DataValidationError, IndexError, InvalidParameterError } from "../core/errors/index";
import { reshape, type Tensor, tensor } from "../ndarray/index";
import { __random } from "../random/random";
import { PlotAccessor } from "./PlotAccessor";
import { Series } from "./Series";
import { StyleAccessor } from "./StyleAccessor";
import type { AggregateFunction, DataFrameData, DataFrameOptions } from "./types";
import { createKey, isRecord, isValidNumber } from "./utils";

const isNumberValue = (value: unknown): value is number => typeof value === "number";

const isIndexLabel = (value: unknown): value is string | number =>
  typeof value === "string" || typeof value === "number";

const isStringArray = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every((entry) => typeof entry === "string");

const isIndexLabelArray = (value: unknown): value is (string | number)[] =>
  Array.isArray(value) && value.every(isIndexLabel);

const ensureUniqueLabels = (labels: readonly string[], labelName: string): void => {
  const seen = new Set<string>();
  for (const label of labels) {
    if (seen.has(label)) {
      throw new DataValidationError(`Duplicate ${labelName} '${label}' is not supported`);
    }
    seen.add(label);
  }
};

const toNumericValues = (values: readonly unknown[]): number[] => values.filter(isValidNumber);

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
      if (!(col in data)) {
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
      if (label === undefined) {
        throw new DataValidationError(`Index label at position ${i} is undefined`);
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
        throw new DataValidationError(`Column '${col}' not found in DataFrame data`);
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
      getColumn: (name: string) => {
        const col = this._data.get(name);
        if (!col) return [];
        return col.slice();
      },
      nRows: this._index.length,
    }));
  }

  /**
   * Style accessor. Provides pandas-like conditional formatting with
   * `df.style.highlight_max()`, `df.style.background_gradient()`, etc.
   *
   * @example
   * ```ts
   * const html = df.style.highlight_max().toHTML();
   * ```
   */
  get style(): StyleAccessor {
    return new StyleAccessor(() => ({
      columns: this._columns.slice(),
      getColumn: (name: string) => {
        const col = this._data.get(name);
        if (!col) return [];
        return col.slice();
      },
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
        result[col] = colData[position];
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
        result[col] = colData[position];
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
    const newData: DataFrameData = {};
    for (const col of this._columns) {
      const colData = this._data.get(col);
      newData[col] = colData ? colData.slice(0, n) : [];
    }

    // Create new DataFrame with sliced data and index
    return new DataFrame(newData, {
      columns: this._columns,
      index: this._index.slice(0, n),
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
    const sliceStart = this._index.length - n;
    const newData: DataFrameData = {};
    for (const col of this._columns) {
      const colData = this._data.get(col);
      newData[col] = colData ? colData.slice(sliceStart) : [];
    }

    // Create new DataFrame with sliced data and index
    return new DataFrame(newData, {
      columns: this._columns,
      index: this._index.slice(sliceStart),
    });
  }

  /**
   * Filter rows based on a boolean predicate function.
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
    const filteredData: DataFrameData = {};
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
    // Validate all columns exist
    for (const col of columns) {
      if (!this._data.has(col)) {
        throw new InvalidParameterError(`Column '${col}' not found in DataFrame`, "columns", col);
      }
    }

    // Build new data with only selected columns (slice to avoid shared mutation)
    const newData: DataFrameData = {};
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
    const columnsToKeep = this._columns.filter((col) => !columns.includes(col));

    // Build new data with remaining columns
    const newData: DataFrameData = {};
    for (const col of columnsToKeep) {
      const colData = this._data.get(col);
      newData[col] = colData ? [...colData] : [];
    }

    return new DataFrame(newData, {
      columns: columnsToKeep,
      index: this._index,
    });
  }

  /**
   * Sort DataFrame by one or more columns.
   *
   * @param by - Column name or array of column names to sort by
   * @param ascending - Sort in ascending order (default: true)
   * @returns New sorted DataFrame
   *
   * @example
   * ```ts
   * const df = new DataFrame({ age: [30, 25, 35], name: ['Bob', 'Alice', 'Carol'] });
   * df.sort('age');  // Sorted by age ascending
   * df.sort(['age'], false);  // Sorted by age descending
   * ```
   */
  sort(by: string | string[], ascending: boolean = true): DataFrame {
    const sortCols = Array.isArray(by) ? by : [by];

    // Validate sort columns exist
    for (const col of sortCols) {
      if (!this._data.has(col)) {
        throw new InvalidParameterError(`Column '${col}' not found in DataFrame`, "by", col);
      }
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
        // ascending/descending (pandas na_position='last' default). Without
        // this, nulls fall into the string fallback and negating for
        // descending puts them first.
        const aMissing =
          aVal === null || aVal === undefined || (isNumberValue(aVal) && Number.isNaN(aVal));
        const bMissing =
          bVal === null || bVal === undefined || (isNumberValue(bVal) && Number.isNaN(bVal));
        if (aMissing || bMissing) {
          if (aMissing && bMissing) continue;
          return aMissing ? 1 : -1;
        }

        // Handle numeric comparison (NaN sorts to end)
        if (isNumberValue(aVal) && isNumberValue(bVal)) {
          const aIsNaN = Number.isNaN(aVal);
          const bIsNaN = Number.isNaN(bVal);
          if (aIsNaN && bIsNaN) continue;
          if (aIsNaN) return 1;
          if (bIsNaN) return -1;
          const diff = aVal - bVal;
          if (diff !== 0) return ascending ? diff : -diff;
        }
        // Handle string comparison
        else if (typeof aVal === "string" && typeof bVal === "string") {
          const cmp = aVal.localeCompare(bVal);
          if (cmp !== 0) return ascending ? cmp : -cmp;
        }
        // Fallback: convert to string
        else {
          const cmp = String(aVal).localeCompare(String(bVal));
          if (cmp !== 0) return ascending ? cmp : -cmp;
        }
      }
      return 0;
    });

    // Build sorted data by gathering from original columns using sorted indices
    const sortedData: DataFrameData = {};
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
    });
  }

  /**
   * Group DataFrame by one or more columns.
   *
   * Returns a DataFrameGroupBy object for performing aggregations.
   *
   * @param by - Column name or array of column names to group by
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
   * ```
   */
  groupBy(by: string | string[]): DataFrameGroupBy {
    return new DataFrameGroupBy(this, by);
  }

  /**
   * Join with another DataFrame using SQL-style join.
   *
   * Uses hash join algorithm for O(n + m) time complexity.
   * Optimized for large datasets with minimal memory overhead.
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
    if (!["inner", "left", "right", "outer"].includes(how)) {
      throw new InvalidParameterError(
        'how must be one of "inner", "left", "right", or "outer"',
        "how",
        how
      );
    }
    // Validate join column exists in both DataFrames
    if (!this._columns.includes(on)) {
      throw new InvalidParameterError(`Join column '${on}' not found in left DataFrame`, "on", on);
    }
    if (!other._columns.includes(on)) {
      throw new InvalidParameterError(`Join column '${on}' not found in right DataFrame`, "on", on);
    }

    // Build hash table from right DataFrame for O(1) lookups
    // Hash map: key value -> array of row indices in right DataFrame
    const rightHash = new Map<string, number[]>();
    const rightData = other._data.get(on) ?? [];

    // Build phase: O(m) where m is rows in right DataFrame
    for (let i = 0; i < rightData.length; i++) {
      const val = rightData[i];
      if (val === null || val === undefined) continue;
      const key = createKey(val);
      const indices = rightHash.get(key) ?? [];
      indices.push(i);
      rightHash.set(key, indices);
    }

    // Track which right rows were matched (for outer join)
    const matchedRightRows = new Set<number>();

    // Detect overlapping non-key columns and build output column mappings
    const rightNonKeyColumns = other._columns.filter((col) => col !== on);
    const overlapping = new Set<string>();
    for (const col of rightNonKeyColumns) {
      if (this._columns.includes(col)) {
        overlapping.add(col);
      }
    }

    // Build left output column names (suffix overlapping ones with _left)
    const leftOutputNames: string[] = [];
    for (const col of this._columns) {
      if (col !== on && overlapping.has(col)) {
        leftOutputNames.push(`${col}_left`);
      } else {
        leftOutputNames.push(col);
      }
    }

    // Build right output column names (suffix overlapping ones with _right, skip join key)
    const rightOutputNames: string[] = [];
    for (const col of rightNonKeyColumns) {
      if (overlapping.has(col)) {
        rightOutputNames.push(`${col}_right`);
      } else {
        rightOutputNames.push(col);
      }
    }

    const allColumns = [...leftOutputNames, ...rightOutputNames];

    // Result data structure
    const resultData: DataFrameData = {};

    // Initialize result columns
    for (const col of allColumns) {
      resultData[col] = [];
    }

    const leftData = this._data.get(on) ?? [];

    // Probe phase: O(n) where n is rows in left DataFrame
    for (let i = 0; i < leftData.length; i++) {
      const leftKey = createKey(leftData[i]);
      const matches = rightHash.get(leftKey) ?? [];

      if (matches.length > 0) {
        // Found match(es) in right DataFrame
        for (const rightIdx of matches) {
          matchedRightRows.add(rightIdx);

          // Add data from left DataFrame
          for (let j = 0; j < this._columns.length; j++) {
            const originalCol = this._columns[j];
            const outputCol = leftOutputNames[j];
            if (originalCol && outputCol) {
              const colData = this._data.get(originalCol);
              resultData[outputCol]?.push(colData?.[i] ?? null);
            }
          }

          // Add data from right DataFrame (skip join column to avoid duplicate)
          for (let j = 0; j < rightNonKeyColumns.length; j++) {
            const originalCol = rightNonKeyColumns[j];
            const outputCol = rightOutputNames[j];
            if (originalCol && outputCol) {
              const colData = other._data.get(originalCol);
              resultData[outputCol]?.push(colData?.[rightIdx] ?? null);
            }
          }
        }
      } else if (how === "left" || how === "outer") {
        // No match - include left row with nulls for right columns
        for (let j = 0; j < this._columns.length; j++) {
          const originalCol = this._columns[j];
          const outputCol = leftOutputNames[j];
          if (originalCol && outputCol) {
            const colData = this._data.get(originalCol);
            resultData[outputCol]?.push(colData?.[i] ?? null);
          }
        }
        for (const col of rightOutputNames) {
          resultData[col]?.push(null);
        }
      }
      // For 'right' and 'inner' joins, skip unmatched left rows
    }

    // For right/outer join: add unmatched right rows
    if (how === "right" || how === "outer") {
      for (let i = 0; i < rightData.length; i++) {
        if (!matchedRightRows.has(i)) {
          // Add values for left columns: join column gets right value, others get null
          for (let j = 0; j < this._columns.length; j++) {
            const originalCol = this._columns[j];
            const outputCol = leftOutputNames[j];
            if (originalCol && outputCol) {
              if (originalCol === on) {
                // For join column, use value from right DataFrame
                const colData = other._data.get(on);
                resultData[outputCol]?.push(colData?.[i] ?? null);
              } else {
                resultData[outputCol]?.push(null);
              }
            }
          }
          // Add data from right DataFrame (skip join column as already added)
          for (let j = 0; j < rightNonKeyColumns.length; j++) {
            const originalCol = rightNonKeyColumns[j];
            const outputCol = rightOutputNames[j];
            if (originalCol && outputCol) {
              const colData = other._data.get(originalCol);
              resultData[outputCol]?.push(colData?.[i] ?? null);
            }
          }
        }
      }
    }

    return new DataFrame(resultData, { columns: allColumns });
  }

  /**
   * Merge with another DataFrame using SQL-style merge.
   *
   * More flexible than join() - supports different column names for join keys.
   * Uses hash join algorithm for O(n + m) complexity.
   *
   * @param other - DataFrame to merge with
   * @param options - Merge configuration
   *   - on: Column name to join on (must exist in both DataFrames)
   *   - left_on: Column name in left DataFrame
   *   - right_on: Column name in right DataFrame
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
   *   left_on: 'emp_id',
   *   right_on: 'employee_id',
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
      readonly left_on?: string;
      readonly right_on?: string;
      readonly how?: "inner" | "left" | "right" | "outer";
      readonly suffixes?: readonly [string, string];
    } = {}
  ): DataFrame {
    const how = options.how ?? "inner";
    if (!["inner", "left", "right", "outer"].includes(how)) {
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
    const suffixes = options.suffixes ?? ["_x", "_y"];

    // Determine join columns
    let leftOn: string;
    let rightOn: string;

    if (options.on) {
      if (typeof options.on !== "string") {
        throw new InvalidParameterError("on must be a string", "on", options.on);
      }
      // Same column name in both DataFrames
      if (options.left_on || options.right_on) {
        throw new InvalidParameterError('Cannot specify both "on" and "left_on"/"right_on"');
      }
      leftOn = options.on;
      rightOn = options.on;
    } else if (options.left_on && options.right_on) {
      if (typeof options.left_on !== "string") {
        throw new InvalidParameterError("left_on must be a string", "left_on", options.left_on);
      }
      if (typeof options.right_on !== "string") {
        throw new InvalidParameterError("right_on must be a string", "right_on", options.right_on);
      }
      // Different column names
      leftOn = options.left_on;
      rightOn = options.right_on;
    } else {
      throw new InvalidParameterError('Must specify either "on" or both "left_on" and "right_on"');
    }

    // Validate columns exist
    if (!this._columns.includes(leftOn)) {
      throw new InvalidParameterError(
        `Column '${leftOn}' not found in left DataFrame`,
        "left_on",
        leftOn
      );
    }
    if (!other._columns.includes(rightOn)) {
      throw new InvalidParameterError(
        `Column '${rightOn}' not found in right DataFrame`,
        "right_on",
        rightOn
      );
    }

    // Build hash table from right DataFrame
    const rightHash = new Map<string, number[]>();
    const rightData = other._data.get(rightOn) ?? [];

    for (let i = 0; i < rightData.length; i++) {
      const val = rightData[i];
      if (val === null || val === undefined) continue;
      const key = createKey(val);
      const indices = rightHash.get(key) ?? [];
      indices.push(i);
      rightHash.set(key, indices);
    }

    const matchedRightRows = new Set<number>();
    const resultData: DataFrameData = {};

    // Handle column name conflicts with suffixes
    const leftColumns = this._columns.map((col) => {
      if (col === leftOn) return col; // Keep join column as-is
      if (other._columns.includes(col) && col !== rightOn) {
        return col + suffixes[0]; // Add suffix for conflicts
      }
      return col;
    });

    const leftColumnSet = new Set(leftColumns);
    const rightColumns: string[] = [];

    // Create a Set of original left columns for checking collisions
    const originalLeftColumns = new Set(this._columns);

    for (const col of other._columns) {
      if (leftOn === rightOn && col === rightOn) {
        continue;
      }

      let resultCol = col;

      // Check if this column existed in the left DataFrame (collision)
      // If so, we must apply the right suffix, unless it's the join key
      if (originalLeftColumns.has(col) && col !== leftOn) {
        resultCol = `${col}${suffixes[1]}`;
      }

      // Ensure uniqueness against already generated left columns (and previous right columns)
      if (leftColumnSet.has(resultCol)) {
        let suffixIndex = 0;
        let candidate = `${resultCol}`;
        // If we didn't suffix it yet (no collision with original), but it collides with a generated name
        // (rare, but possible e.g. left has 'A_x' and we generate 'A_x' from collision)
        // Or if the suffixed name itself collides.

        // Simpler approach: If it collides with result set, append increments
        while (leftColumnSet.has(candidate)) {
          suffixIndex++;
          // If we already added suffix, append number. If not, append suffix then number?
          // Suffix strategy: _x, _y. If _x exists, it stays _x.
          // Here we just need to ensure uniqueness.
          candidate = `${resultCol}_${suffixIndex}`;
        }
        resultCol = candidate;
      }

      rightColumns.push(resultCol);
      leftColumnSet.add(resultCol);
    }

    const allColumns = [...leftColumns, ...rightColumns];

    // Initialize result columns
    for (const col of allColumns) {
      resultData[col] = [];
    }

    const leftData = this._data.get(leftOn) ?? [];

    // Probe phase
    for (let i = 0; i < leftData.length; i++) {
      const key = createKey(leftData[i]);
      const rightIndices = rightHash.get(key) ?? [];

      if (rightIndices.length > 0) {
        for (const rightIdx of rightIndices) {
          matchedRightRows.add(rightIdx);

          // Add left DataFrame data with potential suffix
          for (let j = 0; j < this._columns.length; j++) {
            const originalCol = this._columns[j];
            if (!originalCol) continue;
            const resultCol = leftColumns[j];
            const colData = this._data.get(originalCol);
            if (resultCol) resultData[resultCol]?.push(colData?.[i] ?? null);
          }

          // Add right DataFrame data (excluding join column, with potential suffix)
          let rightColIdx = 0;
          for (const originalCol of other._columns) {
            // Only skip rightOn if it's implicitly excluded from rightColumns (same as leftOn)
            const shouldSkip = leftOn === rightOn && originalCol === rightOn;
            if (shouldSkip || !originalCol) continue;

            const resultCol = rightColumns[rightColIdx];
            const colData = other._data.get(originalCol);
            if (resultCol) resultData[resultCol]?.push(colData?.[rightIdx] ?? null);
            rightColIdx++;
          }
        }
      } else if (how === "left" || how === "outer") {
        // No match - left row with nulls
        for (let j = 0; j < this._columns.length; j++) {
          const originalCol = this._columns[j];
          if (!originalCol) continue;
          const resultCol = leftColumns[j];
          const colData = this._data.get(originalCol);
          if (resultCol) resultData[resultCol]?.push(colData?.[i] ?? null);
        }
        for (const col of rightColumns) {
          resultData[col]?.push(null);
        }
      }
    }

    // Add unmatched right rows for right/outer joins
    if (how === "right" || how === "outer") {
      for (let i = 0; i < rightData.length; i++) {
        if (!matchedRightRows.has(i)) {
          // Add values for left columns (handling join column)
          for (let j = 0; j < this._columns.length; j++) {
            const originalCol = this._columns[j];
            const resultCol = leftColumns[j];

            if (originalCol && resultCol) {
              if (originalCol === leftOn && leftOn === rightOn) {
                // For join column, use value from right DataFrame ONLY if they are the same column conceptually
                const rightJoinData = other._data.get(rightOn);
                resultData[resultCol]?.push(rightJoinData?.[i] ?? null);
              } else {
                resultData[resultCol]?.push(null);
              }
            }
          }

          // Data from right DataFrame
          let rightColIdx = 0;
          for (const originalCol of other._columns) {
            // We only skip rightOn if it was excluded from rightColumns.
            // rightColumns excludes rightOn ONLY if leftOn === rightOn.
            const shouldSkip = leftOn === rightOn && originalCol === rightOn;

            if (shouldSkip) {
              continue;
            }

            const resultCol = rightColumns[rightColIdx];
            const colData = other._data.get(originalCol);
            if (resultCol) resultData[resultCol]?.push(colData?.[i] ?? null);
            rightColIdx++;
          }
        }
      }
    }

    return new DataFrame(resultData, { columns: allColumns });
  }

  /**
   * Concatenate with another DataFrame.
   *
   * @param other - DataFrame to concatenate
   * @param axis - Axis to concatenate along.
   *               - 0 or "rows" or "index": Stack vertically (append rows)
   *               - 1 or "columns": Stack horizontally (append columns)
   * @returns Concatenated DataFrame
   *
   * @example
   * ```ts
   * const df1 = new DataFrame({ a: [1, 2], b: [3, 4] });
   * const df2 = new DataFrame({ a: [5, 6], b: [7, 8] });
   * df1.concat(df2, "rows");  // Stack vertically: 4 rows
   * df1.concat(df2, "columns");  // Stack horizontally: 4 columns
   * ```
   */
  concat(other: DataFrame, axis: Axis = 0): DataFrame {
    const ax = normalizeAxis(axis, 2);

    if (ax === 0) {
      // Concatenate rows (stack vertically)
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
      const newData: DataFrameData = {};

      // Copy data from both DataFrames for each column
      for (const col of this._columns) {
        const thisColData = this._data.get(col) ?? [];
        const otherColData = other._data.get(col) ?? [];
        newData[col] = [...thisColData, ...otherColData];
      }

      // Reset index to sequential integers to avoid duplicate index errors
      const totalRows = this._index.length + other._index.length;
      const newIndex = Array.from({ length: totalRows }, (_, i) => i);

      return new DataFrame(newData, {
        columns: this._columns,
        index: newIndex,
      });
    } else {
      // Concatenate columns (stack horizontally) with index alignment
      // 1. Determine new index (Union of indices)
      const newIndex = [...this._index];
      const seenIndices = new Set(this._index);

      for (const idx of other._index) {
        if (!seenIndices.has(idx)) {
          newIndex.push(idx);
          seenIndices.add(idx);
        }
      }

      // 2. Build new data
      const newData: DataFrameData = {};
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

      // Copy all columns from this DataFrame, suffixing overlapping ones
      for (const col of this._columns) {
        const outputName = overlapping.has(col) ? `${col}_left` : col;
        newData[outputName] = alignColumn(this, col, newIndex);
        newColumns.push(outputName);
      }

      // Add columns from other DataFrame, suffixing overlapping ones
      for (const col of other._columns) {
        const outputName = overlapping.has(col) ? `${col}_right` : col;
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
   * Fill missing values (null or undefined) with a specified value.
   *
   * @param value - Value to use for filling missing values
   * @returns New DataFrame with missing values filled
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, null, 3], b: [4, 5, undefined] });
   * df.fillna(0);  // Replace null/undefined with 0
   * ```
   */
  fillna(value: unknown): DataFrame {
    const newData: DataFrameData = {};

    // Replace null/undefined in each column
    for (const col of this._columns) {
      const colData = this._data.get(col);
      if (colData) {
        newData[col] = colData.map((v) =>
          v === null || v === undefined || (typeof v === "number" && Number.isNaN(v)) ? value : v
        );
      }
    }

    return new DataFrame(newData, {
      columns: this._columns,
      index: this._index,
    });
  }

  /**
   * Drop rows that contain any missing values (null or undefined).
   *
   * @returns New DataFrame with rows containing missing values removed
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, null, 3], b: [4, 5, 6] });
   * df.dropna();  // Only keeps rows 0 and 2
   * ```
   */
  dropna(): DataFrame {
    const newData: DataFrameData = {};
    const newIndex: (string | number)[] = [];

    // Initialize empty arrays for each column
    for (const col of this._columns) {
      newData[col] = [];
    }

    // Check each row for missing values
    for (let i = 0; i < this._index.length; i++) {
      let hasNA = false;

      // Check if any column has null/undefined/NaN at this row
      for (const col of this._columns) {
        const colData = this._data.get(col);
        if (colData) {
          const val = colData[i];
          if (val === null || val === undefined || (typeof val === "number" && Number.isNaN(val))) {
            hasNA = true;
            break;
          }
        }
      }

      // If row has no missing values, keep it
      if (!hasNA) {
        const idx = this._index[i];
        if (idx !== undefined) newIndex.push(idx);
        for (const col of this._columns) {
          const colData = this._data.get(col);
          if (colData) {
            newData[col]?.push(colData[i]);
          }
        }
      }
    }

    return new DataFrame(newData, {
      columns: this._columns,
      index: newIndex,
    });
  }

  /**
   * Generate descriptive statistics.
   *
   * Computes count, mean, std, min, 25%, 50%, 75%, max for numeric columns.
   *
   * @returns DataFrame with statistics
   */
  describe(): DataFrame {
    const stats: DataFrameData = {};
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
        if (isValidNumber(v)) validCount++;
      }
      if (validCount === 0) continue;
      const numericData = new Float64Array(validCount);
      {
        let w = 0;
        for (const v of colData) {
          if (isValidNumber(v)) numericData[w++] = v;
        }
      }

      const sorted = numericData.slice();
      sorted.sort();
      let sum = 0;
      for (let i = 0; i < validCount; i++) sum += numericData[i]!;
      const mean = sum / validCount;
      let variance: number;
      let std: number;

      if (validCount > 1) {
        let sq = 0;
        for (let i = 0; i < validCount; i++) {
          const d = numericData[i]! - mean;
          sq += d * d;
        }
        variance = sq / (validCount - 1);
        std = Math.sqrt(variance);
      } else {
        variance = NaN;
        std = NaN;
      }

      const getPercentile = (p: number) => {
        const idx = (p / 100) * (sorted.length - 1);
        const lower = Math.floor(idx);
        const upper = Math.ceil(idx);
        const weight = idx - lower;
        return (sorted[lower] ?? 0) * (1 - weight) + (sorted[upper] ?? 0) * weight;
      };

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
        getPercentile(25),
        getPercentile(50),
        getPercentile(75),
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
   * Compute correlation matrix.
   *
   * Uses pairwise complete observations (ignores missing values for each pair).
   *
   * @returns DataFrame containing pairwise correlations
   */
  corr(): DataFrame {
    const numericCols: string[] = [];

    // Identify numeric columns
    for (const col of this._columns) {
      const colData = this._data.get(col);
      if (!colData) continue;
      // Check if column has at least one number
      if (colData.some(isValidNumber)) {
        numericCols.push(col);
      }
    }

    const nRows = this._index.length;
    const kCols = numericCols.length;

    // Fast path: when no column contains a missing/invalid value, center each
    // column once and reuse the centered vectors for every pair, exploiting
    // the matrix's symmetry (diagonal = 1). The general path rebuilds the
    // pairwise-valid vectors and recomputes means for all k^2 pairs.
    const centered: (Float64Array | null)[] = new Array(kCols);
    let allValid = true;
    for (let c = 0; c < kCols; c++) {
      const data = this._data.get(numericCols[c] as string);
      if (!data) {
        allValid = false;
        break;
      }
      const col = new Float64Array(nRows);
      let mean = 0;
      let ok = true;
      for (let i = 0; i < nRows; i++) {
        const v = data[i];
        if (!isValidNumber(v)) {
          ok = false;
          break;
        }
        col[i] = v;
        mean += v;
      }
      if (!ok) {
        allValid = false;
        break;
      }
      mean /= nRows;
      for (let i = 0; i < nRows; i++) col[i] = (col[i] as number) - mean;
      centered[c] = col;
    }

    const corrMatrix: DataFrameData = {};

    if (allValid && nRows >= 2) {
      const norms = new Float64Array(kCols);
      for (let c = 0; c < kCols; c++) {
        const col = centered[c] as Float64Array;
        let ss = 0;
        for (let i = 0; i < nRows; i++) ss += col[i]! * col[i]!;
        norms[c] = Math.sqrt(ss);
      }
      const out: number[][] = [];
      for (let a = 0; a < kCols; a++) out.push(new Array(kCols).fill(NaN));
      for (let a = 0; a < kCols; a++) {
        out[a]![a] = norms[a] === 0 ? NaN : 1;
        const ca = centered[a] as Float64Array;
        for (let b = a + 1; b < kCols; b++) {
          const cb = centered[b] as Float64Array;
          let num = 0;
          for (let i = 0; i < nRows; i++) num += ca[i]! * cb[i]!;
          const denom = norms[a]! * norms[b]!;
          const r = denom === 0 ? NaN : num / denom;
          out[a]![b] = r;
          out[b]![a] = r;
        }
      }
      for (let a = 0; a < kCols; a++) corrMatrix[numericCols[a] as string] = out[a] as number[];
      return new DataFrame(corrMatrix, { index: numericCols, columns: numericCols });
    }

    for (const col1 of numericCols) {
      corrMatrix[col1] = [];
      const data1 = this._data.get(col1);

      for (const col2 of numericCols) {
        const data2 = this._data.get(col2);

        if (!data1 || !data2) {
          corrMatrix[col1]?.push(NaN);
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
          corrMatrix[col1]?.push(NaN);
          continue;
        }

        // Compute correlation
        const mean1 = valid1.reduce((a, b) => a + b, 0) / valid1.length;
        const mean2 = valid2.reduce((a, b) => a + b, 0) / valid2.length;

        let num = 0;
        let den1 = 0;
        let den2 = 0;

        for (let k = 0; k < valid1.length; k++) {
          const val1 = valid1[k];
          const val2 = valid2[k];
          // val1 and val2 are guaranteed to be numbers from valid1/valid2 construction
          // However, we check for undefined to satisfy strict null checks (noUncheckedIndexedAccess)
          if (val1 === undefined || val2 === undefined) continue;

          const diff1 = val1 - mean1;
          const diff2 = val2 - mean2;
          num += diff1 * diff2;
          den1 += diff1 * diff1;
          den2 += diff2 * diff2;
        }

        const corr = den1 === 0 || den2 === 0 ? NaN : num / Math.sqrt(den1 * den2);
        corrMatrix[col1]?.push(corr);
      }
    }

    return new DataFrame(corrMatrix, {
      index: numericCols,
      columns: numericCols,
    });
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

    const covMatrix: DataFrameData = {};

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
      const newData: DataFrameData = {};

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

      const newData: DataFrameData = {};

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
   * All columns must contain numeric data.
   *
   * @returns 2D Tensor with shape [rows, columns]
   * @throws {DataValidationError} If data is non-numeric
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, 2, 3], b: [4, 5, 6] });
   * const t = df.toTensor();  // 2D tensor [[1,4], [2,5], [3,6]]
   * ```
   */
  toTensor(): Tensor {
    // Convert to 2D array first
    const arr = this.toArray();

    // Flatten to 1D array in row-major order
    const flat: number[] = [];
    for (const row of arr) {
      for (const val of row) {
        if (typeof val === "number") {
          flat.push(val);
        } else if (val === null || val === undefined) {
          flat.push(NaN);
        } else {
          throw new DataValidationError(
            `Non-numeric value found: ${val}. All data must be numeric (or null/undefined) for tensor conversion.`
          );
        }
      }
    }

    // Create 1D tensor and reshape to [rows, cols]
    const t = tensor(flat);
    const [rows, cols] = this.shape;
    return reshape(t, [rows, cols]);
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
   * Parse CSV string into DataFrame with full type inference and quote handling.
   * Time complexity: O(n) where n is number of characters.
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

    // Strip a leading UTF-8 BOM (common in Excel-exported CSVs); otherwise the
    // first column name becomes "﻿<name>" and lookups fail.
    if (csvString.charCodeAt(0) === 0xfeff) {
      csvString = csvString.slice(1);
    }

    const rows: string[][] = [];
    let fields: string[] = [];
    let currentField = "";
    let inQuotes = false;
    let rowCount = 0;

    // A row is a blank line (skippable) only when it is a single empty field.
    // A multi-field all-empty row like ",," is a real all-null record.
    const isBlankLine = (fs: string[]): boolean => fs.length === 1 && (fs[0] ?? "").trim() === "";

    // Parse character by character to handle newlines in quoted fields
    for (let i = 0; i < csvString.length; i++) {
      const char = csvString[i];
      const nextChar = csvString[i + 1];

      if (char === quoteChar) {
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
        // multi-field rows (",," → an all-null record, like pandas).
        if (!isBlankLine(fields)) {
          if (rowCount >= skipRows) {
            rows.push(fields);
          }
          rowCount++;
        }
        fields = [];
      } else {
        // Regular character (including newlines inside quotes)
        currentField += char;
      }
    }

    // Handle last row if no trailing newline
    if (currentField !== "" || fields.length > 0) {
      fields.push(currentField);
      if (!isBlankLine(fields) && rowCount >= skipRows) {
        rows.push(fields);
      }
    }

    if (inQuotes) {
      throw new DataValidationError("CSV contains an unmatched quote");
    }

    if (rows.length === 0) {
      throw new DataValidationError("CSV contains no data rows");
    }

    let columns: string[];
    let dataRows: string[][];

    if (hasHeader) {
      const firstRow = rows[0];
      if (!firstRow) throw new DataValidationError("CSV has no header row");
      columns = firstRow;
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

    const data: DataFrameData = {};
    for (let colIdx = 0; colIdx < columns.length; colIdx++) {
      const colName = columns[colIdx];
      const colData: unknown[] = [];

      for (const row of dataRows) {
        const value = row[colIdx];
        const trimmed = value === undefined ? "" : value.trim();
        if (trimmed === "" || trimmed === "null" || trimmed === "undefined") {
          // Empty or whitespace-only cells are missing (NaN), NOT numeric 0.
          colData.push(null);
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

      if (colName) data[colName] = colData;
    }

    return new DataFrame(data, { columns });
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
   * Time complexity: O(n × m) where n is rows, m is columns.
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

    const lines: string[] = [];

    const escapeField = (value: unknown): string => {
      const str = String(value ?? "");
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

    if (header) {
      const headerFields = includeIndex ? ["index", ...this._columns] : [...this._columns];
      lines.push(headerFields.map(escapeField).join(delimiter));
    }

    for (let i = 0; i < this._index.length; i++) {
      const rowFields: unknown[] = [];

      if (includeIndex) {
        rowFields.push(this._index[i]);
      }

      for (const col of this._columns) {
        const colData = this._data.get(col);
        rowFields.push(colData?.[i] ?? "");
      }

      lines.push(rowFields.map(escapeField).join(delimiter));
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
   * Serialize DataFrame to JSON string.
   * Time complexity: O(n × m).
   */
  toJsonString(): string {
    return JSON.stringify(
      {
        columns: this._columns,
        index: this._index,
        data: Object.fromEntries(this._data),
      },
      null,
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
      if (!(col in rawData)) {
        throw new DataValidationError(`Missing data for column '${col}'`);
      }
    }
    for (const key of dataKeys) {
      if (!columns.includes(key)) {
        throw new DataValidationError(`Unexpected data column '${key}' not listed in columns`);
      }
    }

    const data: DataFrameData = {};
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

  /**
   * Create DataFrame from a Tensor.
   *
   * @param tensor - Tensor to convert (must be 1D or 2D)
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
  static fromTensor(tensor: Tensor, columns?: string[]): DataFrame {
    const storage = tensor.data;
    let data: unknown[];

    if (storage instanceof BigInt64Array) {
      data = Array.from(storage, (v) => Number(v));
    } else if (ArrayBuffer.isView(storage)) {
      data = Array.from(storage, (v) => Number(v));
    } else if (Array.isArray(storage)) {
      data = [...storage];
    } else {
      throw new DataValidationError("Unsupported tensor storage type");
    }

    if (tensor.ndim === 1) {
      if (columns && columns.length !== 1) {
        throw new DataValidationError(
          `Expected exactly 1 column name for 1D tensor, received ${columns.length}`
        );
      }
      const colName = columns?.[0] ?? "col0";
      return new DataFrame({ [colName]: data });
    }

    if (tensor.ndim === 2) {
      const rows = tensor.shape[0];
      const cols = tensor.shape[1];

      if (rows === undefined || cols === undefined) {
        throw new DataValidationError("Invalid tensor shape");
      }

      if (columns && columns.length !== cols) {
        throw new DataValidationError(
          `Column count (${columns.length}) must match tensor columns (${cols})`
        );
      }

      const dfData: DataFrameData = {};

      for (let c = 0; c < cols; c++) {
        const colName = columns?.[c] ?? `col${c}`;
        const colData: unknown[] = [];

        for (let r = 0; r < rows; r++) {
          colData.push(data[r * cols + c]);
        }

        dfData[colName] = colData;
      }

      return new DataFrame(dfData, {
        columns: columns ?? Array.from({ length: cols }, (_, i) => `col${i}`),
      });
    }

    throw new DataValidationError(
      `Cannot create DataFrame from ${tensor.ndim}D tensor. Only 1D and 2D tensors supported.`
    );
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
   * df.drop_duplicates();  // Keeps first occurrence: [[1, 3], [2, 4]]
   * df.drop_duplicates(undefined, 'last');  // Keeps last occurrence
   * ```
   */
  drop_duplicates(subset?: string[], keep: "first" | "last" | false = "first"): DataFrame {
    // Determine which columns to check for duplicates
    const checkCols = subset ?? this._columns;

    // Validate subset columns exist
    for (const col of checkCols) {
      if (!this._columns.includes(col)) {
        throw new DataValidationError(`Column '${col}' not found in DataFrame`);
      }
    }

    // Track seen row signatures
    const seen = new Map<string, number[]>();
    const keepIndices: number[] = [];

    for (let i = 0; i < this._index.length; i++) {
      const signature: unknown[] = [];
      for (const col of checkCols) {
        signature.push(this._data.get(col)?.[i]);
      }
      const key = createKey(signature);

      const existing = seen.get(key);
      if (existing === undefined) {
        seen.set(key, [i]);
      } else {
        existing.push(i);
      }
    }

    for (const [_key, indices] of seen.entries()) {
      if (keep === "first") {
        const firstIndex = indices[0];
        if (firstIndex !== undefined) {
          keepIndices.push(firstIndex);
        }
      } else if (keep === "last") {
        const lastIndex = indices[indices.length - 1];
        if (lastIndex !== undefined) {
          keepIndices.push(lastIndex);
        }
      } else if (keep === false && indices.length === 1) {
        const onlyIndex = indices[0];
        if (onlyIndex !== undefined) {
          keepIndices.push(onlyIndex);
        }
      }
    }

    keepIndices.sort((a, b) => a - b);

    // Build result DataFrame from kept indices
    const newData: DataFrameData = {};
    const newIndex: (string | number)[] = [];

    for (const col of this._columns) {
      newData[col] = [];
    }

    for (const idx of keepIndices) {
      const label = this._index[idx];
      if (label === undefined) {
        throw new DataValidationError(`Index label at position ${idx} is undefined`);
      }
      newIndex.push(label);
      for (const col of this._columns) {
        newData[col]?.push(this._data.get(col)?.[idx]);
      }
    }

    return new DataFrame(newData, {
      columns: this._columns,
      index: newIndex,
    });
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
    const checkCols = subset ?? this._columns;

    for (const col of checkCols) {
      if (!this._columns.includes(col)) {
        throw new DataValidationError(`Column '${col}' not found in DataFrame`);
      }
    }

    const seen = new Map<string, number[]>();

    for (let i = 0; i < this._index.length; i++) {
      const signature: unknown[] = [];
      for (const col of checkCols) {
        signature.push(this._data.get(col)?.[i]);
      }
      const key = createKey(signature);

      const existing = seen.get(key);
      if (existing === undefined) {
        seen.set(key, [i]);
      } else {
        existing.push(i);
      }
    }

    const isDuplicate: boolean[] = new Array(this._index.length).fill(false);

    for (const [_key, indices] of seen.entries()) {
      if (indices.length > 1) {
        if (keep === "first") {
          for (let i = 1; i < indices.length; i++) {
            const idx = indices[i];
            if (idx !== undefined) isDuplicate[idx] = true;
          }
        } else if (keep === "last") {
          for (let i = 0; i < indices.length - 1; i++) {
            const idx = indices[i];
            if (idx !== undefined) isDuplicate[idx] = true;
          }
        } else if (keep === false) {
          for (const idx of indices) {
            isDuplicate[idx] = true;
          }
        }
      }
    }

    return new Series(isDuplicate, { index: this._index });
  }

  /**
   * Rename columns or index labels.
   * Time complexity: O(m) for columns, O(n) for index.
   *
   * @param mapper - Object mapping old names to new names, or function to transform names
   * @param axis - 0 for index, 1 for columns
   * @returns New DataFrame with renamed labels
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, 2], b: [3, 4] });
   * df.rename({ a: 'x', b: 'y' }, 1);  // Rename columns a->x, b->y
   * df.rename((name) => name.toUpperCase(), 1);  // Uppercase all column names
   * ```
   */
  rename(mapper: Record<string, string> | ((name: string) => string), axis: 0 | 1 = 1): DataFrame {
    if (axis === 1) {
      // Rename columns
      const newColumns = this._columns.map((col) => {
        if (typeof mapper === "function") {
          return mapper(col);
        }
        return mapper[col] ?? col;
      });

      const newData: DataFrameData = {};
      for (let i = 0; i < this._columns.length; i++) {
        const oldCol = this._columns[i];
        const newCol = newColumns[i];
        if (oldCol && newCol) {
          const colData = this._data.get(oldCol);
          if (colData) {
            newData[newCol] = [...colData];
          }
        }
      }

      return new DataFrame(newData, {
        columns: newColumns,
        index: this._index,
      });
    } else {
      // Rename index
      const newIndex = this._index.map((label) => {
        const labelStr = String(label);
        if (typeof mapper === "function") {
          return mapper(labelStr);
        }
        return mapper[labelStr] ?? label;
      });

      const newData: DataFrameData = {};
      for (const col of this._columns) {
        const colData = this._data.get(col);
        if (colData) {
          newData[col] = [...colData];
        }
      }

      return new DataFrame(newData, {
        columns: this._columns,
        index: newIndex,
      });
    }
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
   * df.reset_index();  // Index becomes [0, 1], adds 'index' column with ['x', 'y']
   * df.reset_index(true);  // Index becomes [0, 1], no new column
   * ```
   */
  reset_index(drop: boolean = false): DataFrame {
    const newData: DataFrameData = {};

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
    });
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
   * df.set_index('id');  // Index becomes ['a', 'b', 'c']
   * ```
   */
  set_index(column: string, drop: boolean = true): DataFrame {
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

    const newData: DataFrameData = {};
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
    });
  }

  /**
   * Return boolean DataFrame showing null values.
   * Time complexity: O(n × m).
   *
   * @returns DataFrame of booleans (true = null/undefined, false = not null)
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, null, 3], b: [4, 5, undefined] });
   * df.isnull();  // [[false, false], [true, false], [false, true]]
   * ```
   */
  isnull(): DataFrame {
    const newData: DataFrameData = {};

    for (const col of this._columns) {
      const colData = this._data.get(col);
      if (colData) {
        newData[col] = colData.map((v) => v === null || v === undefined);
      }
    }

    return new DataFrame(newData, {
      columns: this._columns,
      index: this._index,
    });
  }

  /**
   * Return boolean DataFrame showing non-null values.
   * Time complexity: O(n × m).
   *
   * @returns DataFrame of booleans (true = not null, false = null/undefined)
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, null, 3], b: [4, 5, undefined] });
   * df.notnull();  // [[true, true], [false, true], [true, false]]
   * ```
   */
  notnull(): DataFrame {
    const newData: DataFrameData = {};

    for (const col of this._columns) {
      const colData = this._data.get(col);
      if (colData) {
        newData[col] = colData.map((v) => v !== null && v !== undefined);
      }
    }

    return new DataFrame(newData, {
      columns: this._columns,
      index: this._index,
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

    const newData: DataFrameData = {};

    for (const col of this._columns) {
      const colData = this._data.get(col);
      if (colData) {
        newData[col] = colData.map((v) => (replaceSet.has(v) ? value : v));
      }
    }

    return new DataFrame(newData, {
      columns: this._columns,
      index: this._index,
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
    const newData: DataFrameData = {};

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
    });
  }

  /**
   * Return a random sample of rows.
   * Time complexity: O(n) for sampling.
   *
   * @param n - Number of rows to sample
   * @param random_state - Random seed for reproducibility
   * @returns New DataFrame with sampled rows
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, 2, 3, 4, 5], b: [6, 7, 8, 9, 10] });
   * df.sample(3);  // Random 3 rows
   * ```
   */
  sample(n: number, random_state?: number): DataFrame {
    if (!Number.isFinite(n) || !Number.isInteger(n)) {
      throw new InvalidParameterError("n must be a finite integer", "n", n);
    }
    if (random_state !== undefined) {
      if (!Number.isFinite(random_state) || !Number.isInteger(random_state)) {
        throw new InvalidParameterError(
          "random_state must be a finite integer",
          "random_state",
          random_state
        );
      }
    }
    if (n < 0 || n > this._index.length) {
      throw new DataValidationError(`Sample size ${n} must be between 0 and ${this._index.length}`);
    }

    const rng = random_state !== undefined ? this.seededRandom(random_state) : __random;

    const indices = Array.from({ length: this._index.length }, (_, i) => i);

    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const current = indices[i];
      const swap = indices[j];
      if (current === undefined || swap === undefined) {
        throw new DataValidationError("Sample index resolution failed");
      }
      indices[i] = swap;
      indices[j] = current;
    }

    const sampledIndices = indices.slice(0, n);

    const newData: DataFrameData = {};
    const newIndex: (string | number)[] = [];

    for (const col of this._columns) {
      newData[col] = [];
    }

    for (const idx of sampledIndices) {
      const label = this._index[idx];
      if (label === undefined) {
        throw new DataValidationError(`Index label at position ${idx} is undefined`);
      }
      newIndex.push(label);
      for (const col of this._columns) {
        newData[col]?.push(this._data.get(col)?.[idx]);
      }
    }

    return new DataFrame(newData, {
      columns: this._columns,
      index: newIndex,
    });
  }

  /**
   * Seeded random number generator for reproducibility.
   * @private
   */
  private seededRandom(seed: number): () => number {
    let state = seed >>> 0;
    return () => {
      state = (state * 1664525 + 1013904223) % 2 ** 32;
      return state / 2 ** 32;
    };
  }

  /**
   * Return values at the given quantile.
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

      const numericData = toNumericValues(colData);
      if (numericData.length === 0) {
        result.push(NaN);
        resultIndex.push(col);
        continue;
      }

      const sorted = [...numericData].sort((a, b) => a - b);
      const idx = q * (sorted.length - 1);
      const lower = Math.floor(idx);
      const upper = Math.ceil(idx);
      const weight = idx - lower;

      const value = (sorted[lower] ?? 0) * (1 - weight) + (sorted[upper] ?? 0) * weight;

      result.push(value);
      resultIndex.push(col);
    }

    return new Series(result, { index: resultIndex });
  }

  /**
   * Compute numerical rank of values (1 through n) along axis.
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
    const newData: DataFrameData = {};

    for (const col of this._columns) {
      const colData = this._data.get(col);
      if (!colData) continue;

      const numericData = toNumericValues(colData);
      if (numericData.length === 0) {
        newData[col] = colData.map(() => null);
        continue;
      }

      const indexed = colData.map((v, i) => ({ value: v, index: i }));
      const numericIndexed = indexed.filter((item): item is { value: number; index: number } =>
        isValidNumber(item.value)
      );

      numericIndexed.sort((a, b) => {
        if (ascending) {
          return a.value - b.value;
        }
        return b.value - a.value;
      });

      const ranks: (number | null)[] = new Array(colData.length).fill(null);

      let i = 0;
      let denseRank = 0; // Track dense rank separately
      while (i < numericIndexed.length) {
        const currentItem = numericIndexed[i];
        if (!currentItem) {
          break;
        }
        const currentValue = currentItem.value;
        const tieStart = i;

        while (i < numericIndexed.length) {
          const nextItem = numericIndexed[i];
          if (!nextItem || nextItem.value !== currentValue) {
            break;
          }
          i++;
        }

        const tieEnd = i;
        denseRank++; // Increment dense rank for each unique value

        for (let j = tieStart; j < tieEnd; j++) {
          const item = numericIndexed[j];
          if (!item) continue;

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
            // dense method: use denseRank which increments only for unique values
            rank = denseRank;
          }

          ranks[item.index] = rank;
        }
      }

      newData[col] = ranks;
    }

    return new DataFrame(newData, {
      columns: this._columns,
      index: this._index,
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
    const newData: DataFrameData = {};

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
   * df.pct_change();  // [[null], [0.1], [0.1]] (10% increase each time)
   * ```
   */
  pct_change(periods: number = 1): DataFrame {
    if (!Number.isFinite(periods) || !Number.isInteger(periods) || periods < 0) {
      throw new InvalidParameterError("periods must be a non-negative integer", "periods", periods);
    }
    const newData: DataFrameData = {};

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
    });
  }

  /**
   * Return cumulative sum over DataFrame axis.
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
    const newData: DataFrameData = {};

    for (const col of this._columns) {
      const colData = this._data.get(col);
      if (!colData) continue;

      const cumData: unknown[] = [];
      let cumSum = 0;

      for (const value of colData) {
        if (typeof value === "number") {
          cumSum += value;
          cumData.push(cumSum);
        } else {
          cumData.push(null);
        }
      }

      newData[col] = cumData;
    }

    return new DataFrame(newData, {
      columns: this._columns,
      index: this._index,
    });
  }

  /**
   * Return cumulative product over DataFrame axis.
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
    const newData: DataFrameData = {};

    for (const col of this._columns) {
      const colData = this._data.get(col);
      if (!colData) continue;

      const cumData: unknown[] = [];
      let cumProd = 1;

      for (const value of colData) {
        if (typeof value === "number") {
          cumProd *= value;
          cumData.push(cumProd);
        } else {
          cumData.push(null);
        }
      }

      newData[col] = cumData;
    }

    return new DataFrame(newData, {
      columns: this._columns,
      index: this._index,
    });
  }

  /**
   * Return cumulative maximum over DataFrame axis.
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
    const newData: DataFrameData = {};

    for (const col of this._columns) {
      const colData = this._data.get(col);
      if (!colData) continue;

      const cumData: unknown[] = [];
      let cumMax = -Infinity;

      for (const value of colData) {
        if (typeof value === "number") {
          cumMax = Math.max(cumMax, value);
          cumData.push(cumMax);
        } else {
          cumData.push(null);
        }
      }

      newData[col] = cumData;
    }

    return new DataFrame(newData, {
      columns: this._columns,
      index: this._index,
    });
  }

  /**
   * Return cumulative minimum over DataFrame axis.
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
    const newData: DataFrameData = {};

    for (const col of this._columns) {
      const colData = this._data.get(col);
      if (!colData) continue;

      const cumData: unknown[] = [];
      let cumMin = Infinity;

      for (const value of colData) {
        if (typeof value === "number") {
          cumMin = Math.min(cumMin, value);
          cumData.push(cumMin);
        } else {
          cumData.push(null);
        }
      }

      newData[col] = cumData;
    }

    return new DataFrame(newData, {
      columns: this._columns,
      index: this._index,
    });
  }

  /**
   * Shift index by desired number of periods.
   * Time complexity: O(n × m).
   *
   * @param periods - Number of periods to shift (positive = down, negative = up)
   * @param fill_value - Value to use for newly introduced missing values
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
  shift(periods: number = 1, fill_value: unknown = null): DataFrame {
    if (!Number.isFinite(periods) || !Number.isInteger(periods)) {
      throw new InvalidParameterError("periods must be a finite integer", "periods", periods);
    }

    const newData: DataFrameData = {};

    for (const col of this._columns) {
      const colData = this._data.get(col);
      if (!colData) continue;

      const shiftedData: unknown[] = [];
      const rowCount = colData.length;

      if (periods > 0) {
        const shift = Math.min(periods, rowCount);
        for (let i = 0; i < shift; i++) {
          shiftedData.push(fill_value);
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
          shiftedData.push(fill_value);
        }
      } else {
        shiftedData.push(...colData);
      }

      newData[col] = shiftedData;
    }

    return new DataFrame(newData, {
      columns: this._columns,
      index: this._index,
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

    const pivotData: DataFrameData = {};
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
   * @param id_vars - Columns to keep as is
   * @param value_vars - Columns to melt
   * @param var_name - Name for new column with melted variable names
   * @param value_name - Name for new column with melted values.
   *                    Must not conflict with existing columns or var_name.
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
    id_vars: string[],
    value_vars: string[],
    var_name: string = "variable",
    value_name: string = "value"
  ): DataFrame {
    const idVars = [...id_vars];
    const valueVars = [...value_vars];

    ensureUniqueLabels(idVars, "id_var");
    ensureUniqueLabels(valueVars, "value_var");

    for (const idVar of idVars) {
      if (!this._columns.includes(idVar)) {
        throw new DataValidationError(`Column '${idVar}' not found in DataFrame`);
      }
    }

    for (const valueVar of valueVars) {
      if (!this._columns.includes(valueVar)) {
        throw new DataValidationError(`Column '${valueVar}' not found in DataFrame`);
      }
    }

    if (var_name === value_name) {
      throw new DataValidationError("var_name and value_name must be different");
    }

    const reservedNames = new Set([...idVars, ...valueVars]);
    if (reservedNames.has(var_name) || reservedNames.has(value_name)) {
      throw new DataValidationError(
        "var_name and value_name must not conflict with existing columns"
      );
    }

    const newData: DataFrameData = {};
    for (const idVar of idVars) {
      newData[idVar] = [];
    }

    newData[var_name] = [];
    newData[value_name] = [];

    for (let i = 0; i < this._index.length; i++) {
      for (const valueVar of valueVars) {
        for (const idVar of idVars) {
          newData[idVar]?.push(this._data.get(idVar)?.[i]);
        }

        newData[var_name]?.push(valueVar);
        newData[value_name]?.push(this._data.get(valueVar)?.[i]);
      }
    }

    return new DataFrame(newData, {
      columns: [...idVars, var_name, value_name],
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

    const stackSet = new Set(stackCols);
    const idCols = this._columns.filter((c) => !stackSet.has(c));

    // Build output
    const newData: DataFrameData = {};
    for (const id of idCols) {
      newData[id] = [];
    }
    newData[varName] = [];
    newData[valueName] = [];

    const nRows = this._index.length;
    for (let i = 0; i < nRows; i++) {
      for (const sc of stackCols) {
        for (const id of idCols) {
          newData[id]?.push(this._data.get(id)?.[i]);
        }
        newData[varName]?.push(sc);
        newData[valueName]?.push(this._data.get(sc)?.[i]);
      }
    }

    return new DataFrame(newData, {
      columns: [...idCols, varName, valueName],
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
    const newData: DataFrameData = {};
    for (const pv of uniquePivot) {
      newData[pv] = new Array<unknown>(uniqueIdx.length).fill(null);
    }

    // Fill values
    const nRows = indexData.length;
    for (let i = 0; i < nRows; i++) {
      const idx = indexData[i];
      const pv = pivotData[i];
      if (idx === null || idx === undefined || pv === null || pv === undefined) {
        continue;
      }
      const idxKey = typeof idx === "string" || typeof idx === "number" ? idx : String(idx);
      const pvKey = String(pv);
      const rp = rowPos.get(idxKey);
      if (rp !== undefined) {
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
   * @param window - Size of the rolling window
   * @param on - Column to apply rolling calculation to (if omitted, applies to all numeric columns)
   * @returns Rolling object with mean(), sum(), std(), var(), min(), max(), apply() methods
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, 2, 3, 4, 5] });
   * df.rolling(3).mean();  // rolling mean
   * df.rolling(3).sum();   // rolling sum
   * df.rolling(3).std();   // rolling standard deviation
   * ```
   */
  rolling(window: number, on?: string): Rolling {
    if (!Number.isFinite(window) || !Number.isInteger(window) || window <= 0) {
      throw new InvalidParameterError("window must be a positive integer", "window", window);
    }
    if (on && !this._columns.includes(on)) {
      throw new DataValidationError(`Column '${on}' not found in DataFrame`);
    }
    return new Rolling(this, window, on);
  }

  /**
   * Apply a function element-wise to the DataFrame.
   *
   * @param fn - Function to apply to each element
   * @returns New DataFrame with transformed values
   */
  applymap(fn: (value: unknown) => unknown): DataFrame {
    const newData: DataFrameData = {};
    for (const col of this._columns) {
      const colData = this._data.get(col);
      if (!colData) continue;
      newData[col] = colData.map(fn);
    }
    return new DataFrame(newData, {
      columns: this._columns,
      index: this._index,
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
   * one row per element. Non-array values are kept as-is.
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

    const newData: Map<string, unknown[]> = new Map();
    for (const c of this._columns) {
      newData.set(c, []);
    }
    const newIndex: (string | number)[] = [];

    for (let i = 0; i < this._index.length; i++) {
      const val = colData[i];
      const items = Array.isArray(val) ? val : [val];
      for (const item of items) {
        for (const c of this._columns) {
          if (c === column) {
            newData.get(c)!.push(item);
          } else {
            newData.get(c)!.push(this._data.get(c)?.[i]);
          }
        }
        newIndex.push(this._index[i]!);
      }
    }

    const obj: DataFrameData = {};
    for (const c of this._columns) {
      obj[c] = newData.get(c)!;
    }
    // Use numeric index to avoid duplicate label errors after explosion
    const numericIndex = Array.from({ length: newIndex.length }, (_, i) => i);
    return new DataFrame(obj, { columns: this._columns, index: numericIndex });
  }

  /**
   * Convert categorical column(s) into dummy/indicator variables.
   *
   * @param columns - Column name(s) to encode. If omitted, encodes all string columns.
   * @param options - Options: prefix (column name prefix), dropFirst (drop first category)
   * @returns New DataFrame with dummy columns
   */
  getDummies(
    columns?: string | string[],
    options: { prefix?: string; dropFirst?: boolean } = {}
  ): DataFrame {
    const cols = columns
      ? Array.isArray(columns)
        ? columns
        : [columns]
      : this._columns.filter((c) => {
          const d = this._data.get(c);
          return d?.some((v) => typeof v === "string") ?? false;
        });

    const newData: DataFrameData = {};
    const newColumns: string[] = [];

    // Copy non-dummy columns
    for (const c of this._columns) {
      if (!cols.includes(c)) {
        newData[c] = [...(this._data.get(c) ?? [])];
        newColumns.push(c);
      }
    }

    // Create dummy columns for each selected column
    for (const col of cols) {
      const colData = this._data.get(col);
      if (!colData) continue;
      const categories = [...new Set(colData.map(String))].sort();
      const startIdx = options.dropFirst ? 1 : 0;
      for (let ci = startIdx; ci < categories.length; ci++) {
        const cat = categories[ci]!;
        const prefix = options.prefix ?? col;
        const dummyName = `${prefix}_${cat}`;
        newData[dummyName] = colData.map((v) => (String(v) === cat ? 1 : 0));
        newColumns.push(dummyName);
      }
    }

    return new DataFrame(newData, { columns: newColumns, index: this._index });
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
   * // "  a  b\n0 1  3\n1 2  4"
   * ```
   */
  toString(maxRows = 20): string {
    const nRows = this.shape[0] ?? 0;
    const cols = this._columns;

    // Determine which rows to show
    const half = Math.floor(maxRows / 2);
    const showAll = nRows <= maxRows;
    const topCount = showAll ? nRows : half;
    const bottomCount = showAll ? 0 : half;

    // Build header + data rows
    const allRows: string[][] = [];

    // Header row
    allRows.push(["", ...cols]);

    for (let i = 0; i < topCount; i++) {
      const idx = this._index[i];
      const row: string[] = [String(idx ?? i)];
      for (const col of cols) {
        const colData = this._data.get(col);
        const val = colData ? colData[i] : undefined;
        row.push(val === null || val === undefined ? "null" : String(val));
      }
      allRows.push(row);
    }

    if (!showAll) {
      allRows.push(["...", ...cols.map(() => "...")]);
      for (let i = nRows - bottomCount; i < nRows; i++) {
        const idx = this._index[i];
        const row: string[] = [String(idx ?? i)];
        for (const col of cols) {
          const colData = this._data.get(col);
          const val = colData ? colData[i] : undefined;
          row.push(val === null || val === undefined ? "null" : String(val));
        }
        allRows.push(row);
      }
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
        const cell = row[c] ?? "";
        const w = widths[c] ?? 0;
        cells.push(cell.padStart(w));
      }
      lines.push(cells.join("  "));
    }

    return lines.join("\n");
  }

  /** Element-wise absolute value for all numeric columns. */
  abs(): DataFrame {
    const data: DataFrameData = {};
    for (const col of this._columns) {
      const colData = this._data.get(col)!;
      data[col] = colData.map((v) => (typeof v === "number" ? Math.abs(v) : v));
    }
    return new DataFrame(data, {
      columns: [...this._columns],
      index: [...this._index],
    });
  }

  /** Round numeric columns to given number of decimal places. */
  round(decimals = 0): DataFrame {
    const factor = 10 ** decimals;
    const data: DataFrameData = {};
    for (const col of this._columns) {
      const colData = this._data.get(col)!;
      data[col] = colData.map((v) => (typeof v === "number" ? Math.round(v * factor) / factor : v));
    }
    return new DataFrame(data, {
      columns: [...this._columns],
      index: [...this._index],
    });
  }

  /** Count unique values per column. Returns a Series. */
  nunique(): Series {
    const counts: number[] = [];
    for (const col of this._columns) {
      const colData = this._data.get(col)!;
      counts.push(new Set(colData).size);
    }
    return new Series(counts, { index: this._columns });
  }

  /** Return the first n rows ordered by column values (largest). */
  nlargest(n: number, column: string): DataFrame {
    const colData = this._data.get(column);
    if (!colData) throw new IndexError(`Column '${column}' not found`);
    const indices = Array.from({ length: this._index.length }, (_, i) => i);
    indices.sort((a, b) => {
      const va = colData[a],
        vb = colData[b];
      if (typeof va === "number" && typeof vb === "number") return vb - va;
      return 0;
    });
    const selected = indices.slice(0, n);
    const data: DataFrameData = {};
    for (const col of this._columns) {
      const cd = this._data.get(col)!;
      data[col] = selected.map((i) => cd[i]);
    }
    return new DataFrame(data, {
      columns: [...this._columns],
      index: selected.map((i) => this._index[i]!),
    });
  }

  /** Return the first n rows ordered by column values (smallest). */
  nsmallest(n: number, column: string): DataFrame {
    const colData = this._data.get(column);
    if (!colData) throw new IndexError(`Column '${column}' not found`);
    const indices = Array.from({ length: this._index.length }, (_, i) => i);
    indices.sort((a, b) => {
      const va = colData[a],
        vb = colData[b];
      if (typeof va === "number" && typeof vb === "number") return va - vb;
      return 0;
    });
    const selected = indices.slice(0, n);
    const data: DataFrameData = {};
    for (const col of this._columns) {
      const cd = this._data.get(col)!;
      data[col] = selected.map((i) => cd[i]);
    }
    return new DataFrame(data, {
      columns: [...this._columns],
      index: selected.map((i) => this._index[i]!),
    });
  }

  /** Return index label of minimum value for each numeric column. */
  idxmin(): Series {
    const result: (string | number | null)[] = [];
    for (const col of this._columns) {
      const colData = this._data.get(col)!;
      let minVal = Infinity;
      let minIdx: string | number | null = null;
      for (let i = 0; i < colData.length; i++) {
        const v = colData[i];
        if (typeof v === "number" && v < minVal) {
          minVal = v;
          minIdx = this._index[i]!;
        }
      }
      result.push(minIdx);
    }
    return new Series(result, { index: this._columns });
  }

  /** Return index label of maximum value for each numeric column. */
  idxmax(): Series {
    const result: (string | number | null)[] = [];
    for (const col of this._columns) {
      const colData = this._data.get(col)!;
      let maxVal = -Infinity;
      let maxIdx: string | number | null = null;
      for (let i = 0; i < colData.length; i++) {
        const v = colData[i];
        if (typeof v === "number" && v > maxVal) {
          maxVal = v;
          maxIdx = this._index[i]!;
        }
      }
      result.push(maxIdx);
    }
    return new Series(result, { index: this._columns });
  }

  /** Check if values are between left and right (inclusive by default). */
  between(column: string, left: number, right: number, inclusive = true): DataFrame {
    const colData = this._data.get(column);
    if (!colData) throw new IndexError(`Column '${column}' not found`);
    const selected: number[] = [];
    for (let i = 0; i < colData.length; i++) {
      const v = colData[i];
      if (typeof v !== "number") continue;
      if (inclusive ? v >= left && v <= right : v > left && v < right) selected.push(i);
    }
    const data: DataFrameData = {};
    for (const col of this._columns) {
      const cd = this._data.get(col)!;
      data[col] = selected.map((i) => cd[i]);
    }
    return new DataFrame(data, {
      columns: [...this._columns],
      index: selected.map((i) => this._index[i]!),
    });
  }

  /** Add or modify columns functionally. */
  assign(
    columns: Record<string, unknown[] | ((row: Record<string, unknown>, i: number) => unknown)>
  ): DataFrame {
    const nRows = this._index.length;
    const data: DataFrameData = {};
    for (const col of this._columns) {
      data[col] = [...this._data.get(col)!];
    }
    const colNames = [...this._columns];
    for (const [name, valueOrFn] of Object.entries(columns)) {
      if (typeof valueOrFn === "function") {
        const fn = valueOrFn as (row: Record<string, unknown>, i: number) => unknown;
        const arr: unknown[] = [];
        for (let i = 0; i < nRows; i++) {
          const row: Record<string, unknown> = {};
          for (const col of this._columns) row[col] = this._data.get(col)![i];
          arr.push(fn(row, i));
        }
        data[name] = arr;
      } else {
        data[name] = [...(valueOrFn as unknown[])];
      }
      if (!colNames.includes(name)) colNames.push(name);
    }
    return new DataFrame(data, { columns: colNames, index: [...this._index] });
  }

  /** Replace values where condition is false with other. */
  where(
    cond: boolean[] | ((val: unknown, i: number) => boolean),
    other: unknown = null
  ): DataFrame {
    const data: DataFrameData = {};
    for (const col of this._columns) {
      const colData = this._data.get(col)!;
      data[col] = colData.map((v, i) => {
        const keep = typeof cond === "function" ? cond(v, i) : cond[i];
        return keep ? v : other;
      });
    }
    return new DataFrame(data, {
      columns: [...this._columns],
      index: [...this._index],
    });
  }

  /** Replace values where condition is true with other. Inverse of where(). */
  mask(cond: boolean[] | ((val: unknown, i: number) => boolean), other: unknown = null): DataFrame {
    const data: DataFrameData = {};
    for (const col of this._columns) {
      const colData = this._data.get(col)!;
      data[col] = colData.map((v, i) => {
        const shouldMask = typeof cond === "function" ? cond(v, i) : cond[i];
        return shouldMask ? other : v;
      });
    }
    return new DataFrame(data, {
      columns: [...this._columns],
      index: [...this._index],
    });
  }

  /** Cast columns to a target type. */
  astype(column: string, dtype: "number" | "string" | "boolean"): DataFrame {
    const colData = this._data.get(column);
    if (!colData) throw new IndexError(`Column '${column}' not found`);
    const data: DataFrameData = {};
    for (const col of this._columns) {
      if (col === column) {
        switch (dtype) {
          case "number":
            data[col] = colData.map((v) => Number(v));
            break;
          case "string":
            data[col] = colData.map((v) => String(v));
            break;
          case "boolean":
            data[col] = colData.map((v) => Boolean(v));
            break;
        }
      } else {
        data[col] = [...this._data.get(col)!];
      }
    }
    return new DataFrame(data, {
      columns: [...this._columns],
      index: [...this._index],
    });
  }

  /** Print summary of DataFrame (dtypes, non-null counts). */
  info(): string {
    const lines: string[] = [];
    lines.push(`<DataFrame>`);
    lines.push(`RangeIndex: ${this._index.length} entries`);
    lines.push(`Data columns (total ${this._columns.length} columns):`);
    lines.push(` #   Column  Non-Null Count  Dtype`);
    lines.push(`---  ------  --------------  -----`);
    for (let i = 0; i < this._columns.length; i++) {
      const col = this._columns[i]!;
      const colData = this._data.get(col)!;
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

  /** Return whether any element is true (per column). */
  any(axis: 0 | 1 = 0): Series | boolean[] {
    if (axis === 0) {
      const result: boolean[] = [];
      for (const col of this._columns) {
        const colData = this._data.get(col)!;
        result.push(colData.some((v) => Boolean(v)));
      }
      return new Series(result, { index: this._columns });
    }
    return this._index.map((_, i) => {
      for (const col of this._columns) {
        if (this._data.get(col)![i]) return true;
      }
      return false;
    });
  }

  /** Return whether all elements are true (per column). */
  all(axis: 0 | 1 = 0): Series | boolean[] {
    if (axis === 0) {
      const result: boolean[] = [];
      for (const col of this._columns) {
        const colData = this._data.get(col)!;
        result.push(colData.every((v) => Boolean(v)));
      }
      return new Series(result, { index: this._columns });
    }
    return this._index.map((_, i) => {
      for (const col of this._columns) {
        if (!this._data.get(col)![i]) return false;
      }
      return true;
    });
  }

  /** Count occurrences of unique value combinations across specified columns. */
  value_counts(...columns: string[]): DataFrame {
    const cols = columns.length > 0 ? columns : this._columns;
    for (const c of cols) {
      if (!this._data.has(c)) throw new IndexError(`Column '${c}' not found`);
    }
    const counts = new Map<string, number>();
    const keyRows = new Map<string, unknown[]>();
    for (let i = 0; i < this._index.length; i++) {
      const vals = cols.map((c) => this._data.get(c)![i]);
      const key = vals.map((v) => String(v)).join("|||");
      counts.set(key, (counts.get(key) ?? 0) + 1);
      if (!keyRows.has(key)) keyRows.set(key, vals);
    }
    const sorted = [...counts.entries()].sort((a, b) => b[1] - a[1]);
    const data: DataFrameData = {};
    for (const c of cols) data[c] = [];
    data["count"] = [];
    for (const [key, count] of sorted) {
      const vals = keyRows.get(key)!;
      for (let j = 0; j < cols.length; j++) {
        (data[cols[j]!] as unknown[]).push(vals[j]);
      }
      (data["count"] as unknown[]).push(count);
    }
    return new DataFrame(data, { columns: [...cols, "count"] });
  }

  /**
   * Filter rows using a simple expression string.
   *
   * Supports: `column op value` with `and`/`or` connectors.
   * Operators: `==`, `!=`, `>`, `>=`, `<`, `<=`.
   *
   * @param expr - Query expression string, e.g. `"age > 30 and salary > 50000"`
   */
  query(expr: string): DataFrame {
    const tokens = expr.split(/\s+(and|or)\s+/i);
    const conditions: {
      col: string;
      op: string;
      val: unknown;
      connector: string;
      valColumn?: string;
    }[] = [];
    let connector = "and";
    for (const token of tokens) {
      const t = token.trim();
      if (t.toLowerCase() === "and" || t.toLowerCase() === "or") {
        connector = t.toLowerCase();
        continue;
      }
      const match = t.match(/^(\w+)\s*(==|!=|>=|<=|>|<)\s*(.+)$/);
      if (!match) {
        throw new InvalidParameterError(`Invalid query expression: '${t}'`, "expr", expr);
      }
      const col = match[1]!;
      const op = match[2]!;
      const valStr = match[3]!.trim();
      let val: unknown;
      let valColumn: string | undefined;
      if (
        (valStr.startsWith("'") && valStr.endsWith("'")) ||
        (valStr.startsWith('"') && valStr.endsWith('"'))
      ) {
        val = valStr.slice(1, -1);
      } else if (valStr === "true") {
        val = true;
      } else if (valStr === "false") {
        val = false;
      } else if (valStr === "null" || valStr === "NaN") {
        val = null;
      } else if (/^[a-zA-Z_]\w*$/.test(valStr) && this._data.has(valStr)) {
        // Bare identifier matching a column name → column-vs-column comparison
        // (e.g. `a > b`), not the string literal "b".
        valColumn = valStr;
      } else {
        const num = Number(valStr);
        val = Number.isNaN(num) ? valStr : num;
      }
      if (!this._data.has(col)) throw new IndexError(`Column '${col}' not found`);
      conditions.push({ col, op, val, connector, ...(valColumn ? { valColumn } : {}) });
    }
    const evalCondition = (
      c: { col: string; op: string; val: unknown; valColumn?: string },
      i: number
    ): boolean => {
      const cellVal = this._data.get(c.col)![i];
      // For column-vs-column comparisons, resolve the RHS from that column.
      const rhs = c.valColumn !== undefined ? this._data.get(c.valColumn)![i] : c.val;
      c = { ...c, val: rhs };
      switch (c.op) {
        case "==":
          return cellVal === c.val;
        case "!=":
          return cellVal !== c.val;
        case ">":
          return typeof cellVal === "number" && typeof c.val === "number" && cellVal > c.val;
        case ">=":
          return typeof cellVal === "number" && typeof c.val === "number" && cellVal >= c.val;
        case "<":
          return typeof cellVal === "number" && typeof c.val === "number" && cellVal < c.val;
        case "<=":
          return typeof cellVal === "number" && typeof c.val === "number" && cellVal <= c.val;
        default:
          return false;
      }
    };
    const newData: DataFrameData = {};
    for (const col of this._columns) newData[col] = [];
    const newIndex: (string | number)[] = [];
    for (let i = 0; i < this._index.length; i++) {
      let result = evalCondition(conditions[0]!, i);
      for (let c = 1; c < conditions.length; c++) {
        const cond = conditions[c]!;
        const val = evalCondition(cond, i);
        if (cond.connector === "and") result = result && val;
        else result = result || val;
      }
      if (result) {
        for (const col of this._columns) {
          (newData[col] as unknown[]).push(this._data.get(col)![i]);
        }
        newIndex.push(this._index[i]!);
      }
    }
    return new DataFrame(newData, { columns: this._columns, index: newIndex });
  }

  /** Return approximate memory usage in bytes per column. */
  memory_usage(): DataFrame {
    const cols: string[] = [];
    const bytes: number[] = [];
    for (const col of this._columns) {
      const data = this._data.get(col)!;
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
   * Fill NaN/null values by interpolation.
   *
   * @param method - Interpolation method: 'linear' (default) or 'nearest'
   */
  interpolate(method: "linear" | "nearest" = "linear"): DataFrame {
    const newData: DataFrameData = {};
    for (const col of this._columns) {
      const data = this._data.get(col)!;
      const result = [...data];
      if (method === "linear") {
        for (let i = 0; i < result.length; i++) {
          if (!isValidNumber(result[i])) {
            let prevIdx = -1;
            let nextIdx = -1;
            for (let j = i - 1; j >= 0; j--) {
              if (isValidNumber(result[j])) {
                prevIdx = j;
                break;
              }
            }
            for (let j = i + 1; j < result.length; j++) {
              if (isValidNumber(data[j])) {
                nextIdx = j;
                break;
              }
            }
            if (prevIdx >= 0 && nextIdx >= 0) {
              const prevVal = result[prevIdx] as number;
              const nextVal = data[nextIdx] as number;
              const frac = (i - prevIdx) / (nextIdx - prevIdx);
              result[i] = prevVal + frac * (nextVal - prevVal);
            } else if (prevIdx >= 0) {
              result[i] = result[prevIdx];
            } else if (nextIdx >= 0) {
              result[i] = data[nextIdx];
            }
          }
        }
      } else {
        for (let i = 0; i < result.length; i++) {
          if (!isValidNumber(result[i])) {
            let prevIdx = -1;
            let nextIdx = -1;
            for (let j = i - 1; j >= 0; j--) {
              if (isValidNumber(result[j])) {
                prevIdx = j;
                break;
              }
            }
            for (let j = i + 1; j < result.length; j++) {
              if (isValidNumber(data[j])) {
                nextIdx = j;
                break;
              }
            }
            if (prevIdx >= 0 && nextIdx >= 0) {
              result[i] = i - prevIdx <= nextIdx - i ? result[prevIdx] : data[nextIdx];
            } else if (prevIdx >= 0) {
              result[i] = result[prevIdx];
            } else if (nextIdx >= 0) {
              result[i] = data[nextIdx];
            }
          }
        }
      }
      newData[col] = result;
    }
    return new DataFrame(newData, {
      columns: this._columns,
      index: [...this._index],
    });
  }

  /**
   * Pivot with aggregation (unlike pivot() which requires unique index/column pairs).
   *
   * @param options.index - Column to use as new index
   * @param options.columns - Column whose values become new columns
   * @param options.values - Column to aggregate
   * @param options.aggFunc - Aggregation function: 'mean', 'sum', 'count', 'min', 'max', 'first', 'last' (default: 'mean')
   */
  pivot_table(options: {
    index: string;
    columns: string;
    values: string;
    aggFunc?: "mean" | "sum" | "count" | "min" | "max" | "first" | "last";
  }): DataFrame {
    const { index: idxCol, columns: colCol, values: valCol } = options;
    const aggFunc = options.aggFunc ?? "mean";
    for (const c of [idxCol, colCol, valCol]) {
      if (!this._data.has(c)) throw new IndexError(`Column '${c}' not found`);
    }
    const idxData = this._data.get(idxCol)!;
    const colData = this._data.get(colCol)!;
    const valData = this._data.get(valCol)!;

    const rowKeys: unknown[] = [];
    const colKeys: unknown[] = [];
    const groups = new Map<string, number[]>();

    for (let i = 0; i < this._index.length; i++) {
      const rk = idxData[i];
      const ck = colData[i];
      if (!rowKeys.includes(rk)) rowKeys.push(rk);
      if (!colKeys.includes(ck)) colKeys.push(ck);
      // Type-aware composite key so distinct values that stringify the same
      // (e.g. number 1 vs string "1") don't collapse into one group and
      // double-count. `createKey` includes type tags and is collision-safe.
      const key = createKey([rk, ck]);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key)!.push(i);
    }

    const agg = (indices: number[]): unknown => {
      const vals = indices.map((i) => valData[i]).filter(isValidNumber) as number[];
      if (vals.length === 0) return null;
      switch (aggFunc) {
        case "sum":
          return vals.reduce((a, b) => a + b, 0);
        case "count":
          return vals.length;
        case "min":
          return Math.min(...vals);
        case "max":
          return Math.max(...vals);
        case "first":
          return vals[0];
        case "last":
          return vals[vals.length - 1];
        default:
          return vals.reduce((a, b) => a + b, 0) / vals.length;
      }
    };

    const result: DataFrameData = {};
    const newCols = colKeys.map(String);
    for (const ck of newCols) result[ck] = [];
    const newIndex: (string | number)[] = [];

    for (const rk of rowKeys) {
      newIndex.push(rk as string | number);
      for (const ck of colKeys) {
        const key = createKey([rk, ck]);
        const indices = groups.get(key);
        (result[String(ck)] as unknown[]).push(indices ? agg(indices) : null);
      }
    }

    return new DataFrame(result, { columns: newCols, index: newIndex });
  }

  /**
   * Expanding (cumulative) window calculations.
   *
   * @param minPeriods - Minimum number of observations to produce a result (default: 1)
   */
  expanding(minPeriods = 1): Expanding {
    return new Expanding(this, minPeriods);
  }

  /**
   * Exponentially weighted moving calculations.
   *
   * @param options.span - Decay span (alpha = 2 / (span + 1))
   * @param options.alpha - Smoothing factor directly (overrides span)
   */
  ewm(options: { span?: number; alpha?: number }): EWM {
    let alpha: number;
    if (options.alpha !== undefined) {
      alpha = options.alpha;
    } else if (options.span !== undefined) {
      alpha = 2 / (options.span + 1);
    } else {
      throw new InvalidParameterError("ewm requires span or alpha", "options", options);
    }
    if (alpha <= 0 || alpha > 1) {
      throw new InvalidParameterError("alpha must be in (0, 1]", "alpha", alpha);
    }
    return new EWM(this, alpha);
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
    const data: DataFrameData = {};
    for (const col of this._columns) {
      const colData = this._data.get(col);
      data[col] = colData ? [...colData] : [];
    }
    return new DataFrame(data, {
      columns: [...this._columns],
      index: [...this._index],
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
    const data: DataFrameData = {};
    for (const col of this._columns) {
      const colData = this._data.get(col);
      data[col] = colData ? colData.map((v) => valueSet.has(v)) : [];
    }
    return new DataFrame(data, {
      columns: [...this._columns],
      index: [...this._index],
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
      const newData: DataFrameData = {};
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
    const newData: DataFrameData = {};
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
   * Supports simple arithmetic expressions referencing column names.
   * Operators: `+`, `-`, `*`, `/`, `**` (power).
   * The expression must be an assignment like `"c = a + b"` to create a new column,
   * or a comparison like `"a > 2"` to filter rows.
   *
   * @param expr - Expression string
   * @returns New DataFrame with the result
   *
   * @example
   * ```ts
   * const df = new DataFrame({ a: [1, 2, 3], b: [4, 5, 6] });
   * df.eval('c = a + b');  // adds column c = [5, 7, 9]
   * df.eval('a > 1');      // filters to rows where a > 1
   * ```
   */
  eval(expr: string): DataFrame {
    const trimmed = expr.trim();

    // Check if it's an assignment: "newCol = expression".
    // The `=` must be a lone assignment, NOT part of ==, >=, <=, or != — the
    // negative lookahead/lookbehind prevents `eval("a == 3")` from being
    // parsed as `a = (= 3)` and silently overwriting column `a`.
    const assignMatch = trimmed.match(/^([a-zA-Z_]\w*)\s*(?<![=<>!])=(?!=)\s*(.+)$/);
    if (assignMatch) {
      const newColName = assignMatch[1]!;
      const rhsExpr = assignMatch[2]!;
      const values = this.evalExprPerRow(rhsExpr);
      const data: DataFrameData = {};
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
    const data: DataFrameData = {};
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

  private evalExprPerRow(expr: string): unknown[] {
    // Tokenize: split into tokens preserving operators and column names
    const tokens = expr.match(/[a-zA-Z_]\w*|\d+(?:\.\d+)?|\*\*|>=|<=|==|!=|[+\-*/()<>!=%]/g);
    if (!tokens || tokens.length === 0) {
      throw new InvalidParameterError(`eval: could not parse expression '${expr}'`, "expr", expr);
    }

    const n = this._index.length;
    const results: unknown[] = new Array(n);

    // Compile the expression once into a closure tree; the previous
    // implementation re-tokenized, stringified and re-parsed per row
    // (~50x slower on 10K rows). Column leaves reproduce the original
    // Number(String(value)) coercion exactly.
    const identRe = /^[a-zA-Z_]\w*$/;
    type Leaf = { col: unknown[] | null; constVal: number };
    const leaves: Leaf[] = tokens.map((token) => {
      if (identRe.test(token) && this._data.has(token)) {
        return { col: this._data.get(token) as unknown[], constVal: 0 };
      }
      return { col: null, constVal: Number(token) };
    });

    let pos = 0;
    let row = 0;
    const colValue = (leaf: Leaf): number => {
      const val = (leaf.col as unknown[])[row];
      if (val === null || val === undefined) return NaN;
      return Number(String(val));
    };

    type Node = () => number;
    const parseAtom = (): Node => {
      const t = tokens[pos];
      if (t === "(") {
        pos++;
        const inner = parseAdd();
        pos++; // )
        return inner;
      }
      if (t === "-") {
        pos++;
        const inner = parseAtom();
        return () => -inner();
      }
      if (t === undefined) {
        throw new InvalidParameterError("eval: unexpected end of expression", "expr", expr);
      }
      const leaf = leaves[pos]!;
      pos++;
      if (leaf.col) {
        return () => colValue(leaf);
      }
      const c = leaf.constVal;
      return () => c;
    };
    const parsePow = (): Node => {
      let left = parseAtom();
      while (tokens[pos] === "**") {
        pos++;
        const right = parseAtom();
        const l = left;
        left = () => l() ** right();
      }
      return left;
    };
    const parseMul = (): Node => {
      let left = parsePow();
      while (tokens[pos] === "*" || tokens[pos] === "/") {
        const op = tokens[pos];
        pos++;
        const right = parsePow();
        const l = left;
        left = op === "*" ? () => l() * right() : () => l() / right();
      }
      return left;
    };
    const parseAdd = (): Node => {
      let left = parseMul();
      while (tokens[pos] === "+" || tokens[pos] === "-") {
        const op = tokens[pos];
        pos++;
        const right = parseMul();
        const l = left;
        left = op === "+" ? () => l() + right() : () => l() - right();
      }
      return left;
    };

    const leftNode = parseAdd();
    let evalRow: () => unknown = leftNode;
    const cmp = tokens[pos];
    if (
      cmp === ">=" ||
      cmp === "<=" ||
      cmp === "==" ||
      cmp === "!=" ||
      cmp === ">" ||
      cmp === "<"
    ) {
      pos++;
      const rightNode = parseAdd();
      if (cmp === ">=") evalRow = () => leftNode() >= rightNode();
      else if (cmp === "<=") evalRow = () => leftNode() <= rightNode();
      else if (cmp === "==") evalRow = () => leftNode() === rightNode();
      else if (cmp === "!=") evalRow = () => leftNode() !== rightNode();
      else if (cmp === ">") evalRow = () => leftNode() > rightNode();
      else evalRow = () => leftNode() < rightNode();
    }

    for (let i = 0; i < n; i++) {
      row = i;
      results[i] = evalRow();
    }
    return results;
  }

  /**
   * Compute a cross-tabulation of two columns.
   *
   * Returns a DataFrame where rows correspond to unique values of `rowCol`,
   * columns correspond to unique values of `colCol`, and cell values are counts.
   *
   * @param rowCol - Column name for row grouping
   * @param colCol - Column name for column grouping
   * @returns Cross-tabulation DataFrame
   *
   * @example
   * ```ts
   * const df = new DataFrame({
   *   gender: ['M', 'F', 'M', 'F', 'M'],
   *   handed: ['R', 'R', 'L', 'R', 'R'],
   * });
   * df.crosstab('gender', 'handed');
   * // DataFrame with rows M/F, columns L/R, values are counts
   * ```
   */
  crosstab(rowCol: string, colCol: string): DataFrame {
    if (!this._data.has(rowCol)) throw new IndexError(`Column '${rowCol}' not found`);
    if (!this._data.has(colCol)) throw new IndexError(`Column '${colCol}' not found`);

    const rowData = this._data.get(rowCol)!;
    const colData = this._data.get(colCol)!;

    const rowKeys: unknown[] = [];
    const colKeys: unknown[] = [];
    const counts = new Map<string, number>();

    for (let i = 0; i < this._index.length; i++) {
      const rk = rowData[i];
      const ck = colData[i];
      if (!rowKeys.includes(rk)) rowKeys.push(rk);
      if (!colKeys.includes(ck)) colKeys.push(ck);
      const key = `${String(rk)}|||${String(ck)}`;
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }

    const newCols = colKeys.map(String);
    const result: DataFrameData = {};
    for (const ck of newCols) result[ck] = [];
    const newIndex: (string | number)[] = [];

    for (const rk of rowKeys) {
      newIndex.push(rk as string | number);
      for (const ck of colKeys) {
        const key = `${String(rk)}|||${String(ck)}`;
        (result[String(ck)] as unknown[]).push(counts.get(key) ?? 0);
      }
    }

    return new DataFrame(result, { columns: newCols, index: newIndex });
  }
}

/**
 * Expanding (cumulative) window calculations on DataFrame columns.
 */
class Expanding {
  private df: DataFrame;
  private minPeriods: number;

  constructor(df: DataFrame, minPeriods: number) {
    this.df = df;
    this.minPeriods = minPeriods;
  }

  private compute(fn: (vals: number[]) => number | null): DataFrame {
    const newData: DataFrameData = {};
    for (const col of this.df.columns) {
      const colData = this.df.getColumnData(col) as unknown[];
      const result: unknown[] = [];
      const acc: number[] = [];
      for (let i = 0; i < colData.length; i++) {
        const v = colData[i];
        if (isValidNumber(v)) acc.push(v);
        if (acc.length < this.minPeriods) {
          result.push(null);
        } else {
          // fn must not mutate acc; passing it directly avoids an O(n^2)
          // copy per row.
          result.push(fn(acc));
        }
      }
      newData[col] = result;
    }
    return new DataFrame(newData, {
      columns: this.df.columns,
      index: this.df.index,
    });
  }

  /**
   * O(n) streaming aggregation with a running accumulator (same
   * left-to-right accumulation order as the previous per-prefix reduce,
   * so results are bit-identical).
   */
  private computeStreaming(update: (acc: number, v: number, count: number) => number): DataFrame {
    const newData: DataFrameData = {};
    for (const col of this.df.columns) {
      const colData = this.df.getColumnData(col) as unknown[];
      const result: unknown[] = [];
      let acc = 0;
      let count = 0;
      for (let i = 0; i < colData.length; i++) {
        const v = colData[i];
        if (isValidNumber(v)) {
          acc = update(acc, v, count);
          count++;
        }
        result.push(count < this.minPeriods ? null : acc);
      }
      newData[col] = result;
    }
    return new DataFrame(newData, {
      columns: this.df.columns,
      index: this.df.index,
    });
  }

  mean(): DataFrame {
    // Running sum matches reduce()'s accumulation order; divide per row.
    const newData: DataFrameData = {};
    for (const col of this.df.columns) {
      const colData = this.df.getColumnData(col) as unknown[];
      const result: unknown[] = [];
      let sum = 0;
      let count = 0;
      for (let i = 0; i < colData.length; i++) {
        const v = colData[i];
        if (isValidNumber(v)) {
          sum += v;
          count++;
        }
        result.push(count < this.minPeriods ? null : sum / count);
      }
      newData[col] = result;
    }
    return new DataFrame(newData, { columns: this.df.columns, index: this.df.index });
  }

  sum(): DataFrame {
    return this.computeStreaming((acc, v) => acc + v);
  }

  std(): DataFrame {
    return this.compute((vals) => {
      if (vals.length < 2) return null;
      const m = vals.reduce((a, b) => a + b, 0) / vals.length;
      return Math.sqrt(vals.reduce((a, b) => a + (b - m) ** 2, 0) / (vals.length - 1));
    });
  }

  var(): DataFrame {
    return this.compute((vals) => {
      if (vals.length < 2) return null;
      const m = vals.reduce((a, b) => a + b, 0) / vals.length;
      return vals.reduce((a, b) => a + (b - m) ** 2, 0) / (vals.length - 1);
    });
  }

  min(): DataFrame {
    return this.computeStreaming((acc, v, count) => (count === 0 ? v : Math.min(acc, v)));
  }

  max(): DataFrame {
    return this.computeStreaming((acc, v, count) => (count === 0 ? v : Math.max(acc, v)));
  }
}

/**
 * Exponentially weighted moving calculations on DataFrame columns.
 */
class EWM {
  private df: DataFrame;
  private alpha: number;

  constructor(df: DataFrame, alpha: number) {
    this.df = df;
    this.alpha = alpha;
  }

  mean(): DataFrame {
    const newData: DataFrameData = {};
    for (const col of this.df.columns) {
      const colData = this.df.getColumnData(col) as unknown[];
      const result: unknown[] = [];
      let ewm: number | null = null;
      for (let i = 0; i < colData.length; i++) {
        const v = colData[i];
        if (!isValidNumber(v)) {
          result.push(ewm);
          continue;
        }
        if (ewm === null) {
          ewm = v;
        } else {
          ewm = this.alpha * v + (1 - this.alpha) * ewm;
        }
        result.push(ewm);
      }
      newData[col] = result;
    }
    return new DataFrame(newData, {
      columns: this.df.columns,
      index: this.df.index,
    });
  }

  std(): DataFrame {
    const newData: DataFrameData = {};
    for (const col of this.df.columns) {
      const colData = this.df.getColumnData(col) as unknown[];
      const result: unknown[] = [];
      let ewmMean: number | null = null;
      let ewmVar = 0;
      let count = 0;
      for (let i = 0; i < colData.length; i++) {
        const v = colData[i];
        if (!isValidNumber(v)) {
          result.push(null);
          continue;
        }
        count++;
        if (ewmMean === null) {
          ewmMean = v;
          result.push(null);
        } else {
          const diff = v - ewmMean;
          ewmMean = this.alpha * v + (1 - this.alpha) * ewmMean;
          ewmVar = (1 - this.alpha) * (ewmVar + this.alpha * diff * diff);
          result.push(count >= 2 ? Math.sqrt(ewmVar) : null);
        }
      }
      newData[col] = result;
    }
    return new DataFrame(newData, {
      columns: this.df.columns,
      index: this.df.index,
    });
  }

  var(): DataFrame {
    const newData: DataFrameData = {};
    for (const col of this.df.columns) {
      const colData = this.df.getColumnData(col) as unknown[];
      const result: unknown[] = [];
      let ewmMean: number | null = null;
      let ewmVar = 0;
      let count = 0;
      for (let i = 0; i < colData.length; i++) {
        const v = colData[i];
        if (!isValidNumber(v)) {
          result.push(null);
          continue;
        }
        count++;
        if (ewmMean === null) {
          ewmMean = v;
          result.push(null);
        } else {
          const diff = v - ewmMean;
          ewmMean = this.alpha * v + (1 - this.alpha) * ewmMean;
          ewmVar = (1 - this.alpha) * (ewmVar + this.alpha * diff * diff);
          result.push(count >= 2 ? ewmVar : null);
        }
      }
      newData[col] = result;
    }
    return new DataFrame(newData, {
      columns: this.df.columns,
      index: this.df.index,
    });
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

  constructor(df: DataFrame, by: string | string[]) {
    this.df = df;
    this.by = by;
    // Build group map once during construction
    const buildResult = this.buildGroupMap();
    this.groupMap = buildResult.groupMap;
    this.keyValuesMap = buildResult.keyValuesMap;
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

    // Fast path: single column groupBy — avoid array allocation and composite key
    if (groupByCols.length === 1) {
      const colData = this.df.getColumnData(groupByCols[0] as string);
      // When every value is a primitive, the value itself is a collision-free
      // Map key (SameValueZero matches createKey semantics for numbers,
      // strings, booleans, null and NaN) — skipping per-row key-string
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
   * @param operations - Dictionary of column name to aggregation function
   * @returns New DataFrame with aggregated data
   *
   * @example
   * ```ts
   * const grouped = df.groupBy('category');
   * const result = grouped.agg({ value: 'sum', count: 'count' });
   * ```
   */
  agg(operations: Record<string, AggregateFunction | AggregateFunction[]>): DataFrame {
    const groupByCols = Array.isArray(this.by) ? this.by : [this.by];
    const resultData: DataFrameData = {};

    const outputColumns: string[] = [];

    // Initialize result columns (groupby columns + aggregated columns)
    for (const col of groupByCols) {
      resultData[col] = [];
      outputColumns.push(col);
    }

    for (const [col, aggFunc] of Object.entries(operations)) {
      if (Array.isArray(aggFunc)) {
        for (const fn of aggFunc) {
          const outCol = `${col}_${fn}`;
          resultData[outCol] = [];
          outputColumns.push(outCol);
        }
      } else {
        resultData[col] = [];
        outputColumns.push(col);
      }
    }

    // Process each group
    for (const [keyStr, indices] of this.groupMap.entries()) {
      // Add group key values
      const keyParts = this.keyValuesMap.get(keyStr);
      if (!keyParts) {
        throw new DataValidationError(`Missing key values for group: ${keyStr}`);
      }

      for (let i = 0; i < groupByCols.length; i++) {
        const groupCol = groupByCols[i];
        if (groupCol) resultData[groupCol]?.push(keyParts[i]);
      }

      // Apply aggregation functions
      for (const [col, aggFunc] of Object.entries(operations)) {
        // Use raw data array to avoid allocation
        const seriesData = this.df.getColumnData(col);
        const funcs = Array.isArray(aggFunc) ? aggFunc : [aggFunc];

        for (const func of funcs) {
          let result: unknown;

          switch (func) {
            case "count": {
              let count = 0;
              for (const idx of indices) {
                const val = seriesData[idx];
                if (val !== null && val !== undefined) count++;
              }
              result = count;
              break;
            }
            case "first": {
              const firstIdx = indices[0];
              result = firstIdx !== undefined ? seriesData[firstIdx] : undefined;
              break;
            }
            case "last": {
              const lastIdx = indices[indices.length - 1];
              result = lastIdx !== undefined ? seriesData[lastIdx] : undefined;
              break;
            }
            case "sum": {
              let sum = 0;
              let hasNumeric = false;
              for (const idx of indices) {
                const val = seriesData[idx];
                if (val === null || val === undefined) continue;
                if (typeof val !== "number") {
                  throw new DataValidationError("sum() only works on numbers");
                }
                if (isValidNumber(val)) {
                  sum += val;
                  hasNumeric = true;
                }
              }
              // Match Series behavior: throw if empty?
              // But for groupby, usually we return result per group.
              // If we throw here, one empty group crashes the whole groupby.
              // But the test expects a throw for invalid types.
              // For empty groups, we probably want to return 0 or NaN without crashing.
              // But if type is wrong, we crash.
              result = hasNumeric ? sum : 0;
              break;
            }
            case "mean": {
              let sum = 0;
              let count = 0;
              for (const idx of indices) {
                const val = seriesData[idx];
                if (val === null || val === undefined) continue;
                if (typeof val !== "number") {
                  throw new DataValidationError("mean() only works on numbers");
                }
                if (isValidNumber(val)) {
                  sum += val;
                  count++;
                }
              }
              result = count > 0 ? sum / count : NaN;
              break;
            }
            case "median": {
              const nums: number[] = [];
              for (const idx of indices) {
                const val = seriesData[idx];
                if (val === null || val === undefined) continue;
                if (typeof val !== "number") {
                  throw new DataValidationError("median() only works on numbers");
                }
                if (isValidNumber(val)) nums.push(val);
              }
              if (nums.length === 0) {
                result = NaN;
              } else {
                nums.sort((a, b) => a - b);
                const mid = Math.floor(nums.length / 2);
                if (nums.length % 2 === 0) {
                  const v1 = nums[mid - 1];
                  const v2 = nums[mid];
                  result = v1 !== undefined && v2 !== undefined ? (v1 + v2) / 2 : NaN;
                } else {
                  result = nums[mid] ?? NaN;
                }
              }
              break;
            }
            case "min": {
              let min = Infinity;
              let hasNumeric = false;
              for (const idx of indices) {
                const val = seriesData[idx];
                if (val === null || val === undefined) continue;
                if (typeof val !== "number") {
                  throw new DataValidationError("min() only works on numbers");
                }
                if (isValidNumber(val)) {
                  if (val < min) min = val;
                  hasNumeric = true;
                }
              }
              result = hasNumeric ? min : NaN;
              break;
            }
            case "max": {
              let max = -Infinity;
              let hasNumeric = false;
              for (const idx of indices) {
                const val = seriesData[idx];
                if (val === null || val === undefined) continue;
                if (typeof val !== "number") {
                  throw new DataValidationError("max() only works on numbers");
                }
                if (isValidNumber(val)) {
                  if (val > max) max = val;
                  hasNumeric = true;
                }
              }
              result = hasNumeric ? max : NaN;
              break;
            }
            case "std": {
              let sum = 0;
              let count = 0;
              const nums: number[] = [];
              // First pass: sum and collect numbers
              for (const idx of indices) {
                const val = seriesData[idx];
                if (val === null || val === undefined) continue;
                if (typeof val !== "number") {
                  throw new DataValidationError("std() only works on numbers");
                }
                if (isValidNumber(val)) {
                  sum += val;
                  count++;
                  nums.push(val);
                }
              }
              if (count < 2) {
                result = NaN;
              } else {
                const mean = sum / count;
                let sumSq = 0;
                for (const val of nums) {
                  sumSq += (val - mean) ** 2;
                }
                result = Math.sqrt(sumSq / (count - 1));
              }
              break;
            }
            case "var": {
              let sum = 0;
              let count = 0;
              const nums: number[] = [];
              // First pass: sum and collect numbers
              for (const idx of indices) {
                const val = seriesData[idx];
                if (val === null || val === undefined) continue;
                if (typeof val !== "number") {
                  throw new DataValidationError("var() only works on numbers");
                }
                if (isValidNumber(val)) {
                  sum += val;
                  count++;
                  nums.push(val);
                }
              }
              if (count < 2) {
                result = NaN;
              } else {
                const mean = sum / count;
                let sumSq = 0;
                for (const val of nums) {
                  sumSq += (val - mean) ** 2;
                }
                result = sumSq / (count - 1);
              }
              break;
            }
            default:
              throw new DataValidationError(`Unsupported aggregation function: ${func}`);
          }

          const outCol = Array.isArray(aggFunc) ? `${col}_${func}` : col;
          resultData[outCol]?.push(result);
        }
      }
    }

    return new DataFrame(resultData, { columns: outputColumns });
  }

  /**
   * Helper to identify numeric columns (excluding grouping columns).
   * @private
   */
  private getNumericColumns(): string[] {
    const groupByCols = Array.isArray(this.by) ? this.by : [this.by];
    const otherCols = this.df.columns.filter((c) => !groupByCols.includes(c));
    return otherCols.filter((col) => {
      const colData = this.df.get(col);
      // efficient check: look for at least one valid number
      return colData.data.some(isValidNumber);
    });
  }

  /**
   * Helper method to perform same aggregation on all numeric non-grouping columns.
   * @private
   */
  private aggNumeric(operation: AggregateFunction): DataFrame {
    const numericCols = this.getNumericColumns();
    const operations: Record<string, AggregateFunction> = {};
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

    const operations: Record<string, AggregateFunction> = {};
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
}

/**
 * Rolling window calculations on DataFrame columns.
 *
 * Created by DataFrame.rolling(). Provides mean, sum, std, var, min, max, apply.
 */
class Rolling {
  private df: DataFrame;
  private window: number;
  private on: string | undefined;

  constructor(df: DataFrame, window: number, on?: string) {
    this.df = df;
    this.window = window;
    this.on = on;
  }

  private getWindowValues(colData: unknown[], i: number): number[] {
    const start = i - this.window + 1;
    const vals: number[] = [];
    for (let j = Math.max(0, start); j <= i; j++) {
      const v = colData[j];
      if (isValidNumber(v)) vals.push(v);
    }
    return vals;
  }

  private compute(fn: (vals: number[]) => number | null): DataFrame {
    const newData: DataFrameData = {};
    const cols = this.on ? [this.on] : this.df.columns;
    // pandas default: min_periods equals the window, so a window that isn't
    // fully observed (out of bounds OR containing any NaN/null) yields NaN
    // rather than silently averaging over the remaining values.
    const minPeriods = this.window;
    for (const col of cols) {
      const colData = this.df.getColumnData(col) as unknown[];
      const result: unknown[] = [];
      for (let i = 0; i < colData.length; i++) {
        if (i < this.window - 1) {
          result.push(null);
        } else {
          const vals = this.getWindowValues(colData, i);
          result.push(vals.length < minPeriods ? null : fn(vals));
        }
      }
      newData[col] = result;
    }
    return new DataFrame(newData, { columns: cols, index: this.df.index });
  }

  mean(): DataFrame {
    // Sliding running sum in O(n) instead of the generic O(n·window) rebuild:
    // add the entering value, drop the value leaving the window, and track how
    // many of the `window` slots are valid so the min_periods=window rule
    // (any NaN/null in the window ⇒ null) is preserved exactly.
    const cols = this.on ? [this.on] : this.df.columns;
    const w = this.window;
    const newData: DataFrameData = {};
    for (const col of cols) {
      const colData = this.df.getColumnData(col) as unknown[];
      const n = colData.length;
      const result: unknown[] = new Array(n);
      let windowSum = 0;
      let validCount = 0;
      for (let i = 0; i < n; i++) {
        const v = colData[i];
        if (isValidNumber(v)) {
          windowSum += v;
          validCount++;
        }
        if (i >= w) {
          const old = colData[i - w];
          if (isValidNumber(old)) {
            windowSum -= old;
            validCount--;
          }
        }
        result[i] = i < w - 1 || validCount < w ? null : windowSum / w;
      }
      newData[col] = result;
    }
    return new DataFrame(newData, { columns: cols, index: this.df.index });
  }

  sum(): DataFrame {
    return this.compute((vals) => vals.reduce((a, b) => a + b, 0));
  }

  std(): DataFrame {
    return this.compute((vals) => {
      if (vals.length < 2) return null;
      const m = vals.reduce((a, b) => a + b, 0) / vals.length;
      const variance = vals.reduce((a, b) => a + (b - m) ** 2, 0) / (vals.length - 1);
      return Math.sqrt(variance);
    });
  }

  var(): DataFrame {
    return this.compute((vals) => {
      if (vals.length < 2) return null;
      const m = vals.reduce((a, b) => a + b, 0) / vals.length;
      return vals.reduce((a, b) => a + (b - m) ** 2, 0) / (vals.length - 1);
    });
  }

  min(): DataFrame {
    return this.compute((vals) => Math.min(...vals));
  }

  max(): DataFrame {
    return this.compute((vals) => Math.max(...vals));
  }

  apply(fn: (vals: number[]) => number | null): DataFrame {
    return this.compute(fn);
  }

  /**
   * Compute rolling pairwise correlation between two columns.
   *
   * @param col1 - First column name
   * @param col2 - Second column name
   * @returns DataFrame with a single column "{col1}_{col2}" containing rolling Pearson correlation
   */
  corr(col1: string, col2: string): DataFrame {
    const data1 = this.df.getColumnData(col1) as unknown[];
    const data2 = this.df.getColumnData(col2) as unknown[];
    const n = data1.length;
    const result: unknown[] = [];
    const label = `${col1}_${col2}`;

    for (let i = 0; i < n; i++) {
      if (i < this.window - 1) {
        result.push(null);
        continue;
      }
      const start = i - this.window + 1;
      const v1: number[] = [];
      const v2: number[] = [];
      for (let j = Math.max(0, start); j <= i; j++) {
        const a = data1[j];
        const b = data2[j];
        if (isValidNumber(a) && isValidNumber(b)) {
          v1.push(a);
          v2.push(b);
        }
      }
      if (v1.length < 2) {
        result.push(null);
        continue;
      }
      const m1 = v1.reduce((a, b) => a + b, 0) / v1.length;
      const m2 = v2.reduce((a, b) => a + b, 0) / v2.length;
      let num = 0,
        d1 = 0,
        d2 = 0;
      for (let k = 0; k < v1.length; k++) {
        const diff1 = (v1[k] ?? 0) - m1;
        const diff2 = (v2[k] ?? 0) - m2;
        num += diff1 * diff2;
        d1 += diff1 * diff1;
        d2 += diff2 * diff2;
      }
      result.push(d1 === 0 || d2 === 0 ? NaN : num / Math.sqrt(d1 * d2));
    }

    return new DataFrame({ [label]: result }, { index: this.df.index });
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
    const data1 = this.df.getColumnData(col1) as unknown[];
    const data2 = this.df.getColumnData(col2) as unknown[];
    const n = data1.length;
    const result: unknown[] = [];
    const label = `${col1}_${col2}`;

    for (let i = 0; i < n; i++) {
      if (i < this.window - 1) {
        result.push(null);
        continue;
      }
      const start = i - this.window + 1;
      const v1: number[] = [];
      const v2: number[] = [];
      for (let j = Math.max(0, start); j <= i; j++) {
        const a = data1[j];
        const b = data2[j];
        if (isValidNumber(a) && isValidNumber(b)) {
          v1.push(a);
          v2.push(b);
        }
      }
      if (v1.length < 2) {
        result.push(null);
        continue;
      }
      const m1 = v1.reduce((a, b) => a + b, 0) / v1.length;
      const m2 = v2.reduce((a, b) => a + b, 0) / v2.length;
      let covSum = 0;
      for (let k = 0; k < v1.length; k++) {
        covSum += ((v1[k] ?? 0) - m1) * ((v2[k] ?? 0) - m2);
      }
      result.push(covSum / (v1.length - 1));
    }

    return new DataFrame({ [label]: result }, { index: this.df.index });
  }
}
