import {
  DeepboxError,
  DTypeError,
  getConfig,
  InvalidParameterError,
  NotFittedError,
  ShapeError,
} from "../core";
import { CSRMatrix, empty, type Tensor, Tensor as TensorImpl, tensor, zeros } from "../ndarray";
import { assert2D, assertNumericTensor, getShape2D, getStride1D, getStrides2D } from "./_internal";

/**
 * Type representing a category value that can be a string, number, or bigint.
 * Used for categorical encoding operations.
 */
type Category = number | string | bigint;

type CategoryType = "string" | "number" | "bigint";

type CategoriesOption = "auto" | ReadonlyArray<ReadonlyArray<Category>>;

type DropOption = "first" | "if_binary" | null;

function getStringData(t: Tensor): string[] {
  if (t.dtype !== "string") {
    throw new DTypeError("Expected string tensor");
  }
  if (!Array.isArray(t.data)) {
    throw new DeepboxError("Internal error: invalid string tensor storage");
  }
  return t.data;
}

function getNumericData(t: Tensor): ArrayLike<number | bigint> {
  if (t.dtype === "string") {
    throw new DTypeError("Expected numeric tensor");
  }
  if (Array.isArray(t.data)) {
    throw new DeepboxError("Internal error: invalid numeric tensor storage");
  }
  return t.data;
}

function inferCategoryType(values: Category[], paramName: string): CategoryType {
  let hasString = false;
  let hasNumber = false;
  let hasBigInt = false;

  for (const value of values) {
    if (typeof value === "string") {
      hasString = true;
    } else if (typeof value === "number") {
      if (!Number.isFinite(value)) {
        throw new InvalidParameterError("Category values must be finite numbers", paramName, value);
      }
      hasNumber = true;
    } else if (typeof value === "bigint") {
      hasBigInt = true;
    }
  }

  const typeCount = (hasString ? 1 : 0) + (hasNumber ? 1 : 0) + (hasBigInt ? 1 : 0);

  if (typeCount === 0) {
    return "number";
  }
  if (typeCount > 1) {
    throw new InvalidParameterError("Mixed category types are not supported", paramName);
  }
  if (hasString) return "string";
  if (hasBigInt) return "bigint";
  return "number";
}

function sortCategories(values: Iterable<Category>, paramName: string): Category[] {
  const arr = Array.from(values);
  if (arr.length === 0) return arr;

  const categoryType = inferCategoryType(arr, paramName);

  if (categoryType === "string") {
    arr.sort((a, b) => {
      if (typeof a !== "string" || typeof b !== "string") {
        throw new DeepboxError("Internal error: inconsistent category types");
      }
      return a.localeCompare(b);
    });
    return arr;
  }

  if (categoryType === "bigint") {
    arr.sort((a, b) => {
      if (typeof a !== "bigint" || typeof b !== "bigint") {
        throw new DeepboxError("Internal error: inconsistent category types");
      }
      if (a < b) return -1;
      if (a > b) return 1;
      return 0;
    });
    return arr;
  }

  arr.sort((a, b) => {
    if (typeof a !== "number" || typeof b !== "number") {
      throw new DeepboxError("Internal error: inconsistent category types");
    }
    return a - b;
  });
  return arr;
}

function validateCategoryValues(values: ReadonlyArray<Category>, paramName: string): Category[] {
  if (values.length === 0) {
    throw new InvalidParameterError("categories must contain at least one value", paramName);
  }
  const arr = Array.from(values);
  inferCategoryType(arr, paramName);

  const seen = new Set<Category>();
  for (const value of arr) {
    if (seen.has(value)) {
      throw new InvalidParameterError(
        `categories must be unique; duplicate value ${String(value)}`,
        paramName,
        value
      );
    }
    seen.add(value);
  }
  return arr;
}

function resolveCategoriesOption(
  categoriesOption: CategoriesOption,
  nFeatures: number,
  paramName: string
): ReadonlyArray<ReadonlyArray<Category>> | null {
  if (categoriesOption === "auto") {
    return null;
  }
  if (!Array.isArray(categoriesOption)) {
    throw new InvalidParameterError(
      "categories must be 'auto' or an array of category arrays",
      paramName,
      categoriesOption
    );
  }
  if (categoriesOption.length !== nFeatures) {
    throw new InvalidParameterError(
      "categories length must match number of features",
      paramName,
      categoriesOption.length
    );
  }
  return categoriesOption;
}

/**
 * Reads a single value from a 1D tensor at the specified index.
 * Handles both string and numeric dtypes safely.
 *
 * @param t - The tensor to read from
 * @param i - The index to read at
 * @returns The value as a string, number, or bigint
 */
function read1DValue(t: Tensor, i: number): Category {
  const stride = getStride1D(t);
  const idx = t.offset + i * stride;
  if (t.dtype === "string") {
    const value = getStringData(t)[idx];
    if (value === undefined) {
      throw new DeepboxError("Internal error: string tensor access out of bounds");
    }
    return value;
  }
  const value = getNumericData(t)[idx];
  if (value === undefined) {
    throw new DeepboxError("Internal error: numeric tensor access out of bounds");
  }
  return typeof value === "bigint" ? value : Number(value);
}

/**
 * Reads a value from a 2D tensor at the specified row and column.
 * Handles both string and numeric dtypes safely.
 *
 * @param t - The tensor to read from
 * @param row - The row index
 * @param col - The column index
 * @returns The value as a string, number, or bigint
 */
function read2DValue(t: Tensor, row: number, col: number): Category {
  const [stride0, stride1] = getStrides2D(t);
  const idx = t.offset + row * stride0 + col * stride1;
  if (t.dtype === "string") {
    const value = getStringData(t)[idx];
    if (value === undefined) {
      throw new DeepboxError("Internal error: string tensor access out of bounds");
    }
    return value;
  }
  const value = getNumericData(t)[idx];
  if (value === undefined) {
    throw new DeepboxError("Internal error: numeric tensor access out of bounds");
  }
  return typeof value === "bigint" ? value : Number(value);
}

function assert1D(t: Tensor, name: string): void {
  if (t.ndim !== 1) {
    throw new ShapeError(`${name} must be a 1D tensor`);
  }
}

function categoryValueAt(values: Category[], index: number, context: string): Category {
  const value = values[index];
  if (value === undefined) {
    throw new DeepboxError(`Internal error: missing category at index ${index} (${context})`);
  }
  return value;
}

function inferCategoryTypeFromRows(rows: Category[][], paramName: string): CategoryType {
  const values: Category[] = [];
  for (const row of rows) {
    for (const value of row) {
      values.push(value);
    }
  }
  return inferCategoryType(values, paramName);
}

function emptyCategoryVectorFromClasses(classes: Category[], paramName: string): Tensor {
  const categoryType = inferCategoryType(classes, paramName);
  if (categoryType === "string") {
    return empty([0], { dtype: "string" });
  }
  if (categoryType === "bigint") {
    return empty([0], { dtype: "int64" });
  }
  return zeros([0], { dtype: "float64" });
}

function emptyCategoryMatrixFromCategories(
  categories: Category[][],
  nFeatures: number,
  paramName: string
): Tensor {
  const categoryType = inferCategoryTypeFromRows(categories, paramName);
  if (categoryType === "string") {
    return empty([0, nFeatures], { dtype: "string" });
  }
  if (categoryType === "bigint") {
    return empty([0, nFeatures], { dtype: "int64" });
  }
  return zeros([0, nFeatures], { dtype: "float64" });
}

function toCategoryVectorTensor(values: Category[], paramName = "y"): Tensor {
  const categoryType = inferCategoryType(values, paramName);

  if (categoryType === "string") {
    const out = new Array<string>(values.length);
    for (let i = 0; i < values.length; i++) {
      const value = values[i];
      if (typeof value !== "string") {
        throw new DeepboxError("Internal error: expected string category value");
      }
      out[i] = value;
    }
    return tensor(out);
  }

  if (categoryType === "bigint") {
    const out = new BigInt64Array(values.length);
    for (let i = 0; i < values.length; i++) {
      const value = values[i];
      if (typeof value !== "bigint") {
        throw new DeepboxError("Internal error: expected bigint category value");
      }
      out[i] = value;
    }
    return tensor(out);
  }

  const out = new Float64Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const value = values[i];
    if (value === undefined || typeof value !== "number") {
      throw new DeepboxError("Internal error: expected numeric category value");
    }
    out[i] = value;
  }
  return tensor(out);
}

function toCategoryMatrixTensor(values: Category[][], paramName = "X"): Tensor {
  const rows = values.length;
  const cols = rows > 0 ? (values[0]?.length ?? 0) : 0;

  for (let i = 0; i < rows; i++) {
    const row = values[i];
    if (!row) {
      throw new DeepboxError("Internal error: missing row in category matrix");
    }
    if (row.length !== cols) {
      throw new ShapeError("Ragged category matrix cannot be converted to tensor");
    }
  }

  const flat: Category[] = [];
  for (const row of values) {
    for (const value of row) {
      flat.push(value);
    }
  }

  const categoryType = inferCategoryType(flat, paramName);
  if (categoryType === "string") {
    const out = new Array<string[]>(rows);
    for (let i = 0; i < rows; i++) {
      const row = values[i];
      if (!row) {
        throw new DeepboxError("Internal error: missing row in category matrix");
      }
      const outRow = new Array<string>(cols);
      for (let j = 0; j < cols; j++) {
        const value = row[j];
        if (typeof value !== "string") {
          throw new DeepboxError("Internal error: expected string category value");
        }
        outRow[j] = value;
      }
      out[i] = outRow;
    }
    return tensor(out);
  }
  if (categoryType === "number") {
    const out = new Array<number[]>(rows);
    for (let i = 0; i < rows; i++) {
      const row = values[i];
      if (!row) {
        throw new DeepboxError("Internal error: missing row in category matrix");
      }
      const outRow = new Array<number>(cols);
      for (let j = 0; j < cols; j++) {
        const value = row[j];
        if (typeof value !== "number") {
          throw new DeepboxError("Internal error: expected numeric category value");
        }
        outRow[j] = value;
      }
      out[i] = outRow;
    }
    return tensor(out, { dtype: "float64" });
  }

  const data = new BigInt64Array(rows * cols);
  for (let i = 0; i < flat.length; i++) {
    const value = flat[i];
    if (typeof value !== "bigint") {
      throw new DeepboxError("Internal error: expected bigint category value");
    }
    data[i] = value;
  }

  const { defaultDevice } = getConfig();
  return TensorImpl.fromTypedArray({
    data,
    shape: [rows, cols],
    dtype: "int64",
    device: defaultDevice,
  });
}

/**
 * Encode target labels with value between 0 and n_classes-1.
 *
 * This transformer encodes categorical labels (strings or numbers) into integers
 * in the range [0, n_classes-1]. It maintains a mapping of unique classes to
 * their integer representations and can reverse the transformation.
 *
 * **Time Complexity:**
 * - fit: O(n) where n is the number of samples
 * - transform: O(n) with O(1) lookup per sample
 * - inverseTransform: O(n)
 *
 * **Space Complexity:** O(k) where k is the number of unique classes
 *
 * @example
 * ```js
 * import { LabelEncoder } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const y = tensor(['cat', 'dog', 'cat', 'bird']);
 * const encoder = new LabelEncoder();
 * encoder.fit(y);
 * const yEncoded = encoder.transform(y);  // [1, 2, 1, 0]
 * const yDecoded = encoder.inverseTransform(yEncoded); // ['cat', 'dog', 'cat', 'bird']
 * ```
 *
 * @see {@link https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html | Scikit-learn LabelEncoder}
 */
export class LabelEncoder {
  /** Indicates whether the encoder has been fitted to data */
  private fitted = false;
  /** Array of unique classes found during fitting, sorted for consistency */
  private classes_?: Category[];
  /** Map from class value to encoded integer index for O(1) lookup */
  private classToIndex_?: Map<Category, number>;

  /**
   * Fit label encoder to a set of labels.
   * Extracts unique classes and creates an index mapping.
   *
   * @param y - Target labels (1D tensor of strings or numbers)
   * @returns this - Returns self for method chaining
   * @throws {InvalidParameterError} If y is empty
   */
  fit(y: Tensor): this {
    assert1D(y, "y");
    if (y.size === 0) {
      throw new InvalidParameterError("Cannot fit LabelEncoder on empty array", "y");
    }

    // Collect unique classes using a Set for O(n) complexity
    const uniqueSet = new Set<Category>();
    for (let i = 0; i < y.size; i++) {
      uniqueSet.add(read1DValue(y, i));
    }

    // Sort classes for consistent ordering across fits
    this.classes_ = sortCategories(uniqueSet, "y");

    // Build index map for O(1) transform lookups
    this.classToIndex_ = new Map();
    for (let i = 0; i < this.classes_.length; i++) {
      this.classToIndex_.set(categoryValueAt(this.classes_, i, "LabelEncoder.fit"), i);
    }

    this.fitted = true;
    return this;
  }

  /**
   * Transform labels to normalized encoding.
   * Each unique label is mapped to an integer in [0, n_classes-1].
   *
   * @param y - Target labels to encode (1D tensor)
   * @returns Encoded labels as integer tensor
   * @throws {NotFittedError} If encoder is not fitted
   * @throws {InvalidParameterError} If y contains labels not seen during fit
   */
  transform(y: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("LabelEncoder must be fitted before transform");
    }
    assert1D(y, "y");
    if (y.size === 0) {
      return tensor([]);
    }

    const lookup = this.classToIndex_;
    if (!this.classes_ || !lookup) {
      throw new DeepboxError("LabelEncoder internal error: missing fitted state");
    }

    // Pre-allocate result array for better performance
    const result = new Array<number>(y.size);

    // Transform each label using O(1) map lookup
    for (let i = 0; i < y.size; i++) {
      const val = read1DValue(y, i);
      const idx = lookup.get(val);
      if (idx === undefined) {
        throw new InvalidParameterError(
          `Unknown label: ${String(val)}. Label must be present during fit.`,
          "y",
          val
        );
      }
      result[i] = idx;
    }

    return tensor(result, { dtype: "float64" });
  }

  /**
   * Fit label encoder and return encoded labels in one step.
   * Convenience method equivalent to calling fit(y).transform(y).
   *
   * @param y - Target labels (1D tensor)
   * @returns Encoded labels as integer tensor
   */
  fitTransform(y: Tensor): Tensor {
    return this.fit(y).transform(y);
  }

  /**
   * Transform integer labels back to original encoding.
   * Reverses the encoding performed by transform().
   *
   * @param y - Encoded labels (1D integer tensor)
   * @returns Original labels (strings or numbers)
   * @throws {NotFittedError} If encoder is not fitted
   * @throws {InvalidParameterError} If y contains invalid indices
   */
  inverseTransform(y: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("LabelEncoder must be fitted before inverse_transform");
    }
    assert1D(y, "y");
    assertNumericTensor(y, "y");
    const classes = this.classes_;
    if (!classes) {
      throw new DeepboxError("LabelEncoder internal error: missing fitted state");
    }
    if (y.size === 0) {
      return emptyCategoryVectorFromClasses(classes, "y");
    }

    const classesLen = classes.length;

    const result = new Array<Category>(y.size);
    const stride = getStride1D(y);
    const data = getNumericData(y);

    // Map each encoded index back to its original class
    for (let i = 0; i < y.size; i++) {
      const raw = data[y.offset + i * stride];
      if (raw === undefined) {
        throw new DeepboxError("Internal error: numeric tensor access out of bounds");
      }
      const idx = Number(raw);

      // Validate index is in valid range
      if (idx < 0 || idx >= classesLen || !Number.isInteger(idx)) {
        throw new InvalidParameterError(
          `Invalid label index: ${idx}. Must be integer in [0, ${classesLen - 1}]`,
          "y",
          idx
        );
      }

      result[i] = categoryValueAt(classes, idx, "LabelEncoder.inverseTransform");
    }

    // Return tensor with appropriate dtype
    return toCategoryVectorTensor(result, "y");
  }
}

/**
 * Encode categorical features as one-hot numeric array.
 *
 * This encoder transforms categorical features into a binary one-hot encoding.
 * Each categorical feature with n unique values is transformed into n binary features,
 * with only one active (set to 1) per sample.
 *
 * **Time Complexity:**
 * - fit: O(n*m) where n is samples, m is features
 * - transform: O(n*m*k) where k is average categories per feature
 * - Sparse mode is more efficient for high-cardinality features
 *
 * **Space Complexity:**
 * - Dense: O(n * sum(k_i)) where k_i is unique categories for feature i
 * - Sparse: O(nnz) where nnz is number of non-zero elements
 *
 * @example
 * ```js
 * const X = tensor([['red', 'S'], ['blue', 'M'], ['red', 'L']]);
 * const encoder = new OneHotEncoder({ sparse: false });
 * encoder.fit(X);
 * const encoded = encoder.transform(X);
 * // Result: [[1,0,1,0,0], [0,1,0,1,0], [1,0,0,0,1]]
 * ```
 *
 * @see {@link https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html | Scikit-learn OneHotEncoder}
 */
export class OneHotEncoder {
  /** Indicates whether the encoder has been fitted to data */
  private fitted = false;
  /** Array of unique categories for each feature */
  private categories_?: Category[][];
  /** Maps from category value to index for each feature (for O(1) lookup) */
  private categoryToIndex_?: Array<Map<Category, number>>;
  /** Whether to return sparse matrix (CSR) or dense array */
  private sparse: boolean;
  /** How to handle unknown categories during transform */
  private handleUnknown: "error" | "ignore";
  /** Drop policy to avoid collinearity */
  private drop: DropOption;
  /** Per-feature dropped category index */
  private dropIndices_?: Array<number | null>;
  /** Categories configuration */
  private categoriesOption: CategoriesOption;

  /**
   * Creates a new OneHotEncoder instance.
   *
   * @param options - Configuration options
   * @param options.sparse - If true, returns CSRMatrix; if false, returns dense Tensor (default: false)
   * @param options.sparseOutput - Alias for sparse (default: false)
   * @param options.handleUnknown - How to handle unknown categories (default: "error")
   * @param options.drop - If set, drops the first or binary category per feature
   * @param options.categories - "auto" or explicit category list per feature
   */
  constructor(
    options: {
      sparse?: boolean;
      sparseOutput?: boolean;
      handleUnknown?: "error" | "ignore";
      drop?: "first" | "if_binary" | null;
      categories?: CategoriesOption;
    } = {}
  ) {
    const sparseOption = options.sparse ?? options.sparseOutput ?? false;
    if (options.sparse !== undefined && options.sparseOutput !== undefined) {
      if (options.sparse !== options.sparseOutput) {
        throw new InvalidParameterError(
          "sparse and sparseOutput must match when both are provided",
          "sparse",
          options.sparse
        );
      }
    }
    this.sparse = sparseOption;
    this.handleUnknown = options.handleUnknown ?? "error";
    this.drop = options.drop ?? null;
    this.categoriesOption = options.categories ?? "auto";

    if (typeof this.sparse !== "boolean") {
      throw new InvalidParameterError("sparse must be a boolean", "sparse", this.sparse);
    }
    if (this.handleUnknown !== "error" && this.handleUnknown !== "ignore") {
      throw new InvalidParameterError(
        "handleUnknown must be 'error' or 'ignore'",
        "handleUnknown",
        this.handleUnknown
      );
    }
    if (this.drop !== null && this.drop !== "first" && this.drop !== "if_binary") {
      throw new InvalidParameterError(
        "drop must be 'first', 'if_binary', or null",
        "drop",
        this.drop
      );
    }
  }

  /**
   * Fit OneHotEncoder to X.
   * Learns the unique categories for each feature.
   *
   * @param X - Training data (2D tensor of categorical features)
   * @returns this - Returns self for method chaining
   * @throws {ShapeError} If X is not a 2D tensor
   * @throws {InvalidParameterError} If X is empty
   */
  fit(X: Tensor): this {
    assert2D(X, "X");
    const [nSamples, nFeatures] = getShape2D(X);

    if (nSamples === 0 || nFeatures === 0) {
      throw new InvalidParameterError("Cannot fit OneHotEncoder on empty array", "X");
    }

    // Initialize storage for categories and lookup maps
    this.categories_ = [];
    this.categoryToIndex_ = [];

    const explicitCategories = resolveCategoriesOption(
      this.categoriesOption,
      nFeatures,
      "categories"
    );

    // For each feature, collect or validate categories
    for (let j = 0; j < nFeatures; j++) {
      let cats: Category[];

      if (explicitCategories) {
        const featureCats = explicitCategories[j];
        if (!featureCats) {
          throw new InvalidParameterError("Missing categories for feature", "categories", j);
        }
        if (!Array.isArray(featureCats)) {
          throw new InvalidParameterError(
            "categories must be an array of category arrays",
            "categories",
            featureCats
          );
        }
        cats = validateCategoryValues(featureCats, "categories");
      } else {
        const uniqueSet = new Set<Category>();

        // Scan all samples to find unique values in this feature
        for (let i = 0; i < nSamples; i++) {
          uniqueSet.add(read2DValue(X, i, j));
        }

        // Sort categories for consistent ordering
        cats = sortCategories(uniqueSet, "X");
      }

      if (cats.length === 0) {
        throw new InvalidParameterError("Each feature must have at least one category", "X", j);
      }

      this.categories_.push(cats);

      // Build index map for O(1) transform lookups
      const map = new Map<Category, number>();
      for (let k = 0; k < cats.length; k++) {
        map.set(categoryValueAt(cats, k, "OneHotEncoder.fit"), k);
      }
      this.categoryToIndex_.push(map);

      // Validate training data against explicit categories
      if (explicitCategories) {
        for (let i = 0; i < nSamples; i++) {
          const val = read2DValue(X, i, j);
          if (!map.has(val)) {
            throw new InvalidParameterError(
              `Unknown category: ${String(val)} in feature ${j}`,
              "X",
              val
            );
          }
        }
      }
    }

    this.dropIndices_ = this.categories_.map((cats) => {
      if (this.drop === null) return null;
      if (this.drop === "first") return cats.length > 0 ? 0 : null;
      if (this.drop === "if_binary") return cats.length === 2 ? 0 : null;
      return null;
    });

    this.fitted = true;
    return this;
  }

  /**
   * Transform X using one-hot encoding.
   * Each categorical value is converted to a binary vector.
   *
   * @param X - Data to transform (2D tensor)
   * @returns Encoded data as dense Tensor or sparse CSRMatrix
   * @throws {NotFittedError} If encoder is not fitted
   * @throws {InvalidParameterError} If X contains unknown categories
   */
  transform(X: Tensor): Tensor | CSRMatrix {
    if (!this.fitted) {
      throw new NotFittedError("OneHotEncoder must be fitted before transform");
    }
    assert2D(X, "X");
    const [nSamples, nFeatures] = getShape2D(X);

    const categories = this.categories_;
    const categoryMaps = this.categoryToIndex_;
    if (!categories || !categoryMaps) {
      throw new DeepboxError("OneHotEncoder internal error: missing fitted state");
    }
    const fittedFeatures = categories.length;
    if (nFeatures !== fittedFeatures) {
      throw new InvalidParameterError(
        "X has a different feature count than during fit",
        "X",
        nFeatures
      );
    }

    const dropIndices = this.dropIndices_ ?? categories.map(() => null);

    // Calculate total output columns (sum of all category counts minus drops)
    let totalCols = 0;
    for (let j = 0; j < categories.length; j++) {
      const cats = categories[j];
      if (!cats) continue;
      const dropIndex = dropIndices[j] ?? null;
      totalCols += cats.length - (dropIndex === null ? 0 : 1);
    }

    if (nSamples === 0 || nFeatures === 0) {
      return this.sparse
        ? CSRMatrix.fromCOO({
            rows: 0,
            cols: totalCols,
            rowIndices: new Int32Array(0),
            colIndices: new Int32Array(0),
            values: new Float64Array(0),
          })
        : zeros([0, totalCols], { dtype: "float64" });
    }

    if (this.sparse) {
      const rowIdx: number[] = [];
      const colIdx: number[] = [];
      const vals: number[] = [];

      for (let i = 0; i < nSamples; i++) {
        let colOffset = 0;
        for (let j = 0; j < nFeatures; j++) {
          const cats = categories[j];
          const map = categoryMaps[j];
          const dropIndex = dropIndices[j] ?? null;
          if (!cats || !map) {
            throw new DeepboxError("OneHotEncoder internal error: missing fitted categories");
          }
          const outSize = cats.length - (dropIndex === null ? 0 : 1);
          const val = read2DValue(X, i, j);
          const idx = map.get(val);
          if (idx === undefined) {
            if (this.handleUnknown === "ignore") {
              colOffset += outSize;
              continue;
            }
            throw new InvalidParameterError(`Unknown category: ${String(val)}`, "X", val);
          }

          if (dropIndex !== null && idx === dropIndex) {
            colOffset += outSize;
            continue;
          }

          const adjusted = dropIndex !== null && idx > dropIndex ? idx - 1 : idx;
          rowIdx.push(i);
          colIdx.push(colOffset + adjusted);
          vals.push(1);
          colOffset += outSize;
        }
      }

      return CSRMatrix.fromCOO({
        rows: nSamples,
        cols: totalCols,
        rowIndices: Int32Array.from(rowIdx),
        colIndices: Int32Array.from(colIdx),
        values: Float64Array.from(vals),
      });
    }

    const result = Array.from({ length: nSamples }, () => new Array<number>(totalCols).fill(0));

    for (let i = 0; i < nSamples; i++) {
      let colOffset = 0;
      for (let j = 0; j < nFeatures; j++) {
        const cats = categories[j];
        const map = categoryMaps[j];
        const dropIndex = dropIndices[j] ?? null;
        if (!cats || !map) {
          throw new DeepboxError("OneHotEncoder internal error: missing fitted categories");
        }
        const outSize = cats.length - (dropIndex === null ? 0 : 1);
        const val = read2DValue(X, i, j);
        const idx = map.get(val);
        if (idx === undefined) {
          if (this.handleUnknown === "ignore") {
            colOffset += outSize;
            continue;
          }
          throw new InvalidParameterError(`Unknown category: ${String(val)}`, "X", val);
        }
        if (dropIndex !== null && idx === dropIndex) {
          colOffset += outSize;
          continue;
        }
        const row = result[i];
        if (row === undefined) {
          throw new DeepboxError("Internal error: result row access failed");
        }
        const adjusted = dropIndex !== null && idx > dropIndex ? idx - 1 : idx;
        row[colOffset + adjusted] = 1;
        colOffset += outSize;
      }
    }

    return tensor(result, { dtype: "float64", device: X.device });
  }

  fitTransform(X: Tensor): Tensor | CSRMatrix {
    return this.fit(X).transform(X);
  }

  inverseTransform(X: Tensor | CSRMatrix): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("OneHotEncoder must be fitted before inverse_transform");
    }
    const dense = X instanceof CSRMatrix ? X.toDense() : X;
    assert2D(dense, "X");
    assertNumericTensor(dense, "X");
    const [nSamples, nCols] = getShape2D(dense);
    const categories = this.categories_;
    if (!categories) {
      throw new DeepboxError("OneHotEncoder internal error: missing fitted categories");
    }
    const nFeatures = categories.length;
    const dropIndices = this.dropIndices_ ?? categories.map(() => null);
    const totalCols = categories.reduce((sum, cats, idx) => {
      const dropIndex = dropIndices[idx] ?? null;
      return sum + cats.length - (dropIndex === null ? 0 : 1);
    }, 0);
    if (nCols !== totalCols) {
      throw new InvalidParameterError("column count does not match fitted categories", "X", nCols);
    }
    if (nSamples === 0) {
      return emptyCategoryMatrixFromCategories(categories, nFeatures, "X");
    }

    const result = new Array<Category[]>(nSamples);
    for (let i = 0; i < nSamples; i++) {
      result[i] = new Array<Category>(nFeatures);
    }
    const denseData = getNumericData(dense);
    const [stride0, stride1] = getStrides2D(dense);

    for (let i = 0; i < nSamples; i++) {
      let colOffset = 0;
      for (let j = 0; j < nFeatures; j++) {
        const cats = categories[j];
        const dropIndex = dropIndices[j] ?? null;
        if (!cats) {
          throw new DeepboxError("OneHotEncoder internal error: missing fitted categories");
        }
        const outSize = cats.length - (dropIndex === null ? 0 : 1);
        if (outSize === 0) {
          const row = result[i];
          if (!row) {
            throw new DeepboxError("Internal error: result row access failed");
          }
          row[j] = categoryValueAt(cats, dropIndex ?? 0, "OneHotEncoder.inverseTransform");
          continue;
        }

        let maxIdx = 0;
        const rowBase = dense.offset + i * stride0 + colOffset * stride1;
        const first = denseData[rowBase];
        if (first === undefined) {
          throw new DeepboxError("Internal error: numeric tensor access out of bounds");
        }
        let maxVal = Number(first);
        let hasPositive = maxVal > 0;

        for (let k = 1; k < outSize; k++) {
          const raw = denseData[rowBase + k * stride1];
          if (raw === undefined) {
            throw new DeepboxError("Internal error: numeric tensor access out of bounds");
          }
          const val = Number(raw);
          if (val > maxVal) {
            maxVal = val;
            maxIdx = k;
          }
          if (val > 0) {
            hasPositive = true;
          }
        }

        const row = result[i];
        if (row === undefined) {
          throw new DeepboxError("Internal error: result row access failed");
        }
        if (!hasPositive) {
          if (dropIndex !== null) {
            row[j] = categoryValueAt(cats, dropIndex, "OneHotEncoder.inverseTransform");
          } else if (this.handleUnknown === "ignore") {
            throw new InvalidParameterError(
              "Cannot inverse-transform: sample contains no active category (all zeros). This may happen if unknown categories were ignored during transform.",
              "X"
            );
          } else {
            throw new InvalidParameterError("Invalid one-hot encoding: all zeros", "X");
          }
        } else {
          const actualIdx = dropIndex !== null && maxIdx >= dropIndex ? maxIdx + 1 : maxIdx;
          row[j] = categoryValueAt(cats, actualIdx, "OneHotEncoder.inverseTransform");
        }

        colOffset += outSize;
      }
    }

    return toCategoryMatrixTensor(result, "X");
  }
}

/**
 * Encode categorical features as integer array.
 *
 * This encoder transforms categorical features into ordinal integers.
 * Each feature's categories are mapped to integers [0, n_categories-1]
 * based on their sorted order. Unlike OneHotEncoder, this maintains
 * a single column per feature.
 *
 * **Time Complexity:**
 * - fit: O(n*m*log(k)) where n=samples, m=features, k=avg categories
 * - transform: O(n*m*log(k)) due to indexOf lookup
 *
 * **Space Complexity:** O(m*k) where m=features, k=avg categories per feature
 *
 * @example
 * ```js
 * const X = tensor([['low', 'red'], ['high', 'blue'], ['medium', 'red']]);
 * const encoder = new OrdinalEncoder();
 * encoder.fit(X);
 * const encoded = encoder.transform(X);
 * // Result: [[1, 1], [0, 0], [2, 1]] (alphabetically sorted)
 * ```
 *
 * @see {@link https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html | Scikit-learn OrdinalEncoder}
 */
export class OrdinalEncoder {
  /** Indicates whether the encoder has been fitted to data */
  private fitted = false;
  /** Array of unique categories for each feature, sorted */
  private categories_?: Category[][];
  /** Maps from category value to index for each feature (for O(1) lookup) */
  private categoryToIndex_?: Array<Map<Category, number>>;
  /** How to handle unknown categories during transform */
  private handleUnknown: "error" | "useEncodedValue";
  /** Value used for unknown categories when handleUnknown = "useEncodedValue" */
  private unknownValue: number;
  /** Categories configuration */
  private categoriesOption: CategoriesOption;

  /**
   * Creates a new OrdinalEncoder instance.
   *
   * @param options - Configuration options
   * @param options.handleUnknown - How to handle unknown categories
   * @param options.unknownValue - Encoded value for unknown categories when handleUnknown="useEncodedValue"
   * @param options.categories - "auto" or explicit categories per feature
   */
  constructor(
    options: {
      handleUnknown?: "error" | "useEncodedValue";
      unknownValue?: number;
      categories?: CategoriesOption;
    } = {}
  ) {
    this.handleUnknown = options.handleUnknown ?? "error";
    this.unknownValue = options.unknownValue ?? -1;
    this.categoriesOption = options.categories ?? "auto";

    if (this.handleUnknown !== "error" && this.handleUnknown !== "useEncodedValue") {
      throw new InvalidParameterError(
        "handleUnknown must be 'error' or 'useEncodedValue'",
        "handleUnknown",
        this.handleUnknown
      );
    }
    if (!Number.isFinite(this.unknownValue) && !Number.isNaN(this.unknownValue)) {
      throw new InvalidParameterError(
        "unknownValue must be a finite number or NaN",
        "unknownValue",
        this.unknownValue
      );
    }
    if (Number.isFinite(this.unknownValue) && !Number.isInteger(this.unknownValue)) {
      throw new InvalidParameterError(
        "unknownValue must be an integer when finite",
        "unknownValue",
        this.unknownValue
      );
    }
  }

  /**
   * Fit OrdinalEncoder to X.
   * Learns the unique categories for each feature and their ordering.
   *
   * @param X - Training data (2D tensor of categorical features)
   * @returns this - Returns self for method chaining
   * @throws {InvalidParameterError} If X is empty
   */
  fit(X: Tensor): this {
    assert2D(X, "X");
    const [nSamples, nFeatures] = getShape2D(X);

    if (nSamples === 0) {
      throw new InvalidParameterError("Cannot fit OrdinalEncoder on empty array", "X");
    }

    this.categories_ = [];
    this.categoryToIndex_ = [];

    const explicitCategories = resolveCategoriesOption(
      this.categoriesOption,
      nFeatures,
      "categories"
    );

    // For each feature, collect and sort unique categories
    for (let j = 0; j < nFeatures; j++) {
      let sorted: Category[];

      if (explicitCategories) {
        const featureCats = explicitCategories[j];
        if (!featureCats) {
          throw new InvalidParameterError("Missing categories for feature", "categories", j);
        }
        if (!Array.isArray(featureCats)) {
          throw new InvalidParameterError(
            "categories must be an array of category arrays",
            "categories",
            featureCats
          );
        }
        sorted = validateCategoryValues(featureCats, "categories");
      } else {
        const uniqueSet = new Set<Category>();

        // Collect all unique values in this feature
        for (let i = 0; i < nSamples; i++) {
          uniqueSet.add(read2DValue(X, i, j));
        }

        // Sort categories for consistent ordering
        sorted = sortCategories(uniqueSet, "X");
      }

      if (sorted.length === 0) {
        throw new InvalidParameterError("Each feature must have at least one category", "X", j);
      }

      this.categories_.push(sorted);

      // Build index map for O(1) transform lookups
      const map = new Map<Category, number>();
      for (let k = 0; k < sorted.length; k++) {
        map.set(categoryValueAt(sorted, k, "OrdinalEncoder.fit"), k);
      }
      this.categoryToIndex_.push(map);

      if (explicitCategories) {
        for (let i = 0; i < nSamples; i++) {
          const val = read2DValue(X, i, j);
          if (!map.has(val)) {
            throw new InvalidParameterError(
              `Unknown category: ${String(val)} in feature ${j}`,
              "X",
              val
            );
          }
        }
      }

      if (this.handleUnknown === "useEncodedValue") {
        if (
          Number.isFinite(this.unknownValue) &&
          this.unknownValue >= 0 &&
          this.unknownValue < sorted.length
        ) {
          throw new InvalidParameterError(
            "unknownValue must be outside the range of encoded categories",
            "unknownValue",
            this.unknownValue
          );
        }
      }
    }

    this.fitted = true;
    return this;
  }

  /**
   * Transform X using ordinal encoding.
   * Each category is mapped to its index in the sorted categories array.
   *
   * @param X - Data to transform (2D tensor)
   * @returns Encoded data with integer values
   * @throws {NotFittedError} If encoder is not fitted
   * @throws {InvalidParameterError} If X contains unknown categories
   */
  transform(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("OrdinalEncoder must be fitted before transform");
    }
    assert2D(X, "X");
    const [nSamples, nFeatures] = getShape2D(X);
    const fittedFeatures = this.categories_?.length ?? 0;
    if (nFeatures !== fittedFeatures) {
      throw new InvalidParameterError(
        "X has a different feature count than during fit",
        "X",
        nFeatures
      );
    }

    if (nSamples === 0) {
      return zeros([0, nFeatures], { dtype: "float64" });
    }

    // Pre-allocate result array
    const result = new Array<number[]>(nSamples);
    for (let i = 0; i < nSamples; i++) {
      result[i] = new Array<number>(nFeatures);
    }

    // Transform each value to its ordinal index using O(1) map lookup
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) {
        const val = read2DValue(X, i, j);
        const map = this.categoryToIndex_?.[j];
        if (!map) {
          throw new DeepboxError("OrdinalEncoder internal error: missing fitted categories");
        }

        // Use O(1) map lookup instead of O(n) indexOf
        const idx = map.get(val);
        const row = result[i];
        if (!row) {
          throw new DeepboxError("Internal error: result row access failed");
        }
        if (idx === undefined) {
          if (this.handleUnknown === "useEncodedValue") {
            row[j] = this.unknownValue;
            continue;
          }
          throw new InvalidParameterError(
            `Unknown category: ${String(val)} in feature ${j}`,
            "X",
            val
          );
        }

        row[j] = idx;
      }
    }

    return tensor(result, { dtype: "float64" });
  }

  /**
   * Fit encoder and transform X in one step.
   * Convenience method equivalent to calling fit(X).transform(X).
   *
   * @param X - Training data (2D tensor)
   * @returns Encoded data
   */
  fitTransform(X: Tensor): Tensor {
    return this.fit(X).transform(X);
  }

  /**
   * Transform ordinal integers back to original categories.
   * Reverses the encoding performed by transform().
   *
   * @param X - Encoded data (2D integer tensor)
   * @returns Original categorical data
   * @throws {NotFittedError} If encoder is not fitted
   * @throws {InvalidParameterError} If X contains invalid indices
   */
  inverseTransform(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("OrdinalEncoder must be fitted before inverse_transform");
    }
    assert2D(X, "X");
    assertNumericTensor(X, "X");
    const [nSamples, nFeatures] = getShape2D(X);
    const fittedFeatures = this.categories_?.length ?? 0;
    if (nFeatures !== fittedFeatures) {
      throw new InvalidParameterError(
        "X has a different feature count than during fit",
        "X",
        nFeatures
      );
    }

    if (nSamples === 0 || nFeatures === 0) {
      const categoryRows = this.categories_ ?? [];
      const categoryType = inferCategoryTypeFromRows(categoryRows, "X");
      if (categoryType === "string") {
        return empty([0, nFeatures], { dtype: "string" });
      }
      if (categoryType === "bigint") {
        return empty([0, nFeatures], { dtype: "int64" });
      }
      return zeros([0, nFeatures], { dtype: "float64" });
    }

    // Pre-allocate result array
    const result = new Array<Category[]>(nSamples);
    for (let i = 0; i < nSamples; i++) {
      result[i] = new Array<Category>(nFeatures);
    }

    // Map each ordinal index back to its original category
    const [stride0, stride1] = getStrides2D(X);
    const data = getNumericData(X);
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) {
        const raw = data[X.offset + i * stride0 + j * stride1];
        if (raw === undefined) {
          throw new DeepboxError("Internal error: numeric tensor access out of bounds");
        }
        const idx = Number(raw);
        const isUnknownValue =
          this.handleUnknown === "useEncodedValue" &&
          (Number.isNaN(idx) ? Number.isNaN(this.unknownValue) : idx === this.unknownValue);
        if (isUnknownValue) {
          throw new InvalidParameterError(
            "Cannot inverse-transform unknown encoded value",
            "X",
            idx
          );
        }
        const cats = this.categories_?.[j];

        // Validate index is in valid range
        if (!cats || idx < 0 || idx >= cats.length || !Number.isInteger(idx)) {
          throw new InvalidParameterError(
            `Invalid encoded value: ${idx} for feature ${j}. Must be integer in [0, ${(cats?.length ?? 0) - 1}]`,
            "X",
            idx
          );
        }

        const row = result[i];
        if (!row) {
          throw new DeepboxError("Internal error: result row access failed");
        }
        const catVal = cats[idx];
        if (catVal === undefined) {
          throw new DeepboxError("Internal error: category value missing");
        }
        row[j] = catVal;
      }
    }

    return toCategoryMatrixTensor(result, "X");
  }
}

/**
 * Binarize labels in a one-vs-all fashion.
 *
 * This transformer creates a binary matrix representation of labels where
 * each class gets its own column. For multi-class problems, this creates
 * a one-hot encoding of the labels.
 *
 * **Time Complexity:**
 * - fit: O(n) where n is the number of samples
 * - transform: O(n*k) where k is the number of classes
 *
 * **Space Complexity:** O(n*k) for the output matrix
 *
 * @example
 * ```js
 * const y = tensor([0, 1, 2, 0, 1]);
 * const binarizer = new LabelBinarizer();
 * const yBin = binarizer.fitTransform(y);
 * // Result shape: [5, 3] with one-hot encoding
 * ```
 *
 * @see {@link https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html | Scikit-learn LabelBinarizer}
 */
export class LabelBinarizer {
  /** Indicates whether the binarizer has been fitted to data */
  private fitted = false;
  /** Array of unique classes found during fitting, sorted */
  private classes_?: Category[];
  /** Map from class value to index for O(1) lookups */
  private classToIndex_?: Map<Category, number>;
  /** Value used for positive class */
  private posLabel: number;
  /** Value used for negative class */
  private negLabel: number;
  /** Whether to return sparse matrix output */
  private sparse: boolean;

  /**
   * Creates a new LabelBinarizer instance.
   *
   * @param options - Configuration options
   * @param options.posLabel - Value for positive class (default: 1)
   * @param options.negLabel - Value for negative class (default: 0)
   * @param options.sparse - If true, returns CSRMatrix (default: false)
   * @param options.sparseOutput - Alias for sparse (default: false)
   */
  constructor(
    options: {
      posLabel?: number;
      negLabel?: number;
      sparse?: boolean;
      sparseOutput?: boolean;
    } = {}
  ) {
    this.posLabel = options.posLabel ?? 1;
    this.negLabel = options.negLabel ?? 0;
    const sparseOption = options.sparse ?? options.sparseOutput ?? false;
    if (!Number.isFinite(this.posLabel) || !Number.isFinite(this.negLabel)) {
      throw new InvalidParameterError("posLabel and negLabel must be finite numbers", "posLabel");
    }
    if (this.posLabel <= this.negLabel) {
      throw new InvalidParameterError(
        "posLabel must be greater than negLabel",
        "posLabel",
        this.posLabel
      );
    }
    if (options.sparse !== undefined && options.sparseOutput !== undefined) {
      if (options.sparse !== options.sparseOutput) {
        throw new InvalidParameterError(
          "sparse and sparseOutput must match when both are provided",
          "sparse",
          options.sparse
        );
      }
    }
    if (typeof sparseOption !== "boolean") {
      throw new InvalidParameterError("sparse must be a boolean", "sparse", sparseOption);
    }
    if (sparseOption && this.negLabel !== 0) {
      throw new InvalidParameterError(
        "sparse output requires negLabel to be 0",
        "negLabel",
        this.negLabel
      );
    }
    this.sparse = sparseOption;
  }

  /**
   * Fit label binarizer to a set of labels.
   * Learns the unique classes present in the data.
   *
   * @param y - Target labels (1D tensor)
   * @returns this - Returns self for method chaining
   * @throws {InvalidParameterError} If y is empty
   */
  fit(y: Tensor): this {
    assert1D(y, "y");
    if (y.size === 0) {
      throw new InvalidParameterError("Cannot fit LabelBinarizer on empty array", "y");
    }

    // Collect unique classes
    const uniqueSet = new Set<Category>();
    for (let i = 0; i < y.size; i++) {
      uniqueSet.add(read1DValue(y, i));
    }

    // Sort classes for consistent ordering
    this.classes_ = sortCategories(uniqueSet, "y");
    this.classToIndex_ = new Map();
    for (let i = 0; i < this.classes_.length; i++) {
      this.classToIndex_.set(categoryValueAt(this.classes_, i, "LabelBinarizer.fit"), i);
    }

    this.fitted = true;
    return this;
  }

  /**
   * Transform labels to binary matrix.
   * Each label is converted to a binary vector with a single 1.
   *
   * @param y - Labels to transform (1D tensor)
   * @returns Binary matrix (Tensor or CSRMatrix) with shape [n_samples, n_classes]
   * @throws {NotFittedError} If binarizer is not fitted
   * @throws {InvalidParameterError} If y contains unknown labels
   */
  transform(y: Tensor): Tensor | CSRMatrix {
    if (!this.fitted) {
      throw new NotFittedError("LabelBinarizer must be fitted before transform");
    }
    assert1D(y, "y");
    if (y.size === 0) {
      const nClasses = this.classes_?.length ?? 0;
      return this.sparse
        ? CSRMatrix.fromCOO({
            rows: 0,
            cols: nClasses,
            rowIndices: new Int32Array(0),
            colIndices: new Int32Array(0),
            values: new Float64Array(0),
          })
        : zeros([0, nClasses], { dtype: "float64" });
    }

    const nSamples = y.size;
    const nClasses = this.classes_?.length ?? 0;
    const lookup = this.classToIndex_;
    if (!lookup) {
      throw new DeepboxError("LabelBinarizer internal error: missing fitted lookup");
    }

    if (this.sparse) {
      const rowIdx: number[] = [];
      const colIdx: number[] = [];
      const vals: number[] = [];

      for (let i = 0; i < nSamples; i++) {
        const val = read1DValue(y, i);
        const idx = lookup.get(val);

        if (idx === undefined) {
          throw new InvalidParameterError(
            `Unknown label: ${String(val)}. Label must be present during fit.`,
            "y",
            val
          );
        }

        rowIdx.push(i);
        colIdx.push(idx);
        vals.push(this.posLabel);
      }

      return CSRMatrix.fromCOO({
        rows: nSamples,
        cols: nClasses,
        rowIndices: Int32Array.from(rowIdx),
        colIndices: Int32Array.from(colIdx),
        values: Float64Array.from(vals),
      });
    }

    // Pre-allocate binary matrix
    const result = new Array<number[]>(nSamples);
    for (let i = 0; i < nSamples; i++) {
      result[i] = new Array<number>(nClasses).fill(this.negLabel);
    }

    // Set appropriate bit for each label
    for (let i = 0; i < nSamples; i++) {
      const val = read1DValue(y, i);
      const idx = lookup.get(val);

      if (idx === undefined) {
        throw new InvalidParameterError(
          `Unknown label: ${String(val)}. Label must be present during fit.`,
          "y",
          val
        );
      }

      const row = result[i];
      if (!row) {
        throw new DeepboxError("Internal error: result row access failed");
      }
      row[idx] = this.posLabel;
    }

    return tensor(result, { dtype: "float64" });
  }

  /**
   * Fit binarizer and transform labels in one step.
   * Convenience method equivalent to calling fit(y).transform(y).
   *
   * @param y - Target labels (1D tensor)
   * @returns Binary matrix (Tensor or CSRMatrix)
   */
  fitTransform(y: Tensor): Tensor | CSRMatrix {
    return this.fit(y).transform(y);
  }

  /**
   * Transform binary matrix back to labels.
   * Finds the column with maximum value for each row.
   *
   * @param Y - Binary matrix (2D tensor or CSRMatrix)
   * @returns Original labels (1D tensor)
   * @throws {NotFittedError} If binarizer is not fitted
   * @throws {InvalidParameterError} If Y has invalid shape
   */
  inverseTransform(Y: Tensor | CSRMatrix): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("LabelBinarizer must be fitted before inverse_transform");
    }
    if (Y instanceof CSRMatrix) {
      if (this.negLabel !== 0) {
        throw new InvalidParameterError(
          "Sparse inverse transform requires negLabel to be 0",
          "negLabel",
          this.negLabel
        );
      }
      const [rows, cols] = Y.shape;
      if (rows === undefined || cols === undefined) {
        throw new ShapeError("Y must have valid shape");
      }
      const nClasses = this.classes_?.length ?? 0;
      if (cols !== nClasses) {
        throw new InvalidParameterError("column count does not match number of classes", "Y", cols);
      }
      const classes = this.classes_;
      if (!classes) {
        throw new DeepboxError("LabelBinarizer internal error: missing fitted classes");
      }
      if (rows === 0) {
        return emptyCategoryVectorFromClasses(classes, "y");
      }

      const result = new Array<Category>(rows);
      for (let i = 0; i < rows; i++) {
        let maxIdx = 0;
        let maxVal = this.negLabel;
        const start = Y.indptr[i] ?? 0;
        const end = Y.indptr[i + 1] ?? start;
        for (let p = start; p < end; p++) {
          const col = Y.indices[p];
          if (col === undefined) {
            throw new DeepboxError("Internal error: sparse column index missing");
          }
          if (col < 0 || col >= nClasses) {
            throw new InvalidParameterError(
              "column index out of bounds for fitted classes",
              "Y",
              col
            );
          }
          const raw = Y.data[p];
          if (raw === undefined) {
            throw new DeepboxError("Internal error: sparse value missing");
          }
          const val = Number(raw);
          if (val > maxVal) {
            maxVal = val;
            maxIdx = col;
          }
        }

        if (maxVal <= this.negLabel) {
          throw new InvalidParameterError(
            `No active label found for sample ${i}. LabelBinarizer expects exactly one active label.`,
            "Y"
          );
        }
        result[i] = categoryValueAt(classes, maxIdx, "LabelBinarizer.inverseTransform");
      }
      return toCategoryVectorTensor(result, "y");
    }

    assert2D(Y, "Y");
    assertNumericTensor(Y, "Y");
    const [nSamples, nCols] = getShape2D(Y);

    const nClasses = this.classes_?.length ?? 0;
    if (nCols !== nClasses) {
      throw new InvalidParameterError("column count does not match number of classes", "Y", nCols);
    }
    const classes = this.classes_;
    if (!classes) {
      throw new DeepboxError("LabelBinarizer internal error: missing fitted classes");
    }
    if (nSamples === 0) {
      return emptyCategoryVectorFromClasses(classes, "y");
    }
    const result = new Array<Category>(nSamples);
    const [stride0, stride1] = getStrides2D(Y);
    const data = getNumericData(Y);

    // For each sample, find the class with maximum activation
    for (let i = 0; i < nSamples; i++) {
      let maxIdx = 0;
      const rowBase = Y.offset + i * stride0;
      const first = data[rowBase];
      if (first === undefined) {
        throw new DeepboxError("Internal error: numeric tensor access out of bounds");
      }
      let maxVal = Number(first);

      // Find column with highest value
      for (let j = 1; j < nCols; j++) {
        const raw = data[rowBase + j * stride1];
        if (raw === undefined) {
          throw new DeepboxError("Internal error: numeric tensor access out of bounds");
        }
        const val = Number(raw);
        if (val > maxVal) {
          maxVal = val;
          maxIdx = j;
        }
      }

      if (maxVal <= this.negLabel) {
        throw new InvalidParameterError(
          `No active label found for sample ${i}. LabelBinarizer expects exactly one active label.`,
          "Y"
        );
      }

      result[i] = categoryValueAt(classes, maxIdx, "LabelBinarizer.inverseTransform");
    }

    // Return tensor with appropriate dtype
    return toCategoryVectorTensor(result, "y");
  }
}

/**
 * Transform multi-label classification data to binary format.
 *
 * This transformer handles multi-label classification where each sample
 * can belong to multiple classes simultaneously. It creates a binary
 * matrix where each column represents a class and multiple columns can
 * be active (set to 1) for a single sample.
 *
 * **Time Complexity:**
 * - fit: O(n*k) where n is samples, k is avg labels per sample
 * - transform: O(n*k*c) where c is total unique classes
 *
 * **Space Complexity:** O(n*c) for the output matrix
 *
 * @example
 * ```js
 * const y = [['sci-fi', 'action'], ['comedy'], ['action', 'drama']];
 * const binarizer = new MultiLabelBinarizer();
 * const yBin = binarizer.fitTransform(y);
 * // Each row can have multiple 1s
 * ```
 *
 * @see {@link https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html | Scikit-learn MultiLabelBinarizer}
 */
export class MultiLabelBinarizer {
  /** Indicates whether the binarizer has been fitted to data */
  private fitted = false;
  /** Array of all unique classes found across all samples, sorted */
  private classes_?: Category[];
  /** Map from class value to index for O(1) lookups */
  private classToIndex_?: Map<Category, number>;
  /** Whether to return sparse matrix (CSR) or dense array */
  private sparse: boolean;
  /** Optional explicit class ordering */
  private classesOption?: Category[];

  /**
   * Creates a new MultiLabelBinarizer instance.
   *
   * @param options - Configuration options
   * @param options.sparse - If true, returns CSRMatrix; if false, returns dense Tensor (default: false)
   * @param options.sparseOutput - Alias for sparse (default: false)
   * @param options.classes - Explicit class ordering to use instead of sorting
   */
  constructor(
    options: {
      sparse?: boolean;
      sparseOutput?: boolean;
      classes?: ReadonlyArray<Category>;
    } = {}
  ) {
    const sparseOption = options.sparse ?? options.sparseOutput ?? false;
    if (options.sparse !== undefined && options.sparseOutput !== undefined) {
      if (options.sparse !== options.sparseOutput) {
        throw new InvalidParameterError(
          "sparse and sparseOutput must match when both are provided",
          "sparse",
          options.sparse
        );
      }
    }
    this.sparse = sparseOption;
    if (typeof this.sparse !== "boolean") {
      throw new InvalidParameterError("sparse must be a boolean", "sparse", this.sparse);
    }
    if (options.classes !== undefined) {
      this.classesOption = validateCategoryValues(options.classes, "classes");
    }
  }

  /**
   * Fit multi-label binarizer to label sets.
   * Learns all unique classes present across all samples.
   *
   * @param y - Array of label sets, where each element is an array of string/number/bigint labels
   * @returns this - Returns self for method chaining
   * @throws {InvalidParameterError} If y is empty
   */
  fit(y: ReadonlyArray<ReadonlyArray<Category>>): this {
    if (y.length === 0) {
      throw new InvalidParameterError("Cannot fit MultiLabelBinarizer on empty array", "y");
    }
    for (const labels of y) {
      if (!Array.isArray(labels)) {
        throw new InvalidParameterError("MultiLabelBinarizer expects label arrays", "y", labels);
      }
      for (const label of labels) {
        if (typeof label !== "string" && typeof label !== "number" && typeof label !== "bigint") {
          throw new InvalidParameterError(
            "MultiLabelBinarizer labels must be strings, numbers, or bigints",
            "y",
            label
          );
        }
      }
    }

    if (this.classesOption && this.classesOption.length === 0) {
      throw new InvalidParameterError("classes must contain at least one value", "classes");
    }

    if (this.classesOption) {
      this.classes_ = Array.from(this.classesOption);
    } else {
      // Collect all unique labels across all samples
      const uniqueSet = new Set<Category>();
      for (const labels of y) {
        for (const label of labels) {
          uniqueSet.add(label);
        }
      }

      // Sort classes for consistent ordering
      this.classes_ = sortCategories(uniqueSet, "y");
    }
    this.classToIndex_ = new Map();
    for (let i = 0; i < this.classes_.length; i++) {
      this.classToIndex_.set(categoryValueAt(this.classes_, i, "MultiLabelBinarizer.fit"), i);
    }
    if (this.classesOption) {
      for (const labels of y) {
        for (const label of labels) {
          if (!this.classToIndex_.has(label)) {
            throw new InvalidParameterError(
              `Unknown label: ${String(label)}. Label must be present in classes.`,
              "y",
              label
            );
          }
        }
      }
    }

    this.fitted = true;
    return this;
  }

  /**
   * Transform label sets to binary matrix.
   * Each sample can have multiple active (1) columns.
   *
   * @param y - Array of label sets to transform (string/number/bigint labels)
   * @returns Binary matrix (Tensor or CSRMatrix) with shape [n_samples, n_classes]
   * @throws {NotFittedError} If binarizer is not fitted
   * @throws {InvalidParameterError} If y contains unknown labels
   */
  transform(y: ReadonlyArray<ReadonlyArray<Category>>): Tensor | CSRMatrix {
    if (!this.fitted) {
      throw new NotFittedError("MultiLabelBinarizer must be fitted before transform");
    }
    for (const labels of y) {
      if (!Array.isArray(labels)) {
        throw new InvalidParameterError("MultiLabelBinarizer expects label arrays", "y", labels);
      }
      for (const label of labels) {
        if (typeof label !== "string" && typeof label !== "number" && typeof label !== "bigint") {
          throw new InvalidParameterError(
            "MultiLabelBinarizer labels must be strings, numbers, or bigints",
            "y",
            label
          );
        }
      }
    }
    if (y.length === 0) {
      const nClasses = this.classes_?.length ?? 0;
      return this.sparse
        ? CSRMatrix.fromCOO({
            rows: 0,
            cols: nClasses,
            rowIndices: new Int32Array(0),
            colIndices: new Int32Array(0),
            values: new Float64Array(0),
          })
        : zeros([0, nClasses], { dtype: "float64" });
    }

    const nSamples = y.length;
    const nClasses = this.classes_?.length ?? 0;
    const lookup = this.classToIndex_;
    if (!lookup) {
      throw new DeepboxError("MultiLabelBinarizer internal error: missing fitted lookup");
    }

    if (this.sparse) {
      const rowIdx: number[] = [];
      const colIdx: number[] = [];
      const vals: number[] = [];

      for (let i = 0; i < nSamples; i++) {
        const yRow = y[i];
        if (!yRow) continue;
        const seen = new Set<number>();
        for (const label of yRow) {
          const idx = lookup.get(label);
          if (idx === undefined) {
            throw new InvalidParameterError(
              `Unknown label: ${String(label)}. Label must be present during fit.`,
              "y",
              label
            );
          }
          if (seen.has(idx)) continue;
          seen.add(idx);
          rowIdx.push(i);
          colIdx.push(idx);
          vals.push(1);
        }
      }

      return CSRMatrix.fromCOO({
        rows: nSamples,
        cols: nClasses,
        rowIndices: Int32Array.from(rowIdx),
        colIndices: Int32Array.from(colIdx),
        values: Float64Array.from(vals),
      });
    }

    // Pre-allocate binary matrix
    const result = new Array<number[]>(nSamples);
    for (let i = 0; i < nSamples; i++) {
      result[i] = new Array<number>(nClasses).fill(0);
    }

    // Set bits for all labels in each sample
    for (let i = 0; i < nSamples; i++) {
      const yRow = y[i];
      if (!yRow) continue;

      for (const label of yRow) {
        const idx = lookup.get(label);
        if (idx === undefined) {
          throw new InvalidParameterError(
            `Unknown label: ${String(label)}. Label must be present during fit.`,
            "y",
            label
          );
        }

        const row = result[i];
        if (!row) {
          throw new DeepboxError("Internal error: result row access failed");
        }
        row[idx] = 1;
      }
    }

    return tensor(result, { dtype: "float64" });
  }

  /**
   * Fit binarizer and transform label sets in one step.
   * Convenience method equivalent to calling fit(y).transform(y).
   *
   * @param y - Array of label sets (string/number/bigint labels)
   * @returns Binary matrix (Tensor or CSRMatrix)
   */
  fitTransform(y: ReadonlyArray<ReadonlyArray<Category>>): Tensor | CSRMatrix {
    return this.fit(y).transform(y);
  }

  /**
   * Transform binary matrix back to label sets.
   * Finds all active (1) columns for each row.
   *
   * @param Y - Binary matrix (Tensor or CSRMatrix)
   * @returns Array of label sets, one per sample
   * @throws {NotFittedError} If binarizer is not fitted
   * @throws {InvalidParameterError} If Y has invalid shape
   */
  inverseTransform(Y: Tensor | CSRMatrix): Category[][] {
    if (!this.fitted) {
      throw new NotFittedError("MultiLabelBinarizer must be fitted before inverse_transform");
    }
    if (Y instanceof CSRMatrix) {
      const [rows, cols] = Y.shape;
      if (rows === undefined || cols === undefined) {
        throw new ShapeError("Y must have valid shape");
      }
      const fittedClasses = this.classes_?.length ?? 0;
      if (cols !== fittedClasses) {
        throw new InvalidParameterError("column count does not match number of classes", "Y", cols);
      }
      if (rows === 0) {
        return [];
      }

      const classes = this.classes_;
      if (!classes) {
        throw new DeepboxError("MultiLabelBinarizer internal error: missing fitted classes");
      }

      const result: Category[][] = [];
      for (let i = 0; i < rows; i++) {
        const labels: Category[] = [];
        const start = Y.indptr[i] ?? 0;
        const end = Y.indptr[i + 1] ?? start;
        for (let p = start; p < end; p++) {
          const col = Y.indices[p];
          if (col === undefined) {
            throw new DeepboxError("Internal error: sparse column index missing");
          }
          if (col < 0 || col >= fittedClasses) {
            throw new InvalidParameterError(
              "column index out of bounds for fitted classes",
              "Y",
              col
            );
          }
          const raw = Y.data[p];
          if (raw === undefined) {
            throw new DeepboxError("Internal error: sparse value missing");
          }
          const value = Number(raw);
          if (value > 0) {
            labels.push(categoryValueAt(classes, col, "MultiLabelBinarizer.inverseTransform"));
          }
        }
        result.push(labels);
      }
      return result;
    }

    assert2D(Y, "Y");
    assertNumericTensor(Y, "Y");
    const nSamples = Y.shape[0] ?? 0;
    const nClasses = Y.shape[1] ?? 0;
    const fittedClasses = this.classes_?.length ?? 0;
    if (nClasses !== fittedClasses) {
      throw new InvalidParameterError(
        "column count does not match number of classes",
        "Y",
        nClasses
      );
    }

    if (nSamples === 0) {
      return [];
    }

    const classes = this.classes_;
    if (!classes) {
      throw new DeepboxError("MultiLabelBinarizer internal error: missing fitted classes");
    }

    const result: Category[][] = [];
    const [stride0, stride1] = getStrides2D(Y);
    const data = getNumericData(Y);

    // For each sample, collect all active classes
    for (let i = 0; i < nSamples; i++) {
      const labels: Category[] = [];
      const rowBase = Y.offset + i * stride0;

      // Check each class column
      for (let j = 0; j < nClasses; j++) {
        const raw = data[rowBase + j * stride1];
        if (raw === undefined) {
          throw new DeepboxError("Internal error: numeric tensor access out of bounds");
        }
        const val = Number(raw);
        // If this class is active (typically 1, but allow any positive value)
        if (val > 0) {
          labels.push(categoryValueAt(classes, j, "MultiLabelBinarizer.inverseTransform"));
        }
      }

      result.push(labels);
    }

    return result;
  }
}
