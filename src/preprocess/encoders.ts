import {
  DataValidationError,
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
 * Input type accepted by 1D encoder methods (LabelEncoder, LabelBinarizer).
 * Accepts a Tensor directly or a plain JavaScript array of strings, numbers, or booleans.
 */
type EncoderInput1D = Tensor | readonly (string | number | bigint | boolean)[];

/**
 * Input type accepted by 2D encoder methods (OneHotEncoder, OrdinalEncoder).
 * Accepts a Tensor directly or a plain JavaScript array of arrays.
 */
type EncoderInput2D = Tensor | readonly (readonly (string | number | bigint)[])[];

type RawValue = string | number | bigint | boolean;

function isTensorLike(input: unknown): input is Tensor {
  return typeof input === "object" && input !== null && "shape" in input && "dtype" in input;
}

/**
 * Decide the element type of a plain array of values.
 *
 * Booleans count as numbers (true -> 1, false -> 0). When strings are mixed
 * with other values, everything is treated as a string (the other values are
 * converted with `String`), so a table such as `[["red", 1], ["blue", 2]]`
 * works. Numbers and bigints cannot be mixed, because that would silently
 * lose precision or change the category type.
 */
function inferRawKind(
  values: Iterable<unknown>,
  paramName: string
): "string" | "number" | "bigint" | null {
  let hasString = false;
  let hasNumber = false;
  let hasBigInt = false;
  for (const v of values) {
    if (typeof v === "string") hasString = true;
    else if (typeof v === "number" || typeof v === "boolean") hasNumber = true;
    else if (typeof v === "bigint") hasBigInt = true;
    else {
      throw new InvalidParameterError(
        `${paramName} values must be strings, numbers, bigints or booleans`,
        paramName,
        v
      );
    }
  }
  if (hasString) return "string";
  if (hasNumber && hasBigInt) {
    throw new InvalidParameterError(`${paramName} must not mix numbers and bigints`, paramName);
  }
  if (hasBigInt) return "bigint";
  if (hasNumber) return "number";
  return null;
}

/**
 * Coerce a plain 1D array to a Tensor. If already a Tensor, return as-is.
 */
function coerceToTensor1D(input: EncoderInput1D, paramName = "y"): Tensor {
  if (isTensorLike(input)) {
    return input;
  }
  if (!Array.isArray(input)) {
    throw new InvalidParameterError(
      `${paramName} must be a Tensor or an array of values`,
      paramName,
      input
    );
  }
  const arr = input as readonly RawValue[];
  if (arr.length === 0) {
    return tensor([]);
  }
  const kind = inferRawKind(arr, paramName);
  if (kind === "string") {
    return tensor(arr.map((v) => String(v)));
  }
  if (kind === "bigint") {
    const data = new BigInt64Array(arr.length);
    for (let i = 0; i < arr.length; i++) data[i] = arr[i] as bigint;
    return tensor(data);
  }
  const numArr = new Float64Array(arr.length);
  for (let i = 0; i < arr.length; i++) numArr[i] = Number(arr[i]);
  return tensor(numArr);
}

/**
 * Coerce a plain 2D array to a Tensor. If already a Tensor, return as-is.
 */
function coerceToTensor2D(input: EncoderInput2D, paramName = "X"): Tensor {
  if (isTensorLike(input)) {
    return input;
  }
  if (!Array.isArray(input)) {
    throw new InvalidParameterError(
      `${paramName} must be a Tensor or an array of arrays`,
      paramName,
      input
    );
  }
  const arr = input as readonly (readonly RawValue[])[];
  if (arr.length === 0 || (Array.isArray(arr[0]) && arr[0].length === 0)) {
    return tensor([[]]);
  }
  if (!Array.isArray(arr[0])) {
    throw new ShapeError(`${paramName} must be a 2D array (an array of rows)`);
  }
  const nCols = (arr[0] as readonly RawValue[]).length;
  const flat: RawValue[] = [];
  for (const row of arr) {
    if (!Array.isArray(row) || row.length !== nCols) {
      throw new ShapeError(`${paramName} rows must all have the same length (${nCols})`);
    }
    for (const v of row) flat.push(v);
  }
  const kind = inferRawKind(flat, paramName);
  const shape: [number, number] = [arr.length, nCols];
  if (kind === "string") {
    return TensorImpl.fromStringArray({
      data: flat.map((v) => String(v)),
      shape,
      device: getConfig().defaultDevice,
    });
  }
  if (kind === "bigint") {
    const data = new BigInt64Array(flat.length);
    for (let i = 0; i < flat.length; i++) data[i] = flat[i] as bigint;
    return TensorImpl.fromTypedArray({
      data,
      shape,
      dtype: "int64",
      device: getConfig().defaultDevice,
    });
  }
  const data = new Float64Array(flat.length);
  for (let i = 0; i < flat.length; i++) data[i] = Number(flat[i]);
  return TensorImpl.fromTypedArray({
    data,
    shape,
    dtype: "float64",
    device: getConfig().defaultDevice,
  });
}

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
  if (t.dtype === "complex64" || t.dtype === "complex128") {
    throw new DTypeError(`Complex tensors are not supported; received dtype ${t.dtype}`);
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

/**
 * Order two strings by Unicode code point, the order NumPy and scikit-learn
 * use for string categories. `localeCompare` is locale dependent and puts
 * "a" before "B", which would make class order differ between machines.
 */
function compareCodePoints(a: string, b: string): number {
  if (a === b) return 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    let ca = a.charCodeAt(i);
    let cb = b.charCodeAt(i);
    if (ca !== cb) {
      // UTF-16 surrogates (astral code points) must sort above U+E000..U+FFFF.
      if (ca >= 0xd800 && cb >= 0xd800) {
        ca += ca >= 0xe000 ? -0x800 : 0x2000;
        cb += cb >= 0xe000 ? -0x800 : 0x2000;
      }
      return ca - cb;
    }
  }
  return a.length - b.length;
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
      return compareCodePoints(a, b);
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
 * Build a reader for a 1D tensor that resolves storage, offset and stride once.
 * Handles both string and numeric dtypes; bigint stays bigint, everything else
 * becomes a number.
 */
function makeReader1D(t: Tensor): (i: number) => Category {
  const stride = getStride1D(t);
  const base = t.offset;
  if (t.dtype === "string") {
    const data = getStringData(t);
    return (i) => {
      const value = data[base + i * stride];
      if (value === undefined) {
        throw new DeepboxError("Internal error: string tensor access out of bounds");
      }
      return value;
    };
  }
  const data = getNumericData(t);
  return (i) => {
    const value = data[base + i * stride];
    if (value === undefined) {
      throw new DeepboxError("Internal error: numeric tensor access out of bounds");
    }
    return typeof value === "bigint" ? value : Number(value);
  };
}

/**
 * Build a reader for a 2D tensor that resolves storage, offset and strides once.
 */
function makeReader2D(t: Tensor): (row: number, col: number) => Category {
  const [stride0, stride1] = getStrides2D(t);
  const base = t.offset;
  if (t.dtype === "string") {
    const data = getStringData(t);
    return (row, col) => {
      const value = data[base + row * stride0 + col * stride1];
      if (value === undefined) {
        throw new DeepboxError("Internal error: string tensor access out of bounds");
      }
      return value;
    };
  }
  const data = getNumericData(t);
  return (row, col) => {
    const value = data[base + row * stride0 + col * stride1];
    if (value === undefined) {
      throw new DeepboxError("Internal error: numeric tensor access out of bounds");
    }
    return typeof value === "bigint" ? value : Number(value);
  };
}

/**
 * Read a numeric 2D tensor element as a number (used by inverse transforms).
 */
function makeNumberReader2D(t: Tensor): (row: number, col: number) => number {
  const [stride0, stride1] = getStrides2D(t);
  const base = t.offset;
  const data = getNumericData(t);
  return (row, col) => {
    const value = data[base + row * stride0 + col * stride1];
    if (value === undefined) {
      throw new DeepboxError("Internal error: numeric tensor access out of bounds");
    }
    return Number(value);
  };
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
 * Build the category -> index lookup for one feature.
 */
function buildIndexMap(cats: readonly Category[], context: string): Map<Category, number> {
  const map = new Map<Category, number>();
  for (let k = 0; k < cats.length; k++) {
    map.set(categoryValueAt(cats as Category[], k, context), k);
  }
  return map;
}

/**
 * Build the category -> index lookup for each feature.
 */
function buildIndexMaps(
  categories: readonly Category[][],
  context: string
): Map<Category, number>[] {
  return categories.map((cats) => buildIndexMap(cats, context));
}

/**
 * Learn (or validate) the categories of every feature of a 2D tensor.
 *
 * With explicit categories, values of X that are not listed are an error unless
 * `allowUnknown` is true (the encoder will then handle them at transform time).
 */
function learnCategories(
  X: Tensor,
  categoriesOption: CategoriesOption,
  allowUnknown: boolean
): Category[][] {
  const [nSamples, nFeatures] = getShape2D(X);
  const explicitCategories = resolveCategoriesOption(categoriesOption, nFeatures, "categories");
  const read = makeReader2D(X);
  const result: Category[][] = [];

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
      if (!allowUnknown) {
        const known = new Set<Category>(cats);
        for (let i = 0; i < nSamples; i++) {
          const val = read(i, j);
          if (!known.has(val)) {
            throw new InvalidParameterError(
              `Unknown category: ${String(val)} in feature ${j}`,
              "X",
              val
            );
          }
        }
      }
    } else {
      const uniqueSet = new Set<Category>();
      for (let i = 0; i < nSamples; i++) {
        uniqueSet.add(read(i, j));
      }
      cats = sortCategories(uniqueSet, "X");
    }

    if (cats.length === 0) {
      throw new InvalidParameterError("Each feature must have at least one category", "X", j);
    }
    result.push(cats);
  }
  return result;
}

function emptyCSR(rows: number, cols: number): CSRMatrix {
  return CSRMatrix.fromCOO({
    rows,
    cols,
    rowIndices: new Int32Array(0),
    colIndices: new Int32Array(0),
    values: new Float64Array(0),
  });
}

/**
 * Encode target labels with value between 0 and n_classes-1.
 *
 * This transformer encodes categorical labels (strings or numbers) into integers
 * in the range [0, n_classes-1]. It maintains a mapping of unique classes to
 * their integer representations and can reverse the transformation.
 * Classes are sorted: numbers numerically, bigints numerically and strings by
 * Unicode code point (so "B" sorts before "a", as in NumPy).
 *
 * **Time Complexity:**
 * - fit: O(n + k log k) where n is the number of samples and k the number of classes
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
 * @see {@link https://deepbox.dev/docs/preprocess-encoders | Deepbox Encoders}
 */
export class LabelEncoder {
  /** Indicates whether the encoder has been fitted to data */
  private fitted = false;
  /** Array of unique classes found during fitting, sorted for consistency */
  private classes_?: Category[];
  /** Map from class value to encoded integer index for O(1) lookup */
  private classToIndex_?: Map<Category, number>;

  /**
   * The sorted classes learned during fit (a new tensor on every access),
   * or `undefined` before fit.
   */
  get classes(): Tensor | undefined {
    return this.classes_ ? toCategoryVectorTensor(this.classes_, "y") : undefined;
  }

  /**
   * Fit label encoder to a set of labels.
   * Extracts unique classes and creates an index mapping.
   *
   * @param y - Target labels (1D tensor of strings or numbers)
   * @returns this - Returns self for method chaining
   * @throws {InvalidParameterError} If y is empty
   */
  fit(y: EncoderInput1D): this {
    const t = coerceToTensor1D(y);
    assert1D(t, "y");
    if (t.size === 0) {
      throw new InvalidParameterError("Cannot fit LabelEncoder on empty array", "y");
    }

    const read = makeReader1D(t);
    // Collect unique classes using a Set for O(n) complexity
    const uniqueSet = new Set<Category>();
    for (let i = 0; i < t.size; i++) {
      uniqueSet.add(read(i));
    }

    // Sort classes for consistent ordering across fits
    const classes = sortCategories(uniqueSet, "y");

    // Commit only after everything succeeded so a failed refit keeps the old state.
    this.classes_ = classes;
    this.classToIndex_ = buildIndexMap(classes, "LabelEncoder.fit");
    this.fitted = true;
    return this;
  }

  /**
   * Transform labels to normalized encoding.
   * Each unique label is mapped to an integer in [0, n_classes-1].
   *
   * @param y - Target labels to encode (1D tensor)
   * @returns Encoded labels as a float64 tensor of integer values
   * @throws {NotFittedError} If encoder is not fitted
   * @throws {InvalidParameterError} If y contains labels not seen during fit
   */
  transform(y: EncoderInput1D): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("LabelEncoder must be fitted before transform");
    }
    const t = coerceToTensor1D(y);
    assert1D(t, "y");

    const lookup = this.classToIndex_;
    if (!this.classes_ || !lookup) {
      throw new DeepboxError("LabelEncoder internal error: missing fitted state");
    }
    if (t.size === 0) {
      return tensor([]);
    }

    const read = makeReader1D(t);
    const result = new Float64Array(t.size);

    // Transform each label using O(1) map lookup
    for (let i = 0; i < t.size; i++) {
      const val = read(i);
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

    return tensor(result);
  }

  /**
   * Fit label encoder and return encoded labels in one step.
   * Convenience method equivalent to calling fit(y).transform(y).
   *
   * @param y - Target labels (1D tensor)
   * @returns Encoded labels as a float64 tensor of integer values
   */
  fitTransform(y: EncoderInput1D): Tensor {
    return this.fit(y).transform(y);
  }

  /**
   * Transform integer labels back to original encoding.
   * Reverses the encoding performed by transform().
   *
   * @param y - Encoded labels (1D integer tensor or number array)
   * @returns Original labels (strings or numbers)
   * @throws {NotFittedError} If encoder is not fitted
   * @throws {InvalidParameterError} If y contains invalid indices
   */
  inverseTransform(y: EncoderInput1D): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("LabelEncoder must be fitted before inverse_transform");
    }
    const t = coerceToTensor1D(y);
    assert1D(t, "y");
    assertNumericTensor(t, "y");
    const classes = this.classes_;
    if (!classes) {
      throw new DeepboxError("LabelEncoder internal error: missing fitted state");
    }
    if (t.size === 0) {
      return emptyCategoryVectorFromClasses(classes, "y");
    }

    const classesLen = classes.length;
    const read = makeReader1D(t);
    const result = new Array<Category>(t.size);

    // Map each encoded index back to its original class
    for (let i = 0; i < t.size; i++) {
      const idx = Number(read(i));

      // Validate index is in valid range
      if (!Number.isInteger(idx) || idx < 0 || idx >= classesLen) {
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
 * with only one active (set to 1) per sample. Categories of each feature are
 * sorted (numbers numerically, strings by Unicode code point) unless given
 * explicitly through the `categories` option.
 *
 * **Time Complexity:**
 * - fit: O(n*m + sum(k_i log k_i)) where n is samples, m is features, k_i the categories of feature i
 * - transform: O(n*m) with O(1) lookup per value, plus O(n * sum(k_i)) to allocate the dense output
 * - Sparse mode avoids the dense allocation, which matters for high-cardinality features
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
 * // Categories: [blue, red] and [L, M, S]
 * // Result: [[0,1,0,0,1], [1,0,0,1,0], [0,1,1,0,0]]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/preprocess-encoders | Deepbox Encoders}
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
   * @param options.handleUnknown - How to handle unknown categories (default: "error"). With "ignore", an unknown value encodes as all zeros.
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
   * The categories learned for each feature (a copy), or `undefined` before fit.
   */
  get categories(): Category[][] | undefined {
    return this.categories_?.map((c) => c.slice());
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
  fit(X: EncoderInput2D): this {
    const _X = coerceToTensor2D(X);
    assert2D(_X, "X");
    const [nSamples, nFeatures] = getShape2D(_X);

    if (nSamples === 0 || nFeatures === 0) {
      throw new InvalidParameterError("Cannot fit OneHotEncoder on empty array", "X");
    }

    // With handleUnknown="ignore", explicit categories may omit values present in X.
    const categories = learnCategories(_X, this.categoriesOption, this.handleUnknown === "ignore");

    const dropIndices = categories.map((cats): number | null => {
      if (this.drop === "first") return 0;
      if (this.drop === "if_binary") return cats.length === 2 ? 0 : null;
      return null;
    });

    // Commit only after everything succeeded so a failed refit keeps the old state.
    this.categories_ = categories;
    this.categoryToIndex_ = buildIndexMaps(categories, "OneHotEncoder.fit");
    this.dropIndices_ = dropIndices;
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
  transform(X: EncoderInput2D): Tensor | CSRMatrix {
    if (!this.fitted) {
      throw new NotFittedError("OneHotEncoder must be fitted before transform");
    }
    const _X = coerceToTensor2D(X);
    assert2D(_X, "X");
    const [nSamples, nFeatures] = getShape2D(_X);

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

    // Output width of each feature (category count minus the dropped one).
    const outSizes = new Array<number>(nFeatures);
    const colOffsets = new Array<number>(nFeatures);
    let totalCols = 0;
    for (let j = 0; j < nFeatures; j++) {
      const cats = categories[j];
      if (!cats) {
        throw new DeepboxError("OneHotEncoder internal error: missing fitted categories");
      }
      outSizes[j] = cats.length - ((dropIndices[j] ?? null) === null ? 0 : 1);
      colOffsets[j] = totalCols;
      totalCols += outSizes[j] as number;
    }

    if (nSamples === 0) {
      return this.sparse ? emptyCSR(0, totalCols) : zeros([0, totalCols], { dtype: "float64" });
    }

    // Output column of the active entry for each (sample, feature); -1 means all zeros.
    const active = new Int32Array(nSamples * nFeatures).fill(-1);
    const read = makeReader2D(_X);
    let nnz = 0;
    for (let j = 0; j < nFeatures; j++) {
      const map = categoryMaps[j];
      if (!map) {
        throw new DeepboxError("OneHotEncoder internal error: missing fitted categories");
      }
      const dropIndex = dropIndices[j] ?? null;
      const colOffset = colOffsets[j] as number;
      for (let i = 0; i < nSamples; i++) {
        const val = read(i, j);
        const idx = map.get(val);
        if (idx === undefined) {
          if (this.handleUnknown === "ignore") continue;
          throw new InvalidParameterError(
            `Unknown category: ${String(val)} in feature ${j}`,
            "X",
            val
          );
        }
        if (dropIndex !== null && idx === dropIndex) continue;
        const adjusted = dropIndex !== null && idx > dropIndex ? idx - 1 : idx;
        active[i * nFeatures + j] = colOffset + adjusted;
        nnz++;
      }
    }

    if (this.sparse) {
      const rowIdx = new Int32Array(nnz);
      const colIdx = new Int32Array(nnz);
      const vals = new Float64Array(nnz).fill(1);
      let p = 0;
      for (let i = 0; i < nSamples; i++) {
        for (let j = 0; j < nFeatures; j++) {
          const col = active[i * nFeatures + j] as number;
          if (col < 0) continue;
          rowIdx[p] = i;
          colIdx[p] = col;
          p++;
        }
      }
      return CSRMatrix.fromCOO({
        rows: nSamples,
        cols: totalCols,
        rowIndices: rowIdx,
        colIndices: colIdx,
        values: vals,
      });
    }

    const result = new Float64Array(nSamples * totalCols);
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) {
        const col = active[i * nFeatures + j] as number;
        if (col >= 0) result[i * totalCols + col] = 1;
      }
    }
    return TensorImpl.fromTypedArray({
      data: result,
      shape: [nSamples, totalCols],
      dtype: "float64",
      device: _X.device,
    });
  }

  /**
   * Fit encoder and transform X in one step.
   *
   * @param X - Training data (2D tensor)
   * @returns Encoded data as dense Tensor or sparse CSRMatrix
   */
  fitTransform(X: EncoderInput2D): Tensor | CSRMatrix {
    return this.fit(X).transform(X);
  }

  /**
   * Convert one-hot encoded data back to the original categories.
   *
   * For each feature the column with the largest value wins. A block of all
   * zeros decodes to the dropped category when `drop` is set; otherwise it is
   * an error (it cannot be mapped back, for example after unknown categories
   * were ignored).
   *
   * @param X - One-hot data (dense Tensor or CSRMatrix)
   * @returns Matrix of original categories
   * @throws {NotFittedError} If encoder is not fitted
   * @throws {InvalidParameterError} If the column count does not match or a block has no active column
   */
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
    const readNum = makeNumberReader2D(dense);

    for (let i = 0; i < nSamples; i++) {
      const row = new Array<Category>(nFeatures);
      result[i] = row;
      let colOffset = 0;
      for (let j = 0; j < nFeatures; j++) {
        const cats = categories[j];
        const dropIndex = dropIndices[j] ?? null;
        if (!cats) {
          throw new DeepboxError("OneHotEncoder internal error: missing fitted categories");
        }
        const outSize = cats.length - (dropIndex === null ? 0 : 1);
        if (outSize === 0) {
          row[j] = categoryValueAt(cats, dropIndex ?? 0, "OneHotEncoder.inverseTransform");
          continue;
        }

        // NaN never wins the comparison, so it cannot hide a real active column.
        let maxIdx = -1;
        let maxVal = Number.NEGATIVE_INFINITY;
        for (let k = 0; k < outSize; k++) {
          const val = readNum(i, colOffset + k);
          if (val > maxVal) {
            maxVal = val;
            maxIdx = k;
          }
        }

        if (!(maxVal > 0)) {
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
 * based on their sorted order (or the order of the `categories` option).
 * Unlike OneHotEncoder, this maintains a single column per feature.
 *
 * **Time Complexity:**
 * - fit: O(n*m + sum(k_j log k_j)) where n=samples, m=features, k_j=categories of feature j
 * - transform: O(n*m) with O(1) map lookup per value
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
 * @see {@link https://deepbox.dev/docs/preprocess-encoders | Deepbox Encoders}
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
   * @param options.unknownValue - Encoded value for unknown categories when handleUnknown="useEncodedValue" (default -1)
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
    if (typeof this.unknownValue !== "number") {
      throw new InvalidParameterError(
        "unknownValue must be an integer or NaN",
        "unknownValue",
        this.unknownValue
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
   * The categories learned for each feature (a copy), or `undefined` before fit.
   */
  get categories(): Category[][] | undefined {
    return this.categories_?.map((c) => c.slice());
  }

  /**
   * Fit OrdinalEncoder to X.
   * Learns the unique categories for each feature and their ordering.
   *
   * @param X - Training data (2D tensor of categorical features)
   * @returns this - Returns self for method chaining
   * @throws {InvalidParameterError} If X is empty
   */
  fit(X: EncoderInput2D): this {
    const _X = coerceToTensor2D(X);
    assert2D(_X, "X");
    const [nSamples, nFeatures] = getShape2D(_X);

    if (nSamples === 0 || nFeatures === 0) {
      throw new InvalidParameterError("Cannot fit OrdinalEncoder on empty array", "X");
    }

    // With handleUnknown="useEncodedValue", explicit categories may omit values present in X.
    const categories = learnCategories(
      _X,
      this.categoriesOption,
      this.handleUnknown === "useEncodedValue"
    );

    if (this.handleUnknown === "useEncodedValue" && Number.isFinite(this.unknownValue)) {
      for (const cats of categories) {
        if (this.unknownValue >= 0 && this.unknownValue < cats.length) {
          throw new InvalidParameterError(
            "unknownValue must be outside the range of encoded categories",
            "unknownValue",
            this.unknownValue
          );
        }
      }
    }

    // Commit only after everything succeeded so a failed refit keeps the old state.
    this.categories_ = categories;
    this.categoryToIndex_ = buildIndexMaps(categories, "OrdinalEncoder.fit");
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
  transform(X: EncoderInput2D): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("OrdinalEncoder must be fitted before transform");
    }
    const _X = coerceToTensor2D(X);
    assert2D(_X, "X");
    const [nSamples, nFeatures] = getShape2D(_X);
    const maps = this.categoryToIndex_;
    if (!maps) {
      throw new DeepboxError("OrdinalEncoder internal error: missing fitted categories");
    }
    if (nFeatures !== maps.length) {
      throw new InvalidParameterError(
        "X has a different feature count than during fit",
        "X",
        nFeatures
      );
    }

    if (nSamples === 0) {
      return zeros([0, nFeatures], { dtype: "float64" });
    }

    const result = new Float64Array(nSamples * nFeatures);
    const read = makeReader2D(_X);

    // Transform each value to its ordinal index using O(1) map lookup
    for (let j = 0; j < nFeatures; j++) {
      const map = maps[j];
      if (!map) {
        throw new DeepboxError("OrdinalEncoder internal error: missing fitted categories");
      }
      for (let i = 0; i < nSamples; i++) {
        const val = read(i, j);
        const idx = map.get(val);
        if (idx === undefined) {
          if (this.handleUnknown === "useEncodedValue") {
            result[i * nFeatures + j] = this.unknownValue;
            continue;
          }
          throw new InvalidParameterError(
            `Unknown category: ${String(val)} in feature ${j}`,
            "X",
            val
          );
        }
        result[i * nFeatures + j] = idx;
      }
    }

    return TensorImpl.fromTypedArray({
      data: result,
      shape: [nSamples, nFeatures],
      dtype: "float64",
      device: getConfig().defaultDevice,
    });
  }

  /**
   * Fit encoder and transform X in one step.
   * Convenience method equivalent to calling fit(X).transform(X).
   *
   * @param X - Training data (2D tensor)
   * @returns Encoded data
   */
  fitTransform(X: EncoderInput2D): Tensor {
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
  inverseTransform(X: EncoderInput2D): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("OrdinalEncoder must be fitted before inverse_transform");
    }
    const _X = coerceToTensor2D(X);
    assert2D(_X, "X");
    assertNumericTensor(_X, "X");
    const [nSamples, nFeatures] = getShape2D(_X);
    const categories = this.categories_ ?? [];
    if (nFeatures !== categories.length) {
      throw new InvalidParameterError(
        "X has a different feature count than during fit",
        "X",
        nFeatures
      );
    }

    if (nSamples === 0) {
      return emptyCategoryMatrixFromCategories(categories, nFeatures, "X");
    }

    const result = new Array<Category[]>(nSamples);
    const readNum = makeNumberReader2D(_X);

    // Map each ordinal index back to its original category
    for (let i = 0; i < nSamples; i++) {
      const row = new Array<Category>(nFeatures);
      result[i] = row;
      for (let j = 0; j < nFeatures; j++) {
        const idx = readNum(i, j);
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
        const cats = categories[j];

        // Validate index is in valid range
        if (!cats || !Number.isInteger(idx) || idx < 0 || idx >= cats.length) {
          throw new InvalidParameterError(
            `Invalid encoded value: ${idx} for feature ${j}. Must be integer in [0, ${(cats?.length ?? 0) - 1}]`,
            "X",
            idx
          );
        }
        row[j] = categoryValueAt(cats, idx, "OrdinalEncoder.inverseTransform");
      }
    }

    return toCategoryMatrixTensor(result, "X");
  }
}

/**
 * Check that every element of a label-set array is itself an array of
 * strings, numbers or bigints.
 */
function assertLabelSets(y: ReadonlyArray<ReadonlyArray<Category>>): void {
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
 * @see {@link https://deepbox.dev/docs/preprocess-encoders | Deepbox Encoders}
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
   * The sorted classes learned during fit (a new tensor on every access),
   * or `undefined` before fit.
   */
  get classes(): Tensor | undefined {
    return this.classes_ ? toCategoryVectorTensor(this.classes_, "y") : undefined;
  }

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
  fit(y: EncoderInput1D): this {
    const _y = coerceToTensor1D(y);
    assert1D(_y, "y");
    if (_y.size === 0) {
      throw new InvalidParameterError("Cannot fit LabelBinarizer on empty array", "y");
    }

    // Collect unique classes
    const read = makeReader1D(_y);
    const uniqueSet = new Set<Category>();
    for (let i = 0; i < _y.size; i++) {
      uniqueSet.add(read(i));
    }

    // Sort classes for consistent ordering; commit only after everything succeeded.
    const classes = sortCategories(uniqueSet, "y");
    this.classes_ = classes;
    this.classToIndex_ = buildIndexMap(classes, "LabelBinarizer.fit");
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
  transform(y: EncoderInput1D): Tensor | CSRMatrix {
    if (!this.fitted) {
      throw new NotFittedError("LabelBinarizer must be fitted before transform");
    }
    const _y = coerceToTensor1D(y);
    assert1D(_y, "y");
    const nClasses = this.classes_?.length ?? 0;
    if (_y.size === 0) {
      return this.sparse ? emptyCSR(0, nClasses) : zeros([0, nClasses], { dtype: "float64" });
    }

    const nSamples = _y.size;
    const lookup = this.classToIndex_;
    if (!lookup) {
      throw new DeepboxError("LabelBinarizer internal error: missing fitted lookup");
    }

    // Class index of every sample.
    const read = makeReader1D(_y);
    const classIdx = new Int32Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      const val = read(i);
      const idx = lookup.get(val);
      if (idx === undefined) {
        throw new InvalidParameterError(
          `Unknown label: ${String(val)}. Label must be present during fit.`,
          "y",
          val
        );
      }
      classIdx[i] = idx;
    }

    if (this.sparse) {
      const rowIdx = new Int32Array(nSamples);
      for (let i = 0; i < nSamples; i++) rowIdx[i] = i;
      return CSRMatrix.fromCOO({
        rows: nSamples,
        cols: nClasses,
        rowIndices: rowIdx,
        colIndices: classIdx,
        values: new Float64Array(nSamples).fill(this.posLabel),
      });
    }

    const result = new Float64Array(nSamples * nClasses);
    if (this.negLabel !== 0) result.fill(this.negLabel);
    for (let i = 0; i < nSamples; i++) {
      result[i * nClasses + (classIdx[i] as number)] = this.posLabel;
    }
    return TensorImpl.fromTypedArray({
      data: result,
      shape: [nSamples, nClasses],
      dtype: "float64",
      device: getConfig().defaultDevice,
    });
  }

  /**
   * Fit binarizer and transform labels in one step.
   * Convenience method equivalent to calling fit(y).transform(y).
   *
   * @param y - Target labels (1D tensor)
   * @returns Binary matrix (Tensor or CSRMatrix)
   */
  fitTransform(y: EncoderInput1D): Tensor | CSRMatrix {
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

      // Repeated entries of a non-canonical matrix must count as their sum.
      const canon = Y.hasCanonicalFormat ? Y : Y.canonicalize();
      const result = new Array<Category>(rows);
      for (let i = 0; i < rows; i++) {
        let maxIdx = 0;
        let maxVal = this.negLabel;
        const start = canon.indptr[i] ?? 0;
        const end = canon.indptr[i + 1] ?? start;
        for (let p = start; p < end; p++) {
          const col = canon.indices[p];
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
          const raw = canon.data[p];
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
    const readNum = makeNumberReader2D(Y);

    // For each sample, find the class with maximum activation. NaN never wins.
    for (let i = 0; i < nSamples; i++) {
      let maxIdx = -1;
      let maxVal = Number.NEGATIVE_INFINITY;
      for (let j = 0; j < nCols; j++) {
        const val = readNum(i, j);
        if (val > maxVal) {
          maxVal = val;
          maxIdx = j;
        }
      }

      if (maxIdx < 0 || maxVal <= this.negLabel) {
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
 * @see {@link https://deepbox.dev/docs/preprocess-encoders | Deepbox Encoders}
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
   * The classes in column order (a new tensor on every access), or `undefined`
   * before fit.
   */
  get classes(): Tensor | undefined {
    return this.classes_ ? toCategoryVectorTensor(this.classes_, "classes") : undefined;
  }

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
    if (!Array.isArray(y)) {
      throw new InvalidParameterError(
        "MultiLabelBinarizer expects an array of label arrays",
        "y",
        y
      );
    }
    if (y.length === 0) {
      throw new InvalidParameterError("Cannot fit MultiLabelBinarizer on empty array", "y");
    }
    assertLabelSets(y);

    let classes: Category[];
    if (this.classesOption) {
      classes = Array.from(this.classesOption);
    } else {
      // Collect all unique labels across all samples
      const uniqueSet = new Set<Category>();
      for (const labels of y) {
        for (const label of labels) {
          uniqueSet.add(label);
        }
      }

      // Sort classes for consistent ordering
      classes = sortCategories(uniqueSet, "y");
    }
    const lookup = buildIndexMap(classes, "MultiLabelBinarizer.fit");
    if (this.classesOption) {
      for (const labels of y) {
        for (const label of labels) {
          if (!lookup.has(label)) {
            throw new InvalidParameterError(
              `Unknown label: ${String(label)}. Label must be present in classes.`,
              "y",
              label
            );
          }
        }
      }
    }

    // Commit only after everything succeeded so a failed refit keeps the old state.
    this.classes_ = classes;
    this.classToIndex_ = lookup;
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
    if (!Array.isArray(y)) {
      throw new InvalidParameterError(
        "MultiLabelBinarizer expects an array of label arrays",
        "y",
        y
      );
    }
    assertLabelSets(y);
    const nClasses = this.classes_?.length ?? 0;
    if (y.length === 0) {
      return this.sparse ? emptyCSR(0, nClasses) : zeros([0, nClasses], { dtype: "float64" });
    }

    const nSamples = y.length;
    const lookup = this.classToIndex_;
    if (!lookup) {
      throw new DeepboxError("MultiLabelBinarizer internal error: missing fitted lookup");
    }

    // Distinct class indices of every sample (a repeated label counts once).
    const rowIdx: number[] = [];
    const colIdx: number[] = [];
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
      }
    }

    if (this.sparse) {
      return CSRMatrix.fromCOO({
        rows: nSamples,
        cols: nClasses,
        rowIndices: Int32Array.from(rowIdx),
        colIndices: Int32Array.from(colIdx),
        values: new Float64Array(rowIdx.length).fill(1),
      });
    }

    const result = new Float64Array(nSamples * nClasses);
    for (let p = 0; p < rowIdx.length; p++) {
      result[(rowIdx[p] as number) * nClasses + (colIdx[p] as number)] = 1;
    }
    return TensorImpl.fromTypedArray({
      data: result,
      shape: [nSamples, nClasses],
      dtype: "float64",
      device: getConfig().defaultDevice,
    });
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

      // Repeated entries of a non-canonical matrix must count as their sum.
      const canon = Y.hasCanonicalFormat ? Y : Y.canonicalize();
      const result: Category[][] = [];
      for (let i = 0; i < rows; i++) {
        const labels: Category[] = [];
        const start = canon.indptr[i] ?? 0;
        const end = canon.indptr[i + 1] ?? start;
        for (let p = start; p < end; p++) {
          const col = canon.indices[p];
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
          const raw = canon.data[p];
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
    const readNum = makeNumberReader2D(Y);

    // For each sample, collect all active classes
    for (let i = 0; i < nSamples; i++) {
      const labels: Category[] = [];
      for (let j = 0; j < nClasses; j++) {
        // A class is active when its value is positive (typically 1); NaN is inactive.
        if (readNum(i, j) > 0) {
          labels.push(categoryValueAt(classes, j, "MultiLabelBinarizer.inverseTransform"));
        }
      }
      result.push(labels);
    }

    return result;
  }
}

/**
 * Target-based encoding for categorical features.
 *
 * Replaces each category with a smoothed mean of the target for that category:
 * `(n * categoryMean + smooth * targetMean) / (n + smooth)`, which pulls rare
 * categories toward the overall target mean. Categories not seen during fit
 * encode as the overall target mean. Features may be numbers, bigints or strings.
 *
 * `fitTransform` uses cross-fitting (as scikit-learn does), so the training
 * encoding of a row never uses that row's own target. Rows are assigned to
 * `cv` folds deterministically (row `i` goes to fold `i % cv`).
 *
 * @example
 * ```ts
 * import { TargetEncoder } from 'deepbox/preprocess';
 *
 * const enc = new TargetEncoder({ smooth: 10 });
 * enc.fit(tensor([[0],[1],[0],[1],[2]]), tensor([10, 20, 12, 18, 15]));
 * const encoded = enc.transform(tensor([[0],[1],[2]]));
 * ```
 *
 * @category Encoders
 */
export class TargetEncoder {
  private _smooth: number;
  private _cv: number;
  private _encodings: Array<Map<Category, number>> = [];
  private _globalMean = 0;
  private _nFeatures = 0;
  private _fitted = false;

  /**
   * @param options.smooth - Weight of the overall target mean, as a number of pseudo-observations (default 5, must be >= 0).
   * @param options.cv - Number of folds used by `fitTransform` (default 5, must be an integer >= 2).
   */
  constructor(options: { smooth?: number; cv?: number } = {}) {
    this._smooth = options.smooth ?? 5;
    this._cv = options.cv ?? 5;
    if (typeof this._smooth !== "number" || !Number.isFinite(this._smooth) || this._smooth < 0) {
      throw new InvalidParameterError(
        "smooth must be a non-negative finite number",
        "smooth",
        this._smooth
      );
    }
    if (!Number.isInteger(this._cv) || this._cv < 2) {
      throw new InvalidParameterError("cv must be an integer >= 2", "cv", this._cv);
    }
  }

  get isFitted(): boolean {
    return this._fitted;
  }

  /** Mean of the training target, used for unseen categories (`undefined` before fit). */
  get targetMean(): number | undefined {
    return this._fitted ? this._globalMean : undefined;
  }

  /** Learned encoding of every category, one map per feature (copies; `undefined` before fit). */
  get encodings(): Array<Map<Category, number>> | undefined {
    return this._fitted ? this._encodings.map((m) => new Map(m)) : undefined;
  }

  private static prepare(
    X: EncoderInput2D,
    y: EncoderInput1D
  ): {
    xt: Tensor;
    nSamples: number;
    nFeatures: number;
    yVals: Float64Array;
  } {
    const xt = coerceToTensor2D(X, "X");
    const yt = coerceToTensor1D(y, "y");
    if (xt.ndim !== 2) {
      throw new ShapeError(`X must be 2D, got ${xt.ndim}D`);
    }
    if (yt.ndim !== 1) {
      throw new ShapeError(`y must be 1D, got ${yt.ndim}D`);
    }
    assertNumericTensor(yt, "y");
    const [nSamples, nFeatures] = getShape2D(xt);
    if (nSamples !== yt.size) {
      throw new ShapeError("X and y must have same number of samples");
    }
    if (nSamples === 0 || nFeatures === 0) {
      throw new InvalidParameterError("Cannot fit TargetEncoder on empty array", "X");
    }
    const readY = makeReader1D(yt);
    const yVals = new Float64Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      const v = Number(readY(i));
      if (!Number.isFinite(v)) {
        throw new DataValidationError(`y must be finite; found ${v} at index ${i}`);
      }
      yVals[i] = v;
    }
    return { xt, nSamples, nFeatures, yVals };
  }

  /**
   * Fit the encoder to training data.
   *
   * @param X - Feature matrix (2D tensor or array of categories: numbers, bigints or strings)
   * @param y - Numeric target values (1D, finite)
   */
  fit(X: EncoderInput2D, y: EncoderInput1D): this {
    const { xt, nSamples, nFeatures, yVals } = TargetEncoder.prepare(X, y);
    const read = makeReader2D(xt);

    let globalSum = 0;
    for (let i = 0; i < nSamples; i++) globalSum += yVals[i] as number;
    const globalMean = globalSum / nSamples;

    const encodings: Array<Map<Category, number>> = [];
    for (let f = 0; f < nFeatures; f++) {
      const stats = new Map<Category, [number, number]>();
      for (let i = 0; i < nSamples; i++) {
        const cat = read(i, f);
        let entry = stats.get(cat);
        if (entry === undefined) {
          entry = [0, 0];
          stats.set(cat, entry);
        }
        entry[0] += yVals[i] as number;
        entry[1] += 1;
      }
      const featureMap = new Map<Category, number>();
      for (const [cat, [sum, count]] of stats) {
        // (count * catMean + smooth * globalMean) / (count + smooth)
        featureMap.set(cat, (sum + this._smooth * globalMean) / (count + this._smooth));
      }
      encodings.push(featureMap);
    }

    // Commit only after everything succeeded so a failed refit keeps the old state.
    this._encodings = encodings;
    this._globalMean = globalMean;
    this._nFeatures = nFeatures;
    this._fitted = true;
    return this;
  }

  /**
   * Transform categorical features to target-encoded values.
   * Categories that were not seen during fit are encoded as the target mean.
   *
   * @param X - Feature matrix (2D tensor or array of categories)
   * @returns Encoded 2D float64 tensor
   */
  transform(X: EncoderInput2D): Tensor {
    if (!this._fitted) {
      throw new NotFittedError("TargetEncoder is not fitted yet");
    }

    const xt = coerceToTensor2D(X, "X");
    if (xt.ndim !== 2) {
      throw new ShapeError(`X must be 2D, got ${xt.ndim}D`);
    }
    const [nSamples, nFeatures] = getShape2D(xt);
    if (nFeatures !== this._nFeatures) {
      throw new ShapeError(`Expected ${this._nFeatures} features, got ${nFeatures}`);
    }

    const out = new Float64Array(nSamples * nFeatures);
    if (nSamples > 0) {
      const read = makeReader2D(xt);
      for (let f = 0; f < nFeatures; f++) {
        const featureMap = this._encodings[f];
        for (let i = 0; i < nSamples; i++) {
          out[i * nFeatures + f] = featureMap?.get(read(i, f)) ?? this._globalMean;
        }
      }
    }

    return TensorImpl.fromTypedArray({
      data: out,
      shape: [nSamples, nFeatures],
      dtype: "float64",
      device: getConfig().defaultDevice,
    });
  }

  /**
   * Fit and transform in one step using internal cross-fitting (scikit-learn's
   * TargetEncoder.fit_transform behavior): each row's encoding is computed from
   * the OTHER folds, so a row never sees its own target. Within a fold split,
   * both the category statistics and the fallback target mean come from the
   * training folds only. A plain fit(X,y)+transform(X) would leak the target
   * and give optimistically biased cross-validation scores. The encoder is also
   * fit on the full data so later `transform` calls use the complete statistics.
   *
   * @throws {InvalidParameterError} If there are fewer than 2 samples
   */
  fitTransform(X: EncoderInput2D, y: EncoderInput1D): Tensor {
    const { xt, nSamples, nFeatures, yVals } = TargetEncoder.prepare(X, y);
    if (nSamples < 2) {
      throw new InvalidParameterError(
        "TargetEncoder.fitTransform needs at least 2 samples for cross-fitting",
        "X",
        nSamples
      );
    }
    const read = makeReader2D(xt);

    const nFolds = Math.min(this._cv, nSamples);
    const foldOf = (i: number): number => i % nFolds;

    const foldYSum = new Float64Array(nFolds);
    const foldYCount = new Float64Array(nFolds);
    for (let i = 0; i < nSamples; i++) {
      foldYSum[foldOf(i)] = (foldYSum[foldOf(i)] as number) + (yVals[i] as number);
      foldYCount[foldOf(i)] = (foldYCount[foldOf(i)] as number) + 1;
    }
    // Mean of the training folds for each held-out fold.
    const trainMean = new Float64Array(nFolds);
    for (let h = 0; h < nFolds; h++) {
      let s = 0;
      let c = 0;
      for (let k = 0; k < nFolds; k++) {
        if (k === h) continue;
        s += foldYSum[k] as number;
        c += foldYCount[k] as number;
      }
      trainMean[h] = s / c;
    }

    const out = new Float64Array(nSamples * nFeatures);
    const ids = new Int32Array(nSamples);
    for (let f = 0; f < nFeatures; f++) {
      const catIds = new Map<Category, number>();
      for (let i = 0; i < nSamples; i++) {
        const cat = read(i, f);
        let id = catIds.get(cat);
        if (id === undefined) {
          id = catIds.size;
          catIds.set(cat, id);
        }
        ids[i] = id;
      }
      const nCats = catIds.size;
      const totalSum = new Float64Array(nCats);
      const totalCount = new Float64Array(nCats);
      const foldSum = new Float64Array(nCats * nFolds);
      const foldCount = new Float64Array(nCats * nFolds);
      for (let i = 0; i < nSamples; i++) {
        const id = ids[i] as number;
        const y = yVals[i] as number;
        const slot = id * nFolds + foldOf(i);
        totalSum[id] = (totalSum[id] as number) + y;
        totalCount[id] = (totalCount[id] as number) + 1;
        foldSum[slot] = (foldSum[slot] as number) + y;
        foldCount[slot] = (foldCount[slot] as number) + 1;
      }

      for (let i = 0; i < nSamples; i++) {
        const id = ids[i] as number;
        const hold = foldOf(i);
        const slot = id * nFolds + hold;
        const count = (totalCount[id] as number) - (foldCount[slot] as number);
        const mean = trainMean[hold] as number;
        if (count === 0) {
          out[i * nFeatures + f] = mean;
        } else {
          const sum = (totalSum[id] as number) - (foldSum[slot] as number);
          out[i * nFeatures + f] = (sum + this._smooth * mean) / (count + this._smooth);
        }
      }
    }

    // Fit the full-data encodings for subsequent transform() calls.
    this.fit(xt, y);
    return TensorImpl.fromTypedArray({
      data: out,
      shape: [nSamples, nFeatures],
      dtype: "float64",
      device: getConfig().defaultDevice,
    });
  }
}
