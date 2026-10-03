import { DeepboxError, InvalidParameterError, MemoryError, ShapeError } from "../core/errors";
import { type Tensor, zeros } from "../ndarray";
import { __random } from "../random/random";
import { createRandomStream, deriveSeed, shuffleIndicesInPlace } from "./_internal";

/**
 * A single train/test split expressed as sample index arrays.
 *
 * @see {@link https://deepbox.dev/docs/preprocess-splitting | Deepbox Data Splitting}
 */
export type SplitResult = {
  readonly trainIndex: number[];
  readonly testIndex: number[];
};

/** A class or group label read from a 1D tensor or array. */
type Label = string | number | bigint;

/** Upper bound on the number of index entries a materialised split list may hold. */
const MAX_SPLIT_INDEX_ENTRIES = 50_000_000;

/** Upper bound on the number of splits LeavePOut will materialise. */
const MAX_LEAVE_P_OUT_SPLITS = 100_000;

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

function validateNSplits(nSplits: number): void {
  if (!Number.isInteger(nSplits) || nSplits < 2) {
    throw new InvalidParameterError("nSplits must be an integer at least 2", "nSplits", nSplits);
  }
}

function validatePositiveSplitCount(nSplits: number): void {
  if (!Number.isInteger(nSplits) || nSplits < 1) {
    throw new InvalidParameterError("nSplits must be a positive integer", "nSplits", nSplits);
  }
}

function validateRandomState(randomState: number | undefined): void {
  if (randomState !== undefined && (!Number.isSafeInteger(randomState) || randomState < 0)) {
    throw new InvalidParameterError(
      "randomState must be a non-negative safe integer",
      "randomState",
      randomState
    );
  }
}

/** Number of samples along the first axis of `X`. */
function getNSamples(X: Tensor): number {
  const shape0 = X.shape[0];
  if (X.ndim < 1 || shape0 === undefined) {
    throw new ShapeError("X must have valid shape[0]; got a 0-d tensor");
  }
  return shape0;
}

function assertMemoryBudget(entries: number, what: string): void {
  if (entries > MAX_SPLIT_INDEX_ENTRIES) {
    throw new MemoryError(
      `${what} would materialise about ${Math.floor(entries)} index entries, which exceeds the ` +
        `limit of ${MAX_SPLIT_INDEX_ENTRIES}`,
      { requestedBytes: entries * 8 }
    );
  }
}

// ---------------------------------------------------------------------------
// Train/test size resolution
// ---------------------------------------------------------------------------

type SplitSpec = {
  kind: "fraction" | "count";
  value: number;
};

function parseSplitSpec(value: number | undefined, name: string): SplitSpec | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (!Number.isFinite(value) || value <= 0) {
    throw new InvalidParameterError(`${name} must be a positive number`, name, value);
  }
  if (value < 1) {
    return { kind: "fraction", value };
  }
  if (!Number.isInteger(value)) {
    throw new InvalidParameterError(
      `${name} must be an integer when provided as an absolute size`,
      name,
      value
    );
  }
  return { kind: "count", value };
}

/**
 * Convert a size spec to a sample count. Fractions are rounded like scikit-learn
 * (floor for the train side, ceil for the test side), with a small relative
 * tolerance so that products such as `100 * 0.07 = 7.000000000000001` do not
 * round to the wrong integer.
 */
function resolveSplitCount(spec: SplitSpec, nSamples: number, isTrain: boolean): number {
  if (spec.kind === "count") {
    return spec.value;
  }
  const exact = nSamples * spec.value;
  const tolerance = 1e-9 * Math.max(1, Math.abs(exact));
  return isTrain ? Math.floor(exact + tolerance) : Math.ceil(exact - tolerance);
}

function resolveTrainTestCounts(
  nSamples: number,
  trainSize: number | undefined,
  testSize: number | undefined,
  defaultTestSize = 0.25,
  unit: "sample" | "group" = "sample"
): [number, number] {
  const effectiveTestSize =
    trainSize === undefined && testSize === undefined ? defaultTestSize : testSize;
  const trainSpec = parseSplitSpec(trainSize, "trainSize");
  const testSpec = parseSplitSpec(effectiveTestSize, "testSize");

  if (trainSpec?.kind === "count" && trainSpec.value > nSamples) {
    throw new InvalidParameterError(
      `trainSize must not exceed number of ${unit}s`,
      "trainSize",
      trainSpec.value
    );
  }
  if (testSpec?.kind === "count" && testSpec.value > nSamples) {
    throw new InvalidParameterError(
      `testSize must not exceed number of ${unit}s`,
      "testSize",
      testSpec.value
    );
  }

  if (
    trainSpec?.kind === "fraction" &&
    testSpec?.kind === "fraction" &&
    trainSpec.value + testSpec.value > 1 + 1e-12
  ) {
    throw new InvalidParameterError(
      "trainSize and testSize fractions must sum to at most 1",
      "trainSize",
      trainSpec.value
    );
  }

  let nTrain = trainSpec === undefined ? undefined : resolveSplitCount(trainSpec, nSamples, true);
  let nTest = testSpec === undefined ? undefined : resolveSplitCount(testSpec, nSamples, false);

  if (nTrain === undefined && nTest === undefined) {
    throw new DeepboxError("Internal error: failed to resolve split sizes");
  }

  if (nTrain === undefined) {
    nTrain = nSamples - (nTest ?? 0);
  }
  if (nTest === undefined) {
    nTest = nSamples - nTrain;
  }

  if (nTrain + nTest > nSamples) {
    throw new InvalidParameterError(
      `trainSize and testSize exceed number of ${unit}s`,
      "trainSize",
      trainSize
    );
  }

  if (nTrain < 1) {
    throw new InvalidParameterError(`trainSize must be at least 1 ${unit}`, "trainSize", trainSize);
  }
  if (nTest < 1) {
    throw new InvalidParameterError(`testSize must be at least 1 ${unit}`, "testSize", testSize);
  }

  return [nTrain, nTest];
}

// ---------------------------------------------------------------------------
// Labels
// ---------------------------------------------------------------------------

/**
 * Total order over labels: numbers and bigints by value (NaN last), strings by
 * UTF-16 code unit order. Code unit order, unlike `localeCompare`, does not
 * depend on the runtime's locale data, so seeded splits are reproducible
 * across machines.
 */
function compareLabels(a: Label, b: Label): number {
  if (typeof a === "string" || typeof b === "string") {
    if (typeof a === "string" && typeof b === "string") {
      if (a < b) return -1;
      return a > b ? 1 : 0;
    }
    return typeof a === "string" ? 1 : -1;
  }
  const aNaN = typeof a === "number" && Number.isNaN(a);
  const bNaN = typeof b === "number" && Number.isNaN(b);
  if (aNaN || bNaN) {
    if (aNaN && bNaN) return 0;
    return aNaN ? 1 : -1;
  }
  if (a < b) return -1;
  return a > b ? 1 : 0;
}

function isLabel(value: unknown): value is Label {
  return typeof value === "string" || typeof value === "number" || typeof value === "bigint";
}

/** Read a 1D tensor (any offset or stride) into an array of labels. */
function readLabels(t: Tensor, name: string): Label[] {
  if (t.ndim !== 1) {
    throw new ShapeError(`${name} must be a 1D tensor, got ${t.ndim}D`);
  }
  const n = t.shape[0] ?? 0;
  const stride = t.strides[0] ?? 1;
  const data = t.data;
  const out = new Array<Label>(n);
  for (let i = 0; i < n; i++) {
    const value: unknown = data[t.offset + i * stride];
    if (!isLabel(value)) {
      throw new DeepboxError("Internal error: unsupported tensor value type");
    }
    out[i] = value;
  }
  return out;
}

/** Read group labels supplied either as a 1D tensor or as a plain array. */
function readGroups(groups: Tensor | readonly Label[]): Label[] {
  if (Array.isArray(groups)) {
    const out: Label[] = [];
    for (let i = 0; i < groups.length; i++) {
      const value: unknown = groups[i];
      if (!isLabel(value)) {
        throw new InvalidParameterError(
          `groups must contain strings, numbers or bigints; element ${i} is ${typeof value}`,
          "groups",
          value
        );
      }
      out.push(value);
    }
    return out;
  }
  return readLabels(groups as Tensor, "groups");
}

/** Buckets of sample indices per distinct label. */
type LabelGroups = {
  /** Distinct labels in order of first appearance. */
  readonly labels: Label[];
  /** Sample indices (ascending) for each label, aligned with `labels`. */
  readonly indices: number[][];
  /** Index into `labels` for every sample. */
  readonly codes: Int32Array;
};

function groupByLabel(values: readonly Label[]): LabelGroups {
  const lookup = new Map<Label, number>();
  const labels: Label[] = [];
  const indices: number[][] = [];
  const codes = new Int32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const value = values[i] as Label;
    let code = lookup.get(value);
    if (code === undefined) {
      code = labels.length;
      lookup.set(value, code);
      labels.push(value);
      indices.push([]);
    }
    (indices[code] as number[]).push(i);
    codes[i] = code;
  }
  return { labels, indices, codes };
}

/** Same grouping, but with distinct labels sorted ascending (like `np.unique`). */
function groupByLabelSorted(values: readonly Label[]): LabelGroups {
  const raw = groupByLabel(values);
  const order = raw.labels
    .map((_, i) => i)
    .sort((a, b) => compareLabels(raw.labels[a] as Label, raw.labels[b] as Label));
  const remap = new Int32Array(order.length);
  order.forEach((oldCode, newCode) => {
    remap[oldCode] = newCode;
  });
  const codes = new Int32Array(raw.codes.length);
  for (let i = 0; i < codes.length; i++) {
    codes[i] = remap[raw.codes[i] as number] as number;
  }
  return {
    labels: order.map((i) => raw.labels[i] as Label),
    indices: order.map((i) => raw.indices[i] as number[]),
    codes,
  };
}

// ---------------------------------------------------------------------------
// Index helpers
// ---------------------------------------------------------------------------

function range(n: number): number[] {
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) out[i] = i;
  return out;
}

/** Append `source[start, end)` to `target` without spreading (no call-stack limit). */
function appendRange(
  target: number[],
  source: readonly number[],
  start: number,
  end: number
): void {
  for (let i = start; i < end; i++) {
    target.push(source[i] as number);
  }
}

function ascending(a: number, b: number): number {
  return a - b;
}

/**
 * Build one {@link SplitResult} per fold from a sample-to-fold assignment.
 * Train and test indices are both ascending, matching scikit-learn.
 */
function splitsFromFolds(foldOf: ArrayLike<number>, nSplits: number): SplitResult[] {
  const n = foldOf.length;
  const tests: number[][] = Array.from({ length: nSplits }, () => []);
  for (let i = 0; i < n; i++) {
    (tests[foldOf[i] as number] as number[]).push(i);
  }
  const splits: SplitResult[] = [];
  for (let fold = 0; fold < nSplits; fold++) {
    const trainIndex: number[] = [];
    for (let i = 0; i < n; i++) {
      if (foldOf[i] !== fold) trainIndex.push(i);
    }
    splits.push({ trainIndex, testIndex: tests[fold] as number[] });
  }
  return splits;
}

function makeFoldSizes(total: number, nSplits: number): number[] {
  const base = Math.floor(total / nSplits);
  const remainder = total % nSplits;
  return Array.from({ length: nSplits }, (_, i) => base + (i < remainder ? 1 : 0));
}

// ---------------------------------------------------------------------------
// Stratified allocation
// ---------------------------------------------------------------------------

/**
 * Distribute `nDraws` draws over classes in proportion to `counts` using the
 * largest remainder method (scikit-learn's `_approximate_mode`). Ties between
 * equal remainders are broken with `random` when given, otherwise by class order.
 */
function approximateMode(
  counts: readonly number[],
  nDraws: number,
  random: (() => number) | undefined
): number[] {
  let total = 0;
  for (const c of counts) total += c;
  const continuous = counts.map((c) => (c / total) * nDraws);
  const floored = continuous.map((c) => Math.floor(c));
  let need = nDraws - floored.reduce((s, v) => s + v, 0);

  if (need > 0) {
    const remainder = continuous.map((c, i) => c - (floored[i] as number));
    const values = [...new Set(remainder)].sort((a, b) => b - a);
    for (const value of values) {
      const ties: number[] = [];
      for (let i = 0; i < remainder.length; i++) {
        if (remainder[i] === value && (floored[i] as number) < (counts[i] as number)) ties.push(i);
      }
      const addNow = Math.min(ties.length, need);
      if (random !== undefined && addNow < ties.length) {
        shuffleIndicesInPlace(ties, random);
      }
      for (let k = 0; k < addNow; k++) {
        floored[ties[k] as number] = (floored[ties[k] as number] as number) + 1;
      }
      need -= addNow;
      if (need === 0) break;
    }
    // Floating point noise can leave a draw unassigned; give it to any class with room.
    for (let i = 0; i < floored.length && need > 0; i++) {
      while (need > 0 && (floored[i] as number) < (counts[i] as number)) {
        floored[i] = (floored[i] as number) + 1;
        need -= 1;
      }
    }
  }
  return floored;
}

/**
 * Validate stratification inputs and compute how many samples of each class
 * go to the train and test sets. Mirrors scikit-learn's StratifiedShuffleSplit:
 * the train allocation is computed first, then the test allocation from what
 * remains, so both sizes are matched exactly.
 */
function stratifiedCounts(
  classIndices: readonly (readonly number[])[],
  nTrain: number,
  nTest: number,
  random: (() => number) | undefined,
  name: string
): { trainCounts: number[]; testCounts: number[] } {
  const sizes = classIndices.map((ix) => ix.length);
  if (sizes.some((size) => size < 2)) {
    throw new InvalidParameterError(`${name} requires at least 2 samples per class`, name, sizes);
  }
  const nClasses = sizes.length;
  if (nTrain < nClasses) {
    throw new InvalidParameterError(
      "trainSize must be at least the number of classes when stratifying",
      "trainSize",
      nTrain
    );
  }
  if (nTest < nClasses) {
    throw new InvalidParameterError(
      "testSize must be at least the number of classes when stratifying",
      "testSize",
      nTest
    );
  }
  const trainCounts = approximateMode(sizes, nTrain, random);
  const remaining = sizes.map((size, i) => size - (trainCounts[i] as number));
  const testCounts = approximateMode(remaining, nTest, random);
  return { trainCounts, testCounts };
}

// ---------------------------------------------------------------------------
// Row gathering
// ---------------------------------------------------------------------------

type IndexableBuffer = { [index: number]: string | number | bigint; readonly length: number };
type CopyableTypedArray = {
  subarray(begin: number, end: number): CopyableTypedArray;
  set(source: CopyableTypedArray, offset: number): void;
};

/**
 * Copy the rows `rows` of `X` (first axis) into a new tensor of the same dtype.
 * Works for any number of dimensions, any dtype (including string and int64)
 * and arbitrary strides or offsets; contiguous rows use a block copy.
 */
function gatherRows(X: Tensor, rows: readonly number[]): Tensor {
  const innerShape = X.shape.slice(1);
  const rowSize = innerShape.reduce((a, b) => a * b, 1);
  const out = zeros([rows.length, ...innerShape], { dtype: X.dtype, device: X.device });
  if (rows.length === 0 || rowSize === 0) {
    return out;
  }

  const rowStride = X.strides[0] ?? 0;
  const src = X.data as unknown as IndexableBuffer;
  const dst = out.data as unknown as IndexableBuffer;

  // Rows are contiguous blocks when every inner axis is C-ordered.
  let expected = 1;
  let blockCopy = true;
  for (let d = X.ndim - 1; d >= 1; d--) {
    const dim = X.shape[d] ?? 1;
    if (dim !== 1 && X.strides[d] !== expected) {
      blockCopy = false;
      break;
    }
    expected *= dim;
  }

  if (blockCopy) {
    const typed = ArrayBuffer.isView(src) && ArrayBuffer.isView(dst);
    for (let r = 0; r < rows.length; r++) {
      const base = X.offset + (rows[r] as number) * rowStride;
      const outBase = r * rowSize;
      if (typed) {
        (dst as unknown as CopyableTypedArray).set(
          (src as unknown as CopyableTypedArray).subarray(base, base + rowSize),
          outBase
        );
      } else {
        for (let j = 0; j < rowSize; j++) {
          dst[outBase + j] = src[base + j] as string | number | bigint;
        }
      }
    }
    return out;
  }

  // General strided layout: precompute the flat offset of every element within a row.
  const offsets = new Array<number>(rowSize);
  const counter = new Array<number>(innerShape.length).fill(0);
  let current = 0;
  for (let j = 0; j < rowSize; j++) {
    offsets[j] = current;
    for (let d = innerShape.length - 1; d >= 0; d--) {
      const stride = X.strides[d + 1] ?? 0;
      counter[d] = (counter[d] as number) + 1;
      current += stride;
      if ((counter[d] as number) < (innerShape[d] as number)) break;
      current -= (counter[d] as number) * stride;
      counter[d] = 0;
    }
  }
  for (let r = 0; r < rows.length; r++) {
    const base = X.offset + (rows[r] as number) * rowStride;
    const outBase = r * rowSize;
    for (let j = 0; j < rowSize; j++) {
      dst[outBase + j] = src[base + (offsets[j] as number)] as string | number | bigint;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// trainTestSplit
// ---------------------------------------------------------------------------

/**
 * Split arrays into random train and test subsets.
 *
 * `X` and `y` may have any number of dimensions (at least one); rows are taken
 * along the first axis. Inputs are never modified. Sizes follow scikit-learn:
 * with neither `trainSize` nor `testSize` given, the test set gets 25% of the
 * samples; fractions round the test size up and the train size down.
 *
 * With `stratify`, each class contributes to the train and test sets in
 * proportion to its size (largest remainder rounding, so the requested sizes
 * are matched exactly). Every class needs at least 2 samples, and both the
 * train and the test set must be at least as large as the number of classes.
 *
 * @param X - Data tensor; rows are samples (shape `[nSamples, ...]`)
 * @param y - Optional targets with the same number of rows as `X`
 * @param options - Split configuration options
 * @param options.testSize - Fraction (in (0, 1)) or absolute number (integer >= 1) of test samples
 * @param options.trainSize - Fraction (in (0, 1)) or absolute number (integer >= 1) of train samples
 * @param options.randomState - Non-negative integer seed; omit for a non-reproducible split
 * @param options.shuffle - Shuffle before splitting (default `true`). With `false`, the first
 *   rows go to the train set and the following rows to the test set.
 * @param options.stratify - 1D class labels, one per sample, to preserve class proportions
 * @returns `[XTrain, XTest]`, or `[XTrain, XTest, yTrain, yTest]` when `y` is given
 * @throws {InvalidParameterError} If sizes are invalid, `X` is empty, or lengths disagree
 * @throws {ShapeError} If `X` is 0-d or `stratify` is not 1D
 *
 * @example
 * ```ts
 * import { trainTestSplit } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1, 2], [3, 4], [5, 6], [7, 8]]);
 * const y = tensor([0, 1, 0, 1]);
 * const [XTrain, XTest, yTrain, yTest] = trainTestSplit(X, y, { testSize: 0.25 });
 * ```
 *
 * @see {@link https://deepbox.dev/docs/preprocess-splitting | Deepbox Data Splitting}
 */
export function trainTestSplit(
  X: Tensor,
  y?: Tensor,
  options?: {
    testSize?: number;
    trainSize?: number;
    randomState?: number;
    shuffle?: boolean;
    stratify?: Tensor;
  }
): Tensor[] {
  const opts = options ?? {};
  const shuffle = opts.shuffle ?? true;
  const randomState = opts.randomState;

  const nSamples = getNSamples(X);

  if (nSamples === 0) {
    throw new InvalidParameterError(
      `Cannot split empty array: X has shape [${X.shape.join(", ")}]`,
      "X"
    );
  }

  if (y) {
    const yShape0 = y.shape[0];
    if (yShape0 === undefined || yShape0 !== nSamples) {
      throw new InvalidParameterError("X and y must have same number of samples", "y", yShape0);
    }
  }

  if (opts.stratify) {
    if (opts.stratify.ndim !== 1) {
      throw new ShapeError(`stratify must be a 1D tensor, got ${opts.stratify.ndim}D`);
    }
    const stratifyShape0 = opts.stratify.shape[0];
    if (stratifyShape0 === undefined || stratifyShape0 !== nSamples) {
      throw new InvalidParameterError(
        "stratify must have same number of samples as X",
        "stratify",
        stratifyShape0
      );
    }
  }

  const [nTrain, nTest] = resolveTrainTestCounts(nSamples, opts.trainSize, opts.testSize);

  const random = randomState !== undefined ? createRandomStream(randomState) : __random;

  let trainIndices: number[] = [];
  let testIndices: number[] = [];

  if (opts.stratify) {
    const { indices: classIndices } = groupByLabelSorted(readLabels(opts.stratify, "stratify"));
    const { trainCounts, testCounts } = stratifiedCounts(
      classIndices,
      nTrain,
      nTest,
      shuffle ? random : undefined,
      "stratify"
    );

    for (let k = 0; k < classIndices.length; k++) {
      const members = [...(classIndices[k] as number[])];
      if (shuffle) shuffleIndicesInPlace(members, random);
      const nTrainK = trainCounts[k] as number;
      const nTestK = testCounts[k] as number;
      appendRange(trainIndices, members, 0, nTrainK);
      appendRange(testIndices, members, nTrainK, nTrainK + nTestK);
    }

    if (shuffle) {
      shuffleIndicesInPlace(trainIndices, random);
      shuffleIndicesInPlace(testIndices, random);
    } else {
      // Without shuffling, keep the original sample order instead of class blocks.
      trainIndices.sort(ascending);
      testIndices.sort(ascending);
    }
  } else {
    const indices = range(nSamples);
    if (shuffle) shuffleIndicesInPlace(indices, random);
    trainIndices = indices.slice(0, nTrain);
    testIndices = indices.slice(nTrain, nTrain + nTest);
  }

  if (trainIndices.length !== nTrain || testIndices.length !== nTest) {
    throw new DeepboxError("Internal error: resolved split indices do not match requested sizes");
  }

  const XTrain = gatherRows(X, trainIndices);
  const XTest = gatherRows(X, testIndices);

  if (y) {
    return [XTrain, XTest, gatherRows(y, trainIndices), gatherRows(y, testIndices)];
  }

  return [XTrain, XTest];
}

// ---------------------------------------------------------------------------
// K-fold family
// ---------------------------------------------------------------------------

/**
 * K-Folds cross-validator.
 *
 * Splits the samples into `nSplits` consecutive folds (the first `n % nSplits`
 * folds hold one extra sample). Each fold is used once as the test set while
 * the remaining folds form the train set. Train and test indices are always
 * returned in ascending order, also when `shuffle` is enabled.
 *
 * @example
 * ```ts
 * import { KFold } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const kf = new KFold({ nSplits: 2 });
 * const splits = kf.split(tensor([[1], [2], [3], [4]]));
 * // splits[0]: train=[2,3], test=[0,1]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/preprocess-splitting | Deepbox Data Splitting}
 */
export class KFold {
  private readonly nSplits: number;
  private readonly shuffle: boolean;
  private readonly randomState: number | undefined;

  /**
   * @param options.nSplits - Number of folds, an integer of at least 2 (default 5)
   * @param options.shuffle - Shuffle the samples before assigning folds (default `false`)
   * @param options.randomState - Non-negative integer seed used when `shuffle` is `true`
   */
  constructor(
    options: {
      nSplits?: number;
      shuffle?: boolean;
      randomState?: number;
    } = {}
  ) {
    this.nSplits = options.nSplits ?? 5;
    this.shuffle = options.shuffle ?? false;
    this.randomState = options.randomState;
    validateNSplits(this.nSplits);
    validateRandomState(this.randomState);
  }

  /**
   * Generate train/test indices for every fold.
   *
   * @param X - Data tensor; only the number of rows is used
   * @throws {InvalidParameterError} If `nSplits` exceeds the number of samples
   */
  split(X: Tensor): SplitResult[] {
    const nSamples = getNSamples(X);
    if (this.nSplits > nSamples) {
      throw new InvalidParameterError(
        "nSplits must not be greater than number of samples",
        "nSplits",
        this.nSplits
      );
    }
    const order = range(nSamples);

    if (this.shuffle) {
      const random =
        this.randomState !== undefined ? createRandomStream(this.randomState) : __random;
      shuffleIndicesInPlace(order, random);
    }

    const foldOf = new Int32Array(nSamples);
    const foldSizes = makeFoldSizes(nSamples, this.nSplits);
    let position = 0;
    for (let fold = 0; fold < this.nSplits; fold++) {
      const end = position + (foldSizes[fold] as number);
      for (; position < end; position++) {
        foldOf[order[position] as number] = fold;
      }
    }
    return splitsFromFolds(foldOf, this.nSplits);
  }

  /** Number of folds. */
  getNSplits(): number {
    return this.nSplits;
  }
}

/**
 * Stratified K-Folds cross-validator.
 *
 * Every fold keeps (as closely as possible) the class proportions of the full
 * data set, and fold sizes differ by at most one sample. Without shuffling the
 * assignment is identical to scikit-learn's `StratifiedKFold`. Each class must
 * have at least `nSplits` samples. Train and test indices are returned in
 * ascending order.
 *
 * @example
 * ```ts
 * import { StratifiedKFold } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const skf = new StratifiedKFold({ nSplits: 2 });
 * const splits = skf.split(tensor([[1], [2], [3], [4]]), tensor([0, 0, 1, 1]));
 * ```
 *
 * @see {@link https://deepbox.dev/docs/preprocess-splitting | Deepbox Data Splitting}
 */
export class StratifiedKFold {
  private readonly nSplits: number;
  private readonly shuffle: boolean;
  private readonly randomState: number | undefined;

  /**
   * @param options.nSplits - Number of folds, an integer of at least 2 (default 5)
   * @param options.shuffle - Shuffle each class before assigning folds (default `false`)
   * @param options.randomState - Non-negative integer seed used when `shuffle` is `true`
   */
  constructor(
    options: {
      nSplits?: number;
      shuffle?: boolean;
      randomState?: number;
    } = {}
  ) {
    this.nSplits = options.nSplits ?? 5;
    this.shuffle = options.shuffle ?? false;
    this.randomState = options.randomState;
    validateNSplits(this.nSplits);
    validateRandomState(this.randomState);
  }

  /**
   * Generate stratified train/test indices for every fold.
   *
   * @param X - Data tensor; only the number of rows is used
   * @param y - 1D class labels (numbers, bigints or strings), one per sample
   * @throws {InvalidParameterError} If lengths differ or a class has fewer than `nSplits` samples
   * @throws {ShapeError} If `y` is not 1D
   */
  split(X: Tensor, y: Tensor): SplitResult[] {
    const nSamples = getNSamples(X);
    if (this.nSplits > nSamples) {
      throw new InvalidParameterError(
        "nSplits must not be greater than number of samples",
        "nSplits",
        this.nSplits
      );
    }
    const yShape0 = y.shape[0];
    if (yShape0 === undefined || yShape0 !== nSamples) {
      throw new InvalidParameterError("X and y must have same number of samples", "y", yShape0);
    }
    const { labels, indices, codes } = groupByLabel(readLabels(y, "y"));
    const nClasses = labels.length;

    for (let k = 0; k < nClasses; k++) {
      const size = (indices[k] as number[]).length;
      if (size < this.nSplits) {
        throw new InvalidParameterError(
          `Each class must have at least nSplits samples; class ${String(labels[k])} has ${size}`,
          "nSplits",
          this.nSplits
        );
      }
    }

    // Lay the samples out sorted by class (classes in order of first appearance) and deal
    // them round-robin to the folds. This gives every fold an almost equal share of each
    // class and keeps fold sizes within one of each other.
    const allocation: number[][] = Array.from({ length: this.nSplits }, () =>
      new Array<number>(nClasses).fill(0)
    );
    let position = 0;
    for (let k = 0; k < nClasses; k++) {
      const size = (indices[k] as number[]).length;
      for (let j = 0; j < size; j++, position++) {
        const row = allocation[position % this.nSplits] as number[];
        row[k] = (row[k] as number) + 1;
      }
    }

    const random = this.randomState !== undefined ? createRandomStream(this.randomState) : __random;
    const foldsForClass: number[][] = [];
    for (let k = 0; k < nClasses; k++) {
      const folds: number[] = [];
      for (let fold = 0; fold < this.nSplits; fold++) {
        const count = (allocation[fold] as number[])[k] as number;
        for (let c = 0; c < count; c++) folds.push(fold);
      }
      if (this.shuffle) shuffleIndicesInPlace(folds, random);
      foldsForClass.push(folds);
    }

    const foldOf = new Int32Array(nSamples);
    const seen = new Int32Array(nClasses);
    for (let i = 0; i < nSamples; i++) {
      const k = codes[i] as number;
      foldOf[i] = (foldsForClass[k] as number[])[seen[k] as number] as number;
      seen[k] = (seen[k] as number) + 1;
    }
    return splitsFromFolds(foldOf, this.nSplits);
  }

  /** Number of folds. */
  getNSplits(): number {
    return this.nSplits;
  }
}

/**
 * Group K-Fold cross-validator.
 *
 * Guarantees that no group appears in both the train and the test set of a
 * split. Without shuffling, groups are assigned largest first to the fold that
 * currently holds the fewest samples (the same greedy rule as scikit-learn), so
 * fold sizes stay as balanced as the group sizes allow. With `shuffle`, the
 * groups are permuted and cut into `nSplits` chunks of nearly equal group count.
 * Train and test indices are returned in ascending order.
 *
 * @example
 * ```ts
 * import { GroupKFold } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const gkf = new GroupKFold({ nSplits: 2 });
 * const X = tensor([[1], [2], [3], [4]]);
 * const splits = gkf.split(X, undefined, tensor([0, 0, 1, 1]));
 * ```
 *
 * @see {@link https://deepbox.dev/docs/preprocess-splitting | Deepbox Data Splitting}
 */
export class GroupKFold {
  private readonly nSplits: number;
  private readonly shuffle: boolean;
  private readonly randomState: number | undefined;

  /**
   * @param options.nSplits - Number of folds, an integer of at least 2 (default 5)
   * @param options.shuffle - Shuffle the groups before assigning folds (default `false`)
   * @param options.randomState - Non-negative integer seed used when `shuffle` is `true`
   */
  constructor(options: { nSplits?: number; shuffle?: boolean; randomState?: number } = {}) {
    this.nSplits = options.nSplits ?? 5;
    this.shuffle = options.shuffle ?? false;
    this.randomState = options.randomState;
    validateNSplits(this.nSplits);
    validateRandomState(this.randomState);
  }

  /**
   * Generate train/test indices for every fold.
   *
   * @param X - Data tensor; only the number of rows is used
   * @param _y - Unused, accepted for API symmetry with the other cross-validators
   * @param groups - Group label of each sample: a 1D tensor or an array of strings,
   *   numbers or bigints
   * @throws {InvalidParameterError} If lengths differ or there are fewer groups than `nSplits`
   * @throws {ShapeError} If `groups` is a tensor that is not 1D
   */
  split(X: Tensor, _y: Tensor | undefined, groups: Tensor | readonly Label[]): SplitResult[] {
    const nSamples = getNSamples(X);
    const groupLabels = readGroups(groups);
    if (groupLabels.length !== nSamples) {
      throw new InvalidParameterError(
        "X and groups must have same number of samples",
        "groups",
        groupLabels.length
      );
    }
    const { indices } = groupByLabelSorted(groupLabels);
    const nGroups = indices.length;
    if (this.nSplits > nGroups) {
      throw new InvalidParameterError(
        "Number of groups must be at least nSplits",
        "nSplits",
        this.nSplits
      );
    }

    const foldOfGroup = new Int32Array(nGroups);
    if (this.shuffle) {
      const random =
        this.randomState !== undefined ? createRandomStream(this.randomState) : __random;
      const order = range(nGroups);
      shuffleIndicesInPlace(order, random);
      const sizes = makeFoldSizes(nGroups, this.nSplits);
      let position = 0;
      for (let fold = 0; fold < this.nSplits; fold++) {
        const end = position + (sizes[fold] as number);
        for (; position < end; position++) foldOfGroup[order[position] as number] = fold;
      }
    } else {
      // Largest groups first; among equal sizes the later (larger) label goes first,
      // which is the order scikit-learn's reversed stable argsort produces.
      const order = range(nGroups).sort((a, b) => {
        const sizeDiff = (indices[b] as number[]).length - (indices[a] as number[]).length;
        return sizeDiff !== 0 ? sizeDiff : b - a;
      });
      const foldSizes = new Array<number>(this.nSplits).fill(0);
      for (const group of order) {
        let lightest = 0;
        for (let fold = 1; fold < this.nSplits; fold++) {
          if ((foldSizes[fold] as number) < (foldSizes[lightest] as number)) lightest = fold;
        }
        foldOfGroup[group] = lightest;
        foldSizes[lightest] = (foldSizes[lightest] as number) + (indices[group] as number[]).length;
      }
    }

    const foldOf = new Int32Array(nSamples);
    for (let g = 0; g < nGroups; g++) {
      for (const sample of indices[g] as number[]) foldOf[sample] = foldOfGroup[g] as number;
    }
    return splitsFromFolds(foldOf, this.nSplits);
  }

  /** Number of folds. */
  getNSplits(): number {
    return this.nSplits;
  }
}

// ---------------------------------------------------------------------------
// Leave-out cross-validators
// ---------------------------------------------------------------------------

/**
 * Leave-One-Out cross-validator.
 *
 * Each sample is used once as a test set of size one. This materialises
 * `n * (n - 1)` train indices, so very large `n` is rejected with a
 * `MemoryError`.
 *
 * @example
 * ```ts
 * import { LeaveOneOut } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const splits = new LeaveOneOut().split(tensor([[1], [2], [3]]));
 * // 3 splits, each with one test index
 * ```
 *
 * @see {@link https://deepbox.dev/docs/preprocess-splitting | Deepbox Data Splitting}
 */
export class LeaveOneOut {
  /**
   * Generate one split per sample.
   *
   * @param X - Data tensor; only the number of rows is used
   * @throws {InvalidParameterError} If `X` has fewer than 2 samples
   * @throws {MemoryError} If the split list would be too large to materialise
   */
  split(X: Tensor): SplitResult[] {
    const nSamples = getNSamples(X);
    if (nSamples < 2) {
      throw new InvalidParameterError("LeaveOneOut requires at least 2 samples", "X", nSamples);
    }
    assertMemoryBudget(nSamples * (nSamples - 1), "LeaveOneOut");
    const splits: SplitResult[] = [];

    for (let i = 0; i < nSamples; i++) {
      const trainIndex = new Array<number>(nSamples - 1);
      for (let j = 0; j < i; j++) trainIndex[j] = j;
      for (let j = i + 1; j < nSamples; j++) trainIndex[j - 1] = j;
      splits.push({ trainIndex, testIndex: [i] });
    }

    return splits;
  }

  /**
   * Number of splits, which equals the number of samples.
   *
   * @param X - Data tensor; only the number of rows is used
   */
  getNSplits(X: Tensor): number {
    return getNSamples(X);
  }
}

/** C(n, k) as a float; exact while the intermediate products stay below 2^53. */
function binomial(n: number, k: number): number {
  const m = k > n / 2 ? n - k : k;
  let result = 1;
  for (let i = 0; i < m; i++) {
    result = (result * (n - i)) / (i + 1);
  }
  return result;
}

/**
 * Leave-P-Out cross-validator.
 *
 * Every combination of `p` samples is used once as the test set. The number of
 * splits is `C(n, p)`, so the splits are only materialised when they number at
 * most 100,000 and their indices fit a memory budget.
 *
 * @example
 * ```ts
 * import { LeavePOut } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const lpo = new LeavePOut(2);
 * const splits = lpo.split(tensor([[1], [2], [3], [4]])); // C(4, 2) = 6 splits
 * ```
 *
 * @see {@link https://deepbox.dev/docs/preprocess-splitting | Deepbox Data Splitting}
 */
export class LeavePOut {
  private readonly p: number;

  /** @param p - Number of samples left out in each split (positive integer) */
  constructor(p: number) {
    if (!Number.isInteger(p) || p <= 0) {
      throw new InvalidParameterError("p must be a positive integer", "p", p);
    }
    this.p = p;
  }

  private checkSampleCount(nSamples: number): void {
    if (this.p > nSamples) {
      throw new InvalidParameterError("p must not be greater than number of samples", "p", this.p);
    }
    if (this.p === nSamples) {
      throw new InvalidParameterError(
        "p must be less than the number of samples so that the train set is not empty",
        "p",
        this.p
      );
    }
  }

  /**
   * Generate train/test indices for every combination of `p` test samples,
   * in lexicographic order of the test indices.
   *
   * @param X - Data tensor; only the number of rows is used
   * @throws {InvalidParameterError} If `p` is not smaller than the number of samples, or
   *   `C(n, p)` exceeds 100,000
   * @throws {MemoryError} If the indices would not fit the memory budget
   */
  split(X: Tensor): SplitResult[] {
    const nSamples = getNSamples(X);
    this.checkSampleCount(nSamples);
    const p = this.p;

    const nCombos = binomial(nSamples, p);
    if (nCombos > MAX_LEAVE_P_OUT_SPLITS) {
      throw new InvalidParameterError(
        `LeavePOut produces ${Number.isFinite(nCombos) ? Math.floor(nCombos) : "more than 1e308"} ` +
          `splits, which exceeds memory safety limit of ${MAX_LEAVE_P_OUT_SPLITS}`,
        "p",
        p
      );
    }
    assertMemoryBudget(nCombos * nSamples, "LeavePOut");

    const splits: SplitResult[] = [];
    const combo = range(p);
    const isTest = new Uint8Array(nSamples);

    for (;;) {
      isTest.fill(0);
      for (const idx of combo) isTest[idx] = 1;
      const trainIndex: number[] = [];
      for (let i = 0; i < nSamples; i++) {
        if (isTest[i] === 0) trainIndex.push(i);
      }
      splits.push({ trainIndex, testIndex: [...combo] });

      // Advance to the next combination in lexicographic order.
      let pos = p - 1;
      while (pos >= 0 && (combo[pos] as number) === nSamples - p + pos) pos--;
      if (pos < 0) break;
      combo[pos] = (combo[pos] as number) + 1;
      for (let j = pos + 1; j < p; j++) combo[j] = (combo[j - 1] as number) + 1;
    }

    return splits;
  }

  /**
   * Number of splits, `C(n, p)`.
   *
   * @param X - Data tensor; only the number of rows is used
   * @throws {InvalidParameterError} If `p` is not smaller than the number of samples
   */
  getNSplits(X: Tensor): number {
    const nSamples = getNSamples(X);
    this.checkSampleCount(nSamples);
    return Math.round(binomial(nSamples, this.p));
  }
}

/** Read the groups argument of a group splitter and check it against `X`. */
function readSampleGroups(X: Tensor, groups: Tensor | readonly Label[]): LabelGroups {
  const nSamples = getNSamples(X);
  const groupLabels = readGroups(groups);
  if (groupLabels.length !== nSamples) {
    throw new InvalidParameterError(
      "X and groups must have same number of samples",
      "groups",
      groupLabels.length
    );
  }
  return groupByLabelSorted(groupLabels);
}

/** Train/test split whose test set is the union of the given groups. */
function splitFromGroupSet(
  nSamples: number,
  groupIndices: readonly (readonly number[])[],
  testGroups: readonly number[]
): SplitResult {
  const isTest = new Uint8Array(nSamples);
  for (const g of testGroups) {
    for (const sample of groupIndices[g] as readonly number[]) isTest[sample] = 1;
  }
  const trainIndex: number[] = [];
  const testIndex: number[] = [];
  for (let i = 0; i < nSamples; i++) {
    if (isTest[i] === 1) testIndex.push(i);
    else trainIndex.push(i);
  }
  return { trainIndex, testIndex };
}

/**
 * Leave-One-Group-Out cross-validator.
 *
 * Each distinct group is used once as the test set, in ascending order of the group
 * labels, and the other groups form the train set. At least two groups are required.
 * Train and test indices are returned in ascending order.
 *
 * @example
 * ```ts
 * import { LeaveOneGroupOut } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1], [2], [3], [4]]);
 * const splits = new LeaveOneGroupOut().split(X, undefined, tensor([0, 0, 1, 2]));
 * // 3 splits: test=[0,1], test=[2], test=[3]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/preprocess-splitting | Deepbox Data Splitting}
 */
export class LeaveOneGroupOut {
  /**
   * Generate one split per group.
   *
   * @param X - Data tensor; only the number of rows is used
   * @param _y - Unused, accepted for API symmetry with the other cross-validators
   * @param groups - Group label of each sample: a 1D tensor or an array of strings,
   *   numbers or bigints
   * @throws {InvalidParameterError} If lengths differ or there are fewer than 2 groups
   * @throws {ShapeError} If `groups` is a tensor that is not 1D
   */
  split(X: Tensor, _y: Tensor | undefined, groups: Tensor | readonly Label[]): SplitResult[] {
    const nSamples = getNSamples(X);
    const { indices } = readSampleGroups(X, groups);
    if (indices.length < 2) {
      throw new InvalidParameterError(
        "LeaveOneGroupOut requires at least 2 distinct groups",
        "groups",
        indices.length
      );
    }
    assertMemoryBudget(nSamples * indices.length, "LeaveOneGroupOut");
    return indices.map((_, g) => splitFromGroupSet(nSamples, indices, [g]));
  }

  /**
   * Number of splits, which equals the number of distinct groups.
   *
   * @param groups - Group label of each sample
   * @throws {InvalidParameterError} If there are fewer than 2 groups
   */
  getNSplits(groups: Tensor | readonly Label[]): number {
    const n = groupByLabel(readGroups(groups)).labels.length;
    if (n < 2) {
      throw new InvalidParameterError(
        "LeaveOneGroupOut requires at least 2 distinct groups",
        "groups",
        n
      );
    }
    return n;
  }
}

/**
 * Leave-P-Groups-Out cross-validator.
 *
 * Every combination of `nGroups` distinct groups is used once as the test set, in
 * lexicographic order of the ascending group labels. The number of splits is
 * `C(g, nGroups)` for `g` distinct groups, so the splits are only materialised when they
 * number at most 100,000 and their indices fit a memory budget. Train and test indices
 * are returned in ascending order.
 *
 * @example
 * ```ts
 * import { LeavePGroupsOut } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1], [2], [3], [4], [5]]);
 * const splits = new LeavePGroupsOut(2).split(X, undefined, tensor([0, 0, 1, 1, 2]));
 * // C(3, 2) = 3 splits
 * ```
 *
 * @see {@link https://deepbox.dev/docs/preprocess-splitting | Deepbox Data Splitting}
 */
export class LeavePGroupsOut {
  private readonly nGroups: number;

  /** @param nGroups - Number of groups left out in each split (positive integer) */
  constructor(nGroups: number) {
    if (!Number.isInteger(nGroups) || nGroups <= 0) {
      throw new InvalidParameterError("nGroups must be a positive integer", "nGroups", nGroups);
    }
    this.nGroups = nGroups;
  }

  private checkGroupCount(distinct: number): void {
    if (this.nGroups >= distinct) {
      throw new InvalidParameterError(
        `nGroups must be smaller than the number of distinct groups (${distinct}) so that the train set is not empty`,
        "nGroups",
        this.nGroups
      );
    }
  }

  /**
   * Generate train/test indices for every combination of `nGroups` test groups.
   *
   * @param X - Data tensor; only the number of rows is used
   * @param _y - Unused, accepted for API symmetry with the other cross-validators
   * @param groups - Group label of each sample: a 1D tensor or an array of strings,
   *   numbers or bigints
   * @throws {InvalidParameterError} If lengths differ, `nGroups` is not smaller than the
   *   number of distinct groups, or the number of splits exceeds 100,000
   * @throws {MemoryError} If the indices would not fit the memory budget
   */
  split(X: Tensor, _y: Tensor | undefined, groups: Tensor | readonly Label[]): SplitResult[] {
    const nSamples = getNSamples(X);
    const { indices } = readSampleGroups(X, groups);
    const distinct = indices.length;
    this.checkGroupCount(distinct);
    const p = this.nGroups;

    const nCombos = binomial(distinct, p);
    if (nCombos > MAX_LEAVE_P_OUT_SPLITS) {
      throw new InvalidParameterError(
        `LeavePGroupsOut produces ${Number.isFinite(nCombos) ? Math.floor(nCombos) : "more than 1e308"} ` +
          `splits, which exceeds memory safety limit of ${MAX_LEAVE_P_OUT_SPLITS}`,
        "nGroups",
        p
      );
    }
    assertMemoryBudget(nCombos * nSamples, "LeavePGroupsOut");

    const splits: SplitResult[] = [];
    const combo = range(p);
    for (;;) {
      splits.push(splitFromGroupSet(nSamples, indices, combo));
      let pos = p - 1;
      while (pos >= 0 && (combo[pos] as number) === distinct - p + pos) pos--;
      if (pos < 0) break;
      combo[pos] = (combo[pos] as number) + 1;
      for (let j = pos + 1; j < p; j++) combo[j] = (combo[j - 1] as number) + 1;
    }
    return splits;
  }

  /**
   * Number of splits, `C(g, nGroups)` for `g` distinct groups.
   *
   * @param groups - Group label of each sample
   * @throws {InvalidParameterError} If `nGroups` is not smaller than the number of groups
   */
  getNSplits(groups: Tensor | readonly Label[]): number {
    const distinct = groupByLabel(readGroups(groups)).labels.length;
    this.checkGroupCount(distinct);
    return Math.round(binomial(distinct, this.nGroups));
  }
}

/**
 * Predefined split cross-validator.
 *
 * `testFold[i]` names the fold in which sample `i` is a test sample. Samples with
 * `testFold[i] === -1` are never part of a test set (they are always in the train set).
 * One split is produced for each distinct fold value other than -1, in ascending order.
 *
 * @example
 * ```ts
 * import { PredefinedSplit } from 'deepbox/preprocess';
 *
 * const ps = new PredefinedSplit([0, 0, 1, 1, -1]);
 * const splits = ps.split();
 * // Split 0: test=[0,1], train=[2,3,4]
 * // Split 1: test=[2,3], train=[0,1,4]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/preprocess-splitting | Deepbox Data Splitting}
 */
export class PredefinedSplit {
  private readonly testFold: Int32Array;
  private readonly folds: number[];

  /**
   * @param testFold - Fold number of each sample (integers, -1 for "never test"): a 1D
   *   tensor or an array of numbers
   * @throws {InvalidParameterError} If a value is not an integer or no sample is in a fold
   * @throws {ShapeError} If `testFold` is a tensor that is not 1D
   */
  constructor(testFold: Tensor | readonly number[]) {
    const raw: readonly unknown[] = Array.isArray(testFold)
      ? testFold
      : readLabels(testFold as Tensor, "testFold");
    const values = new Int32Array(raw.length);
    for (let i = 0; i < raw.length; i++) {
      const v = Number(raw[i]);
      if (!Number.isInteger(v) || v < -1 || v > 2 ** 31 - 1) {
        throw new InvalidParameterError(
          `testFold must contain integers of at least -1; element ${i} is ${String(raw[i])}`,
          "testFold",
          raw[i]
        );
      }
      values[i] = v;
    }
    this.testFold = values;
    this.folds = [...new Set(values)].filter((f) => f !== -1).sort(ascending);
    if (this.folds.length === 0) {
      throw new InvalidParameterError(
        "testFold must assign at least one sample to a fold (a value other than -1)",
        "testFold",
        raw.length
      );
    }
  }

  /**
   * Generate one split per distinct fold value other than -1.
   *
   * @param X - Optional data tensor; when given, its number of rows must equal the
   *   length of `testFold`
   * @throws {InvalidParameterError} If `X` has a different number of rows
   */
  split(X?: Tensor): SplitResult[] {
    const n = this.testFold.length;
    if (X !== undefined && getNSamples(X) !== n) {
      throw new InvalidParameterError(
        "X and testFold must have same number of samples",
        "testFold",
        n
      );
    }
    assertMemoryBudget(n * this.folds.length, "PredefinedSplit");
    return this.folds.map((fold) => {
      const trainIndex: number[] = [];
      const testIndex: number[] = [];
      for (let i = 0; i < n; i++) {
        if (this.testFold[i] === fold) testIndex.push(i);
        else trainIndex.push(i);
      }
      return { trainIndex, testIndex };
    });
  }

  /** Number of splits, the number of distinct fold values other than -1. */
  getNSplits(): number {
    return this.folds.length;
  }
}

/** Population standard deviation (`np.std`). */
function populationStd(values: ArrayLike<number>): number {
  const n = values.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += values[i] as number;
  mean /= n;
  let acc = 0;
  for (let i = 0; i < n; i++) {
    const d = (values[i] as number) - mean;
    acc += d * d;
  }
  return Math.sqrt(acc / n);
}

/**
 * Stratified Group K-Fold cross-validator.
 *
 * Keeps every group in a single fold (like {@link GroupKFold}) while trying to keep the
 * class distribution of each fold close to the overall one (like {@link StratifiedKFold}).
 * Without shuffling it follows scikit-learn's greedy algorithm: groups are visited from the
 * most to the least unevenly distributed over classes and each goes to the fold whose class
 * proportions end up most even, ties going to the fold with fewer samples. The result is
 * deterministic and matches scikit-learn. With `shuffle`, the groups are visited in a random
 * order before that sort, so equally uneven groups are placed in a seeded random order
 * (scikit-learn's shuffled folds depend on NumPy's generator and are not reproduced).
 * Train and test indices are returned in ascending order.
 *
 * @example
 * ```ts
 * import { StratifiedGroupKFold } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1], [2], [3], [4], [5], [6]]);
 * const y = tensor([0, 0, 1, 1, 0, 1]);
 * const groups = tensor([0, 0, 1, 1, 2, 3]);
 * const splits = new StratifiedGroupKFold({ nSplits: 2 }).split(X, y, groups);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/preprocess-splitting | Deepbox Data Splitting}
 */
export class StratifiedGroupKFold {
  private readonly nSplits: number;
  private readonly shuffle: boolean;
  private readonly randomState: number | undefined;

  /**
   * @param options.nSplits - Number of folds, an integer of at least 2 (default 5)
   * @param options.shuffle - Visit the groups in random order (default `false`)
   * @param options.randomState - Non-negative integer seed used when `shuffle` is `true`
   */
  constructor(options: { nSplits?: number; shuffle?: boolean; randomState?: number } = {}) {
    this.nSplits = options.nSplits ?? 5;
    this.shuffle = options.shuffle ?? false;
    this.randomState = options.randomState;
    validateNSplits(this.nSplits);
    validateRandomState(this.randomState);
  }

  /**
   * Generate train/test indices for every fold.
   *
   * @param X - Data tensor; only the number of rows is used
   * @param y - 1D class labels (numbers, bigints or strings), one per sample
   * @param groups - Group label of each sample: a 1D tensor or an array of strings,
   *   numbers or bigints
   * @throws {InvalidParameterError} If lengths differ, there are fewer groups than `nSplits`,
   *   or every class has fewer than `nSplits` samples
   * @throws {ShapeError} If `y` or `groups` is a tensor that is not 1D
   */
  split(X: Tensor, y: Tensor, groups: Tensor | readonly Label[]): SplitResult[] {
    const nSamples = getNSamples(X);
    const yShape0 = y.shape[0];
    if (yShape0 === undefined || yShape0 !== nSamples) {
      throw new InvalidParameterError("X and y must have same number of samples", "y", yShape0);
    }
    // Sorted classes (like np.unique) so that float sums run in the same order as scikit-learn.
    const classes = groupByLabelSorted(readLabels(y, "y"));
    const nClasses = classes.labels.length;
    const grouped = readSampleGroups(X, groups);
    const nGroups = grouped.indices.length;
    if (this.nSplits > nGroups) {
      throw new InvalidParameterError(
        "Number of groups must be at least nSplits",
        "nSplits",
        this.nSplits
      );
    }
    const classTotals = classes.indices.map((ix) => ix.length);
    if (classTotals.every((count) => this.nSplits > count)) {
      throw new InvalidParameterError(
        "nSplits cannot be greater than the number of members in each class",
        "nSplits",
        this.nSplits
      );
    }

    // Class counts of every group.
    const groupCounts: Float64Array[] = Array.from(
      { length: nGroups },
      () => new Float64Array(nClasses)
    );
    for (let i = 0; i < nSamples; i++) {
      const row = groupCounts[grouped.codes[i] as number] as Float64Array;
      row[classes.codes[i] as number] = (row[classes.codes[i] as number] as number) + 1;
    }

    let order = range(nGroups);
    if (this.shuffle) {
      const random =
        this.randomState !== undefined ? createRandomStream(this.randomState) : __random;
      shuffleIndicesInPlace(order, random);
    }
    // Stable sort: the most unevenly distributed groups first.
    const spread = groupCounts.map((row) => populationStd(row));
    order = order.sort((a, b) => (spread[b] as number) - (spread[a] as number));

    const foldCounts: Float64Array[] = Array.from(
      { length: this.nSplits },
      () => new Float64Array(nClasses)
    );
    const foldTotals = new Float64Array(this.nSplits);
    const foldOfGroup = new Int32Array(nGroups);
    const ratio = new Float64Array(this.nSplits);
    const column = new Float64Array(this.nSplits);
    for (const group of order) {
      const counts = groupCounts[group] as Float64Array;
      let best = 0;
      let minEval = Number.POSITIVE_INFINITY;
      let minSamples = Number.POSITIVE_INFINITY;
      for (let fold = 0; fold < this.nSplits; fold++) {
        // Spread of each class share across the folds if the group joined this fold.
        let evalSum = 0;
        for (let k = 0; k < nClasses; k++) {
          for (let f = 0; f < this.nSplits; f++) {
            const value = (foldCounts[f] as Float64Array)[k] as number;
            ratio[f] =
              (f === fold ? value + (counts[k] as number) : value) / (classTotals[k] as number);
          }
          column.set(ratio);
          evalSum += populationStd(column);
        }
        const foldEval = evalSum / nClasses;
        const samples = foldTotals[fold] as number;
        // Same tolerance as numpy.isclose(foldEval, minEval).
        const close = Math.abs(foldEval - minEval) <= 1e-8 + 1e-5 * Math.abs(minEval);
        if (foldEval < minEval || (close && samples < minSamples)) {
          best = fold;
          minEval = foldEval;
          minSamples = samples;
        }
      }
      const target = foldCounts[best] as Float64Array;
      let added = 0;
      for (let k = 0; k < nClasses; k++) {
        target[k] = (target[k] as number) + (counts[k] as number);
        added += counts[k] as number;
      }
      foldTotals[best] = (foldTotals[best] as number) + added;
      foldOfGroup[group] = best;
    }

    const foldOf = new Int32Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      foldOf[i] = foldOfGroup[grouped.codes[i] as number] as number;
    }
    return splitsFromFolds(foldOf, this.nSplits);
  }

  /** Number of folds. */
  getNSplits(): number {
    return this.nSplits;
  }
}

// ---------------------------------------------------------------------------
// Time series
// ---------------------------------------------------------------------------

/**
 * Time Series cross-validator.
 *
 * Provides train/test indices for time series data. In each split, test
 * indices must be higher than before, and thus shuffling in cross validator is
 * inappropriate. The test sets are the last `nSplits` blocks of `testSize`
 * samples; each train set is everything before its test block, minus `gap`
 * samples, optionally capped to the most recent `maxTrainSize` samples.
 *
 * @example
 * ```ts
 * import { TimeSeriesSplit } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const tscv = new TimeSeriesSplit({ nSplits: 3 });
 * const X = tensor([[1], [2], [3], [4], [5], [6]]);
 * const splits = tscv.split(X);
 * // Split 0: train=[0,1,2], test=[3]
 * // Split 1: train=[0,1,2,3], test=[4]
 * // Split 2: train=[0,1,2,3,4], test=[5]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/preprocess-splitting | Deepbox Data Splitting}
 */
export class TimeSeriesSplit {
  private readonly nSplits: number;
  private readonly maxTrainSize: number | undefined;
  private readonly testSize: number | undefined;
  private readonly gap: number;

  /**
   * @param options.nSplits - Number of splits, an integer of at least 2 (default 5)
   * @param options.maxTrainSize - Cap on the train set size (positive integer)
   * @param options.testSize - Samples per test block (positive integer);
   *   defaults to `floor(n / (nSplits + 1))`
   * @param options.gap - Samples excluded between each train set and its test block (default 0)
   */
  constructor(
    options: {
      nSplits?: number;
      maxTrainSize?: number;
      testSize?: number;
      gap?: number;
    } = {}
  ) {
    this.nSplits = options.nSplits ?? 5;
    this.maxTrainSize = options.maxTrainSize;
    this.testSize = options.testSize;
    this.gap = options.gap ?? 0;

    validateNSplits(this.nSplits);
    if (!Number.isInteger(this.gap) || this.gap < 0) {
      throw new InvalidParameterError("gap must be a non-negative integer", "gap", this.gap);
    }
    if (
      this.maxTrainSize !== undefined &&
      (!Number.isInteger(this.maxTrainSize) || this.maxTrainSize < 1)
    ) {
      throw new InvalidParameterError(
        "maxTrainSize must be a positive integer",
        "maxTrainSize",
        this.maxTrainSize
      );
    }
    if (this.testSize !== undefined && (!Number.isInteger(this.testSize) || this.testSize < 1)) {
      throw new InvalidParameterError(
        "testSize must be a positive integer",
        "testSize",
        this.testSize
      );
    }
  }

  /**
   * Generate the forward-chaining splits, oldest first.
   *
   * @param X - Data tensor; only the number of rows is used
   * @throws {InvalidParameterError} If there are too few samples for `nSplits`, `testSize`
   *   and `gap`
   */
  split(X: Tensor): SplitResult[] {
    const nSamples = getNSamples(X);
    const nSplits = this.nSplits;
    const nFolds = nSplits + 1;

    if (nFolds > nSamples) {
      throw new InvalidParameterError(
        `Cannot have nSplits+1=${nFolds} folds with ${nSamples} samples`,
        "nSplits",
        nSplits
      );
    }

    const testSz = this.testSize ?? Math.floor(nSamples / nFolds);
    if (nSamples - this.gap - testSz * nSplits <= 0) {
      throw new InvalidParameterError(
        `Too many splits=${nSplits} for number of samples=${nSamples} with testSize=${testSz} ` +
          `and gap=${this.gap}`,
        "nSplits",
        nSplits
      );
    }

    const splits: SplitResult[] = [];
    const firstTestStart = nSamples - nSplits * testSz;

    for (let i = 0; i < nSplits; i++) {
      const testStart = firstTestStart + i * testSz;
      const trainEnd = testStart - this.gap;
      const trainStart =
        this.maxTrainSize !== undefined && trainEnd > this.maxTrainSize
          ? trainEnd - this.maxTrainSize
          : 0;

      const trainIndex: number[] = [];
      for (let j = trainStart; j < trainEnd; j++) trainIndex.push(j);

      const testIndex: number[] = [];
      for (let j = testStart; j < testStart + testSz; j++) testIndex.push(j);

      splits.push({ trainIndex, testIndex });
    }

    return splits;
  }

  /** Number of splits. */
  getNSplits(): number {
    return this.nSplits;
  }
}

// ---------------------------------------------------------------------------
// Repeated cross-validators
// ---------------------------------------------------------------------------

/**
 * Repeated K-Fold cross-validator.
 *
 * Repeats K-Fold `nRepeats` times with different shuffling in each repetition.
 * With a `randomState`, repetition `r` shuffles with a seed derived from `randomState` and `r`
 * (a hash, so that neighbouring seeds do not share repetitions).
 *
 * @example
 * ```ts
 * import { RepeatedKFold } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const rkf = new RepeatedKFold({ nSplits: 5, nRepeats: 3 });
 * const X = tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]);
 * const splits = rkf.split(X); // 15 splits (5 folds × 3 repeats)
 * ```
 *
 * @see {@link https://deepbox.dev/docs/preprocess-splitting | Deepbox Data Splitting}
 */
export class RepeatedKFold {
  private readonly nSplits: number;
  private readonly nRepeats: number;
  private readonly randomState: number | undefined;

  /**
   * @param options.nSplits - Folds per repetition, an integer of at least 2 (default 5)
   * @param options.nRepeats - Number of repetitions, a positive integer (default 10)
   * @param options.randomState - Non-negative integer seed
   */
  constructor(
    options: {
      nSplits?: number;
      nRepeats?: number;
      randomState?: number;
    } = {}
  ) {
    this.nSplits = options.nSplits ?? 5;
    this.nRepeats = options.nRepeats ?? 10;
    this.randomState = options.randomState;

    validateNSplits(this.nSplits);
    if (!Number.isInteger(this.nRepeats) || this.nRepeats < 1) {
      throw new InvalidParameterError(
        "nRepeats must be a positive integer",
        "nRepeats",
        this.nRepeats
      );
    }
    validateRandomState(this.randomState);
  }

  /**
   * Generate `nSplits * nRepeats` splits, repetition by repetition.
   *
   * @param X - Data tensor; only the number of rows is used
   */
  split(X: Tensor): SplitResult[] {
    const allSplits: SplitResult[] = [];

    for (let r = 0; r < this.nRepeats; r++) {
      const opts: { nSplits: number; shuffle: boolean; randomState?: number } = {
        nSplits: this.nSplits,
        shuffle: true,
      };
      if (this.randomState !== undefined) {
        opts.randomState = deriveSeed(this.randomState, r);
      }
      for (const fold of new KFold(opts).split(X)) allSplits.push(fold);
    }

    return allSplits;
  }

  /** Total number of splits, `nSplits * nRepeats`. */
  getNSplits(): number {
    return this.nSplits * this.nRepeats;
  }
}

/**
 * Repeated Stratified K-Fold cross-validator.
 *
 * Repeats Stratified K-Fold `nRepeats` times with different shuffling. With a
 * `randomState`, repetition `r` shuffles with a seed derived from `randomState` and `r`
 * (a hash, so that neighbouring seeds do not share repetitions).
 *
 * @example
 * ```ts
 * import { RepeatedStratifiedKFold } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const rskf = new RepeatedStratifiedKFold({ nSplits: 2, nRepeats: 3, randomState: 0 });
 * const splits = rskf.split(tensor([[1], [2], [3], [4]]), tensor([0, 0, 1, 1]));
 * ```
 *
 * @see {@link https://deepbox.dev/docs/preprocess-splitting | Deepbox Data Splitting}
 */
export class RepeatedStratifiedKFold {
  private readonly nSplits: number;
  private readonly nRepeats: number;
  private readonly randomState: number | undefined;

  /**
   * @param options.nSplits - Folds per repetition, an integer of at least 2 (default 5)
   * @param options.nRepeats - Number of repetitions, a positive integer (default 10)
   * @param options.randomState - Non-negative integer seed
   */
  constructor(
    options: {
      nSplits?: number;
      nRepeats?: number;
      randomState?: number;
    } = {}
  ) {
    this.nSplits = options.nSplits ?? 5;
    this.nRepeats = options.nRepeats ?? 10;
    this.randomState = options.randomState;

    validateNSplits(this.nSplits);
    if (!Number.isInteger(this.nRepeats) || this.nRepeats < 1) {
      throw new InvalidParameterError(
        "nRepeats must be a positive integer",
        "nRepeats",
        this.nRepeats
      );
    }
    validateRandomState(this.randomState);
  }

  /**
   * Generate `nSplits * nRepeats` stratified splits, repetition by repetition.
   *
   * @param X - Data tensor; only the number of rows is used
   * @param y - 1D class labels, one per sample
   */
  split(X: Tensor, y: Tensor): SplitResult[] {
    const allSplits: SplitResult[] = [];

    for (let r = 0; r < this.nRepeats; r++) {
      const opts: { nSplits: number; shuffle: boolean; randomState?: number } = {
        nSplits: this.nSplits,
        shuffle: true,
      };
      if (this.randomState !== undefined) {
        opts.randomState = deriveSeed(this.randomState, r);
      }
      for (const fold of new StratifiedKFold(opts).split(X, y)) allSplits.push(fold);
    }

    return allSplits;
  }

  /** Total number of splits, `nSplits * nRepeats`. */
  getNSplits(): number {
    return this.nSplits * this.nRepeats;
  }
}

// ---------------------------------------------------------------------------
// Shuffle-split family
// ---------------------------------------------------------------------------

/**
 * Random permutation cross-validator.
 *
 * Yields indices to split data into train and test sets using random
 * permutations. Unlike KFold, ShuffleSplit allows controlling the
 * train/test size independently and produces overlapping test sets
 * across iterations. With neither `trainSize` nor `testSize` given, the test
 * set gets 10% of the samples (scikit-learn's default for this splitter).
 * Iteration `i` shuffles with a seed derived from `randomState` and `i`
 * (a hash, so that neighbouring seeds do not share iterations).
 *
 * @example
 * ```ts
 * import { ShuffleSplit } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const ss = new ShuffleSplit({ nSplits: 5, testSize: 0.2, randomState: 42 });
 * const X = tensor([[1,2],[3,4],[5,6],[7,8],[9,10]]);
 * const splits = ss.split(X);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/preprocess-splitting | Deepbox Data Splitting}
 */
export class ShuffleSplit {
  private readonly nSplits: number;
  private readonly testSize: number | undefined;
  private readonly trainSize: number | undefined;
  private readonly randomState: number | undefined;

  /**
   * @param options.nSplits - Number of re-shuffling iterations (default 10)
   * @param options.testSize - Fraction (in (0, 1)) or absolute number (integer >= 1) of test
   *   samples
   * @param options.trainSize - Fraction (in (0, 1)) or absolute number (integer >= 1) of train
   *   samples
   * @param options.randomState - Non-negative integer seed
   */
  constructor(
    options: {
      nSplits?: number;
      testSize?: number;
      trainSize?: number;
      randomState?: number;
    } = {}
  ) {
    this.nSplits = options.nSplits ?? 10;
    this.testSize = options.testSize;
    this.trainSize = options.trainSize;
    this.randomState = options.randomState;

    validatePositiveSplitCount(this.nSplits);
    parseSplitSpec(this.testSize, "testSize");
    parseSplitSpec(this.trainSize, "trainSize");
    validateRandomState(this.randomState);
  }

  /**
   * Generate `nSplits` random train/test splits.
   *
   * @param X - Data tensor; only the number of rows is used
   * @throws {InvalidParameterError} If the sizes cannot be satisfied by the number of samples
   */
  split(X: Tensor): SplitResult[] {
    const nSamples = getNSamples(X);

    const [nTrain, nTest] = resolveTrainTestCounts(nSamples, this.trainSize, this.testSize, 0.1);

    const splits: SplitResult[] = [];

    for (let i = 0; i < this.nSplits; i++) {
      const random =
        this.randomState !== undefined ? createRandomStream(this.randomState, i) : __random;

      const indices = range(nSamples);
      shuffleIndicesInPlace(indices, random);

      const testIndex = indices.slice(0, nTest);
      const trainIndex = indices.slice(nTest, nTest + nTrain);
      splits.push({ trainIndex, testIndex });
    }

    return splits;
  }

  /** Number of re-shuffling iterations. */
  getNSplits(): number {
    return this.nSplits;
  }
}

/**
 * Stratified ShuffleSplit cross-validator.
 *
 * Provides train/test indices that preserve the percentage of samples
 * for each class. Like ShuffleSplit but with stratification: every split
 * contains exactly the requested number of train and test samples, divided
 * among the classes by largest remainder rounding. Every class needs at least
 * 2 samples, and both sets must be at least as large as the number of classes.
 * With neither `trainSize` nor `testSize` given, the test set gets 10% of the
 * samples. Iteration `i` shuffles with a seed derived from `randomState` and `i`
 * (a hash, so that neighbouring seeds do not share iterations).
 *
 * @example
 * ```ts
 * import { StratifiedShuffleSplit } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const sss = new StratifiedShuffleSplit({ nSplits: 5, testSize: 0.2, randomState: 42 });
 * const X = tensor([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]);
 * const y = tensor([0,0,0,0,0,1,1,1,1,1]);
 * const splits = sss.split(X, y);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/preprocess-splitting | Deepbox Data Splitting}
 */
export class StratifiedShuffleSplit {
  private readonly nSplits: number;
  private readonly testSize: number | undefined;
  private readonly trainSize: number | undefined;
  private readonly randomState: number | undefined;

  /**
   * @param options.nSplits - Number of re-shuffling iterations (default 10)
   * @param options.testSize - Fraction (in (0, 1)) or absolute number (integer >= 1) of test
   *   samples
   * @param options.trainSize - Fraction (in (0, 1)) or absolute number (integer >= 1) of train
   *   samples
   * @param options.randomState - Non-negative integer seed
   */
  constructor(
    options: {
      nSplits?: number;
      testSize?: number;
      trainSize?: number;
      randomState?: number;
    } = {}
  ) {
    this.nSplits = options.nSplits ?? 10;
    this.testSize = options.testSize;
    this.trainSize = options.trainSize;
    this.randomState = options.randomState;

    validatePositiveSplitCount(this.nSplits);
    parseSplitSpec(this.testSize, "testSize");
    parseSplitSpec(this.trainSize, "trainSize");
    validateRandomState(this.randomState);
  }

  /**
   * Generate `nSplits` stratified random train/test splits.
   *
   * @param X - Data tensor; only the number of rows is used
   * @param y - 1D class labels (numbers, bigints or strings), one per sample
   * @throws {InvalidParameterError} If lengths differ, a class has fewer than 2 samples, or the
   *   sizes are smaller than the number of classes
   * @throws {ShapeError} If `y` is not 1D
   */
  split(X: Tensor, y: Tensor): SplitResult[] {
    const nSamples = getNSamples(X);

    const yShape0 = y.shape[0];
    if (yShape0 === undefined || yShape0 !== nSamples) {
      throw new InvalidParameterError("X and y must have same number of samples", "y", yShape0);
    }

    const [nTrain, nTest] = resolveTrainTestCounts(nSamples, this.trainSize, this.testSize, 0.1);

    const { indices: classIndices } = groupByLabelSorted(readLabels(y, "y"));
    const splits: SplitResult[] = [];

    for (let iter = 0; iter < this.nSplits; iter++) {
      const random =
        this.randomState !== undefined ? createRandomStream(this.randomState, iter) : __random;
      // Per-class counts are drawn for every iteration so that ties between equal remainders
      // are not always resolved the same way (scikit-learn does the same).
      const { trainCounts, testCounts } = stratifiedCounts(
        classIndices,
        nTrain,
        nTest,
        random,
        "y"
      );

      const trainIndex: number[] = [];
      const testIndex: number[] = [];

      for (let k = 0; k < classIndices.length; k++) {
        const members = [...(classIndices[k] as number[])];
        shuffleIndicesInPlace(members, random);
        const nTrainK = trainCounts[k] as number;
        appendRange(trainIndex, members, 0, nTrainK);
        appendRange(testIndex, members, nTrainK, nTrainK + (testCounts[k] as number));
      }

      shuffleIndicesInPlace(trainIndex, random);
      shuffleIndicesInPlace(testIndex, random);
      splits.push({ trainIndex, testIndex });
    }

    return splits;
  }

  /** Number of re-shuffling iterations. */
  getNSplits(): number {
    return this.nSplits;
  }
}

/**
 * Shuffle-Group(s)-Out cross-validation iterator.
 *
 * Provides randomized train/test indices to split data by groups.
 * Ensures that the same group is not in both test and train sets.
 * `testSize` and `trainSize` count groups, not samples: fractions are taken of
 * the number of distinct groups, integers are absolute group counts. With
 * neither given, 20% of the groups form the test set. Train and test indices
 * are returned in ascending order. Iteration `i` shuffles with a seed derived from
 * `randomState` and `i` (a hash, so that neighbouring seeds do not share iterations).
 *
 * @example
 * ```ts
 * import { GroupShuffleSplit } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const gss = new GroupShuffleSplit({ nSplits: 5, testSize: 0.2, randomState: 42 });
 * const X = tensor([[1], [2], [3], [4], [5], [6]]);
 * const splits = gss.split(X, undefined, [0, 0, 1, 1, 2, 2]);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/preprocess-splitting | Deepbox Data Splitting}
 * @category Cross-Validation
 */
export class GroupShuffleSplit {
  private readonly nSplits: number;
  private readonly testSize: number | undefined;
  private readonly trainSize: number | undefined;
  private readonly randomState: number | undefined;

  /**
   * @param options.nSplits - Number of re-shuffling iterations (default 5)
   * @param options.testSize - Fraction (in (0, 1)) of groups or absolute number of groups
   *   (integer >= 1) in the test set
   * @param options.trainSize - Fraction (in (0, 1)) of groups or absolute number of groups
   *   (integer >= 1) in the train set
   * @param options.randomState - Non-negative integer seed
   */
  constructor(
    options: {
      nSplits?: number;
      testSize?: number;
      trainSize?: number;
      randomState?: number;
    } = {}
  ) {
    this.nSplits = options.nSplits ?? 5;
    this.testSize = options.testSize;
    this.trainSize = options.trainSize;
    this.randomState = options.randomState;

    validatePositiveSplitCount(this.nSplits);
    parseSplitSpec(this.testSize, "testSize");
    parseSplitSpec(this.trainSize, "trainSize");
    validateRandomState(this.randomState);
  }

  /**
   * Generate train/test split indices based on groups.
   *
   * @param X - Data tensor; only the number of rows is used
   * @param _y - Unused, accepted for API symmetry with the other cross-validators
   * @param groups - Group label of each sample: an array of strings, numbers or bigints, or a
   *   1D tensor
   * @throws {InvalidParameterError} If `groups` is missing, its length differs from the number
   *   of samples, or the sizes cannot be satisfied by the number of groups
   */
  split(X: Tensor, _y?: Tensor, groups?: Tensor | readonly Label[]): SplitResult[] {
    if (!groups) {
      throw new InvalidParameterError(
        "groups parameter is required for GroupShuffleSplit",
        "groups",
        groups
      );
    }

    const nSamples = getNSamples(X);
    const groupLabels = readGroups(groups);
    if (groupLabels.length !== nSamples) {
      throw new InvalidParameterError(
        "X and groups must have same number of samples",
        "groups",
        groupLabels.length
      );
    }

    const { indices, codes } = groupByLabelSorted(groupLabels);
    const nGroups = indices.length;
    const [nTrainGroups, nTestGroups] = resolveTrainTestCounts(
      nGroups,
      this.trainSize,
      this.testSize,
      0.2,
      "group"
    );

    const splitResults: SplitResult[] = [];
    const role = new Uint8Array(nGroups);

    for (let iter = 0; iter < this.nSplits; iter++) {
      const random =
        this.randomState !== undefined ? createRandomStream(this.randomState, iter) : __random;

      const order = range(nGroups);
      shuffleIndicesInPlace(order, random);

      role.fill(0);
      for (let i = 0; i < nTestGroups; i++) role[order[i] as number] = 2;
      for (let i = nTestGroups; i < nTestGroups + nTrainGroups; i++) role[order[i] as number] = 1;

      const trainIndex: number[] = [];
      const testIndex: number[] = [];
      for (let i = 0; i < nSamples; i++) {
        const r = role[codes[i] as number];
        if (r === 1) trainIndex.push(i);
        else if (r === 2) testIndex.push(i);
      }

      splitResults.push({ trainIndex, testIndex });
    }

    return splitResults;
  }

  /** Number of re-shuffling iterations. */
  getNSplits(): number {
    return this.nSplits;
  }
}
