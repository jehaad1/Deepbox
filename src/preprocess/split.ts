import { DeepboxError, InvalidParameterError, ShapeError } from "../core/errors";
import { type Tensor, zeros } from "../ndarray";
import { createSeededRandom, getShape2D, shuffleIndicesInPlace } from "./_internal";

function validateNSplits(nSplits: number): void {
  if (!Number.isFinite(nSplits) || !Number.isInteger(nSplits) || nSplits < 2) {
    throw new InvalidParameterError("nSplits must be an integer at least 2", "nSplits", nSplits);
  }
}

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

function resolveSplitCount(spec: SplitSpec, nSamples: number, isTrain: boolean): number {
  if (spec.kind === "count") {
    return spec.value;
  }
  const exact = nSamples * spec.value;
  return isTrain ? Math.floor(exact) : Math.ceil(exact);
}

function resolveTrainTestCounts(
  nSamples: number,
  trainSize: number | undefined,
  testSize: number | undefined
): [number, number] {
  const defaultTestSize = trainSize === undefined && testSize === undefined ? 0.25 : testSize;
  const trainSpec = parseSplitSpec(trainSize, "trainSize");
  const testSpec = parseSplitSpec(defaultTestSize, "testSize");

  if (trainSpec?.kind === "count" && trainSpec.value > nSamples) {
    throw new InvalidParameterError(
      "trainSize must not exceed number of samples",
      "trainSize",
      trainSpec.value
    );
  }
  if (testSpec?.kind === "count" && testSpec.value > nSamples) {
    throw new InvalidParameterError(
      "testSize must not exceed number of samples",
      "testSize",
      testSpec.value
    );
  }

  if (
    trainSpec?.kind === "fraction" &&
    testSpec?.kind === "fraction" &&
    trainSpec.value + testSpec.value > 1
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
      "trainSize and testSize exceed number of samples",
      "trainSize",
      trainSize
    );
  }

  if (nTrain < 1) {
    throw new InvalidParameterError("trainSize must be at least 1 sample", "trainSize", trainSize);
  }
  if (nTest < 1) {
    throw new InvalidParameterError("testSize must be at least 1 sample", "testSize", testSize);
  }

  return [nTrain, nTest];
}

function compareLabels(a: unknown, b: unknown): number {
  if (typeof a === "number" && typeof b === "number") return a - b;
  if (typeof a === "bigint" && typeof b === "bigint") {
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
  }
  return String(a).localeCompare(String(b));
}

function makeFoldSizes(total: number, nSplits: number): number[] {
  const base = Math.floor(total / nSplits);
  const remainder = total % nSplits;
  return Array.from({ length: nSplits }, (_, i) => base + (i < remainder ? 1 : 0));
}

function readTensorValue(t: Tensor, indices: number[]): string | number | bigint {
  const value = t.at(...indices);
  if (typeof value === "string" || typeof value === "number" || typeof value === "bigint") {
    return value;
  }
  throw new DeepboxError("Internal error: unsupported tensor value type");
}

function writeTensorValue(t: Tensor, flatIndex: number, value: string | number | bigint): void {
  if (t.dtype === "string") {
    if (typeof value !== "string") {
      throw new DeepboxError("Internal error: expected string value for string tensor");
    }
    t.data[flatIndex] = value;
    return;
  }

  if (typeof value === "string") {
    throw new DeepboxError("Internal error: encountered string value in numeric tensor");
  }

  if (t.data instanceof BigInt64Array) {
    t.data[flatIndex] = typeof value === "bigint" ? value : BigInt(value);
    return;
  }

  t.data[flatIndex] = Number(value);
}

function takeRows2D(X: Tensor, sampleIndices: number[]): Tensor {
  const [, nFeatures] = getShape2D(X);
  const out = zeros([sampleIndices.length, nFeatures], { dtype: X.dtype });

  for (let i = 0; i < sampleIndices.length; i++) {
    const sampleIndex = sampleIndices[i];
    if (sampleIndex === undefined) {
      throw new DeepboxError("Internal error: sample index access failed");
    }
    for (let j = 0; j < nFeatures; j++) {
      const value = readTensorValue(X, [sampleIndex, j]);
      writeTensorValue(out, out.offset + i * nFeatures + j, value);
    }
  }

  return out;
}

function takeVector(y: Tensor, sampleIndices: number[]): Tensor {
  if (y.ndim !== 1) {
    throw new ShapeError(`y must be a 1D tensor, got ${y.ndim}D`);
  }
  const out = zeros([sampleIndices.length], { dtype: y.dtype });

  for (let i = 0; i < sampleIndices.length; i++) {
    const sampleIndex = sampleIndices[i];
    if (sampleIndex === undefined) {
      throw new DeepboxError("Internal error: sample index access failed");
    }
    const value = readTensorValue(y, [sampleIndex]);
    writeTensorValue(out, out.offset + i, value);
  }

  return out;
}

/**
 * Split arrays into random train and test subsets.
 *
 * @param X - Feature matrix (2D tensor)
 * @param y - Optional target labels (1D tensor)
 * @param options - Split configuration options
 * @param options.testSize - Proportion or absolute number of test samples
 * @param options.trainSize - Proportion or absolute number of train samples
 * @param options.randomState - Random seed
 * @param options.shuffle - Whether to shuffle data before splitting
 * @param options.stratify - If not undefined, data is split in stratified fashion using this as class labels
 *
 * @example
 * ```js
 * import { trainTestSplit } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1, 2], [3, 4], [5, 6], [7, 8]]);
 * const y = tensor([0, 1, 0, 1]);
 * const [XTrain, XTest, yTrain, yTest] = trainTestSplit(X, y, { testSize: 0.25 });
 * ```
 *
 * @see {@link https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html | Scikit-learn train_test_split}
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

  const [nSamples] = getShape2D(X);

  if (nSamples === 0) {
    throw new InvalidParameterError("Cannot split empty array", "X");
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

  const indices = Array.from({ length: nSamples }, (_, i) => i);
  const random = randomState !== undefined ? createSeededRandom(randomState) : Math.random;
  const maybeShuffle = (arr: number[]): void => {
    if (!shuffle) return;
    shuffleIndicesInPlace(arr, random);
  };

  let trainIndices: number[] = [];
  let testIndices: number[] = [];

  if (opts.stratify) {
    const stratify = opts.stratify;
    const labelMap = new Map<unknown, number[]>();
    for (let i = 0; i < nSamples; i++) {
      const label = readTensorValue(stratify, [i]);
      let bucket = labelMap.get(label);
      if (bucket === undefined) {
        bucket = [];
        labelMap.set(label, bucket);
      }
      bucket.push(i);
    }

    const labels = Array.from(labelMap.keys()).sort(compareLabels);
    const nClasses = labels.length;
    const classSizes = labels.map((label) => labelMap.get(label)?.length ?? 0);
    const hasSingleton = classSizes.some((size) => size < 2);

    if (hasSingleton && shuffle && randomState === undefined) {
      throw new InvalidParameterError(
        "stratify requires at least 2 samples per class",
        "stratify",
        classSizes
      );
    }

    if (opts.trainSize !== undefined && nTrain < nClasses) {
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

    const testFraction = nTest / nSamples;
    const allowEmptyClassSplits = nTrain < nClasses;
    const counts = labels.map((label) => {
      const size = labelMap.get(label)?.length ?? 0;
      const exact = size * testFraction;
      let testCount = Math.floor(exact);
      let remainder = exact - testCount;
      let min = allowEmptyClassSplits ? 0 : 1;
      let max = allowEmptyClassSplits ? size : size - 1;
      if (size < 2) {
        min = 0;
        max = allowEmptyClassSplits ? size : 0;
        testCount = 0;
        remainder = 0;
      } else {
        if (testCount < min) testCount = min;
        if (testCount > max) testCount = max;
      }
      return { label, size, testCount, remainder, min, max };
    });

    let remaining = nTest - counts.reduce((sum, c) => sum + c.testCount, 0);
    if (remaining !== 0) {
      const order =
        remaining > 0
          ? [...counts].sort((a, b) => {
              if (b.remainder !== a.remainder) return b.remainder - a.remainder;
              return compareLabels(a.label, b.label);
            })
          : [...counts].sort((a, b) => {
              if (a.remainder !== b.remainder) return a.remainder - b.remainder;
              return compareLabels(a.label, b.label);
            });

      let guard = 0;
      while (remaining !== 0 && guard < counts.length * 2) {
        for (const entry of order) {
          if (remaining === 0) break;
          if (remaining > 0 && entry.testCount < entry.max) {
            entry.testCount += 1;
            remaining -= 1;
          } else if (remaining < 0 && entry.testCount > entry.min) {
            entry.testCount -= 1;
            remaining += 1;
          }
        }
        guard += 1;
      }

      if (remaining !== 0) {
        throw new DeepboxError("Internal error: unable to allocate stratified split sizes");
      }
    }

    const remainingTrainPool: number[] = [];
    for (const entry of counts) {
      const labelIndices = [...(labelMap.get(entry.label) ?? [])];
      maybeShuffle(labelIndices);
      testIndices.push(...labelIndices.slice(0, entry.testCount));
      remainingTrainPool.push(...labelIndices.slice(entry.testCount));
    }

    maybeShuffle(testIndices);
    maybeShuffle(remainingTrainPool);
    trainIndices = remainingTrainPool.slice(0, nTrain);
  } else {
    maybeShuffle(indices);
    trainIndices = indices.slice(0, nTrain);
    testIndices = indices.slice(nTrain, nTrain + nTest);
  }

  if (trainIndices.length !== nTrain || testIndices.length !== nTest) {
    throw new DeepboxError("Internal error: resolved split indices do not match requested sizes");
  }

  const XTrain = takeRows2D(X, trainIndices);
  const XTest = takeRows2D(X, testIndices);

  if (y) {
    const yTrain = takeVector(y, trainIndices);
    const yTest = takeVector(y, testIndices);
    return [XTrain, XTest, yTrain, yTest];
  }

  return [XTrain, XTest];
}

/**
 * K-Folds cross-validator.
 *
 * Provides train/test indices to split data in train/test sets.
 *
 * @see {@link https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html | Scikit-learn KFold}
 */
export class KFold {
  private nSplits: number;
  private shuffle: boolean;
  private randomState: number | undefined;

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
  }

  split(X: Tensor): Array<[number[], number[]]> {
    const shape0 = X.shape[0];
    if (shape0 === undefined) {
      throw new ShapeError("X must have valid shape[0]");
    }
    const nSamples = shape0;
    validateNSplits(this.nSplits);
    if (this.nSplits > nSamples) {
      throw new InvalidParameterError(
        "nSplits must not be greater than number of samples",
        "nSplits",
        this.nSplits
      );
    }
    const indices = Array.from({ length: nSamples }, (_, i) => i);

    if (this.shuffle) {
      const random =
        this.randomState !== undefined ? createSeededRandom(this.randomState) : Math.random;
      shuffleIndicesInPlace(indices, random);
    }

    const splits: Array<[number[], number[]]> = [];
    const foldSizes = makeFoldSizes(nSamples, this.nSplits);
    let current = 0;

    for (let i = 0; i < this.nSplits; i++) {
      const foldSize = foldSizes[i] ?? 0;
      const testStart = current;
      const testEnd = current + foldSize;

      const testIndices = indices.slice(testStart, testEnd);
      const trainIndices = [...indices.slice(0, testStart), ...indices.slice(testEnd)];

      splits.push([trainIndices, testIndices]);
      current = testEnd;
    }

    return splits;
  }

  getNSplits(): number {
    return this.nSplits;
  }
}

/**
 * Stratified K-Folds cross-validator.
 *
 * Provides train/test indices while preserving class distribution.
 *
 * @see {@link https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html | Scikit-learn StratifiedKFold}
 */
export class StratifiedKFold {
  private nSplits: number;
  private shuffle: boolean;
  private randomState: number | undefined;

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
  }

  split(X: Tensor, y: Tensor): Array<[number[], number[]]> {
    const shape0 = X.shape[0];
    if (shape0 === undefined) {
      throw new ShapeError("X must have valid shape[0]");
    }
    const nSamples = shape0;
    validateNSplits(this.nSplits);
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
    if (y.ndim !== 1) {
      throw new ShapeError(`y must be a 1D tensor, got ${y.ndim}D`);
    }
    const labelMap = new Map<string | number | bigint, number[]>();
    const random =
      this.randomState !== undefined ? createSeededRandom(this.randomState) : Math.random;

    for (let i = 0; i < nSamples; i++) {
      const label = readTensorValue(y, [i]);
      let bucket = labelMap.get(label);
      if (bucket === undefined) {
        bucket = [];
        labelMap.set(label, bucket);
      }
      bucket.push(i);
    }

    for (const [label, indices] of labelMap.entries()) {
      if (this.shuffle) {
        shuffleIndicesInPlace(indices, random);
      }
      if (indices.length < this.nSplits) {
        throw new InvalidParameterError(
          `Each class must have at least nSplits samples; class ${label} has ${indices.length}`,
          "nSplits",
          this.nSplits
        );
      }
    }

    const foldIndices: number[][] = Array.from({ length: this.nSplits }, () => []);

    for (const indices of labelMap.values()) {
      const foldSizes = makeFoldSizes(indices.length, this.nSplits);
      let start = 0;
      for (let fold = 0; fold < this.nSplits; fold++) {
        const size = foldSizes[fold] ?? 0;
        const end = start + size;
        const target = foldIndices[fold];
        if (!target) {
          throw new DeepboxError("Internal error: stratified fold storage missing");
        }
        target.push(...indices.slice(start, end));
        start = end;
      }
    }

    const splits: Array<[number[], number[]]> = [];

    for (let fold = 0; fold < this.nSplits; fold++) {
      const testIndices = foldIndices[fold] ?? [];
      const trainIndices: number[] = [];
      for (let other = 0; other < this.nSplits; other++) {
        if (other === fold) continue;
        trainIndices.push(...(foldIndices[other] ?? []));
      }
      splits.push([trainIndices, testIndices]);
    }

    return splits;
  }

  getNSplits(): number {
    return this.nSplits;
  }
}

/**
 * Group K-Fold cross-validator.
 *
 * Ensures same group is not in both train and test.
 *
 * @see {@link https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html | Scikit-learn GroupKFold}
 */
export class GroupKFold {
  private nSplits: number;

  constructor(options: { nSplits?: number } = {}) {
    this.nSplits = options.nSplits ?? 5;
  }

  split(X: Tensor, _y: Tensor | undefined, groups: Tensor): Array<[number[], number[]]> {
    const shape0 = X.shape[0];
    if (shape0 === undefined) {
      throw new ShapeError("X must have valid shape[0]");
    }
    const nSamples = shape0;
    validateNSplits(this.nSplits);
    if (groups.ndim !== 1) {
      throw new ShapeError(`groups must be a 1D tensor, got ${groups.ndim}D`);
    }
    const groupsShape0 = groups.shape[0];
    if (groupsShape0 === undefined || groupsShape0 !== nSamples) {
      throw new InvalidParameterError(
        "X and groups must have same number of samples",
        "groups",
        groupsShape0
      );
    }
    const groupMap = new Map<string | number | bigint, number[]>();

    for (let i = 0; i < nSamples; i++) {
      const group = readTensorValue(groups, [i]);
      let bucket = groupMap.get(group);
      if (bucket === undefined) {
        bucket = [];
        groupMap.set(group, bucket);
      }
      bucket.push(i);
    }

    const groupEntries = Array.from(groupMap.entries()).map(([group, indices]) => ({
      group,
      indices,
      size: indices.length,
    }));
    if (this.nSplits > groupEntries.length) {
      throw new InvalidParameterError(
        "Number of groups must be at least nSplits",
        "nSplits",
        this.nSplits
      );
    }
    groupEntries.sort((a, b) => {
      if (b.size !== a.size) return b.size - a.size;
      return compareLabels(a.group, b.group);
    });

    const foldIndices: number[][] = Array.from({ length: this.nSplits }, () => []);
    const foldSizes = new Array<number>(this.nSplits).fill(0);

    for (const entry of groupEntries) {
      let bestFold = 0;
      let bestSize = foldSizes[0] ?? 0;
      for (let fold = 1; fold < this.nSplits; fold++) {
        const size = foldSizes[fold] ?? 0;
        if (size < bestSize) {
          bestSize = size;
          bestFold = fold;
        }
      }
      const target = foldIndices[bestFold];
      if (!target) {
        throw new DeepboxError("Internal error: group fold storage missing");
      }
      target.push(...entry.indices);
      foldSizes[bestFold] = bestSize + entry.size;
    }

    const splits: Array<[number[], number[]]> = [];
    for (let fold = 0; fold < this.nSplits; fold++) {
      const testIndices = foldIndices[fold] ?? [];
      const trainIndices: number[] = [];
      for (let other = 0; other < this.nSplits; other++) {
        if (other === fold) continue;
        trainIndices.push(...(foldIndices[other] ?? []));
      }
      splits.push([trainIndices, testIndices]);
    }

    return splits;
  }

  getNSplits(): number {
    return this.nSplits;
  }
}

/**
 * Leave-One-Out cross-validator.
 *
 * @see {@link https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html | Scikit-learn LeaveOneOut}
 */
export class LeaveOneOut {
  split(X: Tensor): Array<[number[], number[]]> {
    const shape0 = X.shape[0];
    if (shape0 === undefined) {
      throw new ShapeError("X must have valid shape[0]");
    }
    const nSamples = shape0;
    const splits: Array<[number[], number[]]> = [];

    for (let i = 0; i < nSamples; i++) {
      const trainIndices = [
        ...Array.from({ length: i }, (_, j) => j),
        ...Array.from({ length: nSamples - i - 1 }, (_, j) => i + 1 + j),
      ];
      const testIndices = [i];
      splits.push([trainIndices, testIndices]);
    }

    return splits;
  }

  getNSplits(X: Tensor): number {
    const shape0 = X.shape[0];
    if (shape0 === undefined) {
      throw new ShapeError("X must have valid shape[0]");
    }
    return shape0;
  }
}

/**
 * Leave-P-Out cross-validator.
 *
 * @see {@link https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeavePOut.html | Scikit-learn LeavePOut}
 */
export class LeavePOut {
  private p: number;

  constructor(p: number) {
    if (!Number.isFinite(p) || !Number.isInteger(p) || p <= 0) {
      throw new InvalidParameterError("p must be a positive integer", "p", p);
    }
    this.p = p;
  }

  split(X: Tensor): Array<[number[], number[]]> {
    const shape0 = X.shape[0];
    if (shape0 === undefined) {
      throw new ShapeError("X must have valid shape[0]");
    }
    const nSamples = shape0;
    if (this.p > nSamples) {
      throw new InvalidParameterError("p must not be greater than number of samples", "p", this.p);
    }

    // Calculate number of combinations to prevent memory explosion
    let nCombos = 1;
    const k = this.p > nSamples / 2 ? nSamples - this.p : this.p;
    for (let i = 0; i < k; i++) {
      nCombos = (nCombos * (nSamples - i)) / (i + 1);
    }

    // Safety limit: 100,000 splits is generous for in-memory JS arrays
    // For larger splits, a generator approach would be needed, but split() returns Array.
    if (nCombos > 100000) {
      throw new InvalidParameterError(
        `LeavePOut produces ${Math.floor(nCombos)} splits, which exceeds memory safety limit of 100,000`,
        "p",
        this.p
      );
    }

    const splits: Array<[number[], number[]]> = [];
    const allIndices = Array.from({ length: nSamples }, (_, i) => i);

    // Iterative combination generator
    const combine = (start: number, currentCombo: number[]) => {
      if (currentCombo.length === this.p) {
        const testSet = new Set(currentCombo);
        const testIndices = [...currentCombo];
        const trainIndices = allIndices.filter((i) => !testSet.has(i));
        splits.push([trainIndices, testIndices]);
        return;
      }
      for (let i = start; i < nSamples; i++) {
        currentCombo.push(i);
        combine(i + 1, currentCombo);
        currentCombo.pop();
      }
    };

    combine(0, []);

    return splits;
  }

  getNSplits(X: Tensor): number {
    const shape0 = X.shape[0];
    if (shape0 === undefined) {
      throw new ShapeError("X must have valid shape[0]");
    }
    const n = shape0;
    if (this.p > n) {
      throw new InvalidParameterError("p must not be greater than number of samples", "p", this.p);
    }
    // C(n, p) = n! / (p! * (n-p)!)
    let result = 1;
    const k = this.p > n / 2 ? n - this.p : this.p;
    for (let i = 0; i < k; i++) {
      result = (result * (n - i)) / (i + 1);
    }
    return Math.round(result);
  }
}
