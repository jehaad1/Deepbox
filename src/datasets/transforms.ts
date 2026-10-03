/**
 * @see {@link https://deepbox.dev/docs/datasets-builtin | Deepbox documentation}
 */

import { DeviceError, DTypeError, InvalidParameterError, ShapeError } from "../core/errors";
import { gather, reshape, type Tensor, tensor } from "../ndarray";
import type { Dataset } from "./loaders";
import { createRng, normalizeOptionalSeed, shuffleInPlace } from "./utils";

/**
 * Validate that `dataset` is a usable sample-major dataset and return its sample count.
 */
function sampleCount(dataset: Dataset, fnName: string): number {
  if (dataset.data.ndim < 1) {
    throw new ShapeError(`${fnName}: dataset.data must have at least one dimension`);
  }
  const n = dataset.data.shape[0] as number;
  const nTarget = dataset.target.ndim < 1 ? undefined : dataset.target.shape[0];
  if (nTarget !== n) {
    throw new ShapeError(
      `${fnName}: dataset.data has ${n} samples but dataset.target has ` +
        `${nTarget === undefined ? "no sample axis" : `${nTarget} samples`}`
    );
  }
  if (dataset.images !== undefined && dataset.images.shape[0] !== n) {
    throw new ShapeError(
      `${fnName}: dataset.images has ${dataset.images.shape[0] ?? 0} samples; expected ${n}`
    );
  }
  return n;
}

/**
 * Logical row-major values of a numeric tensor as a Float64Array (handles views and strides).
 */
function toFloat64Values(t: Tensor, name: string): Float64Array {
  if (t.dtype === "string" || t.dtype === "complex64" || t.dtype === "complex128") {
    throw new DTypeError(`${name} must be real-valued numeric; received a ${t.dtype} tensor`);
  }
  if (t.isDeviceTensor) {
    throw new DeviceError(`${name} lives on device '${t.device}'; call await t.cpu() first`);
  }
  const flat = reshape(t, [t.size]);
  const src = flat.data as ArrayLike<number | bigint>;
  const out = new Float64Array(flat.size);
  for (let i = 0; i < out.length; i++) out[i] = Number(src[flat.offset + i]);
  return out;
}

/** Number of values per sample (product of all axes after the first; 1 for 1D data). */
function sampleWidth(t: Tensor): number {
  let w = 1;
  for (let i = 1; i < t.ndim; i++) w *= t.shape[i] as number;
  return w;
}

/**
 * Gather the optional `images` tensor alongside data/target and copy the metadata.
 */
function selectSamples(dataset: Dataset, indices: readonly number[]): Dataset {
  const idx = tensor(indices as number[], { dtype: "int32" });
  const out: Dataset = {
    data: gather(dataset.data, idx, 0),
    target: gather(dataset.target, idx, 0),
    featureNames: [...dataset.featureNames],
    description: dataset.description,
  };
  if (dataset.targetNames) out.targetNames = [...dataset.targetNames];
  if (dataset.images) out.images = gather(dataset.images, idx, 0);
  return out;
}

/**
 * A subset of a dataset defined by explicit indices.
 *
 * `data`, `target` (and `images`, when the dataset has them) hold copies of the
 * selected samples, in the order given. Indices may repeat.
 *
 * @throws {InvalidParameterError} If an index is not an integer in `[0, nSamples)`.
 * @throws {ShapeError} If `dataset.data` and `dataset.target` disagree on the sample count.
 *
 * @example
 * ```ts
 * const iris = loadIris();
 * const sub = new Subset(iris, [0, 1, 2, 3, 4]);
 * console.log(sub.data.shape);  // [5, 4]
 * ```
 */
export class Subset {
  readonly data: Tensor;
  readonly target: Tensor;
  readonly featureNames: string[];
  readonly targetNames?: string[];
  readonly description: string;
  readonly images?: Tensor;
  readonly indices: readonly number[];

  constructor(dataset: Dataset, indices: readonly number[]) {
    if (!Array.isArray(indices)) {
      throw new InvalidParameterError("indices must be an array of integers", "indices", indices);
    }
    const n = sampleCount(dataset, "Subset");
    for (const idx of indices) {
      if (!Number.isInteger(idx) || idx < 0 || idx >= n) {
        throw new InvalidParameterError(
          `Subset index ${idx} is out of bounds for dataset with ${n} samples`,
          "indices",
          idx
        );
      }
    }

    const selected = selectSamples(dataset, indices);
    this.data = selected.data;
    this.target = selected.target;
    this.featureNames = selected.featureNames;
    if (selected.targetNames) this.targetNames = selected.targetNames;
    if (selected.images) this.images = selected.images;
    this.description = dataset.description;
    this.indices = [...indices];
  }
}

/**
 * Randomly split a dataset into non-overlapping subsets of given lengths.
 *
 * `lengths` is either absolute sample counts that sum to the dataset size, or
 * fractions in `[0, 1]` that sum to 1. For fractions, each subset gets
 * `floor(fraction * n)` samples and the remaining samples are handed out one at
 * a time to the subsets in order, as in PyTorch's `random_split`.
 *
 * @param dataset - The dataset to split
 * @param lengths - Subset sizes: integers summing to the dataset size, or fractions summing to 1
 * @param seed - Optional seed for reproducible splits
 * @returns Array of Subset instances
 * @throws {InvalidParameterError} If `lengths` are negative or do not add up, or `seed` is not a safe integer
 *
 * @example
 * ```ts
 * const iris = loadIris();
 * const [train, test] = randomSplit(iris, [120, 30], 42);
 * console.log(train.data.shape);  // [120, 4]
 * console.log(test.data.shape);   // [30, 4]
 *
 * const [a, b] = randomSplit(iris, [0.8, 0.2], 42);
 * console.log(a.data.shape[0], b.data.shape[0]); // 120 30
 * ```
 */
export function randomSplit(dataset: Dataset, lengths: readonly number[], seed?: number): Subset[] {
  if (!Array.isArray(lengths)) {
    throw new InvalidParameterError("lengths must be an array of numbers", "lengths", lengths);
  }
  const n = sampleCount(dataset, "randomSplit");
  normalizeOptionalSeed("seed", seed);

  let total = 0;
  let allIntegers = true;
  for (const len of lengths) {
    if (typeof len !== "number" || !Number.isFinite(len) || len < 0) {
      throw new InvalidParameterError(
        `Each length must be a non-negative number; received ${len}`,
        "lengths",
        len
      );
    }
    if (!Number.isInteger(len)) allIntegers = false;
    total += len;
  }

  // Lengths such as [1] or [0, 1] sum to 1 and are fractions, not counts (unless n is 1).
  const asFractions = !allIntegers || (total === 1 && n !== 1);
  let sizes: number[];
  if (!asFractions) {
    if (total !== n) {
      throw new InvalidParameterError(
        `Sum of lengths (${total}) must equal dataset size (${n})`,
        "lengths",
        lengths
      );
    }
    sizes = [...lengths];
  } else {
    // Fractions: they must add up to 1 (within floating-point rounding).
    if (Math.abs(total - 1) > 1e-9 || lengths.some((f) => f > 1)) {
      throw new InvalidParameterError(
        `Fractional lengths must lie in [0, 1] and sum to 1; received sum ${total}`,
        "lengths",
        lengths
      );
    }
    sizes = lengths.map((f) => Math.floor(n * f));
    let remainder = n - sizes.reduce((a, b) => a + b, 0);
    for (let i = 0; remainder > 0; i = (i + 1) % sizes.length, remainder--) {
      sizes[i] = (sizes[i] as number) + 1;
    }
  }

  const indices = Array.from({ length: n }, (_, i) => i);
  const rng = createRng(seed);
  shuffleInPlace(indices, rng);

  const subsets: Subset[] = [];
  let offset = 0;
  for (const len of sizes) {
    subsets.push(new Subset(dataset, indices.slice(offset, offset + len)));
    offset += len;
  }

  return subsets;
}

/**
 * Pick the dtype of a mapped target: keep the source dtype when every value is
 * representable in it, otherwise fall back to the default float dtype so values
 * are not silently truncated (e.g. 0.5 stored in an `int32` target).
 */
function mappedTargetDtype(
  values: readonly number[],
  source: Tensor["dtype"]
): Tensor["dtype"] | undefined {
  const range: Partial<Record<Tensor["dtype"], readonly [number, number]>> = {
    int32: [-2147483648, 2147483647],
    int64: [Number.MIN_SAFE_INTEGER, Number.MAX_SAFE_INTEGER],
    uint8: [0, 255],
    bool: [0, 1],
  };
  const r = range[source];
  if (r === undefined) return source;
  for (const v of values) {
    if (!Number.isInteger(v) || v < r[0] || v > r[1]) return undefined;
  }
  return source;
}

/**
 * Apply a transformation function to every sample in a dataset, producing a new dataset.
 *
 * The function receives one sample's features as a flat `number[]` (the values
 * of `data[i]` in row-major order) and its scalar target, and must return a
 * `{ data: number[]; target: number }` pair. All returned `data` arrays must
 * have the same length; the result `data` has shape `[nSamples, length]`.
 *
 * The result keeps the float dtype of the source `data`. The target keeps the
 * dtype of the source target unless a returned value cannot be represented in
 * it (for example a fractional value for an `int32` target), in which case the
 * default float dtype is used. The source dataset is not modified.
 *
 * @param dataset - Source dataset. `target` must be 1D (one value per sample).
 * @param fn - Mapping function applied to each sample
 * @returns A new Dataset with transformed data
 * @throws {ShapeError} If `target` is not 1D or `fn` returns rows of different lengths
 * @throws {InvalidParameterError} If `fn` does not return `{ data, target }` with an array and a number
 *
 * @example
 * ```ts
 * const iris = loadIris();
 * // Scale features
 * const mapped = mapDataset(iris, (data, target) => ({
 *   data: data.map(v => v * 2),
 *   target,
 * }));
 * ```
 */
export function mapDataset(
  dataset: Dataset,
  fn: (data: number[], target: number) => { data: number[]; target: number }
): Dataset {
  if (typeof fn !== "function") {
    throw new InvalidParameterError("mapDataset expects a function", "fn", fn);
  }
  const n = sampleCount(dataset, "mapDataset");
  if (dataset.target.ndim !== 1) {
    throw new ShapeError(
      `mapDataset requires a 1D target; received shape [${dataset.target.shape.join(", ")}]`
    );
  }
  const width = sampleWidth(dataset.data);
  const values = toFloat64Values(dataset.data, "dataset.data");
  const targets = toFloat64Values(dataset.target, "dataset.target");

  const newData: number[][] = new Array(n);
  const newTargets: number[] = new Array(n);
  let outWidth = width;

  for (let i = 0; i < n; i++) {
    const result = fn(
      Array.from(values.subarray(i * width, (i + 1) * width)),
      targets[i] as number
    );
    if (
      result === null ||
      typeof result !== "object" ||
      !Array.isArray(result.data) ||
      typeof result.target !== "number"
    ) {
      throw new InvalidParameterError(
        `mapDataset callback must return { data: number[], target: number }; sample ${i} did not`,
        "fn",
        result
      );
    }
    if (i === 0) outWidth = result.data.length;
    else if (result.data.length !== outWidth) {
      throw new ShapeError(
        `mapDataset callback returned ${result.data.length} features for sample ${i}; ` +
          `expected ${outWidth} like sample 0`
      );
    }
    newData[i] = result.data;
    newTargets[i] = result.target;
  }

  const srcDtype = dataset.data.dtype;
  const dataDtype = srcDtype === "float32" || srcDtype === "float64" ? srcDtype : undefined;
  const dataTensor =
    n === 0
      ? reshape(tensor([], dataDtype ? { dtype: dataDtype } : {}), [0, width])
      : tensor(newData, dataDtype ? { dtype: dataDtype } : {});
  const tDtype = mappedTargetDtype(newTargets, dataset.target.dtype);

  const mapped: Dataset = {
    data: dataTensor,
    target: tensor(newTargets, tDtype ? { dtype: tDtype } : {}),
    featureNames: [...dataset.featureNames],
    description: dataset.description,
  };
  if (dataset.targetNames) mapped.targetNames = [...dataset.targetNames];
  return mapped;
}

/**
 * Filter a dataset by a predicate function, keeping only samples where the
 * predicate returns true.
 *
 * The predicate receives one sample's features as a flat `number[]` and its
 * scalar target. Samples keep their original order, dtypes and `images`. When
 * nothing matches, the result has zero samples but keeps the feature axes
 * (for example `data` of shape `[0, 4]`).
 *
 * @param dataset - Source dataset. `target` must be 1D (one value per sample).
 * @param predicate - Function that receives sample data and target, returns boolean
 * @returns A new Dataset containing only matching samples
 * @throws {ShapeError} If `target` is not 1D
 *
 * @example
 * ```ts
 * const iris = loadIris();
 * // Keep only class 0 and 1
 * const filtered = filterDataset(iris, (_data, target) => target < 2);
 * ```
 */
export function filterDataset(
  dataset: Dataset,
  predicate: (data: number[], target: number) => boolean
): Dataset {
  if (typeof predicate !== "function") {
    throw new InvalidParameterError(
      "filterDataset expects a predicate function",
      "predicate",
      predicate
    );
  }
  const n = sampleCount(dataset, "filterDataset");
  if (dataset.target.ndim !== 1) {
    throw new ShapeError(
      `filterDataset requires a 1D target; received shape [${dataset.target.shape.join(", ")}]`
    );
  }
  const width = sampleWidth(dataset.data);
  const values = toFloat64Values(dataset.data, "dataset.data");
  const targets = toFloat64Values(dataset.target, "dataset.target");

  const keepIndices: number[] = [];
  for (let i = 0; i < n; i++) {
    const row = Array.from(values.subarray(i * width, (i + 1) * width));
    if (predicate(row, targets[i] as number)) keepIndices.push(i);
  }

  return selectSamples(dataset, keepIndices);
}
