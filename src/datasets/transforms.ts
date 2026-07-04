/**
 * @see {@link https://deepbox.dev/docs/datasets-builtin | Deepbox documentation}
 */

import { InvalidParameterError } from "../core/errors";
import { gather, type Tensor, tensor } from "../ndarray";
import type { Dataset } from "./loaders";
import { createRng, shuffleInPlace } from "./utils";

/**
 * A subset of a dataset defined by explicit indices.
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
  readonly indices: readonly number[];

  constructor(dataset: Dataset, indices: readonly number[]) {
    const n = dataset.data.shape[0] ?? 0;
    for (const idx of indices) {
      if (!Number.isInteger(idx) || idx < 0 || idx >= n) {
        throw new InvalidParameterError(
          `Subset index ${idx} is out of bounds for dataset with ${n} samples`,
          "indices",
          idx
        );
      }
    }

    const idxTensor = tensor(indices as number[], { dtype: "int32" });
    this.data = gather(dataset.data, idxTensor, 0);
    this.target = gather(dataset.target, idxTensor, 0);
    this.featureNames = dataset.featureNames;
    if (dataset.targetNames) this.targetNames = dataset.targetNames;
    this.description = dataset.description;
    this.indices = indices;
  }
}

/**
 * Randomly split a dataset into non-overlapping subsets of given lengths.
 *
 * @param dataset - The dataset to split
 * @param lengths - Array of lengths for each split (must sum to dataset size)
 * @param seed - Optional seed for reproducible splits
 * @returns Array of Subset instances
 *
 * @example
 * ```ts
 * const iris = loadIris();
 * const [train, test] = randomSplit(iris, [120, 30], 42);
 * console.log(train.data.shape);  // [120, 4]
 * console.log(test.data.shape);   // [30, 4]
 * ```
 */
export function randomSplit(dataset: Dataset, lengths: readonly number[], seed?: number): Subset[] {
  const n = dataset.data.shape[0] ?? 0;
  let total = 0;
  for (const len of lengths) {
    if (!Number.isInteger(len) || len < 0) {
      throw new InvalidParameterError(
        `Each length must be a non-negative integer; received ${len}`,
        "lengths",
        len
      );
    }
    total += len;
  }

  if (total !== n) {
    throw new InvalidParameterError(
      `Sum of lengths (${total}) must equal dataset size (${n})`,
      "lengths",
      lengths
    );
  }

  const indices = Array.from({ length: n }, (_, i) => i);
  const rng = createRng(seed);
  shuffleInPlace(indices, rng);

  const subsets: Subset[] = [];
  let offset = 0;
  for (const len of lengths) {
    const subIndices = indices.slice(offset, offset + len);
    subsets.push(new Subset(dataset, subIndices));
    offset += len;
  }

  return subsets;
}

/**
 * Apply a transformation function to every sample in a dataset, producing a new dataset.
 *
 * The function receives a single sample's data tensor (1D) and target scalar value,
 * and must return a `{ data: number[]; target: number }` pair.
 *
 * @param dataset - Source dataset
 * @param fn - Mapping function applied to each sample
 * @returns A new Dataset with transformed data
 *
 * @example
 * ```ts
 * const iris = loadIris();
 * // Standardize features
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
  const n = dataset.data.shape[0] ?? 0;
  const nFeatures = dataset.data.shape[1] ?? 1;
  const newData: number[][] = [];
  const newTargets: number[] = [];

  for (let i = 0; i < n; i++) {
    const sampleData: number[] = [];
    if (dataset.data.ndim === 1) {
      const val = dataset.data.at(i);
      sampleData.push(typeof val === "number" ? val : Number(val));
    } else {
      for (let j = 0; j < nFeatures; j++) {
        const val = dataset.data.at(i, j);
        sampleData.push(typeof val === "number" ? val : Number(val));
      }
    }
    const tVal = dataset.target.at(i);
    const targetNum = typeof tVal === "number" ? tVal : Number(tVal);

    const result = fn(sampleData, targetNum);
    newData.push(result.data);
    newTargets.push(result.target);
  }

  const result: Dataset = {
    data: tensor(newData),
    target: tensor(newTargets, { dtype: dataset.target.dtype }),
    featureNames: dataset.featureNames,
    description: dataset.description,
  };
  if (dataset.targetNames) result.targetNames = dataset.targetNames;
  return result;
}

/**
 * Filter a dataset by a predicate function, keeping only samples where the
 * predicate returns true.
 *
 * @param dataset - Source dataset
 * @param predicate - Function that receives sample data and target, returns boolean
 * @returns A new Dataset containing only matching samples
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
  const n = dataset.data.shape[0] ?? 0;
  const nFeatures = dataset.data.shape[1] ?? 1;
  const keepIndices: number[] = [];

  for (let i = 0; i < n; i++) {
    const sampleData: number[] = [];
    if (dataset.data.ndim === 1) {
      const val = dataset.data.at(i);
      sampleData.push(typeof val === "number" ? val : Number(val));
    } else {
      for (let j = 0; j < nFeatures; j++) {
        const val = dataset.data.at(i, j);
        sampleData.push(typeof val === "number" ? val : Number(val));
      }
    }
    const tVal = dataset.target.at(i);
    const targetNum = typeof tVal === "number" ? tVal : Number(tVal);

    if (predicate(sampleData, targetNum)) {
      keepIndices.push(i);
    }
  }

  if (keepIndices.length === 0) {
    const empty: Dataset = {
      data: tensor([]),
      target: tensor([]),
      featureNames: dataset.featureNames,
      description: dataset.description,
    };
    if (dataset.targetNames) empty.targetNames = dataset.targetNames;
    return empty;
  }

  const idxTensor = tensor(keepIndices, { dtype: "int32" });
  const filtered: Dataset = {
    data: gather(dataset.data, idxTensor, 0),
    target: gather(dataset.target, idxTensor, 0),
    featureNames: dataset.featureNames,
    description: dataset.description,
  };
  if (dataset.targetNames) filtered.targetNames = dataset.targetNames;
  return filtered;
}
