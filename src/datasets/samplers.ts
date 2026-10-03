/**
 * Dataset samplers for controlling index generation in DataLoader.
 *
 * @module datasets/samplers
 * @see {@link https://deepbox.dev/docs/datasets-dataloader | Deepbox Samplers}
 */

import { InvalidParameterError } from "../core/errors";
import {
  assertBoolean,
  assertPositiveInt,
  createPassRng,
  type DatasetRng,
  normalizeOptionalSeed,
  shuffleInPlace,
} from "./utils";

function resolveReshuffle(value: boolean | undefined): boolean {
  if (value === undefined) return false;
  assertBoolean("reshuffleEachIteration", value);
  return value;
}

function isNumberList(value: unknown): value is ArrayLike<number> {
  return Array.isArray(value) || (ArrayBuffer.isView(value) && !(value instanceof DataView));
}

/**
 * Sampler interface: produces an iterable of integer indices.
 */
export interface Sampler {
  /** Total number of indices this sampler will produce. */
  readonly length: number;
  /** Return an iterator of sample indices. */
  [Symbol.iterator](): Iterator<number>;
}

/**
 * Samples elements sequentially, always in the same order.
 *
 * @param dataSourceLength - Number of samples in the data source (a non-negative integer).
 * @throws {InvalidParameterError} If `dataSourceLength` is not a non-negative safe integer.
 *
 * @example
 * ```ts
 * const sampler = new SequentialSampler(10);
 * for (const idx of sampler) { ... } // 0, 1, 2, ..., 9
 * ```
 */
export class SequentialSampler implements Sampler {
  readonly length: number;

  constructor(dataSourceLength: number) {
    if (!Number.isSafeInteger(dataSourceLength) || dataSourceLength < 0) {
      throw new InvalidParameterError(
        "dataSourceLength must be a non-negative integer",
        "dataSourceLength",
        dataSourceLength
      );
    }
    this.length = dataSourceLength;
  }

  *[Symbol.iterator](): Iterator<number> {
    for (let i = 0; i < this.length; i++) yield i;
  }
}

/**
 * Samples elements randomly from a given list of indices (subset),
 * without replacement.
 *
 * Every iteration yields a full permutation of the indices. Without a `seed`
 * each iteration draws from the global generator, so epochs differ. With a
 * `seed`, every iteration yields the same permutation.
 *
 * @param indices - Indices to draw from (non-negative integers). The array is
 *   copied, so later changes to it do not affect the sampler.
 * @param options - `seed` makes the permutation reproducible. With
 *   `reshuffleEachIteration: true` the seeded stream continues across iterations, so each
 *   epoch has a different permutation while the sequence of epochs stays reproducible.
 * @throws {InvalidParameterError} If `indices` is not an array of non-negative
 *   integers or `seed` is not a safe integer.
 *
 * @example
 * ```ts
 * const sampler = new SubsetRandomSampler([0, 2, 4, 6, 8]);
 * for (const idx of sampler) { ... } // random permutation of [0,2,4,6,8]
 * ```
 */
export class SubsetRandomSampler implements Sampler {
  private readonly indices: readonly number[];
  private readonly rngForPass: () => DatasetRng;
  readonly length: number;

  constructor(
    indices: ArrayLike<number>,
    options: { seed?: number; reshuffleEachIteration?: boolean } = {}
  ) {
    if (!isNumberList(indices)) {
      throw new InvalidParameterError("indices must be an array of integers", "indices", indices);
    }
    const copy = Array.from(indices);
    for (let i = 0; i < copy.length; i++) {
      const v = copy[i];
      if (v === undefined || !Number.isSafeInteger(v) || v < 0) {
        throw new InvalidParameterError(
          `indices[${i}] must be a non-negative integer; received ${v}`,
          "indices",
          v
        );
      }
    }
    this.indices = copy;
    this.rngForPass = createPassRng(
      normalizeOptionalSeed("seed", options.seed),
      resolveReshuffle(options.reshuffleEachIteration)
    );
    this.length = copy.length;
  }

  *[Symbol.iterator](): Iterator<number> {
    const arr = [...this.indices];
    shuffleInPlace(arr, this.rngForPass());
    for (const idx of arr) yield idx;
  }
}

/**
 * Samples elements according to given weights (probabilities),
 * with or without replacement.
 *
 * Weights need not sum to one; index `i` is drawn with probability
 * `weights[i] / sum(weights)`. Indices with weight 0 are never drawn. Without
 * a `seed`, each iteration draws from the global generator; with a `seed`,
 * every iteration yields the same sequence.
 *
 * @param weights - Non-negative finite weights, one per sample. The array is copied.
 * @param options.numSamples - Number of indices to draw per iteration. Defaults to `weights.length`.
 * @param options.replacement - Draw with replacement (default `true`). Without replacement,
 *   `numSamples` cannot exceed the number of strictly positive weights.
 * @param options.seed - Seed for reproducible draws.
 * @param options.reshuffleEachIteration - With a `seed`, continue the seeded stream across
 *   iterations so each epoch draws a different sample while the sequence stays reproducible
 *   (default `false`).
 * @throws {InvalidParameterError} If the weights or options are invalid.
 *
 * @example
 * ```ts
 * // Over-sample minority class
 * const sampler = new WeightedRandomSampler(
 *   [0.1, 0.1, 0.4, 0.4],
 *   { numSamples: 8, replacement: true }
 * );
 * ```
 */
export class WeightedRandomSampler implements Sampler {
  private readonly weights: readonly number[];
  private readonly numSamples: number;
  private readonly replacement: boolean;
  private readonly rngForPass: () => DatasetRng;
  readonly length: number;

  constructor(
    weights: ArrayLike<number>,
    options: {
      numSamples?: number;
      replacement?: boolean;
      seed?: number;
      reshuffleEachIteration?: boolean;
    } = {}
  ) {
    if (!isNumberList(weights) || weights.length === 0) {
      throw new InvalidParameterError(
        "weights must be a non-empty array of numbers",
        "weights",
        weights
      );
    }
    const copy = Array.from(weights);
    let total = 0;
    let nPositive = 0;
    for (let i = 0; i < copy.length; i++) {
      const w = copy[i];
      if (w === undefined || !Number.isFinite(w) || w < 0) {
        throw new InvalidParameterError(
          `weights[${i}] must be a non-negative finite number; received ${w}`,
          "weights",
          w
        );
      }
      total += w;
      if (w > 0) nPositive++;
    }
    if (!(total > 0) || !Number.isFinite(total)) {
      throw new InvalidParameterError(
        "weights must sum to a positive finite number",
        "weights",
        total
      );
    }

    const replacement = options.replacement ?? true;
    if (options.replacement !== undefined) assertBoolean("replacement", replacement);
    const numSamples = options.numSamples ?? copy.length;
    assertPositiveInt("numSamples", numSamples);
    if (!replacement && numSamples > nPositive) {
      throw new InvalidParameterError(
        `numSamples (${numSamples}) cannot exceed the number of positive weights (${nPositive}) without replacement`,
        "numSamples",
        numSamples
      );
    }

    this.weights = copy;
    this.replacement = replacement;
    this.numSamples = numSamples;
    this.rngForPass = createPassRng(
      normalizeOptionalSeed("seed", options.seed),
      resolveReshuffle(options.reshuffleEachIteration)
    );
    this.length = numSamples;
  }

  *[Symbol.iterator](): Iterator<number> {
    const rng = this.rngForPass();
    const n = this.weights.length;
    const weights = this.weights;

    if (this.replacement) {
      // Cumulative distribution; draw r in [0, total) and binary-search the first
      // index whose cumulative weight exceeds r.
      const cumWeights = new Float64Array(n);
      let total = 0;
      for (let i = 0; i < n; i++) {
        total += weights[i] as number;
        cumWeights[i] = total;
      }
      for (let s = 0; s < this.numSamples; s++) {
        const r = rng() * total;
        let lo = 0;
        let hi = n - 1;
        while (lo < hi) {
          const mid = (lo + hi) >>> 1;
          if ((cumWeights[mid] as number) <= r) lo = mid + 1;
          else hi = mid;
        }
        // r * total can round up to exactly total; never return a zero-weight tail index.
        while (lo > 0 && weights[lo] === 0) lo--;
        yield lo;
      }
      return;
    }

    // Without replacement: successive weighted draws from the remaining items.
    // A binary sum tree over the weights gives O(log n) per draw. Every node
    // is recomputed from its children when an item is removed (never decremented),
    // so floating-point drift cannot accumulate and the descent cannot fall off the end.
    let size = 1;
    while (size < n) size <<= 1;
    const tree = new Float64Array(2 * size);
    for (let i = 0; i < n; i++) tree[size + i] = weights[i] as number;
    for (let i = size - 1; i >= 1; i--) {
      tree[i] = (tree[2 * i] as number) + (tree[2 * i + 1] as number);
    }
    for (let s = 0; s < this.numSamples; s++) {
      const remaining = tree[1] as number;
      let r = rng() * remaining;
      let node = 1;
      while (node < size) {
        const left = tree[2 * node] as number;
        const right = tree[2 * node + 1] as number;
        if (right <= 0 || (left > 0 && left > r)) {
          node = 2 * node;
        } else {
          r -= left;
          node = 2 * node + 1;
        }
      }
      yield node - size;
      tree[node] = 0;
      for (let i = node >> 1; i >= 1; i >>= 1) {
        tree[i] = (tree[2 * i] as number) + (tree[2 * i + 1] as number);
      }
    }
  }
}
