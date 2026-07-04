/**
 * Dataset samplers for controlling index generation in DataLoader.
 *
 * @module datasets/samplers
 * @see {@link https://deepbox.dev/docs/datasets-dataloader | Deepbox Samplers}
 */

import { InvalidParameterError } from "../core/errors";
import { createRng, shuffleInPlace } from "./utils";

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
 * @example
 * ```ts
 * const sampler = new SequentialSampler(10);
 * for (const idx of sampler) { ... } // 0, 1, 2, ..., 9
 * ```
 */
export class SequentialSampler implements Sampler {
  readonly length: number;

  constructor(dataSourceLength: number) {
    if (!Number.isInteger(dataSourceLength) || dataSourceLength < 0) {
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
 * @example
 * ```ts
 * const sampler = new SubsetRandomSampler([0, 2, 4, 6, 8]);
 * for (const idx of sampler) { ... } // random permutation of [0,2,4,6,8]
 * ```
 */
export class SubsetRandomSampler implements Sampler {
  private readonly indices: readonly number[];
  private readonly seed: number | undefined;
  readonly length: number;

  constructor(indices: readonly number[], options: { seed?: number } = {}) {
    if (!Array.isArray(indices)) {
      throw new InvalidParameterError("indices must be an array of integers", "indices", indices);
    }
    for (let i = 0; i < indices.length; i++) {
      const v = indices[i];
      if (v === undefined || !Number.isInteger(v) || v < 0) {
        throw new InvalidParameterError(
          `indices[${i}] must be a non-negative integer; received ${v}`,
          "indices",
          v
        );
      }
    }
    this.indices = indices;
    this.seed = options.seed;
    this.length = indices.length;
  }

  *[Symbol.iterator](): Iterator<number> {
    const arr = [...this.indices];
    const rng = createRng(this.seed);
    shuffleInPlace(arr, rng);
    for (const idx of arr) yield idx;
  }
}

/**
 * Samples elements according to given weights (probabilities),
 * with or without replacement.
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
  private readonly seed: number | undefined;
  readonly length: number;

  constructor(
    weights: readonly number[],
    options: {
      numSamples?: number;
      replacement?: boolean;
      seed?: number;
    } = {}
  ) {
    if (!Array.isArray(weights) || weights.length === 0) {
      throw new InvalidParameterError(
        "weights must be a non-empty array of numbers",
        "weights",
        weights
      );
    }
    for (let i = 0; i < weights.length; i++) {
      const w = weights[i];
      if (w === undefined || !Number.isFinite(w) || w < 0) {
        throw new InvalidParameterError(
          `weights[${i}] must be a non-negative finite number; received ${w}`,
          "weights",
          w
        );
      }
    }
    const total = weights.reduce((a, b) => a + b, 0);
    if (total <= 0) {
      throw new InvalidParameterError("weights must sum to a positive number", "weights", total);
    }

    this.weights = weights;
    this.replacement = options.replacement ?? true;
    this.numSamples = options.numSamples ?? weights.length;
    this.seed = options.seed;

    if (!Number.isInteger(this.numSamples) || this.numSamples < 1) {
      throw new InvalidParameterError(
        "numSamples must be a positive integer",
        "numSamples",
        this.numSamples
      );
    }
    if (!this.replacement && this.numSamples > weights.length) {
      throw new InvalidParameterError(
        `numSamples (${this.numSamples}) cannot exceed number of weights (${weights.length}) without replacement`,
        "numSamples",
        this.numSamples
      );
    }

    this.length = this.numSamples;
  }

  *[Symbol.iterator](): Iterator<number> {
    const rng = createRng(this.seed);
    const n = this.weights.length;

    // Build cumulative distribution
    const cumWeights = new Float64Array(n);
    let total = 0;
    for (let i = 0; i < n; i++) {
      total += this.weights[i]!;
      cumWeights[i] = total;
    }

    if (this.replacement) {
      for (let s = 0; s < this.numSamples; s++) {
        const r = rng() * total;
        // Binary search for the index
        let lo = 0,
          hi = n - 1;
        while (lo < hi) {
          const mid = (lo + hi) >>> 1;
          if ((cumWeights[mid] ?? 0) <= r) lo = mid + 1;
          else hi = mid;
        }
        yield lo;
      }
    } else {
      // Without replacement: use reservoir-style weighted sampling
      const available = Array.from({ length: n }, (_, i) => i);
      const localWeights = [...this.weights];
      let localTotal = total;

      for (let s = 0; s < this.numSamples; s++) {
        const r = rng() * localTotal;
        let cumSum = 0;
        let chosen = 0;
        for (let i = 0; i < available.length; i++) {
          cumSum += localWeights[available[i]!]!;
          if (cumSum > r) {
            chosen = i;
            break;
          }
        }
        const idx = available[chosen]!;
        yield idx;

        // Remove chosen from available
        localTotal -= localWeights[idx]!;
        available.splice(chosen, 1);
      }
    }
  }
}
