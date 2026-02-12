import { InvalidParameterError } from "../core/errors";
import { gather, Tensor, tensor } from "../ndarray";
import {
  assertBoolean,
  assertPositiveInt,
  createRng,
  normalizeOptionalSeed,
  shuffleInPlace,
} from "./utils";

export type DataLoaderOptions = {
  batchSize?: number;
  shuffle?: boolean;
  dropLast?: boolean;
  seed?: number;
};

/**
 * Data loader for batching and shuffling datasets.
 *
 * Similar to PyTorch's DataLoader. Provides efficient iteration over datasets with support for
 * batching, shuffling, and deterministic reproducibility.
 *
 * @remarks
 * **Iteration Behavior:**
 * - Each iteration creates a fresh shuffle (if enabled), so multiple iterations over the same
 * loader will produce different orderings unless a seed is provided.
 * - With a seed, all iterations produce identical shuffles (deterministic).
 * - The underlying tensors are not copied; batches reference the same data via gather operations.
 *
 * **Shuffling:**
 * - Uses Fisher-Yates shuffle algorithm for uniform random permutation.
 * - When `seed` is provided, shuffling is deterministic and reproducible across runs.
 * - Shuffle happens per iteration, not per construction.
 *
 * @example
 * ```ts
 * import { DataLoader } from 'deepbox/datasets';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1, 2], [3, 4], [5, 6], [7, 8]]);
 * const y = tensor([0, 1, 0, 1]);
 *
 * // Training loop with shuffling
 * const loader = new DataLoader(X, y, {
 * batchSize: 2,
 * shuffle: true,
 * seed: 42  // Deterministic shuffling
 * });
 *
 * for (const [xBatch, yBatch] of loader) {
 * // Train on batch
 * console.log(xBatch.shape, yBatch.shape);  // [2, 2], [2]
 * }
 * ```
 *
 * @example
 * ```ts
 * // Inference without labels
 * const testLoader = new DataLoader(X, undefined, {
 * batchSize: 4,
 * shuffle: false
 * });
 *
 * for (const [xBatch] of testLoader) {
 * // Make predictions
 * }
 * ```
 *
 * @see {@link https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader | PyTorch DataLoader}
 */
export class DataLoader<TTarget extends Tensor | undefined = undefined> {
  private X: Tensor;
  private y: Tensor | undefined;
  private batchSize: number;
  private shuffle: boolean;
  private dropLast: boolean;
  private indices: number[];
  private seed: number | undefined;
  private nSamples: number;

  constructor(X: Tensor, y: TTarget, options?: DataLoaderOptions);
  constructor(X: Tensor, options?: DataLoaderOptions);
  constructor(X: Tensor, yOrOptions?: TTarget | DataLoaderOptions, options?: DataLoaderOptions) {
    this.X = X;

    let rawOpts: DataLoaderOptions | undefined;

    if (yOrOptions instanceof Tensor) {
      this.y = yOrOptions;
      rawOpts = options;
    } else {
      this.y = undefined;
      // supports: new DataLoader(X, options) AND new DataLoader(X, undefined, options)
      rawOpts = yOrOptions === undefined ? options : yOrOptions;
    }

    if (rawOpts !== undefined) {
      if (rawOpts === null || typeof rawOpts !== "object" || Array.isArray(rawOpts)) {
        throw new InvalidParameterError(
          "options must be an object when provided",
          "options",
          rawOpts
        );
      }
    }

    const opts: DataLoaderOptions = rawOpts ?? {};

    this.batchSize = opts.batchSize ?? 1;
    this.shuffle = opts.shuffle ?? false;
    this.dropLast = opts.dropLast ?? false;
    this.seed = normalizeOptionalSeed("seed", opts.seed);

    assertPositiveInt("batchSize", this.batchSize);
    if (opts.shuffle !== undefined) assertBoolean("shuffle", this.shuffle);
    if (opts.dropLast !== undefined) assertBoolean("dropLast", this.dropLast);

    if (this.X.ndim === 0) {
      throw new InvalidParameterError("X must have at least 1 dimension (samples axis)", "X");
    }

    const nSamples = this.X.shape[0];
    if (nSamples === undefined || nSamples === 0) {
      throw new InvalidParameterError("X must have at least 1 sample", "X", nSamples);
    }

    const y = this.y;
    if (y !== undefined) {
      if (y.ndim === 0) {
        throw new InvalidParameterError("y must have at least 1 dimension (samples axis)", "y");
      }
      const ySamples = y.shape[0];
      if (ySamples !== nSamples) {
        throw new InvalidParameterError(
          `X and y must have the same number of samples; X has ${nSamples}, y has ${ySamples}`,
          "y",
          ySamples
        );
      }
    }

    this.nSamples = nSamples;
    this.indices = Array.from({ length: nSamples }, (_, i) => i);
  }

  /**
   * Number of batches in the data loader.
   */
  get length(): number {
    return this.dropLast
      ? Math.floor(this.nSamples / this.batchSize)
      : Math.ceil(this.nSamples / this.batchSize);
  }

  [Symbol.iterator](this: DataLoader<Tensor>): IterableIterator<[Tensor, Tensor]>;
  [Symbol.iterator](this: DataLoader<undefined>): IterableIterator<[Tensor]>;
  [Symbol.iterator](): IterableIterator<[Tensor] | [Tensor, Tensor]> {
    return this.y === undefined ? this.iterateX() : this.iterateXY();
  }

  private prepareIteration(): { indices: number[]; nBatches: number } {
    const indices = [...this.indices];

    if (this.shuffle) {
      const rng = createRng(this.seed);
      shuffleInPlace(indices, rng);
    }

    const nBatches = this.dropLast
      ? Math.floor(this.nSamples / this.batchSize)
      : Math.ceil(this.nSamples / this.batchSize);

    return { indices, nBatches };
  }

  private *iterateX(): IterableIterator<[Tensor]> {
    const { indices, nBatches } = this.prepareIteration();

    for (let i = 0; i < nBatches; i++) {
      const start = i * this.batchSize;
      const end = Math.min(start + this.batchSize, this.nSamples);
      const batchIndices = indices.slice(start, end);

      const indexTensor = tensor(batchIndices, { dtype: "int32" });
      const xBatch = gather(this.X, indexTensor, 0);

      yield [xBatch];
    }
  }

  private *iterateXY(): IterableIterator<[Tensor, Tensor]> {
    const { indices, nBatches } = this.prepareIteration();
    const y = this.y;
    if (y === undefined) {
      throw new InvalidParameterError("Internal error: expected y to be defined", "y");
    }

    for (let i = 0; i < nBatches; i++) {
      const start = i * this.batchSize;
      const end = Math.min(start + this.batchSize, this.nSamples);
      const batchIndices = indices.slice(start, end);

      const indexTensor = tensor(batchIndices, { dtype: "int32" });
      const xBatch = gather(this.X, indexTensor, 0);
      const yBatch = gather(y, indexTensor, 0);

      yield [xBatch, yBatch];
    }
  }
}
