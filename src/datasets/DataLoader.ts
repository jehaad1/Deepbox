import { DeepboxError, InvalidParameterError } from "../core/errors";
import { gather, Tensor, tensor } from "../ndarray";
import type { Sampler } from "./samplers";
import { type Batch, StreamingDataset } from "./streaming";
import {
  assertBoolean,
  assertPositiveInt,
  createRng,
  normalizeOptionalSeed,
  shuffleInPlace,
} from "./utils";

/** Function that merges a list of samples into a mini-batch. */
export type CollateFn<T = [Tensor] | [Tensor, Tensor]> = (batch: T[]) => T;

/** Configuration options for {@link DataLoader}. */
export type DataLoaderOptions = {
  batchSize?: number;
  shuffle?: boolean;
  dropLast?: boolean;
  seed?: number;
  sampler?: Sampler;
  // biome-ignore lint/suspicious/noExplicitAny: collate function works with any batch type
  collateFn?: CollateFn<any>;
};

/**
 * Configuration options for a {@link DataLoader} constructed over a
 * {@link StreamingDataset} (out-of-core / lazy source).
 */
export type StreamingDataLoaderOptions = {
  /** Samples per batch. Default: 1. */
  batchSize?: number;
  /**
   * Window shuffle-buffer size. When set (`>= 1`), samples are shuffled through
   * a bounded buffer of this many elements before batching. Omit to disable
   * shuffling. This is the streaming analogue of `shuffle: true`.
   */
  shuffleBufferSize?: number;
  /**
   * Read-ahead depth. When set (`>= 1`), up to this many batches are fetched
   * concurrently during asynchronous iteration (`for await`). Ignored during
   * synchronous iteration.
   */
  prefetch?: number;
  /** Drop a trailing short batch. Default: false. */
  dropLast?: boolean;
  /** Seed for deterministic shuffle-buffer ordering. */
  seed?: number;
  /**
   * Merge raw samples into a batch. Defaults to the stream's own
   * {@link import('./streaming').defaultCollate | defaultCollate}.
   */
  // biome-ignore lint/suspicious/noExplicitAny: collate accepts the stream's raw sample type
  collateFn?: (samples: any[]) => Batch;
};

/**
 * Data loader for batching and shuffling datasets.
 *
 * Provides efficient iteration over datasets with support for
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
 * @see {@link https://deepbox.dev/docs/datasets-dataloader | Deepbox DataLoader}
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
  private sampler: Sampler | undefined;
  // biome-ignore lint/suspicious/noExplicitAny: collate function works with any batch type
  private collateFn: CollateFn<any> | undefined;

  // ─── Streaming mode ─────────────────────────────────────────────────────
  /** Set when the loader wraps a lazy {@link StreamingDataset} instead of tensors. */
  private stream: StreamingDataset<unknown> | undefined;
  private streamShuffleBuffer: number | undefined;
  private streamPrefetch: number | undefined;
  // biome-ignore lint/suspicious/noExplicitAny: collate accepts the stream's raw sample type
  private streamCollate: ((samples: any[]) => Batch) | undefined;

  constructor(X: Tensor, y: TTarget, options?: DataLoaderOptions);
  constructor(X: Tensor, options?: DataLoaderOptions);
  constructor(dataset: StreamingDataset<unknown>, options?: StreamingDataLoaderOptions);
  constructor(
    source: Tensor | StreamingDataset<unknown>,
    yOrOptions?: TTarget | DataLoaderOptions | StreamingDataLoaderOptions,
    options?: DataLoaderOptions
  ) {
    // ── Streaming source ────────────────────────────────────────────────
    if (source instanceof StreamingDataset) {
      this.stream = source;
      // Unused tensor-mode fields kept consistent for the type checker.
      this.X = undefined as unknown as Tensor;
      this.y = undefined;
      this.indices = [];
      this.nSamples = 0;
      this.shuffle = false;
      this.sampler = undefined;
      this.collateFn = undefined;

      const sOpts = (yOrOptions as StreamingDataLoaderOptions | undefined) ?? {};
      if (sOpts === null || typeof sOpts !== "object" || Array.isArray(sOpts)) {
        throw new InvalidParameterError(
          "options must be an object when provided",
          "options",
          sOpts
        );
      }

      this.batchSize = sOpts.batchSize ?? 1;
      assertPositiveInt("batchSize", this.batchSize);
      this.dropLast = sOpts.dropLast ?? false;
      if (sOpts.dropLast !== undefined) assertBoolean("dropLast", this.dropLast);
      this.seed = normalizeOptionalSeed("seed", sOpts.seed);
      this.streamCollate = sOpts.collateFn;

      if (sOpts.shuffleBufferSize !== undefined) {
        assertPositiveInt("shuffleBufferSize", sOpts.shuffleBufferSize);
        this.streamShuffleBuffer = sOpts.shuffleBufferSize;
      }
      if (sOpts.prefetch !== undefined) {
        assertPositiveInt("prefetch", sOpts.prefetch);
        this.streamPrefetch = sOpts.prefetch;
      }
      return;
    }

    const X = source;
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
    this.sampler = opts.sampler;
    this.collateFn = opts.collateFn;

    if (this.sampler && this.shuffle) {
      throw new InvalidParameterError(
        "Cannot use both sampler and shuffle=true; the sampler controls index order",
        "sampler"
      );
    }

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
   *
   * @throws {@link DeepboxError} In streaming mode, where the length is unknown
   *   without consuming the (potentially unbounded) source.
   */
  get length(): number {
    if (this.stream !== undefined) {
      throw new DeepboxError(
        "length is undefined for a streaming DataLoader: the number of batches " +
          "is not known without consuming the source. Iterate the loader instead."
      );
    }
    return this.dropLast
      ? Math.floor(this.nSamples / this.batchSize)
      : Math.ceil(this.nSamples / this.batchSize);
  }

  /**
   * Build the collated-batch pipeline over the streaming source.
   *
   * @param allowPrefetch - Whether to append the async read-ahead stage
   *   (meaningful only for asynchronous iteration).
   */
  private buildStreamPipeline(allowPrefetch: boolean): StreamingDataset<Batch> {
    const stream = this.stream;
    if (stream === undefined) {
      throw new DeepboxError("Internal error: buildStreamPipeline called without a stream");
    }
    let s: StreamingDataset<unknown> = stream;
    if (this.streamShuffleBuffer !== undefined) {
      s = s.shuffle(this.streamShuffleBuffer, this.seed);
    }
    let batched = s.batch<Batch>(this.batchSize, this.streamCollate, { dropLast: this.dropLast });
    if (allowPrefetch && this.streamPrefetch !== undefined) {
      batched = batched.prefetch(this.streamPrefetch);
    }
    return batched;
  }

  [Symbol.iterator](): IterableIterator<TTarget extends Tensor ? [Tensor, Tensor] : [Tensor]> {
    if (this.stream !== undefined) {
      return this.buildStreamPipeline(false)[Symbol.iterator]() as unknown as IterableIterator<
        TTarget extends Tensor ? [Tensor, Tensor] : [Tensor]
      >;
    }
    return (this.y === undefined ? this.iterateX() : this.iterateXY()) as IterableIterator<
      TTarget extends Tensor ? [Tensor, Tensor] : [Tensor]
    >;
  }

  /**
   * Asynchronous iteration over batches.
   *
   * Streaming loaders drive their lazy pipeline (including
   * {@link StreamingDataLoaderOptions.prefetch | prefetch}); in-memory loaders
   * yield the same batches as synchronous iteration. This lets a single
   * consumer (e.g. {@link import('../nn/Trainer').Trainer.fitAsync | Trainer.fitAsync})
   * work over both.
   */
  async *[Symbol.asyncIterator](): AsyncIterableIterator<
    TTarget extends Tensor ? [Tensor, Tensor] : [Tensor]
  > {
    if (this.stream !== undefined) {
      for await (const batch of this.buildStreamPipeline(true)) {
        yield batch as TTarget extends Tensor ? [Tensor, Tensor] : [Tensor];
      }
      return;
    }
    for (const batch of this) {
      yield batch;
    }
  }

  private prepareIteration(): { indices: number[]; nBatches: number } {
    let indices: number[];

    if (this.sampler) {
      indices = [...this.sampler];
    } else {
      indices = [...this.indices];
      if (this.shuffle) {
        const rng = createRng(this.seed);
        shuffleInPlace(indices, rng);
      }
    }

    const total = indices.length;
    const nBatches = this.dropLast
      ? Math.floor(total / this.batchSize)
      : Math.ceil(total / this.batchSize);

    return { indices, nBatches };
  }

  private *iterateX(): IterableIterator<[Tensor]> {
    const { indices, nBatches } = this.prepareIteration();

    for (let i = 0; i < nBatches; i++) {
      const start = i * this.batchSize;
      const end = Math.min(start + this.batchSize, indices.length);
      const batchIndices = indices.slice(start, end);

      const indexTensor = tensor(batchIndices, { dtype: "int32" });
      const xBatch = gather(this.X, indexTensor, 0);

      if (this.collateFn) {
        yield this.collateFn([[xBatch]]) as [Tensor];
      } else {
        yield [xBatch];
      }
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
      const end = Math.min(start + this.batchSize, indices.length);
      const batchIndices = indices.slice(start, end);

      const indexTensor = tensor(batchIndices, { dtype: "int32" });
      const xBatch = gather(this.X, indexTensor, 0);
      const yBatch = gather(y, indexTensor, 0);

      if (this.collateFn) {
        yield this.collateFn([[xBatch, yBatch]]) as [Tensor, Tensor];
      } else {
        yield [xBatch, yBatch];
      }
    }
  }
}
