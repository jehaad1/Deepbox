import {
  DeepboxError,
  type DType,
  dtypeToTypedArrayCtor,
  IndexError,
  InvalidParameterError,
} from "../core";
import { gather, Tensor, tensor } from "../ndarray";
import { isContiguous } from "../ndarray/tensor/strides";
import type { Sampler } from "./samplers";
import { type Batch, StreamingDataset } from "./streaming";
import {
  assertBoolean,
  assertPositiveInt,
  createPassRng,
  type DatasetRng,
  normalizeOptionalSeed,
  shuffleInPlace,
} from "./utils";

/**
 * Function that post-processes a mini-batch.
 *
 * An in-memory {@link DataLoader} gathers each batch itself and calls the
 * collate function with a single-element list holding that batch
 * (`[[xBatch]]` or `[[xBatch, yBatch]]`); the returned value is what the
 * loader yields.
 */
export type CollateFn<T = [Tensor] | [Tensor, Tensor]> = (batch: T[]) => T;

/** Configuration options for {@link DataLoader}. */
export type DataLoaderOptions = {
  /** Samples per batch (positive integer). Default: 1. */
  batchSize?: number;
  /** Reshuffle the sample order at the start of every iteration. Default: false. */
  shuffle?: boolean;
  /** Drop a trailing batch that has fewer than `batchSize` samples. Default: false. */
  dropLast?: boolean;
  /** Seed for the shuffle. With a seed, every iteration uses the same order unless `reshuffleEachIteration` is set. */
  seed?: number;
  /**
   * With a `seed`, continue one seeded random stream across iterations, so every
   * epoch has a different order while the sequence of epochs stays reproducible
   * (the first epoch matches the default behavior). Default: false. Has no effect
   * without a seed or without `shuffle`.
   */
  reshuffleEachIteration?: boolean;
  /** Custom index order. Mutually exclusive with `shuffle: true`. */
  sampler?: Sampler;
  /** Post-process every batch, see {@link CollateFn}. */
  // biome-ignore lint/suspicious/noExplicitAny: collate function works with any batch type
  collateFn?: CollateFn<any>;
};

/** Dtypes whose tensor storage is one typed-array element per tensor element. */
const ROW_COPY_DTYPES: ReadonlySet<DType> = new Set<DType>([
  "float16",
  "bfloat16",
  "float32",
  "float64",
  "int32",
  "int64",
  "uint8",
  "bool",
]);

type RowStorage = {
  readonly length: number;
  subarray(begin: number, end: number): RowStorage;
  set(source: RowStorage, offset?: number): void;
};

/**
 * Select rows `indices` along axis 0 of `t`.
 *
 * Row-major CPU tensors take a block-copy fast path (one `set` per row);
 * everything else falls back to the general {@link gather}. Both paths return
 * a freshly allocated tensor and raise the same errors for bad indices.
 */
function gatherRows(t: Tensor, indices: readonly number[]): Tensor {
  const nRows = t.shape[0] ?? 0;
  const rowSize = nRows > 0 ? t.size / nRows : 0;
  const dtype = t.dtype;
  if (
    dtype === "string" ||
    t.device !== "cpu" ||
    !ROW_COPY_DTYPES.has(dtype) ||
    rowSize === 0 ||
    Array.isArray(t.data) ||
    !isContiguous(t.shape, t.strides)
  ) {
    return gather(t, tensor(indices as number[], { dtype: "int32" }), 0);
  }

  const Ctor = dtypeToTypedArrayCtor(dtype);
  const out = new Ctor(indices.length * rowSize);
  const src = t.data as unknown as RowStorage;
  const dst = out as unknown as RowStorage;
  for (let j = 0; j < indices.length; j++) {
    const idx = indices[j] as number;
    if (!Number.isInteger(idx)) {
      throw new InvalidParameterError(`sample index ${idx} is not an integer`, "indices", idx);
    }
    if (idx < 0 || idx >= nRows) {
      throw new IndexError(`index ${idx} is out of bounds for axis 0 with size ${nRows}`);
    }
    const start = t.offset + idx * rowSize;
    dst.set(src.subarray(start, start + rowSize), j * rowSize);
  }
  return Tensor.fromTypedArray({
    data: out,
    shape: [indices.length, ...t.shape.slice(1)],
    dtype,
    device: "cpu",
  });
}

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
   * With a `seed`, continue one seeded random stream across iterations, so every
   * pass has a different order while the sequence of passes stays reproducible.
   * Default: false (a seeded loader repeats the same order). Has no effect without
   * a seed or a `shuffleBufferSize`.
   */
  reshuffleEachIteration?: boolean;
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
 * - With a seed, all iterations produce identical shuffles (deterministic). Leave the seed unset
 * if every training epoch should see a different order, or set `reshuffleEachIteration: true` to
 * get a different but reproducible order in every epoch.
 * - Each batch is a copy of the selected rows; the source tensors are never modified.
 *
 * **Shuffling:**
 * - Uses Fisher-Yates shuffle algorithm for uniform random permutation.
 * - When `seed` is provided, shuffling is deterministic and reproducible across runs.
 * - Shuffle happens per iteration, not per construction.
 *
 * **Length:**
 * - `length` is the number of batches per iteration. With a `sampler` it is derived from
 * `sampler.length`.
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
  private rngForPass: () => DatasetRng;
  private reshuffleEachIteration: boolean;
  private shuffledStream: StreamingDataset<unknown> | undefined;
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
      if (
        sOpts === null ||
        typeof sOpts !== "object" ||
        Array.isArray(sOpts) ||
        sOpts instanceof Tensor
      ) {
        throw new InvalidParameterError(
          "options must be an object when provided; a streaming DataLoader takes no y tensor",
          "options",
          sOpts
        );
      }

      this.batchSize = sOpts.batchSize ?? 1;
      assertPositiveInt("batchSize", this.batchSize);
      this.dropLast = sOpts.dropLast ?? false;
      if (sOpts.dropLast !== undefined) assertBoolean("dropLast", this.dropLast);
      this.seed = normalizeOptionalSeed("seed", sOpts.seed);
      this.reshuffleEachIteration = sOpts.reshuffleEachIteration ?? false;
      if (sOpts.reshuffleEachIteration !== undefined) {
        assertBoolean("reshuffleEachIteration", this.reshuffleEachIteration);
      }
      this.rngForPass = createPassRng(this.seed, false);
      if (sOpts.collateFn !== undefined && typeof sOpts.collateFn !== "function") {
        throw new InvalidParameterError("collateFn must be a function", "collateFn");
      }
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

    if (!(source instanceof Tensor)) {
      throw new InvalidParameterError("X must be a Tensor or a StreamingDataset", "X", source);
    }
    const X = source;
    this.X = X;

    let rawOpts: DataLoaderOptions | undefined;

    if (yOrOptions instanceof Tensor) {
      this.y = yOrOptions;
      rawOpts = options;
    } else {
      this.y = undefined;
      if (Array.isArray(yOrOptions)) {
        throw new InvalidParameterError(
          "y must be a Tensor (or omitted); convert arrays with tensor()",
          "y",
          yOrOptions
        );
      }
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
    this.reshuffleEachIteration = opts.reshuffleEachIteration ?? false;
    if (opts.reshuffleEachIteration !== undefined) {
      assertBoolean("reshuffleEachIteration", this.reshuffleEachIteration);
    }
    this.rngForPass = createPassRng(this.seed, this.reshuffleEachIteration);
    this.sampler = opts.sampler;
    this.collateFn = opts.collateFn;

    assertPositiveInt("batchSize", this.batchSize);
    if (opts.shuffle !== undefined) assertBoolean("shuffle", this.shuffle);
    if (opts.dropLast !== undefined) assertBoolean("dropLast", this.dropLast);
    if (this.collateFn !== undefined && typeof this.collateFn !== "function") {
      throw new InvalidParameterError("collateFn must be a function", "collateFn");
    }
    if (this.sampler !== undefined) {
      const sampler = this.sampler as Partial<Sampler> | null;
      if (
        sampler === null ||
        typeof sampler !== "object" ||
        typeof sampler[Symbol.iterator] !== "function" ||
        !Number.isInteger(sampler.length)
      ) {
        throw new InvalidParameterError(
          "sampler must be iterable and expose an integer length",
          "sampler"
        );
      }
    }

    if (this.sampler && this.shuffle) {
      throw new InvalidParameterError(
        "Cannot use both sampler and shuffle=true; the sampler controls index order",
        "sampler"
      );
    }

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
    const total = this.sampler ? this.sampler.length : this.nSamples;
    return this.dropLast ? Math.floor(total / this.batchSize) : Math.ceil(total / this.batchSize);
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
      if (this.reshuffleEachIteration) {
        // One shuffled stream is reused so its seeded generator advances between passes.
        this.shuffledStream ??= stream.shuffle(this.streamShuffleBuffer, this.seed, {
          reshuffleEachIteration: true,
        });
        s = this.shuffledStream;
      } else {
        s = s.shuffle(this.streamShuffleBuffer, this.seed);
      }
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
        shuffleInPlace(indices, this.rngForPass());
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

      const xBatch = gatherRows(this.X, batchIndices);

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

      const xBatch = gatherRows(this.X, batchIndices);
      const yBatch = gatherRows(y, batchIndices);

      if (this.collateFn) {
        yield this.collateFn([[xBatch, yBatch]]) as [Tensor, Tensor];
      } else {
        yield [xBatch, yBatch];
      }
    }
  }
}
