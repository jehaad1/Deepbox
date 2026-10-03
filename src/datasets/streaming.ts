/**
 * Lazy, out-of-core dataset streaming.
 *
 * Provides a {@link StreamingDataset} abstraction that yields samples (or
 * collated batches) on demand from a user-supplied generator / iterator,
 * reading from disk, network, or a computed generator, WITHOUT ever
 * materializing the whole corpus in memory.
 *
 * The pipeline mirrors the ergonomics of `tf.data` / PyTorch `IterableDataset`:
 *
 * - {@link StreamingDataset.map | map}: transform each sample lazily.
 * - {@link StreamingDataset.shuffle | shuffle}: window/reservoir shuffle for
 *   streams (a fixed-size buffer, not a full-corpus permutation).
 * - {@link StreamingDataset.batch | batch}: group consecutive samples and
 *   collate them into batched {@link Tensor}s.
 * - {@link StreamingDataset.prefetch | prefetch}: read ahead N elements
 *   concurrently (async only), overlapping I/O with compute.
 *
 * A dataset built from a synchronous source is iterable both synchronously
 * (`for (const x of ds)`) and asynchronously (`for await (const x of ds)`).
 * Any pipeline that includes {@link StreamingDataset.prefetch | prefetch}, or
 * is built from an async source, is async-only.
 *
 * @module datasets/streaming
 * @see {@link https://deepbox.dev/docs/datasets-dataloader | Deepbox documentation}
 */

import { DeepboxError, InvalidParameterError } from "../core/errors";
import { type Tensor, tensor } from "../ndarray";
import { __randomBelow } from "../random/random";
import { assertBoolean, createPassRng, normalizeOptionalSeed } from "./utils";

/**
 * A raw stream sample. Either an unlabeled feature vector (`number[]`), or a
 * `[features, label]` pair where the label is a scalar class index / regression
 * value (`number`) or a multi-output vector (`number[]`).
 */
export type StreamSample =
  | readonly number[]
  | readonly [ArrayLike<number>, number | readonly number[]];

/** A collated mini-batch: features, optionally paired with targets. */
export type Batch = [Tensor] | [Tensor, Tensor];

/**
 * A function that merges an array of raw samples into a single batch.
 *
 * @typeParam S - The raw sample type produced by the stream.
 */
export type StreamCollateFn<S> = (samples: S[]) => Batch;

/** A source that can be (re)iterated synchronously. */
export type SyncFactory<S> = () => Iterable<S>;

/** A source that can be (re)iterated asynchronously. */
export type AsyncFactory<S> = () => AsyncIterable<S>;

// ─── Default collation ───────────────────────────────────────────────────────

function isArrayLikeNumbers(v: unknown): v is ArrayLike<number> {
  // Typed arrays count; DataView does not (it has no element-wise `length`).
  return Array.isArray(v) || (ArrayBuffer.isView(v) && !(v instanceof DataView));
}

function isLabeledSample(v: unknown): boolean {
  return Array.isArray(v) && v.length === 2 && isArrayLikeNumbers(v[0]);
}

const INT32_MIN = -2147483648;
const INT32_MAX = 2147483647;

/**
 * Pick the dtype for scalar labels: `int32` for integral values in int32 range
 * (classification), `float64` for integral values outside it (exact), else the
 * default float dtype.
 */
function scalarLabelTensor(labels: number[]): Tensor {
  let allInt = true;
  let inInt32 = true;
  for (const v of labels) {
    if (!Number.isInteger(v)) {
      allInt = false;
      break;
    }
    if (v < INT32_MIN || v > INT32_MAX) inInt32 = false;
  }
  if (!allInt) return tensor(labels);
  return tensor(labels, { dtype: inInt32 ? "int32" : "float64" });
}

/**
 * Default collation for {@link StreamSample}s.
 *
 * - `[features, label]` tuples become `[X, y]` where `X` has shape
 *   `[batch, nFeatures]` and `y` has shape `[batch]` (scalar labels) or
 *   `[batch, nOutputs]` (vector labels). Integer scalar labels are collated as
 *   `int32` (classification); otherwise as the default float dtype
 *   (regression).
 * - Bare feature vectors become `[X]` with shape `[batch, nFeatures]`.
 *
 * Whether the batch is labeled is decided by its first sample; every other
 * sample must have the same layout and the same number of features (and
 * outputs).
 *
 * @throws {@link InvalidParameterError} If the batch is empty, or samples are
 *   mixed (labeled and unlabeled) or ragged.
 */
export function defaultCollate(samples: StreamSample[]): Batch {
  if (samples.length === 0) {
    throw new InvalidParameterError("Cannot collate an empty batch", "samples", samples);
  }

  const first = samples[0] as StreamSample;
  const labeled = isLabeledSample(first);

  if (labeled) {
    const xs: number[][] = new Array(samples.length);
    const rawLabels: (number | readonly number[])[] = new Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
      const sample = samples[i];
      if (!isLabeledSample(sample)) {
        throw new InvalidParameterError(
          `samples[${i}] is not a [features, label] pair, but samples[0] is`,
          "samples",
          sample
        );
      }
      const pair = sample as readonly [ArrayLike<number>, number | readonly number[]];
      xs[i] = Array.from(pair[0]);
      rawLabels[i] = pair[1];
    }

    const nFeatures = (xs[0] as number[]).length;
    for (let i = 1; i < xs.length; i++) {
      if ((xs[i] as number[]).length !== nFeatures) {
        throw new InvalidParameterError(
          `samples[${i}] has ${(xs[i] as number[]).length} features; expected ${nFeatures}`,
          "samples",
          samples[i]
        );
      }
    }

    const x = tensor(xs);

    // Multi-output (vector) labels → 2D float target.
    if (isArrayLikeNumbers(rawLabels[0])) {
      const nOutputs = (rawLabels[0] as ArrayLike<number>).length;
      const ys: number[][] = new Array(rawLabels.length);
      for (let i = 0; i < rawLabels.length; i++) {
        const l = rawLabels[i];
        if (!isArrayLikeNumbers(l) || l.length !== nOutputs) {
          throw new InvalidParameterError(
            `samples[${i}] label must be a vector of ${nOutputs} values like samples[0]`,
            "samples",
            l
          );
        }
        ys[i] = Array.from(l);
      }
      return [x, tensor(ys)];
    }

    for (let i = 0; i < rawLabels.length; i++) {
      if (typeof rawLabels[i] !== "number") {
        throw new InvalidParameterError(
          `samples[${i}] label must be a number like samples[0]`,
          "samples",
          rawLabels[i]
        );
      }
    }
    return [x, scalarLabelTensor(rawLabels as number[])];
  }

  const xs: number[][] = new Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const sample = samples[i];
    if (!isArrayLikeNumbers(sample) || isLabeledSample(sample)) {
      throw new InvalidParameterError(
        `samples[${i}] must be a feature vector (array of numbers)`,
        "samples",
        sample
      );
    }
    xs[i] = Array.from(sample);
  }
  const nFeatures = (xs[0] as number[]).length;
  for (let i = 1; i < xs.length; i++) {
    if ((xs[i] as number[]).length !== nFeatures) {
      throw new InvalidParameterError(
        `samples[${i}] has ${(xs[i] as number[]).length} features; expected ${nFeatures}`,
        "samples",
        samples[i]
      );
    }
  }
  return [tensor(xs)];
}

// ─── Internal lazy generators ────────────────────────────────────────────────

function* mapSync<S, T>(src: Iterable<S>, fn: (s: S, index: number) => T): Generator<T> {
  let i = 0;
  for (const item of src) {
    yield fn(item, i++);
  }
}

async function* mapAsync<S, T>(
  src: AsyncIterable<S>,
  fn: (s: S, index: number) => T
): AsyncGenerator<T> {
  let i = 0;
  for await (const item of src) {
    yield fn(item, i++);
  }
}

/**
 * Window/reservoir shuffle: maintains a buffer of at most `bufferSize` items,
 * emitting a uniformly-random buffered item each time a new one arrives, then
 * draining the buffer in random order. Memory stays bounded by `bufferSize`
 * regardless of stream length.
 */
function* shuffleSync<S>(src: Iterable<S>, bufferSize: number, rng: () => number): Generator<S> {
  const buf: S[] = [];
  for (const item of src) {
    if (buf.length < bufferSize) {
      buf.push(item);
      continue;
    }
    const j = __randomBelow(rng, buf.length);
    yield buf[j] as S;
    buf[j] = item;
  }
  while (buf.length > 0) {
    const j = __randomBelow(rng, buf.length);
    yield buf[j] as S;
    const last = buf.pop() as S;
    if (j < buf.length) buf[j] = last;
  }
}

async function* shuffleAsync<S>(
  src: AsyncIterable<S>,
  bufferSize: number,
  rng: () => number
): AsyncGenerator<S> {
  const buf: S[] = [];
  for await (const item of src) {
    if (buf.length < bufferSize) {
      buf.push(item);
      continue;
    }
    const j = __randomBelow(rng, buf.length);
    yield buf[j] as S;
    buf[j] = item;
  }
  while (buf.length > 0) {
    const j = __randomBelow(rng, buf.length);
    yield buf[j] as S;
    const last = buf.pop() as S;
    if (j < buf.length) buf[j] = last;
  }
}

function* batchSync<S, B>(
  src: Iterable<S>,
  batchSize: number,
  collate: (b: S[]) => B,
  dropLast: boolean
): Generator<B> {
  let cur: S[] = [];
  for (const item of src) {
    cur.push(item);
    if (cur.length === batchSize) {
      yield collate(cur);
      cur = [];
    }
  }
  if (cur.length > 0 && !dropLast) {
    yield collate(cur);
  }
}

async function* batchAsync<S, B>(
  src: AsyncIterable<S>,
  batchSize: number,
  collate: (b: S[]) => B,
  dropLast: boolean
): AsyncGenerator<B> {
  let cur: S[] = [];
  for await (const item of src) {
    cur.push(item);
    if (cur.length === batchSize) {
      yield collate(cur);
      cur = [];
    }
  }
  if (cur.length > 0 && !dropLast) {
    yield collate(cur);
  }
}

/**
 * Read-ahead of up to `n` in-flight elements. Pulls `n` `next()` promises
 * eagerly and keeps the pipeline full, overlapping the producer's latency
 * (disk/network) with downstream consumption.
 */
async function* prefetchAsync<S>(src: AsyncIterable<S>, n: number): AsyncGenerator<S> {
  const iterator = src[Symbol.asyncIterator]();
  const queue: Promise<IteratorResult<S>>[] = [];
  const pull = (): void => {
    const p = Promise.resolve(iterator.next());
    // A read that fails while an earlier element is still being consumed would
    // otherwise be reported as an unhandled rejection; the error is rethrown when
    // that element's turn comes.
    p.catch(() => {});
    queue.push(p);
  };
  try {
    for (let i = 0; i < n; i++) pull();
    while (queue.length > 0) {
      const result = await (queue.shift() as Promise<IteratorResult<S>>);
      if (result.done) break;
      pull();
      yield result.value;
    }
  } finally {
    // Drain outstanding requests and release the source iterator.
    await Promise.allSettled(queue);
    if (typeof iterator.return === "function") {
      await iterator.return();
    }
  }
}

function syncToAsync<S>(factory: SyncFactory<S>): AsyncFactory<S> {
  return () =>
    (async function* wrap() {
      for (const item of factory()) {
        yield item;
      }
    })();
}

// ─── StreamingDataset ────────────────────────────────────────────────────────

/**
 * A lazy, re-iterable stream of samples with a chainable transformation
 * pipeline. Nothing is read until iteration begins, and only a bounded working
 * set (shuffle buffer + in-flight prefetch + current batch) is ever held in
 * memory, so corpora far larger than RAM can be processed.
 *
 * Construct one with {@link iterableDataset} (sync source) or
 * {@link asyncIterableDataset} (async source), then chain
 * {@link StreamingDataset.map | map}, {@link StreamingDataset.shuffle | shuffle},
 * {@link StreamingDataset.batch | batch}, and
 * {@link StreamingDataset.prefetch | prefetch}.
 *
 * @typeParam S - The element type produced by this stage of the pipeline.
 *
 * @example
 * ```ts
 * import { iterableDataset, defaultCollate } from 'deepbox/datasets';
 *
 * // A 1e9-row corpus we never fully materialize:
 * const stream = iterableDataset(function* () {
 *   for (let i = 0; i < 1_000_000_000; i++) yield [[i, i * 2], i % 3] as const;
 * });
 *
 * const batches = stream.shuffle(1024, 42).batch(32, defaultCollate);
 * for (const [x, y] of batches) {
 *   // train on [32, 2] / [32]
 *   break;
 * }
 * ```
 */
export class StreamingDataset<S> implements Iterable<S>, AsyncIterable<S> {
  /** Present only when the entire pipeline can be driven synchronously. */
  private readonly syncFactory: SyncFactory<S> | undefined;
  private readonly asyncFactory: AsyncFactory<S>;

  /** @internal Use {@link iterableDataset} / {@link asyncIterableDataset}. */
  private constructor(sync: SyncFactory<S> | undefined, async: AsyncFactory<S>) {
    this.syncFactory = sync;
    this.asyncFactory = async;
  }

  /** @internal Build from a synchronous, re-iterable source. */
  static _fromSync<S>(factory: SyncFactory<S>): StreamingDataset<S> {
    return new StreamingDataset<S>(factory, syncToAsync(factory));
  }

  /** @internal Build from an asynchronous, re-iterable source. */
  static _fromAsync<S>(factory: AsyncFactory<S>): StreamingDataset<S> {
    return new StreamingDataset<S>(undefined, factory);
  }

  /** Whether this pipeline supports synchronous iteration. */
  get isSync(): boolean {
    return this.syncFactory !== undefined;
  }

  /**
   * Synchronous iteration over the stream.
   *
   * @throws {@link DeepboxError} If the pipeline is async-only (built from an
   *   async source, or includes {@link StreamingDataset.prefetch | prefetch}).
   */
  [Symbol.iterator](): Iterator<S> {
    if (this.syncFactory === undefined) {
      throw new DeepboxError(
        "This streaming dataset is async-only (async source or prefetch stage); " +
          "iterate it with `for await` / Symbol.asyncIterator instead."
      );
    }
    return this.syncFactory()[Symbol.iterator]();
  }

  /** Asynchronous iteration over the stream (always available). */
  [Symbol.asyncIterator](): AsyncIterator<S> {
    return this.asyncFactory()[Symbol.asyncIterator]();
  }

  /**
   * Lazily transform every element.
   *
   * @param fn - Mapping applied to each element and its running index.
   * @returns A new stream of the mapped element type.
   */
  map<T>(fn: (sample: S, index: number) => T): StreamingDataset<T> {
    if (typeof fn !== "function") {
      throw new InvalidParameterError("map expects a function", "fn", fn);
    }
    const sync = this.syncFactory;
    const async = this.asyncFactory;
    if (sync !== undefined) {
      return StreamingDataset._fromSync<T>(() => mapSync(sync(), fn));
    }
    return StreamingDataset._fromAsync<T>(() => mapAsync(async(), fn));
  }

  /**
   * Window/reservoir shuffle over a bounded buffer.
   *
   * Emits a uniformly-random buffered element each time a new one arrives, so
   * at most `bufferSize` elements are held at once. Larger buffers approach a
   * true shuffle; a buffer of 1 is a no-op passthrough.
   *
   * @param bufferSize - Maximum elements held in the shuffle buffer (`>= 1`).
   * @param seed - Optional seed for deterministic, reproducible shuffling. With a
   *   seed, every pass over the stream uses the same order; without one, each pass
   *   draws from the global generator and the order differs.
   * @param options.reshuffleEachIteration - With a `seed`, continue one seeded random
   *   stream across passes instead of restarting it, so every pass has a different order
   *   while the sequence of passes stays reproducible (default `false`). Has no effect
   *   without a seed.
   * @returns A new, shuffled stream.
   * @throws {@link InvalidParameterError} If `bufferSize` is not a positive integer
   *   or `seed` is not a safe integer.
   */
  shuffle(
    bufferSize: number,
    seed?: number,
    options: { readonly reshuffleEachIteration?: boolean } = {}
  ): StreamingDataset<S> {
    if (!Number.isSafeInteger(bufferSize) || bufferSize < 1) {
      throw new InvalidParameterError(
        `bufferSize must be a positive integer; received ${bufferSize}`,
        "bufferSize",
        bufferSize
      );
    }
    normalizeOptionalSeed("seed", seed);
    const reshuffle = options.reshuffleEachIteration ?? false;
    if (options.reshuffleEachIteration !== undefined) {
      assertBoolean("reshuffleEachIteration", reshuffle);
    }
    const rngForPass = createPassRng(seed, reshuffle);
    const sync = this.syncFactory;
    const async = this.asyncFactory;
    if (sync !== undefined) {
      return StreamingDataset._fromSync<S>(() => shuffleSync(sync(), bufferSize, rngForPass()));
    }
    return StreamingDataset._fromAsync<S>(() => shuffleAsync(async(), bufferSize, rngForPass()));
  }

  /**
   * Group consecutive elements into collated mini-batches.
   *
   * @param batchSize - Number of samples per batch (`>= 1`).
   * @param collate - Merges a list of samples into a batch. Defaults to
   *   {@link defaultCollate}, which requires `S` to be a {@link StreamSample}.
   * @param options - `dropLast` discards a final short batch (default `false`).
   * @returns A new stream of collated batches.
   * @throws {@link InvalidParameterError} If `batchSize` is not a positive integer.
   */
  batch<B = Batch>(
    batchSize: number,
    collate?: (samples: S[]) => B,
    options: { dropLast?: boolean } = {}
  ): StreamingDataset<B> {
    if (!Number.isSafeInteger(batchSize) || batchSize < 1) {
      throw new InvalidParameterError(
        `batchSize must be a positive integer; received ${batchSize}`,
        "batchSize",
        batchSize
      );
    }
    if (collate !== undefined && typeof collate !== "function") {
      throw new InvalidParameterError("collate must be a function", "collate", collate);
    }
    const dropLast = options.dropLast ?? false;
    if (options.dropLast !== undefined) assertBoolean("dropLast", dropLast);
    const collateFn = collate ?? (defaultCollate as unknown as (samples: S[]) => B);

    const sync = this.syncFactory;
    const async = this.asyncFactory;
    if (sync !== undefined) {
      return StreamingDataset._fromSync<B>(() => batchSync(sync(), batchSize, collateFn, dropLast));
    }
    return StreamingDataset._fromAsync<B>(() =>
      batchAsync(async(), batchSize, collateFn, dropLast)
    );
  }

  /**
   * Read ahead up to `n` elements concurrently, overlapping the producer's
   * latency with downstream work. This stage is inherently asynchronous, so the
   * resulting stream is async-only.
   *
   * @param n - Maximum number of in-flight (read-ahead) elements (`>= 1`).
   * @returns A new async-only stream.
   * @throws {@link InvalidParameterError} If `n` is not a positive integer.
   */
  prefetch(n: number): StreamingDataset<S> {
    if (!Number.isSafeInteger(n) || n < 1) {
      throw new InvalidParameterError(`n must be a positive integer; received ${n}`, "n", n);
    }
    const async = this.asyncFactory;
    return StreamingDataset._fromAsync<S>(() => prefetchAsync(async(), n));
  }

  /**
   * Eagerly drain the stream into an array.
   *
   * @remarks Defeats the point of streaming; use only for small streams or in
   *   tests. Works whether the pipeline is sync or async.
   */
  async toArray(): Promise<S[]> {
    const out: S[] = [];
    for await (const item of this) {
      out.push(item);
    }
    return out;
  }
}

/**
 * Create a {@link StreamingDataset} from a synchronous, re-iterable source.
 *
 * @param factory - A function returning a fresh {@link Iterable} each call (e.g.
 *   a generator function). A fresh iterator is requested per epoch, so passing a
 *   factory, not a one-shot iterator, is what makes multi-epoch training work.
 *
 * @example
 * ```ts
 * const ds = iterableDataset(function* () {
 *   yield* [[[1, 2], 0], [[3, 4], 1]] as const;
 * });
 * ```
 */
export function iterableDataset<S>(factory: SyncFactory<S>): StreamingDataset<S> {
  if (typeof factory !== "function") {
    throw new InvalidParameterError(
      "iterableDataset expects a factory function returning a fresh iterable",
      "factory",
      factory
    );
  }
  return StreamingDataset._fromSync<S>(factory);
}

/**
 * Create a {@link StreamingDataset} from an asynchronous, re-iterable source
 * (e.g. reading rows from disk or a paginated network endpoint).
 *
 * @param factory - A function returning a fresh {@link AsyncIterable} each call.
 */
export function asyncIterableDataset<S>(factory: AsyncFactory<S>): StreamingDataset<S> {
  if (typeof factory !== "function") {
    throw new InvalidParameterError(
      "asyncIterableDataset expects a factory function returning a fresh async iterable",
      "factory",
      factory
    );
  }
  return StreamingDataset._fromAsync<S>(factory);
}
