/**
 * Task pool for batched map / reduce / filter style workloads.
 *
 * The pool splits work into at most `maxWorkers` chunks and runs them on the
 * calling thread, in order. No `worker_threads` or Web Workers are spawned:
 * callbacks are ordinary closures, receive their arguments by reference, and
 * may touch any outer state. It gives batching, ordering guarantees, status
 * counters and a termination switch, but not multi-core speedups.
 *
 * @module core/parallel
 * @see {@link https://deepbox.dev/docs/core-utils | Utilities, serialization & parallelism}
 */

import { DeepboxError } from "../errors/base";
import { InvalidParameterError } from "../errors/invalid_parameter";

type OsLike = {
  readonly availableParallelism?: () => number;
  readonly cpus?: () => readonly unknown[];
};

/** Look up `node:os` without a static import so browser bundles stay clean. */
function loadNodeOs(): OsLike | undefined {
  try {
    const proc = (globalThis as { process?: { getBuiltinModule?: (id: string) => unknown } })
      .process;
    const viaProcess = proc?.getBuiltinModule?.("node:os");
    if (viaProcess !== undefined && viaProcess !== null) return viaProcess as OsLike;
  } catch {
    // fall through to require
  }
  try {
    if (typeof require === "function") return require("node:os") as OsLike;
  } catch {
    // not a CommonJS-capable Node runtime
  }
  return undefined;
}

function detectCpuCount(): number {
  const os = loadNodeOs();
  if (os !== undefined) {
    try {
      const n = os.availableParallelism?.() ?? os.cpus?.().length ?? 0;
      if (Number.isInteger(n) && n > 0) return n;
    } catch {
      // fall through
    }
  }
  const nav = (globalThis as { navigator?: { hardwareConcurrency?: number } }).navigator;
  const hc = nav?.hardwareConcurrency;
  if (typeof hc === "number" && Number.isInteger(hc) && hc > 0) return hc;
  return 4;
}

/**
 * Result of a parallel task execution.
 */
export type TaskResult<T> = {
  readonly value: T;
  readonly workerId: number;
  readonly durationMs: number;
};

/**
 * Options for creating a WorkerPool.
 */
export type WorkerPoolOptions = {
  /** Maximum number of chunks work is split into. Positive integer. Defaults to the CPU count. */
  readonly maxWorkers?: number;
  /**
   * Task timeout in milliseconds (positive). Default: 30000 (30s).
   * Tasks run on the calling thread and cannot be pre-empted, so the value is
   * stored in {@link WorkerPool.taskTimeoutMs} but not enforced.
   */
  readonly taskTimeout?: number;
};

/**
 * Status information about the worker pool.
 */
export type PoolStatus = {
  readonly maxWorkers: number;
  readonly activeWorkers: number;
  /** Always 0: chunks run as soon as they are scheduled. */
  readonly pendingTasks: number;
  readonly completedTasks: number;
  readonly isTerminated: boolean;
};

/** Split `items` into at most `parts` contiguous chunks of near-equal size. */
function splitIntoChunks<T>(
  items: readonly T[],
  parts: number
): { readonly items: T[]; readonly start: number }[] {
  const chunks: { items: T[]; start: number }[] = [];
  if (items.length === 0) return chunks;
  const chunkSize = Math.ceil(items.length / Math.max(1, parts));
  for (let i = 0; i < items.length; i += chunkSize) {
    chunks.push({ items: items.slice(i, i + chunkSize), start: i });
  }
  return chunks;
}

/**
 * A pool that batches work into chunks and runs them in order on the
 * calling thread.
 *
 * Results always keep input order. A callback that throws rejects the
 * returned promise and later chunks are not started.
 *
 * @example
 * ```ts
 * import { WorkerPool } from 'deepbox/core';
 *
 * const pool = new WorkerPool({ maxWorkers: 4 });
 *
 * const results = await pool.map([1, 2, 3, 4], (x) => x * x);
 * console.log(results); // [1, 4, 9, 16]
 *
 * const sum = await pool.reduce([1, 2, 3, 4, 5, 6, 7, 8], (a, b) => a + b, 0);
 * console.log(sum); // 36
 *
 * pool.terminate();
 * ```
 */
export class WorkerPool {
  private readonly maxWorkers: number;
  private activeCount = 0;
  private completedCount = 0;
  private terminated = false;
  /** Task timeout in milliseconds. Stored for callers; not enforced (see {@link WorkerPoolOptions}). */
  readonly taskTimeoutMs: number;

  /**
   * @param options - Pool options
   * @throws {InvalidParameterError} If `maxWorkers` is not a positive integer or
   *   `taskTimeout` is not a positive finite number
   */
  constructor(options: WorkerPoolOptions = {}) {
    const maxWorkers = options.maxWorkers ?? Math.max(1, detectCpuCount());
    if (!Number.isInteger(maxWorkers) || maxWorkers < 1) {
      throw new InvalidParameterError(
        `maxWorkers must be a positive integer; received ${String(maxWorkers)}`,
        "maxWorkers",
        maxWorkers
      );
    }
    const taskTimeout = options.taskTimeout ?? 30_000;
    if (typeof taskTimeout !== "number" || !Number.isFinite(taskTimeout) || taskTimeout <= 0) {
      throw new InvalidParameterError(
        `taskTimeout must be a positive finite number; received ${String(taskTimeout)}`,
        "taskTimeout",
        taskTimeout
      );
    }
    this.maxWorkers = maxWorkers;
    this.taskTimeoutMs = taskTimeout;
  }

  /**
   * Get the current status of the pool.
   */
  status(): PoolStatus {
    return {
      maxWorkers: this.maxWorkers,
      activeWorkers: this.activeCount,
      pendingTasks: 0,
      completedTasks: this.completedCount,
      isTerminated: this.terminated,
    };
  }

  /**
   * Execute a function on a single item.
   *
   * Runs inline: spawning a worker is never worth it for one item.
   *
   * @param fn - Function to execute
   * @param arg - Argument to pass to the function
   * @returns Promise resolving to the function result with timing info
   * @throws {DeepboxError} If the pool has been terminated
   */
  async exec<T, R>(fn: (arg: T) => R, arg: T): Promise<TaskResult<R>> {
    this.ensureNotTerminated();
    const start = Date.now();
    this.activeCount++;
    let value: R;
    try {
      value = fn(arg);
    } finally {
      this.activeCount--;
    }
    this.completedCount++;
    return {
      value,
      workerId: 0,
      durationMs: Date.now() - start,
    };
  }

  /**
   * Map a function over an array of inputs.
   *
   * Items are split into at most `maxWorkers` chunks. `fn` is called with the
   * item only (not the index or array), so functions such as `parseInt` behave
   * as expected.
   *
   * @param items - Array of input items
   * @param fn - Function to apply to each item
   * @returns Promise resolving to array of results (same order as input)
   * @throws {DeepboxError} If the pool has been terminated
   */
  async map<T, R>(items: readonly T[], fn: (item: T) => R): Promise<R[]> {
    this.ensureNotTerminated();
    const out: R[] = [];
    for (const chunk of splitIntoChunks(items, this.maxWorkers)) {
      this.runChunk(() => {
        for (const item of chunk.items) out.push(fn(item));
      });
    }
    return out;
  }

  /**
   * Reduce an array to a single value.
   *
   * Each chunk is reduced separately and the partial results are then
   * combined, so `fn` must be associative. `initial` is applied exactly once,
   * like `Array.prototype.reduce`, and is the left operand of the final
   * combination.
   *
   * @param items - Array of values to reduce
   * @param fn - Associative reducer function
   * @param initial - Initial accumulator value
   * @returns Promise resolving to the reduced value
   * @throws {DeepboxError} If the pool has been terminated
   */
  async reduce<T>(items: readonly T[], fn: (a: T, b: T) => T, initial: T): Promise<T> {
    this.ensureNotTerminated();

    let acc = initial;
    for (const chunk of splitIntoChunks(items, this.maxWorkers)) {
      this.runChunk(() => {
        // Non-empty by construction, so the seedless reduce is safe.
        const partial = chunk.items.reduce((a, b) => fn(a, b));
        acc = fn(acc, partial);
      });
    }
    return acc;
  }

  /**
   * Execute multiple independent tasks.
   *
   * @param tasks - Array of zero-argument functions to execute
   * @returns Promise resolving to array of results (same order as input);
   *   promises returned by tasks are awaited
   * @throws {DeepboxError} If the pool has been terminated
   */
  async all<T>(tasks: readonly (() => T)[]): Promise<T[]> {
    this.ensureNotTerminated();
    return Promise.all(
      tasks.map(async (task) => {
        this.activeCount++;
        try {
          return await task();
        } finally {
          this.activeCount--;
          this.completedCount++;
        }
      })
    );
  }

  /**
   * Execute a function for each item, in order.
   *
   * @param items - Array of input items
   * @param fn - Function to execute for each item (side-effect only)
   * @throws {DeepboxError} If the pool has been terminated
   */
  async forEach<T>(items: readonly T[], fn: (item: T, index: number) => void): Promise<void> {
    this.ensureNotTerminated();
    for (const chunk of splitIntoChunks(items, this.maxWorkers)) {
      this.runChunk(() => {
        for (let j = 0; j < chunk.items.length; j++) {
          fn(chunk.items[j] as T, chunk.start + j);
        }
      });
    }
  }

  /**
   * Filter an array.
   *
   * @param items - Array of items to filter
   * @param predicate - Filter predicate, called with the item only
   * @returns Promise resolving to filtered array (preserving order)
   * @throws {DeepboxError} If the pool has been terminated
   */
  async filter<T>(items: readonly T[], predicate: (item: T) => boolean): Promise<T[]> {
    this.ensureNotTerminated();
    const out: T[] = [];
    for (const chunk of splitIntoChunks(items, this.maxWorkers)) {
      this.runChunk(() => {
        for (const item of chunk.items) {
          if (predicate(item)) out.push(item);
        }
      });
    }
    return out;
  }

  /**
   * Mark the pool as terminated.
   *
   * After termination, no new tasks can be submitted.
   * Safe to call multiple times.
   */
  terminate(): void {
    this.terminated = true;
  }

  /**
   * Whether the pool has been terminated.
   */
  get isTerminated(): boolean {
    return this.terminated;
  }

  private runChunk(body: () => void): void {
    this.activeCount++;
    try {
      body();
    } finally {
      this.activeCount--;
      this.completedCount++;
    }
  }

  private ensureNotTerminated(): void {
    if (this.terminated) {
      throw new DeepboxError("WorkerPool has been terminated and cannot accept new tasks");
    }
  }
}

/**
 * Create a WorkerPool with sensible defaults.
 *
 * @param maxWorkers - Maximum number of chunks (defaults to CPU count)
 * @returns A new WorkerPool instance
 * @throws {InvalidParameterError} If `maxWorkers` is not a positive integer
 */
export function createWorkerPool(maxWorkers?: number): WorkerPool {
  return maxWorkers !== undefined ? new WorkerPool({ maxWorkers }) : new WorkerPool();
}

/**
 * Get the number of available CPU cores.
 *
 * Uses `os.availableParallelism()` in Node.js and `navigator.hardwareConcurrency`
 * in browsers, falling back to 4 when neither is available.
 *
 * @returns Number of available CPU cores (at least 1)
 */
export function availableCores(): number {
  return detectCpuCount();
}
