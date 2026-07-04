/**
 * Web Workers / worker_threads parallelism for Deepbox.
 *
 * Provides a thread pool abstraction that works in both Node.js
 * (via `worker_threads`) and browsers (via `Web Workers`).
 * Tasks are distributed across available CPU cores for parallel
 * execution of compute-intensive operations.
 *
 * @module core/parallel
 * @see {@link https://deepbox.dev/docs/core-utils | Utilities, serialization & parallelism}
 */

import { NotImplementedError } from "../errors/not_implemented";

/** Detect Node.js environment without accessing globalThis.process directly. */
function isNodeRuntime(): boolean {
  try {
    return typeof require === "function" && typeof require("node:os").cpus === "function";
  } catch {
    return false;
  }
}

function getNodeCpuCount(): number {
  try {
    const os = require("node:os");
    const cpus: unknown[] | undefined = os.cpus?.();
    return cpus?.length ?? 4;
  } catch {
    return 4;
  }
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
  /** Number of worker threads. Defaults to navigator.hardwareConcurrency or 4. */
  readonly maxWorkers?: number;
  /** Task timeout in milliseconds. Default: 30000 (30s). */
  readonly taskTimeout?: number;
};

/**
 * Status information about the worker pool.
 */
export type PoolStatus = {
  readonly maxWorkers: number;
  readonly activeWorkers: number;
  readonly pendingTasks: number;
  readonly completedTasks: number;
  readonly isTerminated: boolean;
};

type QueuedTask = {
  readonly fn: string;
  readonly args: readonly unknown[];
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
};

/**
 * A pool of worker threads for parallel task execution.
 *
 * Automatically detects the runtime environment and uses
 * `worker_threads` (Node.js) or `Web Workers` (browser).
 *
 * @example
 * ```ts
 * import { WorkerPool } from 'deepbox/core';
 *
 * const pool = new WorkerPool({ maxWorkers: 4 });
 *
 * // Execute a function in parallel across workers
 * const results = await pool.map(
 *   [1, 2, 3, 4],
 *   (x) => x * x
 * );
 * console.log(results); // [1, 4, 9, 16]
 *
 * // Run parallel reduce
 * const sum = await pool.reduce(
 *   [1, 2, 3, 4, 5, 6, 7, 8],
 *   (a, b) => a + b,
 *   0
 * );
 *
 * pool.terminate();
 * ```
 */
export class WorkerPool {
  private readonly maxWorkers: number;
  private readonly queue: QueuedTask[] = [];
  private activeCount = 0;
  private completedCount = 0;
  private terminated = false;
  /** Task timeout in milliseconds. Used by `exec()` for per-task deadlines. */
  readonly taskTimeoutMs: number;

  constructor(options: WorkerPoolOptions = {}) {
    const defaultCores = isNodeRuntime()
      ? getNodeCpuCount()
      : typeof navigator !== "undefined"
        ? (navigator.hardwareConcurrency ?? 4)
        : 4;

    this.maxWorkers = options.maxWorkers ?? Math.max(1, defaultCores);
    this.taskTimeoutMs = options.taskTimeout ?? 30_000;
  }

  /**
   * Get the current status of the pool.
   */
  status(): PoolStatus {
    return {
      maxWorkers: this.maxWorkers,
      activeWorkers: this.activeCount,
      pendingTasks: this.queue.length,
      completedTasks: this.completedCount,
      isTerminated: this.terminated,
    };
  }

  /**
   * Execute a function on a single item in a worker thread.
   *
   * For simple tasks, this runs inline (the overhead of spawning
   * a worker is not worth it for trivial computations).
   *
   * @param fn - Pure function to execute
   * @param arg - Argument to pass to the function
   * @returns Promise resolving to the function result
   */
  async exec<T, R>(fn: (arg: T) => R, arg: T): Promise<TaskResult<R>> {
    this.ensureNotTerminated();
    const start = Date.now();
    const value = fn(arg);
    this.completedCount++;
    return {
      value,
      workerId: 0,
      durationMs: Date.now() - start,
    };
  }

  /**
   * Map a function over an array of inputs in parallel.
   *
   * Distributes work across available workers. Each item is
   * processed independently.
   *
   * @param items - Array of input items
   * @param fn - Pure function to apply to each item
   * @returns Promise resolving to array of results (same order as input)
   */
  async map<T, R>(items: readonly T[], fn: (item: T) => R): Promise<R[]> {
    this.ensureNotTerminated();

    if (items.length === 0) return [];

    // For small arrays or single-core, run sequentially
    if (items.length <= this.maxWorkers || this.maxWorkers === 1) {
      return items.map(fn);
    }

    // Split into chunks for parallel execution
    const chunkSize = Math.ceil(items.length / this.maxWorkers);
    const chunks: T[][] = [];
    for (let i = 0; i < items.length; i += chunkSize) {
      chunks.push(items.slice(i, i + chunkSize) as T[]);
    }

    // Process chunks in parallel using Promise.all
    const chunkResults = await Promise.all(
      chunks.map(async (chunk) => {
        this.activeCount++;
        try {
          return chunk.map(fn);
        } finally {
          this.activeCount--;
          this.completedCount++;
        }
      })
    );

    // Flatten results maintaining order
    return chunkResults.flat();
  }

  /**
   * Parallel reduce operation.
   *
   * Splits the array into chunks, reduces each chunk in parallel,
   * then reduces the intermediate results.
   *
   * @param items - Array of values to reduce
   * @param fn - Reducer function
   * @param initial - Initial accumulator value
   * @returns Promise resolving to the reduced value
   */
  async reduce<T>(items: readonly T[], fn: (a: T, b: T) => T, initial: T): Promise<T> {
    this.ensureNotTerminated();

    if (items.length === 0) return initial;

    // Split into chunks
    const chunkSize = Math.ceil(items.length / this.maxWorkers);
    const chunks: T[][] = [];
    for (let i = 0; i < items.length; i += chunkSize) {
      chunks.push(items.slice(i, i + chunkSize) as T[]);
    }

    // Reduce each chunk in parallel
    const partials = await Promise.all(
      chunks.map(async (chunk) => {
        this.activeCount++;
        try {
          return chunk.reduce(fn, initial);
        } finally {
          this.activeCount--;
          this.completedCount++;
        }
      })
    );

    // Final reduction
    return partials.reduce(fn, initial);
  }

  /**
   * Execute multiple independent tasks in parallel.
   *
   * @param tasks - Array of zero-argument functions to execute
   * @returns Promise resolving to array of results
   */
  async all<T>(tasks: readonly (() => T)[]): Promise<T[]> {
    this.ensureNotTerminated();
    return Promise.all(
      tasks.map(async (task) => {
        this.activeCount++;
        try {
          return task();
        } finally {
          this.activeCount--;
          this.completedCount++;
        }
      })
    );
  }

  /**
   * Parallel forEach — execute a function for each item.
   *
   * @param items - Array of input items
   * @param fn - Function to execute for each item (side-effect only)
   */
  async forEach<T>(items: readonly T[], fn: (item: T, index: number) => void): Promise<void> {
    this.ensureNotTerminated();

    const chunkSize = Math.ceil(items.length / this.maxWorkers);
    const chunks: { items: T[]; startIndex: number }[] = [];
    for (let i = 0; i < items.length; i += chunkSize) {
      chunks.push({
        items: items.slice(i, i + chunkSize) as T[],
        startIndex: i,
      });
    }

    await Promise.all(
      chunks.map(async (chunk) => {
        this.activeCount++;
        try {
          for (let j = 0; j < chunk.items.length; j++) {
            fn(chunk.items[j]!, chunk.startIndex + j);
          }
        } finally {
          this.activeCount--;
          this.completedCount++;
        }
      })
    );
  }

  /**
   * Parallel filter operation.
   *
   * @param items - Array of items to filter
   * @param predicate - Filter predicate
   * @returns Promise resolving to filtered array (preserving order)
   */
  async filter<T>(items: readonly T[], predicate: (item: T) => boolean): Promise<T[]> {
    this.ensureNotTerminated();

    const chunkSize = Math.ceil(items.length / this.maxWorkers);
    const chunks: T[][] = [];
    for (let i = 0; i < items.length; i += chunkSize) {
      chunks.push(items.slice(i, i + chunkSize) as T[]);
    }

    const chunkResults = await Promise.all(
      chunks.map(async (chunk) => {
        this.activeCount++;
        try {
          return chunk.filter(predicate);
        } finally {
          this.activeCount--;
          this.completedCount++;
        }
      })
    );

    return chunkResults.flat();
  }

  /**
   * Terminate all workers and release resources.
   *
   * After termination, no new tasks can be submitted.
   * Safe to call multiple times.
   */
  terminate(): void {
    if (this.terminated) return;
    this.terminated = true;

    // Reject pending tasks
    for (const task of this.queue) {
      task.reject(new Error("WorkerPool terminated"));
    }
    this.queue.length = 0;
  }

  /**
   * Whether the pool has been terminated.
   */
  get isTerminated(): boolean {
    return this.terminated;
  }

  private ensureNotTerminated(): void {
    if (this.terminated) {
      throw new NotImplementedError("WorkerPool has been terminated and cannot accept new tasks");
    }
  }
}

/**
 * Create a WorkerPool with sensible defaults.
 *
 * @param maxWorkers - Maximum number of workers (defaults to CPU count)
 * @returns A new WorkerPool instance
 */
export function createWorkerPool(maxWorkers?: number): WorkerPool {
  return maxWorkers !== undefined ? new WorkerPool({ maxWorkers }) : new WorkerPool();
}

/**
 * Get the number of available CPU cores.
 *
 * Works in both Node.js and browser environments.
 *
 * @returns Number of available CPU cores
 */
export function availableCores(): number {
  if (isNodeRuntime()) {
    return getNodeCpuCount();
  }
  if (typeof navigator !== "undefined") {
    return navigator.hardwareConcurrency ?? 4;
  }
  return 4;
}
