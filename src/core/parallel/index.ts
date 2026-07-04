/**
 * Parallel execution utilities for Deepbox.
 *
 * @module core/parallel
 * @see {@link https://deepbox.dev/docs/core-utils | Utilities, serialization & parallelism}
 */

export {
  availableCores,
  createWorkerPool,
  type PoolStatus,
  type TaskResult,
  WorkerPool,
  type WorkerPoolOptions,
} from "./WorkerPool";
