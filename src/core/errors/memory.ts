/**
 * @see {@link https://deepbox.dev/docs/core-errors | Errors, warnings & logging}
 */

import { DeepboxError } from "./base";

/**
 * Error thrown when memory allocation fails or memory limits are exceeded.
 *
 * @example
 * ```ts
 * import { MemoryError } from 'deepbox/core';
 *
 * const MAX_MEMORY = 1024;
 * const requestedBytes = 4096;
 * try {
 *   if (requestedBytes > MAX_MEMORY) {
 *     throw new MemoryError(`Cannot allocate ${requestedBytes} bytes`);
 *   }
 * } catch (error) {
 *   console.log((error as Error).message); // Cannot allocate 4096 bytes
 * }
 * ```
 */
export class MemoryError extends DeepboxError {
  override name = "MemoryError";

  /** The amount of memory requested in bytes */
  readonly requestedBytes?: number;

  /** The amount of memory available in bytes */
  readonly availableBytes?: number;

  constructor(
    message: string,
    details?: {
      readonly requestedBytes?: number;
      readonly availableBytes?: number;
      readonly cause?: unknown;
    }
  ) {
    super(message, details?.cause !== undefined ? { cause: details.cause } : undefined);
    if (details?.requestedBytes !== undefined) {
      this.requestedBytes = details.requestedBytes;
    }
    if (details?.availableBytes !== undefined) {
      this.availableBytes = details.availableBytes;
    }
  }
}
