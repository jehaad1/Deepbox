import { DeepboxError } from "./base";

/**
 * Error thrown when memory allocation fails or memory limits are exceeded.
 *
 * @example
 * ```ts
 * if (requestedBytes > MAX_MEMORY) {
 *   throw new MemoryError(`Cannot allocate ${requestedBytes} bytes`);
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
    }
  ) {
    super(message);
    if (details?.requestedBytes !== undefined) {
      this.requestedBytes = details.requestedBytes;
    }
    if (details?.availableBytes !== undefined) {
      this.availableBytes = details.availableBytes;
    }
  }
}
