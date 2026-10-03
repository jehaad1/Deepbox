/**
 * @see {@link https://deepbox.dev/docs/core-errors | Errors, warnings & logging}
 */

import { DeepboxError } from "./base";

/**
 * Error thrown when a method or code path exists in the public API but is not implemented yet.
 *
 * Use this instead of a generic `Error` so callers can reliably detect this specific
 * capability gap.
 */
export class NotImplementedError extends DeepboxError {
  /**
   * Discriminator name for this error type.
   */
  override name = "NotImplementedError";

  /**
   * Create a new NotImplementedError.
   *
   * @param message - Optional error message.
   * @param options - Optional `cause` to chain.
   */
  constructor(message = "Not implemented", options?: { readonly cause?: unknown }) {
    super(message, options);
  }
}
