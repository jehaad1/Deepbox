import { DeepboxError } from "./base";

/**
 * Error thrown when a requested dtype is unsupported or cannot be handled by the current
 * implementation.
 *
 * This is used in low-level dtype utilities (e.g. mapping a `DType` to a TypedArray
 * constructor) to provide a stable, package-specific error type.
 */
export class DTypeError extends DeepboxError {
  /**
   * Discriminator name for this error type.
   */
  override name = "DTypeError";
}
