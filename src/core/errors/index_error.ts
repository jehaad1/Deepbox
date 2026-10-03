/**
 * @see {@link https://deepbox.dev/docs/core-errors | Errors, warnings & logging}
 */

import { DeepboxError } from "./base";

/**
 * Error thrown when an invalid index is used to access tensor elements.
 *
 * @example
 * ```ts
 * if (index < 0 || index >= length) {
 *   throw new IndexError(`Index ${index} is out of bounds for length ${length}`);
 * }
 * ```
 */
export class IndexError extends DeepboxError {
  override name = "IndexError";

  /** The invalid index value */
  readonly index?: number;

  /** The valid range for the index */
  readonly validRange?: readonly [number, number];

  constructor(
    message: string,
    details?: {
      readonly index?: number;
      readonly validRange?: readonly [number, number];
      readonly cause?: unknown;
    }
  ) {
    super(message, details?.cause !== undefined ? { cause: details.cause } : undefined);
    if (details?.index !== undefined) {
      this.index = details.index;
    }
    if (details?.validRange !== undefined) {
      this.validRange = details.validRange;
    }
  }
}
