import type { Shape } from "../types/common";
import { DeepboxError } from "./base";

function cloneShape(shape: Shape): Shape {
  return Object.freeze([...shape]);
}

/**
 * Details about a shape mismatch error.
 *
 * @property expected - The expected shape
 * @property received - The actual shape that was received
 * @property context - Additional context about where the error occurred
 */
export type ShapeErrorDetails = {
  readonly expected?: Shape;
  readonly received?: Shape;
  readonly context?: string;
};

/**
 * Error thrown when tensor shapes are incompatible or invalid.
 *
 * This error is used throughout the library when operations require specific
 * tensor shapes that don't match the provided tensors.
 *
 * @example
 * ```ts
 * import { ShapeError } from 'deepbox/core';
 *
 * if (tensor.shape[0] !== expectedRows) {
 *   throw ShapeError.mismatch([expectedRows, cols], tensor.shape, 'matrix multiplication');
 * }
 * ```
 */
export class ShapeError extends DeepboxError {
  override name = "ShapeError";

  readonly expected: Shape | undefined;
  readonly received: Shape | undefined;
  readonly context: string | undefined;

  constructor(message: string, details: ShapeErrorDetails = {}) {
    // Call parent constructor with error message
    super(message);
    // Store optional shape details for programmatic error handling
    this.expected = details.expected === undefined ? undefined : cloneShape(details.expected);
    this.received = details.received === undefined ? undefined : cloneShape(details.received);
    this.context = details.context;
  }

  /**
   * Create a ShapeError for a shape mismatch.
   *
   * Convenience factory method for the common case of shape mismatches.
   *
   * @param expected - The expected shape
   * @param received - The actual shape that was received
   * @param context - Optional context about where the mismatch occurred
   * @returns A new ShapeError with formatted message and details
   *
   * @example
   * ```ts
   * throw ShapeError.mismatch([3, 4], [3, 5], 'matrix multiplication');
   * // Error: Shape mismatch (matrix multiplication): expected [3,4], received [3,5]
   * ```
   */
  static mismatch(expected: Shape, received: Shape, context?: string): ShapeError {
    // Build context string if provided
    const ctx = context ? ` (${context})` : "";
    // Construct details object with conditional context field
    // Using satisfies ensures type safety without widening types
    const details = {
      expected,
      received,
      ...(context !== undefined ? { context } : {}),
    } satisfies ShapeErrorDetails;
    // Create and return a new ShapeError with formatted message
    return new ShapeError(
      `Shape mismatch${ctx}: expected [${expected}], received [${received}]`,
      details
    );
  }
}
