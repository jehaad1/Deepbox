import type { Shape } from "../types/common";
import { DeepboxError } from "./base";

function cloneShape(shape: Shape): Shape {
  return Object.freeze([...shape]);
}

/**
 * Error thrown when tensor shapes cannot be broadcast together.
 *
 * Broadcasting allows operations on tensors of different shapes by automatically
 * expanding dimensions. This error occurs when shapes are incompatible.
 *
 * @example
 * ```ts
 * // Broadcasting (3, 4) and (4,) works -> (3, 4)
 * // Broadcasting (3, 4) and (3,) fails
 * if (!canBroadcast(shape1, shape2)) {
 *   throw new BroadcastError(shape1, shape2);
 * }
 * ```
 *
 * References:
 * - NumPy broadcasting: https://numpy.org/doc/stable/user/basics.broadcasting.html
 */
export class BroadcastError extends DeepboxError {
  override name = "BroadcastError";

  /** The first shape involved in the broadcast operation */
  readonly shape1: Shape;

  /** The second shape involved in the broadcast operation */
  readonly shape2: Shape;

  constructor(shape1: Shape, shape2: Shape, context?: string) {
    // Build context string if provided (e.g., "in matrix multiplication")
    const ctx = context ? ` (${context})` : "";
    // Create descriptive error message showing both incompatible shapes
    super(`Shapes [${shape1}] and [${shape2}] cannot be broadcast together${ctx}`);
    // Store shapes for programmatic error handling
    this.shape1 = cloneShape(shape1);
    this.shape2 = cloneShape(shape2);
  }
}
