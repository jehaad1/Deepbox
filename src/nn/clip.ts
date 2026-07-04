/**
 * Gradient clipping utilities for neural network training.
 *
 * Prevents exploding gradients by capping gradient magnitudes,
 * essential for training RNNs and deep networks.
 *
 * @example
 * ```ts
 * import { clip_grad_norm_, clip_grad_value_ } from 'deepbox/nn';
 *
 * // Clip by global L2 norm (most common)
 * const totalNorm = clip_grad_norm_(model.parameters(), 1.0);
 *
 * // Clip each gradient element to [-0.5, 0.5]
 * clip_grad_value_(model.parameters(), 0.5);
 * ```
 *
 * @module nn/clip
 * @see {@link https://deepbox.dev/docs/nn-module | Deepbox Gradient Clipping}
 */

import { InvalidParameterError } from "../core";
import type { GradTensor } from "../ndarray";

/**
 * Clip the total norm of gradients of an iterable of parameters.
 *
 * The norm is computed over all gradients together, as if they were
 * concatenated into a single vector. Gradients are modified in-place.
 *
 * @param parameters - Iterable of GradTensors whose gradients will be clipped
 * @param maxNorm - Maximum allowed norm value
 * @param normType - Type of norm (default: 2 for L2 norm)
 * @returns The total norm of the gradients (before clipping)
 * @deprecated Prefer {@link clipGradNorm_}.
 */
export function clip_grad_norm_(
  parameters: Iterable<GradTensor>,
  maxNorm: number,
  normType = 2
): number {
  if (maxNorm < 0) {
    throw new InvalidParameterError(
      `maxNorm must be >= 0; received ${maxNorm}`,
      "maxNorm",
      maxNorm
    );
  }

  const params = Array.from(parameters);
  const grads: GradTensor[] = [];
  for (const p of params) {
    if (p.grad) {
      grads.push(p);
    }
  }

  if (grads.length === 0) {
    return 0;
  }

  let totalNorm: number;

  if (normType === Infinity) {
    totalNorm = 0;
    for (const p of grads) {
      const g = p.grad!;
      for (let i = 0; i < g.size; i++) {
        const absVal = Math.abs(Number(g.data[g.offset + i]));
        if (absVal > totalNorm) totalNorm = absVal;
      }
    }
  } else {
    let normSum = 0;
    for (const p of grads) {
      const g = p.grad!;
      for (let i = 0; i < g.size; i++) {
        normSum += Math.abs(Number(g.data[g.offset + i])) ** normType;
      }
    }
    totalNorm = normSum ** (1.0 / normType);
  }

  const clipCoef = maxNorm / (totalNorm + 1e-6);
  if (clipCoef < 1) {
    for (const p of grads) {
      const g = p.grad!;
      for (let i = 0; i < g.size; i++) {
        g.data[g.offset + i] = Number(g.data[g.offset + i]) * clipCoef;
      }
    }
  }

  return totalNorm;
}

/**
 * Clip the gradients of an iterable of parameters at specified value.
 *
 * Each gradient element is clamped to [-clipValue, clipValue].
 * Gradients are modified in-place.
 *
 * @param parameters - Iterable of GradTensors whose gradients will be clipped
 * @param clipValue - Maximum absolute value for gradient elements
 * @deprecated Prefer {@link clipGradValue_}.
 */
export function clip_grad_value_(parameters: Iterable<GradTensor>, clipValue: number): void {
  if (clipValue < 0) {
    throw new InvalidParameterError(
      `clipValue must be >= 0; received ${clipValue}`,
      "clipValue",
      clipValue
    );
  }

  for (const p of Array.from(parameters)) {
    if (p.grad) {
      const g = p.grad;
      for (let i = 0; i < g.size; i++) {
        const val = Number(g.data[g.offset + i]);
        g.data[g.offset + i] = Math.max(-clipValue, Math.min(clipValue, val));
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Canonical camelCase aliases
//
// The snake_case spellings above mirror PyTorch and remain exported for
// backward compatibility. These camelCase aliases are the recommended names;
// each refers to the exact same in-place function.
// ---------------------------------------------------------------------------

/** Canonical camelCase alias of {@link clip_grad_norm_} (in-place). */
export const clipGradNorm_ = clip_grad_norm_;
/** Convenience alias of {@link clip_grad_norm_} without the trailing underscore. */
export const clipGradNorm = clip_grad_norm_;
/** Canonical camelCase alias of {@link clip_grad_value_} (in-place). */
export const clipGradValue_ = clip_grad_value_;
/** Convenience alias of {@link clip_grad_value_} without the trailing underscore. */
export const clipGradValue = clip_grad_value_;
