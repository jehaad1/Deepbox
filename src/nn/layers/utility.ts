/**
 * Utility layers: Identity, Flatten, Unflatten.
 *
 * @module
 * @see {@link https://deepbox.dev/docs/nn-module | Deepbox documentation}
 */

import { InvalidParameterError, ShapeError } from "../../core";
import { type AnyTensor, GradTensor } from "../../ndarray";
import { Module } from "../module/Module";

/**
 * Identity layer — a no-op that passes input through unchanged.
 *
 * Useful for skip connections and placeholder layers.
 *
 * @example
 * ```ts
 * const identity = new Identity();
 * const output = identity.forward(input); // output === input
 * ```
 *
 * @category Neural Network Layers
 */
export class Identity extends Module {
  forward(input: AnyTensor): AnyTensor {
    return input;
  }

  override toString(): string {
    return "Identity()";
  }
}

/**
 * Flattens a contiguous range of dims into a single dim.
 *
 * @example
 * ```ts
 * const flatten = new Flatten(); // default: startDim=1, endDim=-1
 * // input: (batch, C, H, W) -> output: (batch, C*H*W)
 * ```
 *
 * @category Neural Network Layers
 */
export class Flatten extends Module {
  private readonly startDim: number;
  private readonly endDim: number;

  constructor(startDim = 1, endDim = -1) {
    super();
    this.startDim = startDim;
    this.endDim = endDim;
  }

  forward(input: AnyTensor): AnyTensor {
    const t = GradTensor.isGradTensor(input) ? input : GradTensor.fromTensor(input);
    const shape = t.shape;
    const ndim = shape.length;

    // Resolve negative indices
    const start = this.startDim < 0 ? ndim + this.startDim : this.startDim;
    const end = this.endDim < 0 ? ndim + this.endDim : this.endDim;

    if (start < 0 || start >= ndim) {
      throw new InvalidParameterError(
        `startDim ${this.startDim} out of range for ${ndim}-D tensor`,
        "startDim",
        this.startDim
      );
    }
    if (end < 0 || end >= ndim) {
      throw new InvalidParameterError(
        `endDim ${this.endDim} out of range for ${ndim}-D tensor`,
        "endDim",
        this.endDim
      );
    }
    if (start > end) {
      throw new InvalidParameterError(
        `startDim (${start}) must be <= endDim (${end})`,
        "startDim",
        this.startDim
      );
    }

    // Compute flattened shape
    let flatSize = 1;
    for (let i = start; i <= end; i++) {
      flatSize *= shape[i] ?? 1;
    }

    const newShape: number[] = [];
    for (let i = 0; i < start; i++) {
      newShape.push(shape[i] ?? 1);
    }
    newShape.push(flatSize);
    for (let i = end + 1; i < ndim; i++) {
      newShape.push(shape[i] ?? 1);
    }

    return t.reshape(newShape);
  }

  override toString(): string {
    return `Flatten(start_dim=${this.startDim}, end_dim=${this.endDim})`;
  }
}

/**
 * Unflattens a single dim into multiple dims.
 *
 * @example
 * ```ts
 * const unflatten = new Unflatten(1, [2, 5, 5]);
 * // input: (batch, 50) -> output: (batch, 2, 5, 5)
 * ```
 *
 * @category Neural Network Layers
 */
export class Unflatten extends Module {
  private readonly dim: number;
  private readonly unflattenedSize: readonly number[];

  constructor(dim: number, unflattenedSize: readonly number[]) {
    super();

    if (unflattenedSize.length === 0) {
      throw new InvalidParameterError(
        "unflattenedSize must have at least one element",
        "unflattenedSize",
        unflattenedSize
      );
    }

    for (const s of unflattenedSize) {
      if (!Number.isInteger(s) || s <= 0) {
        throw new InvalidParameterError(
          "All dimensions in unflattenedSize must be positive integers",
          "unflattenedSize",
          unflattenedSize
        );
      }
    }

    this.dim = dim;
    this.unflattenedSize = unflattenedSize;
  }

  forward(input: AnyTensor): AnyTensor {
    const t = GradTensor.isGradTensor(input) ? input : GradTensor.fromTensor(input);
    const shape = t.shape;
    const ndim = shape.length;

    const resolvedDim = this.dim < 0 ? ndim + this.dim : this.dim;

    if (resolvedDim < 0 || resolvedDim >= ndim) {
      throw new InvalidParameterError(
        `dim ${this.dim} out of range for ${ndim}-D tensor`,
        "dim",
        this.dim
      );
    }

    const dimSize = shape[resolvedDim] ?? 1;
    const unflatProduct = this.unflattenedSize.reduce((a, b) => a * b, 1);
    if (dimSize !== unflatProduct) {
      throw new ShapeError(
        `Dimension ${resolvedDim} has size ${dimSize} but unflattenedSize product is ${unflatProduct}`
      );
    }

    const newShape: number[] = [];
    for (let i = 0; i < resolvedDim; i++) {
      newShape.push(shape[i] ?? 1);
    }
    for (const s of this.unflattenedSize) {
      newShape.push(s);
    }
    for (let i = resolvedDim + 1; i < ndim; i++) {
      newShape.push(shape[i] ?? 1);
    }

    return t.reshape(newShape);
  }

  override toString(): string {
    return `Unflatten(dim=${this.dim}, unflattened_size=[${this.unflattenedSize.join(", ")}])`;
  }
}
