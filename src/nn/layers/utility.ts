/**
 * Utility layers: Identity, Flatten, Unflatten.
 *
 * @module
 * @see {@link https://deepbox.dev/docs/nn-module | Deepbox documentation}
 */

import { InvalidParameterError, ShapeError } from "../../core";
import type { AnyTensor, GradTensor, Tensor } from "../../ndarray";
import { Module } from "../module/Module";
import { allPlain, settle, toGradInput } from "./_shared";

/**
 * Identity layer: returns its input unchanged (the same object, no copy).
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
  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: AnyTensor): AnyTensor;
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
 * Negative `startDim` / `endDim` count from the last dimension. A 0-d input is
 * flattened to shape `[1]`, as in PyTorch.
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

  /**
   * @param startDim - First dimension to flatten (default: 1)
   * @param endDim - Last dimension to flatten, inclusive (default: -1)
   * @throws {InvalidParameterError} If either argument is not an integer
   */
  constructor(startDim = 1, endDim = -1) {
    super();
    if (!Number.isInteger(startDim)) {
      throw new InvalidParameterError("startDim must be an integer", "startDim", startDim);
    }
    if (!Number.isInteger(endDim)) {
      throw new InvalidParameterError("endDim must be an integer", "endDim", endDim);
    }
    this.startDim = startDim;
    this.endDim = endDim;
  }

  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: AnyTensor): AnyTensor {
    return settle(this.run(input), allPlain(input));
  }

  private run(input: AnyTensor): GradTensor {
    const t = toGradInput(input);
    const shape = t.shape;
    // A scalar behaves like a 1-element vector for dimension bookkeeping.
    const dims = shape.length === 0 ? [1] : shape;
    const ndim = dims.length;

    // Resolve negative indices
    const start = this.startDim < 0 ? ndim + this.startDim : this.startDim;
    const end = this.endDim < 0 ? ndim + this.endDim : this.endDim;

    if (start < 0 || start >= ndim) {
      throw new InvalidParameterError(
        `startDim ${this.startDim} out of range for ${shape.length}-D tensor`,
        "startDim",
        this.startDim
      );
    }
    if (end < 0 || end >= ndim) {
      throw new InvalidParameterError(
        `endDim ${this.endDim} out of range for ${shape.length}-D tensor`,
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
      flatSize *= dims[i] ?? 1;
    }

    const newShape: number[] = [];
    for (let i = 0; i < start; i++) {
      newShape.push(dims[i] ?? 1);
    }
    newShape.push(flatSize);
    for (let i = end + 1; i < ndim; i++) {
      newShape.push(dims[i] ?? 1);
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
 * At most one entry of `unflattenedSize` may be `-1`; it is inferred from the
 * size of the dimension being split.
 *
 * @example
 * ```ts
 * const unflatten = new Unflatten(1, [2, 5, 5]);
 * // input: (batch, 50) -> output: (batch, 2, 5, 5)
 *
 * const inferred = new Unflatten(1, [2, -1]);
 * // input: (batch, 50) -> output: (batch, 2, 25)
 * ```
 *
 * @category Neural Network Layers
 */
export class Unflatten extends Module {
  private readonly dim: number;
  private readonly unflattenedSize: readonly number[];

  /**
   * @param dim - Dimension to split (negative values count from the end)
   * @param unflattenedSize - Sizes of the new dimensions: positive integers, with at most one `-1`
   * @throws {InvalidParameterError} If `dim` is not an integer or `unflattenedSize` is invalid
   */
  constructor(dim: number, unflattenedSize: readonly number[]) {
    super();

    if (!Number.isInteger(dim)) {
      throw new InvalidParameterError("dim must be an integer", "dim", dim);
    }
    if (unflattenedSize.length === 0) {
      throw new InvalidParameterError(
        "unflattenedSize must have at least one element",
        "unflattenedSize",
        unflattenedSize
      );
    }

    let inferred = 0;
    for (const s of unflattenedSize) {
      if (s === -1) {
        inferred++;
      } else if (!Number.isInteger(s) || s <= 0) {
        throw new InvalidParameterError(
          "All dimensions in unflattenedSize must be positive integers (or -1 for one inferred dimension)",
          "unflattenedSize",
          unflattenedSize
        );
      }
    }
    if (inferred > 1) {
      throw new InvalidParameterError(
        "unflattenedSize can contain at most one -1",
        "unflattenedSize",
        unflattenedSize
      );
    }

    this.dim = dim;
    this.unflattenedSize = [...unflattenedSize];
  }

  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: AnyTensor): AnyTensor {
    return settle(this.run(input), allPlain(input));
  }

  private run(input: AnyTensor): GradTensor {
    const t = toGradInput(input);
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
    const known = this.unflattenedSize.reduce((a, b) => (b === -1 ? a : a * b), 1);
    const hasInferred = this.unflattenedSize.includes(-1);
    let sizes: readonly number[] = this.unflattenedSize;
    if (hasInferred) {
      if (dimSize % known !== 0) {
        throw new ShapeError(
          `Dimension ${resolvedDim} has size ${dimSize}, which is not divisible by the product ${known} of the known unflattenedSize entries`
        );
      }
      const inferredSize = dimSize / known;
      sizes = this.unflattenedSize.map((s) => (s === -1 ? inferredSize : s));
    } else if (dimSize !== known) {
      throw new ShapeError(
        `Dimension ${resolvedDim} has size ${dimSize} but unflattenedSize product is ${known}`
      );
    }

    const newShape: number[] = [];
    for (let i = 0; i < resolvedDim; i++) {
      newShape.push(shape[i] ?? 1);
    }
    for (const s of sizes) {
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
