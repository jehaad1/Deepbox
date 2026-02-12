import type { Shape, TensorStorage } from "./common";
import type { Device } from "./device";
import type { DType } from "./dtype";

/**
 * Type representing a tensor-like object.
 *
 * This type defines the core properties that any tensor implementation
 * must provide. It enables type-safe tensor operations and interoperability
 * between different tensor implementations.
 *
 * @typeParam S - The shape type (extends Shape)
 * @typeParam D - The data type (extends DType)
 *
 * @property shape - Dimensions of the tensor (e.g., [2, 3, 4])
 * @property dtype - Data type of tensor elements
 * @property device - Compute device where tensor resides
 * @property data - Underlying TypedArray storage
 * @property strides - Step sizes for each dimension in memory
 * @property offset - Starting position in the data buffer
 * @property size - Total number of elements in the tensor
 * @property ndim - Number of dimensions (rank) of the tensor
 *
 * @example
 * ```ts
 * import type { TensorLike } from 'deepbox/core';
 *
 * function processTensor<S extends Shape, D extends DType>(
 *   tensor: TensorLike<S, D>
 * ): void {
 *   console.log(`Shape: ${tensor.shape}`);
 *   console.log(`Size: ${tensor.size}`);
 *   console.log(`DType: ${tensor.dtype}`);
 * }
 * ```
 */
export type TensorLike<S extends Shape = Shape, D extends DType = DType> = {
  readonly shape: S;
  readonly dtype: D;
  readonly device: Device;
  readonly data: TensorStorage;
  readonly strides: readonly number[];
  readonly offset: number;
  readonly size: number;
  readonly ndim: number;
};
