/**
 * Foundational type exports for Deepbox Core.
 *
 * This file is a barrel that re-exports core types/constants/type-guards.
 * @see {@link https://deepbox.dev/docs/core-types | Deepbox documentation}
 */

export type {
  Axis,
  ExtendedTypedArray,
  Shape,
  TensorStorage,
  TypedArray,
} from "./common";
export type { Device } from "./device";
export { DEVICES, isDevice } from "./device";
export type { DType, ElementOf, ScalarDType } from "./dtype";
export { DTYPES, isDType } from "./dtype";
export type { TensorLike } from "./tensor";
