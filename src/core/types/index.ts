/**
 * Foundational type exports for Deepbox Core.
 *
 * This file is a barrel that re-exports core types/constants/type-guards.
 */

export type { Axis, Shape, TensorStorage, TypedArray } from "./common";
export type { Device } from "./device";
export { DEVICES, isDevice } from "./device";
export type { DType } from "./dtype";
export { DTYPES, isDType } from "./dtype";
export type { TensorLike } from "./tensor";
