/**
 * Utility exports for Deepbox Core.
 *
 * This file is a barrel that re-exports runtime helpers used across the codebase.
 * @see {@link https://deepbox.dev/docs/core-utils | Utilities, serialization & parallelism}
 */

export { normalizeAxes, normalizeAxis } from "./axis";
export type { NumericDType } from "./dtype_utils";
export { dtypeToTypedArrayCtor, ensureNumericDType } from "./dtype_utils";
export { isTypedArray } from "./type_guards";
export {
  asReadonlyArray,
  getArrayElement,
  getBigIntElement,
  getElementAsNumber,
  getNumericElement,
  getShapeDim,
  getStringElement,
  isBigInt64Array,
  isNumericTypedArray,
  type NumericTypedArray,
} from "./typed_array_access";
export {
  check_array,
  check_is_fitted,
  check_X_y,
  checkArray,
  checkIsFitted,
  checkXY,
  shapesEqual,
  shapeToSize,
  validateArray,
  validateDevice,
  validateDtype,
  validateInteger,
  validateNonNegative,
  validateOneOf,
  validatePositive,
  validateRange,
  validateShape,
} from "./validation";
