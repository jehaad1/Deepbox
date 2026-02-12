/**
 * Utility exports for Deepbox Core.
 *
 * This file is a barrel that re-exports runtime helpers used across the codebase.
 */

export { normalizeAxes, normalizeAxis } from "./axis";
export type { NumericDType } from "./dtypeUtils";
export { dtypeToTypedArrayCtor, ensureNumericDType } from "./dtypeUtils";
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
} from "./typedArrayAccess";
export { isTypedArray } from "./typeGuards";
export {
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
