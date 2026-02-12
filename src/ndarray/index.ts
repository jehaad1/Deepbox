export type { Device, DType, Shape, TensorLike, TypedArray } from "../core";
export type { GradTensorOptions } from "./autograd/index";
// Autograd - gradient tracking and automatic differentiation
export {
  dropout as dropoutGrad,
  GradTensor,
  im2col as im2colGrad,
  logSoftmax as logSoftmaxGrad,
  noGrad,
  parameter,
  softmax as softmaxGrad,
  variance as varianceGrad,
} from "./autograd/index";

// Re-export Tensor class for the union type below
import type { GradTensor as GradTensorClass } from "./autograd/index";
import type { Tensor as TensorClass } from "./tensor/index";

/**
 * Union type representing either a Tensor or GradTensor.
 *
 * This type enables functions to accept both regular tensors and
 * differentiable tensors interchangeably, improving API flexibility.
 *
 * Use this type when a function should work with either tensor type:
 * - `Tensor`: For pure numerical operations without gradient tracking
 * - `GradTensor`: For operations that need automatic differentiation
 *
 * @example
 * ```ts
 * import type { AnyTensor } from 'deepbox/ndarray';
 *
 * function processData(input: AnyTensor): void {
 *   console.log(input.shape);  // Works with both Tensor and GradTensor
 *   console.log(input.dtype);
 * }
 * ```
 */
export type AnyTensor = TensorClass | GradTensorClass;
export { dot } from "./linalg/index";
export {
  elu,
  gelu,
  leakyRelu,
  logSoftmax,
  mish,
  relu,
  sigmoid,
  softmax,
  softplus,
  swish,
} from "./ops/activation";
export { col2im, im2col } from "./ops/conv";
export {
  abs,
  acos,
  acosh,
  add,
  addScalar,
  all,
  allclose,
  any,
  argsort,
  arrayEqual,
  asin,
  asinh,
  atan,
  atan2,
  atanh,
  cbrt,
  ceil,
  clip,
  concatenate,
  cos,
  cosh,
  cumprod,
  cumsum,
  diff,
  div,
  equal,
  exp,
  exp2,
  expm1,
  floor,
  floorDiv,
  greater,
  greaterEqual,
  isclose,
  isfinite,
  isinf,
  isnan,
  less,
  lessEqual,
  log,
  log1p,
  log2,
  log10,
  logicalAnd,
  logicalNot,
  logicalOr,
  logicalXor,
  max,
  maximum,
  mean,
  median,
  min,
  minimum,
  mod,
  mul,
  mulScalar,
  neg,
  notEqual,
  pow,
  prod,
  reciprocal,
  repeat,
  round,
  rsqrt,
  sign,
  sin,
  sinh,
  sort,
  split,
  sqrt,
  square,
  stack,
  std,
  sub,
  sum,
  tan,
  tanh,
  tile,
  trunc,
  variance,
} from "./ops/index";
export { dropoutMask } from "./ops/random";
export type { CSRMatrixInit } from "./sparse";

export { CSRMatrix } from "./sparse";
export type {
  NestedArray,
  SliceRange,
  TensorCreateOptions,
  TensorOptions,
} from "./tensor/index";
export {
  arange,
  empty,
  eye,
  flatten,
  full,
  gather,
  geomspace,
  linspace,
  logspace,
  ones,
  randn,
  reshape,
  slice,
  Tensor,
  tensor,
  transpose,
  zeros,
} from "./tensor/index";
export { expandDims, squeeze, unsqueeze } from "./tensor/shapeOps";
