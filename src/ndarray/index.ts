/**
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox documentation}
 */

export type {
  Device,
  DType,
  ExtendedTypedArray,
  Shape,
  TensorLike,
  TypedArray,
} from "../core";
export type { GradTensorOptions } from "./autograd/index";
// Autograd - gradient tracking and automatic differentiation
export {
  col2imGrad,
  concatGrad,
  customOp,
  dropout as dropoutGrad,
  GradTensor,
  im2col as im2colGrad,
  logSoftmax as logSoftmaxGrad,
  noGrad,
  parameter,
  softmax as softmaxGrad,
  stackGrad,
  variance as varianceGrad,
} from "./autograd/index";
export { Complex, Complex64Array, Complex128Array } from "./tensor/complex";
// Float16 and Complex number arrays
export { BFloat16Array, Float16Array } from "./tensor/float16";

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
export { corrcoef, cov, tensordot } from "./linalg/basic";
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
export { einsum } from "./ops/einsum";
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
  atleast_1d,
  atleast_2d,
  atleast1d,
  atleast2d,
  bartlettWindow,
  bincount,
  blackmanWindow,
  booleanIndex,
  broadcast_to,
  broadcastTo,
  cbrt,
  ceil,
  clip,
  clone,
  column_stack,
  columnStack,
  concatenate,
  contiguous,
  convolve,
  copy,
  correlate,
  cos,
  cosh,
  cross,
  cumprod,
  cumsum,
  delete_,
  detach,
  diag,
  diagonal,
  diff,
  digitize,
  div,
  empty_like,
  emptyLike,
  equal,
  exp,
  exp2,
  expm1,
  type FFTResult,
  fancyIndex,
  fft,
  fft2,
  fftn,
  flip,
  flipLr,
  fliplr,
  flipUd,
  flipud,
  floor,
  floorDiv,
  full_like,
  fullLike,
  gcd,
  gradient,
  greater,
  greaterEqual,
  hammingWindow,
  hannWindow,
  histogram,
  hstack,
  ifft,
  ifft2,
  ifftn,
  index_select,
  indexSelect,
  insert,
  interp,
  intersect1d,
  irfft,
  isclose,
  isfinite,
  isin,
  isinf,
  isnan,
  kaiserWindow,
  lcm,
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
  meshgrid,
  min,
  minimum,
  mod,
  moveaxis,
  mul,
  mulScalar,
  nanmax,
  nanmean,
  nanmin,
  nanstd,
  nansum,
  neg,
  notEqual,
  ones_like,
  onesLike,
  pad,
  pow,
  prod,
  reciprocal,
  repeat,
  rfft,
  roll,
  rot90,
  round,
  rsqrt,
  scatter,
  searchsorted,
  setdiff1d,
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
  swapaxes,
  tan,
  tanh,
  tile,
  trapz,
  tril,
  triu,
  trunc,
  union1d,
  unique,
  variance,
  vstack,
  where,
  zeros_like,
  zerosLike,
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
export { expandDims, squeeze, unsqueeze } from "./tensor/shape_ops";
