/**
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox documentation}
 */

// Arithmetic
export {
  abs,
  add,
  addScalar,
  clip,
  div,
  floorDiv,
  maximum,
  minimum,
  mod,
  mul,
  mulScalar,
  neg,
  pow,
  reciprocal,
  sign,
  sub,
} from "./arithmetic";

// Comparison
export {
  allclose,
  arrayEqual,
  equal,
  greater,
  greaterEqual,
  isclose,
  isfinite,
  isinf,
  isnan,
  less,
  lessEqual,
  notEqual,
} from "./comparison";
// Convolution
export { col2im, im2col } from "./conv";
export { einsum } from "./einsum";
export {
  type FFTNorm,
  type FFTResult,
  fft,
  fft2,
  fftfreq,
  fftn,
  fftshift,
  ifft,
  ifft2,
  ifftn,
  ifftshift,
  irfft,
  rfft,
  rfftfreq,
} from "./fft";
export { delete_, insert } from "./insert_delete";
// Logical
export { logicalAnd, logicalNot, logicalOr, logicalXor } from "./logical";
// Manipulation
export { concatenate, repeat, split, stack, tile } from "./manipulation";
// Math
export {
  cbrt,
  ceil,
  exp,
  exp2,
  expm1,
  floor,
  log,
  log1p,
  log2,
  log10,
  round,
  rsqrt,
  sqrt,
  square,
  trunc,
} from "./math";
// Numerical utilities
export {
  column_stack,
  columnStack,
  digitize,
  gradient,
  hstack,
  interp,
  trapezoid,
  trapz,
  vstack,
} from "./numerical";
// Random ops
export { dropoutMask } from "./random";
// Reduction
export {
  all,
  any,
  argmax,
  argmin,
  cumprod,
  cumsum,
  diff,
  max,
  mean,
  median,
  min,
  nanargmax,
  nanargmin,
  nancumsum,
  nanmedian,
  nanprod,
  nanquantile,
  nanvar,
  prod,
  std,
  sum,
  variance,
} from "./reduction";
export { gcd, intersect1d, lcm, setdiff1d, union1d } from "./setops";
// Signal processing
export {
  bartlettWindow,
  blackmanWindow,
  type ConvolveMode,
  convolve,
  correlate,
  hammingWindow,
  hannWindow,
  kaiserWindow,
  type WindowOptions,
} from "./signal";
// Sorting
export { argsort, sort } from "./sorting";
// Trigonometry
export {
  acos,
  acosh,
  asin,
  asinh,
  atan,
  atan2,
  atanh,
  cos,
  cosh,
  sin,
  sinh,
  tan,
  tanh,
} from "./trigonometry";
export type {
  CrossOptions,
  HistogramOptions,
  LikeOptions,
  PadMode,
  PadWidth,
  UniqueOptions,
  UniqueOutput,
  UniqueResult,
} from "./utils";
// Utils (tensor utilities)
export {
  argwhere,
  atleast_1d,
  atleast_2d,
  atleast1d,
  atleast2d,
  bincount,
  booleanIndex,
  broadcast_to,
  broadcastTo,
  clone,
  contiguous,
  copy,
  countNonzero,
  cross,
  detach,
  diag,
  diagonal,
  empty_like,
  emptyLike,
  fancyIndex,
  flip,
  flipLr,
  fliplr,
  flipUd,
  flipud,
  full_like,
  fullLike,
  histogram,
  index_select,
  indexSelect,
  isin,
  meshgrid,
  moveaxis,
  nanmax,
  nanmean,
  nanmin,
  nanstd,
  nansum,
  nonzero,
  ones_like,
  onesLike,
  pad,
  putAlongAxis,
  roll,
  rot90,
  scatter,
  searchsorted,
  swapaxes,
  takeAlongAxis,
  tril,
  triu,
  unique,
  where,
  zeros_like,
  zerosLike,
} from "./utils";
