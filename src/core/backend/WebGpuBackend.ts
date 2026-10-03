/**
 * WebGPU Backend: GPU execution backend for Deepbox tensors.
 *
 * Implements the {@link KernelBackend} contract with WGSL compute kernels:
 * stride/broadcast-aware element-wise ops, tiled matrix multiplication,
 * multi-pass reductions, and convolution/pooling kernels over float32
 * device buffers (float16 and bfloat16 are also supported). Once registered
 * (`registerBackend('webgpu', backend)` after `await backend.init()`),
 * built-in ndarray ops on `webgpu` tensors dispatch here automatically;
 * see Devices & execution on DeepboxDocs for the accelerated op set.
 *
 * Kernels enqueue GPU work synchronously; reading results back
 * (`await tensor.cpu()`) is asynchronous because WebGPU buffer mapping is
 * asynchronous.
 *
 * In environments without `navigator.gpu` (e.g. Node.js), pass a WebGPU
 * implementation explicitly: `new WebGpuBackend({ gpu })`.
 *
 * @module core/backend
 * @see {@link https://deepbox.dev/docs/devices-and-execution | Devices & execution}
 */

import {
  float16BitsToFloat64,
  float64ToFloat16Bits,
  roundToBFloat16,
} from "../../ndarray/tensor/float16";
import { DeviceError } from "../errors/index";
import type { BackendCapability, BackendInfo } from "./Backend";
import type {
  BinaryKernelOp,
  DeviceBuffer,
  DeviceDType,
  Im2ColParams,
  KernelBackend,
  KernelLayout,
  PoolKernelOp,
  ReduceKernelOp,
  TernaryKernelOp,
  UnaryKernelOp,
} from "./kernels";
import {
  GPU_BUFFER_USAGE,
  GPU_MAP_MODE,
  type GpuAdapter,
  type GpuBindGroupEntry,
  type GpuBuffer,
  type GpuComputePipeline,
  type GpuDevice,
  type GpuDeviceDescriptor,
  type GpuDeviceLostInfo,
  type GpuLike,
} from "./webgpu_types";

/** Maximum tensor rank supported by the WGSL kernels. */
const MAX_RANK = 8;
const WORKGROUP_SIZE = 256;
/** Bytes kept alive in the free-buffer pool before excess buffers are destroyed. */
const POOL_CAP_BYTES = 256 * 1024 * 1024;
/**
 * `maxComputeWorkgroupsPerDimension` guaranteed by every WebGPU device. The
 * backend never asks for more, so launches wider than this are folded into a
 * 2-D grid (1-D kernels) or rejected with a `DeviceError` (matmul).
 */
const MAX_GROUPS_PER_DIM = 65535;
/** `maxStorageBufferBindingSize` / `maxBufferSize` assumed when the device does not report them. */
const DEFAULT_MAX_BUFFER_BYTES = 134217728;

/**
 * Linear thread index of a 1-D kernel. Launches of more than
 * {@link MAX_GROUPS_PER_DIM} workgroups are folded into a 2-D grid, so the
 * index is rebuilt from both grid axes (it equals `gid.x` on a 1-D launch).
 */
const MAIN_1D = /* wgsl */ `fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let i = gid.y * (nwg.x * ${WORKGROUP_SIZE}u) + gid.x;`;

// ─── WGSL kernel sources ─────────────────────────────────────────────────────

/** Layout metadata shared by the strided element-wise kernels. */
const BINARY_META = /* wgsl */ `
struct Meta {
  size: u32,
  ndim: u32,
  a_offset: u32,
  b_offset: u32,
  shape: array<vec4<u32>, 2>,
  a_strides: array<vec4<u32>, 2>,
  b_strides: array<vec4<u32>, 2>,
};
`;

const UNARY_META = /* wgsl */ `
struct Meta {
  size: u32,
  ndim: u32,
  offset: u32,
  _pad: u32,
  shape: array<vec4<u32>, 2>,
  strides: array<vec4<u32>, 2>,
};
`;

/**
 * Layout metadata of the full-reduction passes. `divisor` divides each
 * workgroup partial (the element count for `mean`, 1 otherwise), so a mean
 * accumulates already-scaled partials and half-precision sums cannot overflow.
 */
const REDUCE_META = /* wgsl */ `
struct Meta {
  size: u32,
  ndim: u32,
  offset: u32,
  divisor: f32,
  shape: array<vec4<u32>, 2>,
  strides: array<vec4<u32>, 2>,
};
`;

/** Expressions for each binary op; `x`/`y` are the operand values. */
const BINARY_EXPR: Record<BinaryKernelOp, string> = {
  add: "x + y",
  sub: "x - y",
  mul: "x * y",
  div: "x / y",
  pow: "pow_impl(x, y)",
  maximum: "select(max(x, y), x + y, is_nan(x) || is_nan(y))",
  minimum: "select(min(x, y), x + y, is_nan(x) || is_nan(y))",
};

/**
 * Expressions for each unary op; `x` is the operand value. Ops that call a
 * `*_impl` function get its WGSL source from {@link UNARY_HELPERS}.
 *
 * `exp(x) - 1`, `log(1 + x)`, the common 5-term erf fits and the built-in
 * `tanh` lose most of their digits for small arguments, and the built-in
 * `tanh` returns NaN for large ones, so those ops use the helpers below.
 */
const UNARY_EXPR: Record<UnaryKernelOp, string> = {
  copy: "x",
  step: "select(0.0, 1.0, x > 0.0)",
  neg: "-x",
  abs: "abs(x)",
  exp: "exp(x)",
  log: "log(x)",
  sqrt: "sqrt(x)",
  square: "x * x",
  // NaN-propagating, like the CPU `Math.max(0, x)`.
  relu: "select(max(x, 0.0), x, is_nan(x))",
  sigmoid: "1.0 / (1.0 + exp(-x))",
  tanh: "tanh_impl(x)",
  // Tanh-approximation GELU, the same function as the CPU `gelu` op:
  // 0.5 x (1 + tanh(u)) with u = sqrt(2/pi) (x + 0.044715 x^3). Since
  // 1 + tanh(u) = 2 / (1 + exp(-2u)), it is evaluated as x / (1 + exp(-2u)),
  // which keeps full precision for negative x (no 1 + tanh(-large)
  // cancellation) and stays finite where tanh would overflow.
  // 2 sqrt(2/pi) = 1.5957691216057308.
  gelu: "x / (1.0 + exp(-1.5957691216057308 * (x + 0.044715 * x * x * x)))",
  erf: "erf_impl(x)",
  rsqrt: "inverseSqrt(x)",
  reciprocal: "1.0 / x",
  sign: "select(sign(x), x, is_nan(x))",
  expm1: "expm1_impl(x)",
  log1p: "log1p_impl(x)",
  // Numerically stable softplus: max(x, 0) + log1p(exp(-|x|)). Unlike
  // log(1 + exp(x)) it keeps the tail for very negative x.
  softplus: "max(x, 0.0) + log1p_impl(exp(-abs(x)))",
};

/**
 * `tanh` that is accurate for small |x| (Taylor series up to |x| < 0.5, where
 * the built-in loses up to ~1e-5 relative accuracy) and finite for large |x|
 * (the built-in returns NaN past |x| ~ 44; tanh(20) is already exactly 1 in
 * float32). NaN propagates.
 */
const TANH_IMPL = /* wgsl */ `
fn tanh_impl(x: f32) -> f32 {
  if (is_nan(x)) { return x; }
  if (abs(x) < 0.5) {
    let x2 = x * x;
    return x * (1.0 + x2 * (-0.33333334 + x2 * (0.13333334 + x2 * (-0.053968254
      + x2 * (0.021869489 + x2 * (-0.008863236 + x2 * (0.003592128 + x2 * -0.0014558344)))))));
  }
  return tanh(clamp(x, -20.0, 20.0));
}
`;

/**
 * `exp(x) - 1` without cancellation: a Taylor series for |x| < 0.5 (error
 * below 1e-10 relative), the direct difference elsewhere, where it loses at
 * most one bit.
 */
const EXPM1_IMPL = /* wgsl */ `
fn expm1_impl(x: f32) -> f32 {
  if (abs(x) < 0.5) {
    return x * (1.0 + x * (0.5 + x * (0.16666667 + x * (0.041666668 + x * (0.008333334
      + x * (0.0013888889 + x * (0.00019841270 + x * (0.000024801587 + x * (0.0000027557319
      + x * 0.00000027557319)))))))));
  }
  return exp(x) - 1.0;
}
`;

/**
 * `log(1 + x)` without cancellation: for |x| < 0.5 it uses
 * `2 atanh(s)` with `s = x / (2 + x)` as an odd series in `s` (|s| <= 1/3,
 * error below 1e-8 relative), and the direct form elsewhere.
 */
const LOG1P_IMPL = /* wgsl */ `
fn log1p_impl(x: f32) -> f32 {
  if (abs(x) < 0.5) {
    let s = x / (2.0 + x);
    let s2 = s * s;
    return 2.0 * s * (1.0 + s2 * (0.33333334 + s2 * (0.2 + s2 * (0.14285715 + s2 * (0.11111111
      + s2 * (0.09090909 + s2 * (0.07692308 + s2 * 0.06666667)))))));
  }
  return log(1.0 + x);
}
`;

/**
 * Error function accurate to a few float32 ulps over the whole range:
 *  - |x| < 1: Maclaurin series 2/sqrt(pi) * x * sum((-x^2)^n / (n! (2n+1)));
 *  - 1 <= |x| < 4: erf = 1 - erfc with W. J. Cody's rational approximation of
 *    erfc (CALERF, 1969);
 *  - |x| >= 4: +-1 (erfc(4) = 1.5e-8 is below half an ulp of 1).
 * The 5-term Abramowitz & Stegun 7.1.26 fit is only good to 1.5e-7 absolute,
 * which is a large relative error for small x, where erf(x) ~ 1.13 x.
 */
const ERF_IMPL = /* wgsl */ `
fn erf_impl(x: f32) -> f32 {
  if (is_nan(x)) { return x; }
  let ax = abs(x);
  if (ax < 1.0) {
    let y = -(ax * ax);
    var s = 1.089222104e-09;
    s = s * y + 1.3122533e-08;
    s = s * y + 1.4503852e-07;
    s = s * y + 1.4589169e-06;
    s = s * y + 1.32275132e-05;
    s = s * y + 1.0683761e-04;
    s = s * y + 7.5757576e-04;
    s = s * y + 4.6296296e-03;
    s = s * y + 2.3809524e-02;
    s = s * y + 1.0e-01;
    s = s * y + 3.3333333e-01;
    s = s * y + 1.0e+00;
    return 1.1283792 * x * s;
  }
  if (ax >= 4.0) { return select(1.0, -1.0, x < 0.0); }
  var xnum = 2.1531154e-8 * ax;
  var xden = ax;
  xnum = (xnum + 0.5641885) * ax;     xden = (xden + 15.744926) * ax;
  xnum = (xnum + 8.8831498) * ax;     xden = (xden + 117.69395) * ax;
  xnum = (xnum + 66.119191) * ax;     xden = (xden + 537.1811) * ax;
  xnum = (xnum + 298.63514) * ax;     xden = (xden + 1621.3896) * ax;
  xnum = (xnum + 881.95222) * ax;     xden = (xden + 3290.7992) * ax;
  xnum = (xnum + 1712.0476) * ax;     xden = (xden + 4362.6191) * ax;
  xnum = (xnum + 2051.0784) * ax;     xden = (xden + 3439.3677) * ax;
  let erfc = exp(-(ax * ax)) * ((xnum + 1230.3394) / (xden + 1230.3394));
  let r = 1.0 - erfc;
  return select(r, -r, x < 0.0);
}
`;

/** Unary ops that need helper functions injected into their shader. */
const UNARY_HELPERS: Partial<Record<UnaryKernelOp, string>> = {
  tanh: TANH_IMPL,
  erf: ERF_IMPL,
  expm1: EXPM1_IMPL,
  log1p: LOG1P_IMPL,
  softplus: LOG1P_IMPL,
};

/**
 * `pow` with IEEE/NumPy special cases. WGSL's `pow` (exp2/log2 based) is
 * undefined for a negative base and for a zero base with a non-positive
 * exponent, and it is only accurate to a few 1e-6 for large results.
 *  - `y == 0` gives 1 for every `x` (including NaN), as in `Math.pow`;
 *  - a NaN exponent gives NaN (a zero base must not turn it into 0);
 *  - a zero base gives 0 or +-infinity (sign kept for odd integer `y`);
 *  - integer exponents up to 16 use repeated squaring, which is exact to a
 *    few ulps (the common `x ** 2` and `x ** 3`);
 *  - other negative bases give NaN, or the signed result for integral `y`.
 * Float32 integers of magnitude >= 2^24 are always even, so they are never odd.
 */
const POW_IMPL = /* wgsl */ `
fn pow_impl(x: f32, y: f32) -> f32 {
  if (y == 0.0) { return 1.0; }
  if (is_nan(x)) { return x; }
  if (is_nan(y)) { return y; }
  let y_int = y == floor(y);
  let y_odd = y_int && abs(y) < 16777216.0 && (i32(y) & 1) != 0;
  if (x == 0.0) {
    if (y < 0.0) {
      let r = 1.0 / x;
      return select(abs(r), r, y_odd);
    }
    return select(0.0, x, y_odd);
  }
  if (y_int && abs(y) <= 16.0) {
    var acc = 1.0;
    var base = x;
    for (var e = u32(abs(y)); e > 0u; e = e >> 1u) {
      if ((e & 1u) != 0u) { acc = acc * base; }
      base = base * base;
    }
    return select(acc, 1.0 / acc, y < 0.0);
  }
  if (x > 0.0) {
    return pow(x, y);
  }
  if (y_int) {
    let mag = pow(-x, y);
    return select(mag, -mag, y_odd);
  }
  // Negative base, non-integral exponent: NaN. Routed through a runtime
  // var because WGSL const-expressions must not evaluate to NaN.
  var nan_bits: u32 = 0x7fc00000u;
  return bitcast<f32>(nan_bits);
}
`;

function binaryShader(op: BinaryKernelOp): string {
  return /* wgsl */ `
${BINARY_META}
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> um: Meta;
${NAN_HELPER}
${op === "pow" ? POW_IMPL : ""}
@compute @workgroup_size(${WORKGROUP_SIZE})
${MAIN_1D}
  if (i >= um.size) { return; }
  var rem = i;
  var a_idx = um.a_offset;
  var b_idx = um.b_offset;
  for (var k = 0u; k < um.ndim; k = k + 1u) {
    let d = um.ndim - 1u - k;
    let dim = um.shape[d / 4u][d % 4u];
    let coord = rem % dim;
    rem = rem / dim;
    a_idx = a_idx + coord * um.a_strides[d / 4u][d % 4u];
    b_idx = b_idx + coord * um.b_strides[d / 4u][d % 4u];
  }
  let x = a[a_idx];
  let y = b[b_idx];
  out[i] = ${BINARY_EXPR[op]};
}`;
}

function unaryShader(op: UnaryKernelOp): string {
  return /* wgsl */ `
${UNARY_META}
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> um: Meta;
${NAN_HELPER}
${UNARY_HELPERS[op] ?? ""}
@compute @workgroup_size(${WORKGROUP_SIZE})
${MAIN_1D}
  if (i >= um.size) { return; }
  var rem = i;
  var idx = um.offset;
  for (var k = 0u; k < um.ndim; k = k + 1u) {
    let d = um.ndim - 1u - k;
    let dim = um.shape[d / 4u][d % 4u];
    let coord = rem % dim;
    rem = rem / dim;
    idx = idx + coord * um.strides[d / 4u][d % 4u];
  }
  let x = input[idx];
  out[i] = ${UNARY_EXPR[op]};
}`;
}

const MATMUL_SHADER = /* wgsl */ `
struct Meta {
  m: u32,
  n: u32,
  k: u32,
  a_offset: u32,
  a_stride0: u32,
  a_stride1: u32,
  b_stride0: u32,
  b_stride1: u32,
  b_offset: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> um: Meta;

const TILE: u32 = 16u;

var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row = gid.y;
  let col = gid.x;
  let lr = lid.y;
  let lc = lid.x;

  var acc: f32 = 0.0;
  let tiles = (um.k + TILE - 1u) / TILE;

  for (var t = 0u; t < tiles; t = t + 1u) {
    let a_col = t * TILE + lc;
    let b_row = t * TILE + lr;

    if (row < um.m && a_col < um.k) {
      tile_a[lr][lc] = a[um.a_offset + row * um.a_stride0 + a_col * um.a_stride1];
    } else {
      tile_a[lr][lc] = 0.0;
    }
    if (b_row < um.k && col < um.n) {
      tile_b[lr][lc] = b[um.b_offset + b_row * um.b_stride0 + col * um.b_stride1];
    } else {
      tile_b[lr][lc] = 0.0;
    }
    workgroupBarrier();

    for (var p = 0u; p < TILE; p = p + 1u) {
      acc = acc + tile_a[lr][p] * tile_b[p][lc];
    }
    workgroupBarrier();
  }

  if (row < um.m && col < um.n) {
    out[row * um.n + col] = acc;
  }
}`;

/**
 * NaN test on the bit pattern. Shader compilers (notably Metal with fast
 * math) assume operands are never NaN and fold `x != x` to `false`, which
 * would silently drop NaN propagation, so the integer encoding is inspected.
 */
const NAN_HELPER = /* wgsl */ `
fn is_nan(x: f32) -> bool {
  return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}
`;

/** Reduction combine functions (NaN-propagating for min/max, like NumPy). */
const REDUCE_COMBINE: Record<"sum" | "max" | "min", string> = {
  sum: "return x + y;",
  max: "if (is_nan(x)) { return x; } if (is_nan(y)) { return y; } return max(x, y);",
  min: "if (is_nan(x)) { return x; } if (is_nan(y)) { return y; } return min(x, y);",
};

// Identity elements as runtime expressions: WGSL const-expressions must not
// evaluate to ±inf, so the bit patterns go through a runtime `var`.
const REDUCE_IDENTITY: Record<"sum" | "max" | "min", string> = {
  sum: "0.0",
  max: "neg_inf()", // -inf
  min: "pos_inf()", // +inf
};

const INF_HELPERS = /* wgsl */ `
fn neg_inf() -> f32 {
  var bits: u32 = 0xff800000u;
  return bitcast<f32>(bits);
}
fn pos_inf() -> f32 {
  var bits: u32 = 0x7f800000u;
  return bitcast<f32>(bits);
}
`;

function reduceShader(op: "sum" | "max" | "min"): string {
  return /* wgsl */ `
${REDUCE_META}
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> um: Meta;

var<workgroup> partials: array<f32, ${WORKGROUP_SIZE}>;
${INF_HELPERS}
${NAN_HELPER}
fn combine(x: f32, y: f32) -> f32 {
  ${REDUCE_COMBINE[op]}
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let group = wid.y * nwg.x + wid.x;
  let i = group * ${WORKGROUP_SIZE}u + lid.x;
  if (i < um.size) {
    var rem = i;
    var idx = um.offset;
    for (var k = 0u; k < um.ndim; k = k + 1u) {
      let d = um.ndim - 1u - k;
      let dim = um.shape[d / 4u][d % 4u];
      let coord = rem % dim;
      rem = rem / dim;
      idx = idx + coord * um.strides[d / 4u][d % 4u];
    }
    partials[lid.x] = input[idx];
  } else {
    partials[lid.x] = ${REDUCE_IDENTITY[op]};
  }
  workgroupBarrier();

  for (var stride = ${WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride >> 1u) {
    if (lid.x < stride) {
      partials[lid.x] = combine(partials[lid.x], partials[lid.x + stride]);
    }
    workgroupBarrier();
  }

  // Workgroups past the last real partial (possible on a folded grid) must not write.
  if (lid.x == 0u && i < um.size) {
    out[group] = partials[0] / um.divisor;
  }
}`;
}

/**
 * Reduction along one axis. Each thread owns one output element and reduces
 * the `axisDim` input elements along the reduced axis, accumulating in f32.
 * `divisor` is `axisDim` for mean and `1.0` otherwise, applied once at the
 * end. `outToInStride[j]` is the input stride of the input dimension that
 * output dimension `j` maps to.
 */
function axisReduceShader(op: "sum" | "max" | "min"): string {
  return /* wgsl */ `
struct Meta {
  outSize: u32,
  axisDim: u32,
  outNdim: u32,
  inOffset: u32,
  axisStride: u32,
  divisor: f32,
  _pad0: u32,
  _pad1: u32,
  outShape: array<vec4<u32>, 2>,
  outToInStride: array<vec4<u32>, 2>,
};
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> um: Meta;
${INF_HELPERS}
${NAN_HELPER}
fn combine(x: f32, y: f32) -> f32 {
  ${REDUCE_COMBINE[op]}
}
@compute @workgroup_size(${WORKGROUP_SIZE})
${MAIN_1D}
  if (i >= um.outSize) { return; }
  var rem = i;
  var base = um.inOffset;
  for (var k = 0u; k < um.outNdim; k = k + 1u) {
    let d = um.outNdim - 1u - k;
    let dim = um.outShape[d / 4u][d % 4u];
    let coord = rem % dim;
    rem = rem / dim;
    base = base + coord * um.outToInStride[d / 4u][d % 4u];
  }
  var acc = ${REDUCE_IDENTITY[op]};
  for (var j = 0u; j < um.axisDim; j = j + 1u) {
    acc = combine(acc, input[base + j * um.axisStride]);
  }
  out[i] = acc / um.divisor;
}`;
}

/**
 * Direct batched matrix multiply: one thread per `(batch, row, col)` output
 * element with a k-loop. The batch index (dispatch z) is decoded over the
 * shared batch shape into per-operand base offsets, so broadcasting a batch
 * dimension (stride 0) needs no copy. Output is contiguous `[batch, m, n]`.
 */
const BATCHED_MATMUL_SHADER = /* wgsl */ `
struct Meta {
  m: u32,
  n: u32,
  k: u32,
  batch: u32,
  a_offset: u32,
  b_offset: u32,
  a_s0: u32,
  a_s1: u32,
  b_s0: u32,
  b_s1: u32,
  batch_ndim: u32,
  _pad0: u32,
  batch_shape: array<vec4<u32>, 2>,
  a_batch_strides: array<vec4<u32>, 2>,
  b_batch_strides: array<vec4<u32>, 2>,
};
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> um: Meta;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  let row = gid.y;
  let bi = gid.z;
  if (row >= um.m || col >= um.n || bi >= um.batch) { return; }

  var rem = bi;
  var a_base = um.a_offset;
  var b_base = um.b_offset;
  for (var kk = 0u; kk < um.batch_ndim; kk = kk + 1u) {
    let d = um.batch_ndim - 1u - kk;
    let dim = um.batch_shape[d / 4u][d % 4u];
    let coord = rem % dim;
    rem = rem / dim;
    a_base = a_base + coord * um.a_batch_strides[d / 4u][d % 4u];
    b_base = b_base + coord * um.b_batch_strides[d / 4u][d % 4u];
  }

  var acc: f32 = 0.0;
  for (var p = 0u; p < um.k; p = p + 1u) {
    acc = acc + a[a_base + row * um.a_s0 + p * um.a_s1] * b[b_base + p * um.b_s0 + col * um.b_s1];
  }
  out[(bi * um.m + row) * um.n + col] = acc;
}`;

/** Broadcast-aware ternary select: `cond != 0 ? a : b`, element-wise. */
const TERNARY_META = /* wgsl */ `
struct Meta {
  size: u32,
  ndim: u32,
  c_offset: u32,
  a_offset: u32,
  b_offset: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
  shape: array<vec4<u32>, 2>,
  c_strides: array<vec4<u32>, 2>,
  a_strides: array<vec4<u32>, 2>,
  b_strides: array<vec4<u32>, 2>,
};
`;

const TERNARY_WHERE_SHADER = /* wgsl */ `
${TERNARY_META}
@group(0) @binding(0) var<storage, read> cond: array<f32>;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> um: Meta;

@compute @workgroup_size(${WORKGROUP_SIZE})
${MAIN_1D}
  if (i >= um.size) { return; }
  var rem = i;
  var c_idx = um.c_offset;
  var a_idx = um.a_offset;
  var b_idx = um.b_offset;
  for (var k = 0u; k < um.ndim; k = k + 1u) {
    let d = um.ndim - 1u - k;
    let dim = um.shape[d / 4u][d % 4u];
    let coord = rem % dim;
    rem = rem / dim;
    c_idx = c_idx + coord * um.c_strides[d / 4u][d % 4u];
    a_idx = a_idx + coord * um.a_strides[d / 4u][d % 4u];
    b_idx = b_idx + coord * um.b_strides[d / 4u][d % 4u];
  }
  out[i] = select(b[b_idx], a[a_idx], cond[c_idx] != 0.0);
}`;

/** Shared convolution/pool geometry uniform. */
const CONV_META = /* wgsl */ `
struct Meta {
  batch: u32,
  channels: u32,
  height: u32,
  width: u32,
  outH: u32,
  outW: u32,
  kH: u32,
  kW: u32,
  strideH: u32,
  strideW: u32,
  padH: u32,
  padW: u32,
  inOffset: u32,
  iS0: u32,
  iS1: u32,
  iS2: u32,
  iS3: u32,
  size: u32,
  _p0: u32,
  _p1: u32,
};
`;

/**
 * im2col: one thread per output column element. Decodes the flat index into
 * (batch, out-pixel, tap) and gathers the corresponding padded input value.
 */
const IM2COL_SHADER = /* wgsl */ `
${CONV_META}
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> um: Meta;

@compute @workgroup_size(${WORKGROUP_SIZE})
${MAIN_1D}
  if (i >= um.size) { return; }
  let colSize = um.channels * um.kH * um.kW;
  let outPixels = um.outH * um.outW;
  let b = i / (outPixels * colSize);
  var rem = i % (outPixels * colSize);
  let p = rem / colSize;
  let c = rem % colSize;
  let oh = p / um.outW;
  let ow = p % um.outW;
  let channel = c / (um.kH * um.kW);
  let krem = c % (um.kH * um.kW);
  let kh = krem / um.kW;
  let kw = krem % um.kW;
  // Signed arithmetic for padding (indices can go negative).
  let ih = i32(oh * um.strideH + kh) - i32(um.padH);
  let iw = i32(ow * um.strideW + kw) - i32(um.padW);
  var val: f32 = 0.0;
  if (ih >= 0 && ih < i32(um.height) && iw >= 0 && iw < i32(um.width)) {
    let off = um.inOffset + b * um.iS0 + channel * um.iS1 + u32(ih) * um.iS2 + u32(iw) * um.iS3;
    val = input[off];
  }
  out[i] = val;
}`;

/**
 * col2im: one thread per output image element. Sums the contributions of
 * every sliding window that covered this pixel (gather form, no atomics).
 * Input columns are contiguous `[batch, outH*outW, channels*kH*kW]`.
 */
const COL2IM_SHADER = /* wgsl */ `
${CONV_META}
@group(0) @binding(0) var<storage, read> cols: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> um: Meta;

@compute @workgroup_size(${WORKGROUP_SIZE})
${MAIN_1D}
  if (i >= um.size) { return; }
  let colSize = um.channels * um.kH * um.kW;
  let outPixels = um.outH * um.outW;
  // Decode output image coord (contiguous [batch, channels, height, width]).
  let b = i / (um.channels * um.height * um.width);
  var rem = i % (um.channels * um.height * um.width);
  let channel = rem / (um.height * um.width);
  rem = rem % (um.height * um.width);
  let ih = i32(rem / um.width);
  let iw = i32(rem % um.width);
  var acc: f32 = 0.0;
  // Enumerate every (kh, kw) tap and the output pixel that would read this
  // input location with that tap; accumulate if it lands on a valid pixel.
  for (var kh: u32 = 0u; kh < um.kH; kh = kh + 1u) {
    let ohNumer = ih + i32(um.padH) - i32(kh);
    if (ohNumer < 0 || (ohNumer % i32(um.strideH)) != 0) { continue; }
    let oh = ohNumer / i32(um.strideH);
    if (oh < 0 || oh >= i32(um.outH)) { continue; }
    for (var kw: u32 = 0u; kw < um.kW; kw = kw + 1u) {
      let owNumer = iw + i32(um.padW) - i32(kw);
      if (owNumer < 0 || (owNumer % i32(um.strideW)) != 0) { continue; }
      let ow = owNumer / i32(um.strideW);
      if (ow < 0 || ow >= i32(um.outW)) { continue; }
      let p = u32(oh) * um.outW + u32(ow);
      let c = channel * (um.kH * um.kW) + kh * um.kW + kw;
      acc = acc + cols[b * outPixels * colSize + p * colSize + c];
    }
  }
  out[i] = acc;
}`;

/** 2-D pooling (max / avg). One thread per output element. */
function pool2dShader(op: "max" | "avg"): string {
  return /* wgsl */ `
${CONV_META}
${INF_HELPERS}
${NAN_HELPER}
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> um: Meta;

@compute @workgroup_size(${WORKGROUP_SIZE})
${MAIN_1D}
  if (i >= um.size) { return; }
  // Output is contiguous [batch, channels, outH, outW].
  let b = i / (um.channels * um.outH * um.outW);
  var rem = i % (um.channels * um.outH * um.outW);
  let channel = rem / (um.outH * um.outW);
  rem = rem % (um.outH * um.outW);
  let oh = rem / um.outW;
  let ow = rem % um.outW;
  ${op === "max" ? "var acc: f32 = neg_inf();" : "var acc: f32 = 0.0;\n  var count: f32 = 0.0;"}
  for (var kh: u32 = 0u; kh < um.kH; kh = kh + 1u) {
    let ih = i32(oh * um.strideH + kh) - i32(um.padH);
    if (ih < 0 || ih >= i32(um.height)) { continue; }
    for (var kw: u32 = 0u; kw < um.kW; kw = kw + 1u) {
      let iw = i32(ow * um.strideW + kw) - i32(um.padW);
      if (iw < 0 || iw >= i32(um.width)) { continue; }
      let v = input[um.inOffset + b * um.iS0 + channel * um.iS1 + u32(ih) * um.iS2 + u32(iw) * um.iS3];
      ${
        op === "max"
          ? "if (is_nan(v)) { acc = v; } else if (!is_nan(acc)) { acc = max(acc, v); }"
          : "acc = acc + v;\n      count = count + 1.0;"
      }
    }
  }
  ${op === "avg" ? "out[i] = select(0.0, acc / count, count > 0.0);" : "out[i] = acc;"}
}`;
}

/**
 * 2-D pooling backward. One thread per input element sums contributions from
 * every window that covers it. For `max`, a window contributes only if this
 * element is the window's first-argmax (strict `>` scan, matching the CPU
 * path); for `avg`, every in-range tap of a covering window gets an equal
 * share. Atomic-free (gather form). `input` = pooling input (NCHW, uses the
 * strided layout in Meta); `grad_out` = contiguous `[B,C,outH,outW]`.
 */
function pool2dBackwardShader(op: "max" | "avg"): string {
  return /* wgsl */ `
${CONV_META}
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> grad_out: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> um: Meta;

@compute @workgroup_size(${WORKGROUP_SIZE})
${MAIN_1D}
  if (i >= um.size) { return; }
  // The avg variant never reads the input; touching it keeps binding 0 in the
  // auto-generated bind group layout (unused bindings are dropped from it).
  _ = arrayLength(&input);
  // Decode this input element (contiguous NCHW).
  let b = i / (um.channels * um.height * um.width);
  var rem = i % (um.channels * um.height * um.width);
  let channel = rem / (um.height * um.width);
  rem = rem % (um.height * um.width);
  let ih = i32(rem / um.width);
  let iw = i32(rem % um.width);
  let goPlane = um.outH * um.outW;
  var acc: f32 = 0.0;

  for (var kh: u32 = 0u; kh < um.kH; kh = kh + 1u) {
    let ohNumer = ih + i32(um.padH) - i32(kh);
    if (ohNumer < 0 || (ohNumer % i32(um.strideH)) != 0) { continue; }
    let oh = ohNumer / i32(um.strideH);
    if (oh < 0 || oh >= i32(um.outH)) { continue; }
    for (var kw: u32 = 0u; kw < um.kW; kw = kw + 1u) {
      let owNumer = iw + i32(um.padW) - i32(kw);
      if (owNumer < 0 || (owNumer % i32(um.strideW)) != 0) { continue; }
      let ow = owNumer / i32(um.strideW);
      if (ow < 0 || ow >= i32(um.outW)) { continue; }
      let gv = grad_out[(b * um.channels + channel) * goPlane + u32(oh) * um.outW + u32(ow)];
      ${
        op === "avg"
          ? `
      // Count in-range taps of window (oh,ow).
      var count: f32 = 0.0;
      for (var a: u32 = 0u; a < um.kH; a = a + 1u) {
        let jh = oh * i32(um.strideH) + i32(a) - i32(um.padH);
        if (jh < 0 || jh >= i32(um.height)) { continue; }
        for (var bb: u32 = 0u; bb < um.kW; bb = bb + 1u) {
          let jw = ow * i32(um.strideW) + i32(bb) - i32(um.padW);
          if (jw < 0 || jw >= i32(um.width)) { continue; }
          count = count + 1.0;
        }
      }
      if (count > 0.0) { acc = acc + gv / count; }`
          : `
      // Recompute the window's first-argmax (strict >, row-major scan).
      var best: f32 = 0.0; // overwritten by the first in-range tap
      var bestLocal: i32 = -1;
      var seen: bool = false;
      for (var a: u32 = 0u; a < um.kH; a = a + 1u) {
        let jh = oh * i32(um.strideH) + i32(a) - i32(um.padH);
        if (jh < 0 || jh >= i32(um.height)) { continue; }
        for (var bb: u32 = 0u; bb < um.kW; bb = bb + 1u) {
          let jw = ow * i32(um.strideW) + i32(bb) - i32(um.padW);
          if (jw < 0 || jw >= i32(um.width)) { continue; }
          let v = input[um.inOffset + b * um.iS0 + channel * um.iS1 + u32(jh) * um.iS2 + u32(jw) * um.iS3];
          if (!seen || v > best) { best = v; bestLocal = jh * i32(um.width) + jw; seen = true; }
        }
      }
      if (bestLocal == ih * i32(um.width) + iw) { acc = acc + gv; }`
      }
    }
  }
  out[i] = acc;
}`;
}

const FILL_SHADER = /* wgsl */ `
struct Meta {
  size: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
  value: f32,
  _pad3: f32,
  _pad4: f32,
  _pad5: f32,
};
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> um: Meta;

@compute @workgroup_size(${WORKGROUP_SIZE})
${MAIN_1D}
  if (i < um.size) {
    out[i] = um.value;
  }
}`;

// ─── float16 WGSL variants ───────────────────────────────────────────────────
//
// Half-precision (`shader-f16`) variants of the compute kernels. They are
// generated from the SAME expression tables as the float32 kernels: storage
// buffers are declared `array<f16>`, values are widened to f32 at load
// (`f32(a[i])`), the arithmetic runs in f32 exactly as in the f32 kernels
// (so activations/erf/pow stay accurate), and the result is narrowed back to
// f16 at store (`f16(x)`). Workgroup reduction accumulators stay f32 for
// accuracy: only the storage is half-precision, which is where the
// bandwidth/footprint win comes from.
//
// bfloat16 has NO native WGSL type. Rather than an error-prone manual u32
// bit-twiddle layout, bf16 tensors reuse the float32 kernels: they are stored
// on-device as float32 (values rounded to bf16 at upload, re-rounded at
// download), so the user gets correct bf16 numerics/rounding while the GPU
// computes in f32. This trades the on-device memory saving for correctness in
// a single pass; float16 is the true on-device half-precision path.

/** `enable f16;` directive required at the top of every half-precision shader. */
const F16_ENABLE = "enable f16;\n";

function binaryShaderF16(op: BinaryKernelOp): string {
  return /* wgsl */ `${F16_ENABLE}
${BINARY_META}
@group(0) @binding(0) var<storage, read> a: array<f16>;
@group(0) @binding(1) var<storage, read> b: array<f16>;
@group(0) @binding(2) var<storage, read_write> out: array<f16>;
@group(0) @binding(3) var<uniform> um: Meta;
${NAN_HELPER}
${op === "pow" ? POW_IMPL : ""}
@compute @workgroup_size(${WORKGROUP_SIZE})
${MAIN_1D}
  if (i >= um.size) { return; }
  var rem = i;
  var a_idx = um.a_offset;
  var b_idx = um.b_offset;
  for (var k = 0u; k < um.ndim; k = k + 1u) {
    let d = um.ndim - 1u - k;
    let dim = um.shape[d / 4u][d % 4u];
    let coord = rem % dim;
    rem = rem / dim;
    a_idx = a_idx + coord * um.a_strides[d / 4u][d % 4u];
    b_idx = b_idx + coord * um.b_strides[d / 4u][d % 4u];
  }
  let x = f32(a[a_idx]);
  let y = f32(b[b_idx]);
  out[i] = f16(${BINARY_EXPR[op]});
}`;
}

function unaryShaderF16(op: UnaryKernelOp): string {
  return /* wgsl */ `${F16_ENABLE}
${UNARY_META}
@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read_write> out: array<f16>;
@group(0) @binding(2) var<uniform> um: Meta;
${NAN_HELPER}
${UNARY_HELPERS[op] ?? ""}
@compute @workgroup_size(${WORKGROUP_SIZE})
${MAIN_1D}
  if (i >= um.size) { return; }
  var rem = i;
  var idx = um.offset;
  for (var k = 0u; k < um.ndim; k = k + 1u) {
    let d = um.ndim - 1u - k;
    let dim = um.shape[d / 4u][d % 4u];
    let coord = rem % dim;
    rem = rem / dim;
    idx = idx + coord * um.strides[d / 4u][d % 4u];
  }
  let x = f32(input[idx]);
  out[i] = f16(${UNARY_EXPR[op]});
}`;
}

const MATMUL_SHADER_F16 = /* wgsl */ `${F16_ENABLE}
struct Meta {
  m: u32,
  n: u32,
  k: u32,
  a_offset: u32,
  a_stride0: u32,
  a_stride1: u32,
  b_stride0: u32,
  b_stride1: u32,
  b_offset: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> a: array<f16>;
@group(0) @binding(1) var<storage, read> b: array<f16>;
@group(0) @binding(2) var<storage, read_write> out: array<f16>;
@group(0) @binding(3) var<uniform> um: Meta;

const TILE: u32 = 16u;

var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row = gid.y;
  let col = gid.x;
  let lr = lid.y;
  let lc = lid.x;

  var acc: f32 = 0.0;
  let tiles = (um.k + TILE - 1u) / TILE;

  for (var t = 0u; t < tiles; t = t + 1u) {
    let a_col = t * TILE + lc;
    let b_row = t * TILE + lr;

    if (row < um.m && a_col < um.k) {
      tile_a[lr][lc] = f32(a[um.a_offset + row * um.a_stride0 + a_col * um.a_stride1]);
    } else {
      tile_a[lr][lc] = 0.0;
    }
    if (b_row < um.k && col < um.n) {
      tile_b[lr][lc] = f32(b[um.b_offset + b_row * um.b_stride0 + col * um.b_stride1]);
    } else {
      tile_b[lr][lc] = 0.0;
    }
    workgroupBarrier();

    for (var p = 0u; p < TILE; p = p + 1u) {
      acc = acc + tile_a[lr][p] * tile_b[p][lc];
    }
    workgroupBarrier();
  }

  if (row < um.m && col < um.n) {
    out[row * um.n + col] = f16(acc);
  }
}`;

const BATCHED_MATMUL_SHADER_F16 = /* wgsl */ `${F16_ENABLE}
struct Meta {
  m: u32,
  n: u32,
  k: u32,
  batch: u32,
  a_offset: u32,
  b_offset: u32,
  a_s0: u32,
  a_s1: u32,
  b_s0: u32,
  b_s1: u32,
  batch_ndim: u32,
  _pad0: u32,
  batch_shape: array<vec4<u32>, 2>,
  a_batch_strides: array<vec4<u32>, 2>,
  b_batch_strides: array<vec4<u32>, 2>,
};
@group(0) @binding(0) var<storage, read> a: array<f16>;
@group(0) @binding(1) var<storage, read> b: array<f16>;
@group(0) @binding(2) var<storage, read_write> out: array<f16>;
@group(0) @binding(3) var<uniform> um: Meta;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  let row = gid.y;
  let bi = gid.z;
  if (row >= um.m || col >= um.n || bi >= um.batch) { return; }

  var rem = bi;
  var a_base = um.a_offset;
  var b_base = um.b_offset;
  for (var kk = 0u; kk < um.batch_ndim; kk = kk + 1u) {
    let d = um.batch_ndim - 1u - kk;
    let dim = um.batch_shape[d / 4u][d % 4u];
    let coord = rem % dim;
    rem = rem / dim;
    a_base = a_base + coord * um.a_batch_strides[d / 4u][d % 4u];
    b_base = b_base + coord * um.b_batch_strides[d / 4u][d % 4u];
  }

  var acc: f32 = 0.0;
  for (var p = 0u; p < um.k; p = p + 1u) {
    acc = acc + f32(a[a_base + row * um.a_s0 + p * um.a_s1]) * f32(b[b_base + p * um.b_s0 + col * um.b_s1]);
  }
  out[(bi * um.m + row) * um.n + col] = f16(acc);
}`;

function reduceShaderF16(op: "sum" | "max" | "min"): string {
  return /* wgsl */ `${F16_ENABLE}
${REDUCE_META}
@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read_write> out: array<f16>;
@group(0) @binding(2) var<uniform> um: Meta;

var<workgroup> partials: array<f32, ${WORKGROUP_SIZE}>;
${INF_HELPERS}
${NAN_HELPER}
fn combine(x: f32, y: f32) -> f32 {
  ${REDUCE_COMBINE[op]}
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let group = wid.y * nwg.x + wid.x;
  let i = group * ${WORKGROUP_SIZE}u + lid.x;
  if (i < um.size) {
    var rem = i;
    var idx = um.offset;
    for (var k = 0u; k < um.ndim; k = k + 1u) {
      let d = um.ndim - 1u - k;
      let dim = um.shape[d / 4u][d % 4u];
      let coord = rem % dim;
      rem = rem / dim;
      idx = idx + coord * um.strides[d / 4u][d % 4u];
    }
    partials[lid.x] = f32(input[idx]);
  } else {
    partials[lid.x] = ${REDUCE_IDENTITY[op]};
  }
  workgroupBarrier();

  for (var stride = ${WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride >> 1u) {
    if (lid.x < stride) {
      partials[lid.x] = combine(partials[lid.x], partials[lid.x + stride]);
    }
    workgroupBarrier();
  }

  if (lid.x == 0u && i < um.size) {
    out[group] = f16(partials[0] / um.divisor);
  }
}`;
}

function axisReduceShaderF16(op: "sum" | "max" | "min"): string {
  return /* wgsl */ `${F16_ENABLE}
struct Meta {
  outSize: u32,
  axisDim: u32,
  outNdim: u32,
  inOffset: u32,
  axisStride: u32,
  divisor: f32,
  _pad0: u32,
  _pad1: u32,
  outShape: array<vec4<u32>, 2>,
  outToInStride: array<vec4<u32>, 2>,
};
@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read_write> out: array<f16>;
@group(0) @binding(2) var<uniform> um: Meta;
${INF_HELPERS}
${NAN_HELPER}
fn combine(x: f32, y: f32) -> f32 {
  ${REDUCE_COMBINE[op]}
}
@compute @workgroup_size(${WORKGROUP_SIZE})
${MAIN_1D}
  if (i >= um.outSize) { return; }
  var rem = i;
  var base = um.inOffset;
  for (var k = 0u; k < um.outNdim; k = k + 1u) {
    let d = um.outNdim - 1u - k;
    let dim = um.outShape[d / 4u][d % 4u];
    let coord = rem % dim;
    rem = rem / dim;
    base = base + coord * um.outToInStride[d / 4u][d % 4u];
  }
  var acc = ${REDUCE_IDENTITY[op]};
  for (var j = 0u; j < um.axisDim; j = j + 1u) {
    acc = combine(acc, f32(input[base + j * um.axisStride]));
  }
  out[i] = f16(acc / um.divisor);
}`;
}

const FILL_SHADER_F16 = /* wgsl */ `${F16_ENABLE}
struct Meta {
  size: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
  value: f32,
  _pad3: f32,
  _pad4: f32,
  _pad5: f32,
};
@group(0) @binding(0) var<storage, read_write> out: array<f16>;
@group(0) @binding(1) var<uniform> um: Meta;

@compute @workgroup_size(${WORKGROUP_SIZE})
${MAIN_1D}
  if (i < um.size) {
    out[i] = f16(um.value);
  }
}`;

/**
 * Built-in WGSL compute shader sources, keyed by kernel name.
 *
 * Exposed for inspection, testing and custom integrations; the backend
 * compiles these lazily into cached compute pipelines. float16 variants are
 * registered under the `${name}__f16` key (see {@link f16ShaderName}).
 */
export const WGSL_SHADERS: Record<string, string> = (() => {
  const shaders: Record<string, string> = {};
  for (const op of Object.keys(BINARY_EXPR) as BinaryKernelOp[]) {
    shaders[op] = binaryShader(op);
    shaders[`${op}__f16`] = binaryShaderF16(op);
  }
  for (const op of Object.keys(UNARY_EXPR) as UnaryKernelOp[]) {
    shaders[op] = unaryShader(op);
    shaders[`${op}__f16`] = unaryShaderF16(op);
  }
  shaders["matmul"] = MATMUL_SHADER;
  shaders["matmul__f16"] = MATMUL_SHADER_F16;
  shaders["matmulBatched"] = BATCHED_MATMUL_SHADER;
  shaders["matmulBatched__f16"] = BATCHED_MATMUL_SHADER_F16;
  shaders["reduceSum"] = reduceShader("sum");
  shaders["reduceMax"] = reduceShader("max");
  shaders["reduceMin"] = reduceShader("min");
  shaders["reduceSum__f16"] = reduceShaderF16("sum");
  shaders["reduceMax__f16"] = reduceShaderF16("max");
  shaders["reduceMin__f16"] = reduceShaderF16("min");
  shaders["axisReduceSum"] = axisReduceShader("sum");
  shaders["axisReduceMax"] = axisReduceShader("max");
  shaders["axisReduceMin"] = axisReduceShader("min");
  shaders["axisReduceSum__f16"] = axisReduceShaderF16("sum");
  shaders["axisReduceMax__f16"] = axisReduceShaderF16("max");
  shaders["axisReduceMin__f16"] = axisReduceShaderF16("min");
  shaders["where"] = TERNARY_WHERE_SHADER;
  shaders["im2col"] = IM2COL_SHADER;
  shaders["col2im"] = COL2IM_SHADER;
  shaders["pool2dMax"] = pool2dShader("max");
  shaders["pool2dAvg"] = pool2dShader("avg");
  shaders["pool2dBackwardMax"] = pool2dBackwardShader("max");
  shaders["pool2dBackwardAvg"] = pool2dBackwardShader("avg");
  shaders["fill"] = FILL_SHADER;
  shaders["fill__f16"] = FILL_SHADER_F16;
  return shaders;
})();

// ─── dtype helpers ───────────────────────────────────────────────────────────

/**
 * Bytes per element of on-device storage for a {@link DeviceDType}. float16 is
 * a true 2-byte half; bfloat16 is stored as float32 on-device (host-rounded),
 * so it, like float32, is 4 bytes.
 */
function deviceBytesPerElement(dtype: DeviceDType | undefined): number {
  return dtype === "float16" ? 2 : 4;
}

/** Whether a dtype uses true on-device half (`array<f16>`) storage. */
function isF16Storage(dtype: DeviceDType | undefined): boolean {
  return dtype === "float16";
}

/**
 * Pick the shader variant name for a dtype: the `${name}__f16` half-precision
 * variant for float16, otherwise the float32 shader (also used by bfloat16,
 * which computes in f32 on-device).
 */
function variantName(name: string, dtype: DeviceDType | undefined): string {
  return isF16Storage(dtype) ? `${name}__f16` : name;
}

/**
 * Available shader names.
 */
export type ShaderName = keyof typeof WGSL_SHADERS;

/**
 * Information about a compiled GPU pipeline.
 */
export type GpuPipelineInfo = {
  readonly name: ShaderName;
  readonly workgroupSize: number;
};

/** Device buffer handle owned by the WebGPU backend. */
type GpuDeviceBuffer = DeviceBuffer & {
  gpuBuffer: GpuBuffer;
  freed: boolean;
};

function isGpuDeviceBuffer(buffer: DeviceBuffer): buffer is GpuDeviceBuffer {
  return buffer.device === "webgpu" && typeof (buffer as GpuDeviceBuffer).gpuBuffer === "object";
}

/** Pack layout metadata into the uniform vec4<u32> representation. */
function packDims(values: readonly number[]): number[] {
  const packed = new Array<number>(MAX_RANK).fill(0);
  for (let i = 0; i < values.length; i++) {
    packed[i] = values[i] ?? 0;
  }
  return packed;
}

function sizeOf(shape: readonly number[]): number {
  let size = 1;
  for (const dim of shape) size *= dim;
  return size;
}

function validateRank(layout: KernelLayout, op: string): void {
  if (layout.shape.length > MAX_RANK) {
    throw new DeviceError(
      `webgpu ${op} kernels support tensors up to rank ${MAX_RANK}; received rank ${layout.shape.length}`
    );
  }
}

/** Reject layouts whose shape and stride lists disagree (they would be mis-aligned on the device). */
function validateLayout(layout: KernelLayout, op: string): void {
  validateRank(layout, op);
  if (layout.strides.length !== layout.shape.length) {
    throw new DeviceError(
      `webgpu ${op}: layout has ${layout.shape.length} dimensions but ${layout.strides.length} strides`
    );
  }
}

/** Reject an operand layout that was not broadcast to `outShape` (same rank, one stride per dim). */
function validateOperandLayout(
  layout: KernelLayout,
  outShape: readonly number[],
  op: string
): void {
  if (layout.strides.length !== outShape.length) {
    throw new DeviceError(
      `webgpu ${op}: operand layout has ${layout.strides.length} strides but the output has ` +
        `${outShape.length} dimensions; broadcast operands to the output shape first`
    );
  }
}

/**
 * Reject a launch whose grid exceeds the per-dimension workgroup limit. Only
 * the 2-D/3-D matmul grids can hit it (more than 1,048,560 rows, columns or
 * batch entries); 1-D kernels fold into a 2-D grid instead.
 */
function assertGrid(op: string, x: number, y: number, z: number): void {
  if (x > MAX_GROUPS_PER_DIM || y > MAX_GROUPS_PER_DIM || z > MAX_GROUPS_PER_DIM) {
    throw new DeviceError(
      `webgpu ${op}: workgroup grid ${x}x${y}x${z} exceeds the device limit of ` +
        `${MAX_GROUPS_PER_DIM} per dimension; split the operands into smaller blocks`
    );
  }
}

/** Validate an element count argument (`fill` size). */
function validateCount(value: number, name: string, op: string): void {
  if (!Number.isInteger(value) || value < 0) {
    throw new DeviceError(
      `webgpu ${op}: ${name} must be a non-negative integer; received ${value}`
    );
  }
}

/**
 * WebGPU execution backend.
 *
 * Detects WebGPU availability during `init()` and executes the built-in
 * float32 tensor kernels ({@link KernelBackend}) on the GPU. Buffers are
 * pooled and reused; call {@link WebGpuBackend.dispose} to release
 * everything.
 *
 * @example
 * ```ts
 * import { WebGpuBackend, registerBackend } from 'deepbox/core';
 * import { dot, tensor } from 'deepbox/ndarray';
 *
 * const gpu = new WebGpuBackend();
 * await gpu.init();
 * if (gpu.info().available) {
 *   registerBackend('webgpu', gpu);
 *   const a = tensor([[1, 2], [3, 4]], { device: 'webgpu' });
 *   const b = dot(a, a);           // executes on the GPU
 *   const host = await b.cpu();    // read the result back
 * }
 * ```
 */
export class WebGpuBackend implements KernelBackend {
  private device: GpuDevice | null = null;
  /** Largest single buffer (and storage binding) this device accepts, in bytes. */
  private maxBufferBytes = DEFAULT_MAX_BUFFER_BYTES;
  private pipelines = new Map<string, GpuComputePipeline>();
  private pool = new Map<number, GpuBuffer[]>();
  private uniformPool = new Map<number, GpuBuffer[]>();
  private pooledBytes = 0;
  private disposed = false;
  private initPromise: Promise<void> | null = null;
  private readonly gpuProvider: GpuLike | null;
  /** Whether the requested device enabled the `shader-f16` feature. */
  private f16Supported = false;

  private static readonly CAPABILITIES: readonly BackendCapability[] = [
    "matmul",
    "conv2d",
    "reduction",
    "elementwise",
  ];

  /**
   * @param options.gpu - Explicit WebGPU entry point for runtimes without
   *   `navigator.gpu` (e.g. Node.js with a Dawn binding).
   */
  constructor(options: { readonly gpu?: GpuLike } = {}) {
    this.gpuProvider = options.gpu ?? null;
  }

  info(): BackendInfo {
    return {
      device: "webgpu",
      name: "Deepbox WebGPU Backend",
      available: this.device !== null && !this.disposed,
      capabilities: WebGpuBackend.CAPABILITIES,
    };
  }

  supports(cap: BackendCapability): boolean {
    for (const c of WebGpuBackend.CAPABILITIES) {
      if (c === cap) return true;
    }
    return false;
  }

  /**
   * Initialize the WebGPU backend.
   *
   * Requests a GPU adapter and device. If WebGPU is not available,
   * this method completes without error but the backend reports
   * `available: false`. Calling `init()` again returns the same result; a
   * disposed backend cannot be initialized again.
   *
   * The device is requested with the adapter's maximum buffer size and
   * storage-binding size so tensors larger than the 128 MiB default limit
   * work. If the device is lost later (driver reset, GPU removed), the
   * backend reports `available: false`.
   */
  async init(): Promise<void> {
    if (this.initPromise) return this.initPromise;
    this.initPromise = this.doInit();
    return this.initPromise;
  }

  private async doInit(): Promise<void> {
    const gpu = this.resolveGpu();
    if (!gpu || this.disposed) return;

    try {
      const raw = await gpu.requestAdapter();
      if (!raw) return;
      // The real adapter type is nominal, so it is narrowed to the structural
      // subset used here at this single boundary.
      const adapter = raw as GpuAdapter;
      // Enable true on-device half precision when the adapter advertises it.
      // Guarded so backends on GPUs without the feature still initialize (f16
      // tensors then throw a clear DeviceError at upload time).
      const f16 = adapter.features.has("shader-f16");
      const base: GpuDeviceDescriptor = f16 ? { requiredFeatures: ["shader-f16"] } : {};
      const large = this.largeBufferLimits(adapter);

      let device: GpuDevice;
      let maxBytes = DEFAULT_MAX_BUFFER_BYTES;
      if (large) {
        try {
          device = await adapter.requestDevice({ ...base, requiredLimits: large });
          maxBytes = Math.min(large.maxBufferSize, large.maxStorageBufferBindingSize);
        } catch {
          // The raised limits were refused; fall back to the defaults.
          device = await adapter.requestDevice(base);
        }
      } else {
        device = await adapter.requestDevice(base);
      }
      this.finishInit(device, f16, maxBytes);
    } catch {
      this.device = null;
      this.f16Supported = false;
    }
  }

  /** The adapter's maximum buffer limits when they exceed the defaults, else `null`. */
  private largeBufferLimits(
    adapter: GpuAdapter
  ): { maxBufferSize: number; maxStorageBufferBindingSize: number } | null {
    const limits = adapter.limits;
    const maxBufferSize = limits["maxBufferSize"];
    const maxStorageBufferBindingSize = limits["maxStorageBufferBindingSize"];
    if (
      typeof maxBufferSize !== "number" ||
      typeof maxStorageBufferBindingSize !== "number" ||
      Math.min(maxBufferSize, maxStorageBufferBindingSize) <= DEFAULT_MAX_BUFFER_BYTES
    ) {
      return null;
    }
    return { maxBufferSize, maxStorageBufferBindingSize };
  }

  /** Publish a freshly acquired device unless the backend was disposed in the meantime. */
  private finishInit(device: GpuDevice, f16: boolean, maxBytes: number): void {
    if (this.disposed) {
      device.destroy();
      return;
    }
    this.device = device;
    this.f16Supported = f16;
    // Buffers are addressed with u32 element indices in the kernels.
    this.maxBufferBytes = Math.min(Math.floor(maxBytes / 4) * 4, 0xfffffffc);
    // Some Node bindings omit `lost`, so check it at runtime despite the typing.
    const lost: Promise<GpuDeviceLostInfo> | undefined = device.lost;
    if (lost && typeof lost.then === "function") {
      lost.then(
        () => this.onDeviceLost(device),
        () => this.onDeviceLost(device)
      );
    }
  }

  /** Drop all state tied to a device that was lost (or destroyed). */
  private onDeviceLost(device: GpuDevice): void {
    if (this.device !== device) return;
    this.device = null;
    this.f16Supported = false;
    this.pool.clear();
    this.uniformPool.clear();
    this.pipelines.clear();
    this.pooledBytes = 0;
  }

  /** Whether this backend can execute true on-device float16 kernels. */
  supportsF16(): boolean {
    return this.f16Supported && this.device !== null && !this.disposed;
  }

  /** Ensure the requested dtype can be created on this device. */
  private assertDTypeSupported(dtype: DeviceDType, op: string): void {
    if (dtype !== "float32" && dtype !== "float16" && dtype !== "bfloat16") {
      throw new DeviceError(
        `webgpu ${op}: unsupported device dtype "${String(dtype)}"; ` +
          "use float32, float16 or bfloat16"
      );
    }
    if (dtype === "float16" && !this.f16Supported) {
      throw new DeviceError(
        `webgpu ${op}: float16 tensors require the WebGPU 'shader-f16' feature, ` +
          "which this adapter/GPU does not support. Use float32, or bfloat16 " +
          "(which computes in float32 on-device)."
      );
    }
  }

  /**
   * Resolve the shared dtype of binary/ternary operands, rejecting mixed
   * dtypes (no silent upcast: the caller must cast explicitly).
   */
  private resolveDType(op: string, buffers: readonly DeviceBuffer[]): DeviceDType {
    const dtype = buffers[0]?.dtype ?? "float32";
    for (const buf of buffers) {
      const d = buf.dtype ?? "float32";
      if (d !== dtype) {
        throw new DeviceError(
          `webgpu ${op}: operands have mismatched dtypes ("${dtype}" vs "${d}"). ` +
            "Device kernels do not upcast; cast operands to a common dtype first " +
            "(e.g. move to CPU and `astype`)."
        );
      }
    }
    return dtype;
  }

  /** Reject true-half (f16) buffers on kernels that have no half-precision variant. */
  private assertNotF16(op: string, buffers: readonly DeviceBuffer[]): void {
    for (const buf of buffers) {
      if (isF16Storage(buf.dtype)) {
        throw new DeviceError(
          `webgpu ${op}: float16 tensors are not supported by this kernel. ` +
            "Move to the CPU first with `await t.cpu()`, or cast to float32."
        );
      }
    }
  }

  private resolveGpu(): GpuLike | null {
    if (this.gpuProvider) return this.gpuProvider;
    if (typeof navigator !== "undefined" && "gpu" in navigator) {
      const gpu = (navigator as unknown as { gpu?: GpuLike }).gpu;
      if (gpu && typeof gpu === "object") return gpu;
    }
    return null;
  }

  private requireDevice(op: string): GpuDevice {
    if (this.disposed) {
      throw new DeviceError(`webgpu ${op}: backend has been disposed`);
    }
    if (!this.device) {
      throw new DeviceError(
        `webgpu ${op}: backend is not initialized or WebGPU is unavailable. ` +
          "Call `await backend.init()` and check `backend.info().available` before registering."
      );
    }
    return this.device;
  }

  // ─── KernelBackend: memory ─────────────────────────────────────────────────

  upload(data: Float32Array, dtype: DeviceDType = "float32"): DeviceBuffer {
    const device = this.requireDevice("upload");
    this.assertDTypeSupported(dtype, "upload");
    const bytesPerEl = deviceBytesPerElement(dtype);
    const buffer = this.acquireBuffer(device, data.length, bytesPerEl);

    let payload: ArrayBuffer;
    if (dtype === "float16") {
      // Pack each value into a 2-byte half. Pad to an even element count so
      // the written byte length is a multiple of 4 (a WebGPU writeBuffer
      // requirement); the extra tail half is unused.
      const packed = new Uint16Array(data.length + (data.length & 1));
      for (let i = 0; i < data.length; i++) {
        packed[i] = float64ToFloat16Bits(data[i] ?? 0);
      }
      payload = packed.buffer as ArrayBuffer;
    } else if (dtype === "bfloat16") {
      // Round each value to bfloat16 and back to float32; the on-device
      // storage is float32 and the GPU computes in float32 (download
      // re-rounds), giving correct bf16 numerics without a native bf16 type.
      const rounded = new Float32Array(data.length);
      for (let i = 0; i < data.length; i++) {
        rounded[i] = roundToBFloat16(data[i] ?? 0);
      }
      payload = rounded.buffer as ArrayBuffer;
    } else if (
      data.byteOffset === 0 &&
      data.buffer instanceof ArrayBuffer &&
      data.byteLength === data.buffer.byteLength
    ) {
      // `writeBuffer` copies at call time, so the array's own buffer can be
      // passed without duplicating it.
      payload = data.buffer;
    } else {
      payload = data.buffer.slice(
        data.byteOffset,
        data.byteOffset + data.byteLength
      ) as ArrayBuffer;
    }

    device.queue.writeBuffer(buffer, 0, payload);
    return this.wrap(buffer, data.length, dtype);
  }

  async download(buffer: DeviceBuffer): Promise<Float32Array> {
    const device = this.requireDevice("download");
    const gpuBuf = this.unwrap(buffer, "download");
    const dtype = (buffer.dtype ?? "float32") as DeviceDType;
    if (buffer.size === 0) return new Float32Array(0);
    const dataBytes = buffer.size * deviceBytesPerElement(dtype);
    // Copy/staging sizes must be multiples of 4.
    const byteLength = Math.ceil(dataBytes / 4) * 4;

    const staging = device.createBuffer({
      size: byteLength,
      usage: GPU_BUFFER_USAGE.MAP_READ | GPU_BUFFER_USAGE.COPY_DST,
    });
    try {
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(gpuBuf, 0, staging, 0, byteLength);
      device.queue.submit([encoder.finish()]);

      await staging.mapAsync(GPU_MAP_MODE.READ);
      const mapped = staging.getMappedRange();
      let result: Float32Array;
      if (dtype === "float16") {
        const bits = new Uint16Array(mapped, 0, buffer.size);
        result = new Float32Array(buffer.size);
        for (let i = 0; i < buffer.size; i++) result[i] = float16BitsToFloat64(bits[i] ?? 0);
      } else if (dtype === "bfloat16") {
        // Storage is float32; re-round each value to bfloat16 so chained
        // on-device f32 arithmetic still reads back as bf16 to the user.
        const vals = new Float32Array(mapped, 0, buffer.size);
        result = new Float32Array(buffer.size);
        for (let i = 0; i < buffer.size; i++) {
          result[i] = roundToBFloat16(vals[i] ?? 0);
        }
      } else {
        result = new Float32Array(mapped.slice(0, buffer.size * 4));
      }
      staging.unmap();
      return result;
    } finally {
      staging.destroy();
    }
  }

  free(buffer: DeviceBuffer): void {
    if (!isGpuDeviceBuffer(buffer) || buffer.freed || this.disposed) return;
    buffer.freed = true;
    this.releaseBuffer(buffer.gpuBuffer);
  }

  fill(value: number, size: number, dtype: DeviceDType = "float32"): DeviceBuffer {
    const device = this.requireDevice("fill");
    validateCount(size, "size", "fill");
    this.assertDTypeSupported(dtype, "fill");
    const out = this.acquireBuffer(device, Math.max(size, 1), deviceBytesPerElement(dtype));
    const meta = new ArrayBuffer(32);
    new Uint32Array(meta, 0, 4)[0] = size;
    // bfloat16 fill stores float32 `value`; download re-rounds to bf16.
    new Float32Array(meta, 16, 4)[0] = value;
    this.dispatch(variantName("fill", dtype), [out], meta, Math.ceil(size / WORKGROUP_SIZE));
    return this.wrap(out, size, dtype);
  }

  // ─── KernelBackend: kernels ────────────────────────────────────────────────

  binary(
    op: BinaryKernelOp,
    a: DeviceBuffer,
    aLayout: KernelLayout,
    b: DeviceBuffer,
    bLayout: KernelLayout,
    outShape: readonly number[]
  ): DeviceBuffer {
    const device = this.requireDevice(op);
    validateRank({ shape: outShape, strides: [], offset: 0 }, op);
    validateOperandLayout(aLayout, outShape, op);
    validateOperandLayout(bLayout, outShape, op);
    const dtype = this.resolveDType(op, [a, b]);
    const size = sizeOf(outShape);
    const aBuf = this.unwrap(a, op);
    const bBuf = this.unwrap(b, op);
    const out = this.acquireBuffer(device, Math.max(size, 1), deviceBytesPerElement(dtype));

    // Meta: size, ndim, aOffset, bOffset, shape[8], aStrides[8], bStrides[8]
    const meta = new ArrayBuffer(16 + 3 * 32);
    const u32 = new Uint32Array(meta);
    u32[0] = size;
    u32[1] = outShape.length;
    u32[2] = aLayout.offset;
    u32[3] = bLayout.offset;
    u32.set(packDims(outShape), 4);
    u32.set(packDims(aLayout.strides), 12);
    u32.set(packDims(bLayout.strides), 20);

    this.dispatch(
      variantName(op, dtype),
      [aBuf, bBuf, out],
      meta,
      Math.ceil(size / WORKGROUP_SIZE)
    );
    return this.wrap(out, size, dtype);
  }

  unary(op: UnaryKernelOp, x: DeviceBuffer, layout: KernelLayout): DeviceBuffer {
    const device = this.requireDevice(op);
    validateLayout(layout, op);
    const dtype = (x.dtype ?? "float32") as DeviceDType;
    const size = sizeOf(layout.shape);
    const xBuf = this.unwrap(x, op);
    const out = this.acquireBuffer(device, Math.max(size, 1), deviceBytesPerElement(dtype));

    this.dispatch(
      variantName(op, dtype),
      [xBuf, out],
      this.unaryMeta(size, layout),
      Math.ceil(size / WORKGROUP_SIZE)
    );
    return this.wrap(out, size, dtype);
  }

  matmul(
    a: DeviceBuffer,
    aLayout: KernelLayout,
    b: DeviceBuffer,
    bLayout: KernelLayout
  ): DeviceBuffer {
    const device = this.requireDevice("matmul");
    if (aLayout.shape.length !== 2 || bLayout.shape.length !== 2) {
      throw new DeviceError("webgpu matmul requires rank-2 layouts");
    }
    validateLayout(aLayout, "matmul");
    validateLayout(bLayout, "matmul");
    const dtype = this.resolveDType("matmul", [a, b]);
    const m = aLayout.shape[0] ?? 0;
    const k = aLayout.shape[1] ?? 0;
    const n = bLayout.shape[1] ?? 0;
    if ((bLayout.shape[0] ?? 0) !== k) {
      throw new DeviceError(
        `webgpu matmul: inner dimensions do not match ([${aLayout.shape.join(", ")}] @ ` +
          `[${bLayout.shape.join(", ")}])`
      );
    }
    const aBuf = this.unwrap(a, "matmul");
    const bBuf = this.unwrap(b, "matmul");
    assertGrid("matmul", Math.ceil(n / 16), Math.ceil(m / 16), 1);
    const out = this.acquireBuffer(device, Math.max(m * n, 1), deviceBytesPerElement(dtype));

    const meta = new ArrayBuffer(48);
    const u32 = new Uint32Array(meta);
    u32[0] = m;
    u32[1] = n;
    u32[2] = k;
    u32[3] = aLayout.offset;
    u32[4] = aLayout.strides[0] ?? 0;
    u32[5] = aLayout.strides[1] ?? 0;
    u32[6] = bLayout.strides[0] ?? 0;
    u32[7] = bLayout.strides[1] ?? 0;
    u32[8] = bLayout.offset;

    this.dispatchGrid(
      variantName("matmul", dtype),
      [aBuf, bBuf, out],
      meta,
      Math.ceil(n / 16),
      Math.ceil(m / 16)
    );
    return this.wrap(out, m * n, dtype);
  }

  reduce(op: ReduceKernelOp, x: DeviceBuffer, layout: KernelLayout): DeviceBuffer {
    this.requireDevice(op);
    validateLayout(layout, op);
    const dtype = (x.dtype ?? "float32") as DeviceDType;
    const size = sizeOf(layout.shape);
    if (size === 0) {
      throw new DeviceError(`webgpu ${op}: cannot reduce an empty tensor on the device`);
    }

    const baseOp: "sum" | "max" | "min" = op === "mean" ? "sum" : op;
    const shaderName = variantName(
      baseOp === "sum" ? "reduceSum" : baseOp === "max" ? "reduceMax" : "reduceMin",
      dtype
    );
    const xBuf = this.unwrap(x, op);

    // Pass 1: strided load + workgroup-tree reduce. For `mean` every partial
    // is divided by the element count here, so later passes only add.
    let currentSize = size;
    let current = this.reducePass(
      shaderName,
      xBuf,
      this.reduceMeta(size, layout, op === "mean" ? size : 1),
      currentSize,
      dtype
    );
    currentSize = Math.ceil(currentSize / WORKGROUP_SIZE);

    // Subsequent passes: contiguous partials until a single value remains.
    while (currentSize > 1) {
      const layout1d: KernelLayout = { shape: [currentSize], strides: [1], offset: 0 };
      const next = this.reducePass(
        shaderName,
        current,
        this.reduceMeta(currentSize, layout1d, 1),
        currentSize,
        dtype
      );
      this.releaseBuffer(current);
      current = next;
      currentSize = Math.ceil(currentSize / WORKGROUP_SIZE);
    }

    return this.wrap(current, 1, dtype);
  }

  reduceAxis(
    op: ReduceKernelOp,
    x: DeviceBuffer,
    layout: KernelLayout,
    axis: number
  ): DeviceBuffer {
    const device = this.requireDevice(op);
    validateLayout(layout, op);
    const ndim = layout.shape.length;
    if (!Number.isInteger(axis) || axis < 0 || axis >= ndim) {
      throw new DeviceError(`webgpu ${op}: axis ${axis} out of range for rank ${ndim}`);
    }
    const axisDim = layout.shape[axis] ?? 0;
    if (axisDim === 0) {
      throw new DeviceError(`webgpu ${op}: cannot reduce a zero-length axis on the device`);
    }
    // Output shape = input shape with `axis` removed; outToInStride[j] is the
    // input stride of the input dim that output dim j maps to.
    const outShape: number[] = [];
    const outToInStride: number[] = [];
    for (let d = 0; d < ndim; d++) {
      if (d === axis) continue;
      outShape.push(layout.shape[d] ?? 0);
      outToInStride.push(layout.strides[d] ?? 0);
    }
    const outSize = sizeOf(outShape);
    const dtype = (x.dtype ?? "float32") as DeviceDType;
    const baseOp: "sum" | "max" | "min" = op === "mean" ? "sum" : op;
    const shaderName = variantName(
      baseOp === "sum" ? "axisReduceSum" : baseOp === "max" ? "axisReduceMax" : "axisReduceMin",
      dtype
    );
    const xBuf = this.unwrap(x, op);

    const out = this.acquireBuffer(device, Math.max(outSize, 1), deviceBytesPerElement(dtype));
    // Meta: outSize,axisDim,outNdim,inOffset,axisStride,divisor,pad,pad, outShape[8], outToInStride[8]
    const meta = new ArrayBuffer(32 + 2 * 32);
    const u32 = new Uint32Array(meta);
    const f32 = new Float32Array(meta);
    u32[0] = outSize;
    u32[1] = axisDim;
    u32[2] = outShape.length;
    u32[3] = layout.offset;
    u32[4] = layout.strides[axis] ?? 0;
    f32[5] = op === "mean" ? axisDim : 1;
    u32.set(packDims(outShape), 8);
    u32.set(packDims(outToInStride), 16);

    this.dispatch(shaderName, [xBuf, out], meta, Math.ceil(Math.max(outSize, 1) / WORKGROUP_SIZE));
    return this.wrap(out, outSize, dtype);
  }

  matmulBatched(
    a: DeviceBuffer,
    aLayout: KernelLayout,
    b: DeviceBuffer,
    bLayout: KernelLayout,
    batch: number,
    m: number,
    k: number,
    n: number
  ): DeviceBuffer {
    const device = this.requireDevice("matmulBatched");
    validateLayout(aLayout, "matmulBatched");
    validateLayout(bLayout, "matmulBatched");
    const rank = aLayout.shape.length;
    if (rank < 2 || bLayout.shape.length !== rank) {
      throw new DeviceError(
        "webgpu matmulBatched requires both layouts to have the same rank, at least 2"
      );
    }
    const batchNdim = rank - 2;
    const batchShape = aLayout.shape.slice(0, batchNdim);
    const bBatchShape = bLayout.shape.slice(0, batchNdim);
    if (
      sizeOf(batchShape) !== batch ||
      batchShape.some((d, i) => d !== bBatchShape[i]) ||
      aLayout.shape[batchNdim] !== m ||
      aLayout.shape[batchNdim + 1] !== k ||
      bLayout.shape[batchNdim] !== k ||
      bLayout.shape[batchNdim + 1] !== n
    ) {
      throw new DeviceError(
        `webgpu matmulBatched: layouts [${aLayout.shape.join(", ")}] and [${bLayout.shape.join(", ")}] ` +
          `do not match batch=${batch}, m=${m}, k=${k}, n=${n}`
      );
    }
    // Inner 2-D strides are the last two of each operand.
    const aS0 = aLayout.strides[batchNdim] ?? 0;
    const aS1 = aLayout.strides[batchNdim + 1] ?? 0;
    const bS0 = bLayout.strides[batchNdim] ?? 0;
    const bS1 = bLayout.strides[batchNdim + 1] ?? 0;
    const aBatchStrides = aLayout.strides.slice(0, batchNdim);
    const bBatchStrides = bLayout.strides.slice(0, batchNdim);
    const dtype = this.resolveDType("matmulBatched", [a, b]);
    const aBuf = this.unwrap(a, "matmulBatched");
    const bBuf = this.unwrap(b, "matmulBatched");
    assertGrid("matmulBatched", Math.ceil(n / 16), Math.ceil(m / 16), batch);

    const out = this.acquireBuffer(
      device,
      Math.max(batch * m * n, 1),
      deviceBytesPerElement(dtype)
    );
    // Meta: m,n,k,batch,aOff,bOff,aS0,aS1,bS0,bS1,batchNdim,pad, batchShape[8], aBatchStrides[8], bBatchStrides[8]
    const meta = new ArrayBuffer(48 + 3 * 32);
    const u32 = new Uint32Array(meta);
    u32[0] = m;
    u32[1] = n;
    u32[2] = k;
    u32[3] = batch;
    u32[4] = aLayout.offset;
    u32[5] = bLayout.offset;
    u32[6] = aS0;
    u32[7] = aS1;
    u32[8] = bS0;
    u32[9] = bS1;
    u32[10] = batchNdim;
    u32.set(packDims(batchShape), 12);
    u32.set(packDims(aBatchStrides), 20);
    u32.set(packDims(bBatchStrides), 28);

    this.dispatchGrid(
      variantName("matmulBatched", dtype),
      [aBuf, bBuf, out],
      meta,
      Math.ceil(n / 16),
      Math.ceil(m / 16),
      batch
    );
    return this.wrap(out, batch * m * n, dtype);
  }

  ternary(
    op: TernaryKernelOp,
    cond: DeviceBuffer,
    condLayout: KernelLayout,
    a: DeviceBuffer,
    aLayout: KernelLayout,
    b: DeviceBuffer,
    bLayout: KernelLayout,
    outShape: readonly number[]
  ): DeviceBuffer {
    const device = this.requireDevice(op);
    validateRank({ shape: outShape, strides: [], offset: 0 }, op);
    validateOperandLayout(condLayout, outShape, op);
    validateOperandLayout(aLayout, outShape, op);
    validateOperandLayout(bLayout, outShape, op);
    this.assertNotF16(op, [cond, a, b]);
    // The selected values keep their dtype; a bfloat16/float32 mix computes
    // in float32 (both are float32 on the device).
    const dtype: DeviceDType =
      (a.dtype ?? "float32") === (b.dtype ?? "float32") ? (a.dtype ?? "float32") : "float32";
    const size = sizeOf(outShape);
    const condBuf = this.unwrap(cond, op);
    const aBuf = this.unwrap(a, op);
    const bBuf = this.unwrap(b, op);
    const out = this.acquireBuffer(device, Math.max(size, 1));
    // Meta: size,ndim,cOff,aOff,bOff,pad,pad,pad, shape[8], cStrides[8], aStrides[8], bStrides[8]
    const meta = new ArrayBuffer(32 + 4 * 32);
    const u32 = new Uint32Array(meta);
    u32[0] = size;
    u32[1] = outShape.length;
    u32[2] = condLayout.offset;
    u32[3] = aLayout.offset;
    u32[4] = bLayout.offset;
    u32.set(packDims(outShape), 8);
    u32.set(packDims(condLayout.strides), 16);
    u32.set(packDims(aLayout.strides), 24);
    u32.set(packDims(bLayout.strides), 32);

    this.dispatch(op, [condBuf, aBuf, bBuf, out], meta, Math.ceil(size / WORKGROUP_SIZE));
    return this.wrap(out, size, dtype);
  }

  /**
   * Check convolution/pooling geometry before it reaches a shader, where a
   * zero stride would divide by zero and inconsistent sizes would read out of
   * bounds.
   */
  private validateConvParams(params: Im2ColParams, op: string): void {
    const { batch, channels, height, width, outH, outW, kH, kW } = params;
    const { strideH, strideW, padH, padW } = params;
    const positive = [height, width, kH, kW, strideH, strideW];
    const nonNegative = [batch, channels, padH, padW, outH, outW];
    if (
      positive.some((v) => !Number.isInteger(v) || v < 1) ||
      nonNegative.some((v) => !Number.isInteger(v) || v < 0)
    ) {
      throw new DeviceError(
        `webgpu ${op}: invalid window geometry (sizes, kernel and stride must be positive ` +
          `integers; batch, channels and padding non-negative integers)`
      );
    }
    const expectH = Math.floor((height + 2 * padH - kH) / strideH) + 1;
    const expectW = Math.floor((width + 2 * padW - kW) / strideW) + 1;
    if (outH !== Math.max(expectH, 0) || outW !== Math.max(expectW, 0)) {
      throw new DeviceError(
        `webgpu ${op}: output size ${outH}x${outW} does not match the window geometry ` +
          `(expected ${Math.max(expectH, 0)}x${Math.max(expectW, 0)})`
      );
    }
  }

  /** Check that an NCHW input layout matches the window geometry. */
  private validateConvInput(layout: KernelLayout, params: Im2ColParams, op: string): void {
    const s = layout.shape;
    if (
      layout.strides.length !== 4 ||
      s.length !== 4 ||
      s[0] !== params.batch ||
      s[1] !== params.channels ||
      s[2] !== params.height ||
      s[3] !== params.width
    ) {
      throw new DeviceError(
        `webgpu ${op}: input layout [${s.join(", ")}] does not match ` +
          `[${params.batch}, ${params.channels}, ${params.height}, ${params.width}]`
      );
    }
  }

  private convMeta(p: Im2ColParams, layout: KernelLayout, size: number): ArrayBuffer {
    // 20 u32 slots (see CONV_META). Strides default to contiguous NCHW when a
    // layout is not applicable (col2im input is already contiguous columns).
    const meta = new ArrayBuffer(80);
    const u = new Uint32Array(meta);
    u[0] = p.batch;
    u[1] = p.channels;
    u[2] = p.height;
    u[3] = p.width;
    u[4] = p.outH;
    u[5] = p.outW;
    u[6] = p.kH;
    u[7] = p.kW;
    u[8] = p.strideH;
    u[9] = p.strideW;
    u[10] = p.padH;
    u[11] = p.padW;
    u[12] = layout.offset;
    u[13] = layout.strides[0] ?? 0;
    u[14] = layout.strides[1] ?? 0;
    u[15] = layout.strides[2] ?? 0;
    u[16] = layout.strides[3] ?? 0;
    u[17] = size;
    return meta;
  }

  im2col(x: DeviceBuffer, layout: KernelLayout, params: Im2ColParams): DeviceBuffer {
    const device = this.requireDevice("im2col");
    this.assertNotF16("im2col", [x]);
    this.validateConvParams(params, "im2col");
    this.validateConvInput(layout, params, "im2col");
    const dtype = (x.dtype ?? "float32") as DeviceDType;
    const colSize = params.channels * params.kH * params.kW;
    const size = params.batch * params.outH * params.outW * colSize;
    const xBuf = this.unwrap(x, "im2col");
    const out = this.acquireBuffer(device, Math.max(size, 1));
    this.dispatch(
      "im2col",
      [xBuf, out],
      this.convMeta(params, layout, size),
      Math.ceil(Math.max(size, 1) / WORKGROUP_SIZE)
    );
    return this.wrap(out, size, dtype);
  }

  col2im(cols: DeviceBuffer, params: Im2ColParams): DeviceBuffer {
    const device = this.requireDevice("col2im");
    this.assertNotF16("col2im", [cols]);
    this.validateConvParams(params, "col2im");
    const dtype = (cols.dtype ?? "float32") as DeviceDType;
    const colCount =
      params.batch * params.outH * params.outW * params.channels * params.kH * params.kW;
    if (cols.size < colCount) {
      throw new DeviceError(
        `webgpu col2im: column buffer holds ${cols.size} elements but the geometry needs ${colCount}`
      );
    }
    const size = params.batch * params.channels * params.height * params.width;
    const colsBuf = this.unwrap(cols, "col2im");
    const out = this.acquireBuffer(device, Math.max(size, 1));
    // col2im reads contiguous columns; layout offset/strides are unused by the
    // shader (it decodes the contiguous NCHW output), so pass a zero layout.
    const zeroLayout: KernelLayout = { shape: [], strides: [], offset: 0 };
    this.dispatch(
      "col2im",
      [colsBuf, out],
      this.convMeta(params, zeroLayout, size),
      Math.ceil(Math.max(size, 1) / WORKGROUP_SIZE)
    );
    return this.wrap(out, size, dtype);
  }

  pool2d(
    x: DeviceBuffer,
    layout: KernelLayout,
    op: PoolKernelOp,
    params: Im2ColParams
  ): DeviceBuffer {
    const device = this.requireDevice("pool2d");
    this.assertNotF16("pool2d", [x]);
    this.validateConvParams(params, "pool2d");
    this.validateConvInput(layout, params, "pool2d");
    const dtype = (x.dtype ?? "float32") as DeviceDType;
    const size = params.batch * params.channels * params.outH * params.outW;
    const xBuf = this.unwrap(x, "pool2d");
    const out = this.acquireBuffer(device, Math.max(size, 1));
    this.dispatch(
      op === "max" ? "pool2dMax" : "pool2dAvg",
      [xBuf, out],
      this.convMeta(params, layout, size),
      Math.ceil(Math.max(size, 1) / WORKGROUP_SIZE)
    );
    return this.wrap(out, size, dtype);
  }

  pool2dBackward(
    x: DeviceBuffer,
    xLayout: KernelLayout,
    gradOut: DeviceBuffer,
    op: PoolKernelOp,
    params: Im2ColParams
  ): DeviceBuffer {
    const device = this.requireDevice("pool2dBackward");
    this.assertNotF16("pool2dBackward", [x, gradOut]);
    this.validateConvParams(params, "pool2dBackward");
    this.validateConvInput(xLayout, params, "pool2dBackward");
    const dtype = (x.dtype ?? "float32") as DeviceDType;
    const gradCount = params.batch * params.channels * params.outH * params.outW;
    if (gradOut.size < gradCount) {
      throw new DeviceError(
        `webgpu pool2dBackward: gradient buffer holds ${gradOut.size} elements but the geometry needs ${gradCount}`
      );
    }
    const size = params.batch * params.channels * params.height * params.width;
    const xBuf = this.unwrap(x, "pool2dBackward");
    const gradBuf = this.unwrap(gradOut, "pool2dBackward");
    const out = this.acquireBuffer(device, Math.max(size, 1));
    this.dispatch(
      op === "max" ? "pool2dBackwardMax" : "pool2dBackwardAvg",
      [xBuf, gradBuf, out],
      this.convMeta(params, xLayout, size),
      Math.ceil(Math.max(size, 1) / WORKGROUP_SIZE)
    );
    return this.wrap(out, size, dtype);
  }

  private reducePass(
    shaderName: string,
    input: GpuBuffer,
    meta: ArrayBuffer,
    inputSize: number,
    dtype: DeviceDType = "float32"
  ): GpuBuffer {
    const device = this.requireDevice(shaderName);
    const numGroups = Math.max(1, Math.ceil(inputSize / WORKGROUP_SIZE));
    const out = this.acquireBuffer(device, numGroups, deviceBytesPerElement(dtype));
    this.dispatch(shaderName, [input, out], meta, numGroups);
    return out;
  }

  private unaryMeta(size: number, layout: KernelLayout): ArrayBuffer {
    // Meta: size, ndim, offset, pad, shape[8], strides[8]
    const meta = new ArrayBuffer(16 + 2 * 32);
    const u32 = new Uint32Array(meta);
    u32[0] = size;
    u32[1] = layout.shape.length;
    u32[2] = layout.offset;
    u32.set(packDims(layout.shape), 4);
    u32.set(packDims(layout.strides), 12);
    return meta;
  }

  /** Meta of a full-reduction pass: the unary layout plus the f32 `divisor` in slot 3. */
  private reduceMeta(size: number, layout: KernelLayout, divisor: number): ArrayBuffer {
    const meta = this.unaryMeta(size, layout);
    new Float32Array(meta)[3] = divisor;
    return meta;
  }

  // ─── Pipeline + dispatch plumbing ─────────────────────────────────────────

  /**
   * Get or create a compute pipeline for the given shader.
   *
   * @param name - Shader name from WGSL_SHADERS
   * @returns The compiled compute pipeline, or null if unavailable
   */
  getPipeline(name: ShaderName): GpuComputePipeline | null {
    if (!this.device) return null;
    const existing = this.pipelines.get(name);
    if (existing) return existing;

    const code = WGSL_SHADERS[name];
    if (!code) return null;
    const module = this.device.createShaderModule({ code, label: `deepbox:${name}` });
    const pipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    this.pipelines.set(name, pipeline);
    return pipeline;
  }

  /**
   * Launch a 1-D kernel over `workgroups` workgroups. Counts above the
   * per-dimension limit are folded into a 2-D grid; the shaders rebuild the
   * linear index from `num_workgroups` (see `MAIN_1D`).
   */
  private dispatch(
    name: string,
    buffers: GpuBuffer[],
    meta: ArrayBuffer,
    workgroups: number
  ): void {
    const groups = Math.max(1, workgroups);
    if (groups <= MAX_GROUPS_PER_DIM) {
      this.dispatchGrid(name, buffers, meta, groups, 1);
    } else {
      this.dispatchGrid(
        name,
        buffers,
        meta,
        MAX_GROUPS_PER_DIM,
        Math.ceil(groups / MAX_GROUPS_PER_DIM)
      );
    }
  }

  private dispatchGrid(
    name: string,
    buffers: GpuBuffer[],
    meta: ArrayBuffer,
    groupsX: number,
    groupsY: number,
    groupsZ = 1
  ): void {
    const device = this.requireDevice(name);
    assertGrid(name, groupsX, groupsY, groupsZ);
    const pipeline = this.getPipeline(name);
    if (!pipeline) {
      throw new DeviceError(`webgpu: no kernel named "${name}"`);
    }

    // Reuse a pooled uniform buffer instead of allocating and destroying one
    // per dispatch. WebGPU queue operations are serialized on a single
    // timeline: this dispatch's `writeBuffer` + `submit` complete before the
    // next reuse of the same buffer writes new data, so a returned buffer is
    // safe to hand out again immediately, with no per-op allocation churn.
    const metaBuffer = this.acquireUniform(device, meta.byteLength);
    device.queue.writeBuffer(metaBuffer, 0, meta);

    const entries: GpuBindGroupEntry[] = buffers.map((buffer, i) => ({
      binding: i,
      resource: { buffer },
    }));
    entries.push({ binding: buffers.length, resource: { buffer: metaBuffer } });

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries,
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.max(1, groupsX), Math.max(1, groupsY), Math.max(1, groupsZ));
    pass.end();
    device.queue.submit([encoder.finish()]);

    this.releaseUniform(metaBuffer.size, metaBuffer);
  }

  /** Acquire a uniform buffer of at least `byteLength` bytes from the pool. */
  private acquireUniform(device: GpuDevice, byteLength: number): GpuBuffer {
    // Uniform buffers must be 16-byte aligned; our metas already are.
    const size = Math.max(16, byteLength);
    const list = this.uniformPool.get(size);
    const pooled = list?.pop();
    if (pooled) return pooled;
    return device.createBuffer({
      size,
      usage: GPU_BUFFER_USAGE.UNIFORM | GPU_BUFFER_USAGE.COPY_DST,
    });
  }

  /** Return a uniform buffer to the pool for reuse. */
  private releaseUniform(size: number, buffer: GpuBuffer): void {
    if (this.disposed || !this.device) {
      buffer.destroy();
      return;
    }
    const list = this.uniformPool.get(size) ?? [];
    // Cap per-size retention to avoid unbounded growth on pathological mixes.
    if (list.length < 64) {
      list.push(buffer);
      this.uniformPool.set(size, list);
    } else {
      buffer.destroy();
    }
  }

  // ─── Buffer pool ───────────────────────────────────────────────────────────

  private bucketFor(byteLength: number): number {
    // Round byte size up to the next power of two (min 256 B) so freed
    // buffers are reusable across similar-size allocations. Bucketing on BYTES
    // (not element count) keeps 2-byte f16 and 4-byte f32 buffers from
    // colliding in the pool. Buffers near the device limit use the limit
    // itself as their bucket, since the next power of two would not fit.
    let bucket = 256;
    while (bucket < byteLength) bucket *= 2;
    return Math.min(bucket, this.maxBufferBytes);
  }

  private acquireBuffer(device: GpuDevice, elements: number, bytesPerElement = 4): GpuBuffer {
    const bytes = elements * bytesPerElement;
    if (bytes > this.maxBufferBytes) {
      throw new DeviceError(
        `webgpu: a buffer of ${elements} elements (${bytes} bytes) exceeds the device limit of ` +
          `${this.maxBufferBytes} bytes. Split the tensor or process it in blocks.`
      );
    }
    const bucket = this.bucketFor(bytes);
    const list = this.pool.get(bucket);
    const pooled = list?.pop();
    if (pooled) {
      this.pooledBytes -= bucket;
      return pooled;
    }
    return device.createBuffer({
      size: bucket,
      usage: GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_SRC | GPU_BUFFER_USAGE.COPY_DST,
    });
  }

  private releaseBuffer(buffer: GpuBuffer): void {
    const bucket = buffer.size;
    if (this.disposed || !this.device || this.pooledBytes + bucket > POOL_CAP_BYTES) {
      buffer.destroy();
      return;
    }
    const list = this.pool.get(bucket) ?? [];
    list.push(buffer);
    this.pool.set(bucket, list);
    this.pooledBytes += bucket;
  }

  private wrap(
    buffer: GpuBuffer,
    elements: number,
    dtype: DeviceDType = "float32"
  ): GpuDeviceBuffer {
    return {
      device: "webgpu",
      byteLength: elements * deviceBytesPerElement(dtype),
      size: elements,
      dtype,
      gpuBuffer: buffer,
      freed: false,
    };
  }

  private unwrap(buffer: DeviceBuffer, op: string): GpuBuffer {
    if (!isGpuDeviceBuffer(buffer)) {
      throw new DeviceError(`webgpu ${op}: buffer does not belong to the WebGPU backend`);
    }
    if (buffer.freed) {
      throw new DeviceError(`webgpu ${op}: buffer has already been freed`);
    }
    return buffer.gpuBuffer;
  }

  // ─── Legacy/advanced surface ───────────────────────────────────────────────

  /**
   * Create a GPU buffer from a Float32Array (advanced usage).
   *
   * The buffer is not tracked by the backend's pool: release it with
   * `buffer.destroy()` when done.
   *
   * @param data - Source data
   * @param usage - Buffer usage flags
   * @returns GPU buffer, or null if device unavailable
   */
  createBuffer(data: Float32Array, usage: number): GpuBuffer | null {
    if (!this.device) return null;
    const buffer = this.device.createBuffer({
      size: data.byteLength,
      usage: usage | GPU_BUFFER_USAGE.COPY_SRC,
      mappedAtCreation: true,
    });
    new Float32Array(buffer.getMappedRange()).set(data);
    buffer.unmap();
    return buffer;
  }

  /**
   * Read data back from a GPU buffer (advanced usage).
   *
   * @param buffer - Source GPU buffer
   * @param size - Size in bytes to read; must be a multiple of 4
   * @returns Float32Array with the data (empty when the backend has no device)
   * @throws {DeviceError} If `size` is not a non-negative multiple of 4
   */
  async readBuffer(buffer: GpuBuffer, size: number): Promise<Float32Array> {
    if (!this.device) return new Float32Array(0);
    if (!Number.isInteger(size) || size < 0 || size % 4 !== 0) {
      throw new DeviceError(
        `webgpu readBuffer: size must be a non-negative multiple of 4 bytes; received ${size}`
      );
    }
    if (size === 0) return new Float32Array(0);
    const device = this.device;
    const staging = device.createBuffer({
      size,
      usage: GPU_BUFFER_USAGE.MAP_READ | GPU_BUFFER_USAGE.COPY_DST,
    });
    try {
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
      device.queue.submit([encoder.finish()]);

      await staging.mapAsync(GPU_MAP_MODE.READ);
      const result = new Float32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      return result;
    } finally {
      staging.destroy();
    }
  }

  /**
   * Get the underlying GPU device (for advanced usage).
   *
   * The result is typed as the structural {@link GpuDevice} subset the backend
   * uses. Cast it to `GPUDevice` (from `@webgpu/types`) to reach the rest of
   * the WebGPU API.
   */
  getDevice(): GpuDevice | null {
    return this.device;
  }

  /**
   * List all compiled shader pipelines.
   */
  listPipelines(): GpuPipelineInfo[] {
    return [...this.pipelines.keys()].map((name) => ({
      name: name as ShaderName,
      workgroupSize: WORKGROUP_SIZE,
    }));
  }

  /**
   * List all available built-in shader names.
   */
  listShaders(): ShaderName[] {
    return Object.keys(WGSL_SHADERS) as ShaderName[];
  }

  /**
   * Destroy the pooled buffers and the GPU device. Tensors still holding
   * device buffers become unusable. Safe to call more than once; a disposed
   * backend cannot be initialized again.
   */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const list of this.pool.values()) {
      for (const buffer of list) buffer.destroy();
    }
    this.pool.clear();
    for (const list of this.uniformPool.values()) {
      for (const buffer of list) buffer.destroy();
    }
    this.uniformPool.clear();
    this.pooledBytes = 0;
    this.pipelines.clear();
    this.device?.destroy();
    this.device = null;
  }

  /** Whether {@link WebGpuBackend.dispose} has been called. */
  get isDisposed(): boolean {
    return this.disposed;
  }
}
