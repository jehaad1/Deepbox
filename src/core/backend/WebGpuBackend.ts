/**
 * WebGPU Backend — GPU execution backend for Deepbox tensors.
 *
 * Implements the {@link KernelBackend} contract with WGSL compute kernels:
 * stride/broadcast-aware element-wise ops, tiled matrix multiplication and
 * multi-pass reductions over float32 device buffers. Once registered
 * (`registerBackend('webgpu', backend)` after `await backend.init()`),
 * built-in ndarray ops on `webgpu` tensors dispatch here automatically —
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
  bfloat16BitsToFloat64,
  float16BitsToFloat64,
  float64ToBFloat16Bits,
  float64ToFloat16Bits,
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

/** Maximum tensor rank supported by the WGSL kernels. */
const MAX_RANK = 8;
const WORKGROUP_SIZE = 256;
/** Bytes kept alive in the free-buffer pool before excess buffers are destroyed. */
const POOL_CAP_BYTES = 256 * 1024 * 1024;

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

/** Expressions for each binary op; `x`/`y` are the operand values. */
const BINARY_EXPR: Record<BinaryKernelOp, string> = {
  add: "x + y",
  sub: "x - y",
  mul: "x * y",
  div: "x / y",
  pow: "pow_impl(x, y)",
  maximum: "select(max(x, y), x + y, x != x || y != y)",
  minimum: "select(min(x, y), x + y, x != x || y != y)",
};

/** Expressions for each unary op; `x` is the operand value. */
const UNARY_EXPR: Record<UnaryKernelOp, string> = {
  copy: "x",
  step: "select(0.0, 1.0, x > 0.0)",
  neg: "-x",
  abs: "abs(x)",
  exp: "exp(x)",
  log: "log(x)",
  sqrt: "sqrt(x)",
  square: "x * x",
  relu: "max(x, 0.0)",
  sigmoid: "1.0 / (1.0 + exp(-x))",
  tanh: "tanh(x)",
  // Tanh-approximation GELU, matching the CPU `gelu` op exactly:
  // 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))). sqrt(2/pi)=0.7978845608028654.
  gelu: "0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))",
  erf: "erf_impl(x)",
  rsqrt: "inverseSqrt(x)",
  reciprocal: "1.0 / x",
  sign: "select(sign(x), x, x != x)",
  expm1: "exp(x) - 1.0",
  log1p: "log(1.0 + x)",
  // Numerically stable softplus: for large x, log(1+e^x) -> x.
  softplus: "select(log(1.0 + exp(x)), x, x > 20.0)",
};

/**
 * Abramowitz & Stegun 7.1.26 rational approximation of the error function
 * (max abs error ~1.5e-7, well within float32 precision). erf is odd, so the
 * magnitude is computed on |x| and the sign reapplied.
 */
const ERF_IMPL = /* wgsl */ `
fn erf_impl(x: f32) -> f32 {
  let t = 1.0 / (1.0 + 0.3275911 * abs(x));
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-x * x);
  return select(y, -y, x < 0.0);
}
`;

/** Unary ops that need a helper function injected into their shader. */
const UNARY_HELPERS: Partial<Record<UnaryKernelOp, string>> = {
  erf: ERF_IMPL,
};

/**
 * IEEE-style pow handling negative bases with integral exponents, which
 * WGSL's exp2/log2-based `pow` leaves undefined.
 */
const POW_IMPL = /* wgsl */ `
fn pow_impl(x: f32, y: f32) -> f32 {
  if (x >= 0.0) {
    return pow(x, y);
  }
  if (y == floor(y)) {
    let mag = pow(-x, y);
    let odd = (i32(y) & 1) != 0;
    return select(mag, -mag, odd);
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
${op === "pow" ? POW_IMPL : ""}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
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
${UNARY_HELPERS[op] ?? ""}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
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

/** Reduction combine functions (NaN-propagating for min/max, like NumPy). */
const REDUCE_COMBINE: Record<"sum" | "max" | "min", string> = {
  sum: "return x + y;",
  max: "if (x != x) { return x; } if (y != y) { return y; } return max(x, y);",
  min: "if (x != x) { return x; } if (y != y) { return y; } return min(x, y);",
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
${UNARY_META}
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> um: Meta;

var<workgroup> partials: array<f32, ${WORKGROUP_SIZE}>;
${INF_HELPERS}
fn combine(x: f32, y: f32) -> f32 {
  ${REDUCE_COMBINE[op]}
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let i = gid.x;
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

  if (lid.x == 0u) {
    out[wid.x] = partials[0];
  }
}`;
}

/**
 * Reduction along one axis. Each thread owns one output element and reduces
 * the `axisDim` input elements along the reduced axis. `scale` is `1/axisDim`
 * for mean and `1.0` otherwise, applied once at the end. `outToInStride[j]` is
 * the input stride of the input dimension that output dimension `j` maps to.
 */
function axisReduceShader(op: "sum" | "max" | "min"): string {
  return /* wgsl */ `
struct Meta {
  outSize: u32,
  axisDim: u32,
  outNdim: u32,
  inOffset: u32,
  axisStride: u32,
  scale: f32,
  _pad0: u32,
  _pad1: u32,
  outShape: array<vec4<u32>, 2>,
  outToInStride: array<vec4<u32>, 2>,
};
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> um: Meta;
${INF_HELPERS}
fn combine(x: f32, y: f32) -> f32 {
  ${REDUCE_COMBINE[op]}
}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
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
  out[i] = acc * um.scale;
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
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
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
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
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
 * every sliding window that covered this pixel (gather form — no atomics).
 * Input columns are contiguous `[batch, outH*outW, channels*kH*kW]`.
 */
const COL2IM_SHADER = /* wgsl */ `
${CONV_META}
@group(0) @binding(0) var<storage, read> cols: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> um: Meta;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
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
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> um: Meta;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
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
          ? "if (v != v) { acc = v; } else if (acc == acc) { acc = max(acc, v); }"
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
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= um.size) { return; }
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
      var best: f32 = grad_out[0] * 0.0; // placeholder, set on first valid tap
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
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < um.size) {
    out[gid.x] = um.value;
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
// accuracy — only the storage is half-precision, which is where the
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
${op === "pow" ? POW_IMPL : ""}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
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
${UNARY_HELPERS[op] ?? ""}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
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
${UNARY_META}
@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read_write> out: array<f16>;
@group(0) @binding(2) var<uniform> um: Meta;

var<workgroup> partials: array<f32, ${WORKGROUP_SIZE}>;
${INF_HELPERS}
fn combine(x: f32, y: f32) -> f32 {
  ${REDUCE_COMBINE[op]}
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let i = gid.x;
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

  if (lid.x == 0u) {
    out[wid.x] = f16(partials[0]);
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
  scale: f32,
  _pad0: u32,
  _pad1: u32,
  outShape: array<vec4<u32>, 2>,
  outToInStride: array<vec4<u32>, 2>,
};
@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read_write> out: array<f16>;
@group(0) @binding(2) var<uniform> um: Meta;
${INF_HELPERS}
fn combine(x: f32, y: f32) -> f32 {
  ${REDUCE_COMBINE[op]}
}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
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
  out[i] = f16(acc * um.scale);
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
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < um.size) {
    out[gid.x] = f16(um.value);
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
 * so it — like float32 — is 4 bytes.
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
  gpuBuffer: GPUBuffer;
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
 * import { tensor } from 'deepbox/ndarray';
 *
 * const gpu = new WebGpuBackend();
 * await gpu.init();
 * if (gpu.info().available) {
 *   registerBackend('webgpu', gpu);
 *   const a = tensor([[1, 2], [3, 4]], { device: 'webgpu' });
 *   const b = matmul(a, a);        // executes on the GPU
 *   const host = await b.cpu();    // read the result back
 * }
 * ```
 */
export class WebGpuBackend implements KernelBackend {
  private adapter: GPUAdapter | null = null;
  private device: GPUDevice | null = null;
  private pipelines = new Map<string, GPUComputePipeline>();
  private pool = new Map<number, GPUBuffer[]>();
  private uniformPool = new Map<number, GPUBuffer[]>();
  private pooledBytes = 0;
  private disposed = false;
  private initPromise: Promise<void> | null = null;
  private readonly gpuProvider: GPU | null;
  /** Whether the requested device enabled the `shader-f16` feature. */
  private f16Supported = false;

  private static readonly CAPABILITIES: readonly BackendCapability[] = [
    "matmul",
    "reduction",
    "elementwise",
  ];

  /**
   * @param options.gpu - Explicit WebGPU entry point for runtimes without
   *   `navigator.gpu` (e.g. Node.js with a Dawn binding).
   */
  constructor(options: { readonly gpu?: GPU } = {}) {
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
   * `available: false`.
   */
  async init(): Promise<void> {
    if (this.initPromise) return this.initPromise;
    this.initPromise = this.doInit();
    return this.initPromise;
  }

  private async doInit(): Promise<void> {
    const gpu = this.resolveGpu();
    if (!gpu) return;

    try {
      this.adapter = await gpu.requestAdapter();
      if (!this.adapter) return;
      // Enable true on-device half precision when the adapter advertises it.
      // Guarded so backends on GPUs without the feature still initialize (f16
      // tensors then throw a clear DeviceError at upload time).
      const wantsF16 = this.adapter.features.has("shader-f16");
      this.device = await this.adapter.requestDevice(
        wantsF16 ? { requiredFeatures: ["shader-f16"] } : undefined
      );
      this.f16Supported = wantsF16;
    } catch {
      this.adapter = null;
      this.device = null;
      this.f16Supported = false;
    }
  }

  /** Whether this backend can execute true on-device float16 kernels. */
  supportsF16(): boolean {
    return this.f16Supported && this.device !== null && !this.disposed;
  }

  /** Ensure the requested dtype can be created on this device. */
  private assertDTypeSupported(dtype: DeviceDType, op: string): void {
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
   * dtypes (no silent upcast — the caller must cast explicitly).
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

  private resolveGpu(): GPU | null {
    if (this.gpuProvider) return this.gpuProvider;
    if (typeof navigator !== "undefined" && "gpu" in navigator) {
      const gpu = (navigator as unknown as { gpu?: GPU }).gpu;
      if (gpu && typeof gpu === "object") return gpu;
    }
    return null;
  }

  private requireDevice(op: string): GPUDevice {
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
        rounded[i] = bfloat16BitsToFloat64(float64ToBFloat16Bits(data[i] ?? 0));
      }
      payload = rounded.buffer as ArrayBuffer;
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
    const dataBytes = buffer.size * deviceBytesPerElement(dtype);
    // Copy/staging sizes must be multiples of 4.
    const byteLength = Math.ceil(dataBytes / 4) * 4;

    const staging = device.createBuffer({
      size: byteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(gpuBuf, 0, staging, 0, byteLength);
    device.queue.submit([encoder.finish()]);

    await staging.mapAsync(GPUMapMode.READ);
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
        result[i] = bfloat16BitsToFloat64(float64ToBFloat16Bits(vals[i] ?? 0));
      }
    } else {
      result = new Float32Array(mapped.slice(0, buffer.size * 4));
    }
    staging.unmap();
    staging.destroy();
    return result;
  }

  free(buffer: DeviceBuffer): void {
    if (!isGpuDeviceBuffer(buffer) || buffer.freed || this.disposed) return;
    buffer.freed = true;
    this.releaseBuffer(buffer.gpuBuffer);
  }

  fill(value: number, size: number, dtype: DeviceDType = "float32"): DeviceBuffer {
    const device = this.requireDevice("fill");
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
    validateRank(layout, op);
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
    const dtype = this.resolveDType("matmul", [a, b]);
    const m = aLayout.shape[0] ?? 0;
    const k = aLayout.shape[1] ?? 0;
    const n = bLayout.shape[1] ?? 0;
    const aBuf = this.unwrap(a, "matmul");
    const bBuf = this.unwrap(b, "matmul");
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
    validateRank(layout, op);
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

    // Pass 1: strided load + workgroup-tree reduce.
    let currentSize = size;
    let current = this.reducePass(
      shaderName,
      this.unwrap(x, op),
      this.unaryMeta(size, layout),
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
        this.unaryMeta(currentSize, layout1d),
        currentSize,
        dtype
      );
      this.releaseBuffer(current);
      current = next;
      currentSize = Math.ceil(currentSize / WORKGROUP_SIZE);
    }

    if (op === "mean") {
      const result = this.wrap(current, 1, dtype);
      const divisor = this.fill(size, 1, dtype);
      const scalarLayout: KernelLayout = { shape: [], strides: [], offset: 0 };
      const meanBuf = this.binary("div", result, scalarLayout, divisor, scalarLayout, []);
      this.free(result);
      this.free(divisor);
      return meanBuf;
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
    validateRank(layout, op);
    const ndim = layout.shape.length;
    if (axis < 0 || axis >= ndim) {
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

    const out = this.acquireBuffer(device, Math.max(outSize, 1), deviceBytesPerElement(dtype));
    // Meta: outSize,axisDim,outNdim,inOffset,axisStride,scale,pad,pad, outShape[8], outToInStride[8]
    const meta = new ArrayBuffer(32 + 2 * 32);
    const u32 = new Uint32Array(meta);
    const f32 = new Float32Array(meta);
    u32[0] = outSize;
    u32[1] = axisDim;
    u32[2] = outShape.length;
    u32[3] = layout.offset;
    u32[4] = layout.strides[axis] ?? 0;
    f32[5] = op === "mean" ? 1 / axisDim : 1;
    u32.set(packDims(outShape), 8);
    u32.set(packDims(outToInStride), 16);

    this.dispatch(
      shaderName,
      [this.unwrap(x, op), out],
      meta,
      Math.ceil(Math.max(outSize, 1) / WORKGROUP_SIZE)
    );
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
    validateRank(aLayout, "matmulBatched");
    validateRank(bLayout, "matmulBatched");
    const batchNdim = aLayout.shape.length - 2;
    const batchShape = aLayout.shape.slice(0, batchNdim);
    // Inner 2-D strides are the last two of each operand.
    const aS0 = aLayout.strides[batchNdim] ?? 0;
    const aS1 = aLayout.strides[batchNdim + 1] ?? 0;
    const bS0 = bLayout.strides[batchNdim] ?? 0;
    const bS1 = bLayout.strides[batchNdim + 1] ?? 0;
    const aBatchStrides = aLayout.strides.slice(0, batchNdim);
    const bBatchStrides = bLayout.strides.slice(0, batchNdim);
    const dtype = this.resolveDType("matmulBatched", [a, b]);

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
      [this.unwrap(a, "matmulBatched"), this.unwrap(b, "matmulBatched"), out],
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
    this.assertNotF16(op, [cond, a, b]);
    const size = sizeOf(outShape);
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

    this.dispatch(
      op,
      [this.unwrap(cond, op), this.unwrap(a, op), this.unwrap(b, op), out],
      meta,
      Math.ceil(size / WORKGROUP_SIZE)
    );
    return this.wrap(out, size);
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
    const colSize = params.channels * params.kH * params.kW;
    const size = params.batch * params.outH * params.outW * colSize;
    const out = this.acquireBuffer(device, Math.max(size, 1));
    this.dispatch(
      "im2col",
      [this.unwrap(x, "im2col"), out],
      this.convMeta(params, layout, size),
      Math.ceil(Math.max(size, 1) / WORKGROUP_SIZE)
    );
    return this.wrap(out, size);
  }

  col2im(cols: DeviceBuffer, params: Im2ColParams): DeviceBuffer {
    const device = this.requireDevice("col2im");
    this.assertNotF16("col2im", [cols]);
    const size = params.batch * params.channels * params.height * params.width;
    const out = this.acquireBuffer(device, Math.max(size, 1));
    // col2im reads contiguous columns; layout offset/strides are unused by the
    // shader (it decodes the contiguous NCHW output), so pass a zero layout.
    const zeroLayout: KernelLayout = { shape: [], strides: [], offset: 0 };
    this.dispatch(
      "col2im",
      [this.unwrap(cols, "col2im"), out],
      this.convMeta(params, zeroLayout, size),
      Math.ceil(Math.max(size, 1) / WORKGROUP_SIZE)
    );
    return this.wrap(out, size);
  }

  pool2d(
    x: DeviceBuffer,
    layout: KernelLayout,
    op: PoolKernelOp,
    params: Im2ColParams
  ): DeviceBuffer {
    const device = this.requireDevice("pool2d");
    this.assertNotF16("pool2d", [x]);
    const size = params.batch * params.channels * params.outH * params.outW;
    const out = this.acquireBuffer(device, Math.max(size, 1));
    this.dispatch(
      op === "max" ? "pool2dMax" : "pool2dAvg",
      [this.unwrap(x, "pool2d"), out],
      this.convMeta(params, layout, size),
      Math.ceil(Math.max(size, 1) / WORKGROUP_SIZE)
    );
    return this.wrap(out, size);
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
    const size = params.batch * params.channels * params.height * params.width;
    const out = this.acquireBuffer(device, Math.max(size, 1));
    this.dispatch(
      op === "max" ? "pool2dBackwardMax" : "pool2dBackwardAvg",
      [this.unwrap(x, "pool2dBackward"), this.unwrap(gradOut, "pool2dBackward"), out],
      this.convMeta(params, xLayout, size),
      Math.ceil(Math.max(size, 1) / WORKGROUP_SIZE)
    );
    return this.wrap(out, size);
  }

  private reducePass(
    shaderName: string,
    input: GPUBuffer,
    meta: ArrayBuffer,
    inputSize: number,
    dtype: DeviceDType = "float32"
  ): GPUBuffer {
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

  // ─── Pipeline + dispatch plumbing ─────────────────────────────────────────

  /**
   * Get or create a compute pipeline for the given shader.
   *
   * @param name - Shader name from WGSL_SHADERS
   * @returns The compiled compute pipeline, or null if unavailable
   */
  getPipeline(name: ShaderName): GPUComputePipeline | null {
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

  private dispatch(
    name: string,
    buffers: GPUBuffer[],
    meta: ArrayBuffer,
    workgroups: number
  ): void {
    this.dispatchGrid(name, buffers, meta, Math.max(1, workgroups), 1);
  }

  private dispatchGrid(
    name: string,
    buffers: GPUBuffer[],
    meta: ArrayBuffer,
    groupsX: number,
    groupsY: number,
    groupsZ = 1
  ): void {
    const device = this.requireDevice(name);
    const pipeline = this.getPipeline(name);
    if (!pipeline) {
      throw new DeviceError(`webgpu: no kernel named "${name}"`);
    }

    // Reuse a pooled uniform buffer instead of allocating and destroying one
    // per dispatch. WebGPU queue operations are serialized on a single
    // timeline: this dispatch's `writeBuffer` + `submit` complete before the
    // next reuse of the same buffer writes new data, so a returned buffer is
    // safe to hand out again immediately — no per-op allocation churn.
    const metaBuffer = this.acquireUniform(device, meta.byteLength);
    device.queue.writeBuffer(metaBuffer, 0, meta);

    const entries: GPUBindGroupEntry[] = buffers.map((buffer, i) => ({
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
  private acquireUniform(device: GPUDevice, byteLength: number): GPUBuffer {
    // Uniform buffers must be 16-byte aligned; our metas already are.
    const size = Math.max(16, byteLength);
    const list = this.uniformPool.get(size);
    const pooled = list?.pop();
    if (pooled) return pooled;
    return device.createBuffer({
      size,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  /** Return a uniform buffer to the pool for reuse. */
  private releaseUniform(size: number, buffer: GPUBuffer): void {
    if (this.disposed) {
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
    // colliding in the pool.
    const bytes = Math.max(256, byteLength);
    return 2 ** Math.ceil(Math.log2(bytes));
  }

  private acquireBuffer(device: GPUDevice, elements: number, bytesPerElement = 4): GPUBuffer {
    const bucket = this.bucketFor(elements * bytesPerElement);
    const list = this.pool.get(bucket);
    const pooled = list?.pop();
    if (pooled) {
      this.pooledBytes -= bucket;
      return pooled;
    }
    return device.createBuffer({
      size: bucket,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
  }

  private releaseBuffer(buffer: GPUBuffer): void {
    const bucket = buffer.size;
    if (this.disposed || this.pooledBytes + bucket > POOL_CAP_BYTES) {
      buffer.destroy();
      return;
    }
    const list = this.pool.get(bucket) ?? [];
    list.push(buffer);
    this.pool.set(bucket, list);
    this.pooledBytes += bucket;
  }

  private wrap(
    buffer: GPUBuffer,
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

  private unwrap(buffer: DeviceBuffer, op: string): GPUBuffer {
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
   * @param data - Source data
   * @param usage - Buffer usage flags
   * @returns GPU buffer, or null if device unavailable
   */
  createBuffer(data: Float32Array, usage: GPUBufferUsageFlags): GPUBuffer | null {
    if (!this.device) return null;
    const buffer = this.device.createBuffer({
      size: data.byteLength,
      usage: usage | GPUBufferUsage.COPY_SRC,
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
   * @param size - Size in bytes to read
   * @returns Float32Array with the data
   */
  async readBuffer(buffer: GPUBuffer, size: number): Promise<Float32Array> {
    if (!this.device) return new Float32Array(0);
    const staging = this.device.createBuffer({
      size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
    this.device.queue.submit([encoder.finish()]);

    await staging.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return result;
  }

  /**
   * Get the underlying GPU device (for advanced usage).
   */
  getDevice(): GPUDevice | null {
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
    this.adapter = null;
  }

  get isDisposed(): boolean {
    return this.disposed;
  }
}
