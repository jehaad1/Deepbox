/**
 * Device dispatch machinery tests using an in-process fake kernel backend.
 *
 * The fake implements the full KernelBackend contract over Float32Arrays,
 * so every layer above the kernels, buffer-backed tensor storage, view
 * machinery, op dispatch, transfers, autograd, Module.to, is exercised
 * deterministically without GPU hardware. WGSL kernel correctness itself is
 * covered by test/webgpu-backend.test.ts (skipped when no GPU is present).
 */

import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type {
  Backend,
  BackendCapability,
  BackendInfo,
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
} from "../src/core";
import { DeviceError, DTypeError, registerBackend, resetConfig, setDevice } from "../src/core";
import {
  abs,
  add,
  div,
  dot,
  exp,
  expm1,
  GradTensor,
  gelu,
  log1p,
  max,
  mean,
  min,
  mul,
  neg,
  parameter,
  pow,
  relu,
  sub,
  sum,
  tensor,
  transpose,
  where,
  zeros,
} from "../src/ndarray";
import { softmax as gradSoftmax } from "../src/ndarray/autograd/index";
import {
  bfloat16BitsToFloat64,
  float16BitsToFloat64,
  float64ToBFloat16Bits,
  float64ToFloat16Bits,
} from "../src/ndarray/tensor/float16";
import { Conv2d, Linear, MaxPool2d } from "../src/nn";
import { Adam, SGD } from "../src/optim";

type FakeBuffer = DeviceBuffer & { data: Float32Array; freed: boolean; dtype?: DeviceDType };

/**
 * Round a float32 value through the requested half-precision format and back,
 * so the fake backend's numerics match the real GPU's half storage exactly.
 */
function roundToDType(value: number, dtype: DeviceDType | undefined): number {
  if (dtype === "float16") return float16BitsToFloat64(float64ToFloat16Bits(value));
  if (dtype === "bfloat16") return bfloat16BitsToFloat64(float64ToBFloat16Bits(value));
  return value;
}

const BINARY_FNS: Record<BinaryKernelOp, (x: number, y: number) => number> = {
  add: (x, y) => x + y,
  sub: (x, y) => x - y,
  mul: (x, y) => x * y,
  div: (x, y) => x / y,
  // Math.pow follows the IEEE special cases that the device pow kernel reproduces
  // (y == 0 gives 1, a NaN exponent gives NaN, signed zero and negative bases).
  pow: (x, y) => x ** y,
  maximum: Math.max,
  minimum: Math.min,
};

/**
 * Accurate erf in double precision: the Maclaurin series for |x| < 3 and the
 * continued fraction of the complementary error function elsewhere. The real
 * WGSL kernel is accurate to float32 rounding, so the fake must not use a
 * low-accuracy fit (the Abramowitz and Stegun 7.1.26 form is only good to 1e-7
 * absolute and has a large relative error for small x).
 */
const erfRef = (x: number): number => {
  if (Number.isNaN(x)) return NaN;
  const ax = Math.abs(x);
  if (ax >= 6) return x < 0 ? -1 : 1;
  let result: number;
  if (ax < 3) {
    let term = ax;
    let sum = ax;
    for (let n = 1; n < 200; n++) {
      term *= (-ax * ax) / n;
      const add = term / (2 * n + 1);
      sum += add;
      if (Math.abs(add) < 1e-18 * Math.abs(sum)) break;
    }
    result = (2 / Math.sqrt(Math.PI)) * sum;
  } else {
    // erfc(x) = exp(-x^2) / sqrt(pi) / (x + (1/2) / (x + 1 / (x + (3/2) / (x + ...))))
    let tail = 0;
    for (let k = 60; k >= 1; k--) tail = k / 2 / (ax + tail);
    result = 1 - Math.exp(-ax * ax) / Math.sqrt(Math.PI) / (ax + tail);
  }
  return x < 0 ? -result : result;
};

const UNARY_FNS: Record<UnaryKernelOp, (x: number) => number> = {
  copy: (x) => x,
  step: (x) => (x > 0 ? 1 : 0),
  neg: (x) => -x,
  abs: Math.abs,
  exp: Math.exp,
  log: Math.log,
  sqrt: Math.sqrt,
  square: (x) => x * x,
  relu: (x) => Math.max(x, 0),
  sigmoid: (x) => 1 / (1 + Math.exp(-x)),
  tanh: Math.tanh,
  gelu: (x) => 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))),
  erf: erfRef,
  rsqrt: (x) => 1 / Math.sqrt(x),
  reciprocal: (x) => 1 / x,
  sign: (x) => Math.sign(x),
  // The device kernels use accurate small-argument forms (no exp(x) - 1 or
  // log(1 + x) cancellation) and the stable softplus max(x, 0) + log1p(exp(-|x|)).
  expm1: Math.expm1,
  log1p: Math.log1p,
  softplus: (x) => Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x))),
};

/** Reference KernelBackend over host Float32Arrays. */
class FakeKernelBackend implements KernelBackend {
  allocated = 0;
  freedCount = 0;

  info(): BackendInfo {
    return {
      device: "webgpu",
      name: "Fake Kernel Backend",
      available: true,
      capabilities: ["matmul", "reduction", "elementwise"],
    };
  }

  supports(cap: BackendCapability): boolean {
    return cap === "matmul" || cap === "reduction" || cap === "elementwise";
  }

  async init(): Promise<void> {}
  dispose(): void {}

  private wrap(data: Float32Array, dtype: DeviceDType = "float32"): FakeBuffer {
    this.allocated++;
    // Round to the half format so results match the real GPU's half storage.
    if (dtype !== "float32") {
      for (let i = 0; i < data.length; i++) data[i] = roundToDType(data[i] ?? 0, dtype);
    }
    const bytesPerEl = dtype === "float16" ? 2 : 4;
    return {
      device: "webgpu",
      byteLength: data.length * bytesPerEl,
      size: data.length,
      dtype,
      data,
      freed: false,
    };
  }

  /** Shared dtype of operands; throws on a mismatch (no silent upcast). */
  private dtypeOf(op: string, ...buffers: DeviceBuffer[]): DeviceDType {
    const dtype = (buffers[0]?.dtype ?? "float32") as DeviceDType;
    for (const b of buffers) {
      if (((b.dtype ?? "float32") as DeviceDType) !== dtype) {
        throw new DeviceError(`fake ${op}: mismatched operand dtypes`);
      }
    }
    return dtype;
  }

  private read(buffer: DeviceBuffer, layout: KernelLayout): Float32Array {
    const buf = buffer as FakeBuffer;
    if (buf.freed) throw new DeviceError("fake: buffer already freed");
    const size = layout.shape.reduce((a, b) => a * b, 1);
    const out = new Float32Array(size);
    const ndim = layout.shape.length;
    for (let i = 0; i < size; i++) {
      let rem = i;
      let idx = layout.offset;
      for (let k = ndim - 1; k >= 0; k--) {
        const dim = layout.shape[k] ?? 1;
        idx += (rem % dim) * (layout.strides[k] ?? 0);
        rem = Math.floor(rem / dim);
      }
      out[i] = buf.data[idx] ?? 0;
    }
    return out;
  }

  upload(data: Float32Array, dtype: DeviceDType = "float32"): DeviceBuffer {
    return this.wrap(data.slice(), dtype);
  }

  async download(buffer: DeviceBuffer): Promise<Float32Array> {
    const buf = buffer as FakeBuffer;
    if (buf.freed) throw new DeviceError("fake: buffer already freed");
    return buf.data.slice();
  }

  free(buffer: DeviceBuffer): void {
    const buf = buffer as FakeBuffer;
    if (!buf.freed) {
      buf.freed = true;
      this.freedCount++;
    }
  }

  fill(value: number, size: number, dtype: DeviceDType = "float32"): DeviceBuffer {
    return this.wrap(new Float32Array(Math.max(size, 1)).fill(Math.fround(value)), dtype);
  }

  binary(
    op: BinaryKernelOp,
    a: DeviceBuffer,
    aLayout: KernelLayout,
    b: DeviceBuffer,
    bLayout: KernelLayout,
    outShape: readonly number[]
  ): DeviceBuffer {
    const dtype = this.dtypeOf(op, a, b);
    const av = this.read(a, { ...aLayout, shape: outShape });
    const bv = this.read(b, { ...bLayout, shape: outShape });
    const fn = BINARY_FNS[op];
    const out = new Float32Array(av.length);
    for (let i = 0; i < out.length; i++) out[i] = Math.fround(fn(av[i] ?? 0, bv[i] ?? 0));
    return this.wrap(out, dtype);
  }

  unary(op: UnaryKernelOp, x: DeviceBuffer, layout: KernelLayout): DeviceBuffer {
    const xv = this.read(x, layout);
    const fn = UNARY_FNS[op];
    const out = new Float32Array(xv.length);
    for (let i = 0; i < out.length; i++) out[i] = Math.fround(fn(xv[i] ?? 0));
    return this.wrap(out, (x.dtype ?? "float32") as DeviceDType);
  }

  matmul(
    a: DeviceBuffer,
    aLayout: KernelLayout,
    b: DeviceBuffer,
    bLayout: KernelLayout
  ): DeviceBuffer {
    const dtype = this.dtypeOf("matmul", a, b);
    const m = aLayout.shape[0] ?? 0;
    const k = aLayout.shape[1] ?? 0;
    const n = bLayout.shape[1] ?? 0;
    const av = this.read(a, aLayout);
    const bv = this.read(b, bLayout);
    const out = new Float32Array(m * n);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let acc = 0;
        for (let p = 0; p < k; p++) {
          acc = Math.fround(acc + Math.fround((av[i * k + p] ?? 0) * (bv[p * n + j] ?? 0)));
        }
        out[i * n + j] = acc;
      }
    }
    return this.wrap(out, dtype);
  }

  reduce(op: ReduceKernelOp, x: DeviceBuffer, layout: KernelLayout): DeviceBuffer {
    const xv = this.read(x, layout);
    let acc: number;
    if (op === "max") {
      acc = Number.NEGATIVE_INFINITY;
      for (const v of xv) acc = Number.isNaN(v) || Number.isNaN(acc) ? NaN : Math.max(acc, v);
    } else if (op === "min") {
      acc = Number.POSITIVE_INFINITY;
      for (const v of xv) acc = Number.isNaN(v) || Number.isNaN(acc) ? NaN : Math.min(acc, v);
    } else {
      acc = 0;
      for (const v of xv) acc = Math.fround(acc + v);
      if (op === "mean") acc = Math.fround(acc / xv.length);
    }
    return this.wrap(new Float32Array([Math.fround(acc)]), (x.dtype ?? "float32") as DeviceDType);
  }

  reduceAxis(
    op: ReduceKernelOp,
    x: DeviceBuffer,
    layout: KernelLayout,
    axis: number
  ): DeviceBuffer {
    const xv = this.read(x, layout); // gathered contiguous over layout.shape
    const shape = layout.shape;
    const axisDim = shape[axis] ?? 1;
    const outShape = shape.filter((_, d) => d !== axis);
    const outSize = outShape.reduce((a, b) => a * b, 1);
    const inStrides = contiguousStridesRef(shape);
    const outStrides = contiguousStridesRef(outShape);
    const out = new Float32Array(outSize);
    const combine = (a: number, b: number): number => {
      if (op === "max") return Number.isNaN(a) || Number.isNaN(b) ? NaN : Math.max(a, b);
      if (op === "min") return Number.isNaN(a) || Number.isNaN(b) ? NaN : Math.min(a, b);
      return a + b;
    };
    const identity = op === "max" ? -Infinity : op === "min" ? Infinity : 0;
    out.fill(identity);
    for (let i = 0; i < xv.length; i++) {
      let rem = i;
      const coord: number[] = [];
      for (let d = 0; d < shape.length; d++) {
        coord.push(Math.floor(rem / (inStrides[d] ?? 1)));
        rem -= (coord[d] ?? 0) * (inStrides[d] ?? 1);
      }
      let of = 0;
      let j = 0;
      for (let d = 0; d < shape.length; d++) {
        if (d === axis) continue;
        of += (coord[d] ?? 0) * (outStrides[j] ?? 1);
        j++;
      }
      out[of] = combine(out[of] ?? identity, xv[i] ?? 0);
    }
    if (op === "mean") for (let i = 0; i < out.length; i++) out[i] = (out[i] ?? 0) / axisDim;
    for (let i = 0; i < out.length; i++) out[i] = Math.fround(out[i] ?? 0);
    return this.wrap(out, (x.dtype ?? "float32") as DeviceDType);
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
    const av = this.read(a, aLayout);
    const bv = this.read(b, bLayout);
    // read() gathers each operand contiguously over its full [batch..., X, Y]
    // shape, so batch/row/col indexing into the gathered arrays is row-major.
    const out = new Float32Array(batch * m * n);
    for (let bt = 0; bt < batch; bt++) {
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
          let acc = 0;
          for (let p = 0; p < k; p++) {
            acc = Math.fround(
              acc +
                Math.fround((av[bt * m * k + i * k + p] ?? 0) * (bv[bt * k * n + p * n + j] ?? 0))
            );
          }
          out[bt * m * n + i * n + j] = acc;
        }
      }
    }
    return this.wrap(out);
  }

  ternary(
    _op: TernaryKernelOp,
    cond: DeviceBuffer,
    condLayout: KernelLayout,
    a: DeviceBuffer,
    aLayout: KernelLayout,
    b: DeviceBuffer,
    bLayout: KernelLayout,
    outShape: readonly number[]
  ): DeviceBuffer {
    const cv = this.read(cond, { ...condLayout, shape: outShape });
    const av = this.read(a, { ...aLayout, shape: outShape });
    const bv = this.read(b, { ...bLayout, shape: outShape });
    const out = new Float32Array(cv.length);
    for (let i = 0; i < out.length; i++) out[i] = (cv[i] ?? 0) !== 0 ? (av[i] ?? 0) : (bv[i] ?? 0);
    // Like the device kernel, the selected values keep their dtype.
    const aDtype = (a.dtype ?? "float32") as DeviceDType;
    const bDtype = (b.dtype ?? "float32") as DeviceDType;
    return this.wrap(out, aDtype === bDtype ? aDtype : "float32");
  }

  im2col(x: DeviceBuffer, layout: KernelLayout, p: Im2ColParams): DeviceBuffer {
    const img = this.read(x, layout); // gathered contiguous NCHW
    const is = contiguousStridesRef([p.batch, p.channels, p.height, p.width]);
    const colSize = p.channels * p.kH * p.kW;
    const out = new Float32Array(p.batch * p.outH * p.outW * colSize);
    for (let b = 0; b < p.batch; b++) {
      for (let oh = 0; oh < p.outH; oh++) {
        for (let ow = 0; ow < p.outW; ow++) {
          const pix = oh * p.outW + ow;
          let ci = 0;
          for (let c = 0; c < p.channels; c++) {
            for (let kh = 0; kh < p.kH; kh++) {
              for (let kw = 0; kw < p.kW; kw++) {
                const ih = oh * p.strideH + kh - p.padH;
                const iw = ow * p.strideW + kw - p.padW;
                let v = 0;
                if (ih >= 0 && ih < p.height && iw >= 0 && iw < p.width) {
                  v =
                    img[
                      b * (is[0] ?? 0) + c * (is[1] ?? 0) + ih * (is[2] ?? 0) + iw * (is[3] ?? 0)
                    ] ?? 0;
                }
                out[b * p.outH * p.outW * colSize + pix * colSize + ci] = v;
                ci++;
              }
            }
          }
        }
      }
    }
    return this.wrap(out);
  }

  col2im(cols: DeviceBuffer, p: Im2ColParams): DeviceBuffer {
    const col = (cols as FakeBuffer).data;
    const is = contiguousStridesRef([p.batch, p.channels, p.height, p.width]);
    const colSize = p.channels * p.kH * p.kW;
    const out = new Float32Array(p.batch * p.channels * p.height * p.width);
    for (let b = 0; b < p.batch; b++) {
      for (let oh = 0; oh < p.outH; oh++) {
        for (let ow = 0; ow < p.outW; ow++) {
          const pix = oh * p.outW + ow;
          let ci = 0;
          for (let c = 0; c < p.channels; c++) {
            for (let kh = 0; kh < p.kH; kh++) {
              for (let kw = 0; kw < p.kW; kw++) {
                const ih = oh * p.strideH + kh - p.padH;
                const iw = ow * p.strideW + kw - p.padW;
                if (ih >= 0 && ih < p.height && iw >= 0 && iw < p.width) {
                  const oi =
                    b * (is[0] ?? 0) + c * (is[1] ?? 0) + ih * (is[2] ?? 0) + iw * (is[3] ?? 0);
                  out[oi] =
                    (out[oi] ?? 0) + (col[b * p.outH * p.outW * colSize + pix * colSize + ci] ?? 0);
                }
                ci++;
              }
            }
          }
        }
      }
    }
    return this.wrap(out);
  }

  pool2d(x: DeviceBuffer, layout: KernelLayout, op: PoolKernelOp, p: Im2ColParams): DeviceBuffer {
    const img = this.read(x, layout);
    const is = contiguousStridesRef([p.batch, p.channels, p.height, p.width]);
    const out = new Float32Array(p.batch * p.channels * p.outH * p.outW);
    let oi = 0;
    for (let b = 0; b < p.batch; b++) {
      for (let c = 0; c < p.channels; c++) {
        for (let oh = 0; oh < p.outH; oh++) {
          for (let ow = 0; ow < p.outW; ow++) {
            let acc = op === "max" ? Number.NEGATIVE_INFINITY : 0;
            let count = 0;
            for (let kh = 0; kh < p.kH; kh++) {
              for (let kw = 0; kw < p.kW; kw++) {
                const ih = oh * p.strideH + kh - p.padH;
                const iw = ow * p.strideW + kw - p.padW;
                if (ih < 0 || ih >= p.height || iw < 0 || iw >= p.width) continue;
                const v =
                  img[
                    b * (is[0] ?? 0) + c * (is[1] ?? 0) + ih * (is[2] ?? 0) + iw * (is[3] ?? 0)
                  ] ?? 0;
                if (op === "max")
                  acc = Number.isNaN(v) ? v : Number.isNaN(acc) ? acc : Math.max(acc, v);
                else {
                  acc += v;
                  count++;
                }
              }
            }
            out[oi++] = op === "max" ? acc : count > 0 ? acc / count : 0;
          }
        }
      }
    }
    return this.wrap(out);
  }

  pool2dBackward(
    x: DeviceBuffer,
    xLayout: KernelLayout,
    gradOut: DeviceBuffer,
    op: PoolKernelOp,
    p: Im2ColParams
  ): DeviceBuffer {
    const img = this.read(x, xLayout);
    const go = (gradOut as FakeBuffer).data;
    const is = contiguousStridesRef([p.batch, p.channels, p.height, p.width]);
    const out = new Float32Array(p.batch * p.channels * p.height * p.width);
    const goPlane = p.outH * p.outW;
    for (let i = 0; i < out.length; i++) {
      const b = Math.floor(i / (p.channels * p.height * p.width));
      let rem = i % (p.channels * p.height * p.width);
      const c = Math.floor(rem / (p.height * p.width));
      rem = rem % (p.height * p.width);
      const ih = Math.floor(rem / p.width);
      const iw = rem % p.width;
      let acc = 0;
      for (let kh = 0; kh < p.kH; kh++) {
        const ohN = ih + p.padH - kh;
        if (ohN < 0 || ohN % p.strideH !== 0) continue;
        const oh = ohN / p.strideH;
        if (oh < 0 || oh >= p.outH) continue;
        for (let kw = 0; kw < p.kW; kw++) {
          const owN = iw + p.padW - kw;
          if (owN < 0 || owN % p.strideW !== 0) continue;
          const ow = owN / p.strideW;
          if (ow < 0 || ow >= p.outW) continue;
          const gv = go[(b * p.channels + c) * goPlane + oh * p.outW + ow] ?? 0;
          if (op === "avg") {
            let cnt = 0;
            for (let a = 0; a < p.kH; a++) {
              const jh = oh * p.strideH + a - p.padH;
              if (jh < 0 || jh >= p.height) continue;
              for (let bb = 0; bb < p.kW; bb++) {
                const jw = ow * p.strideW + bb - p.padW;
                if (jw < 0 || jw >= p.width) continue;
                cnt++;
              }
            }
            if (cnt > 0) acc += gv / cnt;
          } else {
            let best = 0;
            let bestLocal = -1;
            let seen = false;
            for (let a = 0; a < p.kH; a++) {
              const jh = oh * p.strideH + a - p.padH;
              if (jh < 0 || jh >= p.height) continue;
              for (let bb = 0; bb < p.kW; bb++) {
                const jw = ow * p.strideW + bb - p.padW;
                if (jw < 0 || jw >= p.width) continue;
                const v =
                  img[
                    b * (is[0] ?? 0) + c * (is[1] ?? 0) + jh * (is[2] ?? 0) + jw * (is[3] ?? 0)
                  ] ?? 0;
                if (!seen || v > best) {
                  best = v;
                  bestLocal = jh * p.width + jw;
                  seen = true;
                }
              }
            }
            if (bestLocal === ih * p.width + iw) acc += gv;
          }
        }
      }
      out[i] = acc;
    }
    return this.wrap(out);
  }
}

function contiguousStridesRef(shape: readonly number[]): number[] {
  const s = new Array<number>(shape.length).fill(1);
  for (let i = shape.length - 2; i >= 0; i--) s[i] = (s[i + 1] ?? 1) * (shape[i + 1] ?? 1);
  return s;
}

const fake = new FakeKernelBackend();
const cpuOnlyBackend: Backend = {
  info: () => ({ device: "wasm", name: "noop", available: true, capabilities: [] }),
  supports: () => false,
  init: async () => {},
  dispose: () => {},
};

beforeAll(() => {
  registerBackend("webgpu", fake);
});

afterAll(() => {
  resetConfig();
});

const dev = { device: "webgpu" as const };
const flat = (t: unknown): number[] => ([t].flat(Infinity) as number[]).flat();

describe("device tensor storage", () => {
  it("creates tensors on the device and blocks synchronous data access", () => {
    const t = tensor([1, 2, 3], dev);
    expect(t.device).toBe("webgpu");
    expect(t.isDeviceTensor).toBe(true);
    expect(t.deviceBuffer).not.toBeNull();
    expect(() => t.data).toThrow(DeviceError);
    expect(() => t.toArray()).toThrow(DeviceError);
    expect(() => t.at(0)).toThrow(DeviceError);
  });

  it("rejects non-float32 dtypes on kernel devices", () => {
    expect(() => tensor([1, 2, 3], { ...dev, dtype: "float64" })).toThrow(DTypeError);
    expect(() => tensor([1, 2, 3], { ...dev, dtype: "int32" })).toThrow(DTypeError);
  });

  it("carries the upload dtype onto the device buffer (bytes are dtype-aware)", () => {
    const f16 = tensor([1.5, 2.5, 3.5], { ...dev, dtype: "float16" });
    expect(f16.dtype).toBe("float16");
    expect(f16.deviceBuffer?.dtype).toBe("float16");
    expect(f16.deviceBuffer?.byteLength).toBe(6); // 3 elements × 2 bytes

    const bf16 = tensor([1, 2, 3], { ...dev, dtype: "bfloat16" });
    expect(bf16.deviceBuffer?.dtype).toBe("bfloat16");

    const f32 = tensor([1, 2, 3], dev);
    // Absent/float32 keeps the original 4-byte behavior.
    expect(f32.deviceBuffer?.byteLength).toBe(12);
  });

  it("round-trips half-precision dtype through cpu()", async () => {
    const f16 = tensor([1, 2, 3, 4], { ...dev, dtype: "float16" });
    const backF16 = await f16.cpu();
    expect(backF16.dtype).toBe("float16");
    expect(backF16.toArray()).toEqual([1, 2, 3, 4]);

    const bf16 = tensor([1, 2, 4], { ...dev, dtype: "bfloat16" });
    const backBf16 = await bf16.cpu();
    expect(backBf16.dtype).toBe("bfloat16");
    expect(backBf16.toArray()).toEqual([1, 2, 4]);
  });

  it("runs half-precision device ops and matches the float32 result", async () => {
    const a = tensor([1, 2, 3, 4], { ...dev, dtype: "float16" });
    const b = tensor([10, 20, 30, 40], { ...dev, dtype: "float16" });
    const summed = await add(a, b).cpu();
    expect(summed.dtype).toBe("float16");
    expect(summed.toArray()).toEqual([11, 22, 33, 44]);
    expect((await sum(a).cpu()).toArray()).toEqual(10);
  });

  it("rejects mixed-dtype device ops instead of silently upcasting", () => {
    const f16 = tensor([1, 2, 3], { ...dev, dtype: "float16" });
    const f32 = tensor([1, 2, 3], dev);
    // Mixed float16/float32 is rejected (no silent upcast), the arithmetic
    // dtype guard catches it before dispatch; the backend has its own guard too.
    expect(() => add(f16, f32)).toThrow(/dtype/i);
  });

  it("zeros/ones create device buffers", async () => {
    const z = zeros([2, 3], dev);
    expect(z.isDeviceTensor).toBe(true);
    expect((await z.cpu()).toArray()).toEqual([
      [0, 0, 0],
      [0, 0, 0],
    ]);
  });

  it("toString prints metadata instead of data", () => {
    const t = tensor([[1, 2]], dev);
    expect(t.toString()).toContain("<webgpu>");
    expect(t.toString()).toContain("shape=[1, 2]");
  });

  it("round-trips to('cpu') exactly, including strided views", async () => {
    const t = tensor(
      [
        [1.5, -2.25, 3],
        [4, 5.5, -6],
      ],
      dev
    );
    const back = await t.cpu();
    expect(back.toArray()).toEqual([
      [1.5, -2.25, 3],
      [4, 5.5, -6],
    ]);
    const view = transpose(t);
    const backView = await view.cpu();
    expect(backView.toArray()).toEqual([
      [1.5, 4],
      [-2.25, 5.5],
      [3, -6],
    ]);
  });

  it("to(same device) returns the same tensor", async () => {
    const t = tensor([1], dev);
    expect(await t.to("webgpu")).toBe(t);
  });

  it("dispose releases the buffer once, views share ownership", async () => {
    const t = tensor([1, 2, 3, 4], dev);
    const view = t.reshape([2, 2]);
    const freedBefore = fake.freedCount;
    t.dispose();
    t.dispose(); // idempotent
    expect(fake.freedCount).toBe(freedBefore); // view still holds the buffer
    view.dispose();
    expect(fake.freedCount).toBe(freedBefore + 1);
  });

  it("using a disposed tensor throws", async () => {
    const t = tensor([1, 2], dev);
    const u = tensor([3, 4], dev);
    t.dispose();
    expect(() => add(t, u)).toThrow(DeviceError);
  });
});

describe("device op dispatch", () => {
  const cpu = async (t: { cpu(): Promise<{ toArray(): unknown }> }) =>
    flat((await t.cpu()).toArray());

  it("binary arithmetic with broadcasting", async () => {
    const a = tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      dev
    );
    const row = tensor([[10, 20, 30]], dev);
    expect(await cpu(add(a, row))).toEqual([11, 22, 33, 14, 25, 36]);
    expect(await cpu(sub(a, row))).toEqual([-9, -18, -27, -6, -15, -24]);
    expect(await cpu(mul(a, a))).toEqual([1, 4, 9, 16, 25, 36]);
    expect(await cpu(div(a, a))).toEqual([1, 1, 1, 1, 1, 1]);
    expect(await cpu(pow(a, tensor([2], dev)))).toEqual([1, 4, 9, 16, 25, 36]);
  });

  it("matches the accurate device kernel numerics for small arguments and the softplus tail", async () => {
    const small = tensor([1e-6, -1e-6, 1e-3], dev);
    const e = await cpu(expm1(small));
    const l = await cpu(log1p(small));
    expect(e[0]).toBeCloseTo(1e-6, 12);
    expect(l[1]).toBeCloseTo(-1e-6, 12);
    expect(e[2]).toBeCloseTo(Math.expm1(1e-3), 7);
    // exp(x) - 1 in float32 would give 0 or 1.19e-7 for 1e-8; the kernel keeps full precision.
    const tiny = await cpu(expm1(tensor([1e-8], dev)));
    expect(tiny[0] ?? 0).toBeCloseTo(1e-8, 12);
  });

  it("evaluates the exact gelu through an accurate erf (scipy reference values)", async () => {
    const x = tensor([1e-3, -0.01, 0.5, 2], dev);
    const out = await cpu(gelu(x, "none"));
    const expected = [0.000500398942213911, -0.004960106436853684, 0.3457312306370065, 1.9544997];
    for (let i = 0; i < expected.length; i++) {
      expect(out[i] ?? Number.NaN).toBeCloseTo(expected[i] ?? 0, 7);
    }
    // A low-accuracy erf fit (error ~1e-7 near zero) is off by ~1e-4 relative here.
    expect(Math.abs((out[0] ?? 0) / 0.000500398942213911 - 1)).toBeLessThan(1e-5);
  });

  it("keeps the selected dtype from the device where kernel", async () => {
    const cond = tensor([1, 0, 1], dev);
    const a = tensor([1.5, 2.5, 3.5], { ...dev, dtype: "float16" });
    const b = tensor([10, 20, 30], { ...dev, dtype: "float16" });
    const out = where(cond, a, b);
    expect(out.dtype).toBe("float16");
    expect(out.deviceBuffer?.dtype).toBe("float16");
    expect(await cpu(out)).toEqual([1.5, 20, 3.5]);
  });

  it("promotes CPU scalar operands like PyTorch", async () => {
    const a = tensor([1, 2, 3], dev);
    expect(await cpu(mul(a, tensor(2)))).toEqual([2, 4, 6]);
    expect(await cpu(add(tensor(10), a))).toEqual([11, 12, 13]);
  });

  it("rejects mixed devices for non-scalar operands", () => {
    const a = tensor([1, 2, 3], dev);
    const c = tensor([1, 2, 3]);
    expect(() => add(a, c)).toThrow(/same device/);
    expect(() => dot(a, c)).toThrow(DeviceError);
  });

  it("unary ops", async () => {
    const t = tensor([-2, 0, 2], dev);
    expect(await cpu(neg(t))).toEqual([2, -0, -2]);
    expect(await cpu(abs(t))).toEqual([2, 0, 2]);
    expect(await cpu(relu(t))).toEqual([0, 0, 2]);
    expect(await cpu(exp(tensor([0], dev)))).toEqual([1]);
  });

  it("ops on views execute with strides (no materialization)", async () => {
    const a = tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      dev
    );
    const tr = transpose(a);
    expect(await cpu(add(tr, tr))).toEqual([2, 8, 4, 10, 6, 12]);
    const sl = a.slice({ start: 0, end: 2 }, { start: 1, end: 3 });
    expect(await cpu(mul(sl, sl))).toEqual([4, 9, 25, 36]);
  });

  it("negative-step slicing on device throws with a transfer hint", () => {
    const a = tensor([1, 2, 3, 4], dev);
    expect(() => a.slice({ start: 3, step: -1 })).toThrow(/cpu/);
  });

  it("reshape: contiguous is a view, non-contiguous copies on device", async () => {
    const a = tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      dev
    );
    expect(await cpu(a.reshape([3, 2]))).toEqual([1, 2, 3, 4, 5, 6]);
    expect(await cpu(transpose(a).reshape([6]))).toEqual([1, 4, 2, 5, 3, 6]);
  });

  it("dot: 1-D and 2-D combinations", async () => {
    const v = tensor([1, 2, 3], dev);
    const m = tensor(
      [
        [1, 0],
        [0, 1],
        [1, 1],
      ],
      dev
    );
    expect(await cpu(dot(v, v))).toEqual([14]);
    expect(await cpu(dot(v, m))).toEqual([4, 5]);
    expect(await cpu(dot(transpose(m), v))).toEqual([4, 5]);
    const sq = tensor(
      [
        [1, 2],
        [3, 4],
      ],
      dev
    );
    expect(await cpu(dot(sq, sq))).toEqual([7, 10, 15, 22]);
  });

  it("dot: batched (ndim>2) operands run on device and match the CPU result", async () => {
    // Two 2x2 matrices per batch: batch 0 = [[1,2],[3,4]], batch 1 = [[5,6],[7,8]].
    const a = tensor(
      [
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ],
      dev
    );
    const out = dot(a, a);
    expect(out.shape).toEqual([2, 2, 2]);
    // batch 0: [[7,10],[15,22]]; batch 1: [[67,78],[91,106]]
    expect(await cpu(out)).toEqual([7, 10, 15, 22, 67, 78, 91, 106]);

    // Mixed rank: [2,2,2] @ [2,2] broadcasts the 2-D operand across the batch.
    const w = tensor(
      [
        [1, 0],
        [0, 1],
      ],
      dev
    );
    const bc = dot(a, w);
    expect(bc.shape).toEqual([2, 2, 2]);
    expect(await cpu(bc)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
  });

  it("full reductions incl. keepdims", async () => {
    const a = tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      dev
    );
    expect(await cpu(sum(a))).toEqual([21]);
    expect(await cpu(mean(a))).toEqual([3.5]);
    expect(await cpu(max(a))).toEqual([6]);
    expect(await cpu(min(a))).toEqual([1]);
    const kd = sum(a, undefined, true);
    expect(kd.shape).toEqual([1, 1]);
    expect(await cpu(kd)).toEqual([21]);
  });

  it("axis reductions run on device and match the CPU result", async () => {
    const a = tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      dev
    );
    // sum over axis 0 -> [5, 7, 9]; axis 1 -> [6, 15]
    const s0 = sum(a, 0);
    expect(s0.shape).toEqual([3]);
    expect(await cpu(s0)).toEqual([5, 7, 9]);
    const s1 = sum(a, 1);
    expect(s1.shape).toEqual([2]);
    expect(await cpu(s1)).toEqual([6, 15]);
    // keepdims
    const s1kd = sum(a, 1, true);
    expect(s1kd.shape).toEqual([2, 1]);
    expect(await cpu(s1kd)).toEqual([6, 15]);
    // mean / max / min along an axis
    expect(await cpu(mean(a, 0))).toEqual([2.5, 3.5, 4.5]);
    expect(await cpu(max(a, 1))).toEqual([3, 6]);
    expect(await cpu(min(a, 0))).toEqual([1, 2, 3]);
  });

  it("unsupported ops fail loudly instead of computing silently", () => {
    const a = tensor([3, 1, 2], dev);
    // sort is not device-accelerated: the data guard must throw.
    expect(() => a.toArray()).toThrow(/not accelerated|Cannot access data/);
  });

  it("setDevice makes new tensors land on the device", async () => {
    setDevice("webgpu");
    try {
      const t = tensor([1, 2]);
      expect(t.device).toBe("webgpu");
      expect(t.isDeviceTensor).toBe(true);
    } finally {
      resetConfig();
    }
  });
});

describe("device autograd", () => {
  it("matmul + relu + mean backward matches the CPU gradients", async () => {
    const wData = [
      [0.5, -1, 2],
      [1.5, 0.25, -0.75],
    ];
    const xData = [
      [1, 2],
      [0.5, -1],
      [-2, 3],
    ];

    const run = async (onDevice: boolean) => {
      const wT = onDevice ? await tensor(wData).to("webgpu") : tensor(wData);
      const xT = onDevice ? await tensor(xData).to("webgpu") : tensor(xData);
      const w = parameter(wT);
      const x = GradTensor.fromTensor(xT, { requiresGrad: false });
      const loss = x.matmul(w).relu().mean();
      loss.backward();
      const g = w.grad;
      if (!g) throw new Error("missing grad");
      return flat((onDevice ? await g.cpu() : g).toArray());
    };

    const ref = await run(false);
    const got = await run(true);
    expect(got.length).toBe(ref.length);
    for (let i = 0; i < ref.length; i++) {
      expect(got[i]).toBeCloseTo(ref[i] ?? 0, 5);
    }
  });

  it("sum backward broadcasts the seed on the device", async () => {
    const w = parameter(
      await tensor([
        [1, 2],
        [3, 4],
      ]).to("webgpu")
    );
    const loss = w.mul(w).sum();
    loss.backward();
    const g = w.grad;
    expect(g).not.toBeNull();
    if (!g) return;
    expect(flat((await g.cpu()).toArray())).toEqual([2, 4, 6, 8]);
  });

  it("softmax forward + backward (axis-reduction backward) matches CPU on device", async () => {
    const xData = [
      [1, 2, 3],
      [0, -1, 2],
    ];
    const tgtData = [
      [0, 0, 1],
      [1, 0, 0],
    ];
    const run = async (onDevice: boolean) => {
      const x = parameter(onDevice ? await tensor(xData).to("webgpu") : tensor(xData));
      const target = GradTensor.fromTensor(
        onDevice ? await tensor(tgtData).to("webgpu") : tensor(tgtData),
        { requiresGrad: false }
      );
      const y = gradSoftmax(x, -1);
      y.mul(target).sum().backward();
      return {
        y: flat(((onDevice ? await y.tensor.cpu() : y.tensor) as { toArray(): unknown }).toArray()),
        g: flat(((onDevice ? await x.grad!.cpu() : x.grad!) as { toArray(): unknown }).toArray()),
      };
    };
    const ref = await run(false);
    const got = await run(true);
    for (let i = 0; i < ref.y.length; i++) expect(got.y[i]).toBeCloseTo(ref.y[i] ?? 0, 5);
    for (let i = 0; i < ref.g.length; i++) expect(got.g[i]).toBeCloseTo(ref.g[i] ?? 0, 4);
  });

  it("full MLP training step (matmul→relu→matmul→softmax→CE) grads match CPU on device", async () => {
    const W1data = [
      [0.1, -0.2, 0.3],
      [0.4, 0.5, -0.6],
    ];
    const W2data = [
      [0.2, -0.1],
      [0.3, 0.4],
      [-0.5, 0.6],
    ];
    const Xdata = [
      [1, -1],
      [0.5, 2],
      [-1, 0.3],
      [2, 1],
    ];
    const Ydata = [
      [1, 0],
      [0, 1],
      [1, 0],
      [0, 1],
    ];
    const run = async (onDevice: boolean) => {
      const mv = async (d: number[][]) => (onDevice ? await tensor(d).to("webgpu") : tensor(d));
      const W1 = parameter(await mv(W1data));
      const W2 = parameter(await mv(W2data));
      const X = GradTensor.fromTensor(await mv(Xdata), { requiresGrad: false });
      const Y = GradTensor.fromTensor(await mv(Ydata), { requiresGrad: false });
      const probs = gradSoftmax(X.matmul(W1).relu().matmul(W2), -1);
      const loss = Y.mul(probs.log()).sum().neg();
      loss.backward();
      return {
        gW1: flat(
          ((onDevice ? await W1.grad!.cpu() : W1.grad!) as { toArray(): unknown }).toArray()
        ),
        gW2: flat(
          ((onDevice ? await W2.grad!.cpu() : W2.grad!) as { toArray(): unknown }).toArray()
        ),
      };
    };
    const ref = await run(false);
    const got = await run(true);
    for (let i = 0; i < ref.gW1.length; i++) expect(got.gW1[i]).toBeCloseTo(ref.gW1[i] ?? 0, 4);
    for (let i = 0; i < ref.gW2.length; i++) expect(got.gW2[i]).toBeCloseTo(ref.gW2[i] ?? 0, 4);
  });

  it("Conv2d forward + backward (im2col/col2im on device) matches CPU", async () => {
    const B = 2;
    const Cin = 2;
    const H = 5;
    const W = 5;
    const xData = Array.from({ length: B }, (_, b) =>
      Array.from({ length: Cin }, (_, c) =>
        Array.from({ length: H }, (_, i) =>
          Array.from({ length: W }, (_, j) =>
            Math.sin((b * Cin * H * W + c * H * W + i * W + j) * 0.3)
          )
        )
      )
    );
    const run = async (onDevice: boolean) => {
      const conv = new Conv2d(Cin, 3, 3, { stride: 1, padding: 1 });
      const c = conv as unknown as {
        weight_: { tensor: { shape: number[] } };
        bias_?: { tensor: { shape: number[] } };
        registerParameter(n: string, p: unknown): void;
        to(d: string): Promise<unknown>;
      };
      const wShape = c.weight_.tensor.shape;
      const wN = wShape.reduce((a, b) => a * b, 1);
      c.weight_ = parameter(
        tensor(Array.from({ length: wN }, (_, i) => Math.cos(i * 0.11))).reshape(wShape)
      ) as never;
      c.registerParameter("weight", c.weight_);
      // Bias is randomized per-instance, pin it so CPU and device runs match.
      if (c.bias_) {
        const bShape = c.bias_.tensor.shape;
        const bN = bShape.reduce((a, b) => a * b, 1);
        c.bias_ = parameter(
          tensor(Array.from({ length: bN }, (_, i) => 0.01 * i)).reshape(bShape)
        ) as never;
        c.registerParameter("bias", c.bias_);
      }
      if (onDevice) await c.to("webgpu");
      const xT = onDevice ? await tensor(xData).to("webgpu") : tensor(xData);
      const x = GradTensor.fromTensor(xT, { requiresGrad: false });
      const y = conv.forward(x);
      y.mul(y).sum().backward();
      const wGrad = (
        c.weight_ as unknown as {
          grad: { cpu(): Promise<{ toArray(): unknown }>; toArray(): unknown };
        }
      ).grad;
      return {
        y: flat(((onDevice ? await y.tensor.cpu() : y.tensor) as { toArray(): unknown }).toArray()),
        gW: flat(((onDevice ? await wGrad.cpu() : wGrad) as { toArray(): unknown }).toArray()),
      };
    };
    const ref = await run(false);
    const got = await run(true);
    expect(got.y.length).toBe(ref.y.length);
    for (let i = 0; i < ref.y.length; i++) expect(got.y[i]).toBeCloseTo(ref.y[i] ?? 0, 4);
    for (let i = 0; i < ref.gW.length; i++) expect(got.gW[i]).toBeCloseTo(ref.gW[i] ?? 0, 4);
  });

  it("SGD and Adam optimizer steps run on device (full loop) matching CPU", async () => {
    const W0 = [
      [0.5, -0.3, 0.2],
      [0.1, 0.4, -0.6],
    ];
    const Xd = [
      [1, 2, 3],
      [4, 5, 6],
      [-1, 0, 2],
    ];
    const train = async (
      makeOpt: (p: GradTensor[]) => { zeroGrad(): void; step(): unknown },
      onDevice: boolean
    ) => {
      const w = parameter(onDevice ? await tensor(W0).to("webgpu") : tensor(W0));
      const opt = makeOpt([w]);
      const x = GradTensor.fromTensor(onDevice ? await tensor(Xd).to("webgpu") : tensor(Xd), {
        requiresGrad: false,
      });
      for (let s = 0; s < 4; s++) {
        opt.zeroGrad();
        w.matmul(x.transpose()).square().sum().backward();
        opt.step();
      }
      return flat(
        ((onDevice ? await w.tensor.cpu() : w.tensor) as { toArray(): unknown }).toArray()
      );
    };
    for (const makeOpt of [
      (p: GradTensor[]) => new SGD(p, { lr: 0.001, momentum: 0.9 }),
      (p: GradTensor[]) => new Adam(p, { lr: 0.01, weightDecay: 0.01 }),
    ]) {
      const ref = await train(makeOpt, false);
      const got = await train(makeOpt, true);
      for (let i = 0; i < ref.length; i++) expect(got[i]).toBeCloseTo(ref[i] ?? 0, 4);
    }
  });

  it("MaxPool2d forward + backward runs on device (incl. overlap/padding) matching CPU", async () => {
    const B = 1;
    const C = 2;
    const H = 5;
    const W = 5;
    const xData = Array.from({ length: B }, () =>
      Array.from({ length: C }, (_, c) =>
        Array.from({ length: H }, (_, i) =>
          Array.from({ length: W }, (_, j) => Math.sin((c * H * W + i * W + j) * 1.7) * 10)
        )
      )
    );
    const run = async (onDevice: boolean, k: number, s: number, p: number) => {
      const pool = new MaxPool2d(k, { stride: s, padding: p });
      const x = GradTensor.fromTensor(onDevice ? await tensor(xData).to("webgpu") : tensor(xData), {
        requiresGrad: true,
      });
      const y = pool.forward(x);
      y.mul(y).sum().backward();
      return {
        y: flat(((onDevice ? await y.tensor.cpu() : y.tensor) as { toArray(): unknown }).toArray()),
        g: flat(((onDevice ? await x.grad!.cpu() : x.grad!) as { toArray(): unknown }).toArray()),
      };
    };
    for (const [k, s, p] of [
      [2, 2, 0],
      [3, 2, 1],
      [2, 1, 0],
    ] as const) {
      const ref = await run(false, k, s, p);
      const got = await run(true, k, s, p);
      for (let i = 0; i < ref.y.length; i++) expect(got.y[i]).toBeCloseTo(ref.y[i] ?? 0, 4);
      for (let i = 0; i < ref.g.length; i++) expect(got.g[i]).toBeCloseTo(ref.g[i] ?? 0, 4);
    }
  });
});

describe("Module.to with a kernel backend", () => {
  it("moves parameters to device memory and back, preserving values", async () => {
    const layer = new Linear(3, 2);
    const before = flat([...layer.parameters()][0]?.tensor.toArray());
    await layer.to("webgpu");
    for (const p of layer.parameters()) {
      expect(p.tensor.isDeviceTensor).toBe(true);
      expect(p.tensor.device).toBe("webgpu");
    }
    await layer.to("cpu");
    const after = flat([...layer.parameters()][0]?.tensor.toArray());
    expect(after).toEqual(before);
  });

  it("forward on the device matches the CPU forward", async () => {
    const layer = new Linear(3, 2);
    const x = [[0.5, -1, 2]];
    const refOut = flat(
      layer.forward(GradTensor.fromTensor(tensor(x), { requiresGrad: false })).tensor.toArray()
    );
    await layer.to("webgpu");
    const out = layer.forward(
      GradTensor.fromTensor(await tensor(x).to("webgpu"), { requiresGrad: false })
    );
    const got = flat((await out.tensor.cpu()).toArray());
    for (let i = 0; i < refOut.length; i++) {
      expect(got[i]).toBeCloseTo(refOut[i] ?? 0, 5);
    }
  });
});

describe("host-accelerator registration semantics", () => {
  it("a backend without kernels keeps host storage for its device", () => {
    registerBackend("wasm", cpuOnlyBackend);
    const t = tensor([1, 2, 3], { device: "wasm" });
    expect(t.isDeviceTensor).toBe(false);
    expect(t.device).toBe("wasm");
    // Host storage stays readable and CPU ops work.
    expect(flat(add(t, t).toArray())).toEqual([2, 4, 6]);
  });
});

describe("dispatchMatmul (internal 2-D path)", () => {
  it("executes strict 2-D matmul on the device with shape validation", async () => {
    const { matmul } = await import("../src/ndarray/linalg/basic");
    const a = tensor(
      [
        [1, 2],
        [3, 4],
      ],
      dev
    );
    const b = tensor(
      [
        [5, 6],
        [7, 8],
      ],
      dev
    );
    const out = matmul(a, b);
    expect(flat((await out.cpu()).toArray())).toEqual([19, 22, 43, 50]);

    const cpuM = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(() => matmul(a, cpuM)).toThrow(/same device/);
    expect(() => matmul(a, tensor([1, 2], dev))).toThrow(/2D/);
    expect(() => matmul(a, zeros([3, 2], dev))).toThrow();
  });
});
