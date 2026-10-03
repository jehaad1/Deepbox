/**
 * Wave 2 regression tests for the ndarray group: issues that wave 1 auditors
 * reported outside their own scope.
 *
 * Device behaviour is exercised with a small in-process kernel backend that
 * enforces the same dtype rules as the WebGPU backend (no silent upcast).
 */

import { afterAll, describe, expect, it } from "vitest";
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
} from "../../src/core";
import {
  DataValidationError,
  DeviceError,
  DTypeError,
  InvalidParameterError,
  registerBackend,
  resetConfig,
  ShapeError,
} from "../../src/core";
import * as nd from "../../src/ndarray";
import {
  addScalar,
  argsort,
  clip,
  concatenate,
  dot,
  dropoutGrad,
  GradTensor,
  gather,
  gelu,
  logSoftmaxGrad,
  mulScalar,
  repeat,
  reshape,
  roll,
  softmaxGrad,
  sort,
  split,
  stack,
  stackGrad,
  sum,
  tensor,
  tensordot,
  tile,
  transpose,
  where,
} from "../../src/ndarray";
import { geluDerivative } from "../../src/ndarray/ops/activation";
import {
  dispatchCol2im,
  dispatchPool2d,
  dispatchReduce,
} from "../../src/ndarray/ops/device_dispatch";
import * as opsIndex from "../../src/ndarray/ops/index";
import {
  cbrt,
  exp2,
  expm1,
  floor,
  log1p,
  log2,
  log10,
  rsqrt,
  trunc,
} from "../../src/ndarray/ops/math";
import { roundToBFloat16 } from "../../src/ndarray/tensor/float16";
import { normalizeIndex, normalizeRange } from "../../src/ndarray/tensor/slice_helpers";
import { isContiguous, isDenseLayout } from "../../src/ndarray/tensor/strides";
import { Tensor } from "../../src/ndarray/tensor/Tensor";
import { setSeed } from "../../src/random";

// ---------------------------------------------------------------------------
// A small kernel backend over Float32Array that rejects mixed dtypes.
// ---------------------------------------------------------------------------

type FakeBuffer = DeviceBuffer & { data: Float32Array; freed: boolean };

const BINARY: Record<BinaryKernelOp, (x: number, y: number) => number> = {
  add: (x, y) => x + y,
  sub: (x, y) => x - y,
  mul: (x, y) => x * y,
  div: (x, y) => x / y,
  pow: (x, y) => x ** y,
  maximum: Math.max,
  minimum: Math.min,
};

/** Abramowitz and Stegun 7.1.26, absolute error below 1.5e-7. */
function erfApprox(x: number): number {
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
      t *
      Math.exp(-x * x);
  return x < 0 ? -y : y;
}

const UNARY: Record<UnaryKernelOp, (x: number) => number> = {
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
  erf: erfApprox,
  rsqrt: (x) => 1 / Math.sqrt(x),
  reciprocal: (x) => 1 / x,
  sign: Math.sign,
  expm1: Math.expm1,
  log1p: Math.log1p,
  softplus: (x) => Math.log1p(Math.exp(x)),
};

class MiniBackend implements KernelBackend {
  col2imCalls = 0;

  info(): BackendInfo {
    return { device: "webgpu", name: "Mini", available: true, capabilities: ["elementwise"] };
  }
  supports(cap: BackendCapability): boolean {
    return cap === "elementwise" || cap === "reduction";
  }
  async init(): Promise<void> {}
  dispose(): void {}

  private wrap(data: Float32Array, dtype: DeviceDType = "float32"): FakeBuffer {
    if (dtype === "bfloat16") {
      for (let i = 0; i < data.length; i++) data[i] = roundToBFloat16(data[i] ?? 0);
    }
    return {
      device: "webgpu",
      byteLength: data.length * 4,
      size: data.length,
      dtype,
      data,
      freed: false,
    };
  }

  private sameDType(op: string, ...bufs: DeviceBuffer[]): DeviceDType {
    const dtype = (bufs[0]?.dtype ?? "float32") as DeviceDType;
    for (const b of bufs) {
      if (((b.dtype ?? "float32") as DeviceDType) !== dtype) {
        throw new DeviceError(`mini ${op}: mismatched operand dtypes`);
      }
    }
    return dtype;
  }

  private read(buffer: DeviceBuffer, layout: KernelLayout): Float32Array {
    const buf = buffer as FakeBuffer;
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
    return (buffer as FakeBuffer).data.slice();
  }
  free(buffer: DeviceBuffer): void {
    (buffer as FakeBuffer).freed = true;
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
    const dtype = this.sameDType(op, a, b);
    const av = this.read(a, { ...aLayout, shape: outShape });
    const bv = this.read(b, { ...bLayout, shape: outShape });
    const fn = BINARY[op];
    const out = new Float32Array(av.length);
    for (let i = 0; i < out.length; i++) out[i] = Math.fround(fn(av[i] ?? 0, bv[i] ?? 0));
    return this.wrap(out, dtype);
  }

  unary(op: UnaryKernelOp, x: DeviceBuffer, layout: KernelLayout): DeviceBuffer {
    const xv = this.read(x, layout);
    const fn = UNARY[op];
    const out = new Float32Array(xv.length);
    for (let i = 0; i < out.length; i++) out[i] = Math.fround(fn(xv[i] ?? 0));
    return this.wrap(out, (x.dtype ?? "float32") as DeviceDType);
  }

  matmul(): DeviceBuffer {
    throw new Error("mini: matmul is not implemented");
  }
  matmulBatched(): DeviceBuffer {
    throw new Error("mini: matmulBatched is not implemented");
  }

  reduce(op: ReduceKernelOp, x: DeviceBuffer, layout: KernelLayout): DeviceBuffer {
    const xv = this.read(x, layout);
    let acc = 0;
    if (op === "max") acc = Math.max(...xv);
    else if (op === "min") acc = Math.min(...xv);
    else {
      for (const v of xv) acc += v;
      if (op === "mean") acc /= xv.length;
    }
    return this.wrap(new Float32Array([Math.fround(acc)]), (x.dtype ?? "float32") as DeviceDType);
  }
  reduceAxis(): DeviceBuffer {
    throw new Error("mini: reduceAxis is not implemented");
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
    const dtype = this.sameDType("where", a, b);
    const cv = this.read(cond, { ...condLayout, shape: outShape });
    const av = this.read(a, { ...aLayout, shape: outShape });
    const bv = this.read(b, { ...bLayout, shape: outShape });
    const out = new Float32Array(cv.length);
    for (let i = 0; i < out.length; i++) out[i] = (cv[i] ?? 0) !== 0 ? (av[i] ?? 0) : (bv[i] ?? 0);
    return this.wrap(out, dtype);
  }

  im2col(): DeviceBuffer {
    throw new Error("mini: im2col must not be reached");
  }
  col2im(_cols: DeviceBuffer, _p: Im2ColParams): DeviceBuffer {
    this.col2imCalls++;
    throw new Error("mini: col2im must not be reached");
  }
  pool2d(_x: DeviceBuffer, _l: KernelLayout, _op: PoolKernelOp, _p: Im2ColParams): DeviceBuffer {
    throw new Error("mini: pool2d must not be reached");
  }
  pool2dBackward(): DeviceBuffer {
    throw new Error("mini: pool2dBackward must not be reached");
  }
}

const mini = new MiniBackend();

// Registered at load time because some describe blocks create device tensors eagerly.
registerBackend("webgpu", mini as Backend);

afterAll(() => {
  resetConfig();
});

const dev = { device: "webgpu" as const };

/** Read a device or host tensor back as a flat number array. */
async function values(t: Tensor): Promise<number[]> {
  const host = await t.cpu();
  return Array.from(host.data as Float32Array).slice(host.offset, host.offset + host.size);
}

function expectClose(
  actual: ArrayLike<number>,
  expected: readonly number[],
  rel = 1e-9,
  abs = 1e-12
): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    const e = expected[i] as number;
    const a = actual[i] as number;
    expect(Math.abs(a - e), `index ${i}: ${a} vs ${e}`).toBeLessThanOrEqual(
      rel * Math.abs(e) + abs
    );
  }
}

// ---------------------------------------------------------------------------
// Public entry point: barrel exports
// ---------------------------------------------------------------------------

describe("ndarray barrel exports", () => {
  it("exposes the fft helpers, trapezoid and the half-precision rounding helpers", () => {
    for (const name of [
      "fftfreq",
      "rfftfreq",
      "fftshift",
      "ifftshift",
      "trapezoid",
      "roundToFloat16",
      "roundToBFloat16",
    ]) {
      expect(typeof (nd as Record<string, unknown>)[name], name).toBe("function");
    }
    expect(typeof opsIndex.trapezoid).toBe("function");
    expect(nd.trapezoid).toBe(nd.trapz);
  });

  it("the exported helpers work through the public entry", () => {
    expect(nd.fftfreq(4).toArray()).toEqual([0, 0.25, -0.5, -0.25]);
    expect(nd.rfftfreq(4).toArray()).toEqual([0, 0.25, 0.5]);
    expect(nd.fftshift(nd.fftfreq(4)).toArray()).toEqual([-0.5, -0.25, 0, 0.25]);
    expect(nd.ifftshift(nd.fftshift(nd.fftfreq(4))).toArray()).toEqual([0, 0.25, -0.5, -0.25]);
    expect(nd.roundToBFloat16(1.00001)).toBe(1);
  });

  it("exports the option and result types by name", () => {
    const mode: nd.ConvolveMode = "same";
    const win: nd.WindowOptions = {};
    const like: nd.LikeOptions = {};
    const hist: nd.HistogramOptions = {};
    const padMode: nd.PadMode = "edge";
    const padWidth: nd.PadWidth = [1, 1];
    const axes: nd.TensordotAxes = 1;
    const norm: nd.FFTNorm = "ortho";
    const nested: nd.StringNestedArray = ["a", ["b"]];
    const gelumode: nd.GeluApproximation = "none";
    expect([mode, win, like, hist, padMode, padWidth, axes, norm, nested, gelumode]).toHaveLength(
      10
    );
    expect(nd.pad(tensor([1, 2]), padWidth, padMode).toArray()).toEqual([1, 1, 2, 2]);
  });
});

// ---------------------------------------------------------------------------
// Device dispatch
// ---------------------------------------------------------------------------

describe("device scalars take the dtype of the device operand", () => {
  it("mul, add and clip with a host scalar work for bfloat16 tensors", async () => {
    const x = tensor([1, 2, 3, 4], { ...dev, dtype: "bfloat16" });
    const doubled = mulScalar(x, 2);
    expect(doubled.dtype).toBe("bfloat16");
    expect(await values(doubled)).toEqual([2, 4, 6, 8]);
    const shifted = addScalar(x, 1);
    expect(shifted.dtype).toBe("bfloat16");
    expect(await values(shifted)).toEqual([2, 3, 4, 5]);
    const clipped = clip(x, 2, 3);
    expect(clipped.dtype).toBe("bfloat16");
    expect(await values(clipped)).toEqual([2, 2, 3, 3]);
  });

  it("a 0-d host tensor combines with a bfloat16 device tensor in either order", async () => {
    const x = tensor([1, 2, 3], { ...dev, dtype: "bfloat16" });
    const two = tensor(2, { dtype: "bfloat16" });
    expect(await values(nd.mul(x, two))).toEqual([2, 4, 6]);
    expect(await values(nd.sub(two, x))).toEqual([1, 0, -1]);
  });

  it("where() with a host scalar branch keeps the value dtype", async () => {
    const x = tensor([1, 2, 3, 4], { ...dev, dtype: "bfloat16" });
    const cond = tensor([1, 0, 1, 0], dev);
    const zero = tensor(0, { dtype: "bfloat16" });
    const out = where(cond, x, zero);
    expect(out.dtype).toBe("bfloat16");
    expect(await values(out)).toEqual([1, 0, 3, 0]);
    const out2 = where(cond, zero, x);
    expect(await values(out2)).toEqual([0, 2, 0, 4]);
  });

  it("float32 device tensors are unaffected", async () => {
    const x = tensor([1, 2], dev);
    expect(await values(mulScalar(x, 3))).toEqual([3, 6]);
  });
});

describe("device dot and reductions follow the CPU rules", () => {
  it("dot rejects 0-d operands on a device, like the CPU implementation", () => {
    const cpuMsg = (() => {
      try {
        dot(tensor([1, 2, 3]), tensor(2));
      } catch (e) {
        return (e as Error).message;
      }
      return "";
    })();
    expect(cpuMsg).toMatch(/0-d/);
    const x = tensor([1, 2, 3], dev);
    expect(() => dot(x, tensor(2))).toThrow(ShapeError);
    expect(() => dot(x, tensor(2))).toThrow(/does not accept 0-d tensors/);
    expect(() => dot(tensor(2), x)).toThrow(ShapeError);
  });

  it("an empty axis list reduces nothing on a device tensor", async () => {
    const x = tensor([1, 2, 3], dev);
    const out = dispatchReduce("sum", x, [], false);
    expect(out).not.toBeNull();
    expect(out?.shape).toEqual([3]);
    expect(await values(out as Tensor)).toEqual([1, 2, 3]);
    const cpu = sum(tensor([1, 2, 3]), []);
    expect(cpu.shape).toEqual([3]);
    // Through the public op as well.
    const viaOp = sum(x, []);
    expect(viaOp.shape).toEqual([3]);
    expect(await values(viaOp)).toEqual([1, 2, 3]);
  });

  it("an omitted axis is still a full reduction", async () => {
    const x = tensor([1, 2, 3], dev);
    expect(await values(sum(x))).toEqual([6]);
  });
});

describe("window geometry is validated before any kernel runs", () => {
  it("col2im reports a kernel that does not fit instead of reaching the backend", () => {
    const cols = tensor([[[1, 2, 3, 4]]], dev);
    const before = mini.col2imCalls;
    expect(() => dispatchCol2im(cols, [1, 1, 2, 2], [5, 5], [1, 1], [0, 0])).toThrow(
      InvalidParameterError
    );
    expect(() => dispatchCol2im(cols, [1, 1, 2, 2], [5, 5], [1, 1], [0, 0])).toThrow(
      /does not fit/
    );
    expect(() => dispatchCol2im(cols, [1, 1, 2, 2], [2, 2], [0, 1], [0, 0])).toThrow(
      InvalidParameterError
    );
    expect(() => dispatchCol2im(cols, [1, 2, 2], [1, 1], [1, 1], [0, 0])).toThrow(ShapeError);
    expect(mini.col2imCalls).toBe(before);
  });

  it("an input with a zero-sized spatial dimension gets the kernel-too-large message", () => {
    const empty = Tensor.fromDeviceBuffer({
      owner: new (class {
        buffer = mini.fill(0, 1);
        backend = mini;
        acquire(): void {}
        release(): void {}
        isFreed = false;
      })() as never,
      shape: [1, 1, 0, 3],
      device: "webgpu",
    });
    expect(() => dispatchPool2d(empty, "max", [1, 1], [1, 1], [0, 0])).toThrow(/does not fit/);
  });
});

describe("device-only functions give a precise hint", () => {
  const x = tensor([1, 2, 3, 4], dev);

  it("concatenate, stack, split, tile and repeat name themselves in the error", () => {
    expect(() => concatenate([x, x])).toThrow(DeviceError);
    expect(() => concatenate([x, x])).toThrow(/concatenate is not available on device "webgpu"/);
    expect(() => stack([x, x])).toThrow(/stack is not available on device "webgpu"/);
    expect(() => split(x, 2)).toThrow(/split is not available on device "webgpu"/);
    expect(() => tile(x, [2])).toThrow(/tile is not available on device "webgpu"/);
    expect(() => repeat(x, 2)).toThrow(/repeat is not available on device "webgpu"/);
    expect(() => concatenate([tensor([1, 2]), x])).toThrow(/concatenate is not available/);
  });

  it("host tensors are unaffected", () => {
    expect(concatenate([tensor([1, 2]), tensor([3])]).toArray()).toEqual([1, 2, 3]);
  });

  it("math functions without a kernel say so, and the ones with a kernel run on the device", async () => {
    for (const fn of [cbrt, exp2, log2, log10, floor, trunc]) {
      expect(() => fn(x)).toThrow(DeviceError);
      expect(() => fn(x)).toThrow(/has no kernel on device "webgpu"/);
    }
    const pos = tensor([1, 4, 9, 16], dev);
    const r = rsqrt(pos);
    expect(r.isDeviceTensor).toBe(true);
    expectClose(await values(r), [1, 0.5, 1 / 3, 0.25], 1e-6);
    const e = expm1(tensor([0, 1e-3], dev));
    expect(e.isDeviceTensor).toBe(true);
    expectClose(await values(e), [0, Math.expm1(1e-3)], 1e-6);
    const l = log1p(tensor([0, 1e-3], dev));
    expect(l.isDeviceTensor).toBe(true);
    expectClose(await values(l), [0, Math.log1p(1e-3)], 1e-6);
  });

  it("math functions on host tensors keep their results", () => {
    expect(rsqrt(tensor([4, 16], { dtype: "float64" })).toArray()).toEqual([0.5, 0.25]);
    expect(cbrt(tensor([8, 27], { dtype: "float64" })).toArray()).toEqual([2, 3]);
    expect(exp2(tensor([3], { dtype: "float64" })).toArray()).toEqual([8]);
    expect(log2(tensor([8], { dtype: "float64" })).toArray()).toEqual([3]);
    expect(
      Object.is((nd.round(tensor([-0.4], { dtype: "float64" })).toArray() as number[])[0], -0)
    ).toBe(true);
    expect(nd.log(tensor([1, Math.E])).dtype).toBe("float32");
    expect(nd.log(tensor([1, Math.E], { dtype: "int32" })).dtype).toBe("float32");
    expect(() => nd.exp(tensor(["a"]))).toThrow(DTypeError);
  });

  it("math functions read strided views through their strides", () => {
    const t = transpose(
      tensor(
        [
          [1, 4],
          [9, 16],
        ],
        { dtype: "float64" }
      )
    );
    expect(nd.sqrt(t).toArray()).toEqual([
      [1, 3],
      [2, 4],
    ]);
    expect(expm1(t).shape).toEqual([2, 2]);
  });
});

// ---------------------------------------------------------------------------
// GradTensor on devices
// ---------------------------------------------------------------------------

describe("gelu and hardtanh backward on device tensors", () => {
  const data = [-3, -1, -0.25, 0, 0.5, 2];

  it("gelu (tanh) gradient on the device matches the host gradient", async () => {
    const host = GradTensor.fromTensor(tensor(data), { requiresGrad: true });
    host.gelu().sum().backward();
    const onDevice = GradTensor.fromTensor(tensor(data, dev), { requiresGrad: true });
    onDevice.gelu().sum().backward();
    const g = onDevice.grad as Tensor;
    expect(g.isDeviceTensor).toBe(true);
    expectClose(await values(g), Array.from((host.grad as Tensor).data as Float32Array), 1e-4);
  });

  it("gelu (exact) gradient on the device matches the host gradient", async () => {
    const host = GradTensor.fromTensor(tensor(data), { requiresGrad: true });
    host.gelu("none").sum().backward();
    const onDevice = GradTensor.fromTensor(tensor(data, dev), { requiresGrad: true });
    onDevice.gelu("none").sum().backward();
    const g = onDevice.grad as Tensor;
    expect(g.isDeviceTensor).toBe(true);
    // The fake erf kernel is accurate to 1.5e-7 only.
    const hostGrad = Array.from((host.grad as Tensor).data as Float32Array);
    const devGrad = await values(g);
    for (let i = 0; i < data.length; i++) {
      expect(devGrad[i]).toBeCloseTo(hostGrad[i] as number, 5);
    }
  });

  it("hardtanh gradient on the device is 1 strictly inside the interval and 0 elsewhere", async () => {
    const x = GradTensor.fromTensor(tensor([-2, -1, -0.5, 0, 0.5, 1, 2], dev), {
      requiresGrad: true,
    });
    x.hardtanh(-1, 1).sum().backward();
    expect(await values(x.grad as Tensor)).toEqual([0, 0, 1, 1, 1, 0, 0]);
    const x2 = GradTensor.fromTensor(tensor([-2, 0.5, 3], dev), { requiresGrad: true });
    x2.hardtanh(0, 2).sum().backward();
    expect(await values(x2.grad as Tensor)).toEqual([0, 1, 0]);
  });

  it("works for bfloat16 device tensors (scalars take the operand dtype)", async () => {
    const x = GradTensor.fromTensor(tensor([-2, 0.5, 2], { ...dev, dtype: "bfloat16" }), {
      requiresGrad: true,
    });
    x.hardtanh(-1, 1).sum().backward();
    expect(await values(x.grad as Tensor)).toEqual([0, 1, 0]);
  });
});

// ---------------------------------------------------------------------------
// Exact GELU
// ---------------------------------------------------------------------------

describe("exact (erf) GELU", () => {
  // scipy.special.ndtr and torch.nn.functional.gelu in float64.
  const x = [-8, -3.5, -1, -0.25, 0, 0.3, 1, 2.5, 6];
  const exact = [
    -4.976768459417392e-15, -0.0008142017766243376, -0.15865525393145707, -0.10032341857926907, 0,
    0.18537342665668577, 0.8413447460685429, 2.4844758366855597, 5.999999994080474,
  ];
  const exactGrad = [
    -3.9807546004751306e-14, -0.0028217603536246205, -0.08331547058768635, 0.304626645116364, 0.5,
    0.7323277668271099, 1.0833154705876864, 1.037611085908145, 1.0000000354687093,
  ];
  const tanhGrad = [
    0, -0.002422643760116816, -0.08296408384578258, 0.30464590484893955, 0.5, 0.7322954516388018,
    1.0829640838457826, 1.037951576212666, 1.0000000007709977,
  ];
  const tanhValues = [
    0, -0.0006161976553737403, -0.1588080093917233, -0.100324649298315, 0, 0.18537092354275922,
    0.8411919906082768, 2.484915733910001, 5.9999999999156035,
  ];

  it("matches scipy over the whole range, including the far tails", () => {
    const out = gelu(tensor(x, { dtype: "float64" }), "none");
    expect(out.dtype).toBe("float64");
    expectClose(out.data as Float64Array, exact, 1e-12, 0);
  });

  it("the default stays the tanh approximation", () => {
    const out = gelu(tensor(x, { dtype: "float64" }));
    expectClose(out.data as Float64Array, tanhValues, 1e-9, 1e-15);
    expect(Math.abs(((out.data as Float64Array)[3] as number) - exact[3]!)).toBeGreaterThan(1e-7);
    const explicit = gelu(tensor(x, { dtype: "float64" }), "tanh");
    expect(Array.from(explicit.data as Float64Array)).toEqual(Array.from(out.data as Float64Array));
  });

  it("handles infinities and NaN", () => {
    const out = gelu(tensor([-Infinity, Infinity, Number.NaN], { dtype: "float64" }), "none")
      .data as Float64Array;
    expect(out[0]).toBe(0);
    expect(out[1]).toBe(Infinity);
    expect(out[2]).toBeNaN();
  });

  it("rejects an unknown variant", () => {
    expect(() => gelu(tensor([1]), "erf" as never)).toThrow(InvalidParameterError);
    expect(() => geluDerivative(tensor([1]), "erf" as never)).toThrow(InvalidParameterError);
    expect(() => new GradTensor([1]).gelu("x" as never)).toThrow(InvalidParameterError);
  });

  it("GradTensor.gelu gradient matches torch for both variants", () => {
    const e = GradTensor.fromTensor(tensor(x, { dtype: "float64" }), { requiresGrad: true });
    e.gelu("none").sum().backward();
    expectClose((e.grad as Tensor).data as Float64Array, exactGrad, 1e-12, 1e-16);
    const t = GradTensor.fromTensor(tensor(x, { dtype: "float64" }), { requiresGrad: true });
    t.gelu().sum().backward();
    expectClose((t.grad as Tensor).data as Float64Array, tanhGrad, 1e-12, 1e-16);
  });

  it("the derivative at infinity is the limit", () => {
    const d = geluDerivative(tensor([-Infinity, Infinity], { dtype: "float64" }), "none")
      .data as Float64Array;
    expect(Array.from(d)).toEqual([0, 1]);
    const d2 = geluDerivative(tensor([-Infinity, Infinity], { dtype: "float64" }))
      .data as Float64Array;
    expect(Array.from(d2)).toEqual([0, 1]);
  });

  it("the exact form runs on the device through the erf kernel", async () => {
    const out = gelu(tensor([-1, 0, 1, 2], dev), "none");
    expect(out.isDeviceTensor).toBe(true);
    const host = Array.from(
      gelu(tensor([-1, 0, 1, 2], { dtype: "float64" }), "none").data as Float64Array
    );
    const got = await values(out);
    for (let i = 0; i < host.length; i++) expect(got[i]).toBeCloseTo(host[i] as number, 5);
  });
});

// ---------------------------------------------------------------------------
// Autograd
// ---------------------------------------------------------------------------

describe("dropout on non-float input", () => {
  it("converts integer input to float32 so the scale 1 / (1 - p) is not truncated", () => {
    setSeed(7);
    const x = GradTensor.fromTensor(tensor(new Array(200).fill(1), { dtype: "int32" }));
    const y = dropoutGrad(x, 0.3);
    expect(y.dtype).toBe("float32");
    const data = y.tensor.data as Float32Array;
    const kept = Array.from(data).filter((v) => v !== 0);
    expect(kept.length).toBeGreaterThan(100);
    for (const v of kept) expect(v).toBeCloseTo(1 / 0.7, 5);
  });

  it("bool input is converted as well and float input keeps its dtype", () => {
    setSeed(1);
    const b = GradTensor.fromTensor(tensor([true, true, true, true], { dtype: "bool" }));
    expect(dropoutGrad(b, 0.5).dtype).toBe("float32");
    const f = GradTensor.fromTensor(tensor([1, 2, 3], { dtype: "float64" }));
    expect(dropoutGrad(f, 0.5).dtype).toBe("float64");
    expect(dropoutGrad(b, 0.5, false)).toBe(b);
  });
});

describe("stackGrad keeps integer values exact", () => {
  it("int64 values above 2^53 survive stacking", () => {
    const big = 2n ** 62n + 1n;
    const a = GradTensor.fromTensor(tensor(new BigInt64Array([big, 2n]), { dtype: "int64" }));
    const b = GradTensor.fromTensor(tensor(new BigInt64Array([3n, big + 2n]), { dtype: "int64" }));
    const s = stackGrad([a, b]);
    expect(s.shape).toEqual([2, 2]);
    expect(Array.from(s.tensor.data as BigInt64Array)).toEqual([big, 2n, 3n, big + 2n]);
  });

  it("the gradient is split back to each part", () => {
    const a = GradTensor.fromTensor(tensor([1, 2, 3]), { requiresGrad: true });
    const b = GradTensor.fromTensor(tensor([4, 5, 6]), { requiresGrad: true });
    const w = GradTensor.fromTensor(
      tensor([
        [1, 2, 3],
        [10, 20, 30],
      ])
    );
    stackGrad([a, b]).mul(w).sum().backward();
    expect(Array.from((a.grad as Tensor).data as Float32Array)).toEqual([1, 2, 3]);
    expect(Array.from((b.grad as Tensor).data as Float32Array)).toEqual([10, 20, 30]);
  });

  it("0-d parts and mixed dtypes follow the first part", () => {
    const a = GradTensor.fromTensor(tensor(1), { requiresGrad: true });
    const b = GradTensor.fromTensor(tensor(2, { dtype: "float64" }), { requiresGrad: true });
    const s = stackGrad([a, b]);
    expect(s.shape).toEqual([2]);
    expect(s.dtype).toBe("float32");
    s.sum().backward();
    expect((a.grad as Tensor).dtype).toBe("float32");
    expect((b.grad as Tensor).dtype).toBe("float64");
  });
});

describe("fused softmax and log-softmax", () => {
  const w2 = tensor(
    [
      [1, -2, 0.5],
      [3, 0.25, -1],
    ],
    { dtype: "float64" }
  );
  const input2 = () =>
    GradTensor.fromTensor(
      tensor(
        [
          [1, 2, 3],
          [0.5, -1, 2],
        ],
        { dtype: "float64" }
      ),
      { requiresGrad: true }
    );

  it("log-softmax along the last axis matches torch", () => {
    const x = input2();
    const y = logSoftmaxGrad(x, 1);
    y.mul(GradTensor.fromTensor(w2)).sum().backward();
    expectClose(
      y.tensor.data as Float64Array,
      [
        -2.4076059644443806, -1.4076059644443804, -0.4076059644443804, -1.7413112966571571,
        -3.241311296657157, -0.24131129665715703,
      ]
    );
    expectClose(
      (x.grad as Tensor).data as Float64Array,
      [
        1.0450152865851903, -1.8776357644726012, 0.8326204778874109, 2.6055966176849177,
        0.16199671014095324, -2.7675933278258706,
      ]
    );
  });

  it("softmax along axis 0 matches torch", () => {
    const x = input2();
    const y = softmaxGrad(x, 0);
    y.mul(GradTensor.fromTensor(w2)).sum().backward();
    expectClose(
      y.tensor.data as Float64Array,
      [
        0.6224593312018546, 0.9525741268224334, 0.7310585786300049, 0.37754066879814546,
        0.04742587317756679, 0.2689414213699951,
      ]
    );
    expectClose(
      (x.grad as Tensor).data as Float64Array,
      [
        -0.4700074244031891, -0.10164748439455214, 0.2949178998622228, 0.47000742440318893,
        0.10164748439455232, -0.2949178998622228,
      ]
    );
  });

  it("softmax along a middle axis of a 3-D tensor matches torch", () => {
    const a = [
      [
        [0.1257302210933933, -0.1321048632913019],
        [0.6404226504432821, 0.10490011715303971],
        [-0.535669373161111, 0.36159505490948474],
      ],
      [
        [1.3040000451301372, 0.9470809631292422],
        [-0.7037352358069926, -1.2654214710460525],
        [-0.6232744625373522, 0.0413259793472436],
      ],
    ];
    const w = [
      [
        [-2.3250307746388343, -0.21879166393254573],
        [-1.2459109472530652, -0.7322673547034516],
        [-0.5442589828573099, -0.31630015636915454],
      ],
      [
        [0.4116305363741328, 1.0425133694426776],
        [-0.12853466294403426, 1.3664634705496859],
        [-0.6651946734866135, 0.3515100700930197],
      ],
    ];
    const x = GradTensor.fromTensor(tensor(a, { dtype: "float64" }), { requiresGrad: true });
    const y = softmaxGrad(x, 1);
    y.mul(GradTensor.fromTensor(tensor(w, { dtype: "float64" })))
      .sum()
      .backward();
    expectClose(
      y.tensor.data as Float64Array,
      [
        0.31355311937931274, 0.2560285589797887, 0.5246131925145855, 0.3245027391395449,
        0.1618336881061018, 0.41946870188066643, 0.7813496198646922, 0.6606490126240229,
        0.10492936666975322, 0.07229249138182496, 0.11372101346555454, 0.26705849599415216,
      ]
    );
    expectClose(
      (x.grad as Tensor).data as Float64Array,
      [
        -0.2678713661507828, 0.05313259012531668, 0.1179385346504563, -0.09928150332194477,
        0.14993283150032632, 0.04614891319662812, 0.13996850687912274, 0.10644320204997491,
        -0.03788247641605863, 0.035066864073930296, -0.10208603046306415, -0.14151006612390526,
      ]
    );
  });

  it("keeps the float32 dtype and a large-logit lane stays finite", () => {
    const x = GradTensor.fromTensor(tensor([[1000, 1001, 1002]]), { requiresGrad: true });
    const y = logSoftmaxGrad(x, -1);
    expect(y.dtype).toBe("float32");
    const d = y.tensor.data as Float32Array;
    expect(d[2]).toBeCloseTo(-0.40760594606399536, 6);
    const s = softmaxGrad(x, -1);
    expect(Array.from(s.tensor.data as Float32Array).reduce((p, c) => p + c, 0)).toBeCloseTo(1, 6);
  });

  it("a lane with NaN or an infinite maximum gives NaN, like the composite formula", () => {
    const x = GradTensor.fromTensor(
      tensor(
        [
          [1, 2, 3],
          [Infinity, 1, 1],
          [Number.NaN, 1, 2],
          [-Infinity, -Infinity, -Infinity],
          [-Infinity, 0, 0],
        ],
        { dtype: "float64" }
      ),
      { requiresGrad: true }
    );
    const y = logSoftmaxGrad(x, 1).tensor.data as Float64Array;
    expect(Number.isFinite(y[0] as number)).toBe(true);
    for (let i = 3; i < 12; i++) expect(y[i]).toBeNaN();
    expect(y[12]).toBe(-Infinity);
    expect(y[13]).toBeCloseTo(-Math.LN2, 12);
  });

  it("integer inputs compute in float32 and invalid axes are rejected", () => {
    // 1.5.0 dtype rule: integer input to a fractional op computes in float32.
    const xi = GradTensor.fromTensor(tensor([[1, 2, 3]], { dtype: "int32" }));
    const out = logSoftmaxGrad(xi, 1);
    expect(out.dtype).toBe("float32");
    // torch.log_softmax([[1., 2., 3.]], 1)
    const expected = [-2.4076059644, -1.4076059644, -0.4076059644];
    const got = (out.toArray() as number[][])[0] ?? [];
    for (let i = 0; i < expected.length; i++) {
      expect(got[i]).toBeCloseTo(expected[i] ?? Number.NaN, 6);
    }
    const x = input2();
    expect(() => softmaxGrad(x, 5)).toThrow(InvalidParameterError);
  });
});

// ---------------------------------------------------------------------------
// Tensor, indexing and shape
// ---------------------------------------------------------------------------

describe("gather", () => {
  it("matches numpy.take on strided input, repeated indices and negative axes", () => {
    const base = tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      { dtype: "int32" }
    );
    const t = transpose(base); // [[1,4],[2,5],[3,6]] as a strided view
    expect(gather(t, tensor([2, 0, 2], { dtype: "int32" }), 0).toArray()).toEqual([
      [3, 6],
      [1, 4],
      [3, 6],
    ]);
    expect(gather(t, tensor([1, 1, 0], { dtype: "int32" }), -1).toArray()).toEqual([
      [4, 4, 1],
      [5, 5, 2],
      [6, 6, 3],
    ]);
  });

  it("keeps int64 exact and works for string tensors and 3-D input", () => {
    const big = 2n ** 60n + 3n;
    const t = tensor(new BigInt64Array([big, 1n, 2n, 3n]), { dtype: "int64" }).reshape([2, 2]);
    const g = gather(t, tensor([1, 0], { dtype: "int32" }), 0);
    expect(Array.from(g.data as BigInt64Array)).toEqual([2n, 3n, big, 1n]);
    const s = tensor([
      ["a", "b"],
      ["c", "d"],
    ]);
    expect(gather(s, tensor([1, 1], { dtype: "int32" }), 1).toArray()).toEqual([
      ["b", "b"],
      ["d", "d"],
    ]);
    const cube = tensor(
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
      { dtype: "int32" }
    );
    expect(gather(cube, tensor([1], { dtype: "int32" }), 1).toArray()).toEqual([
      [[3, 4]],
      [[7, 8]],
    ]);
  });

  it("validates indices", () => {
    const t = tensor([1, 2, 3]);
    expect(() => gather(t, tensor([3], { dtype: "int32" }), 0)).toThrow();
    expect(() => gather(t, tensor([0.5]), 0)).toThrow(InvalidParameterError);
  });

  it("is fast: no allocation per output element", () => {
    const t = nd.randn([4000, 64]);
    const idx = tensor(
      Array.from({ length: 4000 }, (_, i) => (i * 7919) % 4000),
      { dtype: "int32" }
    );
    const start = performance.now();
    const out = gather(t, idx, 0);
    const elapsed = performance.now() - start;
    expect(out.shape).toEqual([4000, 64]);
    expect(elapsed).toBeLessThan(2000);
  });
});

describe("complex dtypes", () => {
  it("tensors cannot hold complex data, so gather never sees an imaginary part", () => {
    const c = new nd.Complex128Array(2);
    expect(() =>
      Tensor.fromTypedArray({ data: c as never, shape: [2], dtype: "complex128", device: "cpu" })
    ).toThrow(DTypeError);
    expect(() => tensor([1, 2]).astype("complex64")).toThrow(DTypeError);
  });
});

describe("slice bounds are validated in one place", () => {
  it("normalizeRange and normalizeIndex reject NaN and fractional bounds", () => {
    expect(() => normalizeRange({ start: 1.5 }, 5)).toThrow(InvalidParameterError);
    expect(() => normalizeRange({ end: Number.NaN }, 5)).toThrow(InvalidParameterError);
    expect(() => normalizeRange({ step: 0.5 }, 5)).toThrow(InvalidParameterError);
    expect(() => normalizeRange(2.5, 5)).toThrow(InvalidParameterError);
    expect(() => normalizeIndex(Number.NaN, 5)).toThrow(InvalidParameterError);
    expect(normalizeRange({ start: -Infinity, end: Infinity }, 5)).toEqual({
      start: 0,
      end: 5,
      step: 1,
    });
  });

  it("Tensor.slice and the standalone slice() reject bad bounds without a second check", () => {
    const t = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(() => t.slice({ start: 0.5 })).toThrow(InvalidParameterError);
    expect(() => t.slice(0, { end: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => nd.slice(t, 1.2)).toThrow(InvalidParameterError);
    expect(() => nd.slice(t, Number.POSITIVE_INFINITY)).toThrow(InvalidParameterError);
    expect(() => t.slice(0, 0, 0)).toThrow(ShapeError);
    expect(t.slice({ start: -Infinity, end: Infinity }).toArray()).toEqual([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(t.slice(1, { start: 1 }).toArray()).toEqual([5, 6]);
  });
});

describe("contiguity of size-1 axes", () => {
  it("isDenseLayout ignores the stride of size-1 axes, isContiguous stays strict", () => {
    expect(isContiguous([3, 1], [1, 3])).toBe(false);
    expect(isDenseLayout([3, 1], [1, 3])).toBe(true);
    expect(isDenseLayout([1, 3], [7, 1])).toBe(true);
    expect(isDenseLayout([2, 3], [1, 2])).toBe(false);
    expect(isDenseLayout([2, 3], [3, 1])).toBe(true);
    expect(isDenseLayout([0, 3], [99, 99])).toBe(true);
    expect(isDenseLayout([], [])).toBe(true);
    expect(isDenseLayout([2], [])).toBe(false);
  });

  it("reshape and astype do not copy a transposed row vector", () => {
    const row = tensor([[1, 2, 3]], { dtype: "float64" });
    const col = transpose(row); // shape [3, 1], strides [1, 3]
    expect(col.shape).toEqual([3, 1]);
    const flat = col.reshape([3]);
    expect(flat.data).toBe(row.data);
    expect(flat.toArray()).toEqual([1, 2, 3]);
    expect(reshape(col, [-1]).data).toBe(row.data);
    const cast = col.astype("float32");
    expect(cast.toArray()).toEqual([[1], [2], [3]]);
  });

  it("a genuinely strided view is still copied into row-major order", () => {
    const m = tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      { dtype: "float64" }
    );
    const flat = transpose(m).reshape([6]);
    expect(flat.toArray()).toEqual([1, 4, 2, 5, 3, 6]);
    expect(flat.data).not.toBe(m.data);
  });

  it("astype on a strided tensor reads the right elements", () => {
    const m = tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      { dtype: "int32" }
    );
    expect(transpose(m).astype("float64").toArray()).toEqual([
      [1, 4],
      [2, 5],
      [3, 6],
    ]);
    expect(transpose(m).astype("string").toArray()).toEqual([
      ["1", "4"],
      ["2", "5"],
      ["3", "6"],
    ]);
  });
});

describe("reshape has one implementation", () => {
  it("the function and the method agree, including errors", () => {
    const t = tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      { dtype: "float64" }
    );
    expect(reshape(t, [3, -1]).toArray()).toEqual(t.reshape([3, -1]).toArray());
    expect(() => reshape(t, [-1, -1])).toThrow(ShapeError);
    expect(() => t.reshape([-1, -1])).toThrow(ShapeError);
    expect(() => reshape(t, [4, -1])).toThrow(ShapeError);
    expect(() => reshape(t, [7])).toThrow(ShapeError);
    expect(() => reshape(t, [2.5, -1])).toThrow(DataValidationError);
    expect(() => t.reshape([2.5, -1])).toThrow(DataValidationError);
    expect(() => reshape(t, [0, -1])).toThrow(ShapeError);
  });

  it("handles strings, int64 and strided views through the same code", () => {
    const s = tensor([
      ["a", "b", "c"],
      ["d", "e", "f"],
    ]);
    expect(reshape(transpose(s), [6]).toArray()).toEqual(["a", "d", "b", "e", "c", "f"]);
    const big = 2n ** 55n + 1n;
    const i = tensor(new BigInt64Array([big, 2n, 3n, 4n]), { dtype: "int64" }).reshape([2, 2]);
    expect(Array.from(reshape(transpose(i), [4]).data as BigInt64Array)).toEqual([big, 3n, 2n, 4n]);
  });
});

// ---------------------------------------------------------------------------
// Linear algebra
// ---------------------------------------------------------------------------

describe("tensordot result dtype", () => {
  it("float32 with bool or uint8 stays float32, like NumPy", () => {
    const f = tensor([[1.5, 2.5]], { dtype: "float32" });
    const u = tensor([[2], [4]], { dtype: "uint8" });
    const out = tensordot(f, u, 1);
    expect(out.dtype).toBe("float32");
    expect(out.toArray()).toEqual([[13]]);
    const b = tensor([[1], [0]], { dtype: "bool" });
    const out2 = tensordot(b.reshape([1, 2]), f.reshape([2, 1]).astype("float32"), 1);
    expect(out2.dtype).toBe("float32");
    expect(out2.toArray()).toEqual([[1.5]]);
  });

  it("other mixes still give float64", () => {
    const f32 = tensor([1, 2], { dtype: "float32" });
    expect(tensordot(f32, tensor([1, 2], { dtype: "int32" }), 1).dtype).toBe("float64");
    expect(tensordot(f32, tensor([1, 2], { dtype: "float64" }), 1).dtype).toBe("float64");
    expect(tensordot(tensor([1, 2], { dtype: "int64" }), tensor([0.5, 2]), 1).dtype).toBe(
      "float64"
    );
    expect(
      tensordot(tensor([1, 2], { dtype: "int32" }), tensor([1, 2], { dtype: "uint8" }), 1).dtype
    ).toBe("int32");
  });
});

// ---------------------------------------------------------------------------
// Sorting and rolling
// ---------------------------------------------------------------------------

describe("sorting details", () => {
  it("descending argsort keeps ties in their original order, also N-D", () => {
    expect(argsort(tensor([1, 2, 2, 3]), -1, true).toArray()).toEqual([3, 1, 2, 0]);
    const m = tensor([
      [2, 1, 2, 1],
      [0, 0, 5, 5],
    ]);
    expect(argsort(m, 1, true).toArray()).toEqual([
      [0, 2, 1, 3],
      [2, 3, 0, 1],
    ]);
    expect(argsort(m, 0, true).toArray()).toEqual([
      [0, 0, 1, 1],
      [1, 1, 0, 0],
    ]);
  });

  it("sort orders -0 before +0 for short and long lanes alike", () => {
    const build = (n: number): Float64Array => {
      const v = new Float64Array(n);
      for (let i = 0; i < n; i++) v[i] = ((i * 2654435761) % 1000) - 500;
      v[3] = 0;
      v[4] = -0;
      v[5] = 0;
      v[6] = -0;
      return v;
    };
    for (const n of [64, 9000]) {
      const out = sort(tensor(build(n), { dtype: "float64" })).data as Float64Array;
      const firstZero = out.indexOf(0);
      const zeros = Array.from(out.slice(firstZero, firstZero + 4));
      expect(zeros.map((z) => (Object.is(z, -0) ? "-0" : "+0"))).toEqual(["-0", "-0", "+0", "+0"]);
      for (let i = 1; i < out.length; i++)
        expect(out[i] as number).toBeGreaterThanOrEqual(out[i - 1] as number);
    }
  });
});

describe("roll", () => {
  it("rolls strided input along one or several axes", () => {
    const m = tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      { dtype: "int32" }
    );
    expect(roll(transpose(m), 1, 0).toArray()).toEqual([
      [3, 6],
      [1, 4],
      [2, 5],
    ]);
    expect(roll(m, [1, -1], [0, 1]).toArray()).toEqual([
      [5, 6, 4],
      [2, 3, 1],
    ]);
    expect(roll(transpose(m), 2).toArray()).toEqual([
      [3, 6],
      [1, 4],
      [2, 5],
    ]);
  });
});
