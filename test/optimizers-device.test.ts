/**
 * On-device optimizer parity tests.
 *
 * Every optimizer that supports a device (kernel-backed) fast path composes its
 * update rule from device-dispatched tensor ops instead of reading parameters
 * back to the host. These tests run a tiny training loop on both the CPU and an
 * in-process fake `webgpu` kernel backend and assert the final weights match,
 * proving the device math is identical to the host update rule.
 *
 * The fake backend mirrors the one in test/device-dispatch.test.ts: a reference
 * KernelBackend over host Float32Arrays, so the whole stack above the kernels
 * (buffer storage, op dispatch, autograd, optimizer state) is exercised
 * deterministically without GPU hardware.
 */

import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type {
  BackendCapability,
  BackendInfo,
  BinaryKernelOp,
  DeviceBuffer,
  Im2ColParams,
  KernelBackend,
  KernelLayout,
  PoolKernelOp,
  ReduceKernelOp,
  TernaryKernelOp,
  UnaryKernelOp,
} from "../src/core";
import { DeviceError, registerBackend, resetConfig } from "../src/core";
import { GradTensor, parameter, tensor } from "../src/ndarray";
import {
  AdaDelta,
  Adagrad,
  Adam,
  Adamax,
  AdamW,
  ASGD,
  LAMB,
  LARS,
  LBFGS,
  Lion,
  Nadam,
  RAdam,
  RMSprop,
  Rprop,
  SGD,
  SparseAdam,
} from "../src/optim";

type FakeBuffer = DeviceBuffer & { data: Float32Array; freed: boolean };

const BINARY_FNS: Record<BinaryKernelOp, (x: number, y: number) => number> = {
  add: (x, y) => x + y,
  sub: (x, y) => x - y,
  mul: (x, y) => x * y,
  div: (x, y) => x / y,
  pow: (x, y) => x ** y,
  maximum: Math.max,
  minimum: Math.min,
};

const erfRef = (x: number): number => {
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
      t *
      Math.exp(-x * x);
  return x < 0 ? -y : y;
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
  expm1: (x) => Math.exp(x) - 1,
  log1p: (x) => Math.log(1 + x),
  softplus: (x) => (x > 20 ? x : Math.log(1 + Math.exp(x))),
};

function contiguousStridesRef(shape: readonly number[]): number[] {
  const s = new Array<number>(shape.length).fill(1);
  for (let i = shape.length - 2; i >= 0; i--) s[i] = (s[i + 1] ?? 1) * (shape[i + 1] ?? 1);
  return s;
}

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

  private wrap(data: Float32Array): FakeBuffer {
    this.allocated++;
    return { device: "webgpu", byteLength: data.byteLength, size: data.length, data, freed: false };
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

  upload(data: Float32Array): DeviceBuffer {
    return this.wrap(data.slice());
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

  fill(value: number, size: number): DeviceBuffer {
    return this.wrap(new Float32Array(Math.max(size, 1)).fill(Math.fround(value)));
  }

  binary(
    op: BinaryKernelOp,
    a: DeviceBuffer,
    aLayout: KernelLayout,
    b: DeviceBuffer,
    bLayout: KernelLayout,
    outShape: readonly number[]
  ): DeviceBuffer {
    const av = this.read(a, { ...aLayout, shape: outShape });
    const bv = this.read(b, { ...bLayout, shape: outShape });
    const fn = BINARY_FNS[op];
    const out = new Float32Array(av.length);
    for (let i = 0; i < out.length; i++) out[i] = Math.fround(fn(av[i] ?? 0, bv[i] ?? 0));
    return this.wrap(out);
  }

  unary(op: UnaryKernelOp, x: DeviceBuffer, layout: KernelLayout): DeviceBuffer {
    const xv = this.read(x, layout);
    const fn = UNARY_FNS[op];
    const out = new Float32Array(xv.length);
    for (let i = 0; i < out.length; i++) out[i] = Math.fround(fn(xv[i] ?? 0));
    return this.wrap(out);
  }

  matmul(
    a: DeviceBuffer,
    aLayout: KernelLayout,
    b: DeviceBuffer,
    bLayout: KernelLayout
  ): DeviceBuffer {
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
    return this.wrap(out);
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
    return this.wrap(new Float32Array([Math.fround(acc)]));
  }

  reduceAxis(
    op: ReduceKernelOp,
    x: DeviceBuffer,
    layout: KernelLayout,
    axis: number
  ): DeviceBuffer {
    const xv = this.read(x, layout);
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
    return this.wrap(out);
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
    return this.wrap(out);
  }

  im2col(x: DeviceBuffer, layout: KernelLayout, p: Im2ColParams): DeviceBuffer {
    const img = this.read(x, layout);
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

const fake = new FakeKernelBackend();

beforeAll(() => {
  registerBackend("webgpu", fake);
});

afterAll(() => {
  resetConfig();
});

const flat = (t: unknown): number[] => ([t].flat(Infinity) as number[]).flat();

// Starting weights and a fixed input for the toy regression loss.
const W0 = [
  [0.5, -0.3, 0.2],
  [0.1, 0.4, -0.6],
];
const Xd = [
  [1, 2, 3],
  [4, 5, 6],
  [-1, 0, 2],
];

type Opt = { zeroGrad(): void; step(closure?: () => number): unknown };

/**
 * Run `steps` optimization steps of the toy loss
 * `sum((w @ xᵀ)²)` and return the final flattened weights, on CPU or device.
 */
async function train(
  makeOpt: (p: GradTensor[]) => Opt,
  onDevice: boolean,
  steps: number
): Promise<number[]> {
  const w = parameter(onDevice ? await tensor(W0).to("webgpu") : tensor(W0));
  const opt = makeOpt([w]);
  const x = GradTensor.fromTensor(onDevice ? await tensor(Xd).to("webgpu") : tensor(Xd), {
    requiresGrad: false,
  });
  for (let s = 0; s < steps; s++) {
    opt.zeroGrad();
    w.matmul(x.transpose()).square().sum().backward();
    opt.step();
  }
  return flat(((onDevice ? await w.tensor.cpu() : w.tensor) as { toArray(): unknown }).toArray());
}

describe("optimizers: device (webgpu) update matches CPU", () => {
  const cases: Array<{
    name: string;
    make: (p: GradTensor[]) => Opt;
    steps?: number;
  }> = [
    {
      name: "SGD (momentum + nesterov + weightDecay)",
      make: (p) => new SGD(p, { lr: 0.001, momentum: 0.9, weightDecay: 0.01, nesterov: true }),
    },
    { name: "Adam (weightDecay)", make: (p) => new Adam(p, { lr: 0.01, weightDecay: 0.01 }) },
    { name: "Adam (amsgrad)", make: (p) => new Adam(p, { lr: 0.01, amsgrad: true }) },
    { name: "AdamW", make: (p) => new AdamW(p, { lr: 0.01, weightDecay: 0.02 }) },
    {
      name: "AdamW (amsgrad)",
      make: (p) => new AdamW(p, { lr: 0.01, weightDecay: 0.02, amsgrad: true }),
    },
    { name: "RMSprop", make: (p) => new RMSprop(p, { lr: 0.005, weightDecay: 0.01 }) },
    { name: "RMSprop (momentum)", make: (p) => new RMSprop(p, { lr: 0.005, momentum: 0.9 }) },
    {
      name: "RMSprop (centered + momentum)",
      make: (p) => new RMSprop(p, { lr: 0.005, momentum: 0.9, centered: true, weightDecay: 0.01 }),
    },
    {
      name: "Adagrad",
      make: (p) => new Adagrad(p, { lr: 0.05, weightDecay: 0.01, lrDecay: 0.01 }),
    },
    { name: "Adamax", make: (p) => new Adamax(p, { lr: 0.01, weightDecay: 0.01 }) },
    { name: "Nadam", make: (p) => new Nadam(p, { lr: 0.01, weightDecay: 0.01 }) },
    {
      name: "RAdam (fallback branch, beta2=0.999)",
      make: (p) => new RAdam(p, { lr: 0.01, weightDecay: 0.01 }),
    },
    {
      name: "RAdam (rectified branch, beta2=0.9)",
      make: (p) => new RAdam(p, { lr: 0.01, beta2: 0.9 }),
      steps: 8,
    },
    { name: "AdaDelta", make: (p) => new AdaDelta(p, { lr: 1.0, rho: 0.9, weightDecay: 0.01 }) },
    { name: "ASGD", make: (p) => new ASGD(p, { lr: 0.01, weightDecay: 0.01 }) },
    { name: "Rprop", make: (p) => new Rprop(p, { lr: 0.01 }) },
    { name: "Lion", make: (p) => new Lion(p, { lr: 0.01, weightDecay: 0.01 }) },
    { name: "LAMB", make: (p) => new LAMB(p, { lr: 0.01, weightDecay: 0.02 }) },
    { name: "LARS", make: (p) => new LARS(p, { lr: 0.05, momentum: 0.9, weightDecay: 1e-4 }) },
  ];

  for (const { name, make, steps } of cases) {
    it(`${name} trajectory matches host`, async () => {
      const n = steps ?? 5;
      const ref = await train(make, false, n);
      const got = await train(make, true, n);
      expect(got.length).toBe(ref.length);
      for (let i = 0; i < ref.length; i++) {
        expect(got[i]).toBeCloseTo(ref[i] ?? 0, 4);
      }
    });
  }
});

describe("optimizers that cannot run on device throw clearly", () => {
  it("SparseAdam throws a DeviceError on device parameters", async () => {
    const w = parameter(await tensor(W0).to("webgpu"));
    const x = GradTensor.fromTensor(await tensor(Xd).to("webgpu"), { requiresGrad: false });
    w.matmul(x.transpose()).square().sum().backward();
    const opt = new SparseAdam([w], { lr: 0.01 });
    expect(() => opt.step()).toThrow(DeviceError);
    expect(() => opt.step()).toThrow(/cpu/i);
  });

  it("LBFGS throws a DeviceError on device parameters", async () => {
    const w = parameter(await tensor(W0).to("webgpu"));
    const x = GradTensor.fromTensor(await tensor(Xd).to("webgpu"), { requiresGrad: false });
    const opt = new LBFGS([w], { lr: 1 });
    const closure = () => {
      opt.zeroGrad();
      const loss = w.matmul(x.transpose()).square().sum();
      loss.backward();
      return 0;
    };
    expect(() => opt.step(closure)).toThrow(DeviceError);
  });
});
