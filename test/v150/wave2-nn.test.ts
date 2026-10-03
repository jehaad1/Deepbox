/**
 * Wave 2 regression tests for the nn module group: gradient clipping, Module
 * helpers, Linear input casts, BatchNorm on a device, padding and dropout dtypes,
 * Transformer stack registration, SpectralNorm over every weight-bearing layer,
 * InstanceNorm accessors and the nn barrel exports.
 */

import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
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
} from "../../src/core";
import { DeviceError, registerBackend, resetConfig } from "../../src/core";
import { GradTensor, parameter, randn, type Tensor, tensor, transpose } from "../../src/ndarray";
import type {
  BinaryCrossEntropyWithLogitsOptions,
  CrossEntropyLossOptions,
  CtcLossOptions,
  EmbeddingBagMode,
  LossReduction,
  NllLossOptions,
  RNNNonlinearity,
} from "../../src/nn";
import {
  AlphaDropout,
  BatchNorm1d,
  Conv1d,
  Conv2d,
  Conv3d,
  ConvTranspose1d,
  ConvTranspose2d,
  clip_grad_norm_,
  clip_grad_value_,
  Dropout,
  Dropout2d,
  Embedding,
  GRU,
  InstanceNorm2d,
  Linear,
  LSTM,
  Module,
  ModuleDict,
  MultiheadAttention,
  ReflectionPad2d,
  RNN,
  Sequential,
  SpectralNorm,
  TransformerDecoder,
  TransformerDecoderLayer,
  TransformerEncoder,
  TransformerEncoderLayer,
  ZeroPad2d,
} from "../../src/nn";

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
      name: "Wave2 nn fake kernel backend",
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

  private unsupported(op: string): never {
    throw new DeviceError(`fake: ${op} is not supported`);
  }

  im2col(_x: DeviceBuffer, _layout: KernelLayout, _p: Im2ColParams): DeviceBuffer {
    return this.unsupported("im2col");
  }

  col2im(_cols: DeviceBuffer, _p: Im2ColParams): DeviceBuffer {
    return this.unsupported("col2im");
  }

  pool2d(
    _x: DeviceBuffer,
    _layout: KernelLayout,
    _op: PoolKernelOp,
    _p: Im2ColParams
  ): DeviceBuffer {
    return this.unsupported("pool2d");
  }

  pool2dBackward(
    _x: DeviceBuffer,
    _xLayout: KernelLayout,
    _gradOut: DeviceBuffer,
    _op: PoolKernelOp,
    _p: Im2ColParams
  ): DeviceBuffer {
    return this.unsupported("pool2dBackward");
  }
}

const fake = new FakeKernelBackend();

beforeAll(() => {
  registerBackend("webgpu", fake);
});

afterAll(() => {
  resetConfig();
});

const flat = (t: Tensor | GradTensor): number[] => {
  const raw = GradTensor.isGradTensor(t) ? t.tensor : t;
  return ([raw.toArray()].flat(Infinity) as number[]).map(Number);
};

describe("clip_grad_norm_ / clip_grad_value_ on strided and device gradients", () => {
  it("reads a transposed (strided) gradient in logical order", () => {
    const base = tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      { dtype: "float64" }
    );
    const p = parameter(
      tensor(
        [
          [0, 0],
          [0, 0],
          [0, 0],
        ],
        { dtype: "float64" }
      )
    );
    p.setGrad(transpose(base));
    const norm = clip_grad_norm_([p], 1);
    expect(norm).toBeCloseTo(Math.sqrt(91), 12);
    const coef = 1 / (Math.sqrt(91) + 1e-6);
    // The view [[1,4],[2,5],[3,6]] scaled by coef, read back in logical order.
    const expected = [1, 4, 2, 5, 3, 6].map((v) => v * coef);
    const got = flat(p.grad as Tensor);
    for (let i = 0; i < expected.length; i++) {
      expect(got[i]).toBeCloseTo(expected[i] as number, 12);
    }
  });

  it("clips a strided gradient by value", () => {
    const base = tensor(
      [
        [1, -8, 3],
        [-4, 5, 20],
      ],
      { dtype: "float64" }
    );
    const p = parameter(
      tensor(
        [
          [0, 0],
          [0, 0],
          [0, 0],
        ],
        { dtype: "float64" }
      )
    );
    p.setGrad(transpose(base));
    clip_grad_value_([p], 4);
    expect(flat(p.grad as Tensor)).toEqual([1, -4, -4, 4, 3, 4]);
  });

  it("scales and clamps a tied parameter once", () => {
    const shared = parameter(tensor([3, 4], { dtype: "float64" }));
    shared.setGrad(tensor([3, 4], { dtype: "float64" }));
    expect(clip_grad_norm_([shared, shared], 1)).toBeCloseTo(5, 12);
    const c = 1 / (5 + 1e-6);
    expect(flat(shared.grad as Tensor)[0]).toBeCloseTo(3 * c, 12);
    shared.setGrad(tensor([10, -10], { dtype: "float64" }));
    clip_grad_value_([shared, shared], 2);
    expect(flat(shared.grad as Tensor)).toEqual([2, -2]);
  });

  it("throws DeviceError for device gradients before changing any gradient", async () => {
    const host = parameter(tensor([3, 4], { dtype: "float32" }));
    host.setGrad(tensor([3, 4], { dtype: "float32" }));
    const dev = parameter(await tensor([1, 2], { dtype: "float32" }).to("webgpu"));
    dev.setGrad(await tensor([1, 2], { dtype: "float32" }).to("webgpu"));
    expect(() => clip_grad_norm_([host, dev], 0.5)).toThrow(DeviceError);
    expect(() => clip_grad_value_([host, dev], 0.5)).toThrow(DeviceError);
    expect(flat(host.grad as Tensor)).toEqual([3, 4]);
  });
});

describe("Module helpers", () => {
  it("parameters() and namedParameters() report a shared module once", () => {
    const shared = new Linear(2, 2);
    const model = new Sequential(shared, shared);
    expect([...model.parameters()]).toHaveLength(2);
    const names = [...model.namedParameters()].map(([n]) => n);
    expect(names).toEqual(["0.weight", "0.bias"]);
    expect([...model.namedParameters("", true, false)]).toHaveLength(4);
  });

  it("freezeParameters resolves names held in a ModuleDict without replacing the parameter", () => {
    const fc = new Linear(2, 2);
    const holder = new ModuleDict({ fc });
    const weight = [...fc.parameters()][0] as GradTensor;
    holder.freezeParameters(["fc.weight"]);
    expect([...fc.parameters()][0]).toBe(weight);
    expect(weight.requiresGrad).toBe(false);
    expect([...holder.parameters()]).toContain(weight);
    holder.unfreezeParameters(["fc.weight"]);
    expect(weight.requiresGrad).toBe(true);
    expect(() => holder.freezeParameters(["fc.nope"])).toThrow(/Unknown parameter name/);
  });

  it("stateDict and loadStateDict round-trip", () => {
    const a = new Sequential(new Linear(3, 2), new Linear(2, 1));
    const b = new Sequential(new Linear(3, 2), new Linear(2, 1));
    b.loadStateDict(a.stateDict());
    const x = tensor([[1, 2, 3]]);
    expect(flat(b.forward(x))).toEqual(flat(a.forward(x)));
  });

  it("the Module class comment carries a well-formed @see tag", () => {
    const text = readFileSync("src/nn/module/Module.ts", "utf8");
    expect(text).toContain(
      "@see {@link https://deepbox.dev/docs/nn-module | Deepbox Module & Sequential}"
    );
    expect(text).not.toMatch(/^ \* \{ https/m);
  });

  it("parameters() and modules() terminate on a module cycle", () => {
    class Node extends Module {
      link(other: Node): void {
        this.registerModule("other", other);
      }
      forward(x: Tensor): Tensor {
        return x;
      }
    }
    const a = new Node();
    const b = new Node();
    a.link(b);
    b.link(a);
    expect([...a.parameters()]).toHaveLength(0);
    expect([...a.modules()]).toHaveLength(2);
  });
});

describe("Linear input casts", () => {
  const lin = new Linear(3, 2);
  const logical = [
    [1, 4, 7],
    [2, 5, 8],
  ];
  const stored = (dtype: "float64" | "int64"): Tensor =>
    tensor(
      [
        [1, 2],
        [4, 5],
        [7, 8],
      ],
      { dtype }
    );
  const reference = flat(lin.forward(tensor(logical, { dtype: "float32" })));

  it("honours strides of a transposed float64 view", () => {
    const view = transpose(stored("float64"));
    expect(view.strides).not.toEqual([3, 1]);
    const out = lin.forward(view);
    expect(out.dtype).toBe("float32");
    expect(flat(out)).toEqual(reference);
  });

  it("converts int64 (BigInt) input by value", () => {
    const out = lin.forward(tensor(logical, { dtype: "int64" }));
    expect(out.dtype).toBe("float32");
    expect(flat(out)).toEqual(reference);
  });

  it("converts a strided int64 view", () => {
    expect(flat(lin.forward(transpose(stored("int64"))))).toEqual(reference);
  });

  it("keeps the autograd link when casting a GradTensor", () => {
    const x = parameter(tensor(logical, { dtype: "float64" }));
    const y = lin.forward(x);
    expect(y.dtype).toBe("float32");
    y.sum().backward();
    expect(x.grad).not.toBeNull();
    expect(x.grad?.shape).toEqual([2, 3]);
  });
});

describe("BatchNorm buffers stay on the device after a training step", () => {
  it("keeps running_mean and running_var on webgpu", async () => {
    const bn = new BatchNorm1d(3);
    await bn.to("webgpu");
    const x = await tensor(
      [
        [1, 2, 3],
        [4, 5, 7],
        [0, 1, 2],
        [3, 3, 3],
      ],
      { dtype: "float32" }
    ).to("webgpu");
    for (let step = 0; step < 2; step++) {
      bn.forward(GradTensor.fromTensor(x, { requiresGrad: false }));
      for (const [, buffer] of bn.namedBuffers()) expect(buffer.device).toBe("webgpu");
    }
    const mean = await [...bn.buffers()][0]?.cpu();
    // Momentum 0.1, two updates from zero with the same batch mean.
    const batchMean = [2, 2.75, 3.75];
    const expected = batchMean.map((m) => m * (1 - 0.9 * 0.9));
    const got = flat(mean as Tensor);
    for (let i = 0; i < 3; i++) expect(got[i]).toBeCloseTo(expected[i] as number, 5);
  });
});

describe("padding and dropout keep a float32 graph float32", () => {
  const cases: Array<[string, Module, number[]]> = [
    ["ZeroPad2d", new ZeroPad2d(1), [1, 2, 3, 3]],
    ["ReflectionPad2d", new ReflectionPad2d(1), [1, 2, 3, 3]],
    ["Dropout", new Dropout(0.5), [4, 6]],
    ["Dropout2d", new Dropout2d(0.5), [2, 3, 3, 3]],
    ["AlphaDropout", new AlphaDropout(0.5), [4, 6]],
  ];
  for (const [name, layer, shape] of cases) {
    it(`${name} returns float32 outputs and gradients`, () => {
      const a = parameter(randn(shape, { dtype: "float32" }));
      const w = parameter(randn(shape, { dtype: "float32" }));
      const out = layer.forward(a.mul(w)) as GradTensor;
      expect(out.dtype).toBe("float32");
      out.sum().backward();
      expect(a.grad?.dtype).toBe("float32");
      expect(w.grad?.dtype).toBe("float32");
    });
  }
});

describe("Transformer stacks register a real ModuleList", () => {
  it("TransformerEncoder exposes a `layers` child with numbered children", () => {
    const enc = new TransformerEncoder(new TransformerEncoderLayer(4, 2, 8), 3);
    const names = [...enc.namedModules()].map(([n]) => n);
    expect(names).toContain("layers");
    expect(names).toContain("layers.0");
    expect(names).toContain("layers.2");
    expect([...enc.namedChildren()].map(([n]) => n)).toEqual(["layers"]);
    const keys = Object.keys(enc.stateDict().parameters);
    expect(keys).toContain("layers.0.self_attn.out_proj_weight");
    expect(keys).toContain("layers.2.linear2.bias");
  });

  it("TransformerDecoder does the same and still runs", () => {
    const dec = new TransformerDecoder(new TransformerDecoderLayer(4, 2, 8), 2);
    expect([...dec.namedChildren()].map(([n]) => n)).toEqual(["layers"]);
    const keys = Object.keys(dec.stateDict().parameters);
    expect(keys).toContain("layers.1.multihead_attn.out_proj_weight");
    const out = dec.forward(randn([1, 3, 4]), randn([1, 5, 4]));
    expect(out.shape).toEqual([1, 3, 4]);
  });

  it("freezes a nested layer parameter by its dotted path", () => {
    const enc = new TransformerEncoder(new TransformerEncoderLayer(4, 2, 8), 2);
    enc.freezeParameters(["layers.1.linear1.weight"]);
    const frozen = [...enc.namedParameters()].filter(([, p]) => !p.requiresGrad);
    expect(frozen.map(([n]) => n)).toEqual(["layers.1.linear1.weight"]);
    expect(() => enc.freezeParameters(["layers.7.linear1.weight"])).toThrow(/Unknown parameter/);
  });

  it("state dicts round-trip between two stacks", () => {
    const a = new TransformerEncoder(new TransformerEncoderLayer(4, 2, 8), 2);
    const b = new TransformerEncoder(new TransformerEncoderLayer(4, 2, 8), 2);
    b.loadStateDict(a.stateDict());
    a.eval();
    b.eval();
    const x = randn([1, 3, 4]);
    const ya = flat(a.forward(x));
    const yb = flat(b.forward(x));
    for (let i = 0; i < ya.length; i++) expect(yb[i]).toBeCloseTo(ya[i] as number, 6);
  });
});

describe("SpectralNorm substitutes the weight in every weight-bearing layer", () => {
  type Case = { name: string; make: () => Module; weight: string; input: () => Tensor };
  const cases: Case[] = [
    { name: "Linear", make: () => new Linear(4, 3), weight: "weight", input: () => randn([2, 4]) },
    {
      name: "Conv1d",
      make: () => new Conv1d(2, 3, 3),
      weight: "weight",
      input: () => randn([1, 2, 6]),
    },
    {
      name: "Conv2d",
      make: () => new Conv2d(1, 2, 3),
      weight: "weight",
      input: () => randn([1, 1, 5, 5]),
    },
    {
      name: "Conv3d",
      make: () => new Conv3d(1, 2, 2),
      weight: "weight",
      input: () => randn([1, 1, 3, 3, 3]),
    },
    {
      name: "ConvTranspose1d",
      make: () => new ConvTranspose1d(1, 2, 3),
      weight: "weight",
      input: () => randn([1, 1, 4]),
    },
    {
      name: "ConvTranspose2d",
      make: () => new ConvTranspose2d(1, 2, 3),
      weight: "weight",
      input: () => randn([1, 1, 4, 4]),
    },
    {
      name: "GRU input weight",
      make: () => new GRU(3, 4),
      weight: "weight_ih_l0",
      input: () => randn([2, 5, 3]),
    },
    {
      name: "GRU hidden weight",
      make: () => new GRU(3, 4),
      weight: "weight_hh_l0",
      input: () => randn([2, 5, 3]),
    },
    {
      name: "LSTM",
      make: () => new LSTM(3, 4),
      weight: "weight_ih_l0",
      input: () => randn([2, 5, 3]),
    },
    {
      name: "RNN",
      make: () => new RNN(3, 4),
      weight: "weight_ih_l0",
      input: () => randn([2, 5, 3]),
    },
    {
      name: "MultiheadAttention query weight",
      make: () => new MultiheadAttention(4, 2),
      weight: "in_proj_weight_q",
      input: () => randn([2, 3, 4]),
    },
    {
      name: "MultiheadAttention output weight",
      make: () => new MultiheadAttention(4, 2),
      weight: "out_proj_weight",
      input: () => randn([2, 3, 4]),
    },
    {
      name: "TransformerEncoderLayer",
      make: () => new TransformerEncoderLayer(4, 2, 8),
      weight: "linear1.weight",
      input: () => randn([2, 3, 4]),
    },
    {
      name: "TransformerEncoder (nested ModuleList)",
      make: () => new TransformerEncoder(new TransformerEncoderLayer(4, 2, 8), 2),
      weight: "layers.1.linear2.weight",
      input: () => randn([2, 3, 4]),
    },
  ];
  for (const c of cases) {
    it(`${c.name}: forward equals the module run with weight / sigma`, () => {
      const module = c.make();
      module.eval();
      const x = c.input();
      const sn = new SpectralNorm(module, c.weight);
      sn.eval();
      const viaWrapper = flat(sn.forward(x));
      const sigma = sn.spectralNormValue;
      const entry = [...module.namedParameters()].find(([n]) => n === c.weight);
      const data = entry?.[1].tensor.data as Float32Array;
      for (let i = 0; i < data.length; i++) data[i] = (data[i] as number) / sigma;
      const direct = flat(module.forward(x as never) as Tensor);
      expect(direct).toHaveLength(viaWrapper.length);
      for (let i = 0; i < direct.length; i++) {
        expect(viaWrapper[i]).toBeCloseTo(direct[i] as number, 5);
      }
    });
  }
});

describe("InstanceNorm exposes weight and bias", () => {
  it("returns the registered group_norm parameters", () => {
    const inorm = new InstanceNorm2d(3);
    const named = new Map(inorm.namedParameters());
    expect(inorm.weight).toBe(named.get("group_norm.weight"));
    expect(inorm.bias).toBe(named.get("group_norm.bias"));
    expect(inorm.weight?.shape).toEqual([3]);
  });

  it("is undefined without affine", () => {
    const inorm = new InstanceNorm2d(3, { affine: false });
    expect(inorm.weight).toBeUndefined();
    expect(inorm.bias).toBeUndefined();
  });
});

describe("nn barrel and source conventions", () => {
  it("re-exports the option and alias types", () => {
    const nonlinearity: RNNNonlinearity = "relu";
    const mode: EmbeddingBagMode = "mean";
    const reduction: LossReduction = "none";
    const ce: CrossEntropyLossOptions = { reduction };
    const bce: BinaryCrossEntropyWithLogitsOptions = { reduction };
    const nll: NllLossOptions = { reduction };
    const ctc: CtcLossOptions = { blank: 0 };
    expect([nonlinearity, mode, ce.reduction, bce.reduction, nll.reduction, ctc.blank]).toEqual([
      "relu",
      "mean",
      "none",
      "none",
      "none",
      0,
    ]);
    expect(new RNN(2, 3, { nonlinearity }).forward(randn([1, 2, 2]))).toBeDefined();
    expect(new Embedding(4, 2).numEmbeddings).toBe(4);
  });

  it("every nn source file links to the Deepbox docs with a @see tag", () => {
    const files: string[] = [];
    const walk = (dir: string): void => {
      for (const entry of readdirSync(dir)) {
        const full = join(dir, entry);
        if (statSync(full).isDirectory()) walk(full);
        else if (full.endsWith(".ts")) files.push(full);
      }
    };
    walk("src/nn");
    expect(files.length).toBeGreaterThan(20);
    for (const file of files) {
      expect(readFileSync(file, "utf8"), file).toMatch(/@see \{@link https:\/\/deepbox\.dev\/docs/);
    }
  });
});
