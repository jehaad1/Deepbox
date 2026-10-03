/**
 * v1.5.0 regression tests for src/nn: Trainer, gradient clipping, containers,
 * weight initialization and training utilities.
 *
 * Reference values come from PyTorch 2.12 (clip_grad_norm_, init.sparse_,
 * init.calculate_gain) and NumPy 2.4 (overflow-safe norms).
 */
import { describe, expect, it } from "vitest";
import {
  DataValidationError,
  DTypeError,
  IndexError,
  InvalidParameterError,
  ShapeError,
} from "../../src/core";
import {
  type AnyTensor,
  GradTensor,
  parameter,
  type Tensor,
  tensor,
  transpose,
  zeros,
} from "../../src/ndarray";
import { roundToFloat16 } from "../../src/ndarray/tensor/float16";
import {
  calculateGain,
  clipGradNorm_,
  clipGradValue_,
  constant_,
  EarlyStopping,
  eye_,
  kaiming_normal_,
  Linear,
  ModelCheckpoint,
  Module,
  ModuleDict,
  ModuleList,
  mseLoss,
  normal_,
  orthogonal_,
  ParameterDict,
  ParameterList,
  ReLU,
  Sequential,
  sparse_,
  Trainer,
  trunc_normal_,
  uniform_,
  xavier_uniform_,
} from "../../src/nn";
import { calculateGain as calculateGainInit, makeRng } from "../../src/nn/init";
import { Adam } from "../../src/optim";
import { setSeed } from "../../src/random";

function flat(t: Tensor): number[] {
  const out: number[] = [];
  const walk = (v: unknown): void => {
    if (Array.isArray(v)) v.forEach(walk);
    else out.push(Number(v as number | bigint));
  };
  walk(t.toArray());
  return out;
}

function gradParam(values: number[], grad: number[], dtype: "float64" | "float32" = "float64") {
  const p = GradTensor.fromTensor(tensor(values, { dtype }), { requiresGrad: true });
  p.setGrad(tensor(grad, { dtype }));
  return p;
}

// ─────────────────────────────────────────────────────────────────────────────
// clip.ts
// ─────────────────────────────────────────────────────────────────────────────

describe("clip_grad_norm_ (v1.5.0)", () => {
  it("matches torch for the global L2 norm and scaling", () => {
    // torch: clip_grad_norm_([a, b], 1.0) with grads [3, 4] and [12] -> 13.0
    const a = gradParam([0, 0], [3, 4]);
    const b = gradParam([0], [12]);
    const norm = clipGradNorm_([a, b], 1.0);
    expect(norm).toBeCloseTo(13, 12);
    const coef = 1 / (13 + 1e-6);
    expect(flat(a.grad as Tensor)[0]).toBeCloseTo(3 * coef, 12);
    expect(flat(a.grad as Tensor)[1]).toBeCloseTo(4 * coef, 12);
    expect(flat(b.grad as Tensor)[0]).toBeCloseTo(12 * coef, 12);
    // torch float32 values 0.23076921701431274, 0.307692289352417, 0.923076868057251
    expect(flat(a.grad as Tensor)[0]).toBeCloseTo(0.23076921701431274, 6);
    expect(flat(b.grad as Tensor)[0]).toBeCloseTo(0.923076868057251, 6);
  });

  it("matches torch for p-norms of the concatenated gradient", () => {
    // grads [3, -4] and [12]; torch norms: p=1 -> 19, p=3 -> 12.207056, p=0.5 -> 51.784614
    const cases: Array<[number, number]> = [
      [1, 19],
      [3, 12.207056045532227],
      [0.5, 51.78461456298828],
      [Number.POSITIVE_INFINITY, 12],
    ];
    for (const [p, expected] of cases) {
      const a = gradParam([0, 0], [3, -4]);
      const b = gradParam([0], [12]);
      expect(clipGradNorm_([a, b], 1000, p)).toBeCloseTo(expected, 5);
    }
  });

  it("does not overflow or underflow for extreme gradient magnitudes", () => {
    const big = gradParam([0, 0], [1e200, 1e200]);
    const bigNorm = clipGradNorm_([big], Number.POSITIVE_INFINITY);
    expect(bigNorm / 1e200).toBeCloseTo(Math.SQRT2, 12);

    const small = gradParam([0, 0], [1e-200, 1e-200]);
    const smallNorm = clipGradNorm_([small], 1.0);
    expect(smallNorm / 1e-200).toBeCloseTo(Math.SQRT2, 12);
    // Already below maxNorm: untouched.
    expect(flat(small.grad as Tensor)).toEqual([1e-200, 1e-200]);

    // Clipping a huge gradient yields a unit-norm result instead of NaN.
    const clipped = clipGradNorm_([big], 1.0);
    expect(clipped / 1e200).toBeCloseTo(Math.SQRT2, 12);
    const g = flat(big.grad as Tensor);
    expect(Math.hypot(g[0] as number, g[1] as number)).toBeCloseTo(1, 6);
  });

  it("clips gradients stored in a non-contiguous view", () => {
    const base = tensor([
      [3, 0],
      [4, 0],
    ]);
    const view = transpose(base); // shape [2, 2], non-contiguous
    const p = GradTensor.fromTensor(
      tensor([
        [0, 0],
        [0, 0],
      ]),
      { requiresGrad: true }
    );
    p.setGrad(view);
    const norm = clipGradNorm_([p], 1.0);
    expect(norm).toBeCloseTo(5, 12);
    const coef = 1 / (5 + 1e-6);
    const got = flat(view);
    expect(got[0]).toBeCloseTo(3 * coef, 6);
    expect(got[1]).toBeCloseTo(4 * coef, 6);
    expect(got.slice(2)).toEqual([0, 0]);
  });

  it("counts a parameter listed twice once", () => {
    const p = gradParam([0, 0], [3, 4]);
    const norm = clipGradNorm_([p, p], 1.0);
    expect(norm).toBeCloseTo(5, 12);
    const coef = 1 / (5 + 1e-6);
    // Scaled once, not twice.
    expect(flat(p.grad as Tensor)[0]).toBeCloseTo(3 * coef, 12);
  });

  it("accepts a single GradTensor", () => {
    const p = gradParam([0, 0], [3, 4]);
    expect(clipGradNorm_(p, 100)).toBeCloseTo(5, 12);
  });

  it("validates arguments and optionally rejects non-finite norms", () => {
    const p = gradParam([0, 0], [3, 4]);
    expect(() => clipGradNorm_([p], Number.NaN)).toThrow(InvalidParameterError);
    expect(() => clipGradNorm_([p], 1, 0)).toThrow(InvalidParameterError);
    expect(() => clipGradNorm_([p], 1, -1)).toThrow(InvalidParameterError);
    expect(() => clipGradNorm_([p], 1, Number.NaN)).toThrow(InvalidParameterError);

    const bad = gradParam([0, 0], [Number.NaN, 1]);
    expect(() => clipGradNorm_([bad], 1, 2, true)).toThrow(DataValidationError);
    expect(Number.isNaN(clipGradNorm_([bad], 1))).toBe(true);
    const inf = gradParam([0, 0], [Number.POSITIVE_INFINITY, 1]);
    expect(() => clipGradNorm_([inf], 1, 2, true)).toThrow(DataValidationError);
  });

  it("rejects integer gradients instead of writing garbage", () => {
    const p = GradTensor.fromTensor(tensor([0, 0], { dtype: "int32" }), { requiresGrad: true });
    p.setGrad(tensor([30, 40], { dtype: "int32" }));
    expect(() => clipGradNorm_([p], 1)).toThrow(DTypeError);
  });

  it("rounds clipped float16 gradients to representable values", () => {
    const p = GradTensor.fromTensor(tensor([0, 0], { dtype: "float16" }), { requiresGrad: true });
    p.setGrad(tensor([3, 4], { dtype: "float16" }));
    clipGradNorm_([p], 1.0);
    const g = flat(p.grad as Tensor);
    expect(g[0]).toBe(roundToFloat16(3 / (5 + 1e-6)));
    expect(g[1]).toBe(roundToFloat16(4 / (5 + 1e-6)));
  });
});

describe("clip_grad_value_ (v1.5.0)", () => {
  it("clamps non-contiguous gradients and validates clipValue", () => {
    const view = transpose(
      tensor([
        [10, -10],
        [5, 1],
      ])
    );
    const p = GradTensor.fromTensor(
      tensor([
        [0, 0],
        [0, 0],
      ]),
      { requiresGrad: true }
    );
    p.setGrad(view);
    clipGradValue_([p], 3);
    expect(flat(view)).toEqual([3, 3, -3, 1]);
    expect(() => clipGradValue_([p], Number.NaN)).toThrow(InvalidParameterError);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// init.ts
// ─────────────────────────────────────────────────────────────────────────────

describe("weight initialization (v1.5.0)", () => {
  it("sparse_ zeroes ceil(sparsity * rows) entries per column like torch", () => {
    // torch.nn.init.sparse_(w[10, 6], 0.25) -> 3 zeros per column; w[8, 3], 0.3 -> 3 zeros
    for (const [rows, cols, sparsity, zerosPerCol] of [
      [10, 6, 0.25, 3],
      [8, 3, 0.3, 3],
      [4, 5, 0.5, 2],
      [5, 4, 0, 0],
      [5, 4, 1, 5],
    ] as const) {
      const t = tensor(Array.from({ length: rows }, () => new Array<number>(cols).fill(1)));
      sparse_(t, sparsity, 1);
      const values = flat(t);
      for (let j = 0; j < cols; j++) {
        let z = 0;
        for (let i = 0; i < rows; i++) if (values[i * cols + j] === 0) z++;
        expect(z).toBe(zerosPerCol);
      }
    }
  });

  it("sparse_ validates its arguments", () => {
    const t = tensor([
      [0, 0],
      [0, 0],
    ]);
    expect(() => sparse_(t, 1.5)).toThrow(InvalidParameterError);
    expect(() => sparse_(t, -0.1)).toThrow(InvalidParameterError);
    expect(() => sparse_(t, 0.5, -1)).toThrow(InvalidParameterError);
  });

  it("orthogonal_ gives orthonormal rows, columns and flattened conv kernels", () => {
    setSeed(7);
    const tall = zeros([64, 32], { dtype: "float64" });
    orthogonal_(tall);
    const tv = flat(tall);
    for (let a = 0; a < 32; a++) {
      for (let b = 0; b < 32; b++) {
        let dot = 0;
        for (let i = 0; i < 64; i++) dot += (tv[i * 32 + a] as number) * (tv[i * 32 + b] as number);
        expect(Math.abs(dot - (a === b ? 1 : 0))).toBeLessThan(1e-12);
      }
    }

    const wide = zeros([8, 3, 3, 3], { dtype: "float64" }); // flattened to [8, 27]
    orthogonal_(wide, 2);
    const wv = flat(wide);
    for (let a = 0; a < 8; a++) {
      for (let b = 0; b < 8; b++) {
        let dot = 0;
        for (let j = 0; j < 27; j++) dot += (wv[a * 27 + j] as number) * (wv[b * 27 + j] as number);
        expect(Math.abs(dot - (a === b ? 4 : 0))).toBeLessThan(1e-12);
      }
    }
  });

  it("orthogonal_ writes through a non-contiguous view", () => {
    setSeed(11);
    const base = zeros([4, 2], { dtype: "float64" });
    const view = transpose(base); // [2, 4] view over the [4, 2] buffer
    orthogonal_(view);
    const v = flat(view);
    const dot = (a: number, b: number) => {
      let s = 0;
      for (let j = 0; j < 4; j++) s += (v[a * 4 + j] as number) * (v[b * 4 + j] as number);
      return s;
    };
    expect(dot(0, 0)).toBeCloseTo(1, 12);
    expect(dot(1, 1)).toBeCloseTo(1, 12);
    expect(dot(0, 1)).toBeCloseTo(0, 12);
  });

  it("fills non-contiguous views in logical order and accepts GradTensor", () => {
    const base = zeros([3, 2]);
    const view = transpose(base);
    constant_(view, 7);
    expect(flat(base)).toEqual([7, 7, 7, 7, 7, 7]);

    const p = parameter([
      [0, 0],
      [0, 0],
    ]);
    const returned = constant_(p, 2);
    expect(returned).toBe(p);
    expect(flat(p.tensor)).toEqual([2, 2, 2, 2]);
    xavier_uniform_(p);
    expect(flat(p.tensor).every((x) => x !== 2)).toBe(true);
  });

  it("random fills reject non-floating dtypes instead of truncating", () => {
    const ints = tensor([0, 0, 0], { dtype: "int32" });
    expect(() => uniform_(ints, -1, 1)).toThrow(DTypeError);
    expect(() => normal_(ints)).toThrow(DTypeError);
    expect(() =>
      orthogonal_(
        tensor(
          [
            [0, 0],
            [0, 0],
          ],
          { dtype: "int32" }
        )
      )
    ).toThrow(DTypeError);
  });

  it("constant_ handles integer, bool, int64 and half-precision dtypes", () => {
    expect(flat(constant_(tensor([0, 0], { dtype: "int64" }), 3.9))).toEqual([3, 3]);
    expect(flat(constant_(tensor([0, 0], { dtype: "bool" }), 5))).toEqual([1, 1]);
    expect(flat(constant_(tensor([0, 0], { dtype: "int32" }), -2.7))).toEqual([-2, -2]);
    expect(flat(constant_(tensor([0], { dtype: "float16" }), 0.1))[0]).toBe(roundToFloat16(0.1));
    expect(() => constant_(tensor([0], { dtype: "int64" }), Number.NaN)).toThrow(
      InvalidParameterError
    );
    expect(() => constant_(tensor(["a"]), 1)).toThrow(DTypeError);
  });

  it("random fills round half-precision values", () => {
    setSeed(3);
    const t = tensor(new Array<number>(16).fill(0), { dtype: "float16" });
    uniform_(t, -1, 1);
    for (const v of flat(t)) expect(v).toBe(roundToFloat16(v));
  });

  it("validates distribution parameters", () => {
    const t = zeros([2, 2]);
    expect(() => uniform_(t, 1, 0)).toThrow(InvalidParameterError);
    expect(() => uniform_(t, Number.NaN, 1)).toThrow(InvalidParameterError);
    expect(() => normal_(t, 0, -1)).toThrow(InvalidParameterError);
    expect(() => xavier_uniform_(t, -1)).toThrow(InvalidParameterError);
    expect(() => kaiming_normal_(t, 0, "fan_sideways" as "fan_in")).toThrow(InvalidParameterError);
    expect(() => kaiming_normal_(t, Number.NaN)).toThrow(InvalidParameterError);
  });

  it("zero-element tensors are a no-op instead of dividing by zero", () => {
    const empty = zeros([0, 3]);
    expect(() => kaiming_normal_(empty)).not.toThrow();
    expect(() => xavier_uniform_(empty)).not.toThrow();
    expect(() => orthogonal_(empty)).not.toThrow();
    expect(() => sparse_(empty)).not.toThrow();
  });

  it("kaiming_normal_ and xavier_uniform_ follow the PyTorch scale", () => {
    setSeed(2024);
    const w = zeros([512, 256]);
    kaiming_normal_(w, 0, "fan_in", "relu"); // std = sqrt(2 / 256)
    const v = flat(w);
    const mean = v.reduce((a, b) => a + b, 0) / v.length;
    const sd = Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / v.length);
    expect(sd / Math.sqrt(2 / 256)).toBeCloseTo(1, 1);

    const u = zeros([512, 256]);
    xavier_uniform_(u); // bound = sqrt(6 / (512 + 256))
    const bound = Math.sqrt(6 / 768);
    const maxAbs = flat(u).reduce((m, x) => Math.max(m, Math.abs(x)), 0);
    expect(maxAbs).toBeLessThanOrEqual(bound);
    expect(maxAbs).toBeGreaterThan(0.99 * bound);
  });

  it("calculateGain matches torch and is exported from nn", () => {
    expect(calculateGain("leaky_relu", 0.2)).toBeCloseTo(1.3867504905630728, 14);
    expect(calculateGain("conv_transpose2d")).toBe(1);
    expect(calculateGain).toBe(calculateGainInit);
    expect(() => calculateGain("leaky_relu", Number.NaN)).toThrow(InvalidParameterError);
  });

  it("trunc_normal_ stays inside [a, b] and honors mean/std", () => {
    setSeed(5);
    const t = zeros([4000]);
    trunc_normal_(t, 1, 0.5, 0.5, 1.5);
    const v = flat(t);
    expect(Math.min(...v)).toBeGreaterThanOrEqual(0.5);
    expect(Math.max(...v)).toBeLessThanOrEqual(1.5);
    // Symmetric truncation around the mean keeps the mean at 1.
    expect(v.reduce((a, b) => a + b, 0) / v.length).toBeCloseTo(1, 1);
    expect(() => trunc_normal_(t, 0, 1, 2, -2)).toThrow(InvalidParameterError);
    expect(() => trunc_normal_(t, 0, 1, 40, 41)).toThrow(InvalidParameterError);
  });

  it("eye_ fills rectangular and strided tensors", () => {
    expect(flat(eye_(zeros([2, 3])))).toEqual([1, 0, 0, 0, 1, 0]);
    const base = zeros([3, 2]);
    eye_(transpose(base)); // [2, 3] identity view
    expect(flat(base)).toEqual([1, 0, 0, 1, 0, 0]);
    expect(() => eye_(zeros([3]))).toThrow(InvalidParameterError);
  });

  it("makeRng is deterministic and not limited to a short cycle", () => {
    const a = makeRng(123);
    const b = makeRng(123);
    const seen = new Set<number>();
    let mismatches = 0;
    let outOfRange = 0;
    for (let i = 0; i < 300000; i++) {
      const x = a();
      if (x !== b()) mismatches++;
      if (!(x >= 0 && x < 1)) outOfRange++;
      seen.add(x);
    }
    expect(mismatches).toBe(0);
    expect(outOfRange).toBe(0);
    // The previous LCG had only 233280 distinct outputs.
    expect(seen.size).toBeGreaterThan(299000);
    expect(makeRng(1)()).not.toBe(makeRng(2)());
    expect(() => makeRng(Number.NaN)).toThrow(InvalidParameterError);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// containers
// ─────────────────────────────────────────────────────────────────────────────

describe("containers (v1.5.0)", () => {
  it("ParameterDict and ParameterList stay consistent after freezeParameters", () => {
    const w = parameter([[1, 2]]);
    const dict = new ParameterDict({ w });
    dict.freezeParameters();
    const frozen = [...dict.parameters()][0] as GradTensor;
    expect(frozen.requiresGrad).toBe(false);
    expect(dict.get("w")).toBe(frozen);
    expect([...dict.values()][0]).toBe(frozen);
    expect([...dict][0]?.[1]).toBe(frozen);

    const list = new ParameterList([parameter([1, 2]), parameter([3])]);
    list.freezeParameters();
    expect(list.get(0).requiresGrad).toBe(false);
    expect(list.get(-1)).toBe([...list.parameters()][1]);
    expect(list.length).toBe(2);
  });

  it("ParameterList and ParameterDict reject invalid contents and keys", () => {
    const list = new ParameterList();
    expect(() => list.append(tensor([1]) as unknown as GradTensor)).toThrow(InvalidParameterError);
    expect(() => list.get(0)).toThrow(InvalidParameterError);
    expect(() => list.get(0.5)).toThrow(InvalidParameterError);
    list.extend([parameter([1]), parameter([2])]);
    expect(list.length).toBe(2);

    const dict = new ParameterDict();
    expect(() => dict.set("a.b", parameter([1]))).toThrow(InvalidParameterError);
    expect(() => dict.set("", parameter([1]))).toThrow(InvalidParameterError);
    dict.update({ a: parameter([1]), b: parameter([2]) });
    expect(dict.length).toBe(2);
    expect(dict.pop("a")).toBeInstanceOf(GradTensor);
    expect(dict.has("a")).toBe(false);
    dict.clear();
    expect(dict.length).toBe(0);
    expect([...dict.parameters()]).toEqual([]);
  });

  it("ModuleList supports extend/pop and keeps parameter names contiguous", () => {
    const list = new ModuleList([new Linear(2, 2), new Linear(2, 3)]);
    list.extend([new Linear(3, 4)]);
    expect([...list.namedParameters()].map(([n]) => n)).toEqual([
      "0.weight",
      "0.bias",
      "1.weight",
      "1.bias",
      "2.weight",
      "2.bias",
    ]);
    const removed = list.pop(0);
    expect(removed).toBeInstanceOf(Linear);
    expect(list.length).toBe(2);
    expect([...list.namedParameters()].map(([n]) => n)).toEqual([
      "0.weight",
      "0.bias",
      "1.weight",
      "1.bias",
    ]);
    expect(list.pop()).toBeInstanceOf(Linear);
    expect(list.length).toBe(1);
    expect(() => list.insert(1.5, new ReLU())).toThrow(InvalidParameterError);
    expect(() => list.insert(5, new ReLU())).toThrow(InvalidParameterError);
    expect(() => list.append(undefined as unknown as Module)).toThrow(InvalidParameterError);
    expect(() => list.get(1.2)).toThrow(InvalidParameterError);
  });

  it("ModuleList.toString indents nested containers", () => {
    const list = new ModuleList([new Sequential(new ReLU())]);
    expect(list.toString()).toBe("ModuleList(\n  (0): Sequential(\n    (0): ReLU()\n  )\n)");
  });

  it("ModuleDict validates keys, supports update/pop/clear and re-keys in place", () => {
    const dict = new ModuleDict({ enc: new Linear(2, 2) });
    expect(() => dict.set("a.b", new ReLU())).toThrow(InvalidParameterError);
    expect(() => dict.set("", new ReLU())).toThrow(InvalidParameterError);
    expect(() => dict.set("x", {} as unknown as Module)).toThrow(InvalidParameterError);
    dict.update(new Map([["dec", new Linear(2, 2)]]));
    expect([...dict.keys()]).toEqual(["enc", "dec"]);
    const replacement = new Linear(2, 5);
    dict.set("enc", replacement);
    expect([...dict.keys()]).toEqual(["enc", "dec"]);
    expect(dict.get("enc")).toBe(replacement);
    expect(dict.pop("dec")).toBeInstanceOf(Linear);
    expect([...dict.namedParameters()].map(([n]) => n)).toEqual(["enc.weight", "enc.bias"]);
    dict.clear();
    expect(dict.length).toBe(0);
    expect([...dict.parameters()]).toEqual([]);
    expect(() => dict.pop("enc")).toThrow(InvalidParameterError);
  });

  it("Sequential supports append/extend/insert with renumbered parameters", () => {
    const seq = new Sequential(new Linear(2, 3), new Linear(3, 1));
    seq.insert(1, new ReLU());
    expect(seq.length).toBe(3);
    expect(seq.getLayer(1)).toBeInstanceOf(ReLU);
    expect([...seq.namedParameters()].map(([n]) => n)).toEqual([
      "0.weight",
      "0.bias",
      "2.weight",
      "2.bias",
    ]);
    seq.append(new ReLU()).extend([new ReLU()]);
    expect(seq.length).toBe(5);
    const out = seq.forward(tensor([[1, 2]]));
    expect(out.shape).toEqual([1, 1]);

    expect(() => seq.getLayer(1.5)).toThrow(IndexError);
    expect(() => seq.getLayer(5)).toThrow(IndexError);
    expect(() => seq.insert(9, new ReLU())).toThrow(IndexError);
    expect(() => seq.append(undefined as unknown as Module)).toThrow(InvalidParameterError);
    expect(() => new Sequential(new ReLU(), undefined as unknown as Module)).toThrow(
      InvalidParameterError
    );
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// training.ts
// ─────────────────────────────────────────────────────────────────────────────

describe("training utilities (v1.5.0)", () => {
  it("EarlyStopping and ModelCheckpoint reject non-numeric metrics", () => {
    const es = new EarlyStopping({ patience: 2 });
    expect(() => es.step("0.5" as unknown as number)).toThrow(InvalidParameterError);
    const cp = new ModelCheckpoint();
    expect(() => cp.step(new Linear(1, 1), undefined as unknown as number)).toThrow(
      InvalidParameterError
    );
  });

  it("NaN never counts as an improvement", () => {
    const es = new EarlyStopping({ patience: 2 });
    expect(es.step(1)).toBe(false);
    expect(es.step(Number.NaN)).toBe(false);
    expect(es.step(Number.NaN)).toBe(true);
    expect(es.best).toBe(1);

    const model = new Linear(1, 1);
    const cp = new ModelCheckpoint();
    expect(cp.step(model, 1)).toBe(true);
    expect(cp.step(model, Number.NaN)).toBe(false);
    expect(cp.best).toBe(1);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Trainer.ts
// ─────────────────────────────────────────────────────────────────────────────

class Passthrough extends Module {
  forward(x: AnyTensor): AnyTensor {
    return x;
  }
}

class Regressor extends Module {
  private readonly fc = new Linear(2, 1);
  constructor() {
    super();
    this.registerModule("fc", this.fc);
  }
  forward(x: AnyTensor): AnyTensor {
    const gx = x instanceof GradTensor ? x : parameter((x as Tensor).toArray() as number[][]);
    return this.fc.forward(gx);
  }
}

const noopOptimizer = { step() {}, zeroGrad() {} };
const constantLoss = (value: number) => (_o: AnyTensor, _t: Tensor) => tensor([value]);

function batch(rows: number, target = 0): readonly [Tensor, Tensor] {
  return [tensor(Array.from({ length: rows }, () => [1, 2])), tensor([target])];
}

describe("Trainer (v1.5.0)", () => {
  it("runs callbacks for the epoch that triggers early stopping", () => {
    const epochs: number[] = [];
    const trainer = new Trainer(new Passthrough(), noopOptimizer, constantLoss(5), {
      epochs: 50,
      earlyStopping: { patience: 2 },
      callbacks: [(info) => epochs.push(info.epoch)],
    });
    const result = trainer.fit([batch(1)]);
    expect(result.stoppedEarly).toBe(true);
    expect(result.history.length).toBe(3);
    expect(epochs).toEqual([1, 2, 3]);
    expect(result.bestEpoch).toBe(1);
  });

  it("resets early-stopping state on every fit", () => {
    const trainer = new Trainer(new Passthrough(), noopOptimizer, constantLoss(5), {
      epochs: 50,
      earlyStopping: { patience: 2 },
    });
    const first = trainer.fit([batch(1)]);
    const second = trainer.fit([batch(1)]);
    expect(second.history.length).toBe(first.history.length);
    expect(second.history.map((h) => h.epoch)).toEqual([1, 2, 3]);
    expect(second.bestEpoch).toBe(1);
  });

  it("fitAsync matches fit, including callbacks on the stopping epoch", async () => {
    const epochs: number[] = [];
    const trainer = new Trainer(new Passthrough(), noopOptimizer, constantLoss(5), {
      epochs: 50,
      earlyStopping: { patience: 2 },
      callbacks: [(info) => epochs.push(info.epoch)],
    });
    const result = await trainer.fitAsync([batch(1)]);
    expect(result.history.length).toBe(3);
    expect(epochs).toEqual([1, 2, 3]);
    const again = await trainer.fitAsync([batch(1)]);
    expect(again.history.length).toBe(3);
  });

  it("weights the epoch loss by batch size", () => {
    // batches of 3 samples (loss 1) and 1 sample (loss 5): (3 * 1 + 1 * 5) / 4 = 2
    const losses = [1, 5];
    let call = 0;
    const lossFn = () => tensor([losses[call++ % 2] as number]);
    const trainer = new Trainer(new Passthrough(), noopOptimizer, lossFn, { epochs: 1 });
    const result = trainer.fit([batch(3), batch(1)], [batch(3), batch(1)]);
    expect(result.history[0]?.trainLoss).toBeCloseTo(2, 12);
    expect(result.history[0]?.valLoss).toBeCloseTo(2, 12);
  });

  it("throws on empty or exhausted data instead of reporting a loss of 0", () => {
    const trainer = new Trainer(new Passthrough(), noopOptimizer, constantLoss(1), { epochs: 2 });
    expect(() => trainer.fit([])).toThrow(InvalidParameterError);
    expect(() => trainer.fit([batch(1)], [])).toThrow(/valData yielded no batches/);

    function* once(): Generator<readonly [Tensor, Tensor]> {
      yield batch(1);
    }
    expect(() => trainer.fit(once())).toThrow(/epoch 2/);
  });

  it("fitAsync reports empty data too", async () => {
    const trainer = new Trainer(new Passthrough(), noopOptimizer, constantLoss(1), { epochs: 1 });
    await expect(trainer.fitAsync([])).rejects.toThrow(InvalidParameterError);
  });

  it("rejects losses with more than one element", () => {
    const trainer = new Trainer(new Passthrough(), noopOptimizer, () => tensor([1, 2, 3]), {
      epochs: 1,
    });
    expect(() => trainer.fit([batch(1)])).toThrow(ShapeError);
  });

  it("clips the global gradient norm before each optimizer step", () => {
    const model = new Regressor();
    const norms: number[] = [];
    const optimizer = {
      zeroGrad: () => model.zeroGrad(),
      step: () => {
        let sq = 0;
        for (const p of model.parameters()) {
          if (p.grad) for (const g of flat(p.grad)) sq += g * g;
        }
        norms.push(Math.sqrt(sq));
      },
    };
    const lossFn = (out: AnyTensor, target: Tensor): AnyTensor =>
      mseLoss(out as GradTensor, parameter(target.toArray() as number[][]));
    const data: Array<readonly [Tensor, Tensor]> = [[tensor([[10, 20]]), tensor([[1000]])]];

    new Trainer(model, optimizer, lossFn, { epochs: 1 }).fit(data);
    expect(norms[0] as number).toBeGreaterThan(1);

    norms.length = 0;
    new Trainer(model, optimizer, lossFn, { epochs: 1, maxGradNorm: 0.5 }).fit(data);
    expect(norms[0] as number).toBeLessThanOrEqual(0.5 + 1e-6);
    expect(norms[0] as number).toBeGreaterThan(0.49);
  });

  it("trains a real model from plain Tensor batches", () => {
    // Plain input tensors used to leave the loss without a graph, so nothing was learned.
    setSeed(1);
    const model = new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
    const optimizer = new Adam(model.parameters(), { lr: 0.05 });
    const x = tensor([
      [0, 0],
      [1, 0],
      [0, 1],
      [1, 1],
    ]);
    const y = tensor([[0], [1], [2], [3]]);
    const lossFn = (out: AnyTensor, target: Tensor): AnyTensor =>
      mseLoss(out as GradTensor, parameter(target.toArray() as number[][]));
    const result = new Trainer(model, optimizer, lossFn, { epochs: 60 }).fit([[x, y]]);
    const first = result.history[0]?.trainLoss as number;
    const last = result.history[59]?.trainLoss as number;
    expect(last).toBeLessThan(first / 5);
  });

  it("validates constructor arguments", () => {
    const m = new Passthrough();
    expect(() => new Trainer({} as Module, noopOptimizer, constantLoss(1))).toThrow(
      InvalidParameterError
    );
    expect(() => new Trainer(m, noopOptimizer, constantLoss(1), { maxGradNorm: 0 })).toThrow(
      InvalidParameterError
    );
    expect(
      () => new Trainer(m, noopOptimizer, constantLoss(1), { maxGradNorm: Number.NaN })
    ).toThrow(InvalidParameterError);
    expect(() => new Trainer(m, {} as typeof noopOptimizer, constantLoss(1))).toThrow(
      InvalidParameterError
    );
    expect(() => new Trainer(m, noopOptimizer, undefined as never)).toThrow(InvalidParameterError);
    expect(
      () => new Trainer(m, noopOptimizer, constantLoss(1), { callbacks: [1 as never] })
    ).toThrow(InvalidParameterError);
  });
});
