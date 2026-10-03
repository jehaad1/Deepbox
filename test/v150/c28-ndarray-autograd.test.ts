/**
 * Regression tests for src/ndarray/autograd and src/ndarray/index (v1.5.0 review).
 *
 * Reference values come from PyTorch 2.12 (float64 autograd) and NumPy 2.4 (slice scatter masks).
 */
import { describe, expect, it } from "vitest";
import { DeepboxError, InvalidParameterError, ShapeError } from "../../src/core";
import * as ndarray from "../../src/ndarray";
import {
  concatGrad,
  customOp,
  dropoutGrad,
  GradTensor,
  logSoftmaxGrad,
  noGrad,
  parameter,
  softmaxGrad,
  stackGrad,
  type Tensor,
  tensor,
  varianceGrad,
} from "../../src/ndarray";
import { clipGradNorm_ } from "../../src/nn";

function nums(t: Tensor | GradTensor | null): number[] {
  if (t === null) throw new Error("expected a tensor");
  const flat = (v: unknown): number[] =>
    Array.isArray(v) ? v.flatMap(flat) : [Number(v as number | bigint)];
  return flat(t.toArray());
}

function expectClose(actual: number[], expected: number[], tol = 1e-9): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    const e = expected[i] as number;
    const a = actual[i] as number;
    if (Number.isNaN(e)) expect(Number.isNaN(a)).toBe(true);
    else expect(Math.abs(a - e)).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(e)));
  }
}

const f64 = (data: number | number[] | number[][] | number[][][]) =>
  parameter(tensor(data, { dtype: "float64" }));
const f32 = (data: number | number[] | number[][] | number[][][]) =>
  parameter(tensor(data, { dtype: "float32" }));

describe("GradTensor construction", () => {
  it("honors options.dtype in the data constructor", () => {
    const g = new GradTensor([1, 2, 3], { requiresGrad: true, dtype: "float64" });
    expect(g.dtype).toBe("float64");
    expect(new GradTensor([1, 2, 3]).dtype).toBe("float32");
  });
});

describe("leaf gradient storage", () => {
  it("does not share gradient buffers between leaves", () => {
    const x = f32([1, 2, 3]);
    const y = f32([4, 5, 6]);
    x.add(y).sum().backward();
    const gx = x.grad as Tensor;
    const gy = y.grad as Tensor;
    expect(gx).not.toBe(gy);
    gx.data[0] = 100;
    expect(nums(y.grad)).toEqual([1, 1, 1]);
  });

  it("clipGradNorm_ scales each shared-source gradient exactly once", () => {
    const x = f32([1, 2, 3]);
    const y = f32([4, 5, 6]);
    x.add(y).sum().backward();
    // torch.nn.utils.clip_grad_norm_([x, y], 1.0) -> total norm sqrt(6), coefficient 1/(sqrt(6)+1e-6)
    const norm = clipGradNorm_([x, y], 1.0);
    expect(norm).toBeCloseTo(Math.sqrt(6), 6);
    const coef = 1 / (Math.sqrt(6) + 1e-6);
    expectClose(nums(x.grad), [coef, coef, coef], 1e-6);
    expectClose(nums(y.grad), [coef, coef, coef], 1e-6);
  });

  it("accumulates when backward() is called on a leaf twice", () => {
    const x = f32([1, 2, 3]);
    x.backward();
    x.backward();
    expect(nums(x.grad)).toEqual([2, 2, 2]);
  });
});

describe("backward seed validation", () => {
  it("rejects a seed gradient with a different number of elements", () => {
    const x = f32([1, 2, 3]);
    const y = x.sum();
    expect(() => y.backward(tensor([1, 1, 1]))).toThrow(ShapeError);
    expect(x.grad).toBeNull();
  });

  it("reshapes a same-size seed with another shape", () => {
    const x = f32([1, 2, 3]);
    x.sum().backward(tensor([2]));
    expect(nums(x.grad)).toEqual([2, 2, 2]);
  });
});

describe("accumulateGrad shape check", () => {
  it("rejects a customOp gradient with the wrong shape", () => {
    const x = f32([1, 2, 3]);
    const out = customOp(tensor([2, 4, 6]), [[x, () => tensor([[1, 1, 1]])]]);
    expect(() => out.backward()).toThrow(ShapeError);
  });
});

describe("pow", () => {
  it("has a zero gradient for exponent 0, also at x = 0", () => {
    const x = f64([0, 1, 2]);
    const y = x.pow(0);
    expect(nums(y)).toEqual([1, 1, 1]);
    y.sum().backward();
    // torch: (x ** 0).sum().backward() -> [0, 0, 0]
    expect(nums(x.grad)).toEqual([0, 0, 0]);
  });

  it("promotes integer tensors for fractional exponents instead of truncating", () => {
    const x = parameter(tensor([0, 1, 4], { dtype: "int32" }));
    const y = x.sqrt();
    expect(y.dtype).toBe("float64");
    expectClose(nums(y), [0, 1, 2]);
    const z = parameter(tensor([2, 3], { dtype: "int32" })).pow(2);
    expect(z.dtype).toBe("int32");
    expect(nums(z)).toEqual([4, 9]);
  });
});

describe("dtype preservation", () => {
  it("keeps float32 through gelu, elu and tanhshrink and matches torch gradients", () => {
    const xs = [-1.5, -0.3, 0, 0.7, 2.1];
    const cases: Array<[string, (g: GradTensor) => GradTensor, number[], number[]]> = [
      [
        "gelu",
        (g) => g.gelu(),
        [-0.10042842301976707, -0.11462907645724074, 0, 0.5305701347051167, 2.062669004961112],
        [-0.1277107931514331, 0.26770454836119817, 0.5, 0.976357218656104, 1.0753508892075991],
      ],
      [
        "elu",
        (g) => g.elu(1.3),
        [-1.0099307918070413, -0.33693631311376676, 0, 0.7, 2.1],
        [0.2900692081929588, 0.9630636868862332, 1.3, 1, 1],
      ],
      [
        "tanhshrink",
        (g) => g.tanhshrink(),
        [-0.5948517463551336, -0.008687387548409087, 0, 0.09563222288283646, 1.1295480633865462],
        [0.8192933610763514, 0.08486303817337082, 0, 0.3652604100175414, 0.9417769612768031],
      ],
    ];
    for (const [name, fn, y, g] of cases) {
      const x = f32(xs);
      const out = fn(x);
      expect(out.dtype, name).toBe("float32");
      out.sum().backward();
      expect((x.grad as Tensor).dtype, name).toBe("float32");
      expectClose(nums(out), y, 1e-5);
      expectClose(nums(x.grad), g, 1e-5);
    }
  });

  it("allows repeated backward with zeroGrad on a float32 tensor", () => {
    const x = f32([0.5, -1, 2]);
    x.gelu().sum().backward();
    x.zeroGrad();
    expect(() => x.tanhshrink().sum().backward()).not.toThrow();
  });
});

describe("max / min backward", () => {
  it("works for float32 with axis reductions", () => {
    const x = f32([
      [1, 3, 3],
      [2, 2, 2],
      [5, 1, 0],
    ]);
    const m = x.max(1);
    expect(m.dtype).toBe("float32");
    m.sum().backward();
    // torch.amax(x, 1): ties share the gradient equally
    expectClose(nums(x.grad), [0, 0.5, 0.5, 1 / 3, 1 / 3, 1 / 3, 1, 0, 0], 1e-6);

    const y = f32([
      [1, 3, 3],
      [1, 2, 2],
      [5, 1, 3],
    ]);
    y.min(0).sum().backward();
    // torch.amin(y, 0)
    expectClose(nums(y.grad), [0.5, 0, 0, 0.5, 0, 1, 0, 1, 0], 1e-6);
  });

  it("sends the gradient to the NaN element when the maximum is NaN", () => {
    const x = f32([
      [1, Number.NaN, 3],
      [4, 5, 6],
    ]);
    const m = x.max(1);
    m.sum().backward();
    expect(Number.isNaN(nums(m)[0] as number)).toBe(true);
    expect(nums(x.grad)).toEqual([0, 1, 0, 0, 0, 1]);
  });

  it("matches torch for a full reduction with keepdims", () => {
    const x = f64([
      [1, 5],
      [5, 2],
    ]);
    x.max(undefined, true).sum().backward();
    expectClose(nums(x.grad), [0, 0.5, 0.5, 0]);
  });
});

describe("div backward", () => {
  it("does not overflow b*b for large float32 denominators", () => {
    const a = f32([1e30]);
    const b = f32([1e20]);
    a.div(b).sum().backward();
    // torch float32: a.grad = 1e-20, b.grad = -1e-10
    expect(Math.abs((nums(a.grad)[0] as number) / 1e-20 - 1)).toBeLessThan(1e-5);
    expect(Math.abs((nums(b.grad)[0] as number) / -1e-10 - 1)).toBeLessThan(1e-5);
  });

  it("applies the upstream gradient last so large seeds do not overflow", () => {
    const a = f32([1e20]);
    const b = f32([1e10]);
    a.div(b).backward(tensor([1e30]));
    // torch float32: a.grad = 1e20, b.grad = -1e30
    expect(Math.abs((nums(a.grad)[0] as number) / 1e20 - 1)).toBeLessThan(1e-5);
    expect(Math.abs((nums(b.grad)[0] as number) / -1e30 - 1)).toBeLessThan(1e-5);
  });

  it("matches torch for broadcast division", () => {
    const a = f64([
      [1, 2],
      [3, 4],
    ]);
    const b = f64([2, 4]);
    a.div(b).sum().backward();
    expectClose(nums(a.grad), [0.5, 0.25, 0.5, 0.25]);
    // -sum(a)/b^2 per column: -(1+3)/4, -(2+4)/16
    expectClose(nums(b.grad), [-1, -0.375]);
  });
});

describe("gather backward", () => {
  it("scatter-adds repeated indices (torch.index_select reference)", () => {
    const a = f64([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const idx = GradTensor.fromTensor(tensor([2, 0, 2], { dtype: "int32" }));
    const y = a.gather(idx, 1);
    expect(nums(y)).toEqual([3, 1, 3, 6, 4, 6]);
    const w = GradTensor.fromTensor(
      tensor(
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        { dtype: "float64" }
      )
    );
    y.mul(w).sum().backward();
    expect(nums(a.grad)).toEqual([2, 0, 4, 5, 0, 10]);
  });

  it("does not record the (non-differentiable) indices as a graph parent", () => {
    const a = GradTensor.fromTensor(tensor([1, 2, 3], { dtype: "float64" }));
    const idx = parameter(tensor([0, 2], { dtype: "int32" }));
    const out = a.gather(idx, 0);
    expect(out.requiresGrad).toBe(false);
  });

  it("gathers along the leading axis of a 3-D tensor", () => {
    const a = f64([[[1, 2]], [[3, 4]], [[5, 6]]]);
    const idx = GradTensor.fromTensor(tensor([1, 1], { dtype: "int32" }));
    a.gather(idx, 0).sum().backward();
    expect(nums(a.grad)).toEqual([0, 0, 2, 2, 0, 0]);
  });
});

describe("slice backward", () => {
  const base = () =>
    f64(Array.from({ length: 4 }, (_, r) => Array.from({ length: 6 }, (_, c) => r * 6 + c)));

  it.each([
    [
      "negative start with strided columns",
      [{ start: -3 }, { start: 1, step: 2 }],
      [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ],
    [
      "integer row with negative step",
      [2, { start: 5, end: 0, step: -2 }],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    ],
    [
      "out-of-range bounds",
      [{ start: -100, end: 100, step: 3 }, { step: 4 }],
      [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    ],
    ["empty range", [{ start: 3, end: 1 }], new Array<number>(24).fill(0)],
  ] as const)("matches NumPy: %s", (_name, ranges, expected) => {
    const x = base();
    x.slice(...(ranges as unknown as Parameters<GradTensor["slice"]>))
      .sum()
      .backward();
    expect(nums(x.grad)).toEqual([...expected]);
  });
});

describe("stackGrad / concatGrad", () => {
  it("rejects parts with different shapes", () => {
    expect(() => stackGrad([f32([1, 2, 3]), f32([1, 2])])).toThrow(ShapeError);
  });

  it("rejects an empty list", () => {
    expect(() => stackGrad([])).toThrow(DeepboxError);
    expect(() => concatGrad([])).toThrow(DeepboxError);
  });

  it("backpropagates through stack with per-part weights", () => {
    const a = f64([1, 2]);
    const b = f64([3, 4]);
    const s = stackGrad([a, b]);
    expect(s.shape).toEqual([2, 2]);
    const w = GradTensor.fromTensor(
      tensor(
        [
          [1, 2],
          [3, 4],
        ],
        { dtype: "float64" }
      )
    );
    s.mul(w).sum().backward();
    expect(nums(a.grad)).toEqual([1, 2]);
    expect(nums(b.grad)).toEqual([3, 4]);
  });

  it("concatenates along a negative axis and splits the gradient", () => {
    const a = f64([[1, 2]]);
    const b = f64([[3]]);
    const c = concatGrad([a, b], -1);
    expect(nums(c)).toEqual([1, 2, 3]);
    const w = GradTensor.fromTensor(tensor([[10, 20, 30]], { dtype: "float64" }));
    c.mul(w).sum().backward();
    expect(nums(a.grad)).toEqual([10, 20]);
    expect(nums(b.grad)).toEqual([30]);
  });
});

describe("variance / softmax / logSoftmax", () => {
  it("variance accepts string axes and matches torch.var(correction=0)", () => {
    const x = f64([
      [1, 2, 3],
      [4, 5, 7],
    ]);
    const v = varianceGrad(x, "columns", 0);
    expectClose(nums(v), [0.6666666666666666, 1.555555555555556]);
    v.mul(GradTensor.fromTensor(tensor([1, 2], { dtype: "float64" })))
      .sum()
      .backward();
    expectClose(
      nums(x.grad),
      [
        -0.6666666666666666, 0, 0.6666666666666666, -1.7777777777777772, -0.444444444444444,
        2.2222222222222223,
      ]
    );
  });

  it("softmax and logSoftmax stay finite and sum correctly for large logits", () => {
    const x = f32([[1000, 1001, 1002]]);
    const s = softmaxGrad(x, -1);
    expect(nums(s).reduce((p, c) => p + c, 0)).toBeCloseTo(1, 6);
    const l = logSoftmaxGrad(x, -1);
    expectClose(nums(l), [-2.4076059644443806, -1.4076059644443806, -0.4076059644443806], 1e-5);
    expect(l.dtype).toBe("float32");
  });
});

describe("matmul backward", () => {
  it("matches torch for a 2-D by 1-D product", () => {
    const a = f64([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const b = f64([1, -1, 2]);
    const y = a.matmul(b);
    expect(nums(y)).toEqual([5, 11]);
    y.mul(GradTensor.fromTensor(tensor([1, 2], { dtype: "float64" })))
      .sum()
      .backward();
    expect(nums(a.grad)).toEqual([1, -1, 2, 2, -2, 4]);
    expect(nums(b.grad)).toEqual([9, 12, 15]);
  });

  it("matches torch for batched matmul", () => {
    const a = f64([
      [
        [1, 2],
        [3, 4],
      ],
      [
        [0, 1],
        [1, 0],
      ],
    ]);
    const b = f64([
      [
        [1, 0],
        [2, 1],
      ],
      [
        [3, 1],
        [0, 2],
      ],
    ]);
    const y = a.matmul(b);
    const w = GradTensor.fromTensor(
      tensor(
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
        { dtype: "float64" }
      )
    );
    y.mul(w).sum().backward();
    expect(nums(a.grad)).toEqual([1, 4, 3, 10, 21, 12, 29, 16]);
    expect(nums(b.grad)).toEqual([10, 14, 14, 20, 7, 8, 5, 6]);
  });
});

describe("dropout validation", () => {
  it("validates p in every mode", () => {
    const x = f32([1, 2, 3]);
    expect(() => dropoutGrad(x, 1)).toThrow(InvalidParameterError);
    expect(() => dropoutGrad(x, -0.1)).toThrow(InvalidParameterError);
    expect(() => dropoutGrad(x, 1.5, false)).toThrow(InvalidParameterError);
    expect(() => dropoutGrad(x, Number.NaN)).toThrow(InvalidParameterError);
    expect(dropoutGrad(x, 0.5, false)).toBe(x);
    expect(dropoutGrad(x, 0)).toBe(x);
  });
});

describe("noGrad", () => {
  it("rejects async callbacks and other thenables", () => {
    expect(() => noGrad(async () => 1)).toThrow(DeepboxError);
    // biome-ignore lint/suspicious/noThenProperty: the point is to pass a non-Promise thenable
    const thenable = { then: () => undefined };
    expect(() => noGrad(() => thenable)).toThrow(DeepboxError);
    const x = f32([1]);
    expect(noGrad(() => x.mul(x)).requiresGrad).toBe(false);
    expect(x.mul(x).requiresGrad).toBe(true);
  });
});

describe("ndarray index exports", () => {
  it("re-exports hardtanh and tanhshrink", () => {
    const t = tensor([-2, 0.5, 2], { dtype: "float64" });
    expectClose(nums(ndarray.hardtanh(t)), [-1, 0.5, 1]);
    expectClose(nums(ndarray.tanhshrink(t)), [
      -2 - Math.tanh(-2),
      0.5 - Math.tanh(0.5),
      2 - Math.tanh(2),
    ]);
  });
});
