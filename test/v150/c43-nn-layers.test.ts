/**
 * Regression tests for src/nn/layers/upsample.ts, src/nn/layers/utility.ts,
 * src/nn/losses/cross_entropy.ts, src/nn/losses/index.ts and src/nn/module/Module.ts (v1.5.0).
 *
 * Reference values come from PyTorch 2.12 (float64) with the inputs written in each test.
 */
import { describe, expect, it } from "vitest";
import { DTypeError, InvalidParameterError, ShapeError } from "../../src/core";
import { GradTensor, parameter, type Tensor, tensor, transpose } from "../../src/ndarray";
import {
  binaryCrossEntropyLoss,
  binaryCrossEntropyWithLogitsLoss,
  cosineEmbeddingLoss,
  crossEntropyLoss,
  ctcLoss,
  Flatten,
  gaussianNLLLoss,
  huberLoss,
  klDivLoss,
  Linear,
  Module,
  maeLoss,
  marginRankingLoss,
  mseLoss,
  nllLoss,
  ReLU,
  rmseLoss,
  Sequential,
  Sigmoid,
  smoothL1Loss,
  tripletMarginLoss,
  Unflatten,
  Upsample,
} from "../../src/nn";

const UP: Record<string, { shape: number[]; out: number[]; grad: number[] }> = {
  up_bil_noac: {
    shape: [1, 1, 4, 6],
    out: [
      1.0, 1.5, 2.166666666667, 2.833333333333, 3.5, 4.0, 3.5, 4.0, 4.666666666667, 5.333333333333,
      6.0, 6.5, 6.5, 7.0, 7.666666666667, 8.333333333333, 9.1875, 9.875, 9.0, 9.5, 10.166666666667,
      10.833333333333, 12.0, 13.0,
    ],
    grad: [
      1.025, 1.322916666667, 1.620833333333, 1.91875, 2.3125, 2.583333333333, 2.854166666667, 3.125,
      4.0625, 4.360416666667, 4.658333333333, 4.95625,
    ],
  },
  up_bil_ac: {
    shape: [1, 1, 4, 6],
    out: [
      1.0, 1.6, 2.2, 2.8, 3.4, 4.0, 3.666666666667, 4.266666666667, 4.866666666667, 5.466666666667,
      6.066666666667, 6.666666666667, 6.333333333333, 6.933333333333, 7.533333333333,
      8.133333333333, 8.866666666667, 9.666666666667, 9.0, 9.6, 10.2, 10.8, 11.8, 13.0,
    ],
    grad: [
      0.893333333333, 1.333333333333, 1.653333333333, 1.72, 2.293333333333, 2.933333333333,
      3.253333333333, 3.12, 3.693333333333, 4.533333333333, 4.853333333333, 4.52,
    ],
  },
  up_near_15: {
    shape: [1, 1, 4, 6],
    out: [
      1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0, 7.0, 8.0, 9.0,
      9.0, 10.0, 11.0, 11.0, 13.0,
    ],
    grad: [2.6, 1.6, 3.8, 2.2, 3.1, 1.7, 3.7, 2.0, 4.3, 2.3, 4.9, 2.6],
  },
  up_near_size: {
    shape: [1, 1, 5, 7],
    out: [
      1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0,
      7.0, 8.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 9.0, 9.0, 10.0, 10.0, 11.0, 11.0, 13.0,
    ],
    grad: [2.8, 3.6, 4.4, 2.5, 8.4, 9.2, 10.0, 5.3, 6.3, 6.7, 7.1, 3.7],
  },
  up_bil_noac_sf: {
    shape: [1, 1, 6, 6],
    out: [
      1.0, 1.5, 2.166666666667, 2.833333333333, 3.5, 4.0, 2.0, 2.5, 3.166666666667, 3.833333333333,
      4.5, 5.0, 4.0, 4.5, 5.166666666667, 5.833333333333, 6.5, 7.0, 6.0, 6.5, 7.166666666667,
      7.833333333333, 8.625, 9.25, 8.0, 8.5, 9.166666666667, 9.833333333333, 10.875, 11.75, 9.0,
      9.5, 10.166666666667, 10.833333333333, 12.0, 13.0,
    ],
    grad: [
      2.125, 2.558333333333, 2.991666666667, 3.425, 5.5, 5.933333333333, 6.366666666667, 6.8, 8.875,
      9.308333333333, 9.741666666667, 10.175,
    ],
  },
};
const CE: Record<string, { loss: number[]; grad: number[] }> = {
  ce_weight: {
    loss: [1.026105990258],
    grad: [
      -0.06199979293, 0.044078721946, 0.017921070984, 0.042223467154, -0.051644796142,
      0.009421328988, 0.010451280427, 0.017231248346, -0.027682528773, 0.227429428517,
      -0.329620058042, 0.102190629525,
    ],
  },
  ce_ignore: {
    loss: [1.049826634668],
    grad: [
      -0.113666287038, 0.080810990235, 0.032855296803, 0.0, 0.0, 0.0, 0.038321361566,
      0.063181243936, -0.101502605502, 0.208476976141, -0.302151719872, 0.093674743732,
    ],
  },
  ce_ls: {
    loss: [0.896497861115],
    grad: [
      -0.068583048612, 0.052274909343, 0.016308139269, 0.020695300335, -0.018839130681,
      -0.001856169654, 0.020407687841, 0.039052599619, -0.05946028746, 0.148024398772,
      -0.209947123238, 0.061922724465,
    ],
  },
  ce_all: {
    loss: [1.523236153335],
    grad: [
      -0.053056806615, 0.033480210399, 0.019576596216, 0.0, 0.0, 0.0, 0.001755405803,
      -0.00379684853, 0.002041442727, 0.308559057745, -0.446238416942, 0.137679359197,
    ],
  },
  ce_none: {
    loss: [0.417030016278, 0.153178207122, 0.363135506369, 2.369314381356],
    grad: [
      -0.340998861114, 0.242432970705, 0.098565890409, 0.116114534674, -0.142023189392,
      0.025908654717, 0.114964084698, 0.189543731809, -0.304507816507, 0.625430928422,
      -0.906455159617, 0.281024231195,
    ],
  },
  ce_sum: {
    loss: [3.302658111125],
    grad: [
      -0.340998861114, 0.242432970705, 0.098565890409, 0.116114534674, -0.142023189392,
      0.025908654717, 0.114964084698, 0.189543731809, -0.304507816507, 0.625430928422,
      -0.906455159617, 0.281024231195,
    ],
  },
  ce_soft_w: {
    loss: [1.160178539102],
    grad: [
      0.01446282743, -0.030300520922, 0.015837693493, 0.02580010892, -0.024635145359,
      -0.001164963561, -0.024133080943, -0.057352660343, 0.081485741286, 0.096993505316,
      -0.124275168895, 0.027281663579,
    ],
  },
  ce_soft_ls: {
    loss: [1.134414527781],
    grad: [
      0.017250284721, 0.000608242676, -0.017858527398, -0.013471366331, 0.049494202652,
      -0.036022836321, -0.031258978825, -0.012614067048, 0.043873045873, 0.078857732106,
      -0.054113789904, -0.024743942201,
    ],
  },
  ce_masked: {
    loss: [0.220094849281],
    grad: [-0.134470710685, 0.0, 0.134470710685, 0.059601461011, -0.059601461011, 0.0],
  },
};
const BCE: Record<string, { loss: number[]; grad: number[] }> = {
  bce_multi: {
    loss: [0.263190471936],
    grad: [
      -0.01986715367, 0.044823570228, -0.0629234448, 0.007904312196, -0.075027667115,
      -0.00299770166,
    ],
  },
  bce_pw: {
    loss: [0.243326231101],
    grad: [
      -0.039734307341, 0.044823570228, -0.0314617224, 0.007904312196, -0.075027667115,
      -0.00149885083,
    ],
  },
  bce_none: {
    loss: [
      0.126928011043, 0.313261687518, 0.47407698418, 0.048587351574, 0.598138869382, 0.018149927918,
    ],
    grad: [
      -0.119202922022, 0.26894142137, -0.377540668798, 0.047425873178, -0.450166002688,
      -0.017986209962,
    ],
  },
  bce_sum: {
    loss: [1.579142831614],
    grad: [
      -0.119202922022, 0.26894142137, -0.377540668798, 0.047425873178, -0.450166002688,
      -0.017986209962,
    ],
  },
};
const NLL: Record<string, { loss: number[]; grad: number[] }> = {
  nll_w: {
    loss: [1.524921866336],
    grad: [
      -0.285714285714, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.142857142857, 0.0, -0.571428571429,
      0.0,
    ],
  },
};
const NLL_LP: number[] = [
  -0.417030016278, -1.417030016278, -2.317030016278, -2.153178207122, -0.153178207122,
  -3.653178207122, -2.163135506369, -1.663135506369, -0.363135506369, -0.469314381356,
  -2.369314381356, -1.269314381356,
];
const HUBER: Record<string, { loss: number[]; grad: number[] }> = {
  huber: { loss: [0.0, 0.5, 0.5, 2.5, 0.02], grad: [0.0, 1.0, -1.0, 1.0, -0.2] },
  sl1_b0: { loss: [0.0, 1.0, 1.0, 3.0, 0.2], grad: [0.0, 1.0, -1.0, 1.0, -1.0] },
  sl1_b2: { loss: [0.0, 0.25, 0.25, 2.0, 0.01], grad: [0.0, 0.5, -0.5, 1.0, -0.1] },
};
const KL: Record<string, { loss: number[]; grad: number[] }> = {
  kl_mean: {
    loss: [0.08139858362],
    grad: [-0.0, -0.1, -0.066666666667, -0.083333333333, -0.041666666667, -0.041666666667],
  },
  kl_bm: { loss: [0.24419575086], grad: [-0.0, -0.3, -0.2, -0.25, -0.125, -0.125] },
  kl_none: {
    loss: [0.0, 0.002081669433, 0.439201736379, -0.128159156906, 0.112633626407, 0.062633626407],
    grad: [-0.0, -0.6, -0.4, -0.5, -0.25, -0.25],
  },
};
const KL_Q: number[] = [
  -1.314295072821, -0.514295072821, -2.014295072821, -0.436828866749, -1.836828866749,
  -1.636828866749,
];
const KL_LOGT: number[] = [0.24419575086];
const KL_LT: number[] = [
  -69.077552789821,
  -0.510825623766,
  -0.916290731874,
  -Math.LN2,
  -1.38629436112,
  -1.38629436112,
];
const COS: number[] = [0.521908556266, 0.0, 1.0];
const MR: { loss: number[]; grad: number[] } = {
  loss: [0.7],
  grad: [-0.333333333333, -0.0, 0.333333333333],
};
const G_ROW: number[] = [-0.09657359028, -0.09657359028, 0.40907359028, 0.59657359028];
const G_ROW0: number[] = [-0.09657359028, -0.09657359028, 0.40907359028, 0.59657359028];
const CTC_LP: number[] = [
  -0.55718507422, -2.122884071795, -1.975183977268, -1.779283458713, -1.535558496309,
  -1.40196568208, -0.776297841455, -2.545349260061, -2.933959270644, -1.53542440849,
  -1.024735830981, -0.987361402382, -3.360318947512, -0.699257122301, -1.768725746145,
  -1.211429535517, -1.414637314264, -1.958749254366, -1.204827731132, -1.151355293938,
  -3.130717932089, -1.024239073683, -1.677071545155, -0.890809583519, -3.050426131099,
  -1.334270653426, -0.797587578869, -1.431723493527, -1.354350488584, -1.673136164899,
  -1.221893591591, -1.348783399925, -1.644639382677, -2.17971250571, -1.107412793404,
  -1.012165588003, -2.770444723615, -3.364542451856, -1.035624218178, -0.601875487277,
  -1.41272238838, -0.908624028614, -3.427346440666, -1.136410723931, -1.676387011283,
  -1.587585927493, -1.219799464988, -1.1607742297,
];
const CTC: Record<string, number[]> = {
  ctc_mean: [2.545935904671],
  ctc_none: [6.631225247102, 5.762926787281],
  ctc_pad: [6.631225247102, 5.762926787281],
  ctc_empty: [11.013569561284, 5.762926787281],
};

const f64 = (data: number[], shape?: number[]): Tensor => {
  const t = tensor(data, { dtype: "float64" });
  return shape ? t.reshape(shape) : t;
};
const i32 = (data: number[]): Tensor => tensor(data, { dtype: "int32" });

function values(t: Tensor | GradTensor): number[] {
  const tt = GradTensor.isGradTensor(t) ? t.tensor : t;
  const arr = tt.toArray();
  return Array.isArray(arr) ? (arr.flat(10) as number[]) : [Number(arr)];
}

function expectClose(actual: number[], expected: readonly number[], tol = 1e-9): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(Math.abs((actual[i] as number) - (expected[i] as number))).toBeLessThan(tol);
  }
}

/** Test helper that exposes the protected registration methods. */
class Holder extends Module {
  constructor(setup?: (self: Holder) => void) {
    super();
    setup?.(this);
  }
  add(name: string, child: Module): void {
    this.registerModule(name, child);
  }
  param(name: string, p: GradTensor): void {
    this.registerParameter(name, p);
  }
  buf(name: string, b: Tensor): void {
    this.registerBuffer(name, b);
  }
  forward(x: Tensor | GradTensor): Tensor | GradTensor {
    return x;
  }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

describe("Module (v1.5.0)", () => {
  it("freezeParameters / unfreezeParameters keep the parameter objects", () => {
    const model = new Sequential(new Linear(2, 3), new ReLU(), new Linear(3, 1));
    const before = [...model.parameters()];
    model.freezeParameters();
    expect([...model.parameters()]).toEqual(before);
    for (const p of before) expect(p.requiresGrad).toBe(false);
    model.unfreezeParameters(["0.weight"]);
    expect(before[0]?.requiresGrad).toBe(true);
    expect(before[1]?.requiresGrad).toBe(false);
    expect([...model.parameters()][0]).toBe(before[0]);
  });

  it("freezing clears an existing gradient", () => {
    const layer = new Linear(2, 2);
    const out = layer.forward(
      GradTensor.fromTensor(tensor([[1, 2]], { dtype: "float32" }), { requiresGrad: true })
    );
    out.sum().backward();
    const w = [...layer.parameters()][0] as GradTensor;
    expect(w.grad).not.toBeNull();
    layer.freezeParameters();
    expect(w.grad).toBeNull();
  });

  it("an unknown name throws and leaves every parameter untouched", () => {
    const model = new Sequential(new Linear(2, 3), new Linear(3, 1));
    expect(() => model.freezeParameters(["0.weight", "5.weight"])).toThrow(InvalidParameterError);
    for (const p of model.parameters()) expect(p.requiresGrad).toBe(true);
  });

  it("resolves parameter names whose child module key contains a dot", () => {
    const inner = new Linear(2, 2);
    const holder = new Holder((h) => h.add("layers.0", inner));
    holder.freezeParameters(["layers.0.weight"]);
    const [weight, bias] = [...inner.parameters()] as GradTensor[];
    expect(weight?.requiresGrad).toBe(false);
    expect(bias?.requiresGrad).toBe(true);
    holder.unfreezeParameters(["layers.0.weight"]);
    expect(weight?.requiresGrad).toBe(true);
  });

  it("yields a shared parameter and a repeated child module only once", () => {
    const shared = new Linear(2, 2);
    const seq = new Sequential(shared, new ReLU(), shared);
    expect(seq.modules().next().value).toBe(seq);
    expect([...seq.modules()].filter((m) => m === shared).length).toBe(1);
    expect([...seq.parameters()].length).toBe(2);
    expect([...seq.namedParameters()].map(([n]) => n)).toEqual(["0.weight", "0.bias"]);
    // an optimizer built from parameters() updates each tensor once
    expect(seq.summary()).toContain("Total params: 6");
    // the state dict still lists both names so the model can be rebuilt
    const sd = seq.stateDict();
    expect(Object.keys(sd.parameters).sort()).toEqual(["0.bias", "0.weight", "2.bias", "2.weight"]);
    expect([...seq.namedParameters("", true, false)].length).toBe(4);
  });

  it("tied parameters registered under two names appear once in parameters()", () => {
    const w = parameter(f64([1, 2, 3, 4], [2, 2]));
    const holder = new Holder((h) => {
      h.param("a", w);
      h.param("b", w);
    });
    expect([...holder.parameters()].length).toBe(1);
    expect(Object.keys(holder.stateDict().parameters).sort()).toEqual(["a", "b"]);
    holder.loadStateDict(holder.stateDict());
  });

  it("stateDict copies tensors in logical order for strided views and offsets", () => {
    const base = f64([1, 2, 3, 4, 5, 6]).reshape([2, 3]);
    const transposed = transpose(base); // shape [3, 2], non-contiguous
    const sliced = f64([9, 8, 7, 6, 5]).slice({ start: 1, end: 4 }); // offset 1
    const holder = new Holder((h) => {
      h.param("t", GradTensor.fromTensor(transposed, { requiresGrad: true }));
      h.param("s", GradTensor.fromTensor(sliced, { requiresGrad: true }));
      h.buf("b", transposed);
    });
    const sd = holder.stateDict();
    expect(sd.parameters.t?.data).toEqual([1, 4, 2, 5, 3, 6]);
    expect(sd.parameters.t?.shape).toEqual([3, 2]);
    expect(sd.parameters.s?.data).toEqual([8, 7, 6]);
    expect(sd.buffers.b?.data).toEqual([1, 4, 2, 5, 3, 6]);

    // the saved state loads back into the same layout
    const target = new Holder((h) => {
      h.param("t", GradTensor.fromTensor(f64([0, 0, 0, 0, 0, 0], [3, 2]), { requiresGrad: true }));
      h.param("s", GradTensor.fromTensor(f64([0, 0, 0]), { requiresGrad: true }));
      h.buf("b", f64([0, 0, 0, 0, 0, 0], [3, 2]));
    });
    target.loadStateDict(sd);
    expect(target.stateDict()).toEqual(sd);

    // and into a strided target through its strides
    const destBase = f64([0, 0, 0, 0, 0, 0], [2, 3]);
    const strided = new Holder((h) => {
      h.param("t", GradTensor.fromTensor(transpose(destBase), { requiresGrad: true }));
      h.param("s", GradTensor.fromTensor(f64([0, 0, 0]), { requiresGrad: true }));
      h.buf("b", f64([0, 0, 0, 0, 0, 0], [3, 2]));
    });
    strided.loadStateDict(sd);
    expect(values(destBase)).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it("loadStateDict validates everything before writing anything", () => {
    const model = new Sequential(new Linear(2, 2), new Linear(2, 1));
    const before = model.stateDict();
    const bad = model.stateDict();
    for (const entry of Object.values(bad.parameters)) {
      entry.data = entry.data.map(() => 42);
    }
    const last = bad.parameters["1.bias"];
    if (!last) throw new Error("missing entry");
    last.dtype = "float64"; // layer parameters are float32
    expect(() => model.loadStateDict(bad)).toThrow(DTypeError);
    expect(model.stateDict()).toEqual(before);

    const wrongType = model.stateDict();
    const first = wrongType.parameters["0.weight"];
    if (!first) throw new Error("missing entry");
    first.data = first.data.map(() => "x");
    expect(() => model.loadStateDict(wrongType)).toThrow(DTypeError);
    expect(model.stateDict()).toEqual(before);
  });

  it("loadStateDict does not mistake inherited object keys for entries", () => {
    const holder = new Holder((h) => h.param("valueOf", parameter(f64([1]))));
    expect(() => holder.loadStateDict({ parameters: {}, buffers: {} })).toThrow(
      /missing parameter: valueOf/
    );
  });

  it("children() / namedChildren() list direct children once", () => {
    const a = new ReLU();
    const holder = new Holder((h) => {
      h.add("x", a);
      h.add("y", a);
      h.add("z", new Sigmoid());
    });
    expect([...holder.namedChildren()].map(([n]) => n)).toEqual(["x", "z"]);
    expect([...holder.children()].length).toBe(2);
  });

  it("rejects empty registration names and self registration", () => {
    const holder = new Holder();
    expect(() => holder.add("", new ReLU())).toThrow(InvalidParameterError);
    expect(() => holder.param("", parameter(f64([1])))).toThrow(InvalidParameterError);
    expect(() => holder.buf("", f64([1]))).toThrow(InvalidParameterError);
    expect(() => holder.add("me", holder)).toThrow(InvalidParameterError);
  });

  it("modules() terminates on a cycle", () => {
    const a = new Holder();
    const b = new Holder();
    a.add("b", b);
    b.add("a", a);
    expect([...a.modules()].length).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// Flatten / Unflatten
// ---------------------------------------------------------------------------

describe("Flatten and Unflatten (v1.5.0)", () => {
  it("Flatten turns a 0-d tensor into shape [1] like PyTorch", () => {
    const out = new Flatten(0).forward(tensor(3));
    expect(out.shape).toEqual([1]);
  });

  it("Flatten rejects non-integer dims in the constructor", () => {
    expect(() => new Flatten(0.5)).toThrow(InvalidParameterError);
    expect(() => new Flatten(1, Number.NaN)).toThrow(InvalidParameterError);
  });

  it("Flatten keeps working for ordinary shapes and reports the tensor rank in errors", () => {
    const x = f64(
      Array.from({ length: 24 }, (_, i) => i),
      [2, 3, 4]
    );
    expect(new Flatten().forward(x).shape).toEqual([2, 12]);
    expect(new Flatten(0, -1).forward(x).shape).toEqual([24]);
    expect(() => new Flatten(5).forward(x)).toThrow(/out of range for 3-D tensor/);
  });

  it("Unflatten infers one -1 entry", () => {
    const x = f64(
      Array.from({ length: 100 }, (_, i) => i),
      [2, 50]
    );
    expect(new Unflatten(1, [2, -1]).forward(x).shape).toEqual([2, 2, 25]);
    expect(new Unflatten(1, [-1, 5, 5]).forward(x).shape).toEqual([2, 2, 5, 5]);
    expect(new Unflatten(-1, [-1]).forward(x).shape).toEqual([2, 50]);
  });

  it("Unflatten rejects an indivisible inferred size, two -1 entries and bad dims", () => {
    const x = f64(
      Array.from({ length: 14 }, (_, i) => i),
      [2, 7]
    );
    expect(() => new Unflatten(1, [2, -1]).forward(x)).toThrow(ShapeError);
    expect(() => new Unflatten(1, [-1, -1])).toThrow(InvalidParameterError);
    expect(() => new Unflatten(1, [-2])).toThrow(InvalidParameterError);
    expect(() => new Unflatten(0.5, [2])).toThrow(InvalidParameterError);
    expect(() => new Unflatten(1, [7, 2]).forward(x)).toThrow(ShapeError);
  });

  it("Unflatten copies unflattenedSize", () => {
    const sizes = [2, 3];
    const layer = new Unflatten(1, sizes);
    sizes[0] = 5;
    const x = f64(
      Array.from({ length: 12 }, (_, i) => i),
      [2, 6]
    );
    expect(layer.forward(x).shape).toEqual([2, 2, 3]);
  });
});

// ---------------------------------------------------------------------------
// Upsample
// ---------------------------------------------------------------------------

describe("Upsample (v1.5.0)", () => {
  const input = (dtype: "float32" | "float64" = "float64"): Tensor =>
    tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13], { dtype }).reshape([1, 1, 3, 4]);

  function runWithGrad(layer: Upsample, dtype: "float32" | "float64" = "float64") {
    const x = GradTensor.fromTensor(input(dtype), { requiresGrad: true });
    const y = layer.forward(x);
    const c = Array.from({ length: y.size }, (_, i) => i * 0.1 + 0.3);
    const weights = GradTensor.fromTensor(tensor(c, { dtype }).reshape([...y.shape]));
    y.mul(weights).sum().backward();
    return { x, y };
  }

  const cases: Array<[string, ConstructorParameters<typeof Upsample>[0]]> = [
    ["up_bil_noac", { size: [4, 6], mode: "bilinear", alignCorners: false }],
    ["up_bil_ac", { size: [4, 6], mode: "bilinear", alignCorners: true }],
    ["up_near_15", { scaleFactor: 1.5, mode: "nearest" }],
    ["up_near_size", { size: [5, 7], mode: "nearest" }],
    ["up_bil_noac_sf", { scaleFactor: [2, 1.5], mode: "bilinear", alignCorners: false }],
  ];

  for (const [name, options] of cases) {
    it(`matches PyTorch for ${name} (output and input gradient)`, () => {
      const ref = UP[name] as { shape: number[]; out: number[]; grad: number[] };
      const { x, y } = runWithGrad(new Upsample(options));
      expect([...y.shape]).toEqual(ref.shape);
      expectClose(values(y), ref.out);
      expectClose(values(x.grad as Tensor), ref.grad);
    });
  }

  it("uses floor(size * scaleFactor) for the output size (PyTorch), not rounding", () => {
    // 3 * 1.5 = 4.5 -> 4, 4 * 1.5 = 6
    const y = new Upsample({ scaleFactor: 1.5 }).forward(input());
    expect([...y.shape]).toEqual([1, 1, 4, 6]);
  });

  it("nearest indices use a float32 ratio and an unchanged axis is copied (PyTorch)", () => {
    // torch.nn.functional.interpolate(arange(33).reshape(1,1,33,1), scale_factor=(1.1, 1), mode="nearest")
    // picks 30 for output row 33 (33 / 1.1 is exactly 30, but 33 / 1.1 in doubles is just below).
    const column = (n: number): Tensor =>
      f64(Array.from({ length: n }, (_, i) => i)).reshape([1, 1, n, 1]);
    const y = new Upsample({ scaleFactor: [1.1, 1] }).forward(column(33));
    expect([...y.shape]).toEqual([1, 1, 36, 1]);
    expect(values(y)).toEqual([
      0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 20, 21, 22,
      23, 24, 25, 26, 27, 28, 29, 30, 30, 31,
    ]);
    // 7 * 1.1 floors to 7: the axis is copied in both modes
    for (const mode of ["nearest", "bilinear"] as const) {
      const same = new Upsample({ scaleFactor: [1.1, 1], mode, alignCorners: false }).forward(
        column(7)
      );
      expect(values(same)).toEqual([0, 1, 2, 3, 4, 5, 6]);
    }
  });

  it("keeps float32 inputs in float32 and returns a gradient of the same dtype", () => {
    const { x, y } = runWithGrad(new Upsample({ scaleFactor: 2, mode: "bilinear" }), "float32");
    expect(y.dtype).toBe("float32");
    expect(x.grad?.dtype).toBe("float32");
  });

  it("can be used twice on one float32 graph (gradients accumulate without a dtype clash)", () => {
    const x = GradTensor.fromTensor(input("float32"), { requiresGrad: true });
    const up = new Upsample({ scaleFactor: 2 });
    const a = up.forward(x).sum();
    const b = up.forward(x).sum();
    expect(() => a.add(b).backward()).not.toThrow();
    // every input pixel is copied 4 times per call
    expectClose(values(x.grad as Tensor), new Array(12).fill(8), 1e-6);
  });

  it("integer inputs produce float64 output", () => {
    const x = tensor([1, 2, 3, 4], { dtype: "int32" }).reshape([1, 1, 2, 2]);
    const y = new Upsample({ scaleFactor: 2 }).forward(x);
    expect(y.dtype).toBe("float64");
    expect(values(y).slice(0, 4)).toEqual([1, 1, 2, 2]);
  });

  it("works on a non-contiguous input", () => {
    const base = tensor([1, 2, 3, 4, 5, 6], { dtype: "float64" }).reshape([1, 1, 2, 3]);
    const view = transpose(base, [0, 1, 3, 2]); // (1, 1, 3, 2)
    const y = new Upsample({ size: [3, 4] }).forward(view);
    expectClose(
      values(y),
      values(new Upsample({ size: [3, 4] }).forward(f64(values(view), [1, 1, 3, 2])))
    );
    expectClose(values(y).slice(0, 4), [1, 1, 4, 4]);
  });

  it("rejects a zero-size output and empty spatial dims instead of returning garbage", () => {
    const x = tensor([1, 2], { dtype: "float64" }).reshape([1, 1, 1, 2]);
    expect(() => new Upsample({ scaleFactor: 0.4 }).forward(x)).toThrow(ShapeError);
    const empty = tensor([], { dtype: "float64" }).reshape([1, 1, 0, 2]);
    expect(() => new Upsample({ scaleFactor: 2 }).forward(empty)).toThrow(ShapeError);
  });

  it("supports an empty batch", () => {
    const empty = tensor([], { dtype: "float64" }).reshape([0, 1, 2, 2]);
    expect([...new Upsample({ scaleFactor: 2 }).forward(empty).shape]).toEqual([0, 1, 4, 4]);
  });

  it("a non-finite neighbour does not leak through zero-weight taps", () => {
    const x = tensor([1, Number.POSITIVE_INFINITY, 3, 4], { dtype: "float64" }).reshape([
      1, 1, 2, 2,
    ]);
    const y = values(new Upsample({ scaleFactor: 2, mode: "nearest" }).forward(x));
    expect(y.filter((v) => v === Number.POSITIVE_INFINITY).length).toBe(4);
    const bil = values(new Upsample({ size: [2, 2], mode: "bilinear" }).forward(x));
    expect(bil).toEqual([1, Number.POSITIVE_INFINITY, 3, 4]);
  });

  it("validates options and describes itself", () => {
    expect(() => new Upsample({ scaleFactor: [2] as unknown as [number, number] })).toThrow(
      InvalidParameterError
    );
    expect(() => new Upsample({ scaleFactor: [2, 0] })).toThrow(/positive/);
    expect(() => new Upsample({ size: 0 })).toThrow(/positive integers/);
    expect(() => new Upsample({ size: 2.5 })).toThrow(/positive integers/);
    expect(() => new Upsample({ size: 4, alignCorners: 1 as unknown as boolean })).toThrow(
      InvalidParameterError
    );
    expect(new Upsample({ size: 4 }).toString()).toBe("Upsample(size=[4, 4], mode='nearest')");
    expect(new Upsample({ scaleFactor: [2, 3], mode: "bilinear" }).toString()).toBe(
      "Upsample(scale_factor=[2, 3], mode='bilinear', align_corners=true)"
    );
  });
});

// ---------------------------------------------------------------------------
// Cross entropy
// ---------------------------------------------------------------------------

describe("crossEntropyLoss (v1.5.0)", () => {
  const X = [2.0, 1.0, 0.1, 0.5, 2.5, -1.0, -0.3, 0.2, 1.5, 1.2, -0.7, 0.4];
  const Y = [0, 1, 2, 1];
  const Yig = [0, -100, 2, 1];
  const W = [1.0, 2.0, 0.5];
  const P = [0.7, 0.2, 0.1, 0.1, 0.8, 0.1, 0.2, 0.2, 0.6, 0.3, 0.3, 0.4];
  const logits = () => GradTensor.fromTensor(f64(X, [4, 3]), { requiresGrad: true });

  function check(name: string, target: Tensor, options: Parameters<typeof crossEntropyLoss>[2]) {
    const ref = CE[name] as { loss: number[]; grad: number[] };
    const x = logits();
    const loss = crossEntropyLoss(x, target, options);
    expectClose(values(loss), ref.loss);
    (loss.size > 1 ? loss.sum() : loss).backward();
    expectClose(values(x.grad as Tensor), ref.grad);
  }

  it("class weights (mean divides by the sum of target weights)", () => {
    check("ce_weight", i32(Y), { weight: W });
  });
  it("ignoreIndex", () => {
    check("ce_ignore", i32(Yig), { ignoreIndex: -100 });
  });
  it("labelSmoothing", () => {
    check("ce_ls", i32(Y), { labelSmoothing: 0.1 });
  });
  it("weights + smoothing + ignoreIndex together", () => {
    check("ce_all", i32(Yig), { weight: W, labelSmoothing: 0.2, ignoreIndex: -100 });
  });
  it("reduction none and sum", () => {
    check("ce_none", i32(Y), { reduction: "none" });
    check("ce_sum", i32(Y), { reduction: "sum" });
  });
  it("probability targets with weights and with smoothing", () => {
    check("ce_soft_w", f64(P, [4, 3]), { weight: W });
    check("ce_soft_ls", f64(P, [4, 3]), { labelSmoothing: 0.3 });
  });

  it("returns a number for plain tensors and a Tensor for reduction none", () => {
    expect(typeof crossEntropyLoss(f64(X, [4, 3]), i32(Y))).toBe("number");
    const none = crossEntropyLoss(f64(X, [4, 3]), i32(Y), { reduction: "none" });
    expect([...none.shape]).toEqual([4]);
    expectClose(values(none), (CE.ce_none as { loss: number[] }).loss);
  });

  it("-Infinity logits of unused classes do not turn the loss into NaN", () => {
    const ref = CE.ce_masked as { loss: number[]; grad: number[] };
    const x = GradTensor.fromTensor(
      f64([1, Number.NEGATIVE_INFINITY, 0, 0, 2, Number.NEGATIVE_INFINITY], [2, 3]),
      { requiresGrad: true }
    );
    const loss = crossEntropyLoss(x, i32([0, 1]));
    expectClose(values(loss), ref.loss);
    loss.backward();
    expectClose(values(x.grad as Tensor), ref.grad);
  });

  it("every sample ignored gives NaN, as in PyTorch", () => {
    const v = crossEntropyLoss(f64(X, [4, 3]), i32([-100, -100, -100, -100]), {
      ignoreIndex: -100,
    });
    expect(Number.isNaN(v)).toBe(true);
  });

  it("a float32 target of class indices and a float64 soft target against float32 logits work", () => {
    const x = GradTensor.fromTensor(
      tensor(
        [
          [1, 2],
          [3, 1],
        ],
        { dtype: "float32" }
      ),
      {
        requiresGrad: true,
      }
    );
    const soft = tensor(
      [
        [0, 1],
        [1, 0],
      ],
      { dtype: "float64" }
    );
    expect(() => crossEntropyLoss(x, soft).backward()).not.toThrow();
  });

  it("validates the options", () => {
    const x = f64(X, [4, 3]);
    expect(() => crossEntropyLoss(x, i32(Y), { labelSmoothing: 1.5 })).toThrow(
      InvalidParameterError
    );
    expect(() => crossEntropyLoss(x, i32(Y), { labelSmoothing: Number.NaN })).toThrow(
      InvalidParameterError
    );
    expect(() => crossEntropyLoss(x, i32(Y), { ignoreIndex: 0.5 })).toThrow(InvalidParameterError);
    expect(() => crossEntropyLoss(x, i32(Y), { weight: [1, 2] })).toThrow(ShapeError);
    expect(() => crossEntropyLoss(x, i32(Y), { weight: [1, Number.NaN, 2] })).toThrow(
      InvalidParameterError
    );
    expect(() => crossEntropyLoss(x, i32(Y), { reduction: "avg" as unknown as "mean" })).toThrow(
      InvalidParameterError
    );
    expect(() => crossEntropyLoss(tensor([[], []], { dtype: "float64" }), i32([0, 0]))).toThrow(
      ShapeError
    );
  });
});

describe("binaryCrossEntropyWithLogitsLoss (v1.5.0)", () => {
  const BX = [2.0, -1.0, 0.5, -3.0, 0.2, 4.0];
  const BZ = [1, 0, 1, 0, 1, 1];
  const PW = [2.0, 1.0, 0.5];

  function check(name: string, options?: Parameters<typeof binaryCrossEntropyWithLogitsLoss>[2]) {
    const ref = BCE[name] as { loss: number[]; grad: number[] };
    const x = GradTensor.fromTensor(f64(BX, [2, 3]), { requiresGrad: true });
    const loss = binaryCrossEntropyWithLogitsLoss(x, f64(BZ, [2, 3]), options);
    expectClose(values(loss), ref.loss);
    (loss.size > 1 ? loss.sum() : loss).backward();
    expectClose(values(x.grad as Tensor), ref.grad);
  }

  it("accepts same-shaped multi-label (N, C) input and matches PyTorch", () => {
    check("bce_multi");
    check("bce_sum", { reduction: "sum" });
    check("bce_none", { reduction: "none" });
  });

  it("posWeight (number-free tensor broadcast over labels)", () => {
    check("bce_pw", { posWeight: f64(PW) });
  });

  it("integer labels are converted to the logit dtype", () => {
    const v = binaryCrossEntropyWithLogitsLoss(f64([2, -1]), i32([1, 0]));
    expect(v).toBeCloseTo((Math.log1p(Math.exp(-2)) + Math.log1p(Math.exp(-1))) / 2, 12);
  });

  it("stays finite for extreme logits", () => {
    const none = binaryCrossEntropyWithLogitsLoss(f64([-800, 800, 0]), f64([1, 0, 1]), {
      reduction: "none",
    });
    expectClose(values(none), [800, 800, Math.LN2]);
  });

  it("has the exact gradient sigmoid(x) - z at x = 0 (zero-initialised logits)", () => {
    // torch.nn.functional.binary_cross_entropy_with_logits, float64, reduction="sum"
    const plain = GradTensor.fromTensor(f64([0, 1, -1]), { requiresGrad: true });
    binaryCrossEntropyWithLogitsLoss(plain, f64([1, 0, 1]), { reduction: "sum" }).backward();
    expectClose(values(plain.grad as Tensor), [-0.5, 0.7310585786300049, -0.7310585786300049]);

    const weighted = GradTensor.fromTensor(f64([0, 1, -1]), { requiresGrad: true });
    binaryCrossEntropyWithLogitsLoss(weighted, f64([1, 0, 1]), {
      reduction: "sum",
      posWeight: 2,
    }).backward();
    expectClose(values(weighted.grad as Tensor), [-1, 0.7310585786300049, -1.4621171572600098]);
  });

  it("keeps the (N,) / (N, 1) pairing and its errors", () => {
    const v = binaryCrossEntropyWithLogitsLoss(f64([0.5, -0.5], [2, 1]), f64([1, 0]));
    expect(v).toBeCloseTo(Math.log1p(Math.exp(-0.5)), 12);
    expect(() => binaryCrossEntropyWithLogitsLoss(f64([1, 2, 3]), f64([1, 0]))).toThrow(
      /Batch size mismatch/
    );
  });
});

// ---------------------------------------------------------------------------
// Other losses
// ---------------------------------------------------------------------------

describe("nllLoss (v1.5.0)", () => {
  const LP = NLL_LP;
  it("class weights and ignoreIndex match PyTorch, with gradients", () => {
    const ref = NLL.nll_w as { loss: number[]; grad: number[] };
    const x = GradTensor.fromTensor(f64(LP, [4, 3]), { requiresGrad: true });
    const loss = nllLoss(x, i32([0, -100, 2, 1]), { weight: [1, 2, 0.5], ignoreIndex: -100 });
    expectClose(values(loss), ref.loss);
    loss.backward();
    expectClose(values(x.grad as Tensor), ref.grad);
  });

  it("still accepts a reduction string and returns Tensors for Tensor inputs", () => {
    const none = nllLoss(f64(LP, [4, 3]), i32([0, 1, 2, 1]), "none");
    expect([...none.shape]).toEqual([4]);
    const mean = nllLoss(f64(LP, [4, 3]), i32([0, 1, 2, 1]));
    expect(mean.shape).toEqual([]);
  });

  it("rejects fractional and NaN targets instead of rounding them", () => {
    expect(() => nllLoss(f64(LP, [4, 3]), f64([0, 1.4, 2, 1]))).toThrow(/integer/);
    expect(() => nllLoss(f64(LP, [4, 3]), f64([0, Number.NaN, 2, 1]))).toThrow(
      InvalidParameterError
    );
    expect(() => nllLoss(f64(LP, [4, 3]), i32([0, 3, 2, 1]))).toThrow(/out of range/);
  });
});

describe("regression losses (v1.5.0)", () => {
  const HP = [1.0, 2.5, -0.5, 3.0, 0.0];
  const HT = [1.0, 1.5, 0.5, 0.0, 0.2];

  function grad(fn: (p: GradTensor) => GradTensor) {
    const p = GradTensor.fromTensor(f64(HP), { requiresGrad: true });
    const loss = fn(p);
    loss.sum().backward();
    return { loss, grad: values(p.grad as Tensor) };
  }

  it("huberLoss supports autograd and matches PyTorch (zero diff, boundary, linear zone)", () => {
    const ref = HUBER.huber as { loss: number[]; grad: number[] };
    const { loss, grad: g } = grad((p) => huberLoss(p, f64(HT), 1.0, "none"));
    expectClose(values(loss), ref.loss);
    expectClose(g, ref.grad);
    expectClose(values(huberLoss(f64(HP), f64(HT), 1.0, "none")), ref.loss);
  });

  it("smoothL1Loss supports autograd and beta = 0 (L1)", () => {
    for (const [name, beta] of [
      ["sl1_b0", 0],
      ["sl1_b2", 2],
    ] as const) {
      const ref = HUBER[name] as { loss: number[]; grad: number[] };
      const { loss, grad: g } = grad((p) => smoothL1Loss(p, f64(HT), beta, "none"));
      expectClose(values(loss), ref.loss);
      expectClose(g, ref.grad);
      expectClose(values(smoothL1Loss(f64(HP), f64(HT), beta, "none")), ref.loss);
    }
    expect(() => smoothL1Loss(f64(HP), f64(HT), -1)).toThrow(/non-negative/);
  });

  it("autograd losses match (N, 1) with (N,) element-wise instead of broadcasting to (N, N)", () => {
    const pred = GradTensor.fromTensor(f64([1, 2, 3], [3, 1]), { requiresGrad: true });
    const target = f64([1, 2, 5]);
    const loss = mseLoss(pred, target);
    expect(values(loss)[0]).toBeCloseTo(4 / 3, 12);
    loss.backward();
    expectClose(values(pred.grad as Tensor), [0, 0, -4 / 3]);
    expect(values(maeLoss(pred, target))[0]).toBeCloseTo(2 / 3, 12);
    expect(values(rmseLoss(pred, target))[0]).toBeCloseTo(Math.sqrt(4 / 3), 12);
    expect(
      values(binaryCrossEntropyLoss(GradTensor.fromTensor(f64([0.5, 0.5], [2, 1])), f64([1, 0])))[0]
    ).toBeCloseTo(Math.LN2, 6);
  });

  it("autograd losses accept targets of a different dtype", () => {
    const pred = GradTensor.fromTensor(tensor([1, 2], { dtype: "float32" }), {
      requiresGrad: true,
    });
    expect(() => mseLoss(pred, f64([1, 3])).backward()).not.toThrow();
    expect(values(mseLoss(pred, i32([1, 3])))[0]).toBeCloseTo(0.5, 6);
  });

  it("binaryCrossEntropyLoss accepts (N, 1) with (N,) on plain tensors too", () => {
    const v = binaryCrossEntropyLoss(f64([0.25, 0.75], [2, 1]), f64([0, 1]));
    expect(values(v)[0]).toBeCloseTo(-Math.log(0.75), 9);
  });
});

describe("klDivLoss (v1.5.0)", () => {
  const Q = KL_Q;
  const P = [0.0, 0.6, 0.4, 0.5, 0.25, 0.25];

  it("matches PyTorch for mean, batchmean and none, with input gradients", () => {
    for (const [name, reduction] of [
      ["kl_mean", "mean"],
      ["kl_bm", "batchmean"],
      ["kl_none", "none"],
    ] as const) {
      const ref = KL[name] as { loss: number[]; grad: number[] };
      const x = GradTensor.fromTensor(f64(Q, [2, 3]), { requiresGrad: true });
      const loss = klDivLoss(x, f64(P, [2, 3]), reduction);
      expectClose(values(loss), ref.loss);
      (loss.size > 1 ? loss.sum() : loss).backward();
      expectClose(values(x.grad as Tensor), ref.grad);
      expectClose(values(klDivLoss(f64(Q, [2, 3]), f64(P, [2, 3]), reduction)), ref.loss);
    }
  });

  it("supports log-space targets", () => {
    expectClose(
      values(klDivLoss(f64(Q, [2, 3]), f64(KL_LT, [2, 3]), "batchmean", true)),
      KL_LOGT,
      1e-9
    );
  });

  it("rejects an unknown reduction instead of silently averaging", () => {
    expect(() => klDivLoss(f64(Q, [2, 3]), f64(P, [2, 3]), "avg" as unknown as "mean")).toThrow(
      InvalidParameterError
    );
  });

  it("NaN and negative targets give NaN, not 0", () => {
    const q = f64([-1, -1, -1]);
    expect(
      values(klDivLoss(q, f64([0.5, Number.NaN, -0.2]), "none"))
        .slice(1)
        .every(Number.isNaN)
    ).toBe(true);
    expect(values(klDivLoss(q, f64([0.5, 0, 0.5]), "none"))[1]).toBe(0);
  });
});

describe("embedding losses (v1.5.0)", () => {
  it("cosineEmbeddingLoss matches PyTorch (including a zero vector)", () => {
    const a = f64([1, 2, 3, 0.5, -1, 2, 0, 0, 0], [3, 3]);
    const b = f64([2, 1, 0, -0.5, 1, -2, 1, 2, 3], [3, 3]);
    expectClose(values(cosineEmbeddingLoss(a, b, f64([1, -1, 1]), -0.2, "none")), COS, 1e-9);
  });

  it("cosineEmbeddingLoss rejects bad labels, mismatched shapes and >2D inputs", () => {
    const a = f64([1, 2, 3, 4], [2, 2]);
    expect(() => cosineEmbeddingLoss(a, a, f64([1, 0]))).toThrow(InvalidParameterError);
    expect(() => cosineEmbeddingLoss(a, a, f64([1, Number.NaN]))).toThrow(InvalidParameterError);
    expect(() => cosineEmbeddingLoss(a, f64([1, 2, 3], [1, 3]), f64([1, 1]))).toThrow(ShapeError);
    expect(() => cosineEmbeddingLoss(a, a, f64([1]))).toThrow(ShapeError);
    const cube = f64([1, 2, 3, 4], [1, 2, 2]);
    expect(() => cosineEmbeddingLoss(cube, cube, f64([1]))).toThrow(ShapeError);
  });

  it("tripletMarginLoss rejects inputs with more than two dimensions", () => {
    const cube = f64([1, 2, 3, 4], [1, 2, 2]);
    expect(() => tripletMarginLoss(cube, cube, cube)).toThrow(ShapeError);
  });

  it("marginRankingLoss with float64 GradTensors (used to fail with a dtype mismatch)", () => {
    const ref = MR as { loss: number[]; grad: number[] };
    const x1 = GradTensor.fromTensor(f64([0.5, 1.5, -1.0]), { requiresGrad: true });
    const loss = marginRankingLoss(x1, f64([1.0, 0.5, -2.0]), f64([1, 1, -1]), 0.3);
    expectClose(values(loss), ref.loss);
    loss.backward();
    expectClose(values(x1.grad as Tensor), ref.grad);
  });

  it("marginRankingLoss accepts a single shared label and a same-size label of another shape", () => {
    const x1 = f64([1, 2, 3]);
    const x2 = f64([0, 0, 0]);
    expectClose(values(marginRankingLoss(x1, x2, f64([-1]), 0, "none")), [1, 2, 3]);
    expectClose(values(marginRankingLoss(x1, x2, f64([1]), 0, "none")), [0, 0, 0]);
    expectClose(values(marginRankingLoss(x1, x2, f64([1, 1, 1], [3, 1]), 5, "none")), [4, 3, 2]);
    const g = GradTensor.fromTensor(x1, { requiresGrad: true });
    expectClose(values(marginRankingLoss(g, x2, f64([1, 1, 1], [3, 1]), 5, "none")), [4, 3, 2]);
    expect(() => marginRankingLoss(x1, x2, f64([1, 1]))).toThrow(ShapeError);
    expect(() => marginRankingLoss(g, f64([1, 2]), f64([1, 1, 1]))).toThrow(ShapeError);
  });
});

describe("probabilistic losses (v1.5.0)", () => {
  const GI = f64([0.5, 1.0, -1.0, 2.0], [2, 2]);
  const GT = f64([0.0, 1.5, -0.5, 1.0], [2, 2]);

  it("gaussianNLLLoss accepts a per-sample variance with or without a trailing 1", () => {
    expectClose(
      values(gaussianNLLLoss(GI, GT, f64([0.5, 2.0], [2, 1]), { reduction: "none" })),
      G_ROW
    );
    expectClose(values(gaussianNLLLoss(GI, GT, f64([0.5, 2.0]), { reduction: "none" })), G_ROW0);
    expect(() => gaussianNLLLoss(GI, GT, f64([1, 2, 3]))).toThrow(ShapeError);
  });

  it("gaussianNLLLoss rejects a negative variance instead of silently clamping it", () => {
    expect(() => gaussianNLLLoss(GI, GT, f64([1, -0.1, 1, 1], [2, 2]))).toThrow(
      InvalidParameterError
    );
    // zero is allowed and clamped to eps
    expect(
      Number.isFinite(values(gaussianNLLLoss(GI, GT, f64([0, 1, 1, 1], [2, 2])))[0] as number)
    ).toBe(true);
  });
});

describe("ctcLoss (v1.5.0)", () => {
  const LP = f64(CTC_LP, [6, 2, 4]);
  const lengths = (a: number[]) => i32(a);

  it("mean divides each loss by its target length before averaging (PyTorch)", () => {
    const v = ctcLoss(LP, i32([1, 2, 2, 3, 1]), lengths([6, 5]), lengths([3, 2]));
    expectClose(values(v), CTC.ctc_mean as number[], 1e-9);
    const none = ctcLoss(LP, i32([1, 2, 2, 3, 1]), lengths([6, 5]), lengths([3, 2]), {
      reduction: "none",
    });
    expectClose(values(none), CTC.ctc_none as number[], 1e-9);
    const sum = ctcLoss(LP, i32([1, 2, 2, 3, 1]), lengths([6, 5]), lengths([3, 2]), {
      reduction: "sum",
    });
    expect(values(sum)[0]).toBeCloseTo(
      (CTC.ctc_none as number[]).reduce((a, b) => a + b, 0),
      9
    );
  });

  it("accepts padded (N, S) targets", () => {
    const padded = i32([1, 2, 2, 3, 1, 0]).reshape([2, 3]);
    const v = ctcLoss(LP, padded, lengths([6, 5]), lengths([3, 2]), { reduction: "none" });
    expectClose(values(v), CTC.ctc_pad as number[], 1e-9);
  });

  it("allows an empty target", () => {
    const v = ctcLoss(LP, i32([3, 1]), lengths([6, 5]), lengths([0, 2]), { reduction: "none" });
    expectClose(values(v), CTC.ctc_empty as number[], 1e-9);
  });

  it("zeroInfinity replaces impossible alignments", () => {
    const inf = ctcLoss(LP, i32([1, 2, 2, 3, 1]), lengths([2, 5]), lengths([3, 2]), {
      reduction: "none",
    });
    expect(values(inf)[0]).toBe(Number.POSITIVE_INFINITY);
    const zeroed = ctcLoss(LP, i32([1, 2, 2, 3, 1]), lengths([2, 5]), lengths([3, 2]), {
      reduction: "none",
      zeroInfinity: true,
    });
    expect(values(zeroed)[0]).toBe(0);
    // a repeated label needs an extra blank frame: 3 frames cannot emit [2, 2, 3]
    const tight = ctcLoss(LP, i32([2, 2, 3, 1]), lengths([3, 5]), lengths([3, 1]), {
      reduction: "none",
    });
    expect(values(tight)[0]).toBe(Number.POSITIVE_INFINITY);
  });

  it("reports unclear input as typed errors", () => {
    const tg = i32([1, 2, 2, 3, 1]);
    expect(() => ctcLoss(LP, i32([1, 2, 2]), lengths([6, 5]), lengths([3, 2]))).toThrow(ShapeError);
    expect(() => ctcLoss(LP, tg, lengths([6, 5]), f64([3, 1.5]))).toThrow(InvalidParameterError);
    expect(() => ctcLoss(LP, tg, lengths([7, 5]), lengths([3, 2]))).toThrow(InvalidParameterError);
    expect(() => ctcLoss(LP, tg, lengths([6, 5]), lengths([-1, 2]))).toThrow(InvalidParameterError);
    expect(() => ctcLoss(LP, i32([1, 2, 9, 3, 1]), lengths([6, 5]), lengths([3, 2]))).toThrow(
      /out of range/
    );
    expect(() =>
      ctcLoss(LP, tg, lengths([6, 5]), lengths([3, 2]), { reduction: "x" as "mean" })
    ).toThrow(InvalidParameterError);
  });
});
