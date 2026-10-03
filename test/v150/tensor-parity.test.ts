import { describe, expect, it } from "vitest";
import type { Backend } from "../../src/core";
import {
  DeviceError,
  DTypeError,
  InvalidParameterError,
  isBackendAvailable,
  registerBackend,
  ShapeError,
  unregisterBackend,
} from "../../src/core";
import { GradTensor, noGrad, parameter, Tensor, tensor } from "../../src/ndarray";

// Reference values from torch 2.12 (float32 inputs, float64 printout).
const REF = JSON.parse(
  '{"elu":[[0.30000001192092896,-0.4891640543937683,2.5],[-0.3523902893066406,1.899999976158142,-0.6224377751350403]],"gelu_tanh":[[0.18537093698978424,-0.1382972002029419,2.4849157333374023],[-0.16942985355854034,1.8454512357711792,-0.030321503058075905]],"gelu_none":[[0.1853734254837036,-0.1380835920572281,2.48447585105896],[-0.16937455534934998,1.8454384803771973,-0.030587567016482353]],"expm1":[[0.349858820438385,-0.6988058090209961,11.182494163513184],[-0.5034146904945374,5.68589448928833,-0.8891968727111816]],"log1p":[[0.2623642683029175,0.7884573936462402,1.2527629137039185],[0.5306282639503479,1.0647107362747192,1.1631507873535156]],"tan":[[0.30933627486228943,-2.5721518993377686,-0.7470223307609558],[-0.8422883749008179,-2.927097797393799,1.3738229274749756]],"maximum":[[1.100000023841858,-0.4000000059604645,2.5],[0.20000000298023224,1.899999976158142,3.0]],"minimum":[[0.30000001192092896,-1.2000000476837158,0.6000000238418579],[-0.699999988079071,-1.5,-2.200000047683716]],"maximum_s":[[0.5,0.5,2.5],[0.5,1.899999976158142,0.5]],"minimum_s":[[0.30000001192092896,-1.2000000476837158,0.5],[-0.699999988079071,0.5,-2.200000047683716]],"softplus":[[0.8543552756309509,0.26328244805336,2.578889846801758],[0.40318605303764343,2.039386749267578,0.10508330911397934]],"mish":[[0.20800139009952545,-0.3088357746601105,2.4713923931121826],[-0.2678702473640442,1.8367435932159424,-0.23033608496189117]],"swish":[[0.172332763671875,-0.2777702808380127,2.310354471206665],[-0.2322685569524765,1.6527940034866333,-0.21945106983184814]],"selu":[[0.3152103126049042,-1.2285699844360352,2.6267526149749756],[-0.8850530385971069,1.9963319301605225,-1.5632964372634888]],"logSigmoid":[[-0.5543552041053772,-1.4632824659347534,-0.07888973504304886],[-1.103186011314392,-0.13938675820827484,-2.3050832748413086]],"hardtanh":[[0.30000001192092896,-0.5,0.800000011920929],[-0.5,0.800000011920929,-0.5]],"leakyRelu":[[0.30000001192092896,-0.24000000953674316,2.5],[-0.14000000059604645,1.899999976158142,-0.4400000274181366]],"tanhshrink":[[0.008687376976013184,-0.3663454055786133,1.5133857727050781],[-0.09563219547271729,0.9437625408172607,-1.224256992340088]],"gather":[[2.5,0.30000001192092896,0.30000001192092896],[-2.200000047683716,-0.699999988079071,-0.699999988079071]],"gather0":[[-0.699999988079071,1.899999976158142,-2.200000047683716],[-0.699999988079071,1.899999976158142,-2.200000047683716],[0.30000001192092896,-1.2000000476837158,2.5]],"flip1":[[[2.5,-1.2000000476837158,0.30000001192092896],[-2.200000047683716,1.899999976158142,-0.699999988079071]],[[3.0,2.0,1.0],[6.0,5.0,4.0]]],"flipAll":[[[-2.200000047683716,1.899999976158142,-0.699999988079071],[2.5,-1.2000000476837158,0.30000001192092896]],[[6.0,5.0,4.0],[3.0,2.0,1.0]]],"sort1":[[[1.0,1.0,2.0,3.0],[-1.0,0.0,5.0,5.0]],[[4.0,1.0,3.0,2.0],[6.0,7.0,8.0,5.0]]],"sort1d":[[[3.0,2.0,1.0,1.0],[5.0,5.0,0.0,-1.0]],[[1.0,3.0,2.0,4.0],[7.0,5.0,6.0,8.0]]],"sort0":[[[0.0,1.0,2.0,-1.0],[3.0,5.0,5.0,1.0]],[[5.0,2.0,3.0,8.0],[1.0,6.0,7.0,4.0]]],"argsort1":[[1,3,2,0],[3,0,1,2]],"argsort1d":[[0,2,1,3],[1,2,0,3]],"argmax1":[2,1],"argmin0":[1,0,1],"argmaxAll":2,"argminKeep":[[1],[2]],"powT":[[[0.25,1.2247449159622192,0.5],[5.196152210235596,0.0,1.0]],[[1.0,0.8164966106414795,-0.75],[10.392304420471191,0.0,0.0]],[[-0.1732867956161499,0.9931826591491699,1.0397207736968994],[22.834226608276367,0.0,5.497744560241699]]],"powBc":[[[0.25,1.2247449159622192,2.8284270763397217],[9.0,0.8366600275039673,3.9528470039367676]],[[1.0,0.8164966106414795,6.3639607429504395],[24.0,2.9880714416503906,14.230250358581543]],[39.37675476074219,-0.4988957643508911,27.613292694091797]],"clipMin":[[[0.30000001192092896,0.30000001192092896,2.5],[0.30000001192092896,1.899999976158142,0.30000001192092896]],[[1.0,0.0,3.0],[0.0,5.0,0.0]]],"clipMax":[[[0.30000001192092896,-1.2000000476837158,0.30000001192092896],[-0.699999988079071,0.30000001192092896,-2.200000047683716]],[[1.0,2.0,0.0],[4.0,0.0,6.0]]],"maxNumGrad":[[[0.30000001192092896,0.30000001192092896,2.5],[0.30000001192092896,1.899999976158142,0.30000001192092896]],[[0.5,0.0,3.0],[0.0,5.0,0.0]]],"minNumGrad":[[[0.30000001192092896,-1.2000000476837158,0.30000001192092896],[-0.699999988079071,0.30000001192092896,-2.200000047683716]],[[0.5,2.0,0.0],[4.0,0.0,6.0]]],"comp":[[[-0.6875,-1.25,1.5625],[-1.1875,0.25,-0.375]],[[0.625,1.25,1.875],[2.5,3.125,3.75]]],"gt":[[false,false,true],[false,true,false]],"ge":[[true,false,true],[false,true,false]],"lt":[[true,true,false],[true,false,true]],"le":[[true,true,false],[true,false,true]],"eq":[[true,true,true],[true,true,true]],"ne":[[false,true,true],[true,true,true]],"floor":[0.0,1.0,2.0,-1.0,-2.0,2.0,1234.0,-4.0],"ceil":[1.0,2.0,3.0,-0.0,-1.0,3.0,1235.0,-3.0],"round":[0.0,2.0,2.0,-0.0,-2.0,2.0,1234.0,-3.0],"round2":[0.5,1.5,2.5,-0.5,-1.7000000476837158,2.3499999046325684,1234.5,-3.1700000762939453],"dot":[[[9.199999809265137,10.799999237060547],[-6.0,-7.000000953674316]],[[5.0,11.0,17.0],[11.0,25.0,39.0]]]}'
) as Record<string, unknown>;

const X = [
  [0.3, -1.2, 2.5],
  [-0.7, 1.9, -2.2],
];
const Y = [
  [1.1, -0.4, 0.6],
  [0.2, -1.5, 3.0],
];
const P = [
  [0.3, 1.2, 2.5],
  [0.7, 1.9, 2.2],
];
const W = [
  [1, 2, 3],
  [4, 5, 6],
];

function flat(input: unknown): number[] {
  const value = input instanceof Tensor || input instanceof GradTensor ? input.toArray() : input;
  return (Array.isArray(value) ? (value as unknown[]).flat(5) : [value]).map((v) =>
    typeof v === "boolean" ? (v ? 1 : 0) : Number(v)
  );
}
function close(actual: unknown, expected: unknown, tol = 2e-5): void {
  const got = flat(actual);
  const want = flat(expected);
  expect(got.length).toBe(want.length);
  for (let i = 0; i < want.length; i++) {
    const w = want[i] as number;
    const g = got[i] as number;
    if (Number.isNaN(w)) expect(Number.isNaN(g)).toBe(true);
    else expect(Math.abs(g - w)).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(w)));
  }
}
function ref(name: string): unknown {
  return REF[name];
}
function pair(name: string): [unknown, unknown] {
  const v = REF[name] as [unknown, unknown];
  return [v[0], v[1]];
}
/** Backward of `(f(x) * weights).sum()` on a fresh leaf; returns the output and the gradient. */
function weighted(
  data: number[][],
  fn: (x: GradTensor) => GradTensor,
  weights: number[][] = W
): { out: GradTensor; grad: Tensor } {
  const x = parameter(data);
  const out = fn(x);
  out
    .mul(GradTensor.fromTensor(tensor(weights)))
    .sum()
    .backward();
  const grad = x.grad;
  if (grad === null) throw new Error("expected a gradient");
  return { out, grad };
}

// ---------------------------------------------------------------------------
// Public surface
// ---------------------------------------------------------------------------

/** Public member names: prototype methods and getters plus instance fields, without `_` names. */
function memberNames(ctor: { prototype: object }, instance: object): Set<string> {
  const names = new Set<string>();
  for (const key of Object.getOwnPropertyNames(ctor.prototype)) names.add(key);
  for (const key of Object.getOwnPropertyNames(instance)) names.add(key);
  names.delete("constructor");
  for (const key of [...names]) if (key.startsWith("_")) names.delete(key);
  return names;
}

/**
 * Members that only make sense with gradient tracking. `backward`, `grad` and
 * `requiresGrad` are shared: on a plain Tensor they report "no gradient".
 */
const GRAD_ONLY = ["setGrad", "setRequiresGrad", "zeroGrad", "accumulateGrad", "hasGrad", "tensor"];

/**
 * Members that expose or guard raw storage. `GradTensor` reaches them through
 * `.tensor`: `fill` mutates data in place (not allowed on a tracked leaf),
 * `deviceBuffer` is the backend integration handle, and the rest are private
 * fields and helpers of `Tensor`.
 */
const STORAGE_ONLY = [
  "fill",
  "deviceBuffer",
  "bufferOwner",
  "disposed",
  "assertAlive",
  "isNumericTensor",
  "isStringTensor",
];

describe("Tensor and GradTensor share one method surface", () => {
  const tensorNames = memberNames(Tensor, tensor([1, 2]));
  const gradNames = memberNames(GradTensor, parameter([1, 2]));

  it("have the same public members apart from the allowlists", () => {
    const onlyTensor = [...tensorNames].filter((n) => !gradNames.has(n)).sort();
    const onlyGrad = [...gradNames].filter((n) => !tensorNames.has(n)).sort();
    expect(onlyTensor).toEqual([...STORAGE_ONLY].sort());
    expect(onlyGrad).toEqual([...GRAD_ONLY].sort());
  });

  it("keeps the allowlists free of members that both classes have", () => {
    for (const name of [...GRAD_ONLY, ...STORAGE_ONLY]) {
      expect(tensorNames.has(name) && gradNames.has(name)).toBe(false);
    }
  });
});

// ---------------------------------------------------------------------------
// Tensor: new fluent methods
// ---------------------------------------------------------------------------

describe("Tensor activation and math methods match torch", () => {
  const x = tensor(X);
  const y = tensor(Y);

  it("elu, selu, leakyRelu, hardtanh, tanhshrink", () => {
    close(x.elu(0.7), ref("elu"));
    close(x.selu(), ref("selu"));
    close(x.leakyRelu(0.2), ref("leakyRelu"));
    close(x.hardtanh(-0.5, 0.8), ref("hardtanh"));
    close(x.tanhshrink(), ref("tanhshrink"));
  });

  it("defaults equal the functional ops", () => {
    close(x.elu(), x.elu(1));
    close(x.leakyRelu(), x.leakyRelu(0.01));
    close(x.hardtanh(), x.hardtanh(-1, 1));
  });

  it("gelu takes the same options as the op", () => {
    close(x.gelu(), ref("gelu_tanh"));
    close(x.gelu({ approximate: "tanh" }), ref("gelu_tanh"));
    close(x.gelu("tanh"), ref("gelu_tanh"));
    close(x.gelu({ approximate: "none" }), ref("gelu_none"));
    close(x.gelu("none"), ref("gelu_none"));
    expect(() => x.gelu({ approximate: "bad" as "tanh" })).toThrow(InvalidParameterError);
  });

  it("softplus, mish, swish, logSigmoid", () => {
    close(x.softplus(), ref("softplus"));
    close(x.mish(), ref("mish"));
    close(x.swish(), ref("swish"));
    close(x.logSigmoid(), ref("logSigmoid"));
  });

  it("expm1, log1p, tan", () => {
    close(x.expm1(), ref("expm1"));
    close(tensor(P).log1p(), ref("log1p"));
    close(x.tan(), ref("tan"));
  });

  it("maximum and minimum take a tensor or a number", () => {
    close(x.maximum(y), ref("maximum"));
    close(x.minimum(y), ref("minimum"));
    close(x.maximum(0.5), ref("maximum_s"));
    close(x.minimum(0.5), ref("minimum_s"));
    close(tensor([Number.NaN, 1]).maximum(0), [Number.NaN, 1]);
  });

  it("maximum keeps the integer dtype for a whole number", () => {
    const r = tensor([1, 5, 3], { dtype: "int32" }).maximum(2);
    expect(r.dtype).toBe("int32");
    expect(r.toArray()).toEqual([2, 5, 3]);
  });

  it("gather selects along an axis", () => {
    close(x.gather(tensor([2, 0, 0]), 1), ref("gather"));
    close(x.gather(tensor([1, 1, 0]), 0), ref("gather0"));
    close(x.gather(tensor([1]), -1), [[-1.2], [1.9]]);
  });

  it("detach returns the tensor itself", () => {
    expect(x.detach()).toBe(x);
  });

  it("rejects bad operands", () => {
    expect(() => x.maximum("a" as unknown as number)).toThrow(InvalidParameterError);
  });
});

// ---------------------------------------------------------------------------
// GradTensor: new methods
// ---------------------------------------------------------------------------

describe("GradTensor.item", () => {
  it("reads a scalar loss", () => {
    const loss = parameter([1, 2, 3]).sum();
    expect(loss.item()).toBe(6);
    expect(parameter([[2.5]]).item()).toBe(2.5);
  });

  it("returns a bigint for int64", () => {
    const t = GradTensor.fromTensor(tensor([7], { dtype: "int64" }));
    expect(t.item()).toBe(7n);
  });

  it("throws for more than one element", () => {
    expect(() => parameter([1, 2]).item()).toThrow(ShapeError);
  });
});

describe("GradTensor squeeze and unsqueeze", () => {
  it("squeeze is a differentiable view", () => {
    const x = parameter([[[1, 2, 3]]]);
    const y = x.squeeze();
    expect(y.shape).toEqual([3]);
    y.mul(GradTensor.fromTensor(tensor([1, 2, 3])))
      .sum()
      .backward();
    expect(x.grad?.shape).toEqual([1, 1, 3]);
    close(x.grad, [1, 2, 3]);
  });

  it("squeeze with one axis or a list keeps the other size-1 axes", () => {
    const x = parameter(tensor([1, 2]).reshape([1, 2, 1]));
    expect(x.squeeze(0).shape).toEqual([2, 1]);
    expect(x.squeeze(-1).shape).toEqual([1, 2]);
    expect(x.squeeze([0, 2]).shape).toEqual([2]);
  });

  it("unsqueeze adds an axis and passes the gradient through", () => {
    const x = parameter([1, 2, 3]);
    const y = x.unsqueeze(1);
    expect(y.shape).toEqual([3, 1]);
    expect(x.unsqueeze(-1).shape).toEqual([3, 1]);
    expect(x.unsqueeze(0).shape).toEqual([1, 3]);
    y.mul(GradTensor.fromTensor(tensor([[4], [5], [6]])))
      .sum()
      .backward();
    expect(x.grad?.shape).toEqual([3]);
    close(x.grad, [4, 5, 6]);
  });

  it("track nothing when the input does not require grad", () => {
    const y = GradTensor.fromTensor(tensor([[1, 2]])).squeeze();
    expect(y.requiresGrad).toBe(false);
  });
});

describe("GradTensor non-differentiable results are plain Tensors", () => {
  const x = parameter(X);

  it("argmax and argmin match torch", () => {
    const a = x.argmax(1);
    expect(a).toBeInstanceOf(Tensor);
    expect(a.dtype).toBe("int32");
    close(a, ref("argmax1"));
    close(x.argmin(0), ref("argmin0"));
    expect(x.argmax().item()).toBe(ref("argmaxAll"));
    close(x.argmin(1, true), ref("argminKeep"));
    expect(x.argmin(1, true).shape).toEqual([2, 1]);
  });

  it("comparisons give bool tensors and accept number, Tensor and GradTensor", () => {
    const y = parameter(Y);
    const r = x.gt(0.3);
    expect(r).toBeInstanceOf(Tensor);
    expect(r.dtype).toBe("bool");
    close(r, ref("gt"));
    close(x.ge(0.3), ref("ge"));
    close(x.lt(y), ref("lt"));
    close(x.le(tensor(Y)), ref("le"));
    close(x.eq(x), ref("eq"));
    close(x.ne(0.3), ref("ne"));
  });

  it("comparisons of an integer tensor with a whole number keep exact integer rules", () => {
    const t = GradTensor.fromTensor(tensor([1, 2, 3], { dtype: "int32" }));
    close(t.eq(2), [0, 1, 0]);
    close(t.ge(2), [0, 1, 1]);
  });

  it("isnan, any and all", () => {
    const t = parameter([
      [1, Number.NaN],
      [0, 0],
    ]);
    expect(t.isnan().dtype).toBe("bool");
    close(t.isnan(), [
      [0, 1],
      [0, 0],
    ]);
    const b = GradTensor.fromTensor(
      tensor([
        [1, 0],
        [1, 1],
      ])
    );
    expect(Number(b.any().item())).toBe(1);
    expect(Number(b.all().item())).toBe(0);
    close(b.all(1), [0, 1]);
    expect(b.any([0, 1], true).shape).toEqual([1, 1]);
    expect(b.all(0, true).shape).toEqual([1, 2]);
  });

  it("argsort matches torch and keeps ties in order", () => {
    const t = parameter([
      [3, 1, 2, 1],
      [0, 5, 5, -1],
    ]);
    close(t.argsort(1), ref("argsort1"));
    close(t.argsort(1, true), ref("argsort1d"));
    expect(t.argsort()).toBeInstanceOf(Tensor);
  });
});

describe("GradTensor rounding has a zero gradient", () => {
  const V = [0.5, 1.5, 2.5, -0.5, -1.7, 2.347, 1234.5, -3.173];

  it("floor, ceil and round match torch", () => {
    close(parameter(V).floor(), ref("floor"));
    close(parameter(V).ceil(), ref("ceil"));
    close(parameter(V).round(), ref("round"));
    close(parameter(V).round(2), ref("round2"), 1e-4);
  });

  it("returns GradTensor and backward gives zeros", () => {
    for (const fn of [
      (t: GradTensor) => t.floor(),
      (t: GradTensor) => t.ceil(),
      (t: GradTensor) => t.round(),
      (t: GradTensor) => t.round(1),
    ]) {
      const x = parameter(V);
      const y = fn(x);
      expect(GradTensor.isGradTensor(y)).toBe(true);
      expect(y.requiresGrad).toBe(true);
      y.sum().backward();
      expect(x.grad?.dtype).toBe("float32");
      close(x.grad, new Array(V.length).fill(0));
    }
  });

  it("a rounded value still lets other paths contribute", () => {
    const x = parameter([1.4, 2.6]);
    x.floor().add(x.mul(x)).sum().backward();
    close(x.grad, [2.8, 5.2]);
  });
});

describe("GradTensor flip and sort are differentiable", () => {
  it("flip matches torch values and gradients", () => {
    const [o1, g1] = pair("flip1");
    const a = weighted(X, (x) => x.flip([1]));
    close(a.out, o1);
    close(a.grad, g1);
    const [o2, g2] = pair("flipAll");
    const b = weighted(X, (x) => x.flip());
    close(b.out, o2);
    close(b.grad, g2);
  });

  const S2 = [
    [3, 1, 2, 1],
    [0, 5, 5, -1],
  ];
  const W2 = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
  ];

  it("sort matches torch values and gradients, ties included", () => {
    for (const [name, axis, descending] of [
      ["sort1", 1, false],
      ["sort1d", 1, true],
      ["sort0", 0, false],
    ] as const) {
      const [o, g] = pair(name);
      const r = weighted(S2, (x) => x.sort(axis, descending), W2);
      close(r.out, o);
      close(r.grad, g);
    }
  });

  it("sort defaults to the last axis, ascending", () => {
    const r = parameter([3, 1, 2]).sort();
    close(r, [1, 2, 3]);
    const x = parameter([3, 1, 2]);
    x.sort()
      .mul(GradTensor.fromTensor(tensor([10, 20, 30])))
      .sum()
      .backward();
    close(x.grad, [30, 10, 20]);
  });

  it("sort of a 0-d tensor passes the gradient through", () => {
    const x = parameter(tensor(4));
    x.sort()
      .mul(GradTensor.fromTensor(tensor(3)))
      .backward();
    close(x.grad, 3);
  });

  it("sort records nothing inside noGrad", () => {
    const x = parameter([3, 1, 2]);
    const y = noGrad(() => x.sort());
    expect(y.requiresGrad).toBe(false);
  });
});

describe("GradTensor.dot", () => {
  it("equals matmul, with values and gradients from torch", () => {
    const m = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const x = parameter(X);
    const out = x.dot(m);
    const [o, g] = pair("dot");
    close(out, o);
    out
      .mul(
        GradTensor.fromTensor(
          tensor([
            [1, 2],
            [3, 4],
          ])
        )
      )
      .sum()
      .backward();
    close(x.grad, g);
  });

  it("1-D dot is a scalar", () => {
    expect(
      parameter([1, 2, 3])
        .dot(parameter([4, 5, 6]))
        .item()
    ).toBe(32);
  });
});

describe("GradTensor device methods", () => {
  const hostBackend: Backend = {
    info: () => ({ device: "wasm", name: "host", available: true, capabilities: [] }),
    supports: () => false,
    init: async () => {},
    dispose: () => {},
  };

  it("exposes isDeviceTensor and dispose on host tensors", () => {
    const x = parameter([1, 2]);
    expect(x.isDeviceTensor).toBe(false);
    expect(() => x.dispose()).not.toThrow();
    expect(x.toArray()).toEqual([1, 2]);
  });

  it("cpu() returns the same object when already on the CPU", async () => {
    const x = parameter([1, 2]);
    expect(await x.cpu()).toBe(x);
    expect(await x.to("cpu")).toBe(x);
  });

  it("to() throws without a backend", async () => {
    if (isBackendAvailable("webgpu")) return;
    await expect(parameter([1]).to("webgpu")).rejects.toThrow(DeviceError);
  });

  it("to() keeps requiresGrad and moves the gradient", async () => {
    const had = isBackendAvailable("wasm");
    registerBackend("wasm", hostBackend);
    try {
      const w = parameter([1, 2, 3]);
      w.mul(w).sum().backward();
      const moved = await w.to("wasm");
      expect(moved).not.toBe(w);
      expect(moved.device).toBe("wasm");
      expect(moved.requiresGrad).toBe(true);
      expect(moved.grad?.device).toBe("wasm");
      close(moved.grad, [2, 4, 6]);
      close(moved, [1, 2, 3]);

      const back = await moved.cpu();
      expect(back.device).toBe("cpu");
      expect(back.requiresGrad).toBe(true);
      expect(back.grad?.device).toBe("cpu");
      close(back.grad, [2, 4, 6]);

      const frozen = await GradTensor.fromTensor(tensor([1, 2])).to("wasm");
      expect(frozen.requiresGrad).toBe(false);
      expect(frozen.grad).toBeNull();
    } finally {
      if (!had) unregisterBackend("wasm");
    }
  });
});

// ---------------------------------------------------------------------------
// Shared methods: same argument forms on both classes
// ---------------------------------------------------------------------------

describe("shared methods accept the same arguments", () => {
  it("add, sub, mul, div take a number or a Tensor on GradTensor", () => {
    const C = [
      [0.5, 2, 4],
      [1, 0.25, 8],
    ];
    const [o, g] = pair("comp");
    const r = weighted(X, (x) => x.mul(2.5).add(tensor(C)).div(4).sub(1));
    close(r.out, o);
    close(r.grad, g);
    const t = tensor(X).mul(2.5).add(tensor(C)).div(4).sub(1);
    close(t, o);
  });

  it("a number operand does not change the dtype", () => {
    expect(parameter([1, 2]).add(1).dtype).toBe("float32");
    expect(parameter([1, 2]).mul(0.5).dtype).toBe("float32");
    const ints = GradTensor.fromTensor(tensor([1, 2], { dtype: "int32" }));
    expect(ints.add(1).dtype).toBe("int32");
  });

  it("rejects an operand that is neither number nor tensor", () => {
    expect(() => parameter([1]).add("a" as unknown as number)).toThrow(InvalidParameterError);
    expect(() => parameter([1]).matmul(3 as unknown as Tensor)).toThrow(InvalidParameterError);
  });

  it("pow with a GradTensor exponent matches torch values and both gradients", () => {
    const B = [
      [0.5, 1.5, 2],
      [3, 0, 2.5],
    ];
    const E = [
      [2, 0.5, -1],
      [1.5, 2, 0],
    ];
    const [o, gb, ge] = REF.powT as [unknown, unknown, unknown];
    const b = parameter(B);
    const e = parameter(E);
    const out = b.pow(e);
    close(out, o);
    out
      .mul(GradTensor.fromTensor(tensor(W)))
      .sum()
      .backward();
    close(b.grad, gb);
    close(e.grad, ge);
  });

  it("pow with a broadcast exponent reduces its gradient", () => {
    const B = [
      [0.5, 1.5, 2],
      [3, 0.7, 2.5],
    ];
    const [o, gb, ge] = REF.powBc as [unknown, unknown, unknown];
    const b = parameter(B);
    const e = parameter([2, 0.5, 1.5]);
    const out = b.pow(e);
    close(out, o);
    out
      .mul(GradTensor.fromTensor(tensor(W)))
      .sum()
      .backward();
    close(b.grad, gb);
    expect(e.grad?.shape).toEqual([3]);
    close(e.grad, ge);
  });

  it("pow with a constant Tensor exponent only gives the base a gradient", () => {
    const b = parameter([2, 3]);
    b.pow(tensor([2, 2]))
      .sum()
      .backward();
    close(b.grad, [4, 6]);
    close(tensor([2, 3]).pow(tensor([2, 2])), [4, 9]);
  });

  it("pow with an integer exponent keeps the float dtype of the base, as Tensor does", () => {
    const base = parameter([1, 2, 3], { dtype: "float32" });
    const r = base.pow(tensor([2, 2, 2], { dtype: "int32" }));
    expect(r.dtype).toBe("float32");
    expect(r.dtype).toBe(
      tensor([1, 2, 3], { dtype: "float32" }).pow(tensor([2, 2, 2], { dtype: "int32" })).dtype
    );
    r.sum().backward();
    close(base.grad, [2, 4, 6]);
  });

  it("sort and flip give torch gradients on a non-contiguous view", () => {
    const data = [
      [3, 1, 2, 1],
      [0, 5, 5, -1],
      [2, 2, 7, 4],
    ];
    const w = GradTensor.fromTensor(
      tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
      ])
    );
    const a = parameter(data);
    a.transpose().sort(-1).mul(w).sum().backward();
    close(a.grad, [
      [3, 4, 7, 11],
      [1, 6, 8, 10],
      [2, 5, 9, 12],
    ]);
    const b = parameter(data);
    b.transpose().flip([0]).mul(w).sum().backward();
    close(b.grad, [
      [10, 7, 4, 1],
      [11, 8, 5, 2],
      [12, 9, 6, 3],
    ]);
  });

  it("pow of integer operands without tracking stays integer", () => {
    const a = GradTensor.fromTensor(tensor([2, 3], { dtype: "int32" }));
    const r = a.pow(tensor([2, 2], { dtype: "int32" }));
    expect(r.dtype).toBe("int32");
    expect(r.toArray()).toEqual([4, 9]);
  });

  it("clip takes optional bounds", () => {
    for (const [name, lo, hi] of [
      ["clipMin", 0.3, undefined],
      ["clipMax", undefined, 0.3],
    ] as const) {
      const [o, g] = pair(name);
      const r = weighted(X, (x) => x.clip(lo, hi));
      close(r.out, o);
      close(r.grad, g);
      close(tensor(X).clip(lo, hi), o);
    }
  });

  it("maximum and minimum take a number on GradTensor, ties share the gradient", () => {
    const [o1, g1] = pair("maxNumGrad");
    const a = weighted(X, (x) => x.maximum(0.3));
    close(a.out, o1);
    close(a.grad, g1);
    const [o2, g2] = pair("minNumGrad");
    const b = weighted(X, (x) => x.minimum(0.3));
    close(b.out, o2);
    close(b.grad, g2);
    close(parameter(X).maximum(tensor(Y)), ref("maximum"));
  });

  it("gather takes a Tensor or a GradTensor of indices", () => {
    const x = parameter(X);
    close(x.gather(tensor([2, 0, 0]), 1), ref("gather"));
    close(x.gather(GradTensor.fromTensor(tensor([2, 0, 0])), 1), ref("gather"));
    x.gather(tensor([2, 0, 0]), 1)
      .sum()
      .backward();
    close(x.grad, [
      [2, 0, 1],
      [2, 0, 1],
    ]);
  });

  it("matmul takes a Tensor", () => {
    const a = parameter([
      [1, 2],
      [3, 4],
    ]);
    const r = a.matmul(tensor([[1], [1]]));
    close(r, [[3], [7]]);
    r.sum().backward();
    close(a.grad, [
      [1, 1],
      [1, 1],
    ]);
  });

  it("gelu takes the options object on GradTensor", () => {
    const x = parameter(X);
    close(x.gelu(), ref("gelu_tanh"));
    close(x.gelu({ approximate: "none" }), ref("gelu_none"));
    close(x.gelu("none"), ref("gelu_none"));
    close(x.gelu({}), ref("gelu_tanh"));
    expect(() => x.gelu({ approximate: "bad" as "tanh" })).toThrow(InvalidParameterError);
  });

  it("astype takes any dtype, and rejects string like Tensor.astype does on numbers", () => {
    const x = parameter([1.5, 2.5]);
    expect(x.astype("float64").dtype).toBe("float64");
    expect(() => x.astype("string")).toThrow(DTypeError);
    expect(() => x.astype("complex64")).toThrow(DTypeError);
  });

  it("reductions, softmax, transpose, reshape and slice agree between the classes", () => {
    const t = tensor(X);
    const g = parameter(X);
    close(g.sum(0, true), t.sum(0, true));
    close(g.sum([0, 1]), t.sum([0, 1]));
    close(g.mean(1), t.mean(1));
    close(g.max(0, true), t.max(0, true));
    close(g.min([0, 1]), t.min([0, 1]));
    close(g.prod(1), t.prod(1));
    close(g.std(1, true, 1), t.std(1, true, 1));
    close(g.var(0, false, 1), t.var(0, false, 1));
    close(g.cumsum(1), t.cumsum(1));
    close(g.softmax(0), t.softmax(0));
    close(g.logSoftmax(), t.logSoftmax());
    close(g.transpose([1, 0]), t.transpose([1, 0]));
    close(g.reshape([3, -1]), t.reshape([3, -1]));
    close(g.slice({ start: 1 }, { start: 0, end: 2 }), t.slice({ start: 1 }, { start: 0, end: 2 }));
    close(g.astype("float64"), t.astype("float64"));
    close(g.pow(2), t.pow(2));
    close(g.clip(-1, 1), t.clip(-1, 1));
    close(g.squeeze(), t.squeeze());
    close(g.unsqueeze(1), t.unsqueeze(1));
    close(g.flip(0), t.flip(0));
    close(g.sort(0, true), t.sort(0, true));
    expect(g.shape).toEqual(t.shape);
    expect(g.flatten().shape).toEqual(t.flatten().shape);
    expect(g.T.shape).toEqual(t.T.shape);
  });

  it("every activation method agrees between the classes", () => {
    const t = tensor(X);
    const g = parameter(X);
    for (const name of [
      "elu",
      "gelu",
      "selu",
      "softplus",
      "mish",
      "swish",
      "logSigmoid",
      "hardtanh",
      "leakyRelu",
      "tanhshrink",
      "expm1",
      "tan",
      "sigmoid",
      "tanh",
      "relu",
      "abs",
      "exp",
      "sin",
      "cos",
      "neg",
      "square",
    ] as const) {
      close(g[name](), t[name](), 1e-5);
    }
  });
});

describe("GradTensor in a training loop reads the loss with item()", () => {
  it("loss.item() falls as the weights fit", () => {
    const x = GradTensor.fromTensor(
      tensor([
        [1, 2],
        [3, 1],
        [2, 2],
      ])
    );
    const y = GradTensor.fromTensor(tensor([5, 5, 6]));
    let weights = [0, 0];
    const losses: number[] = [];
    for (let step = 0; step < 50; step++) {
      const w = parameter(weights);
      const diff = x.matmul(w).sub(y);
      const loss = diff.mul(diff).mean();
      loss.backward();
      losses.push(loss.item() as number);
      const g = w.grad?.toArray() as number[];
      weights = weights.map((v, i) => v - 0.05 * (g[i] as number));
    }
    expect(losses[losses.length - 1] as number).toBeLessThan((losses[0] as number) / 10);
  });
});
