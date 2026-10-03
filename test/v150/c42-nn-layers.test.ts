import { describe, expect, it } from "vitest";
import { DTypeError, InvalidParameterError, ShapeError } from "../../src/core";
import { GradTensor, parameter, type Tensor, tensor, transpose } from "../../src/ndarray";
import { Tensor as TensorClass } from "../../src/ndarray/tensor/Tensor";
import {
  BatchNorm1d,
  BatchNorm2d,
  ConstantPad2d,
  GRU,
  GroupNorm,
  InstanceNorm1d,
  InstanceNorm2d,
  LayerNorm,
  Linear,
  LocalResponseNorm,
  LSTM,
  packPaddedSequence,
  packSequence,
  padPackedSequence,
  ReflectionPad2d,
  ReplicationPad2d,
  RMSNorm,
  RNN,
  SpectralNorm,
  unpackSequence,
  ZeroPad2d,
} from "../../src/nn";

const f64 = { dtype: "float64" as const };

function flat(t: Tensor | GradTensor): number[] {
  const arr = (GradTensor.isGradTensor(t) ? t.tensor : t).toArray() as unknown;
  return (arr as number[]).flat(Number.POSITIVE_INFINITY) as number[];
}

function expectClose(actual: readonly number[], expected: readonly number[], tol = 1e-9): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(Math.abs((actual[i] as number) - (expected[i] as number))).toBeLessThan(tol);
  }
}

/** Deterministic test data: sin(a * k + b). */
function gen(n: number, a = 1.3, b = 0.4): number[] {
  return Array.from({ length: n }, (_, k) => Math.sin(a * k + b));
}

function cosWeights(n: number): number[] {
  return Array.from({ length: n }, (_, k) => Math.cos(0.37 * k));
}

function weightedSum(t: GradTensor, w: number[]): GradTensor {
  return t.mul(GradTensor.fromTensor(tensor(w, f64).reshape([...t.shape]))).sum();
}

// ─── Linear ────────────────────────────────────────────────────────────────

describe("v1.5.0 Linear", () => {
  function makeLayer(): Linear {
    const layer = new Linear(3, 2, f64);
    layer.loadStateDict({
      parameters: {
        weight: { data: [0.5, -1, 2, 1.5, 0.25, -0.75], shape: [2, 3], dtype: "float64" },
        bias: { data: [0.1, -0.2], shape: [2], dtype: "float64" },
      },
      buffers: {},
    });
    return layer;
  }
  // numpy: X @ W.T + b
  const expected = [4.6, -0.45, 9.1, 2.55];

  it("reads an offset view of the input instead of crashing", () => {
    const view = TensorClass.fromTypedArray({
      data: new Float32Array([9, 9, 1, 2, 3, 4, 5, 6]),
      shape: [2, 3],
      dtype: "float32",
      device: "cpu",
      offset: 2,
    });
    expectClose(flat(makeLayer().forward(view)), expected, 1e-6);
  });

  it("reads a strided (transposed) input with a dtype conversion", () => {
    const view = transpose(
      tensor([
        [1, 4],
        [2, 5],
        [3, 6],
      ])
    );
    expectClose(flat(makeLayer().forward(view)), expected, 1e-6);
    const ints = transpose(
      tensor(
        [
          [1, 4],
          [2, 5],
          [3, 6],
        ],
        { dtype: "int32" }
      )
    );
    expectClose(flat(makeLayer().forward(ints)), expected, 1e-6);
  });

  it("keeps the autograd link when a GradTensor input has another dtype", () => {
    const layer = new Linear(2, 2);
    const upstream = parameter(tensor([[1, 2]], f64));
    const out = layer.forward(upstream.mul(GradTensor.scalar(2, f64)));
    expect(out.dtype).toBe("float32");
    out.sum().backward();
    expect(upstream.grad).not.toBeNull();
    expect(upstream.grad?.dtype).toBe("float64");
    // d(sum(x W^T))/dx = 2 * column sums of W
    const w = flat(layer.getWeight());
    expectClose(flat(upstream.grad as Tensor), [2 * (w[0]! + w[2]!), 2 * (w[1]! + w[3]!)], 1e-5);
  });

  it("returns the layer dtype for float32 input into a float64 layer", () => {
    const out = makeLayer().forward(tensor([[1, 2, 3]]));
    expect(out.dtype).toBe("float64");
  });

  it("accepts integer, int64 and boolean input", () => {
    const layer = makeLayer();
    expectClose(flat(layer.forward(tensor([[1, 2, 3]], { dtype: "int64" }))), [4.6, -0.45], 1e-12);
    expectClose(flat(layer.forward(tensor([[1, 0, 1]], { dtype: "bool" }))), [2.6, 0.55], 1e-12);
  });

  it("getWeight and getBias return the live parameter tensors", () => {
    const layer = makeLayer();
    const params = new Map(layer.namedParameters());
    expect(layer.getWeight()).toBe(params.get("weight")?.tensor);
    expect(layer.getBias()).toBe(params.get("bias")?.tensor);
    expect(new Linear(2, 2, { bias: false }).getBias()).toBeUndefined();
  });

  it("rejects string input and scalar input", () => {
    const layer = makeLayer();
    expect(() => layer.forward(tensor([["a", "b", "c"]]))).toThrow(DTypeError);
    expect(() => layer.forward(tensor(1))).toThrow(ShapeError);
  });
});

// ─── Normalization ─────────────────────────────────────────────────────────

describe("v1.5.0 BatchNorm", () => {
  // torch.nn.BatchNorm2d(2, dtype=float64) on sin(1.3 k + 0.4), shape (3, 2, 2, 2)
  const bn2dY = [
    0.54791925572, 1.391865730223, 0.199971173168, -1.281636097047, -0.904392279831, 0.81029429489,
    1.323819854916, -0.116131748254, -1.372400698566, -0.627889432281, 1.039726847298,
    1.187386383778, -0.417695279612, -1.426730067331, -0.359679938301, 1.220224168617,
    0.998787437003, -0.680477237226, -1.359595640718, -0.043657721351, 1.343032602712,
    0.760451290204, -0.950270918365, -1.282921979645,
  ];
  const bn2dEvalY = [
    0.911856270666, 0.996396286328, 0.61239162108, -0.059554330773, -0.704733295326,
    -1.017896587973, -0.852653604544, -0.286720702138, 0.414318385787, 0.919529434264,
    0.992347483434, 0.59852506696, -0.077623251365, -0.717128322061, -1.019683131773,
    -0.842991425944, -0.269121271854, 0.429987613774, 0.926942667574, 0.988018162607,
    0.584017894998, -0.094775436286, -0.729320793299, -1.021181579592,
  ];

  it("matches PyTorch in float64: batch stats, running stats and eval mode", () => {
    const bn = new BatchNorm2d(2, f64);
    const y = bn.forward(tensor(gen(24), f64).reshape([3, 2, 2, 2]));
    expectClose(flat(y), bn2dY, 1e-9);
    const buffers = bn.stateDict().buffers;
    expectClose(
      Array.from(buffers.running_mean?.data as number[]),
      [-0.000158090534, 0.00067794909]
    );
    expectClose(
      Array.from(buffers.running_var?.data as number[]),
      [0.955551878511, 0.954296307833]
    );

    bn.eval();
    const evalY = bn.forward(tensor(gen(24, 0.7, 1.1), f64).reshape([3, 2, 2, 2]));
    expectClose(flat(evalY), bn2dEvalY, 1e-9);
  });

  it("accepts integer input", () => {
    // torch.nn.BatchNorm1d(3, affine=False)(float64([[1, 2, 3], [4, 5, 7]]))
    const bn = new BatchNorm1d(3, { affine: false });
    const y = bn.forward(
      tensor(
        [
          [1, 2, 3],
          [4, 5, 7],
        ],
        { dtype: "int32" }
      )
    );
    expect(y.dtype).toBe("float32");
    expectClose(
      flat(y),
      [
        -0.999997777785, -0.999997777785, -0.999998750002, 0.999997777785, 0.999997777785,
        0.999998750002,
      ],
      1e-6
    );
  });

  it("computes in the parameter dtype: float64 input is cast to float32 by default", () => {
    const bn = new BatchNorm1d(3);
    const y = bn.forward(
      tensor(
        [
          [1, 2, 3],
          [4, 5, 7],
        ],
        f64
      )
    );
    expect(y.dtype).toBe("float32");
    expectClose(
      flat(y),
      [
        -0.999997777778, -0.999997777778, -0.999998750002, 0.999997777778, 0.999997777778,
        0.999998750002,
      ],
      1e-6
    );
    // the running statistics keep the dtype of the layer
    expect(bn.stateDict().buffers.running_mean?.dtype).toBe("float32");
  });

  it("supports a float64 layer through the dtype option", () => {
    const bn = new BatchNorm1d(2, f64);
    const params = Array.from(bn.parameters());
    expect(params.map((p) => p.dtype)).toEqual(["float64", "float64"]);
    expect(bn.stateDict().buffers.running_var?.dtype).toBe("float64");
    expect(() => new BatchNorm1d(2, { dtype: "int32" as never })).toThrow(InvalidParameterError);
  });

  it("rejects a single value per channel when batch statistics are needed", () => {
    const bn = new BatchNorm1d(3);
    expect(() => bn.forward(tensor([[1, 2, 3]]))).toThrow(/more than one value per channel/);
    expect(() => bn.forward(tensor([[1, 2, 3]]))).toThrow(InvalidParameterError);
    // running statistics need no batch statistics, so a batch of one is fine in eval mode
    bn.eval();
    expect(bn.forward(tensor([[1, 2, 3]])).shape).toEqual([1, 3]);
    // a length-2 sequence gives two values per channel
    expect(new BatchNorm1d(3).forward(tensor(gen(6)).reshape([1, 3, 2])).shape).toEqual([1, 3, 2]);
  });

  it("running statistics written by loadStateDict are used in eval mode", () => {
    const bn = new BatchNorm1d(2, { affine: false, eps: 1e-5, ...f64 });
    bn.forward(
      tensor(
        [
          [1, 2],
          [3, 5],
        ],
        f64
      )
    );
    const state = bn.stateDict();
    state.buffers.running_mean = { data: [1, 2], shape: [2], dtype: "float64" };
    state.buffers.running_var = { data: [4, 9], shape: [2], dtype: "float64" };
    bn.loadStateDict(state);
    bn.eval();
    const y = bn.forward(tensor([[3, 5]], f64));
    expectClose(flat(y), [2 / Math.sqrt(4 + 1e-5), 3 / Math.sqrt(9 + 1e-5)], 1e-12);
  });
});

describe("v1.5.0 LayerNorm / GroupNorm / RMSNorm", () => {
  const ints = tensor(
    [
      [1, 2, 3],
      [4, 5, 7],
    ],
    { dtype: "int32" }
  );

  it("accept integer input", () => {
    // torch LayerNorm/GroupNorm(1 group) without affine on float64([[1,2,3],[4,5,7]])
    const normalized = [
      -1.224735685908, 0, 1.224735685908, -1.06904153145, -0.267260382863, 1.336301914313,
    ];
    expectClose(
      flat(new LayerNorm(3, { elementwiseAffine: false }).forward(ints)),
      normalized,
      1e-5
    );
    expectClose(flat(new GroupNorm(1, 3, { affine: false }).forward(ints)), normalized, 1e-5);
    expectClose(
      flat(new RMSNorm(3, { elementwiseAffine: false }).forward(ints)),
      [
        0.462909553912, 0.925819107824, 1.388728661736, 0.730296621624, 0.91287077703,
        1.278019087842,
      ],
      1e-5
    );
  });

  it("compute in the parameter dtype: float64 input is cast to float32 by default", () => {
    const x = tensor(
      [
        [1, 2, 3],
        [4, 5, 7],
      ],
      f64
    );
    for (const layer of [new LayerNorm(3), new GroupNorm(1, 3), new RMSNorm(3)]) {
      expect(layer.forward(x).dtype).toBe("float32");
    }
    for (const layer of [new LayerNorm(3, f64), new GroupNorm(1, 3, f64), new RMSNorm(3, f64)]) {
      expect(layer.forward(x).dtype).toBe("float64");
    }
    expectClose(
      flat(new LayerNorm(3, f64).forward(x)),
      [-1.224735685908, 0, 1.224735685908, -1.06904153145, -0.267260382863, 1.336301914313],
      1e-9
    );
  });

  it("LayerNorm bias=false and RMSNorm elementwiseAffine=false drop parameters", () => {
    expect(Array.from(new LayerNorm(3, { bias: false }).namedParameters()).map(([n]) => n)).toEqual(
      ["weight"]
    );
    expect(Array.from(new LayerNorm(3).namedParameters()).map(([n]) => n)).toEqual([
      "weight",
      "bias",
    ]);
    expect(Array.from(new RMSNorm(3, { elementwiseAffine: false }).parameters())).toHaveLength(0);
    expect(Array.from(new RMSNorm(3).parameters())).toHaveLength(1);
  });

  it("validate eps and shapes", () => {
    expect(() => new GroupNorm(2, 4, { eps: 0 })).toThrow(/eps/);
    expect(() => new GroupNorm(2, 4, { eps: Number.NaN })).toThrow(/eps/);
    expect(() => new RMSNorm(4, { eps: -1 })).toThrow(/eps/);
    expect(() => new RMSNorm([0])).toThrow(/positive integers/);
    expect(() => new RMSNorm([2.5])).toThrow(/positive integers/);
    expect(() => new RMSNorm([])).toThrow(/at least one/);
  });
});

describe("v1.5.0 InstanceNorm", () => {
  // torch.nn.InstanceNorm2d(2, dtype=float64) on sin(1.3 k + 0.4) reshaped to (1, 2, 3, 3)
  const expected = [
    0.478787168241, 1.324957952513, 0.129922031891, -1.355590170242, -0.955299797577,
    0.744366816152, 1.253394099936, -0.173944109492, -1.446593991422, -0.784998518013,
    0.988630015614, 1.145676435926, -0.543932630754, -1.604915943407, -0.482930463377,
    1.17831245256, 0.945088040678, -0.840929389229,
  ];

  it("matches PyTorch for batched and unbatched input", () => {
    const norm = new InstanceNorm2d(2, f64);
    expectClose(flat(norm.forward(tensor(gen(18), f64).reshape([1, 2, 3, 3]))), expected, 1e-9);
    expectClose(flat(norm.forward(tensor(gen(18), f64).reshape([2, 3, 3]))), expected, 1e-9);
    expect(norm.forward(tensor(gen(18), f64).reshape([2, 3, 3])).shape).toEqual([2, 3, 3]);
  });

  it("validates the input rank", () => {
    expect(() => new InstanceNorm2d(2).forward(tensor(gen(8)).reshape([2, 4]))).toThrow(ShapeError);
    expect(() => new InstanceNorm1d(2).forward(tensor(gen(16)).reshape([1, 2, 2, 4]))).toThrow(
      /InstanceNorm1d expects/
    );
    expect(new InstanceNorm1d(2).forward(tensor(gen(8)).reshape([2, 4])).shape).toEqual([2, 4]);
  });

  it("rejects a single spatial element per channel", () => {
    expect(
      () => new InstanceNorm2d(2).forward(tensor(gen(4)).reshape([1, 2, 1, 2])).shape
    ).not.toThrow();
    expect(() => new InstanceNorm2d(2).forward(tensor(gen(2)).reshape([1, 2, 1, 1]))).toThrow(
      InvalidParameterError
    );
  });
});

describe("v1.5.0 LocalResponseNorm", () => {
  const x = () => tensor(gen(10), f64).reshape([1, 5, 2]);
  const w = Array.from({ length: 10 }, (_, k) => Math.cos(0.9 * k));
  const options = { alpha: 0.5, beta: 0.6, k: 1.5 };

  function run(size: number): { y: number[]; dx: number[] } {
    const input = parameter(x());
    const y = new LocalResponseNorm(size, options).forward(input);
    weightedSum(y, w).backward();
    return { y: flat(y), dx: flat(input.grad as Tensor) };
  }

  it("matches PyTorch for odd size, forward and gradient", () => {
    // torch.nn.LocalResponseNorm(3, alpha=0.5, beta=0.6, k=1.5)
    const { y, dx } = run(3);
    expectClose(
      y,
      [
        0.301884756659, 0.696069904911, 0.106642562411, -0.631434369809, -0.456368431805,
        0.421221427783, 0.645303369259, -0.056889354913, -0.68760764982, -0.347751096858,
      ],
      1e-9
    );
    expectClose(
      dx,
      [
        0.761021387815, 0.327859943709, -0.183565318703, -0.532485318371, -0.59254341468,
        -0.178824740815, 0.392954379565, 0.75639772244, 0.423913353784, -0.186607042395,
      ],
      1e-9
    );
  });

  it("uses PyTorch's asymmetric window for even size, forward and gradient", () => {
    // torch.nn.LocalResponseNorm(4, ...): window = size // 2 before, (size - 1) // 2 after
    const { y, dx } = run(4);
    expectClose(
      y,
      [
        0.302734988418, 0.714269409447, 0.107606918205, -0.650487447698, -0.462100000035,
        0.410600043654, 0.66488235944, -0.055192626299, -0.69387997591, -0.343318552669,
      ],
      1e-9
    );
    expectClose(
      dx,
      [
        0.752296904901, 0.367345838136, -0.187353821283, -0.572171300696, -0.634626476052,
        -0.175987968781, 0.413872631656, 0.73399622377, 0.430267292585, -0.184580942524,
      ],
      1e-9
    );
  });

  it("size 2 default options", () => {
    const y = new LocalResponseNorm(2).forward(x());
    expectClose(
      flat(y),
      [
        0.389416127799, 0.99162824194, 0.141119100163, -0.916103318625, -0.631256733186,
        0.578414300914, 0.940685281688, -0.075150161622, -0.980868285742, -0.449643960174,
      ],
      1e-9
    );
  });

  it("reads views with a storage offset correctly", () => {
    const view = TensorClass.fromTypedArray({
      data: new Float64Array([7, 7, ...gen(10)]),
      shape: [1, 5, 2],
      dtype: "float64",
      device: "cpu",
      offset: 2,
    });
    const viaView = flat(new LocalResponseNorm(3, options).forward(view));
    expectClose(viaView, run(3).y, 1e-12);
  });

  it("works on integer input and validates parameters", () => {
    const out = new LocalResponseNorm(3).forward(tensor([[[1], [2], [3]]], { dtype: "int32" }));
    expect(out.dtype).toBe("float32");
    expect(flat(out)[0]).toBeCloseTo(1 / (1 + (1e-4 / 3) * 5) ** 0.75, 6);
    expect(() => new LocalResponseNorm(3, { alpha: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => new LocalResponseNorm(3, { k: Number.POSITIVE_INFINITY })).toThrow(
      InvalidParameterError
    );
  });
});

// ─── Padding ───────────────────────────────────────────────────────────────

describe("v1.5.0 padding layers", () => {
  const x = () =>
    tensor([
      [
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
      ],
    ]);

  it("match torch.nn.functional.pad", () => {
    expectClose(
      flat(new ReflectionPad2d([2, 1, 1, 0]).forward(x())),
      [6, 5, 4, 5, 6, 5, 3, 2, 1, 2, 3, 2, 6, 5, 4, 5, 6, 5],
      1e-6
    );
    expectClose(
      flat(new ReplicationPad2d([2, 1, 1, 2]).forward(x())),
      [1, 1, 1, 2, 3, 3, 1, 1, 1, 2, 3, 3, 4, 4, 4, 5, 6, 6, 4, 4, 4, 5, 6, 6, 4, 4, 4, 5, 6, 6],
      1e-6
    );
    expectClose(
      flat(new ConstantPad2d([1, 0, 0, 1], 7).forward(x())),
      [7, 1, 2, 3, 7, 4, 5, 6, 7, 7, 7, 7],
      1e-6
    );
    expectClose(flat(new ZeroPad2d([-1, 0, 0, 0]).forward(x())), [2, 3, 5, 6], 1e-6);
  });

  it("keep the input dtype", () => {
    expect(new ZeroPad2d(1).forward(x()).dtype).toBe("float32");
    expect(new ZeroPad2d(1).forward(tensor([[[[1, 2]]]], f64)).dtype).toBe("float64");
    const ints = new ConstantPad2d(1, 3).forward(tensor([[[[1, 2]]]], { dtype: "int32" }));
    expect(ints.dtype).toBe("int32");
    expect(flat(ints)).toEqual([3, 3, 3, 3, 3, 1, 2, 3, 3, 3, 3, 3]);
    const big = new ConstantPad2d([1, 0, 0, 0], 3).forward(
      tensor([[[[1, 2]]]], { dtype: "int64" })
    );
    expect(big.dtype).toBe("int64");
    expect(big.data).toBeInstanceOf(BigInt64Array);
    expect(Array.from(big.data as BigInt64Array)).toEqual([3n, 1n, 2n]);
    const bools = new ConstantPad2d([1, 0, 0, 0], 1).forward(
      tensor([[[[0, 1]]]], { dtype: "bool" })
    );
    expect(bools.dtype).toBe("bool");
    expect(flat(bools)).toEqual([1, 0, 1]);
  });

  it("float64 inputs are padded exactly", () => {
    const value = 0.1234567890123456;
    const out = new ZeroPad2d(1).forward(tensor([[[[value]]]], f64));
    expect(flat(out)[4]).toBe(value);
  });

  it("returns a gradient with the input dtype and PyTorch's value", () => {
    // F.pad(p, (1, 1, 1, 1), mode="replicate").sum().backward()
    const p = parameter(x());
    new ReplicationPad2d(1).forward(p).sum().backward();
    expect(p.grad?.dtype).toBe("float32");
    expectClose(flat(p.grad as Tensor), [4, 2, 4, 4, 2, 4], 1e-6);
  });

  it("accepts unbatched 3D input", () => {
    const out = new ZeroPad2d(1).forward(
      tensor([
        [
          [1, 2],
          [3, 4],
        ],
      ])
    );
    expect(out.shape).toEqual([1, 4, 4]);
  });

  it("reads strided input", () => {
    const view = transpose(
      tensor([
        [
          [
            [1, 4],
            [2, 5],
            [3, 6],
          ],
        ],
      ]),
      [0, 1, 3, 2]
    );
    expectClose(
      flat(new ReflectionPad2d(1).forward(view)),
      [5, 4, 5, 6, 5, 2, 1, 2, 3, 2, 5, 4, 5, 6, 5, 2, 1, 2, 3, 2],
      1e-6
    );
  });

  it("validates input and padding", () => {
    expect(() =>
      new ReflectionPad2d(1).forward(
        tensor([
          [1, 2],
          [3, 4],
        ])
      )
    ).toThrow(/expects 3D \(C, H, W\) or 4D/);
    expect(() => new ReflectionPad2d(2).forward(x())).toThrow(/less than input height/);
    expect(() => new ZeroPad2d(1.5)).toThrow(/integers/);
    expect(() => new ZeroPad2d([1, 2, 3] as never)).toThrow(/\[left, right, top, bottom\]/);
    expect(() => new ReplicationPad2d(1).forward(tensor([[[[]]]]).reshape([1, 1, 0, 0]))).toThrow(
      ShapeError
    );
  });
});

// ─── Packed sequences ──────────────────────────────────────────────────────

describe("v1.5.0 packed sequences", () => {
  const seqs = () => [
    tensor(
      [
        [1, 2],
        [3, 4],
        [5, 6],
      ],
      f64
    ),
    tensor([[7, 8]], f64),
    tensor(
      [
        [9, 10],
        [11, 12],
      ],
      f64
    ),
  ];

  it("packs in torch.nn.utils.rnn.pack_sequence order and keeps the dtype", () => {
    const packed = packSequence(seqs());
    expect(flat(packed.data)).toEqual([1, 2, 9, 10, 7, 8, 3, 4, 11, 12, 5, 6]);
    expect(packed.data.shape).toEqual([6, 2]);
    expect(packed.data.dtype).toBe("float64");
    expect(packed.batchSizes).toEqual([3, 2, 1]);
    expect(packed.sortedIndices).toEqual([0, 2, 1]);
    expect(packed.unsortedIndices).toEqual([0, 2, 1]);
  });

  it("keeps float64 precision and integer dtypes", () => {
    const value = 0.1234567890123;
    expect(flat(packSequence([tensor([[value]], f64)]).data)[0]).toBe(value);
    const packed = packSequence([
      tensor([1, 2, 3], { dtype: "int32" }),
      tensor([4], { dtype: "int32" }),
    ]);
    expect(packed.data.dtype).toBe("int32");
    const [back] = unpackSequence(packed);
    expect(back[0]?.dtype).toBe("int32");
  });

  it("round-trips 1D sequences as 1D and 2D sequences as 2D", () => {
    const [oneD] = unpackSequence(packSequence([tensor([1, 2, 3]), tensor([4])]));
    expect(oneD.map((t) => t.shape)).toEqual([[3], [1]]);
    const [twoD] = unpackSequence(packSequence([tensor([[1], [2], [3]]), tensor([[4]])]));
    expect(twoD.map((t) => t.shape)).toEqual([
      [3, 1],
      [1, 1],
    ]);
  });

  it("unpacks to the original order", () => {
    const [unpacked, lengths] = unpackSequence(packSequence(seqs()));
    expect(lengths).toEqual([3, 1, 2]);
    expect(unpacked.map((t) => flat(t))).toEqual([
      [1, 2, 3, 4, 5, 6],
      [7, 8],
      [9, 10, 11, 12],
    ]);
  });

  it("reads strided sequences", () => {
    const packed = packSequence([
      transpose(
        tensor([
          [1, 3, 5],
          [2, 4, 6],
        ])
      ),
    ]);
    expect(flat(packed.data)).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it("rejects a 2D sequence whose feature size differs from an earlier 1D sequence", () => {
    expect(() => packSequence([tensor([[1, 2, 3]]), tensor([1, 2])])).toThrow(/same feature/);
    expect(() => packSequence([tensor([1, 2]), tensor([[1, 2, 3]])])).toThrow(/same feature/);
  });

  it("enforcesSorted verifies the order", () => {
    expect(() => packSequence([tensor([1]), tensor([1, 2])], true)).toThrow(/not sorted/);
    expect(() =>
      packPaddedSequence(
        tensor([
          [[1], [0]],
          [[1], [2]],
        ]).reshape([2, 2, 1]),
        [1, 2],
        true
      )
    ).toThrow(/not sorted/);
    expect(packSequence([tensor([1, 2]), tensor([1])], true).batchSizes).toEqual([2, 1]);
  });

  it("padPackedSequence supports paddingValue and validates totalLength", () => {
    const packed = packSequence(seqs());
    const [padded, lengths] = padPackedSequence(packed, 4, -1);
    expect(lengths).toEqual([3, 1, 2]);
    expect(padded.shape).toEqual([3, 4, 2]);
    expect(padded.dtype).toBe("float64");
    expect(flat(padded)).toEqual([
      1, 2, 3, 4, 5, 6, -1, -1, 7, 8, -1, -1, -1, -1, -1, -1, 9, 10, 11, 12, -1, -1, -1, -1,
    ]);
    expect(() => padPackedSequence(packed, 2)).toThrow(
      /at least the longest sequence length \(3\)/
    );
    expect(() => padPackedSequence(packed, 3.5)).toThrow(InvalidParameterError);
  });

  it("packPaddedSequence requires integer lengths and keeps the dtype", () => {
    const padded = tensor(
      [
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [0, 0],
        ],
      ],
      f64
    );
    expect(() => packPaddedSequence(padded, [2, 1.5])).toThrow(/out of range/);
    const packed = packPaddedSequence(padded, [2, 1]);
    expect(packed.data.dtype).toBe("float64");
    expect(flat(packed.data)).toEqual([1, 2, 5, 6, 3, 4]);
  });

  it("rejects an inconsistent packed sequence", () => {
    const packed = packSequence(seqs());
    expect(() => unpackSequence({ ...packed, batchSizes: [3, 2, 3] })).toThrow(
      InvalidParameterError
    );
    expect(() => unpackSequence({ ...packed, batchSizes: [3, 2] })).toThrow(ShapeError);
  });

  it("rejects string tensors", () => {
    expect(() => packSequence([tensor(["a", "b"])])).toThrow(DTypeError);
  });
});

// ─── Recurrent layers ──────────────────────────────────────────────────────

describe("v1.5.0 recurrent layers", () => {
  /** Fill every parameter, in registration order, with 0.6 * sin(0.83 * k + 0.2). */
  function fillParameters(module: RNN | GRU | LSTM): void {
    const parameters: Record<string, { data: number[]; shape: number[]; dtype: "float64" }> = {};
    let counter = 0;
    for (const [name, p] of module.namedParameters()) {
      const data = Array.from(
        { length: p.size },
        (_, i) => 0.6 * Math.sin(0.83 * (counter + i) + 0.2)
      );
      counter += p.size;
      parameters[name] = { data, shape: [...p.shape], dtype: "float64" };
    }
    module.loadStateDict({ parameters, buffers: {} });
  }

  const input = () => GradTensor.fromTensor(tensor(gen(16), f64).reshape([4, 2, 2]));
  const state = (scale: number, fn: (k: number) => number) =>
    tensor(
      Array.from({ length: 16 }, (_, k) => scale * fn(k)),
      f64
    ).reshape([4, 2, 2]);

  function loss(
    output: GradTensor,
    states: GradTensor[],
    weightsOut: number[],
    weightsState: number[]
  ): GradTensor {
    let total = weightedSum(output, weightsOut);
    for (const s of states) total = total.add(weightedSum(s, weightsState));
    return total;
  }

  function gradSum(module: RNN | GRU | LSTM, name: string): number {
    const p = new Map(module.namedParameters()).get(name) as GradTensor;
    const w = cosWeights(p.size);
    return flat(p.grad as Tensor).reduce((a, v, i) => a + v * (w[i] as number), 0);
  }

  const options = { numLayers: 2, bidirectional: true, batchFirst: false, ...f64 };

  it("RNN(relu) matches PyTorch, two layers, bidirectional, seq-first", () => {
    const rnn = new RNN(2, 2, { ...options, nonlinearity: "relu" });
    fillParameters(rnn);
    const [out, h] = rnn.forwardWithState(input());
    expect(out.shape).toEqual([4, 2, 4]);
    expect(h.shape).toEqual([4, 2, 2]);
    const w = cosWeights(32);
    const wh = cosWeights(16);
    expect(weightedSum(out, w).tensor.at()).toBeCloseTo(0.3186177755482944, 10);
    expect(weightedSum(h, wh).tensor.at()).toBeCloseTo(0.17685563443971203, 10);
    loss(out, [h], w, wh).backward();
    expect(gradSum(rnn, "weight_ih_l0")).toBeCloseTo(-2.0327856595437086, 9);
    expect(gradSum(rnn, "weight_hh_l1_reverse")).toBeCloseTo(1.1999777765828503, 9);
  });

  it("GRU matches PyTorch with an initial state", () => {
    const gru = new GRU(2, 2, options);
    fillParameters(gru);
    const h0 = state(0.3, Math.sin);
    const [out, h] = gru.forwardWithState(input(), h0);
    const w = cosWeights(32);
    const wh = cosWeights(16);
    expect(weightedSum(out, w).tensor.at()).toBeCloseTo(0.49148780834880135, 10);
    expect(weightedSum(h, wh).tensor.at()).toBeCloseTo(0.8781014859022233, 10);
    loss(out, [h], w, wh).backward();
    expect(gradSum(gru, "weight_ih_l0")).toBeCloseTo(-0.5527410324037846, 9);
    expect(gradSum(gru, "weight_hh_l1_reverse")).toBeCloseTo(-0.1747416709258243, 9);
  });

  it("LSTM matches PyTorch with initial hidden and cell states", () => {
    const lstm = new LSTM(2, 2, options);
    fillParameters(lstm);
    const h0 = state(0.3, Math.sin);
    const c0 = state(0.2, Math.cos);
    const [out, [h, c]] = lstm.forwardWithState(input(), h0, c0);
    const w = cosWeights(32);
    const wh = cosWeights(16);
    expect(weightedSum(out, w).tensor.at()).toBeCloseTo(0.30740004565421053, 10);
    expect(weightedSum(h, wh).tensor.at()).toBeCloseTo(-0.14502616106949756, 10);
    expect(weightedSum(c, wh).tensor.at()).toBeCloseTo(1.0430682663151036, 10);
    loss(out, [h, c], w, wh).backward();
    expect(gradSum(lstm, "weight_ih_l0")).toBeCloseTo(-0.4398501798365735, 9);
    expect(gradSum(lstm, "weight_hh_l1_reverse")).toBeCloseTo(-0.05072440571160415, 9);
  });

  it("unbatched GRU matches PyTorch", () => {
    const gru = new GRU(2, 2, f64);
    fillParameters(gru);
    const [out, h] = gru.forwardWithState(tensor(gen(8), f64).reshape([4, 2]));
    expect(out.shape).toEqual([4, 2]);
    expect(h.shape).toEqual([1, 2]);
    expectClose(
      flat(out),
      [
        0.0227056543, -0.3191731836, -0.1885180804, -0.4873510658, -0.2084450096, -0.6248112073,
        -0.13779249, -0.5306659203,
      ],
      1e-9
    );
    expectClose(flat(h), [-0.13779249, -0.5306659203], 1e-9);
  });

  it("validates nonlinearity and dtype, and exposes the configuration", () => {
    expect(() => new RNN(2, 2, { nonlinearity: "sigmoid" as never })).toThrow(
      /nonlinearity must be 'tanh' or 'relu'/
    );
    expect(() => new LSTM(2, 2, { dtype: "int32" as never })).toThrow(InvalidParameterError);
    const lstm = new LSTM(3, 5, { numLayers: 2, bidirectional: true });
    expect([lstm.inputSize, lstm.hiddenSize, lstm.numLayers]).toEqual([3, 5, 2]);
    expect([lstm.bidirectional, lstm.batchFirst]).toEqual([true, true]);
    const rnn = new RNN(3, 5, { dtype: "float64" });
    expect(Array.from(rnn.parameters()).every((p) => p.dtype === "float64")).toBe(true);
    expect(rnn.nonlinearity).toBe("tanh");
  });
});

// ─── SpectralNorm ──────────────────────────────────────────────────────────

describe("v1.5.0 SpectralNorm", () => {
  function makeWrapped(iterations = 100): { linear: Linear; sn: SpectralNorm } {
    const linear = new Linear(3, 2, { bias: false, ...f64 });
    linear.loadStateDict({
      parameters: { weight: { data: [1, 2, 0, 0, 1, 3], shape: [2, 3], dtype: "float64" } },
      buffers: {},
    });
    return { linear, sn: new SpectralNorm(linear, "weight", iterations) };
  }

  it("registers the wrapped module so parameters and state are visible", () => {
    const { sn } = makeWrapped();
    expect(Array.from(sn.parameters())).toHaveLength(1);
    expect(Array.from(sn.namedParameters()).map(([n]) => n)).toEqual(["module.weight"]);
    expect(Array.from(sn.namedBuffers()).map(([n]) => n)).toEqual(["weight_u", "weight_v"]);
  });

  it("estimates the largest singular value right after construction", () => {
    // numpy.linalg.svd(W)[1][0]
    expect(new SpectralNorm(makeWrapped().linear).spectralNormValue).toBeCloseTo(
      3.2713242148580175,
      6
    );
  });

  it("propagates PyTorch's gradients to the input and to the raw weight", () => {
    // W_sn = W / sigma(W), sigma = u^T W v with u, v held constant (autograd in PyTorch)
    const { linear, sn } = makeWrapped();
    const x = parameter(tensor([[1, -1, 2]], f64));
    const y = sn.forward(x);
    expectClose(flat(y), [-0.30568660711099904, 1.528433035554995], 1e-6);
    weightedSum(y, [0.7, -1.3]).backward();
    expectClose(
      flat(x.grad as Tensor),
      [0.21398062497769932, 0.030568660711099867, -1.1921777677328962],
      1e-6
    );
    const weight = new Map(linear.namedParameters()).get("weight") as GradTensor;
    expectClose(
      flat(weight.grad as Tensor),
      [
        0.23651450273102656, -0.10467371757796597, 0.6206787056346348, -0.33315343735122,
        0.7090026505164806, -0.24538990603052502,
      ],
      1e-6
    );
  });

  it("leaves the wrapped weight untouched, also when the wrapped forward throws", () => {
    const { linear, sn } = makeWrapped();
    const original = new Map(linear.namedParameters()).get("weight") as GradTensor;
    const before = flat(original.tensor);
    expect(() => sn.forward(tensor([[1, 2]], f64))).toThrow(ShapeError);
    const after = new Map(linear.namedParameters()).get("weight") as GradTensor;
    expect(after).toBe(original);
    expect(flat(after.tensor)).toEqual(before);
    // the wrapped layer still uses the raw weight outside of the wrapper
    expectClose(flat(linear.forward(tensor([[1, -1, 2]], f64))), [-1, 5], 1e-12);
  });

  it("only refines the power iteration vectors in training mode", () => {
    const { sn } = makeWrapped(1);
    const seed = sn.stateDict();
    seed.buffers.weight_u = { data: [1, 0], shape: [2], dtype: "float64" };
    sn.loadStateDict(seed);
    sn.eval();
    sn.forward(tensor([[1, 2, 3]], f64));
    expect(Array.from(sn.stateDict().buffers.weight_u?.data as number[])).toEqual([1, 0]);
    sn.train();
    sn.forward(tensor([[1, 2, 3]], f64));
    expect(Array.from(sn.stateDict().buffers.weight_u?.data as number[])).not.toEqual([1, 0]);
  });

  it("rejects weights that are not float or have no elements to normalize", () => {
    expect(() => new SpectralNorm(new Linear(2, 2), "bias")).toThrow(ShapeError);
    expect(() => new SpectralNorm(new Linear(2, 2), "weight", 1, Number.NaN)).toThrow(
      InvalidParameterError
    );
  });
});

// ─── Review follow-ups ─────────────────────────────────────────────────────

describe("v1.5.0 review follow-ups", () => {
  it("LocalResponseNorm converts non-float input to the dtype option", () => {
    const ints = tensor([[[1], [2], [3]]], { dtype: "int32" });
    expect(new LocalResponseNorm(3, { dtype: "float64" }).forward(ints).dtype).toBe("float64");
    expect(() => new LocalResponseNorm(3, { dtype: "int32" as never })).toThrow(
      InvalidParameterError
    );
  });

  it("ConstantPad2d rejects a non-finite value for int64 input", () => {
    const layer = new ConstantPad2d(1, Number.NaN);
    expect(() => layer.forward(tensor([[[[1, 2]]]], { dtype: "int64" }))).toThrow(
      InvalidParameterError
    );
    expect(layer.forward(tensor([[[[1, 2]]]], f64)).shape).toEqual([1, 1, 3, 4]);
  });

  it("packed sequences reject indices that are not a permutation", () => {
    const packed = packSequence([tensor([[1], [2]], f64), tensor([[3]], f64)]);
    expect(() => unpackSequence({ ...packed, sortedIndices: [0, 0] })).toThrow(
      InvalidParameterError
    );
    expect(() => padPackedSequence({ ...packed, sortedIndices: [0, 5] })).toThrow(
      InvalidParameterError
    );
    expect(() => unpackSequence({ ...packed, unsortedIndices: [1, 0] })).toThrow(
      InvalidParameterError
    );
  });

  it("recurrent layers reject a 2D initial state for batched input", () => {
    const gru = new GRU(2, 3, f64);
    const x = tensor(gen(12), f64).reshape([2, 3, 2]);
    expect(() => gru.forward(x, tensor(gen(3), f64).reshape([1, 3]))).toThrow(ShapeError);
    expect(gru.forward(x, tensor(gen(6), f64).reshape([1, 2, 3])).shape).toEqual([2, 3, 3]);
    // unbatched input takes a 2D state
    expect(
      gru.forward(tensor(gen(6), f64).reshape([3, 2]), tensor(gen(3), f64).reshape([1, 3])).shape
    ).toEqual([3, 3]);
  });
});
