/**
 * Regression tests for src/nn/layers/activations.ts and src/nn/layers/attention.ts (v1.5.0).
 *
 * Reference values come from PyTorch 2.12 (float32) with the same deterministic weights
 * and inputs, produced by the generators `gen` and `inp` below.
 */
import { describe, expect, it } from "vitest";
import { DTypeError, InvalidParameterError, ShapeError } from "../../src/core";
import { GradTensor, parameter, type Tensor, tensor, transpose } from "../../src/ndarray";
import type { Module } from "../../src/nn";
import {
  causalMask,
  FullTransformer,
  GELU,
  GLU,
  Hardsigmoid,
  Hardswish,
  Hardtanh,
  LeakyReLU,
  Mish,
  MultiheadAttention,
  PositionalEncoding,
  PReLU,
  SELU,
  Softmax2d,
  Softmin,
  Softplus,
  Softsign,
  TransformerDecoder,
  TransformerDecoderLayer,
  TransformerEncoder,
  TransformerEncoderLayer,
} from "../../src/nn";

const REF: Record<string, number[]> = {
  mha_self: [
    0.018344931, 0.031481054, 0.13377681, 0.099248096, 0.015456237, 0.051453765, 0.14028743,
    0.080456033, 0.015255101, 0.037948854, 0.13803953, 0.093553312, 0.010088824, 0.13102764,
    0.16008502, 0.0044723153, 0.0041091293, 0.093913972, 0.15933439, 0.041449871, 0.0072468147,
    0.11354959, 0.15975749, 0.021890979,
  ],
  mha_cross: [
    0.00091312826, 0.048699841, 0.15433112, 0.085756697, 0.0067625493, 0.07496924, 0.15324548,
    0.059290417, -0.0019818097, 0.051126584, 0.15766612, 0.083934732, 0.0099485815, 0.060181748,
    0.14737783, 0.07301385, 0.0035693794, 0.064024121, 0.15445384, 0.070454665, 0.0073519349,
    0.047127381, 0.14760716, 0.08610981,
  ],
  mha_kv_shared: [
    0.0022974461, 0.043685798, 0.15203755, 0.090354808, 0.0033743382, 0.077163525, 0.15703161,
    0.057782724, 9.611249e-5, 0.038420655, 0.15328409, 0.095846005, 0.0080710649, 0.068963923,
    0.15084794, 0.064860962, 0.0031521842, 0.057290118, 0.15364987, 0.07704287, 0.0078571886,
    0.05349211, 0.14825611, 0.079862759,
  ],
  mha_causal: [
    0.051497426, 0.042983577, 0.10271022, 0.082111858, 0.0097802952, 0.11572571, 0.15761864,
    0.019326994, 0.015255101, 0.037948854, 0.13803953, 0.093553312, 0.0040283352, 0.25848404,
    0.18925884, -0.11769358, -0.010917962, 0.066505142, 0.1693911, 0.070682414, 0.0072468221,
    0.11354959, 0.15975749, 0.021890976,
  ],
  mha_boolmask: [
    -0.016436979, -0.030078635, 0.1573953, 0.16509083, 0.016157694, 0.12117415, 0.15222928,
    0.01290123, 0.0065748394, 0.087453999, 0.15569721, 0.047250263, 0.0051401407, -0.063718162,
    0.12971787, 0.19371125, 0.0076574683, 0.12514873, 0.16145027, 0.01059881, 0.0087040216,
    0.092647634, 0.15450986, 0.041841313,
  ],
  mha_2d: [
    0.018344931, 0.03148105, 0.13377681, 0.099248096, 0.015456222, 0.051453758, 0.14028743,
    0.080456041, 0.015255094, 0.037948854, 0.13803953, 0.093553312,
  ],
  enc: [
    1.2175741, 0.83241397, -0.73868513, -1.2822406, -1.6177661, -0.55303919, 0.65946633, 0.93099809,
    1.7202007, 0.05530991, -1.2382265, -0.55705202, -1.5154555, 0.66553342, 1.1187119, -0.6318447,
    -0.17367229, -1.3019606, -0.38578486, 1.3701553, 0.54658914, 1.1673726, -0.19750014, -1.5265388,
  ],
  enc_masked: [
    1.2277566, 0.83041036, -0.76756239, -1.262025, -1.6720049, -0.47691342, 0.70026696, 0.87158388,
    1.7202007, 0.05530991, -1.2382265, -0.55705202, -1.3640423, 0.84058005, 1.0318187, -0.82605034,
    -0.2043305, -1.3114978, -0.34777403, 1.3681443, 0.54658914, 1.1673726, -0.19750014, -1.5265388,
  ],
  dec_causal: [
    0.84845209, 0.60020137, -0.59510571, -1.0461254, -1.3197324, -0.22955476, 0.66801757, 0.8415311,
    1.1894438, 0.027049534, -0.92884165, -0.44041157, -1.0582682, 0.68593705, 0.85608464,
    -0.62584525, -0.20851207, -0.97654021, -0.23251325, 1.4073926, 0.34812424, 0.88641095,
    -0.13998765, -1.299088,
  ],
  dec_nomask: [
    0.84172297, 0.60222471, -0.57320637, -1.0641847, -1.2880526, -0.28885615, 0.63919443, 0.9020437,
    1.1894438, 0.027049534, -0.92884165, -0.44041157, -1.155274, 0.56087172, 0.93044269,
    -0.46518782, -0.18202144, -0.97027111, -0.26678818, 1.4083682, 0.34812424, 0.88641095,
    -0.13998765, -1.299088,
  ],
  pe: [
    0, 1, 0, 1, 0, 1, 0.84147096, 0.54030228, 0.046399225, 0.998923, 0.0021544329, 0.99999768,
    0.90929741, -0.41614684, 0.0926985, 0.99569422, 0.0043088561, 0.9999907, 0.14112, -0.9899925,
    0.1387981, 0.99032068, 0.006463259, 0.99997914,
  ],
  hardsigmoid_t: [0, 0.58333331, 0.33333334, 1, 0.5, 1],
  hardswish_t: [-0, 0.29166666, -0.33333334, 3, 0, 7],
  softsign_t: [-0.77777779, 0.33333334, -0.5, 0.75, 0, 0.875],
  selu_t: [-1.7050093, 0.52535051, -1.1113307, 3.1521029, 0, 7.354907],
  softplus_b2: [0.00045573323, 0.063464006, 0.34657359, 0.65663084, 3.0012378, 7.0000004],
  softplus_grad: [1, 3.7835059e-44, 0.5, 1, 0.006692851, 0.95257413],
  mish_grad: [1, 3.7835059e-44, 0.60000002, 1, -0.026747495, 1.021107],
  mish_val: [100, -3.7835059e-42, 0, 20, -0.033576235, 2.9865351],
  sm2d_3d: [
    0.52226138, 0.36052275, 0.31161606, 0.40136451, 0.57489341, 0.68358469, 0.47773859, 0.63947725,
    0.68838394, 0.59863555, 0.42510661, 0.31641534,
  ],
  sm2d_4d: [
    0.59186119, 0.38574773, 0.14623396, 0.11205352, 0.14317216, 0.45774543, 0.74582601, 0.69769347,
    0.26496658, 0.15650682, 0.10794004, 0.19025302, 0.32475743, 0.5195871, 0.57978857, 0.32398483,
    0.096927792, 0.074811816, 0.17997633, 0.53527933, 0.57831484, 0.40560108, 0.24023508,
    0.14073586,
  ],
  prelu_ch: [
    0.62160999, -0.023692904, -0.18248767, -0.17661625, -0.051460922, 0.67252183, 0.99705118,
    0.55135006, -0.064054735, -0.18889666, -0.25172147, -0.025640966,
  ],
};

function gen(n: number, seed: number, amp = 0.3): number[] {
  return Array.from({ length: n }, (_, k) => Math.sin(0.37 * k + seed) * amp);
}

function inp(shape: number[], seed: number, dtype: "float32" | "float64" = "float32"): Tensor {
  const n = shape.reduce((a, b) => a * b, 1);
  const flat = Array.from({ length: n }, (_, k) => Math.cos(0.91 * k + seed));
  // Build through nested arrays so the tensor has the requested shape.
  const t = tensor(flat, { dtype });
  return t.reshape(shape);
}

function setParams(module: Module, values: Record<string, number[]>): void {
  const sd = module.stateDict();
  for (const [name, data] of Object.entries(values)) {
    const entry = sd.parameters[name];
    if (!entry) throw new Error(`no parameter ${name}`);
    entry.data = data;
  }
  module.loadStateDict(sd);
}

function attnParams(prefix: string, off: number): Record<string, number[]> {
  return {
    [`${prefix}in_proj_weight_q`]: gen(16, 1 + off),
    [`${prefix}in_proj_weight_k`]: gen(16, 2 + off),
    [`${prefix}in_proj_weight_v`]: gen(16, 3 + off),
    [`${prefix}out_proj_weight`]: gen(16, 4 + off),
    [`${prefix}in_proj_bias_q`]: gen(4, 5 + off, 0.1),
    [`${prefix}in_proj_bias_k`]: gen(4, 6 + off, 0.1),
    [`${prefix}in_proj_bias_v`]: gen(4, 7 + off, 0.1),
    [`${prefix}out_proj_bias`]: gen(4, 8 + off, 0.1),
  };
}

function ffnNormParams(norms: number): Record<string, number[]> {
  const p: Record<string, number[]> = {
    "linear1.weight": gen(32, 9),
    "linear1.bias": gen(8, 10, 0.1),
    "linear2.weight": gen(32, 11),
    "linear2.bias": gen(4, 12, 0.1),
  };
  for (let i = 0; i < norms; i++) {
    p[`norm${i + 1}.weight`] = gen(4, 13 + 2 * i, 0.2).map((v) => 1 + v);
    p[`norm${i + 1}.bias`] = gen(4, 14 + 2 * i, 0.1);
  }
  return p;
}

function values(t: Tensor | GradTensor): number[] {
  const tt = GradTensor.isGradTensor(t) ? t.tensor : t;
  return Array.from(tt.reshape([tt.size]).data as ArrayLike<number>, Number);
}

function expectClose(actual: number[], expected: number[], tol = 1e-5): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(Math.abs((actual[i] as number) - (expected[i] as number))).toBeLessThan(tol);
  }
}

function makeMha(): MultiheadAttention {
  const mha = new MultiheadAttention(4, 2);
  setParams(mha, attnParams("", 0));
  mha.eval();
  return mha;
}

describe("MultiheadAttention (PyTorch parity)", () => {
  const q = inp([2, 3, 4], 0.1);
  const k = inp([2, 5, 4], 0.7);
  const v = inp([2, 5, 4], 1.3);

  it("self-attention matches torch", () => {
    expectClose(values(makeMha().forward(q, q, q)), REF.mha_self as number[]);
  });

  it("cross-attention matches torch", () => {
    expectClose(values(makeMha().forward(q, k, v)), REF.mha_cross as number[]);
  });

  it("uses key as value when value is omitted", () => {
    // Previously the value silently defaulted to the query.
    const out = makeMha().forward(q, k);
    expect(out.shape).toEqual([2, 3, 4]);
    expectClose(values(out), REF.mha_kv_shared as number[]);
  });

  it("causal mask matches torch", () => {
    expectClose(values(makeMha().forward(q, q, q, causalMask(3))), REF.mha_causal as number[]);
  });

  it("boolean mask marks positions that must not be attended (true = masked)", () => {
    const flags = [
      [0, 1, 0, 1, 0],
      [1, 0, 0, 0, 1],
      [0, 0, 1, 0, 0],
    ];
    const mask = tensor(flags.flat(), { dtype: "uint8" }).reshape([3, 5]);
    const boolMask = tensor(flags.flat().map((f) => f === 1) as unknown as number[], {
      dtype: "bool",
    }).reshape([3, 5]);
    const out = makeMha().forward(q, k, v, boolMask);
    expectClose(values(out), REF.mha_boolmask as number[]);
    // A uint8 mask is numeric and is added to the scores, so it must differ.
    const numeric = makeMha().forward(q, k, v, mask);
    expect(Math.abs((values(numeric)[0] as number) - (values(out)[0] as number))).toBeGreaterThan(
      1e-4
    );
  });

  it("2D input matches the batched result", () => {
    const q2 = inp([3, 4], 0.1);
    const out = makeMha().forward(q2, q2, q2);
    expect(out.shape).toEqual([3, 4]);
    expectClose(values(out), (REF.mha_2d as number[]).slice(0, 12));
  });

  it("accepts float64 inputs (cast to the float32 parameters)", () => {
    const q64 = inp([2, 3, 4], 0.1, "float64");
    const out = makeMha().forward(q64, q64, q64);
    expect(out.dtype).toBe("float32");
    expectClose(values(out), REF.mha_self as number[]);
  });

  it("gradients flow back through a float64 input", () => {
    const x = parameter(inp([1, 3, 4], 0.1, "float64"));
    const out = makeMha().forward(x, x, x);
    out.sum().backward();
    expect(x.grad).not.toBeNull();
    expect(x.grad?.shape).toEqual([1, 3, 4]);
  });

  it("rejects masks that cannot broadcast to the score shape", () => {
    const mha = makeMha();
    expect(() => mha.forward(q, q, q, tensor([[0, 0]]))).toThrow(ShapeError);
    expect(() => mha.forward(q, q, q, tensor([0, 0, 0]))).toThrow(ShapeError);
  });

  it("rejects string dtype for key and value", () => {
    const s = tensor([["a", "b", "c", "d"]]);
    expect(() => makeMha().forward(inp([1, 4], 0), s)).toThrow(DTypeError);
  });
});

describe("causalMask", () => {
  it("validates seqLen", () => {
    expect(() => causalMask(-1)).toThrow(InvalidParameterError);
    expect(() => causalMask(2.5)).toThrow(InvalidParameterError);
    expect(causalMask(0).shape).toEqual([0, 0]);
  });
  it("is zero on and below the diagonal", () => {
    expect(values(causalMask(2))).toEqual([0, -1e9, 0, 0]);
  });
});

describe("Transformer layers (PyTorch parity)", () => {
  const q = inp([2, 3, 4], 0.1);
  const mem = inp([2, 5, 4], 2.0);

  function makeEnc(): TransformerEncoderLayer {
    const enc = new TransformerEncoderLayer(4, 2, 8, { dropout: 0 });
    setParams(enc, {
      ...Object.fromEntries(
        Object.entries(attnParams("", 0)).map(([key, val]) => [`self_attn.${key}`, val])
      ),
      ...ffnNormParams(2),
    });
    return enc.eval();
  }

  function makeDec(): TransformerDecoderLayer {
    const dec = new TransformerDecoderLayer(4, 2, 8, { dropout: 0 });
    setParams(dec, {
      ...Object.fromEntries(
        Object.entries(attnParams("", 0)).map(([key, val]) => [`self_attn.${key}`, val])
      ),
      ...Object.fromEntries(
        Object.entries(attnParams("", 20)).map(([key, val]) => [`multihead_attn.${key}`, val])
      ),
      ...ffnNormParams(3),
    });
    return dec.eval();
  }

  it("encoder layer matches torch", () => {
    expectClose(values(makeEnc().forward(q)), REF.enc as number[]);
  });

  it("encoder layer applies a source mask", () => {
    expectClose(values(makeEnc().forward(q, causalMask(3))), REF.enc_masked as number[]);
  });

  it("decoder layer is causal by default and matches torch with tgt_mask", () => {
    expectClose(values(makeDec().forward(q, mem)), REF.dec_causal as number[]);
  });

  it("decoder layer honors an explicit tgtMask instead of the causal one", () => {
    const zero = tensor([
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ]);
    expectClose(values(makeDec().forward(q, mem, zero)), REF.dec_nomask as number[]);
  });

  it("float64 inputs work in encoder and decoder layers", () => {
    const q64 = inp([2, 3, 4], 0.1, "float64");
    const mem64 = inp([2, 5, 4], 2.0, "float64");
    expectClose(values(makeEnc().forward(q64)), REF.enc as number[]);
    expectClose(values(makeDec().forward(q64, mem64)), REF.dec_causal as number[]);
  });

  it("clone copies weights, mode and trainability instead of re-initializing", () => {
    const enc = makeEnc();
    const copy = enc.clone();
    expect(copy.training).toBe(false);
    expectClose(values(copy.forward(q)), values(enc.forward(q)), 1e-12);
    const a = Array.from(enc.namedParameters());
    const b = new Map(copy.namedParameters());
    for (const [name, p] of a) {
      const other = b.get(name);
      expect(other).toBeDefined();
      expect(other).not.toBe(p);
      expect(values(other as GradTensor)).toEqual(values(p));
    }
    const dec = makeDec();
    expectClose(values(dec.clone().forward(q, mem)), values(dec.forward(q, mem)), 1e-12);
  });

  it("TransformerEncoder layers all start from the template weights", () => {
    const template = makeEnc();
    const stack = new TransformerEncoder(template, 3);
    stack.eval();
    const layer0 = new Map(template.namedParameters());
    for (const [name, p] of stack.namedParameters()) {
      const m = /^layers\.(\d+)\.(.*)$/.exec(name);
      if (!m) continue;
      expect(values(p)).toEqual(values(layer0.get(m[2] as string) as GradTensor));
    }
    // identical layers: output of the stack equals applying the template three times
    let ref: GradTensor = GradTensor.fromTensor(q);
    for (let i = 0; i < 3; i++) ref = template.forward(ref);
    expectClose(values(stack.forward(q)), values(ref));
  });

  it("TransformerDecoder passes masks through and requires memory", () => {
    const stack = new TransformerDecoder(makeDec(), 2);
    stack.eval();
    expect(stack.forward(q, mem).shape).toEqual([2, 3, 4]);
    expect(() => stack.forward(q)).toThrow(InvalidParameterError);
  });

  it("FullTransformer supports masks and an optional final norm", () => {
    const plain = new FullTransformer(4, 2, 1, 1, 8, { dropout: 0 });
    const withNorm = new FullTransformer(4, 2, 1, 1, 8, { dropout: 0, finalNorm: true });
    const names = (m: Module) => Array.from(m.namedParameters()).map(([n]) => n);
    expect(names(plain).some((n) => n.startsWith("encoder.norm."))).toBe(false);
    expect(names(withNorm)).toContain("encoder.norm.weight");
    expect(names(withNorm)).toContain("decoder.norm.weight");
    plain.eval();
    const src = inp([2, 5, 4], 0.3);
    const tgt = inp([2, 3, 4], 0.4);
    expect(plain.forward(src, tgt, causalMask(5), causalMask(3)).shape).toEqual([2, 3, 4]);
    expect(() => plain.forward(src)).toThrow(InvalidParameterError);
  });
});

describe("PositionalEncoding", () => {
  it("matches the sinusoidal table", () => {
    const pe = new PositionalEncoding(6, { dropout: 0, maxLen: 8 });
    const zeros = tensor(new Array(24).fill(0)).reshape([4, 6]);
    expectClose(values(pe.forward(zeros)), REF.pe as number[], 1e-6);
  });

  it("works with the default float32 input dtype (used to throw a dtype mismatch)", () => {
    const pe = new PositionalEncoding(4, { dropout: 0, maxLen: 10 });
    const out = pe.forward(inp([2, 3, 4], 0));
    expect(out.dtype).toBe("float32");
    expect(out.shape).toEqual([2, 3, 4]);
  });

  it("keeps float64 for float64 input and casts integer input to float32", () => {
    const pe = new PositionalEncoding(4, { dropout: 0, maxLen: 10 });
    expect(pe.forward(inp([3, 4], 0, "float64")).dtype).toBe("float64");
    expect(pe.forward(tensor([[1, 2, 3, 4]], { dtype: "int32" })).dtype).toBe("float32");
  });

  it("rejects a wrong last dimension instead of broadcasting", () => {
    const pe = new PositionalEncoding(4, { dropout: 0 });
    expect(() => pe.forward(inp([2, 3, 1], 0))).toThrow(ShapeError);
    expect(() => pe.forward(inp([2, 3, 5], 0))).toThrow(ShapeError);
  });

  it("validates maxLen", () => {
    expect(() => new PositionalEncoding(4, { maxLen: 0 })).toThrow(InvalidParameterError);
    expect(() => new PositionalEncoding(4, { maxLen: 2.5 })).toThrow(InvalidParameterError);
  });
});

describe("activations on strided views", () => {
  const x = tensor([
    [-3.5, -1, 0],
    [0.5, 3, 7],
  ]);
  const xt = transpose(x); // shape [3, 2], non-contiguous

  it("Hardsigmoid / Hardswish / Softsign / SELU read through the strides", () => {
    expectClose(values(new Hardsigmoid().forward(xt)), REF.hardsigmoid_t as number[], 1e-6);
    expectClose(values(new Hardswish().forward(xt)), REF.hardswish_t as number[], 1e-6);
    expectClose(values(new Softsign().forward(xt)), REF.softsign_t as number[], 1e-6);
    expectClose(values(new SELU().forward(xt)), REF.selu_t as number[], 1e-5);
  });

  it("GradTensor and Tensor paths agree for the strided input", () => {
    const g = GradTensor.fromTensor(xt);
    for (const layer of [new Hardsigmoid(), new Hardswish(), new Softsign(), new SELU()]) {
      expectClose(values(layer.forward(g)), values(layer.forward(xt)), 1e-6);
    }
  });
});

describe("gradients at the kinks (PyTorch conventions)", () => {
  const kinks = [-4, -3, -1, 0, 1, 3, 4];

  function gradOf(apply: (x: GradTensor) => GradTensor): number[] {
    const p = parameter(tensor(kinks));
    apply(p).sum().backward();
    return values(p.grad as Tensor);
  }

  it("Hardsigmoid has zero gradient at -3 and 3", () => {
    const g = 0.1666666716337204;
    expectClose(
      gradOf((x) => new Hardsigmoid().forward(x)),
      [0, 0, g, g, g, 0, 0],
      1e-6
    );
  });

  it("Hardswish matches torch at -3 and 3", () => {
    expectClose(
      gradOf((x) => new Hardswish().forward(x)),
      [0, 0, 0.1666666567325592, 0.5, 0.8333333730697632, 1, 1],
      1e-6
    );
  });

  it("PReLU passes the slope as the gradient of x at x = 0, and trains the slope", () => {
    const layer = new PReLU();
    const p = parameter(tensor(kinks));
    layer.forward(p).sum().backward();
    expectClose(values(p.grad as Tensor), [0.25, 0.25, 0.25, 0.25, 1, 1, 1], 1e-7);
    const slope = Array.from(layer.namedParameters()).find(([n]) => n === "weight")?.[1];
    expectClose(values(slope?.grad as Tensor), [-8], 1e-6);
  });
});

describe("Softplus / Mish", () => {
  const big = [100, -100, 0, 20, -5, 3];

  it("gradients stay finite for large inputs (used to be NaN)", () => {
    const p = parameter(tensor(big));
    const out = new Softplus().forward(p);
    out.sum().backward();
    expect(values(out).every(Number.isFinite)).toBe(true);
    expectClose(values(p.grad as Tensor), REF.softplus_grad as number[], 1e-6);
  });

  it("Mish value and gradient match torch for large inputs", () => {
    const p = parameter(tensor(big));
    const out = new Mish().forward(p);
    out.sum().backward();
    expectClose(values(out), REF.mish_val as number[], 1e-5);
    expectClose(values(p.grad as Tensor), REF.mish_grad as number[], 1e-5);
  });

  it("beta scales the transition", () => {
    const x = tensor([
      [-3.5, -1, 0],
      [0.5, 3, 7],
    ]);
    const layer = new Softplus(2);
    expectClose(values(layer.forward(x)), REF.softplus_b2 as number[], 1e-7);
    expectClose(values(layer.forward(GradTensor.fromTensor(x))), REF.softplus_b2 as number[], 1e-5);
    expect(layer.toString()).toBe("Softplus(beta=2)");
    expect(new Softplus().toString()).toBe("Softplus()");
    expect(() => new Softplus(0)).toThrow(InvalidParameterError);
    expect(() => new Softplus(Number.NaN)).toThrow(InvalidParameterError);
  });
});

describe("Softmax2d", () => {
  it("normalizes over channels for 3-D (C, H, W) input", () => {
    const out = new Softmax2d().forward(inp([2, 2, 3], 0.3));
    expect(out.shape).toEqual([2, 2, 3]);
    expectClose(values(out), REF.sm2d_3d as number[], 1e-6);
  });

  it("normalizes over channels for 4-D (N, C, H, W) input, both paths", () => {
    const t = inp([2, 3, 2, 2], 0.5);
    expectClose(values(new Softmax2d().forward(t)), REF.sm2d_4d as number[], 1e-6);
    expectClose(
      values(new Softmax2d().forward(GradTensor.fromTensor(t))),
      REF.sm2d_4d as number[],
      1e-6
    );
  });

  it("rejects other ranks (used to return garbage shapes)", () => {
    expect(() => new Softmax2d().forward(inp([2, 3], 0))).toThrow(ShapeError);
    expect(() => new Softmax2d().forward(inp([2, 3, 4, 5, 6], 0))).toThrow(ShapeError);
  });
});

describe("PReLU", () => {
  function perChannel(): PReLU {
    const p = new PReLU(3);
    setParams(p, { weight: [0.1, 0.2, 0.3] });
    return p;
  }

  it("applies one slope per channel along axis 1 (matches torch)", () => {
    const out = perChannel().forward(inp([2, 3, 2], 0.9));
    expect(out.shape).toEqual([2, 3, 2]);
    expectClose(values(out), REF.prelu_ch as number[], 1e-6);
  });

  it("handles 4-D input (used to fail the broadcast)", () => {
    expect(perChannel().forward(inp([2, 3, 4, 5], 0.2)).shape).toEqual([2, 3, 4, 5]);
  });

  it("rejects a channel mismatch with a ShapeError", () => {
    expect(() => perChannel().forward(inp([2, 5, 3], 0))).toThrow(ShapeError);
    expect(() => perChannel().forward(tensor(1))).toThrow(ShapeError);
  });

  it("keeps the input shape for scalar input and a single shared slope", () => {
    expect(new PReLU().forward(tensor(-2)).shape).toEqual([]);
    expect(values(new PReLU(1, 0.5).forward(tensor(-2)))).toEqual([-1]);
  });

  it("keeps the autograd graph when casting a float64 input", () => {
    const x = parameter(inp([2, 3], 0.4, "float64"));
    const layer = perChannel();
    layer.forward(x).sum().backward();
    expect(x.grad).not.toBeNull();
    expect(x.grad?.shape).toEqual([2, 3]);
  });

  it("validates init and string input", () => {
    expect(() => new PReLU(1, Number.NaN)).toThrow(InvalidParameterError);
    expect(() => new PReLU().forward(tensor(["a"]))).toThrow(DTypeError);
  });
});

describe("constructor validation", () => {
  it("Hardtanh rejects minVal > maxVal and NaN bounds at construction", () => {
    expect(() => new Hardtanh(2, 1)).toThrow(InvalidParameterError);
    expect(() => new Hardtanh(Number.NaN, 1)).toThrow(InvalidParameterError);
    expect(new Hardtanh(1, 1).toString()).toBe("Hardtanh(minVal=1, maxVal=1)");
  });

  it("LeakyReLU rejects non-finite alpha", () => {
    expect(() => new LeakyReLU(Number.NaN)).toThrow(InvalidParameterError);
    expect(() => new LeakyReLU(Number.POSITIVE_INFINITY)).toThrow(InvalidParameterError);
  });

  it("GLU rejects a non-integer dim", () => {
    expect(() => new GLU(0.5)).toThrow(InvalidParameterError);
  });

  it("Softmin accepts a named axis", () => {
    const out = new Softmin("columns").forward(tensor([[1, 2, 3]]));
    // float32 input gives a float32 output, so the sum is exact only to float32 precision
    expect(values(out).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
  });

  it("GELU documents and uses the tanh approximation", () => {
    const g = values(new GELU().forward(tensor([1])))[0] as number;
    expect(g).toBeCloseTo(0.8411919906082768, 6);
  });
});
