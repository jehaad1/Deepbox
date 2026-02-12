import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import {
  elu,
  gelu,
  leakyRelu,
  logSoftmax,
  mish,
  relu,
  sigmoid,
  softmax,
  softplus,
  swish,
} from "../src/ndarray/ops/activation";
import { tanh } from "../src/ndarray/ops/trigonometry";
import { transpose } from "../src/ndarray/tensor/shape";
import { toNum2D, toNumArr } from "./_helpers";

describe("deepbox/ndarray - Activation Branches", () => {
  it("covers BigInt paths", () => {
    const t = tensor([-1, 0, 2], { dtype: "int64" });
    expect(sigmoid(t).dtype).toBe("float64");
    expect(relu(t).dtype).toBe("float64");
    expect(leakyRelu(t, 0.1).dtype).toBe("float64");
    expect(elu(t).dtype).toBe("float64");
    expect(gelu(t).dtype).toBe("float64");
    expect(tanh(t).dtype).toBe("float64");
    expect(swish(t).dtype).toBe("float64");
    expect(mish(t).dtype).toBe("float64");
    expect(softplus(t).dtype).toBe("float64");
  });

  it("promotes non-float activations to float outputs", () => {
    const tInt = tensor([-1, 0, 1], { dtype: "int32" });
    const sigmoidInt = sigmoid(tInt);
    expect(sigmoidInt.dtype).toBe("float64");
    expect(toNumArr(sigmoidInt.toArray())[1]).toBeCloseTo(0.5, 6);

    const leaky = leakyRelu(tInt, 0.1);
    expect(leaky.dtype).toBe("float64");
    expect(toNumArr(leaky.toArray())[0]).toBeCloseTo(-0.1, 6);

    const tBool = tensor([0, 1], { dtype: "bool" });
    expect(sigmoid(tBool).dtype).toBe("float64");
    expect(leakyRelu(tBool).dtype).toBe("float64");
  });

  it("covers softmax/logSoftmax axis branches", () => {
    const x = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const sm = softmax(x, 1);
    expect(sm.shape).toEqual([2, 3]);

    const smNeg = softmax(x, -1);
    expect(smNeg.shape).toEqual([2, 3]);

    const lsm = logSoftmax(x, 0);
    expect(lsm.shape).toEqual([2, 3]);

    expect(() => softmax(x, 5)).toThrow(/out of bounds/);
    expect(() => logSoftmax(x, -3)).toThrow(/out of bounds/);
  });

  it("computes softmax/logSoftmax on strided views", () => {
    const x = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const xT = transpose(x);

    const sm = softmax(xT, 1);
    const expected = toNum2D(xT.toArray()).map((row) => {
      const maxVal = Math.max(...row);
      const exps = row.map((v) => Math.exp(v - maxVal));
      const sumExp = exps.reduce((acc, v) => acc + v, 0);
      return exps.map((v) => v / sumExp);
    });
    const actual = toNum2D(sm.toArray());
    for (let i = 0; i < expected.length; i++) {
      const expRow = expected[i] ?? [];
      const actRow = actual[i] ?? [];
      for (let j = 0; j < expRow.length; j++) {
        expect(actRow[j]).toBeCloseTo(expRow[j] ?? 0, 6);
      }
    }

    const lsm = logSoftmax(xT, 1);
    const actualLog = toNum2D(lsm.toArray());
    for (let i = 0; i < expected.length; i++) {
      const expRow = expected[i] ?? [];
      const actRow = actualLog[i] ?? [];
      for (let j = 0; j < expRow.length; j++) {
        expect(actRow[j]).toBeCloseTo(Math.log(expRow[j] ?? 1), 6);
      }
    }
  });

  it("throws on string dtype", () => {
    const s = tensor(["a", "b"]);
    expect(() => sigmoid(s)).toThrow();
    expect(() => relu(s)).toThrow();
    expect(() => leakyRelu(s)).toThrow();
    expect(() => elu(s)).toThrow();
    expect(() => gelu(s)).toThrow();
    expect(() => tanh(s)).toThrow();
    expect(() => swish(s)).toThrow();
    expect(() => mish(s)).toThrow();
    expect(() => softplus(s)).toThrow();
    expect(() => softmax(s)).toThrow();
    expect(() => logSoftmax(s)).toThrow();
  });
});
