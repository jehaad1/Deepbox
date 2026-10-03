import { describe, expect, it } from "vitest";
import { DeepboxError } from "../../src/core";
import { noGrad, tensor } from "../../src/ndarray";
import { GroupNorm, LayerNorm, Linear, mseLoss, ReLU, RMSNorm, Sequential } from "../../src/nn";
import { Adam } from "../../src/optim";

// Issues found by the release owner's review of wave 3.
describe("1.5.0 release review", () => {
  it("normalization layers without parameters keep the input float dtype", () => {
    const x = tensor(
      [
        [0.5, -1.2, 0.3, 1],
        [2.1, 0.4, -0.7, 0],
      ],
      { dtype: "float64" }
    );
    const ln = new LayerNorm(4, { elementwiseAffine: false }).forward(x);
    expect(ln.dtype).toBe("float64");
    // torch.nn.LayerNorm(4, elementwise_affine=False) in float64
    const expected = [
      0.4267943594143175, -1.6462068148837965, 0.1829118683204217, 1.0365005871490567,
    ];
    const row = (ln.toArray() as number[][])[0] ?? [];
    for (let i = 0; i < expected.length; i++)
      expect(row[i]).toBeCloseTo(expected[i] ?? Number.NaN, 14);

    expect(new GroupNorm(2, 4, { affine: false }).forward(x.reshape([1, 4, 2])).dtype).toBe(
      "float64"
    );
    expect(new RMSNorm(4, { elementwiseAffine: false }).forward(x).dtype).toBe("float64");
    // With parameters, the layer computes in its parameter dtype.
    expect(new LayerNorm(4).forward(x).dtype).toBe("float32");
  });
});

describe("1.5.0 training loop with plain tensors", () => {
  it("type-checks and trains without wrapping data in parameter()", () => {
    const X = tensor([
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ]);
    const y = tensor([[0], [1], [1], [0]]);
    const model = new Sequential(new Linear(2, 16), new ReLU(), new Linear(16, 1));
    const optimizer = new Adam(model.parameters(), { lr: 0.05 });
    const losses: number[] = [];
    for (let epoch = 0; epoch < 200; epoch++) {
      optimizer.zeroGrad();
      const loss = mseLoss(model.forward(X), y);
      loss.backward();
      optimizer.step();
      losses.push(Number(loss.item()));
    }
    expect(losses[losses.length - 1]).toBeLessThan((losses[0] ?? 0) / 10);
  });

  it("plain tensors report no gradient and backward() explains why", () => {
    const t = tensor([1, 2]);
    expect(t.requiresGrad).toBe(false);
    expect(t.grad).toBeNull();
    expect(() => t.backward()).toThrow(DeepboxError);
    const model = new Sequential(new Linear(2, 1));
    const loss = noGrad(() => mseLoss(model.forward(tensor([[1, 2]])), tensor([[0]])));
    expect(loss.requiresGrad).toBe(false);
    expect(() => loss.backward()).toThrow(/does not track gradients/);
  });
});

describe("1.5.0 binomial sampler for large means (BTPE)", () => {
  it("draws quickly for huge n and matches the binomial moments", async () => {
    const { binomial, setSeed } = await import("../../src/random");
    setSeed(11);
    const start = performance.now();
    const big = Array.from(
      binomial(1e12, 0.3, [2000], { dtype: "int64" }).data as BigInt64Array,
      Number
    );
    // Inversion needed about 4.6e5 steps per draw here; BTPE needs a few.
    expect(performance.now() - start).toBeLessThan(2000);
    const mean = big.reduce((a, b) => a + b, 0) / big.length;
    const sd = Math.sqrt(1e12 * 0.3 * 0.7);
    expect(Math.abs(mean - 3e11)).toBeLessThan((5 * sd) / Math.sqrt(big.length));

    const xs = Array.from(binomial(5000, 0.37, [200000]).data as Int32Array);
    const m = xs.reduce((a, b) => a + b, 0) / xs.length;
    const v = xs.reduce((a, b) => a + (b - m) ** 2, 0) / xs.length;
    const trueVar = 5000 * 0.37 * 0.63;
    expect(Math.abs(m - 1850)).toBeLessThan(5 * Math.sqrt(trueVar / xs.length));
    expect(Math.abs(v / trueVar - 1)).toBeLessThan(0.02);
  });
});
