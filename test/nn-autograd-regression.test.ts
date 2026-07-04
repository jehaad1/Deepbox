import { beforeEach, describe, expect, it } from "vitest";
import { setSeed } from "../src/core";
import * as db from "../src/ndarray";
import * as nn from "../src/nn";

// Finite-difference checks are sensitive near max-pool tie points; a fixed
// seed keeps the random inputs identical regardless of suite ordering.
beforeEach(() => {
  setSeed(1234);
});

/**
 * Regression tests for the v1.0.0 audit: layers that previously returned
 * detached tensors (no gradient) must now propagate gradients to their inputs
 * and parameters. Verified against finite differences.
 */

function flat(t: { toArray(): unknown }): number[] {
  const o: number[] = [];
  const rec = (a: unknown) => (Array.isArray(a) ? a.forEach(rec) : o.push(a as number));
  rec(t.toArray());
  return o;
}

/** Max finite-difference error of the input gradient for a layer. */
function inputGradError(
  makeLayer: () => { forward: (x: db.GradTensor) => db.GradTensor },
  shape: number[]
): number {
  const layer = makeLayer();
  const x0 = db.randn(shape, { dtype: "float64" });
  const x = db.GradTensor.fromTensor(x0, { requiresGrad: true });
  layer.forward(x).sum().backward();
  const grad = x.grad;
  if (!grad) throw new Error("missing input gradient");
  const gA = flat(grad);
  const xd = x0.data as Float64Array;
  const eps = 3e-3;
  let e = 0;
  for (let i = 0; i < Math.min(xd.length, 16); i++) {
    const o = xd[i] as number;
    xd[i] = o + eps;
    const lp = layer.forward(db.GradTensor.fromTensor(x0)).sum().tensor.toArray() as number;
    xd[i] = o - eps;
    const lm = layer.forward(db.GradTensor.fromTensor(x0)).sum().tensor.toArray() as number;
    xd[i] = o;
    e = Math.max(e, Math.abs((lp - lm) / (2 * eps) - (gA[i] as number)));
  }
  return e;
}

describe("nn autograd regression (previously-detached layers)", () => {
  it("Conv3d propagates input and weight gradients", () => {
    const layer = new nn.Conv3d(2, 3, 2, { stride: 1, padding: 1 });
    expect(inputGradError(() => layer, [1, 2, 4, 4, 4])).toBeLessThan(1e-3);
    expect(layer.weight.grad).not.toBeNull();
  });

  it("ConvTranspose2d works with default bias and propagates gradients", () => {
    const layer = new nn.ConvTranspose2d(2, 3, 2, { stride: 2 });
    expect(inputGradError(() => layer, [1, 2, 3, 3])).toBeLessThan(1e-3);
    expect(layer.weight.grad).not.toBeNull();
  });

  it("ConvTranspose1d propagates gradients", () => {
    const layer = new nn.ConvTranspose1d(2, 3, 3, { stride: 2 });
    expect(inputGradError(() => layer, [1, 2, 5])).toBeLessThan(1e-3);
  });

  it("pooling layers route gradients", () => {
    expect(inputGradError(() => new nn.MaxPool1d(2), [1, 2, 6])).toBeLessThan(1e-3);
    expect(inputGradError(() => new nn.MaxPool3d(2), [1, 1, 4, 4, 4])).toBeLessThan(1e-3);
    expect(inputGradError(() => new nn.AvgPool3d(2), [1, 1, 4, 4, 4])).toBeLessThan(1e-3);
    expect(inputGradError(() => new nn.AdaptiveAvgPool2d([2, 2]), [1, 2, 5, 5])).toBeLessThan(1e-3);
    expect(inputGradError(() => new nn.AdaptiveMaxPool2d([2, 2]), [1, 2, 5, 5])).toBeLessThan(1e-3);
  });

  it("MaxPool2d uses -inf padding (no zero contamination on negative inputs)", () => {
    const neg = db.tensor(
      [
        [
          [
            [-1, -2, -3],
            [-4, -5, -6],
            [-7, -8, -9],
          ],
        ],
      ],
      { dtype: "float64" }
    );
    const mp = new nn.MaxPool2d(2, { stride: 1, padding: 1 });
    const out = flat(mp.forward(db.GradTensor.fromTensor(neg)));
    // Every output must be negative — a spurious 0 would indicate 0-padding.
    expect(out.every((v) => v < 0)).toBe(true);
  });

  it("Embedding propagates gradient to the weight table (duplicate indices accumulate)", () => {
    const emb = new nn.Embedding(5, 3);
    const idx = db.tensor([0, 2, 2, 4], { dtype: "int32" });
    emb.forward(idx).sum().backward();
    const g = emb.weight.grad?.toArray() as number[][];
    expect(g[2]).toEqual([2, 2, 2]); // index 2 used twice
    expect(g[1]).toEqual([0, 0, 0]); // index 1 unused
  });

  it("RNN/LSTM/GRU are trainable (all params receive gradients)", () => {
    for (const layer of [new nn.RNN(3, 4), new nn.LSTM(3, 4), new nn.GRU(3, 4)]) {
      const y = layer.forward(db.randn([2, 3, 3], { dtype: "float64" }));
      y.sum().backward();
      for (const p of layer.parameters()) expect(p.grad).not.toBeNull();
    }
  });

  it("TransformerDecoderLayer self-attention is causally masked (no future leakage)", () => {
    const dec = new nn.TransformerDecoderLayer(4, 2, 8, { dropout: 0 });
    dec.eval();
    // Parameters default to float32, so use float32 inputs (dtype must match).
    const seq = db.randn([1, 3, 4], { dtype: "float32" });
    const mem = db.randn([1, 3, 4], { dtype: "float32" });
    const base = flat(dec.forward(seq, mem));
    // Perturb only the LAST timestep; the first timestep output must be
    // unchanged if self-attention cannot see the future.
    const seqData = seq.data as Float32Array;
    const perturbed = db.tensor(Array.from(seqData), { dtype: "float32" }).reshape([1, 3, 4]);
    (perturbed.data as Float32Array)[8] += 1.0; // an element of timestep index 2
    const after = flat(dec.forward(perturbed, mem));
    // first-timestep block is the first 4 outputs
    let maxDiff = 0;
    for (let i = 0; i < 4; i++) maxDiff = Math.max(maxDiff, Math.abs(base[i]! - after[i]!));
    expect(maxDiff).toBeLessThan(1e-9);
  });

  it("crossEntropyLoss accepts float64 logits", () => {
    const pred = db.GradTensor.fromTensor(
      db.tensor(
        [
          [2, 0.5, 0.1],
          [0.1, 1.5, 0.2],
        ],
        { dtype: "float64" }
      ),
      { requiresGrad: true }
    );
    const loss = nn.crossEntropyLoss(pred, db.tensor([0, 1], { dtype: "int32" }));
    expect(Number.isFinite((loss as db.GradTensor).tensor.toArray() as number)).toBe(true);
  });
});
