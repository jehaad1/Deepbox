import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { MultiheadAttention, TransformerEncoderLayer } from "../src/nn/layers/attention";

describe("deepbox/nn - Attention", () => {
  it("MultiheadAttention handles 2D inputs", () => {
    const mha = new MultiheadAttention(4, 2, { bias: false });
    const x = tensor([
      [1, 0, 0, 1],
      [0, 1, 1, 0],
      [1, 1, 0, 0],
    ]);
    const out = mha.forward(x, x, x);
    expect(out.shape).toEqual([3, 4]);
    expect(mha.toString()).toContain("MultiheadAttention");
  });

  it("MultiheadAttention handles batched 3D inputs", () => {
    const mha = new MultiheadAttention(4, 2);
    const x = tensor([
      [
        [1, 0, 0, 1],
        [0, 1, 1, 0],
      ],
      [
        [0, 0, 1, 1],
        [1, 1, 0, 0],
      ],
    ]);
    const out = mha.forward(x, x, x);
    expect(out.shape).toEqual([2, 2, 4]);
  });

  it("throws for invalid head configuration", () => {
    expect(() => new MultiheadAttention(5, 2)).toThrow();
  });

  it("validates integer dimensions", () => {
    expect(() => new MultiheadAttention(4.5, 2)).toThrow();
    expect(() => new MultiheadAttention(4, 2.5)).toThrow();
  });

  it("validates dropout range", () => {
    expect(() => new MultiheadAttention(4, 2, { dropout: -0.1 })).toThrow();
    expect(() => new MultiheadAttention(4, 2, { dropout: 1 })).toThrow();
  });

  it("MultiheadAttention validates input shapes and dtypes", () => {
    const mha = new MultiheadAttention(4, 2);
    const q = tensor([
      [1, 0, 0, 1],
      [0, 1, 1, 0],
    ]);
    const kBad = tensor([
      [1, 0, 0],
      [0, 1, 1],
    ]);
    expect(() => mha.forward(q, kBad, q)).toThrow();

    const q3 = tensor([
      [
        [1, 0, 0, 1],
        [0, 1, 1, 0],
      ],
    ]);
    const k3 = tensor([
      [
        [1, 0, 0, 1],
        [0, 1, 1, 0],
      ],
      [
        [1, 1, 0, 0],
        [0, 0, 1, 1],
      ],
    ]);
    expect(() => mha.forward(q3, k3, q3)).toThrow();

    const s = tensor([
      ["a", "b", "c", "d"],
      ["e", "f", "g", "h"],
    ]);
    expect(() => mha.forward(s)).toThrow();
  });

  it("TransformerEncoderLayer preserves shape", () => {
    const layer = new TransformerEncoderLayer(4, 2, 8);
    const x2d = tensor([
      [1, 2, 3, 4],
      [2, 3, 4, 5],
    ]);
    const out2d = layer.forward(x2d);
    expect(out2d.shape).toEqual([2, 4]);

    const x3d = tensor([
      [
        [1, 2, 3, 4],
        [4, 3, 2, 1],
      ],
    ]);
    const out3d = layer.forward(x3d);
    expect(out3d.shape).toEqual([1, 2, 4]);
  });

  it("TransformerEncoderLayer validates dropout", () => {
    expect(() => new TransformerEncoderLayer(4, 2, 8, { dropout: 1 })).toThrow();
    expect(() => new TransformerEncoderLayer(4, 2, 8, { dropout: -0.1 })).toThrow();
  });
});
