import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { MultiheadAttention } from "../src/nn/layers/attention";

describe("nn attention extra branches", () => {
  it("validates constructor parameters", () => {
    expect(() => new MultiheadAttention(0, 1)).toThrow(/embedDim/);
    expect(() => new MultiheadAttention(4, 0)).toThrow(/numHeads/);
    expect(() => new MultiheadAttention(5, 2)).toThrow(/divisible/);
    expect(() => new MultiheadAttention(4, 2, { dropout: 1 })).toThrow(/dropout/);
  });

  it("rejects string tensors and rank mismatches", () => {
    const mha = new MultiheadAttention(4, 2);
    const s = tensor(["a", "b", "c", "d"]);
    expect(() => mha.forward(s)).toThrow(/string/);

    const q = tensor([[1, 2, 3, 4]]);
    const k = tensor([1, 2, 3, 4]);
    expect(() => mha.forward(q, k, q)).toThrow(/same rank/);
  });

  it("validates embedDim and batch mismatches", () => {
    const mha = new MultiheadAttention(4, 2);
    const q = tensor([[[1, 2, 3, 4]]]);
    const badEmbed = tensor([[[1, 2]]]);
    expect(() => mha.forward(badEmbed, badEmbed, badEmbed)).toThrow(/embedDim/);

    const k = tensor([[[1, 2, 3, 4]], [[5, 6, 7, 8]]]);
    expect(() => mha.forward(q, k, q)).toThrow(/batch size mismatch/);
  });

  it("validates key/value sequence length mismatch", () => {
    const mha = new MultiheadAttention(4, 2);
    const q = tensor([[[1, 0, 0, 1]]]);
    const k = tensor([
      [
        [1, 0, 0, 1],
        [0, 1, 1, 0],
      ],
    ]);
    const v = tensor([[[1, 1, 1, 1]]]);
    expect(() => mha.forward(q, k, v)).toThrow(/sequence length/i);
  });
});
