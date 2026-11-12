// NN-13: dropout layers accept p = 1. As torch.nn.functional.dropout(x, 1.0, training=True),
// dropout2d and alpha_dropout, the training output is all zeros and the gradient is zero;
// evaluation mode returns the input unchanged.
import { describe, expect, it } from "vitest";
import { InvalidParameterError } from "../../src/core";
import { GradTensor, type Tensor, tensor } from "../../src/ndarray";
import { AlphaDropout, Dropout, Dropout2d } from "../../src/nn";

function values(t: Tensor): number[] {
  return Array.from(t.reshape([t.size]).data as ArrayLike<number>, Number);
}

describe("NN-13: dropout probability 1", () => {
  const layers: ReadonlyArray<readonly [string, () => Dropout | Dropout2d | AlphaDropout]> = [
    ["Dropout", () => new Dropout(1)],
    ["Dropout2d", () => new Dropout2d(1)],
    ["AlphaDropout", () => new AlphaDropout(1)],
  ];

  it.each(layers)("%s zeroes everything in training mode", (_name, make) => {
    const layer = make().train();
    const x = tensor([
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
    ]);
    const out = layer.forward(x);
    expect(values(out)).toEqual([0, 0, 0, 0, 0, 0, 0, 0]);
    expect([...out.shape]).toEqual([1, 2, 2, 2]);
  });

  it.each(layers)("%s is the identity in evaluation mode", (_name, make) => {
    const layer = make().eval();
    const x = tensor([[[[1, 2]], [[3, 4]]]]);
    expect(values(layer.forward(x))).toEqual([1, 2, 3, 4]);
  });

  it.each(layers)("%s gives a zero gradient", (_name, make) => {
    const layer = make().train();
    const x = GradTensor.fromTensor(tensor([[[[1, 2]], [[3, 4]]]]), { requiresGrad: true });
    layer.forward(x).sum().backward();
    expect(values(x.grad as Tensor)).toEqual([0, 0, 0, 0]);
  });

  it("still rejects probabilities outside [0, 1]", () => {
    for (const bad of [-0.1, 1.5, Number.NaN]) {
      expect(() => new Dropout(bad)).toThrow(InvalidParameterError);
      expect(() => new Dropout2d(bad)).toThrow(InvalidParameterError);
      expect(() => new AlphaDropout(bad)).toThrow(InvalidParameterError);
    }
  });
});
