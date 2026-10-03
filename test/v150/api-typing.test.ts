import { describe, expect, expectTypeOf, it } from "vitest";
import { type AnyTensor, GradTensor, parameter, type Tensor, tensor } from "../../src/ndarray";
import { Linear, maeLoss, mseLoss, ReLU, Sequential, smoothL1Loss } from "../../src/nn";
import { figure, plot, type RenderedPNG, type RenderedSVG, show } from "../../src/plot";

describe("1.5.0 typing ergonomics", () => {
  it("loss functions accept the AnyTensor returned by Module.forward", () => {
    const model = new Sequential(new Linear(2, 4), new ReLU(), new Linear(4, 1));
    const x = parameter([
      [1, 2],
      [3, 4],
    ]);
    const y = tensor([[1], [2]]);
    const pred = model.forward(x);
    expectTypeOf(pred).toEqualTypeOf<AnyTensor>();
    const loss = mseLoss(pred, y);
    expectTypeOf(loss).toEqualTypeOf<AnyTensor>();
    expect(GradTensor.isGradTensor(loss)).toBe(true);
    expectTypeOf(maeLoss(pred, y)).toEqualTypeOf<AnyTensor>();
    expectTypeOf(smoothL1Loss(pred, y, 1)).toEqualTypeOf<AnyTensor>();
    // Existing narrow overloads still win for concrete inputs.
    expectTypeOf(mseLoss(tensor([1]), tensor([1]))).toEqualTypeOf<Tensor>();
    expectTypeOf(
      mseLoss(
        x,
        tensor([
          [1, 2],
          [3, 4],
        ])
      )
    ).toEqualTypeOf<GradTensor>();
  });

  it("show() narrows its return type from the format option", () => {
    figure();
    plot(tensor([0, 1, 2]), tensor([0, 1, 4]));
    const svg = show({ format: "svg" });
    expectTypeOf(svg).toEqualTypeOf<RenderedSVG>();
    expect(svg.svg).toContain("<svg");
    expectTypeOf(show()).toEqualTypeOf<RenderedSVG>();
    expectTypeOf(show({ format: "png" })).toEqualTypeOf<Promise<RenderedPNG>>();
  });
});
