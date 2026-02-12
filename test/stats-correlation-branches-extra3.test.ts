import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { corrcoef, cov } from "../src/stats/correlation";
import { numData } from "./_helpers";

describe("stats correlation extra branches", () => {
  it("corrcoef handles y provided and invalid shapes", () => {
    const x = tensor([1, 2, 3]);
    const y = tensor([1, 2, 4]);
    const out = numData(corrcoef(x, y));
    expect(out.length).toBe(4);
    expect(out[0]).toBe(1);

    expect(() => corrcoef(tensor([[1, 2]]), undefined)).toThrow(/at least 2 observations/);
    expect(() => corrcoef(tensor([[[1]]]))).toThrow(/1D or 2D/);
    expect(() => corrcoef(tensor([[1, 2]]))).toThrow(/at least 2 observations/);
  });

  it("corrcoef returns NaN for zero-variance columns", () => {
    const data = tensor([
      [1, 1],
      [1, 1],
      [2, 1],
    ]);
    const out = numData(corrcoef(data));
    expect(Number.isNaN(out[1] ?? 0)).toBe(true);
  });

  it("cov handles 1D/2D inputs and ddof validation", () => {
    expect(() => cov(tensor([1, 2]), undefined, -1)).toThrow(/ddof/);

    const oneD = cov(tensor([1, 2, 3]));
    expect(oneD.shape).toEqual([1, 1]);

    const twoD = cov(
      tensor([
        [1, 2],
        [3, 4],
        [5, 6],
      ])
    );
    expect(twoD.shape).toEqual([2, 2]);

    const x = tensor([1, 2, 3]);
    const y = tensor([2, 3, 4]);
    const xy = cov(x, y, 0);
    expect(xy.shape).toEqual([2, 2]);

    expect(() => cov(tensor([]), undefined, 1)).toThrow(/at least one element/);
    expect(() => cov(tensor([1, 2]), undefined, 2)).toThrow(/ddof/);
  });
});
