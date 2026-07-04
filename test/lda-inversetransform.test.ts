import { describe, expect, it } from "vitest";
import { LatentDirichletAllocation } from "../src/ml/decomposition";
import { tensor } from "../src/ndarray";

describe("LatentDirichletAllocation.inverseTransform()", () => {
  it("reconstructs approximate word distributions from topic distributions", () => {
    // Simple document-term matrix (non-negative)
    const X = tensor([
      [1, 2, 0, 1],
      [0, 1, 3, 0],
      [2, 0, 1, 2],
      [1, 1, 1, 1],
    ]);

    const lda = new LatentDirichletAllocation({
      nComponents: 2,
      maxIter: 20,
      randomState: 42,
    });
    lda.fit(X);

    const transformed = lda.transform(X);
    expect(transformed.shape).toEqual([4, 2]);

    const reconstructed = lda.inverseTransform(transformed);
    expect(reconstructed.shape).toEqual([4, 4]);

    // Reconstructed values should be non-negative (they are probabilities weighted by topic weights)
    for (let i = 0; i < reconstructed.size; i++) {
      expect(Number(reconstructed.data[i])).toBeGreaterThanOrEqual(0);
    }
  });

  it("throws NotFittedError if not fitted", () => {
    const lda = new LatentDirichletAllocation({ nComponents: 2 });
    expect(() => lda.inverseTransform(tensor([[0.5, 0.5]]))).toThrow();
  });

  it("round-trips produce correct shapes", () => {
    const X = tensor([
      [3, 1, 0],
      [0, 2, 3],
      [1, 1, 1],
    ]);

    const lda = new LatentDirichletAllocation({
      nComponents: 2,
      maxIter: 10,
      randomState: 123,
    });

    const Xt = lda.fitTransform(X);
    expect(Xt.shape).toEqual([3, 2]);

    const Xr = lda.inverseTransform(Xt);
    expect(Xr.shape).toEqual([3, 3]);
  });
});
