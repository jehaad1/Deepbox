import { describe, expect, it } from "vitest";
import { tensor } from "../../src/ndarray";
import { Binarizer } from "../../src/preprocess";

// scikit-learn 1.9: Binarizer().fit_transform(np.float32 array).dtype == float32.
describe("PP-8 Binarizer output dtype", () => {
  it("keeps float32 input as float32", () => {
    const out = new Binarizer({ threshold: 1.5 }).fitTransform(
      tensor(
        [
          [1, 2],
          [3, 0.5],
        ],
        { dtype: "float32" }
      )
    );
    expect(out.dtype).toBe("float32");
    expect(out.toArray()).toEqual([
      [0, 1],
      [1, 0],
    ]);
  });

  it("returns float64 for float64 and integer input", () => {
    expect(new Binarizer().fitTransform(tensor([[1, -1]], { dtype: "float64" })).dtype).toBe(
      "float64"
    );
    expect(new Binarizer().fitTransform(tensor([[1, 0]], { dtype: "int32" })).dtype).toBe(
      "float64"
    );
  });
});
