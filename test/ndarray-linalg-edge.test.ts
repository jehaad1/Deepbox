import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { dot, matmul } from "../src/ndarray/linalg/basic";

describe("deepbox/ndarray - Linalg Edge Cases", () => {
  it("handles int64 matmul and dot", () => {
    const a = tensor(
      [
        [1, 2],
        [3, 4],
      ],
      { dtype: "int64" }
    );
    const b = tensor(
      [
        [5, 6],
        [7, 8],
      ],
      { dtype: "int64" }
    );
    const c = matmul(a, b);
    expect(c.dtype).toBe("int64");
    expect(c.toArray()).toEqual([
      [19n, 22n],
      [43n, 50n],
    ]);

    const d = dot(tensor([1, 2], { dtype: "int64" }), tensor([3, 4], { dtype: "int64" }));
    expect(d.dtype).toBe("int64");
  });

  it("validates string dtype and shapes", () => {
    const s = tensor(["a", "b"]);
    expect(() => dot(s, s)).toThrow();
    const a = tensor([[1, 2]]);
    const b = tensor([[1, 2, 3]]);
    expect(() => matmul(a, b)).toThrow();
  });
});
