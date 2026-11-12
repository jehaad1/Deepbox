import { describe, expect, it } from "vitest";
import { ShapeError } from "../../src/core";
import { dot, parameter, tensor } from "../../src/ndarray";
import { matmul } from "../../src/ndarray/linalg/basic";

// Reference: numpy.matmul (numpy 2.x).

const f64 = { dtype: "float64" } as const;

describe("matmul follows numpy.matmul", () => {
  it("promotes a 1-D left operand: [1, 2] @ [[1, 2], [3, 4]] = [7, 10]", () => {
    const out = matmul(
      tensor([1, 2], f64),
      tensor(
        [
          [1, 2],
          [3, 4],
        ],
        f64
      )
    );
    expect(out.shape).toEqual([2]);
    expect(out.toArray()).toEqual([7, 10]);
  });

  it("promotes a 1-D right operand: [[1, 2], [3, 4]] @ [1, 2] = [5, 11]", () => {
    const out = matmul(
      tensor(
        [
          [1, 2],
          [3, 4],
        ],
        f64
      ),
      tensor([1, 2], f64)
    );
    expect(out.shape).toEqual([2]);
    expect(out.toArray()).toEqual([5, 11]);
  });

  it("multiplies two 1-D vectors into a 0-d tensor", () => {
    const out = matmul(tensor([1, 2], f64), tensor([3, 4], f64));
    expect(out.shape).toEqual([]);
    expect(out.item()).toBe(11);
  });

  it("broadcasts a 2-D right operand over a batch", () => {
    const a = tensor(
      [
        [
          [0, 1],
          [2, 3],
        ],
        [
          [4, 5],
          [6, 7],
        ],
      ],
      f64
    );
    const out = matmul(
      a,
      tensor(
        [
          [1, 2],
          [3, 4],
        ],
        f64
      )
    );
    expect(out.toArray()).toEqual([
      [
        [3, 4],
        [11, 16],
      ],
      [
        [19, 28],
        [27, 40],
      ],
    ]);
  });

  it("multiplies a batch by a 1-D vector", () => {
    const a = tensor(
      [
        [
          [0, 1],
          [2, 3],
        ],
        [
          [4, 5],
          [6, 7],
        ],
      ],
      f64
    );
    const out = matmul(a, tensor([0, 1], f64));
    expect(out.toArray()).toEqual([
      [1, 3],
      [5, 7],
    ]);
  });

  it("broadcasts batch dimensions of size 1", () => {
    const a = tensor(
      [
        [
          [0, 1, 2],
          [3, 4, 5],
        ],
      ],
      f64
    );
    const b = tensor(
      [
        [
          [0, 1],
          [2, 3],
          [4, 5],
        ],
        [
          [6, 7],
          [8, 9],
          [10, 11],
        ],
      ],
      f64
    );
    const out = matmul(a, b);
    expect(out.shape).toEqual([2, 2, 2]);
    expect(out.toArray()).toEqual([
      [
        [10, 13],
        [28, 40],
      ],
      [
        [28, 31],
        [100, 112],
      ],
    ]);
  });

  it("agrees with dot on batched operands and through Tensor.matmul", () => {
    const a = tensor(
      [
        [
          [0, 1],
          [2, 3],
        ],
        [
          [4, 5],
          [6, 7],
        ],
      ],
      f64
    );
    const b = tensor(
      [
        [
          [1, 0],
          [0, 1],
        ],
        [
          [2, 0],
          [0, 2],
        ],
      ],
      f64
    );
    expect(matmul(a, b).toArray()).toEqual(dot(a, b).toArray());
    expect(a.matmul(b).toArray()).toEqual(dot(a, b).toArray());
  });

  it("rejects 0-d operands and mismatched shapes", () => {
    expect(() => matmul(tensor(2, f64), tensor([1, 2], f64))).toThrow(ShapeError);
    expect(() =>
      matmul(
        tensor([[1, 2]], f64),
        tensor(
          [
            [1, 2],
            [3, 4],
            [5, 6],
          ],
          f64
        )
      )
    ).toThrow(/mismatch/);
    expect(() => matmul(tensor([[[1, 2, 3]]], f64), tensor([[[1]], [[2]], [[3]]], f64))).toThrow(
      ShapeError
    );
  });

  it("keeps the GradTensor matmul gradient working for batched operands", () => {
    const a = parameter([
      [
        [1, 2],
        [3, 4],
      ],
    ]);
    const b = parameter([
      [5, 6],
      [7, 8],
    ]);
    const loss = a.matmul(b).sum();
    loss.backward();
    expect(a.grad?.shape).toEqual([1, 2, 2]);
    expect(b.grad?.shape).toEqual([2, 2]);
  });
});
