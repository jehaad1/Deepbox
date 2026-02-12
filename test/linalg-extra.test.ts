import { describe, expect, it } from "vitest";
import { InvalidParameterError } from "../src/core";
import { eig } from "../src/linalg/decomposition/eig";
import { norm } from "../src/linalg/norms";
import { tensor } from "../src/ndarray";

describe("Linalg extra tests", () => {
  it("should throw InvalidParameterError for matrix with complex eigenvalues", () => {
    const a = tensor([
      [0, -1],
      [1, 0],
    ]);
    expect(() => eig(a)).toThrow(InvalidParameterError);
  });

  it("should calculate matrix norm correctly for axis=[0,1]", () => {
    const a = tensor([
      [1, 2],
      [3, 4],
    ]);
    const result = norm(a, 1, [0, 1]);
    expect(result).toBe(6);
  });

  it("should throw InvalidParameterError for invalid matrix norm order", () => {
    const a = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(() => norm(a, 3)).toThrow(InvalidParameterError);
  });
});
