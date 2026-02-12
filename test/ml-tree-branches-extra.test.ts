import { describe, expect, it } from "vitest";
import { DecisionTreeClassifier, DecisionTreeRegressor } from "../src/ml/tree";
import { tensor } from "../src/ndarray";

describe("ml tree extra branches", () => {
  it("DecisionTreeRegressor predict errors before fit and on shape mismatch", () => {
    const reg = new DecisionTreeRegressor();
    expect(() => reg.predict(tensor([[1, 2]]))).toThrow(/fitted/i);

    reg.fit(
      tensor([
        [1, 2],
        [2, 3],
      ]),
      tensor([1, 2])
    );
    expect(() => reg.predict(tensor([1, 2]))).toThrow(/2-dimensional/);
    expect(() => reg.predict(tensor([[1, 2, 3]]))).toThrow(/features/);
  });

  it("DecisionTreeClassifier predict errors before fit and on shape mismatch", () => {
    const clf = new DecisionTreeClassifier();
    expect(() => clf.predict(tensor([[1, 2]]))).toThrow(/fitted/i);

    clf.fit(
      tensor([
        [1, 2],
        [2, 3],
      ]),
      tensor([0, 1])
    );
    expect(() => clf.predict(tensor([1, 2]))).toThrow(/2-dimensional/);
    expect(() => clf.predict(tensor([[1, 2, 3]]))).toThrow(/features/);
  });
});
