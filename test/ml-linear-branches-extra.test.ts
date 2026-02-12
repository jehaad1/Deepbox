import { describe, expect, it } from "vitest";
import { Lasso } from "../src/ml/linear/Lasso";
import { LinearRegression } from "../src/ml/linear/LinearRegression";
import { Ridge } from "../src/ml/linear/Ridge";
import { tensor } from "../src/ndarray";

describe("ml linear extra branches", () => {
  it("Ridge setParams validates types and unknowns", () => {
    const model = new Ridge();
    expect(() => model.setParams({ alpha: "bad" })).toThrow(/alpha must be/);
    expect(() => model.setParams({ fitIntercept: 1 })).toThrow(/fitIntercept must be a boolean/);
    expect(() => model.setParams({ solver: "oops" })).toThrow(/Invalid solver/);
    expect(() => model.setParams({ unknown: 1 })).toThrow(/Unknown parameter/);
  });

  it("Ridge getters throw before fitting", () => {
    const model = new Ridge();
    expect(() => model.coef).toThrow(/fitted/);
    expect(() => model.intercept).toThrow(/fitted/);
    expect(() => model.nIter).toThrow(/fitted/);
  });

  it("Lasso setParams validates and errors on unknown", () => {
    const model = new Lasso();
    expect(() => model.setParams({ alpha: "bad" })).toThrow(/alpha must be/);
    expect(() => model.setParams({ fitIntercept: 1 })).toThrow(/boolean/);
    expect(() => model.setParams({ unknown: 1 })).toThrow(/Unknown parameter/);
  });

  it("LinearRegression setParams validation errors", () => {
    const model = new LinearRegression();
    expect(() => model.setParams({ fitIntercept: 1 })).toThrow(/boolean/);
    expect(() => model.setParams({ normalize: "no" })).toThrow(/boolean/);
    expect(() => model.setParams({ unknown: 1 })).toThrow(/Unknown parameter/);
  });

  it("LinearRegression predict throws before fit", () => {
    const model = new LinearRegression();
    expect(() => model.predict(tensor([[1, 2]]))).toThrow(/fit/);
  });
});
