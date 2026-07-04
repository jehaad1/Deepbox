import { afterEach, describe, expect, it } from "vitest";
import {
  DecisionTreeClassifier,
  DecisionTreeRegressor,
  get_output,
  getEstimatorTags,
  KMeans,
  LinearRegression,
  PCA,
  reset_output,
  set_output,
} from "../src/ml";

describe("set_output / get_output / reset_output", () => {
  afterEach(() => {
    reset_output();
  });

  it("defaults to 'default'", () => {
    expect(get_output()).toBe("default");
  });

  it("sets output to 'array'", () => {
    set_output("array");
    expect(get_output()).toBe("array");
  });

  it("sets output back to 'default'", () => {
    set_output("array");
    set_output("default");
    expect(get_output()).toBe("default");
  });

  it("reset_output restores default", () => {
    set_output("array");
    reset_output();
    expect(get_output()).toBe("default");
  });

  it("throws on invalid output type", () => {
    expect(() => set_output("invalid" as "default")).toThrow();
  });
});

describe("getEstimatorTags", () => {
  it("infers classifier tags from DecisionTreeClassifier", () => {
    const clf = new DecisionTreeClassifier();
    const tags = getEstimatorTags(clf);
    expect(tags.estimatorType).toBe("classifier");
    expect(tags.hasPredictProba).toBe(true);
    expect(tags.requiresY).toBe(true);
  });

  it("infers regressor tags from DecisionTreeRegressor", () => {
    const reg = new DecisionTreeRegressor();
    const tags = getEstimatorTags(reg);
    expect(tags.estimatorType).toBe("regressor");
    expect(tags.hasPredictProba).toBe(false);
    expect(tags.requiresY).toBe(true);
  });

  it("infers regressor tags from LinearRegression", () => {
    const lr = new LinearRegression();
    const tags = getEstimatorTags(lr);
    expect(tags.estimatorType).toBe("regressor");
    expect(tags.requiresY).toBe(true);
  });

  it("infers clusterer tags from KMeans", () => {
    const km = new KMeans({ nClusters: 2 });
    const tags = getEstimatorTags(km);
    expect(tags.estimatorType).toBe("clusterer");
    expect(tags.requiresY).toBe(false);
  });

  it("infers transformer tags from PCA", () => {
    const pca = new PCA({ nComponents: 2 });
    const tags = getEstimatorTags(pca);
    expect(tags.estimatorType).toBe("transformer");
    expect(tags.requiresY).toBe(false);
  });

  it("provides default values for non-inferred fields", () => {
    const clf = new DecisionTreeClassifier();
    const tags = getEstimatorTags(clf);
    expect(tags.multiOutput).toBe(false);
    expect(tags.requiresPositiveX).toBe(false);
    expect(tags.requiresPositiveY).toBe(false);
    expect(tags.supportsSparse).toBe(false);
    expect(tags.supportsSampleWeight).toBe(false);
    expect(tags.hasDecisionFunction).toBe(false);
  });
});
