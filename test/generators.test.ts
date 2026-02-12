import { describe, expect, it } from "vitest";
import {
  makeBlobs,
  makeCircles,
  makeClassification,
  makeGaussianQuantiles,
  makeMoons,
  makeRegression,
} from "../src/datasets";
import { numRawData } from "./_helpers";

describe("Dataset Generators", () => {
  it("should generate classification dataset", () => {
    const [X, y] = makeClassification({
      nSamples: 100,
      nFeatures: 20,
      nClasses: 2,
    });
    expect(X).toBeDefined();
    expect(y).toBeDefined();
    expect(X.shape[0]).toBe(100);
    expect(X.shape[1]).toBe(20);
    expect(y.shape[0]).toBe(100);
  });

  it("should be deterministic when randomState is provided", () => {
    const [X1, y1] = makeClassification({
      nSamples: 10,
      nFeatures: 5,
      nInformative: 2,
      nRedundant: 2,
      nClasses: 2,
      randomState: 42,
    });
    const [X2, y2] = makeClassification({
      nSamples: 10,
      nFeatures: 5,
      nInformative: 2,
      nRedundant: 2,
      nClasses: 2,
      randomState: 42,
    });
    expect(numRawData(X1.data).slice(0, 5)).toEqual(numRawData(X2.data).slice(0, 5));
    expect(numRawData(y1.data).slice(0, 5)).toEqual(numRawData(y2.data).slice(0, 5));
  });

  it("should generate regression dataset", () => {
    const [X, y] = makeRegression({ nSamples: 100, nFeatures: 10 });
    expect(X).toBeDefined();
    expect(y).toBeDefined();
    expect(X.shape[0]).toBe(100);
    expect(X.shape[1]).toBe(10);
  });

  it("should generate blobs", () => {
    const [X, y] = makeBlobs({ nSamples: 300, centers: 3 });
    expect(X).toBeDefined();
    expect(y).toBeDefined();
    expect(X.shape[0]).toBe(300);
    expect(y.shape[0]).toBe(300);
  });

  it("should support explicit centers array", () => {
    const [X, y] = makeBlobs({
      nSamples: 5,
      nFeatures: 2,
      centers: [
        [0, 0],
        [10, 10],
      ],
      randomState: 1,
    });
    expect(X.shape[0]).toBe(5);
    expect(X.shape[1]).toBe(2);
    expect(y.shape[0]).toBe(5);
  });

  it("should generate moons", () => {
    const [X, y] = makeMoons({ nSamples: 201, noise: 0.1, randomState: 0 });
    expect(X).toBeDefined();
    expect(y).toBeDefined();
    expect(X.shape[0]).toBe(201);
    expect(y.shape[0]).toBe(201);
  });

  it("should generate circles", () => {
    const [X, y] = makeCircles({ nSamples: 201, factor: 0.5, randomState: 0 });
    expect(X).toBeDefined();
    expect(y).toBeDefined();
    expect(X.shape[0]).toBe(201);
    expect(y.shape[0]).toBe(201);
  });

  it("should generate Gaussian quantiles", () => {
    const [X, y] = makeGaussianQuantiles({
      nSamples: 100,
      nFeatures: 2,
      nClasses: 3,
    });
    expect(X).toBeDefined();
    expect(y).toBeDefined();
    expect(X.shape[0]).toBe(100);
    expect(y.shape[0]).toBe(100);
  });

  it("should handle nSamples < nClasses in Gaussian quantiles", () => {
    const [X, y] = makeGaussianQuantiles({
      nSamples: 2,
      nFeatures: 2,
      nClasses: 3,
      randomState: 1,
    });
    expect(X.shape[0]).toBe(2);
    expect(y.shape[0]).toBe(2);
  });
});
