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

const invalidSeeds = [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, 1.5];

describe("makeClassification - Comprehensive Tests", () => {
  it("should generate dataset with default parameters", () => {
    const [X, y] = makeClassification();
    expect(X.shape[0]).toBe(100);
    expect(X.shape[1]).toBe(20);
    expect(y.shape[0]).toBe(100);
  });

  it("should generate dataset with custom parameters", () => {
    const [X, y] = makeClassification({
      nSamples: 500,
      nFeatures: 30,
      nClasses: 5,
    });
    expect(X.shape[0]).toBe(500);
    expect(X.shape[1]).toBe(30);
    expect(y.shape[0]).toBe(500);
  });

  it("should use nInformative and nRedundant parameters", () => {
    const [X, _y] = makeClassification({
      nSamples: 100,
      nFeatures: 20,
      nInformative: 10,
      nRedundant: 5,
      nClasses: 3,
    });
    expect(X.shape[0]).toBe(100);
    expect(X.shape[1]).toBe(20);
  });

  it("should be deterministic with same seed", () => {
    const [X1, y1] = makeClassification({
      nSamples: 50,
      nFeatures: 10,
      randomState: 42,
    });
    const [X2, y2] = makeClassification({
      nSamples: 50,
      nFeatures: 10,
      randomState: 42,
    });
    const data1 = numRawData(X1.data);
    const data2 = numRawData(X2.data);
    expect(data1).toEqual(data2);
    expect(numRawData(y1.data)).toEqual(numRawData(y2.data));
  });

  it("should produce different results with different seeds", () => {
    const [X1, _y1] = makeClassification({
      nSamples: 50,
      nFeatures: 10,
      randomState: 42,
    });
    const [X2, _y2] = makeClassification({
      nSamples: 50,
      nFeatures: 10,
      randomState: 123,
    });
    const data1 = numRawData(X1.data);
    const data2 = numRawData(X2.data);
    expect(data1).not.toEqual(data2);
  });

  it("should generate all class labels", () => {
    const [, y] = makeClassification({
      nSamples: 1000,
      nClasses: 5,
      randomState: 1,
    });
    const labels = numRawData(y.data);
    const uniqueLabels = new Set(labels);
    expect(uniqueLabels.size).toBeGreaterThanOrEqual(3); // Should have most classes
  });

  it("should throw for invalid randomState", () => {
    for (const seed of invalidSeeds) {
      expect(() => makeClassification({ randomState: seed })).toThrow();
    }
  });

  it("should throw error if nInformative > nFeatures", () => {
    expect(() =>
      makeClassification({
        nFeatures: 10,
        nInformative: 15,
      })
    ).toThrow();
  });

  it("should throw error if nInformative + nRedundant > nFeatures", () => {
    expect(() =>
      makeClassification({
        nFeatures: 10,
        nInformative: 6,
        nRedundant: 5,
      })
    ).toThrow();
  });

  it("should throw error for invalid nSamples", () => {
    expect(() => makeClassification({ nSamples: 0 })).toThrow();
    expect(() => makeClassification({ nSamples: -1 })).toThrow();
    expect(() => makeClassification({ nSamples: 1.5 })).toThrow();
  });

  it("should throw error for invalid nFeatures", () => {
    expect(() => makeClassification({ nFeatures: 0 })).toThrow();
    expect(() => makeClassification({ nFeatures: -5 })).toThrow();
  });

  it("should throw error for invalid nClasses", () => {
    expect(() => makeClassification({ nClasses: 0 })).toThrow();
    expect(() => makeClassification({ nClasses: -1 })).toThrow();
  });

  it("should handle single sample", () => {
    const [X, y] = makeClassification({ nSamples: 1, nFeatures: 5 });
    expect(X.shape[0]).toBe(1);
    expect(y.shape[0]).toBe(1);
  });

  it("should handle single feature", () => {
    const [X, _y] = makeClassification({
      nSamples: 10,
      nFeatures: 1,
      nInformative: 1,
      nRedundant: 0,
    });
    expect(X.shape[1]).toBe(1);
  });

  it("should handle binary classification", () => {
    const [_X, y] = makeClassification({ nSamples: 100, nClasses: 2 });
    const labels = numRawData(y.data);
    const uniqueLabels = new Set(labels);
    expect(uniqueLabels.size).toBeLessThanOrEqual(2);
  });

  it("should handle large number of classes", () => {
    const [_X, y] = makeClassification({
      nSamples: 1000,
      nClasses: 20,
      randomState: 5,
    });
    expect(y.shape[0]).toBe(1000);
  });

  it("should generate features with reasonable variance", () => {
    const [X] = makeClassification({ nSamples: 1000, nFeatures: 10 });
    const data = numRawData(X.data);
    const mean = data.reduce((a, b) => a + b, 0) / data.length;
    const variance = data.reduce((a, b) => a + (b - mean) ** 2, 0) / data.length;
    expect(variance).toBeGreaterThan(0.5); // Should have reasonable spread
  });

  it("should handle nInformative = nFeatures", () => {
    const [X, _y] = makeClassification({
      nSamples: 100,
      nFeatures: 10,
      nInformative: 10,
      nRedundant: 0,
    });
    expect(X.shape[1]).toBe(10);
  });

  it("should handle nRedundant = 0", () => {
    const [X, _y] = makeClassification({
      nSamples: 100,
      nFeatures: 10,
      nInformative: 5,
      nRedundant: 0,
    });
    expect(X.shape[1]).toBe(10);
  });

  it("should handle nInformative = 0 (all noise)", () => {
    expect(() =>
      makeClassification({
        nSamples: 100,
        nFeatures: 10,
        nInformative: 0,
        nRedundant: 0,
      })
    ).toThrow(); // Should throw because nInformative must be positive
  });

  it("should not generate NaN or Infinity values", () => {
    const [X, y] = makeClassification({
      nSamples: 1000,
      nFeatures: 20,
      nInformative: 10,
      nRedundant: 5,
      randomState: 1,
    });
    const xData = numRawData(X.data);
    const yData = numRawData(y.data);

    expect(xData.every((v) => Number.isFinite(v))).toBe(true);
    expect(yData.every((v) => Number.isFinite(v))).toBe(true);
  });
});

describe("makeRegression - Comprehensive Tests", () => {
  it("should generate dataset with default parameters", () => {
    const [X, y] = makeRegression();
    expect(X.shape[0]).toBe(100);
    expect(X.shape[1]).toBe(100);
    expect(y.shape[0]).toBe(100);
  });

  it("should generate dataset with custom parameters", () => {
    const [X, y] = makeRegression({
      nSamples: 500,
      nFeatures: 50,
    });
    expect(X.shape[0]).toBe(500);
    expect(X.shape[1]).toBe(50);
    expect(y.shape[0]).toBe(500);
  });

  it("should be deterministic with same seed", () => {
    const [X1, y1] = makeRegression({
      nSamples: 50,
      nFeatures: 10,
      randomState: 42,
    });
    const [X2, y2] = makeRegression({
      nSamples: 50,
      nFeatures: 10,
      randomState: 42,
    });
    expect(numRawData(X1.data)).toEqual(numRawData(X2.data));
    expect(numRawData(y1.data)).toEqual(numRawData(y2.data));
  });

  it("should add noise when specified", () => {
    const [, y1] = makeRegression({
      nSamples: 1000,
      nFeatures: 10,
      noise: 0,
      randomState: 1,
    });
    const [, y2] = makeRegression({
      nSamples: 1000,
      nFeatures: 10,
      noise: 10,
      randomState: 1,
    });
    const data1 = numRawData(y1.data);
    const data2 = numRawData(y2.data);

    // With noise, values should be different
    let differences = 0;
    for (let i = 0; i < data1.length; i++) {
      if (Math.abs((data1[i] ?? 0) - (data2[i] ?? 0)) > 0.1) {
        differences++;
      }
    }
    expect(differences).toBeGreaterThan(900); // Most should be different
  });

  it("should throw for invalid randomState", () => {
    for (const seed of invalidSeeds) {
      expect(() => makeRegression({ randomState: seed })).toThrow();
    }
  });

  it("should throw error for negative noise", () => {
    expect(() => makeRegression({ noise: -1 })).toThrow();
  });

  it("should throw error for invalid nSamples", () => {
    expect(() => makeRegression({ nSamples: 0 })).toThrow();
    expect(() => makeRegression({ nSamples: -1 })).toThrow();
    expect(() => makeRegression({ nSamples: 2.5 })).toThrow();
  });

  it("should throw error for invalid nFeatures", () => {
    expect(() => makeRegression({ nFeatures: 0 })).toThrow();
    expect(() => makeRegression({ nFeatures: -5 })).toThrow();
  });

  it("should handle single sample", () => {
    const [X, y] = makeRegression({ nSamples: 1, nFeatures: 5 });
    expect(X.shape[0]).toBe(1);
    expect(y.shape[0]).toBe(1);
  });

  it("should handle single feature", () => {
    const [X, _y] = makeRegression({ nSamples: 10, nFeatures: 1 });
    expect(X.shape[1]).toBe(1);
  });

  it("should generate continuous targets", () => {
    const [, y] = makeRegression({ nSamples: 100, nFeatures: 10 });
    const data = numRawData(y.data);
    const uniqueValues = new Set(data);
    expect(uniqueValues.size).toBeGreaterThan(50); // Should be mostly unique
  });

  it("should handle large datasets", () => {
    const [X, _y] = makeRegression({ nSamples: 10000, nFeatures: 100 });
    expect(X.shape[0]).toBe(10000);
    expect(X.shape[1]).toBe(100);
  });

  it("should generate targets with reasonable range", () => {
    const [, y] = makeRegression({
      nSamples: 1000,
      nFeatures: 10,
      noise: 0,
    });
    const data = numRawData(y.data);
    const max = Math.max(...data);
    const min = Math.min(...data);
    expect(max - min).toBeGreaterThan(5); // Should have reasonable spread
  });

  it("should handle zero noise", () => {
    const [, y] = makeRegression({
      nSamples: 100,
      nFeatures: 10,
      noise: 0,
      randomState: 1,
    });
    expect(y.shape[0]).toBe(100);
  });

  it("should handle large noise", () => {
    const [, y] = makeRegression({
      nSamples: 100,
      nFeatures: 10,
      noise: 100,
      randomState: 1,
    });
    expect(y.shape[0]).toBe(100);
  });

  it("should throw for non-finite noise", () => {
    expect(() => makeRegression({ noise: Number.POSITIVE_INFINITY })).toThrow();
    expect(() => makeRegression({ noise: Number.NaN })).toThrow();
  });

  it("should not generate NaN or Infinity values", () => {
    const [X, y] = makeRegression({
      nSamples: 1000,
      nFeatures: 50,
      noise: 0.1,
      randomState: 1,
    });
    const xData = numRawData(X.data);
    const yData = numRawData(y.data);

    expect(xData.every((v) => Number.isFinite(v))).toBe(true);
    expect(yData.every((v) => Number.isFinite(v))).toBe(true);
  });
});

describe("makeBlobs - Comprehensive Tests", () => {
  it("should generate dataset with default parameters", () => {
    const [X, y] = makeBlobs();
    expect(X.shape[0]).toBe(100);
    expect(X.shape[1]).toBe(2);
    expect(y.shape[0]).toBe(100);
  });

  it("should generate dataset with custom parameters", () => {
    const [X, _y] = makeBlobs({
      nSamples: 300,
      nFeatures: 5,
      centers: 4,
    });
    expect(X.shape[0]).toBe(300);
    expect(X.shape[1]).toBe(5);
  });

  it("should be deterministic with same seed", () => {
    const [X1, y1] = makeBlobs({
      nSamples: 50,
      centers: 3,
      randomState: 42,
    });
    const [X2, y2] = makeBlobs({
      nSamples: 50,
      centers: 3,
      randomState: 42,
    });
    expect(numRawData(X1.data)).toEqual(numRawData(X2.data));
    expect(numRawData(y1.data)).toEqual(numRawData(y2.data));
  });

  it("should accept explicit center coordinates", () => {
    const centers = [
      [0, 0],
      [5, 5],
      [10, 0],
    ];
    const [X, _y] = makeBlobs({
      nSamples: 90,
      nFeatures: 2,
      centers,
      clusterStd: 0.5,
    });
    expect(X.shape[0]).toBe(90);
    expect(X.shape[1]).toBe(2);
  });

  it("should distribute samples evenly across centers", () => {
    const [, y] = makeBlobs({
      nSamples: 300,
      centers: 3,
      randomState: 1,
    });
    const labels = numRawData(y.data);
    const counts = [0, 0, 0];
    for (const label of labels) {
      counts[label]++;
    }
    // Each center should have roughly 100 samples
    for (const count of counts) {
      expect(count).toBeGreaterThan(90);
      expect(count).toBeLessThan(110);
    }
  });

  it("should infer nFeatures from centers when not provided", () => {
    const [X, y] = makeBlobs({
      nSamples: 12,
      centers: [
        [0, 0, 0],
        [1, 1, 1],
      ],
      randomState: 7,
    });
    expect(X.shape).toEqual([12, 3]);
    expect(y.shape[0]).toBe(12);
  });

  it("should respect shuffle=false ordering", () => {
    const [, y] = makeBlobs({
      nSamples: 12,
      centers: 3,
      shuffle: false,
      randomState: 1,
    });
    const labels = numRawData(y.data);
    expect(labels.slice(0, 4)).toEqual([0, 0, 0, 0]);
    expect(labels.slice(4, 8)).toEqual([1, 1, 1, 1]);
    expect(labels.slice(8, 12)).toEqual([2, 2, 2, 2]);
  });

  it("should throw error for empty centers array", () => {
    expect(() =>
      makeBlobs({
        nSamples: 100,
        centers: [],
      })
    ).toThrow();
  });

  it("should throw error for invalid centers number", () => {
    expect(() => makeBlobs({ centers: 0 })).toThrow();
    expect(() => makeBlobs({ centers: -1 })).toThrow();
    expect(() => makeBlobs({ centers: 2.5 })).toThrow();
    expect(() => makeBlobs({ centers: Number.POSITIVE_INFINITY })).toThrow();
  });

  it("should throw error for mismatched center dimensions", () => {
    expect(() =>
      makeBlobs({
        nSamples: 100,
        nFeatures: 3,
        centers: [
          [0, 0],
          [1, 1],
        ],
      })
    ).toThrow();
  });

  it("should throw error for invalid clusterStd", () => {
    expect(() => makeBlobs({ clusterStd: 0 })).toThrow();
    expect(() => makeBlobs({ clusterStd: -1 })).toThrow();
  });

  it("should throw error for invalid nSamples", () => {
    expect(() => makeBlobs({ nSamples: 0 })).toThrow();
    expect(() => makeBlobs({ nSamples: -1 })).toThrow();
  });

  it("should handle single center", () => {
    const [_X, y] = makeBlobs({ nSamples: 100, centers: 1 });
    const labels = numRawData(y.data);
    const uniqueLabels = new Set(labels);
    expect(uniqueLabels.size).toBe(1);
  });

  it("should handle many centers", () => {
    const [_X, y] = makeBlobs({
      nSamples: 1000,
      centers: 20,
      randomState: 1,
    });
    expect(y.shape[0]).toBe(1000);
  });

  it("should respect clusterStd parameter", () => {
    // Use center at origin to avoid center location affecting variance
    const [X1] = makeBlobs({
      nSamples: 1000,
      centers: [[0, 0]],
      clusterStd: 1.0,
      randomState: 42,
    });
    const [X2] = makeBlobs({
      nSamples: 1000,
      centers: [[0, 0]],
      clusterStd: 10.0,
      randomState: 43,
    });

    const data1 = numRawData(X1.data);
    const data2 = numRawData(X2.data);

    // Calculate variance properly: Var(X) = E[(X - mean)^2]
    const mean1 = data1.reduce((a, b) => a + b, 0) / data1.length;
    const mean2 = data2.reduce((a, b) => a + b, 0) / data2.length;
    const variance1 = data1.reduce((a, b) => a + (b - mean1) ** 2, 0) / data1.length;
    const variance2 = data2.reduce((a, b) => a + (b - mean2) ** 2, 0) / data2.length;

    // Variance should scale with square of clusterStd (10^2 / 1^2 = 100x)
    // Use conservative 50x threshold to account for sampling variance
    expect(variance2).toBeGreaterThan(variance1 * 50);
  });

  it("should handle high-dimensional features", () => {
    const [X, _y] = makeBlobs({
      nSamples: 100,
      nFeatures: 50,
      centers: 3,
    });
    expect(X.shape[1]).toBe(50);
  });

  it("should handle uneven sample distribution", () => {
    const [, y] = makeBlobs({
      nSamples: 101,
      centers: 3,
      randomState: 1,
    });
    expect(y.shape[0]).toBe(101);
  });

  it("should throw for invalid randomState", () => {
    for (const seed of invalidSeeds) {
      expect(() => makeBlobs({ randomState: seed })).toThrow();
    }
  });

  it("should not generate NaN or Infinity values", () => {
    const [X, y] = makeBlobs({
      nSamples: 1000,
      nFeatures: 10,
      centers: 5,
      clusterStd: 1.0,
      randomState: 1,
    });
    const xData = numRawData(X.data);
    const yData = numRawData(y.data);

    expect(xData.every((v) => Number.isFinite(v))).toBe(true);
    expect(yData.every((v) => Number.isFinite(v))).toBe(true);
  });
});

describe("makeMoons - Comprehensive Tests", () => {
  it("should generate dataset with default parameters", () => {
    const [X, y] = makeMoons();
    expect(X.shape[0]).toBe(100);
    expect(X.shape[1]).toBe(2);
    expect(y.shape[0]).toBe(100);
  });

  it("should generate dataset with custom parameters", () => {
    const [X, _y] = makeMoons({ nSamples: 200, noise: 0.1 });
    expect(X.shape[0]).toBe(200);
    expect(X.shape[1]).toBe(2);
  });

  it("should be deterministic with same seed", () => {
    const [X1, y1] = makeMoons({
      nSamples: 50,
      noise: 0.1,
      randomState: 42,
    });
    const [X2, y2] = makeMoons({
      nSamples: 50,
      noise: 0.1,
      randomState: 42,
    });
    expect(numRawData(X1.data)).toEqual(numRawData(X2.data));
    expect(numRawData(y1.data)).toEqual(numRawData(y2.data));
  });

  it("should generate binary labels", () => {
    const [, y] = makeMoons({ nSamples: 100 });
    const labels = numRawData(y.data);
    const uniqueLabels = new Set(labels);
    expect(uniqueLabels.size).toBe(2);
    expect(uniqueLabels.has(0)).toBe(true);
    expect(uniqueLabels.has(1)).toBe(true);
  });

  it("should split samples evenly between moons", () => {
    const [, y] = makeMoons({ nSamples: 100 });
    const labels = numRawData(y.data);
    const count0 = labels.filter((l) => l === 0).length;
    const count1 = labels.filter((l) => l === 1).length;
    expect(count0).toBe(50);
    expect(count1).toBe(50);
  });

  it("should respect shuffle=false ordering", () => {
    const [, y] = makeMoons({ nSamples: 4, shuffle: false, randomState: 3 });
    const labels = numRawData(y.data);
    expect(labels).toEqual([0, 0, 1, 1]);
  });

  it("should handle odd number of samples", () => {
    const [X, y] = makeMoons({ nSamples: 101 });
    expect(X.shape[0]).toBe(101);
    const labels = numRawData(y.data);
    const count0 = labels.filter((l) => l === 0).length;
    const count1 = labels.filter((l) => l === 1).length;
    expect(count0 + count1).toBe(101);
  });

  it("should throw error for negative noise", () => {
    expect(() => makeMoons({ noise: -1 })).toThrow();
  });

  it("should throw error for invalid nSamples", () => {
    expect(() => makeMoons({ nSamples: 0 })).toThrow();
    expect(() => makeMoons({ nSamples: -1 })).toThrow();
  });

  it("should handle zero noise", () => {
    const [X, _y] = makeMoons({ nSamples: 100, noise: 0 });
    expect(X.shape[0]).toBe(100);
  });

  it("should handle large noise", () => {
    const [X, _y] = makeMoons({ nSamples: 100, noise: 5 });
    expect(X.shape[0]).toBe(100);
  });

  it("should generate points in expected range", () => {
    const [X] = makeMoons({ nSamples: 1000, noise: 0 });
    const data = numRawData(X.data);
    // Without noise, points should be roughly in [-1, 2] x [-0.5, 1.5]
    for (let i = 0; i < data.length; i += 2) {
      const x = data[i] ?? 0;
      const y = data[i + 1] ?? 0;
      expect(x).toBeGreaterThan(-1.5);
      expect(x).toBeLessThan(2.5);
      expect(y).toBeGreaterThan(-1);
      expect(y).toBeLessThan(2);
    }
  });

  it("should handle single sample", () => {
    const [X, y] = makeMoons({ nSamples: 1 });
    expect(X.shape[0]).toBe(1);
    expect(y.shape[0]).toBe(1);
  });

  it("should throw for non-finite noise", () => {
    expect(() => makeMoons({ noise: Number.POSITIVE_INFINITY })).toThrow();
    expect(() => makeMoons({ noise: Number.NaN })).toThrow();
  });

  it("should throw for invalid randomState", () => {
    for (const seed of invalidSeeds) {
      expect(() => makeMoons({ randomState: seed })).toThrow();
    }
  });

  it("should not generate NaN or Infinity values", () => {
    const [X, y] = makeMoons({
      nSamples: 1000,
      noise: 0.1,
      randomState: 1,
    });
    const xData = numRawData(X.data);
    const yData = numRawData(y.data);

    expect(xData.every((v) => Number.isFinite(v))).toBe(true);
    expect(yData.every((v) => Number.isFinite(v))).toBe(true);
  });
});

describe("makeCircles - Comprehensive Tests", () => {
  it("should generate dataset with default parameters", () => {
    const [X, y] = makeCircles();
    expect(X.shape[0]).toBe(100);
    expect(X.shape[1]).toBe(2);
    expect(y.shape[0]).toBe(100);
  });

  it("should generate dataset with custom parameters", () => {
    const [X, _y] = makeCircles({
      nSamples: 200,
      noise: 0.05,
      factor: 0.5,
    });
    expect(X.shape[0]).toBe(200);
    expect(X.shape[1]).toBe(2);
  });

  it("should be deterministic with same seed", () => {
    const [X1, y1] = makeCircles({
      nSamples: 50,
      factor: 0.5,
      randomState: 42,
    });
    const [X2, y2] = makeCircles({
      nSamples: 50,
      factor: 0.5,
      randomState: 42,
    });
    expect(numRawData(X1.data)).toEqual(numRawData(X2.data));
    expect(numRawData(y1.data)).toEqual(numRawData(y2.data));
  });

  it("should generate binary labels", () => {
    const [, y] = makeCircles({ nSamples: 100 });
    const labels = numRawData(y.data);
    const uniqueLabels = new Set(labels);
    expect(uniqueLabels.size).toBe(2);
  });

  it("should split samples evenly between circles", () => {
    const [, y] = makeCircles({ nSamples: 100 });
    const labels = numRawData(y.data);
    const count0 = labels.filter((l) => l === 0).length;
    const count1 = labels.filter((l) => l === 1).length;
    expect(count0).toBe(50);
    expect(count1).toBe(50);
  });

  it("should respect shuffle=false ordering", () => {
    const [, y] = makeCircles({ nSamples: 4, shuffle: false, randomState: 3 });
    const labels = numRawData(y.data);
    expect(labels).toEqual([0, 0, 1, 1]);
  });

  it("should throw error for invalid factor", () => {
    expect(() => makeCircles({ factor: 0 })).toThrow();
    expect(() => makeCircles({ factor: 1 })).toThrow();
    expect(() => makeCircles({ factor: -0.5 })).toThrow();
    expect(() => makeCircles({ factor: 1.5 })).toThrow();
  });

  it("should throw error for negative noise", () => {
    expect(() => makeCircles({ noise: -1 })).toThrow();
  });

  it("should throw error for invalid nSamples", () => {
    expect(() => makeCircles({ nSamples: 0 })).toThrow();
    expect(() => makeCircles({ nSamples: -1 })).toThrow();
  });

  it("should handle factor close to 0", () => {
    const [X, _y] = makeCircles({
      nSamples: 100,
      factor: 0.1,
      noise: 0,
    });
    expect(X.shape[0]).toBe(100);
  });

  it("should handle factor close to 1", () => {
    const [X, _y] = makeCircles({
      nSamples: 100,
      factor: 0.99,
      noise: 0,
    });
    expect(X.shape[0]).toBe(100);
  });

  it("should respect factor parameter", () => {
    const [X] = makeCircles({
      nSamples: 1000,
      factor: 0.5,
      noise: 0,
      randomState: 1,
    });
    const data = numRawData(X.data);

    // Calculate distances from origin
    const distances: number[] = [];
    for (let i = 0; i < data.length; i += 2) {
      const x = data[i] ?? 0;
      const y = data[i + 1] ?? 0;
      distances.push(Math.sqrt(x * x + y * y));
    }

    // Should have two clusters of distances: ~0.5 and ~1.0
    const innerCount = distances.filter((d) => d < 0.75).length;
    const outerCount = distances.filter((d) => d >= 0.75).length;
    expect(innerCount).toBeGreaterThan(400);
    expect(outerCount).toBeGreaterThan(400);
  });

  it("should handle odd number of samples", () => {
    const [X, _y] = makeCircles({ nSamples: 101 });
    expect(X.shape[0]).toBe(101);
  });

  it("should handle single sample", () => {
    const [X, _y] = makeCircles({ nSamples: 1 });
    expect(X.shape[0]).toBe(1);
  });

  it("should not generate NaN or Infinity values", () => {
    const [X, y] = makeCircles({
      nSamples: 1000,
      noise: 0.05,
      factor: 0.5,
      randomState: 1,
    });
    const xData = numRawData(X.data);
    const yData = numRawData(y.data);

    expect(xData.every((v) => Number.isFinite(v))).toBe(true);
    expect(yData.every((v) => Number.isFinite(v))).toBe(true);
  });

  it("should throw for invalid randomState", () => {
    for (const seed of invalidSeeds) {
      expect(() => makeCircles({ randomState: seed })).toThrow();
    }
  });
});

describe("makeGaussianQuantiles - Comprehensive Tests", () => {
  it("should generate dataset with default parameters", () => {
    const [X, y] = makeGaussianQuantiles();
    expect(X.shape[0]).toBe(100);
    expect(X.shape[1]).toBe(2);
    expect(y.shape[0]).toBe(100);
  });

  it("should generate dataset with custom parameters", () => {
    const [X, _y] = makeGaussianQuantiles({
      nSamples: 500,
      nFeatures: 5,
      nClasses: 4,
    });
    expect(X.shape[0]).toBe(500);
    expect(X.shape[1]).toBe(5);
  });

  it("should be deterministic with same seed", () => {
    const [X1, y1] = makeGaussianQuantiles({
      nSamples: 50,
      nClasses: 3,
      randomState: 42,
    });
    const [X2, y2] = makeGaussianQuantiles({
      nSamples: 50,
      nClasses: 3,
      randomState: 42,
    });
    expect(numRawData(X1.data)).toEqual(numRawData(X2.data));
    expect(numRawData(y1.data)).toEqual(numRawData(y2.data));
  });

  it("should generate all class labels", () => {
    const [, y] = makeGaussianQuantiles({
      nSamples: 1000,
      nClasses: 5,
      randomState: 1,
    });
    const labels = numRawData(y.data);
    const uniqueLabels = new Set(labels);
    expect(uniqueLabels.size).toBe(5);
  });

  it("should distribute samples roughly evenly across classes", () => {
    const [, y] = makeGaussianQuantiles({
      nSamples: 1000,
      nClasses: 4,
      randomState: 1,
    });
    const labels = numRawData(y.data);
    const counts = [0, 0, 0, 0];
    for (const label of labels) {
      counts[label]++;
    }
    // Each class should have roughly 250 samples
    for (const count of counts) {
      expect(count).toBeGreaterThan(200);
      expect(count).toBeLessThan(300);
    }
  });

  it("should throw error for invalid nSamples", () => {
    expect(() => makeGaussianQuantiles({ nSamples: 0 })).toThrow();
    expect(() => makeGaussianQuantiles({ nSamples: -1 })).toThrow();
  });

  it("should throw error for invalid nFeatures", () => {
    expect(() => makeGaussianQuantiles({ nFeatures: 0 })).toThrow();
  });

  it("should throw error for invalid nClasses", () => {
    expect(() => makeGaussianQuantiles({ nClasses: 0 })).toThrow();
  });

  it("should handle single class", () => {
    const [_X, y] = makeGaussianQuantiles({
      nSamples: 100,
      nClasses: 1,
    });
    const labels = numRawData(y.data);
    const uniqueLabels = new Set(labels);
    expect(uniqueLabels.size).toBe(1);
  });

  it("should handle many classes", () => {
    const [_X, y] = makeGaussianQuantiles({
      nSamples: 1000,
      nClasses: 20,
      randomState: 1,
    });
    expect(y.shape[0]).toBe(1000);
  });

  it("should handle high-dimensional features", () => {
    const [X, _y] = makeGaussianQuantiles({
      nSamples: 100,
      nFeatures: 50,
      nClasses: 3,
    });
    expect(X.shape[1]).toBe(50);
  });

  it("should handle nSamples < nClasses", () => {
    const [_X, y] = makeGaussianQuantiles({
      nSamples: 5,
      nClasses: 10,
      randomState: 1,
    });
    expect(y.shape[0]).toBe(5);
    // Not all classes will be present
    const labels = numRawData(y.data);
    const uniqueLabels = new Set(labels);
    expect(uniqueLabels.size).toBeLessThanOrEqual(5);
  });

  it("should assign classes based on distance from origin", () => {
    const [X, y] = makeGaussianQuantiles({
      nSamples: 1000,
      nFeatures: 2,
      nClasses: 3,
      randomState: 1,
    });
    const data = numRawData(X.data);
    const labels = numRawData(y.data);

    // Calculate average distance for each class
    const classDists: { [key: number]: number[] } = { 0: [], 1: [], 2: [] };
    for (let i = 0; i < labels.length; i++) {
      const x = data[i * 2] ?? 0;
      const y = data[i * 2 + 1] ?? 0;
      const dist = Math.sqrt(x * x + y * y);
      const label = labels[i] ?? 0;
      classDists[label]?.push(dist);
    }

    const avgDist0 =
      (classDists[0]?.reduce((a, b) => a + b, 0) ?? 0) / (classDists[0]?.length ?? 1);
    const avgDist1 =
      (classDists[1]?.reduce((a, b) => a + b, 0) ?? 0) / (classDists[1]?.length ?? 1);
    const avgDist2 =
      (classDists[2]?.reduce((a, b) => a + b, 0) ?? 0) / (classDists[2]?.length ?? 1);

    // Classes should be ordered by distance
    expect(avgDist0).toBeLessThan(avgDist1);
    expect(avgDist1).toBeLessThan(avgDist2);
  });

  it("should throw for invalid randomState", () => {
    for (const seed of invalidSeeds) {
      expect(() => makeGaussianQuantiles({ randomState: seed })).toThrow();
    }
  });

  it("should not generate NaN or Infinity values", () => {
    const [X, y] = makeGaussianQuantiles({
      nSamples: 1000,
      nFeatures: 10,
      nClasses: 5,
      randomState: 1,
    });
    const xData = numRawData(X.data);
    const yData = numRawData(y.data);

    expect(xData.every((v) => Number.isFinite(v))).toBe(true);
    expect(yData.every((v) => Number.isFinite(v))).toBe(true);
  });
});
