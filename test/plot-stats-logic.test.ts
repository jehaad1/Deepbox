import { describe, expect, it } from "vitest";
import {
  calculateQuartiles,
  calculateWhiskers,
  kernelDensityEstimation,
} from "../src/plot/utils/statistics";

describe("Plot Statistics Logic", () => {
  describe("calculateQuartiles", () => {
    it("should handle odd number of elements", () => {
      const data = [1, 2, 3, 4, 5];
      // Median = 3
      // Lower half = [1, 2] -> Median = 1.5
      // Upper half = [4, 5] -> Median = 4.5
      const { q1, median, q3 } = calculateQuartiles(data);
      expect(median).toBe(3);
      expect(q1).toBe(1.5);
      expect(q3).toBe(4.5);
    });

    it("should handle even number of elements", () => {
      const data = [1, 2, 3, 4];
      // Median = 2.5
      // Lower half = [1, 2] -> Median = 1.5
      // Upper half = [3, 4] -> Median = 3.5
      const { q1, median, q3 } = calculateQuartiles(data);
      expect(median).toBe(2.5);
      expect(q1).toBe(1.5);
      expect(q3).toBe(3.5);
    });

    it("should handle single element", () => {
      const { q1, median, q3 } = calculateQuartiles([10]);
      expect(median).toBe(10);
      expect(q1).toBe(10);
      expect(q3).toBe(10);
    });

    it("should handle empty array", () => {
      const { median } = calculateQuartiles([]);
      expect(median).toBe(0);
    });
  });

  describe("calculateWhiskers", () => {
    it("should identify outliers", () => {
      // Q1=2, Q3=8, IQR=6. Bounds: [-7, 17]
      // Let's create a clear outlier case
      // Data: [1, 2, 3, 4, 5, 100]
      // Median = 3.5
      // Lower = [1, 2, 3] -> Median = 2
      // Upper = [4, 5, 100] -> Median = 5
      // IQR = 3. Bounds: 2 - 4.5 = -2.5, 5 + 4.5 = 9.5
      // 100 is outlier
      const data = [1, 2, 3, 4, 5, 100];
      const { q1, q3 } = calculateQuartiles(data);
      const { outliers, upperWhisker } = calculateWhiskers(data, q1, q3);

      expect(outliers).toContain(100);
      expect(outliers.length).toBe(1);
      expect(upperWhisker).toBe(5);
    });

    it("should collapse whiskers to quartiles if all values are outliers", () => {
      // Data: [0, 100]. Q1=0, Q3=100. IQR=100. Bounds: [-150, 250].
      // Wait, that example had them inside.
      // Let's use manually calculated quartiles where data is outside.
      // Q1=10, Q3=11. IQR=1. Bounds: [8.5, 12.5].
      // Data: [0, 100]. Both are outliers.
      const data = [0, 100];
      const q1 = 10;
      const q3 = 11;
      const { outliers, lowerWhisker, upperWhisker } = calculateWhiskers(data, q1, q3);

      expect(outliers).toContain(0);
      expect(outliers).toContain(100);
      expect(outliers.length).toBe(2);
      // Whiskers should be at quartiles (length 0) instead of extending to outliers
      expect(lowerWhisker).toBe(q1);
      expect(upperWhisker).toBe(q3);
    });
  });

  describe("kernelDensityEstimation", () => {
    it("should produce a valid PDF (approximate)", () => {
      const data = [0, 0, 0, 1, 1, 1];
      const points = [-1, 0, 0.5, 1, 2];
      const kde = kernelDensityEstimation(data, points, 0.5);

      expect(kde.length).toBe(points.length);
      // Peak should be near 0 and 1
      expect(kde[1]).toBeGreaterThan(kde[0]);
      expect(kde[3]).toBeGreaterThan(kde[4]);
    });

    it("should auto-calculate bandwidth", () => {
      const data = [1, 2, 3];
      const points = [1, 2, 3];
      const kde = kernelDensityEstimation(data, points, 0); // 0 triggers auto
      expect(kde.every((v) => Number.isFinite(v))).toBe(true);
    });
  });
});
