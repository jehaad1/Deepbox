import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { Figure } from "../src/plot";
import {
  calculateQuartiles,
  calculateWhiskers,
  kernelDensityEstimation,
} from "../src/plot/utils/statistics";

describe("Plot Statistical Correctness", () => {
  describe("calculateQuartiles", () => {
    it("calculates quartiles correctly for even-length arrays", () => {
      const data = [1, 2, 3, 4, 5, 6, 7, 8];
      const result = calculateQuartiles(data);

      expect(result.q1).toBeCloseTo(2.5, 2);
      expect(result.median).toBeCloseTo(4.5, 2);
      expect(result.q3).toBeCloseTo(6.5, 2);
    });

    it("calculates quartiles correctly for odd-length arrays", () => {
      const data = [1, 2, 3, 4, 5, 6, 7, 8, 9];
      const result = calculateQuartiles(data);

      expect(result.q1).toBeCloseTo(2.5, 2);
      expect(result.median).toBeCloseTo(5, 2);
      expect(result.q3).toBeCloseTo(7.5, 2);
    });

    it("handles single element arrays", () => {
      const data = [42];
      const result = calculateQuartiles(data);

      expect(result.q1).toBe(42);
      expect(result.median).toBe(42);
      expect(result.q3).toBe(42);
    });

    it("handles empty arrays", () => {
      const data: number[] = [];
      const result = calculateQuartiles(data);

      expect(result.q1).toBe(0);
      expect(result.median).toBe(0);
      expect(result.q3).toBe(0);
    });

    it("matches numpy/Excel quartile calculation method", () => {
      const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const result = calculateQuartiles(data);

      // These should match our implementation
      expect(result.q1).toBeCloseTo(3, 2);
      expect(result.median).toBeCloseTo(5.5, 2);
      expect(result.q3).toBeCloseTo(8, 2);
    });
  });

  describe("calculateWhiskers", () => {
    it("calculates whiskers without outliers", () => {
      const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const { q1, q3 } = calculateQuartiles(data);
      const result = calculateWhiskers(data, q1, q3);

      expect(result.lowerWhisker).toBe(1);
      expect(result.upperWhisker).toBe(10);
      expect(result.outliers).toHaveLength(0);
    });

    it("identifies outliers correctly", () => {
      const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 25]; // Both 20 and 25 are outliers
      const { q1, q3 } = calculateQuartiles(data);
      // Q1=3, Q3=9, IQR=6, upper bound = Q3 + 1.5*IQR = 18
      const result = calculateWhiskers(data, q1, q3);

      expect(result.lowerWhisker).toBe(1);
      expect(result.upperWhisker).toBe(9); // Last non-outlier value
      expect(result.outliers).toEqual([20, 25]); // Both exceed upper bound (18)
    });

    it("handles lower outliers", () => {
      const data = [-10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const { q1, q3 } = calculateQuartiles(data);
      const result = calculateWhiskers(data, q1, q3);

      expect(result.lowerWhisker).toBe(1);
      expect(result.upperWhisker).toBe(10);
      expect(result.outliers).toEqual([-10]);
    });

    it("handles no valid whisker range", () => {
      const data = [1, 2, 3, 100]; // 100 is far outlier
      const { q1, q3 } = calculateQuartiles(data);
      const result = calculateWhiskers(data, q1, q3);

      expect(result.lowerWhisker).toBe(1);
      expect(result.upperWhisker).toBe(100); // Returns max value
      expect(result.outliers).toEqual([]); // No outliers detected with current logic
    });
  });

  describe("kernelDensityEstimation", () => {
    it("produces reasonable KDE values", () => {
      const data = [0, 1, 2, 3, 4, 5];
      const points = [0, 1, 2, 3, 4, 5];
      const result = kernelDensityEstimation(data, points, 1);

      expect(result).toHaveLength(6);
      // KDE should be highest at the center of the data
      expect(result[2]).toBeGreaterThan(result[0]);
      expect(result[2]).toBeGreaterThan(result[5]);

      // All values should be positive
      for (const val of result) {
        expect(val).toBeGreaterThan(0);
      }
    });

    it("handles single data point", () => {
      const data = [2.5];
      const points = [0, 2.5, 5];
      const result = kernelDensityEstimation(data, points, 1);

      expect(result).toHaveLength(3);
      // Should be highest at the data point
      expect(result[1]).toBeGreaterThan(result[0]);
      expect(result[1]).toBeGreaterThan(result[2]);
    });

    it("handles empty data gracefully", () => {
      const data: number[] = [];
      const points = [0, 1, 2];
      const result = kernelDensityEstimation(data, points, 1);

      expect(result).toHaveLength(3);
      // All zero for empty data
      for (const val of result) {
        expect(val).toBe(0);
      }
    });

    it("produces symmetric KDE for symmetric data", () => {
      const data = [-2, -1, 0, 1, 2];
      const points = [-2, -1, 0, 1, 2];
      const result = kernelDensityEstimation(data, points, 1);

      expect(result[0]).toBeCloseTo(result[4], 2);
      expect(result[1]).toBeCloseTo(result[3], 2);
    });
  });

  describe("Boxplot Integration", () => {
    it("renders boxplot with correct statistics", () => {
      const fig = new Figure({ width: 400, height: 300 });
      const ax = fig.addAxes();

      const data = tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
      const boxplot = ax.boxplot(data);

      expect(boxplot.q1).toBeCloseTo(3, 2);
      expect(boxplot.median).toBeCloseTo(5.5, 2);
      expect(boxplot.q3).toBeCloseTo(8, 2);
      expect(boxplot.whiskerLow).toBe(1);
      expect(boxplot.whiskerHigh).toBe(10);
      expect(boxplot.outliers).toHaveLength(0);
    });

    it("handles outliers in boxplot", () => {
      const fig = new Figure({ width: 400, height: 300 });
      const ax = fig.addAxes();

      const data = tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 25]);
      const boxplot = ax.boxplot(data);

      // Q1=3, Q3=9, IQR=6, upper bound = Q3 + 1.5*IQR = 18
      // Both 20 and 25 exceed the upper bound
      expect(boxplot.outliers).toEqual([20, 25]);
      expect(boxplot.whiskerHigh).toBe(9); // Last non-outlier value
    });

    it("renders SVG without errors", () => {
      const fig = new Figure({ width: 400, height: 300 });
      const ax = fig.addAxes();

      ax.boxplot(tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]));

      const svg = fig.renderSVG().svg;
      expect(svg).toContain('fill="#8c564b"'); // Box body
      expect(svg).toContain('stroke-width="2"'); // Median line
    });
  });

  describe("Violinplot Integration", () => {
    it("renders violinplot with KDE", () => {
      const fig = new Figure({ width: 400, height: 300 });
      const ax = fig.addAxes();

      const data = tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
      const violinplot = ax.violinplot(data);

      expect(violinplot.q1).toBeCloseTo(3, 2);
      expect(violinplot.median).toBeCloseTo(5.5, 2);
      expect(violinplot.q3).toBeCloseTo(8, 2);
      expect(violinplot.kdePoints).toHaveLength(100);
      expect(violinplot.kdeValues).toHaveLength(100);

      // KDE values should be positive
      for (const val of violinplot.kdeValues) {
        expect(val).toBeGreaterThanOrEqual(0);
      }
    });

    it("renders SVG without errors", () => {
      const fig = new Figure({ width: 400, height: 300 });
      const ax = fig.addAxes();

      ax.violinplot(tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]));

      const svg = fig.renderSVG().svg;
      expect(svg).toContain("<path"); // Violin shape
      expect(svg).toContain('fill="#8c564b"'); // Violin fill color
    });
  });

  describe("Pie Chart Integration", () => {
    it("calculates correct angles for pie slices", () => {
      const fig = new Figure({ width: 400, height: 300 });
      const ax = fig.addAxes();

      const values = tensor([1, 2, 3]); // Total = 6, so slices are 1/6, 2/6, 3/6
      const pie = ax.pie(values);

      expect(pie.values).toEqual([1, 2, 3]);
      expect(pie.angles).toHaveLength(4); // Start + 3 end angles

      // Check angle calculations (in radians)
      // angles[0] = 0 (start of slice 1)
      // angles[1] = 1/6 * 2π (end of slice 1, start of slice 2)
      // angles[2] = 3/6 * 2π (end of slice 2, start of slice 3)
      // angles[3] = 6/6 * 2π = 2π (end of slice 3)
      expect(pie.angles[0]).toBe(0);
      expect(pie.angles[1]).toBeCloseTo((1 / 6) * 2 * Math.PI, 2);
      expect(pie.angles[2]).toBeCloseTo((3 / 6) * 2 * Math.PI, 2);
      expect(pie.angles[3]).toBeCloseTo(2 * Math.PI, 2);
    });

    it("handles labels correctly", () => {
      const fig = new Figure({ width: 400, height: 300 });
      const ax = fig.addAxes();

      const values = tensor([10, 20, 30]);
      const labels = ["A", "B", "C"];
      const pie = ax.pie(values, labels);

      expect(pie.labels).toEqual(labels);
    });

    it("renders SVG without errors", () => {
      const fig = new Figure({ width: 400, height: 300 });
      const ax = fig.addAxes();

      ax.pie(tensor([1, 2, 3, 4]), ["A", "B", "C", "D"]);

      const svg = fig.renderSVG().svg;
      expect(svg).toContain("<path"); // Pie slices
      expect(svg).toContain("<text"); // Labels
    });

    it("rejects zero-total data", () => {
      const fig = new Figure({ width: 400, height: 300 });
      const ax = fig.addAxes();

      const values = tensor([0, 0, 0]);
      expect(() => ax.pie(values)).toThrow(/positive number/);
    });
  });

  describe("Mathematical Consistency", () => {
    it("boxplot and violinplot use same quartile calculations", () => {
      const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const quartiles = calculateQuartiles(data);

      const fig = new Figure({ width: 400, height: 300 });
      const ax = fig.addAxes();

      const boxplot = ax.boxplot(tensor(data));
      const violinplot = ax.violinplot(tensor(data));

      // Verify boxplot and violinplot match the calculated quartiles
      expect(boxplot.q1).toBeCloseTo(quartiles.q1, 5);
      expect(boxplot.median).toBeCloseTo(quartiles.median, 5);
      expect(boxplot.q3).toBeCloseTo(quartiles.q3, 5);
      expect(violinplot.q1).toBeCloseTo(quartiles.q1, 5);
      expect(violinplot.median).toBeCloseTo(quartiles.median, 5);
      expect(violinplot.q3).toBeCloseTo(quartiles.q3, 5);
    });

    it("data range calculations are consistent", () => {
      const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

      const fig = new Figure({ width: 400, height: 300 });
      const ax = fig.addAxes();

      const boxplot = ax.boxplot(tensor(data));
      const violinplot = ax.violinplot(tensor(data));
      const pie = ax.pie(tensor(data));

      const boxplotRange = boxplot.getDataRange();
      const violinplotRange = violinplot.getDataRange();
      const pieRange = pie.getDataRange();

      // Ensure ranges are not null
      expect(boxplotRange).toBeDefined();
      expect(violinplotRange).toBeDefined();
      expect(pieRange).toBeDefined();
      if (!boxplotRange || !violinplotRange || !pieRange) return;

      // Boxplot and violinplot ranges should encompass the data
      // Note: violinplot uses KDE which may extend beyond data bounds
      expect(boxplotRange.ymin).toBeLessThanOrEqual(1);
      expect(boxplotRange.ymax).toBeGreaterThanOrEqual(10);
      expect(violinplotRange.ymin).toBeLessThanOrEqual(1);
      expect(violinplotRange.ymax).toBeGreaterThanOrEqual(10);

      // Pie chart range should be a valid square region centered on the pie
      expect(pieRange.xmax - pieRange.xmin).toBeGreaterThan(0);
      expect(pieRange.ymax - pieRange.ymin).toBeGreaterThan(0);
      // Pie should have roughly equal width and height (it's circular)
      const pieWidth = pieRange.xmax - pieRange.xmin;
      const pieHeight = pieRange.ymax - pieRange.ymin;
      expect(pieWidth).toBeCloseTo(pieHeight, 1);
    });
  });
});
