import { bench, describe } from "vitest";
import { tensor } from "../../src/ndarray";
import { bar, Figure, figure, heatmap, hist, plot, scatter } from "../../src/plot";

describe("Plot Performance Benchmarks", () => {
  describe("Line Plots", () => {
    bench("line plot - 100 points", () => {
      const x = tensor(Array.from({ length: 100 }, (_, i) => i));
      const y = tensor(Array.from({ length: 100 }, (_, i) => Math.sin(i / 10)));
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.plot(x, y);
      fig.renderSVG();
    });

    bench("line plot - 1,000 points", () => {
      const x = tensor(Array.from({ length: 1000 }, (_, i) => i));
      const y = tensor(Array.from({ length: 1000 }, (_, i) => Math.sin(i / 100)));
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.plot(x, y);
      fig.renderSVG();
    });

    bench("line plot - 10,000 points", () => {
      const x = tensor(Array.from({ length: 10000 }, (_, i) => i));
      const y = tensor(Array.from({ length: 10000 }, (_, i) => Math.sin(i / 1000)));
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.plot(x, y);
      fig.renderSVG();
    });

    bench("line plot - 100,000 points", () => {
      const x = tensor(Array.from({ length: 100000 }, (_, i) => i));
      const y = tensor(Array.from({ length: 100000 }, (_, i) => Math.sin(i / 10000)));
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.plot(x, y);
      fig.renderSVG();
    });
  });

  describe("Scatter Plots", () => {
    bench("scatter plot - 100 points", () => {
      const x = tensor(Array.from({ length: 100 }, () => Math.random()));
      const y = tensor(Array.from({ length: 100 }, () => Math.random()));
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.scatter(x, y);
      fig.renderSVG();
    });

    bench("scatter plot - 1,000 points", () => {
      const x = tensor(Array.from({ length: 1000 }, () => Math.random()));
      const y = tensor(Array.from({ length: 1000 }, () => Math.random()));
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.scatter(x, y);
      fig.renderSVG();
    });

    bench("scatter plot - 10,000 points", () => {
      const x = tensor(Array.from({ length: 10000 }, () => Math.random()));
      const y = tensor(Array.from({ length: 10000 }, () => Math.random()));
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.scatter(x, y);
      fig.renderSVG();
    });
  });

  describe("Bar Charts", () => {
    bench("bar chart - 10 bars", () => {
      const x = tensor(Array.from({ length: 10 }, (_, i) => i));
      const height = tensor(Array.from({ length: 10 }, () => Math.random() * 100));
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.bar(x, height);
      fig.renderSVG();
    });

    bench("bar chart - 100 bars", () => {
      const x = tensor(Array.from({ length: 100 }, (_, i) => i));
      const height = tensor(Array.from({ length: 100 }, () => Math.random() * 100));
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.bar(x, height);
      fig.renderSVG();
    });

    bench("bar chart - 1,000 bars", () => {
      const x = tensor(Array.from({ length: 1000 }, (_, i) => i));
      const height = tensor(Array.from({ length: 1000 }, () => Math.random() * 100));
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.bar(x, height);
      fig.renderSVG();
    });
  });

  describe("Histograms", () => {
    bench("histogram - 100 values, 10 bins", () => {
      const data = tensor(Array.from({ length: 100 }, () => Math.random()));
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.hist(data, 10);
      fig.renderSVG();
    });

    bench("histogram - 1,000 values, 20 bins", () => {
      const data = tensor(Array.from({ length: 1000 }, () => Math.random()));
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.hist(data, 20);
      fig.renderSVG();
    });

    bench("histogram - 10,000 values, 50 bins", () => {
      const data = tensor(Array.from({ length: 10000 }, () => Math.random()));
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.hist(data, 50);
      fig.renderSVG();
    });
  });

  describe("Heatmaps", () => {
    bench("heatmap - 10×10", () => {
      const data = tensor(Array.from({ length: 100 }, () => Math.random())).reshape([10, 10]);
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.heatmap(data);
      fig.renderSVG();
    });

    bench("heatmap - 50×50", () => {
      const data = tensor(Array.from({ length: 2500 }, () => Math.random())).reshape([50, 50]);
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.heatmap(data);
      fig.renderSVG();
    });

    bench("heatmap - 100×100", () => {
      const data = tensor(Array.from({ length: 10000 }, () => Math.random())).reshape([100, 100]);
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.heatmap(data);
      fig.renderSVG();
    });

    bench("heatmap - 1000×1000", () => {
      const data = tensor(Array.from({ length: 1000000 }, () => Math.random())).reshape([
        1000, 1000,
      ]);
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.heatmap(data);
      fig.renderSVG();
    });
  });

  describe("PNG Rendering", () => {
    bench("PNG encoding - 640×480", async () => {
      const fig = new Figure({ width: 640, height: 480 });
      const ax = fig.addAxes();
      ax.plot(tensor([1, 2, 3]), tensor([1, 4, 9]));
      await fig.renderPNG();
    });

    bench("PNG encoding - 1280×720", async () => {
      const fig = new Figure({ width: 1280, height: 720 });
      const ax = fig.addAxes();
      ax.plot(tensor([1, 2, 3]), tensor([1, 4, 9]));
      await fig.renderPNG();
    });

    bench("PNG encoding - 1920×1080", async () => {
      const fig = new Figure({ width: 1920, height: 1080 });
      const ax = fig.addAxes();
      ax.plot(tensor([1, 2, 3]), tensor([1, 4, 9]));
      await fig.renderPNG();
    });
  });

  describe("Global Functions", () => {
    bench("global plot function", () => {
      figure();
      plot(tensor([1, 2, 3, 4, 5]), tensor([1, 4, 9, 16, 25]));
      const fig = figure();
      fig.renderSVG();
    });

    bench("global scatter function", () => {
      figure();
      scatter(tensor([1, 2, 3, 4, 5]), tensor([1, 4, 9, 16, 25]));
      const fig = figure();
      fig.renderSVG();
    });

    bench("global bar function", () => {
      figure();
      bar(tensor([1, 2, 3, 4, 5]), tensor([10, 20, 15, 25, 30]));
      const fig = figure();
      fig.renderSVG();
    });

    bench("global hist function", () => {
      figure();
      hist(tensor(Array.from({ length: 1000 }, () => Math.random())), 20);
      const fig = figure();
      fig.renderSVG();
    });

    bench("global heatmap function", () => {
      figure();
      heatmap(tensor(Array.from({ length: 100 }, () => Math.random())).reshape([10, 10]));
      const fig = figure();
      fig.renderSVG();
    });
  });

  describe("Complex Plots", () => {
    bench("multiple plots on same axes", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      for (let i = 0; i < 5; i++) {
        const x = tensor(Array.from({ length: 100 }, (_, j) => j));
        const y = tensor(Array.from({ length: 100 }, (_, j) => Math.sin(j / 10 + i)));
        ax.plot(x, y);
      }
      fig.renderSVG();
    });

    bench("subplot grid - 2×2", () => {
      const fig = new Figure({ width: 800, height: 600 });
      for (let i = 0; i < 4; i++) {
        const ax = fig.addAxes();
        const x = tensor(Array.from({ length: 50 }, (_, j) => j));
        const y = tensor(Array.from({ length: 50 }, (_, j) => Math.sin(j / 5 + i)));
        ax.plot(x, y);
      }
      fig.renderSVG();
    });

    bench("mixed plot types", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.plot(tensor([1, 2, 3, 4, 5]), tensor([1, 4, 9, 16, 25]));
      ax.scatter(tensor([1.5, 2.5, 3.5]), tensor([2, 6, 12]));
      ax.bar(tensor([1, 2, 3]), tensor([5, 10, 15]));
      fig.renderSVG();
    });
  });

  describe("Color Parsing", () => {
    bench("hex color parsing", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      for (let i = 0; i < 100; i++) {
        ax.plot(tensor([i, i + 1]), tensor([i, i + 1]), { color: "#ff0000" });
      }
      fig.renderSVG();
    });

    bench("named color parsing", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      for (let i = 0; i < 100; i++) {
        ax.plot(tensor([i, i + 1]), tensor([i, i + 1]), {
          color: "forestgreen",
        });
      }
      fig.renderSVG();
    });

    bench("RGB color parsing", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      for (let i = 0; i < 100; i++) {
        ax.plot(tensor([i, i + 1]), tensor([i, i + 1]), {
          color: "rgb(255, 0, 0)",
        });
      }
      fig.renderSVG();
    });

    bench("HSL color parsing", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      for (let i = 0; i < 100; i++) {
        ax.plot(tensor([i, i + 1]), tensor([i, i + 1]), {
          color: "hsl(120, 100%, 50%)",
        });
      }
      fig.renderSVG();
    });
  });
});
