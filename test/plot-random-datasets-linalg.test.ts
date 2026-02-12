import { describe, expect, it } from "vitest";
import { loadIris, makeBlobs, makeMoons } from "../src/datasets";
import { eig, svd } from "../src/linalg";
import { slice, tensor } from "../src/ndarray";
import { Figure } from "../src/plot";
import { rand, randint, randn } from "../src/random";

const countOccurrences = (haystack: string, needle: string): number => {
  let count = 0;
  let pos = 0;
  for (;;) {
    const next = haystack.indexOf(needle, pos);
    if (next === -1) return count;
    count += 1;
    pos = next + needle.length;
  }
};

describe("Plot - Random, datasets, linalg", () => {
  it("plots random distributions", () => {
    const fig = new Figure({ width: 300, height: 200 });
    const ax = fig.addAxes();
    ax.plot(tensor([0, 1, 2, 3]), randn([4]));
    ax.scatter(tensor([0, 1, 2, 3]), rand([4]));
    ax.bar(tensor([0, 1, 2]), randint(0, 5, [3]));
    const svg = fig.renderSVG().svg;
    expect(svg).toContain("polyline");
    expect(svg).toContain("<circle");
    expect(svg).toContain("<rect");
  });

  it("plots dataset outputs (Iris, blobs, moons)", () => {
    const iris = loadIris();
    const fig = new Figure({ width: 300, height: 200 });
    const ax = fig.addAxes();
    const nIris = iris.data.shape[0] ?? 0;
    const x = slice(iris.data, { start: 0, end: nIris }, 0);
    const y = slice(iris.data, { start: 0, end: nIris }, 1);
    ax.scatter(x, y);

    const [blobsX] = makeBlobs({ nSamples: 50, nFeatures: 2, centers: 3, randomState: 42 });
    const nBlobs = blobsX.shape[0] ?? 0;
    const bx = slice(blobsX, { start: 0, end: nBlobs }, 0);
    const by = slice(blobsX, { start: 0, end: nBlobs }, 1);
    ax.scatter(bx, by);

    const [moonsX] = makeMoons({ nSamples: 50, noise: 0.05, randomState: 7 });
    const nMoons = moonsX.shape[0] ?? 0;
    const mx = slice(moonsX, { start: 0, end: nMoons }, 0);
    const my = slice(moonsX, { start: 0, end: nMoons }, 1);
    ax.scatter(mx, my);

    const svg = fig.renderSVG().svg;
    expect(svg).toContain("<circle");
  });

  it("plots SVD singular values and eigenvalues", () => {
    const A = tensor([
      [3, 2],
      [2, 3],
    ]);
    const [, S] = svd(A);
    const [values] = eig(A);

    const fig = new Figure({ width: 280, height: 180 });
    const ax = fig.addAxes();
    const xs = tensor([0, 1]);
    ax.plot(xs, S);
    ax.scatter(xs, values);
    const svg = fig.renderSVG().svg;
    expect(svg).toContain("polyline");
    expect(svg).toContain("<circle");
  });

  it("renders heatmap from eigenvectors", () => {
    const A = tensor([
      [1, 2],
      [2, 1],
    ]);
    const [, vectors] = eig(A);
    const fig = new Figure({ width: 200, height: 140 });
    fig.addAxes().heatmap(vectors);
    const svg = fig.renderSVG().svg;
    expect(svg).toContain("<rect");
  });

  it("asserts canvas primitives counts for bars and lines", () => {
    const fig = new Figure({ width: 260, height: 160 });
    const ax = fig.addAxes();
    ax.plot(tensor([0, 1, 2]), tensor([1, 2, 3]));
    ax.barh(tensor([0, 1, 2]), tensor([1, 2, 3]));
    const svg = fig.renderSVG().svg;
    const rects = countOccurrences(svg, "<rect");
    const lines = countOccurrences(svg, "<polyline");
    expect(rects).toBeGreaterThan(1);
    expect(lines).toBe(1);
  });

  it("handles contour and contourf edge cases with NaNs and Infs", () => {
    const Z = tensor([
      [1, NaN, 3],
      [Infinity, 5, 6],
      [7, 8, -Infinity],
    ]);
    const fig = new Figure({ width: 240, height: 160 });
    const ax = fig.addAxes();
    ax.contour(tensor([]), tensor([]), Z);
    ax.contourf(tensor([]), tensor([]), Z);
    ax.heatmap(Z);
    const svg = fig.renderSVG().svg;
    expect(svg).toContain("<rect");
  });
});
