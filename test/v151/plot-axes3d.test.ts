import { describe, expect, it } from "vitest";
import { ShapeError } from "../../src/core";
import { tensor } from "../../src/ndarray";
import { Figure } from "../../src/plot";

function drawn(fig: Figure): number {
  const { svg } = fig.renderSVG();
  return (svg.match(/<(path|circle|polygon|polyline)/g) ?? []).length;
}

const X = [
  [0, 1],
  [0, 1],
];
const Y = [
  [0, 0],
  [1, 1],
];
const Z = [
  [0, 1],
  [1, 2],
];

describe("PL-1 Axes 3D methods accept tensors and validate", () => {
  it("draws surface, wireframe and scatter3d from tensors like from arrays", () => {
    const fromTensors = new Figure({ width: 300, height: 200 });
    const a = fromTensors.addAxes();
    a.surface(tensor(X), tensor(Y), tensor(Z));
    a.wireframe(tensor(X), tensor(Y), tensor(Z));
    a.scatter3d(tensor([0, 1]), tensor([0, 1]), tensor([0, 1]));

    const fromArrays = new Figure({ width: 300, height: 200 });
    const b = fromArrays.addAxes();
    b.surface(X, Y, Z);
    b.wireframe(X, Y, Z);
    b.scatter3d([0, 1], [0, 1], [0, 1]);

    expect(drawn(fromTensors)).toBeGreaterThan(3);
    expect(fromTensors.renderSVG().svg).toBe(fromArrays.renderSVG().svg);
  });

  it("still accepts Float64Array grids", () => {
    const fig = new Figure({ width: 300, height: 200 });
    const ax = fig.addAxes();
    ax.surface(
      X.map((r) => Float64Array.from(r)),
      Y.map((r) => Float64Array.from(r)),
      Z.map((r) => Float64Array.from(r))
    );
    ax.scatter3d(new Float64Array([0, 1]), new Float64Array([0, 1]), new Float64Array([0, 1]));
    expect(drawn(fig)).toBeGreaterThan(1);
  });

  it("throws the top-level validation errors from the Axes methods", () => {
    const ax = new Figure().addAxes();
    expect(() => ax.surface([], [], [])).toThrow(ShapeError);
    expect(() => ax.surface(X, Y, [[0, 1], [1]])).toThrow(/rectangular 2 x 2 grid/);
    expect(() => ax.wireframe(X, Y, [[0, 1]])).toThrow(/same, non-zero number of rows/);
    expect(() => ax.surface(tensor([0, 1]), tensor([0, 1]), tensor([0, 1]))).toThrow(ShapeError);
    expect(() => ax.scatter3d(tensor([0, 1]), tensor([0, 1]), tensor([0]))).toThrow(ShapeError);
    expect(() => ax.scatter3d(tensor(X), tensor([0, 1]), tensor([0, 1]))).toThrow(ShapeError);
  });
});
