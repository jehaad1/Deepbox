import { describe, expect, it } from "vitest";
import { loadIris, loadLeafShapes } from "../src/datasets";

describe("Datasets - Branch Coverage", () => {
  it("loads built-in datasets", () => {
    const iris = loadIris();
    expect(iris.data.shape[0]).toBeGreaterThan(0);
    const leafShapes = loadLeafShapes();
    expect(leafShapes.data.shape[0]).toBeGreaterThan(0);
  });
});
