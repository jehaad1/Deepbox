import { describe, expect, it, vi } from "vitest";

vi.mock("node:zlib", () => ({
  deflateSync: () => {
    throw new Error("deflate failed");
  },
}));

describe("PNG encoder when zlib.deflateSync throws", () => {
  it("falls back to deflateUncompressed in catch block", async () => {
    const { pngEncodeRGBA } = await import("../src/plot/renderers/png");
    const rgba = new Uint8ClampedArray(4);
    rgba.set([100, 150, 200, 255]);
    const png = await pngEncodeRGBA(1, 1, rgba);
    expect(png[0]).toBe(137);
  });
});
