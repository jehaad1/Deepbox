import { describe, expect, it, vi } from "vitest";

vi.mock("node:zlib", () => ({}));

describe("PNG encoder without zlib deflateSync", () => {
  it("falls back to deflateUncompressed when dynamic import yields no deflateSync", async () => {
    const { pngEncodeRGBA } = await import("../src/plot/renderers/png");
    const rgba = new Uint8ClampedArray([10, 20, 30, 255]);
    const png = await pngEncodeRGBA(1, 1, rgba);
    expect(png[0]).toBe(137);
    expect(png[1]).toBe(80);
  });
});
