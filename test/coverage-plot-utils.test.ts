import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { normalizeColor, parseHexColorToRGBA } from "../src/plot/utils/colors";
import { buildContourGrid } from "../src/plot/utils/contours";

describe("plot utils branch coverage", () => {
  it("normalizeColor and parseHexColorToRGBA cover common formats", () => {
    expect(normalizeColor(undefined, "#ff0000")).toBe("#ff0000");
    expect(normalizeColor("   ", "#00ff00")).toBe("#00ff00");
    expect(normalizeColor("#AABBCCDD", "#000000")).toBe("#aabbccdd");
    expect(normalizeColor("#aabbcc", "#000000")).toBe("#aabbcc");
    expect(normalizeColor("rgb(255,0,0)", "#000000")).toBe("#ff0000");
    expect(normalizeColor("rgba(255,0,0,0.5)", "#000000")).toBe("#ff000080");
    expect(normalizeColor("hsl(0,100%,50%)", "#000000")).toBe("#ff0000");
    expect(normalizeColor("hsla(120,100%,50%,0.25)", "#000000")).toBe("#00ff0040");

    const named = parseHexColorToRGBA("red");
    expect(named.r).toBe(255);
    expect(named.g).toBe(0);
    expect(named.b).toBe(0);

    const cached = parseHexColorToRGBA("red");
    expect(cached).toBe(named);

    const invalidHex = parseHexColorToRGBA("#gggggg");
    expect(invalidHex.r).toBe(0);
    expect(invalidHex.g).toBe(0);
    expect(invalidHex.b).toBe(0);

    const rgbClamp = parseHexColorToRGBA("rgba(10,20,30,2)");
    expect(rgbClamp.a).toBe(255);

    const invalidRgb = parseHexColorToRGBA("rgb(10,20)");
    expect(invalidRgb.r).toBe(0);

    const gray = parseHexColorToRGBA("hsl(0,0%,50%)");
    expect(gray.r).toBe(gray.g);
    expect(gray.g).toBe(gray.b);
  });

  it("buildContourGrid handles empty coords and extent", () => {
    const Z = tensor([
      [1, 2],
      [3, 4],
    ]);
    const grid = buildContourGrid(tensor([]), tensor([]), Z, {
      xmin: 0,
      xmax: 1,
      ymin: 0,
      ymax: 2,
    });
    expect(grid.xCoords.length).toBe(2);
    expect(grid.yCoords.length).toBe(2);
  });

  it("buildContourGrid throws on invalid extent", () => {
    const Z = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(() =>
      buildContourGrid(tensor([]), tensor([]), Z, {
        xmin: 1,
        xmax: 0,
        ymin: 0,
        ymax: 1,
      })
    ).toThrow(/extent/i);
  });

  it("buildContourGrid handles 1D coords and mismatches", () => {
    const Z = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const X = tensor([0, 1, 2]);
    const Y = tensor([0, 1]);
    const grid = buildContourGrid(X, Y, Z);
    expect(grid.xCoords.length).toBe(3);
    expect(grid.yCoords.length).toBe(2);

    expect(() => buildContourGrid(tensor([0, 1]), Y, Z)).toThrow(/length/i);
    expect(() => buildContourGrid(X, tensor([0]), Z)).toThrow(/length/i);
  });

  it("buildContourGrid handles meshgrid and errors", () => {
    const Z = tensor([
      [1, 2],
      [3, 4],
    ]);
    const X = tensor([
      [0, 1],
      [0, 1],
    ]);
    const Y = tensor([
      [0, 0],
      [1, 1],
    ]);
    const grid = buildContourGrid(X, Y, Z);
    expect(grid.xCoords.length).toBe(2);

    const badX = tensor([
      [0, 1],
      [0, 2],
    ]);
    expect(() => buildContourGrid(badX, Y, Z)).toThrow(/rectilinear/i);

    expect(() => buildContourGrid(tensor([[0, 1, 2]]), Y, Z)).toThrow(/shape/i);
  });

  it("buildContourGrid validates coordinate presence and finiteness", () => {
    const Z = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(() => buildContourGrid(tensor([0, 1]), tensor([]), Z)).toThrow(/both X and Y/i);
    expect(() => buildContourGrid(tensor([0, Number.NaN]), tensor([0, 1]), Z)).toThrow(/finite/i);
  });

  it("buildContourGrid rejects all-NaN Z", () => {
    const Z = tensor([
      [Number.NaN, Number.NaN],
      [Number.NaN, Number.NaN],
    ]);
    expect(() => buildContourGrid(tensor([]), tensor([]), Z)).toThrow(/finite Z/i);
  });
});
