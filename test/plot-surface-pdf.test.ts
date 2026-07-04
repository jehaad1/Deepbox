import { describe, expect, it, vi } from "vitest";
import { InvalidParameterError, ShapeError } from "../src/core";
import { RasterCanvas } from "../src/plot/canvas/RasterCanvas";
import { Scatter3D, Surface3D, Wireframe3D } from "../src/plot/plots/Surface3D";
import { svgToPdf } from "../src/plot/renderers/pdf";

const grid2x2 = {
  x: [new Float64Array([0, 1]), new Float64Array([0, 1])],
  y: [new Float64Array([0, 0]), new Float64Array([1, 1])],
  z: [new Float64Array([0, 0.5]), new Float64Array([0.5, 1])],
};

describe("Surface3D and related drawables", () => {
  it("Surface3D rejects mismatched grid row counts", () => {
    expect(
      () =>
        new Surface3D(
          [new Float64Array([0])],
          [new Float64Array([0]), new Float64Array([1])],
          [new Float64Array([0])]
        )
    ).toThrow(ShapeError);
  });

  it("Surface3D getDataRange and drawSVG run for a minimal grid", () => {
    const { x, y, z } = grid2x2;
    const s = new Surface3D(x, y, z);
    const range = s.getDataRange();
    expect(range).not.toBeNull();
    expect(Number.isFinite(range!.xmin)).toBe(true);

    const pushed: string[] = [];
    s.drawSVG({
      transform: { xToPx: (v) => v * 50, yToPx: (v) => v * 50 },
      push: (el) => pushed.push(el),
    });
    expect(pushed.some((p) => p.includes("<path"))).toBe(true);
  });

  it("Surface3D drawRaster draws grid lines", () => {
    const { x, y, z } = grid2x2;
    const s = new Surface3D(x, y, z);
    const canvas = new RasterCanvas(120, 120);
    const spy = vi.spyOn(canvas, "drawLineRGBA");
    s.drawRaster({
      transform: { xToPx: (v) => v * 50 + 10, yToPx: (v) => v * 50 + 10 },
      canvas,
    });
    expect(spy).toHaveBeenCalled();
  });

  it("Wireframe3D drawSVG emits polylines", () => {
    const { x, y, z } = grid2x2;
    const w = new Wireframe3D(x, y, z);
    const pushed: string[] = [];
    w.drawSVG({
      transform: { xToPx: (v) => v * 40, yToPx: (v) => v * 40 },
      push: (el) => pushed.push(el),
    });
    expect(pushed.some((p) => p.includes("<polyline"))).toBe(true);
  });

  it("Scatter3D rejects non-positive size", () => {
    expect(
      () =>
        new Scatter3D(new Float64Array([1]), new Float64Array([2]), new Float64Array([3]), {
          size: 0,
        })
    ).toThrow(InvalidParameterError);
  });

  it("Scatter3D drawSVG and drawRaster execute", () => {
    const sc = new Scatter3D(
      new Float64Array([0, 1]),
      new Float64Array([0, 1]),
      new Float64Array([0, 1])
    );
    const pushed: string[] = [];
    sc.drawSVG({
      transform: { xToPx: (v) => v * 100, yToPx: (v) => v * 100 },
      push: (el) => pushed.push(el),
    });
    expect(pushed.some((p) => p.includes("<circle"))).toBe(true);

    const canvas = new RasterCanvas(80, 80);
    const spy = vi.spyOn(canvas, "drawCircleRGBA");
    sc.drawRaster({
      transform: { xToPx: (v) => v * 30 + 10, yToPx: (v) => v * 30 + 10 },
      canvas,
    });
    expect(spy).toHaveBeenCalled();
  });
});

describe("svgToPdf", () => {
  it("produces a PDF header and EOF for a tiny SVG", () => {
    const svg = `<svg xmlns="http://www.w3.org/2000/svg"><rect x="10" y="20" width="30" height="40" fill="#112233"/></svg>`;
    const pdf = svgToPdf(svg, 200, 200);
    const head = new TextDecoder().decode(pdf.slice(0, 8));
    expect(head.startsWith("%PDF-1.4")).toBe(true);
    const tail = new TextDecoder().decode(pdf.slice(-32));
    expect(tail).toContain("%%EOF");
  });

  it("handles line, path, and text elements", () => {
    const svg = `
      <svg xmlns="http://www.w3.org/2000/svg">
        <line x1="0" y1="0" x2="50" y2="50" stroke="#000000" stroke-width="2"/>
        <path d="M10 10 L20 20 Z" fill="none" stroke="#ff0000" stroke-width="1"/>
        <text x="5" y="15" font-size="12" fill="#000000">Hi</text>
      </svg>`;
    const pdf = svgToPdf(svg, 100, 100);
    expect(pdf.byteLength).toBeGreaterThan(200);
  });

  it("handles rgb() fills, short hex, and circles", () => {
    const svg = `
      <svg xmlns="http://www.w3.org/2000/svg">
        <rect x="1" y="2" width="3" height="4" fill="rgb(200, 100, 50)"/>
        <rect x="5" y="5" width="2" height="2" fill="#abc"/>
        <circle cx="40" cy="40" r="5" fill="#112233"/>
      </svg>`;
    const pdf = svgToPdf(svg, 80, 80);
    expect(pdf.byteLength).toBeGreaterThan(200);
  });

  it("handles polylines with fill vs stroke-only", () => {
    const svg = `
      <svg xmlns="http://www.w3.org/2000/svg">
        <polyline points="0,0 20,0 20,20" stroke="#000000" stroke-width="1" fill="#ff0000"/>
        <polyline points="30,30 40,40" stroke="#0000ff" stroke-width="2" fill="none"/>
      </svg>`;
    const pdf = svgToPdf(svg, 120, 120);
    expect(pdf.byteLength).toBeGreaterThan(200);
  });

  it("handles path fill-only, stroke-only, and fill+stroke", () => {
    const svg = `
      <svg xmlns="http://www.w3.org/2000/svg">
        <path d="M5 5 L15 5 L10 15 Z" fill="#00aa00"/>
        <path d="M20 20 L40 20" fill="none" stroke="#aa00aa" stroke-width="1"/>
        <path d="M50 50 L60 50 L60 60 Z" fill="#cccccc" stroke="#333333" stroke-width="0.5"/>
      </svg>`;
    const pdf = svgToPdf(svg, 200, 200);
    expect(pdf.byteLength).toBeGreaterThan(250);
  });

  it("escapes parentheses in PDF text strings", () => {
    const svg = `
      <svg xmlns="http://www.w3.org/2000/svg">
        <text x="10" y="20" font-size="10" fill="#000000">Label (units)</text>
      </svg>`;
    const pdf = svgToPdf(svg, 100, 100);
    const text = new TextDecoder().decode(pdf);
    expect(text).toContain("\\(");
    expect(text).toContain("\\)");
  });

  it("ignores whitespace-only text nodes", () => {
    const svg = `<svg xmlns="http://www.w3.org/2000/svg"><text x="1" y="2">   </text></svg>`;
    const pdf = svgToPdf(svg, 50, 50);
    expect(new TextDecoder().decode(pdf)).not.toContain("BT");
  });
});
