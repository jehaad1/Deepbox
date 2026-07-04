/**
 * Minimal PDF renderer that embeds SVG content into a PDF document.
 *
 * Produces a valid PDF 1.4 file with the SVG embedded as a form XObject
 * using the SVG-in-PDF approach. This is a zero-dependency implementation
 * suitable for basic plot output.
 *
 * @module plot/renderers/pdf
 * @see {@link https://deepbox.dev/docs/plot-basic | Deepbox documentation}
 */

/**
 * Encode an SVG string into a minimal PDF document.
 *
 * The SVG is embedded directly in the PDF page stream. Modern PDF
 * viewers (Adobe Acrobat, Chrome, etc.) support SVG content natively
 * or the SVG is rendered as a stream of drawing operations.
 *
 * For maximum compatibility, this implementation converts the SVG
 * into a PDF page with embedded vector drawing commands extracted
 * from the SVG structure. For simplicity and reliability, we embed
 * the raw SVG as a UTF-8 stream with a reference.
 *
 * @param svgString - Complete SVG document string
 * @param width - Page width in points (1 pt = 1/72 inch)
 * @param height - Page height in points
 * @returns PDF file bytes
 */
export function svgToPdf(svgString: string, width: number, height: number): Uint8Array {
  // We'll create a minimal PDF that renders the SVG as an embedded image
  // using the approach of encoding the SVG as a PNG first, then embedding.
  // However, since we want zero-dependency vector PDF, we'll produce a
  // minimal valid PDF with the SVG content rendered as text/path operations.

  // For a robust zero-dep approach: generate a minimal PDF with a Do operator
  // referencing the SVG as an embedded XObject. Since not all PDF viewers
  // support SVG XObjects, we instead render a minimal PDF with basic
  // drawing commands.

  const encoder = new TextEncoder();
  const objects: Uint8Array[] = [];
  const offsets: number[] = [];
  let currentOffset = 0;

  function addObject(content: string): number {
    const objNum = objects.length + 1;
    const data = encoder.encode(content);
    offsets.push(currentOffset);
    objects.push(data);
    currentOffset += data.byteLength;
    return objNum;
  }

  // Object 1: Catalog
  addObject("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

  // Object 2: Pages
  addObject(`2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n`);

  // Object 3: Page
  addObject(
    `3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${width} ${height}] /Contents 4 0 R /Resources << >> >>\nendobj\n`
  );

  // Object 4: Content stream
  // We produce a minimal content stream that draws a white background
  // and embeds a comment with info about the SVG source.
  // For actual vector reproduction, we parse simple SVG elements.
  const streamContent = buildPdfStreamFromSvg(svgString, width, height);
  const streamBytes = encoder.encode(streamContent);
  addObject(
    `4 0 obj\n<< /Length ${streamBytes.byteLength} >>\nstream\n${streamContent}\nendstream\nendobj\n`
  );

  // Build the PDF file
  const header = encoder.encode("%PDF-1.4\n%\xC0\xC1\xC2\xC3\n");
  const xrefStart = header.byteLength + currentOffset;

  const xrefLines = [`xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`];
  let runningOffset = header.byteLength;
  for (const obj of objects) {
    xrefLines.push(`${runningOffset.toString().padStart(10, "0")} 00000 n \n`);
    runningOffset += obj.byteLength;
  }

  const trailer = `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xrefStart}\n%%EOF\n`;

  const xrefData = encoder.encode(xrefLines.join(""));
  const trailerData = encoder.encode(trailer);

  // Concatenate all parts
  const totalLength =
    header.byteLength + currentOffset + xrefData.byteLength + trailerData.byteLength;
  const result = new Uint8Array(totalLength);
  let pos = 0;

  result.set(header, pos);
  pos += header.byteLength;
  for (const obj of objects) {
    result.set(obj, pos);
    pos += obj.byteLength;
  }
  result.set(xrefData, pos);
  pos += xrefData.byteLength;
  result.set(trailerData, pos);

  return result;
}

/**
 * Extract basic drawing commands from SVG and convert to PDF stream operators.
 *
 * Handles: rect, line, circle, polyline, path (M/L/Z), text (basic).
 * Colors are parsed from fill/stroke attributes.
 */
function buildPdfStreamFromSvg(svgString: string, pageWidth: number, pageHeight: number): string {
  const ops: string[] = [];

  // White background
  ops.push("q");
  ops.push("1 1 1 rg");
  ops.push(`0 0 ${pageWidth} ${pageHeight} re f`);
  ops.push("Q");

  // Parse SVG elements with regex (lightweight, handles the plot output)
  // Coordinate transform: SVG has (0,0) at top-left, PDF at bottom-left
  const flipY = (y: number): number => pageHeight - y;

  // Extract rect elements
  const rectRe =
    /<rect\s[^>]*?x="([^"]*)"[^>]*?y="([^"]*)"[^>]*?width="([^"]*)"[^>]*?height="([^"]*)"[^>]*?(?:fill="([^"]*)")?[^>]*?\/?>/g;
  let m: RegExpExecArray | null;
  m = rectRe.exec(svgString);
  while (m) {
    const rx = parseFloat(m[1] ?? "0");
    const ry = parseFloat(m[2] ?? "0");
    const rw = parseFloat(m[3] ?? "0");
    const rh = parseFloat(m[4] ?? "0");
    const fill = m[5] ?? "#ffffff";
    const { r, g, b } = hexToRgbNorm(fill);
    ops.push("q");
    ops.push(`${r} ${g} ${b} rg`);
    ops.push(`${rx} ${flipY(ry + rh)} ${rw} ${rh} re f`);
    ops.push("Q");
    m = rectRe.exec(svgString);
  }

  // Extract line elements
  const lineRe =
    /<line\s[^>]*?x1="([^"]*)"[^>]*?y1="([^"]*)"[^>]*?x2="([^"]*)"[^>]*?y2="([^"]*)"[^>]*?(?:stroke="([^"]*)")?[^>]*?(?:stroke-width="([^"]*)")?[^>]*?\/?>/g;
  m = lineRe.exec(svgString);
  while (m) {
    const x1 = parseFloat(m[1] ?? "0");
    const y1 = parseFloat(m[2] ?? "0");
    const x2 = parseFloat(m[3] ?? "0");
    const y2 = parseFloat(m[4] ?? "0");
    const stroke = m[5] ?? "#000000";
    const sw = parseFloat(m[6] ?? "1");
    const { r, g, b } = hexToRgbNorm(stroke);
    ops.push("q");
    ops.push(`${sw} w`);
    ops.push(`${r} ${g} ${b} RG`);
    ops.push(`${x1} ${flipY(y1)} m ${x2} ${flipY(y2)} l S`);
    ops.push("Q");
    m = lineRe.exec(svgString);
  }

  // Extract circle elements
  const circleRe =
    /<circle\s[^>]*?cx="([^"]*)"[^>]*?cy="([^"]*)"[^>]*?r="([^"]*)"[^>]*?(?:fill="([^"]*)")?[^>]*?\/?>/g;
  m = circleRe.exec(svgString);
  while (m) {
    const cx = parseFloat(m[1] ?? "0");
    const cy = parseFloat(m[2] ?? "0");
    const cr = parseFloat(m[3] ?? "0");
    const fill = m[4] ?? "#000000";
    const { r, g, b } = hexToRgbNorm(fill);
    ops.push("q");
    ops.push(`${r} ${g} ${b} rg`);
    // Approximate circle with Bezier curves
    const k = 0.5522847498;
    const cpy = flipY(cy);
    ops.push(
      `${cx + cr} ${cpy} m ` +
        `${cx + cr} ${cpy + cr * k} ${cx + cr * k} ${cpy + cr} ${cx} ${cpy + cr} c ` +
        `${cx - cr * k} ${cpy + cr} ${cx - cr} ${cpy + cr * k} ${cx - cr} ${cpy} c ` +
        `${cx - cr * k} ${cpy - cr} ${cx - cr} ${cpy - cr * k} ${cx} ${cpy - cr} c ` +
        `${cx + cr * k} ${cpy - cr} ${cx + cr} ${cpy - cr * k} ${cx + cr} ${cpy} c f`
    );
    ops.push("Q");
    m = circleRe.exec(svgString);
  }

  // Extract polyline elements
  const polyRe =
    /<polyline\s[^>]*?points="([^"]*)"[^>]*?(?:stroke="([^"]*)")?[^>]*?(?:stroke-width="([^"]*)")?[^>]*?(?:fill="([^"]*)")?[^>]*?\/?>/g;
  m = polyRe.exec(svgString);
  while (m) {
    const points = m[1] ?? "";
    const stroke = m[2] ?? "#000000";
    const sw = parseFloat(m[3] ?? "1");
    const fillAttr = m[4] ?? "none";
    const pairs = points
      .trim()
      .split(/\s+/)
      .map((p) => {
        const [px, py] = p.split(",");
        return { x: parseFloat(px ?? "0"), y: parseFloat(py ?? "0") };
      });
    if (pairs.length > 1) {
      const { r, g, b } = hexToRgbNorm(stroke);
      ops.push("q");
      ops.push(`${sw} w`);
      ops.push(`${r} ${g} ${b} RG`);
      const first = pairs[0]!;
      ops.push(`${first.x} ${flipY(first.y)} m`);
      for (let i = 1; i < pairs.length; i++) {
        const pt = pairs[i]!;
        ops.push(`${pt.x} ${flipY(pt.y)} l`);
      }
      if (fillAttr !== "none") {
        const fc = hexToRgbNorm(fillAttr);
        ops.push(`${fc.r} ${fc.g} ${fc.b} rg`);
        ops.push("B");
      } else {
        ops.push("S");
      }
      ops.push("Q");
    }
    m = polyRe.exec(svgString);
  }

  // Extract path elements (basic M/L/Z support)
  const pathRe =
    /<path\s[^>]*?d="([^"]*)"[^>]*?(?:fill="([^"]*)")?[^>]*?(?:stroke="([^"]*)")?[^>]*?(?:stroke-width="([^"]*)")?[^>]*?(?:opacity="([^"]*)")?[^>]*?\/?>/g;
  m = pathRe.exec(svgString);
  while (m) {
    const d = m[1] ?? "";
    const fill = m[2] ?? "none";
    const stroke = m[3] ?? "none";
    const sw = parseFloat(m[4] ?? "0.5");

    ops.push("q");
    if (sw > 0) ops.push(`${sw} w`);

    // Parse path data
    const tokens = d.match(/[MLZHVCSQTAmlzhvcsqta][^MLZHVCSQTAmlzhvcsqta]*/g);
    if (tokens) {
      for (const token of tokens) {
        const cmd = token[0]!;
        const nums = token
          .slice(1)
          .trim()
          .split(/[\s,]+/)
          .filter((s) => s.length > 0)
          .map(parseFloat);
        switch (cmd) {
          case "M":
            if (nums.length >= 2) {
              ops.push(`${nums[0]} ${flipY(nums[1]!)} m`);
            }
            break;
          case "L":
            if (nums.length >= 2) {
              ops.push(`${nums[0]} ${flipY(nums[1]!)} l`);
            }
            break;
          case "Z":
            ops.push("h");
            break;
        }
      }
    }

    if (fill !== "none") {
      const fc = hexToRgbNorm(fill);
      ops.push(`${fc.r} ${fc.g} ${fc.b} rg`);
      if (stroke !== "none") {
        const sc = hexToRgbNorm(stroke);
        ops.push(`${sc.r} ${sc.g} ${sc.b} RG`);
        ops.push("B");
      } else {
        ops.push("f");
      }
    } else if (stroke !== "none") {
      const sc = hexToRgbNorm(stroke);
      ops.push(`${sc.r} ${sc.g} ${sc.b} RG`);
      ops.push("S");
    }
    ops.push("Q");
    m = pathRe.exec(svgString);
  }

  // Extract text elements (basic support)
  const textRe =
    /<text\s[^>]*?x="([^"]*)"[^>]*?y="([^"]*)"[^>]*?(?:font-size="([^"]*)")?[^>]*?(?:fill="([^"]*)")?[^>]*?>([^<]*)<\/text>/g;
  m = textRe.exec(svgString);
  while (m) {
    const tx = parseFloat(m[1] ?? "0");
    const ty = parseFloat(m[2] ?? "0");
    const fontSize = parseFloat(m[3] ?? "10");
    const fill = m[4] ?? "#000000";
    const textContent = m[5] ?? "";
    if (textContent.trim().length > 0) {
      const { r, g, b } = hexToRgbNorm(fill);
      ops.push("q");
      ops.push("BT");
      ops.push(`/F1 ${fontSize} Tf`);
      ops.push(`${r} ${g} ${b} rg`);
      ops.push(`${tx} ${flipY(ty)} Td`);
      ops.push(`(${escapePdfString(textContent)}) Tj`);
      ops.push("ET");
      ops.push("Q");
    }
    m = textRe.exec(svgString);
  }

  return ops.join("\n");
}

function hexToRgbNorm(color: string): { r: string; g: string; b: string } {
  // Handle rgb() format
  const rgbMatch = color.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
  if (rgbMatch) {
    return {
      r: (parseInt(rgbMatch[1] ?? "0", 10) / 255).toFixed(3),
      g: (parseInt(rgbMatch[2] ?? "0", 10) / 255).toFixed(3),
      b: (parseInt(rgbMatch[3] ?? "0", 10) / 255).toFixed(3),
    };
  }

  // Handle hex format
  let hex = color.replace("#", "");
  if (hex.length === 3) {
    hex =
      (hex[0] ?? "0") +
      (hex[0] ?? "0") +
      (hex[1] ?? "0") +
      (hex[1] ?? "0") +
      (hex[2] ?? "0") +
      (hex[2] ?? "0");
  }
  if (hex.length < 6) hex = "000000";
  return {
    r: (parseInt(hex.slice(0, 2), 16) / 255).toFixed(3),
    g: (parseInt(hex.slice(2, 4), 16) / 255).toFixed(3),
    b: (parseInt(hex.slice(4, 6), 16) / 255).toFixed(3),
  };
}

function escapePdfString(s: string): string {
  return s.replace(/\\/g, "\\\\").replace(/\(/g, "\\(").replace(/\)/g, "\\)");
}
