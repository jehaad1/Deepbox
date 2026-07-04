/**
 * Zero-dependency Excel (.xlsx) reader and writer for Deepbox DataFrames.
 *
 * XLSX files are ZIP archives containing Office Open XML spreadsheet files.
 * This module implements a minimal ZIP reader/writer and XML parser/serializer
 * to handle common spreadsheet data without any external dependencies.
 *
 * Supports:
 * - Reading: Extracts cell values (strings, numbers, booleans) from Sheet1
 * - Writing: Creates valid .xlsx files with a single worksheet
 *
 * Limitations:
 * - Single sheet only (reads Sheet1, writes Sheet1)
 * - No formula evaluation
 * - No cell formatting/styling
 * - No merged cells
 * - No date type (dates are stored as numbers or strings)
 *
 * @module dataframe/io/xlsx
 * @see {@link https://deepbox.dev/docs/dataframe-overview | Deepbox documentation}
 */

import { DataValidationError } from "../../core";

// ─── Mini ZIP Utilities (zero-dependency) ────────────────────────────────────

/** CRC-32 lookup table. */
const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let j = 0; j < 8; j++) {
      c = (c & 1) !== 0 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    table[i] = c;
  }
  return table;
})();

function crc32(data: Uint8Array): number {
  let crc = 0xffffffff;
  for (let i = 0; i < data.length; i++) {
    crc = CRC_TABLE[(crc ^ data[i]!) & 0xff]! ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function encodeUTF8(str: string): Uint8Array {
  const encoder = new TextEncoder();
  return encoder.encode(str);
}

function decodeUTF8(data: Uint8Array): string {
  const decoder = new TextDecoder("utf-8");
  return decoder.decode(data);
}

function writeUint16LE(value: number): Uint8Array {
  return new Uint8Array([value & 0xff, (value >> 8) & 0xff]);
}

function writeUint32LE(value: number): Uint8Array {
  return new Uint8Array([
    value & 0xff,
    (value >> 8) & 0xff,
    (value >> 16) & 0xff,
    (value >> 24) & 0xff,
  ]);
}

function readUint16LE(data: Uint8Array, offset: number): number {
  return data[offset]! | (data[offset + 1]! << 8);
}

function readUint32LE(data: Uint8Array, offset: number): number {
  return (
    (data[offset]! |
      (data[offset + 1]! << 8) |
      (data[offset + 2]! << 16) |
      ((data[offset + 3]! << 24) >>> 0)) >>>
    0
  );
}

function concatBytes(...arrays: Uint8Array[]): Uint8Array {
  let totalLen = 0;
  for (const arr of arrays) totalLen += arr.length;
  const result = new Uint8Array(totalLen);
  let offset = 0;
  for (const arr of arrays) {
    result.set(arr, offset);
    offset += arr.length;
  }
  return result;
}

type ZipEntry = { name: string; data: Uint8Array };

function createZip(entries: ZipEntry[]): Uint8Array {
  const parts: Uint8Array[] = [];
  const centralDir: Uint8Array[] = [];
  let localOffset = 0;

  for (const entry of entries) {
    const nameBytes = encodeUTF8(entry.name);
    const crc = crc32(entry.data);
    const size = entry.data.length;

    // Local file header
    const localHeader = concatBytes(
      new Uint8Array([0x50, 0x4b, 0x03, 0x04]), // signature
      writeUint16LE(20), // version needed
      writeUint16LE(0), // flags
      writeUint16LE(0), // compression: stored
      writeUint16LE(0), // mod time
      writeUint16LE(0), // mod date
      writeUint32LE(crc),
      writeUint32LE(size),
      writeUint32LE(size),
      writeUint16LE(nameBytes.length),
      writeUint16LE(0), // extra field length
      nameBytes
    );

    parts.push(localHeader);
    parts.push(entry.data);

    // Central directory entry
    const cdEntry = concatBytes(
      new Uint8Array([0x50, 0x4b, 0x01, 0x02]),
      writeUint16LE(20), // version made by
      writeUint16LE(20), // version needed
      writeUint16LE(0), // flags
      writeUint16LE(0), // compression
      writeUint16LE(0), // mod time
      writeUint16LE(0), // mod date
      writeUint32LE(crc),
      writeUint32LE(size),
      writeUint32LE(size),
      writeUint16LE(nameBytes.length),
      writeUint16LE(0), // extra field length
      writeUint16LE(0), // comment length
      writeUint16LE(0), // disk number
      writeUint16LE(0), // internal attrs
      writeUint32LE(0), // external attrs
      writeUint32LE(localOffset),
      nameBytes
    );
    centralDir.push(cdEntry);

    localOffset += localHeader.length + entry.data.length;
  }

  let cdSize = 0;
  for (const cd of centralDir) cdSize += cd.length;

  // End of central directory
  const eocd = concatBytes(
    new Uint8Array([0x50, 0x4b, 0x05, 0x06]),
    writeUint16LE(0), // disk number
    writeUint16LE(0), // central dir disk
    writeUint16LE(entries.length),
    writeUint16LE(entries.length),
    writeUint32LE(cdSize),
    writeUint32LE(localOffset),
    writeUint16LE(0) // comment length
  );

  return concatBytes(...parts, ...centralDir, eocd);
}

/**
 * Inflate a raw-deflate stream using Node's zlib (accessed via
 * `process.getBuiltinModule` so browser bundles stay import-free).
 */
/** Default cap on a single decompressed zip entry (guards against zip bombs). */
const DEFAULT_MAX_ENTRY_BYTES = 256 * 1024 * 1024;

function inflateRaw(data: Uint8Array, maxEntryBytes: number, entryName: string): Uint8Array {
  const proc = (
    globalThis as {
      process?: { getBuiltinModule?: (id: string) => unknown };
    }
  ).process;
  const zlib = proc?.getBuiltinModule?.("node:zlib") as
    | {
        inflateRawSync?: (buf: Uint8Array, opts?: { maxOutputLength?: number }) => Uint8Array;
      }
    | undefined;
  if (!zlib?.inflateRawSync) {
    throw new DataValidationError(
      "readXlsx: this file uses deflate-compressed zip entries, which require a Node.js runtime to read"
    );
  }
  try {
    return new Uint8Array(zlib.inflateRawSync(data, { maxOutputLength: maxEntryBytes }));
  } catch (err) {
    if (err instanceof RangeError || (err as { code?: string }).code === "ERR_BUFFER_TOO_LARGE") {
      throw new DataValidationError(
        `readXlsx: zip entry "${entryName}" decompresses beyond the ${maxEntryBytes}-byte limit ` +
          "(possible decompression bomb); raise maxEntryBytes explicitly if the file is trusted"
      );
    }
    throw err;
  }
}

function readZip(data: Uint8Array, maxEntryBytes = DEFAULT_MAX_ENTRY_BYTES): ZipEntry[] {
  const entries: ZipEntry[] = [];

  // Find End of Central Directory
  let eocdOffset = -1;
  for (let i = data.length - 22; i >= 0; i--) {
    if (data[i] === 0x50 && data[i + 1] === 0x4b && data[i + 2] === 0x05 && data[i + 3] === 0x06) {
      eocdOffset = i;
      break;
    }
  }
  if (eocdOffset < 0) return entries;

  const numEntries = readUint16LE(data, eocdOffset + 8);
  const cdOffset = readUint32LE(data, eocdOffset + 16);

  let pos = cdOffset;
  for (let e = 0; e < numEntries; e++) {
    if (pos + 46 > data.length) break;
    if (
      data[pos] !== 0x50 ||
      data[pos + 1] !== 0x4b ||
      data[pos + 2] !== 0x01 ||
      data[pos + 3] !== 0x02
    )
      break;

    const method = readUint16LE(data, pos + 10);
    const compressedSize = readUint32LE(data, pos + 20);
    const nameLen = readUint16LE(data, pos + 28);
    const extraLen = readUint16LE(data, pos + 30);
    const commentLen = readUint16LE(data, pos + 32);
    const localOffset = readUint32LE(data, pos + 42);

    const name = decodeUTF8(data.subarray(pos + 46, pos + 46 + nameLen));

    // Read from local file header
    const localNameLen = readUint16LE(data, localOffset + 26);
    const localExtraLen = readUint16LE(data, localOffset + 28);
    const dataStart = localOffset + 30 + localNameLen + localExtraLen;
    const rawData = data.slice(dataStart, dataStart + compressedSize);

    if (compressedSize > maxEntryBytes) {
      throw new DataValidationError(
        `readXlsx: zip entry "${name}" exceeds the ${maxEntryBytes}-byte limit; ` +
          "raise maxEntryBytes explicitly if the file is trusted"
      );
    }

    let fileData: Uint8Array;
    if (method === 0) {
      fileData = rawData;
    } else if (method === 8) {
      fileData = inflateRaw(rawData, maxEntryBytes, name);
    } else {
      throw new DataValidationError(
        `readXlsx: unsupported zip compression method ${method} for entry "${name}"`
      );
    }

    entries.push({ name, data: fileData });
    pos += 46 + nameLen + extraLen + commentLen;
  }

  return entries;
}

// ─── XML Utilities ───────────────────────────────────────────────────────────

function escapeXml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/** Decode the XML entities {@link escapeXml} produces (plus apostrophes). */
function unescapeXml(s: string): string {
  return s
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&#(\d+);/g, (_, code: string) => String.fromCodePoint(Number(code)))
    .replace(/&amp;/g, "&");
}

/** Convert column index (0-based) to Excel column letter (A, B, ..., Z, AA, ...) */
function colLetter(index: number): string {
  let letter = "";
  let n = index;
  while (n >= 0) {
    letter = String.fromCharCode((n % 26) + 65) + letter;
    n = Math.floor(n / 26) - 1;
  }
  return letter;
}

/** Parse Excel cell reference (e.g. "A1") to [col, row] (0-based). */
function parseCellRef(ref: string): [number, number] {
  const match = /^([A-Z]+)(\d+)$/.exec(ref);
  if (!match) return [0, 0];
  const colStr = match[1]!;
  const rowStr = match[2]!;
  let col = 0;
  for (let i = 0; i < colStr.length; i++) {
    col = col * 26 + (colStr.charCodeAt(i) - 64);
  }
  return [col - 1, parseInt(rowStr, 10) - 1];
}

// ─── Simple XML tag extractor ────────────────────────────────────────────────

function extractTagContent(xml: string, tag: string): string[] {
  const results: string[] = [];
  const openTag = `<${tag}`;
  const closeTag = `</${tag}>`;
  let pos = 0;
  while (pos < xml.length) {
    const start = xml.indexOf(openTag, pos);
    if (start < 0) break;
    const end = xml.indexOf(closeTag, start);
    if (end < 0) break;
    results.push(xml.substring(start, end + closeTag.length));
    pos = end + closeTag.length;
  }
  return results;
}

function getAttr(element: string, attr: string): string | undefined {
  const regex = new RegExp(`${attr}="([^"]*)"`, "");
  const match = regex.exec(element);
  return match?.[1];
}

function getInnerText(element: string, tag: string): string | undefined {
  const open = `<${tag}>`;
  const close = `</${tag}>`;
  const start = element.indexOf(open);
  if (start < 0) return undefined;
  const end = element.indexOf(close, start);
  if (end < 0) return undefined;
  return element.substring(start + open.length, end);
}

// ─── XLSX Reader ─────────────────────────────────────────────────────────────

/** Options for reading an XLSX file. */
export type XlsxReadOptions = {
  /** Sheet name to read. Default: first sheet. */
  readonly sheet?: string;
  /** Whether the first row contains headers. Default: true. */
  readonly header?: boolean;
  /**
   * Maximum decompressed size per zip entry in bytes (guards against
   * decompression bombs in untrusted files). Default: 256 MiB.
   */
  readonly maxEntryBytes?: number;
};

/**
 * Locate the worksheet zip entry for `sheetName` (or the workbook's first
 * sheet when omitted) by resolving workbook.xml sheet names through the
 * workbook relationships. Falls back to the first `xl/worksheets/sheet*`
 * entry for files without workbook metadata.
 */
function resolveSheetEntry(entries: ZipEntry[], sheetName?: string): ZipEntry | undefined {
  const fallback = entries.find((e) => e.name.startsWith("xl/worksheets/sheet"));

  const workbook = entries.find((e) => e.name === "xl/workbook.xml");
  if (!workbook) {
    if (sheetName !== undefined) {
      throw new DataValidationError(
        `readXlsx: sheet "${sheetName}" not found (file has no workbook metadata)`
      );
    }
    return fallback;
  }

  const workbookXml = decodeUTF8(workbook.data);
  const sheets: { name: string; rid: string }[] = [];
  const sheetTagRe = /<sheet\b[^>]*\/?>/g;
  let m = sheetTagRe.exec(workbookXml);
  while (m) {
    const tag = m[0];
    const name = getAttr(tag, "name");
    const rid = getAttr(tag, "r:id");
    if (name !== undefined && rid !== undefined) {
      sheets.push({ name: unescapeXml(name), rid });
    }
    m = sheetTagRe.exec(workbookXml);
  }

  const relsEntry = entries.find((e) => e.name === "xl/_rels/workbook.xml.rels");
  const ridToTarget = new Map<string, string>();
  if (relsEntry) {
    const relsXml = decodeUTF8(relsEntry.data);
    const relRe = /<Relationship\b[^>]*\/?>/g;
    let rm = relRe.exec(relsXml);
    while (rm) {
      const id = getAttr(rm[0], "Id");
      const target = getAttr(rm[0], "Target");
      if (id !== undefined && target !== undefined) {
        ridToTarget.set(id, target.replace(/^\//, "").replace(/^xl\//, ""));
      }
      rm = relRe.exec(relsXml);
    }
  }

  const pick = (rid: string): ZipEntry | undefined => {
    const target = ridToTarget.get(rid);
    if (!target) return undefined;
    const full = target.startsWith("worksheets/") ? `xl/${target}` : `xl/${target}`;
    return entries.find((e) => e.name === full);
  };

  if (sheetName === undefined) {
    const first = sheets[0];
    return (first && pick(first.rid)) ?? fallback;
  }

  const wanted = sheets.find((sh) => sh.name === sheetName);
  if (!wanted) {
    throw new DataValidationError(
      `readXlsx: sheet "${sheetName}" not found; available sheets: ` +
        (sheets.map((sh) => sh.name).join(", ") || "(none)")
    );
  }
  return pick(wanted.rid) ?? fallback;
}

/**
 * Read cell data from an XLSX buffer.
 *
 * Returns an object with `columns` (header names) and `data`
 * (array of objects mapping column names to values).
 *
 * @param buffer - The raw .xlsx file as a Uint8Array
 * @param options - Read options
 * @returns Parsed spreadsheet data
 */
export function readXlsx(
  buffer: Uint8Array,
  options: XlsxReadOptions = {}
): { columns: string[]; data: Record<string, string | number | boolean | null>[] } {
  const { header = true } = options;
  const entries = readZip(buffer, options.maxEntryBytes ?? DEFAULT_MAX_ENTRY_BYTES);

  // Extract shared strings
  const sharedStrings: string[] = [];
  const ssiEntry = entries.find((e) => e.name === "xl/sharedStrings.xml");
  if (ssiEntry) {
    const ssiXml = decodeUTF8(ssiEntry.data);
    const siTags = extractTagContent(ssiXml, "si");
    for (const si of siTags) {
      const t = getInnerText(si, "t");
      sharedStrings.push(t !== undefined ? unescapeXml(t) : "");
    }
  }

  // Resolve the requested sheet via workbook.xml + its relationships.
  const sheetEntry = resolveSheetEntry(entries, options.sheet);
  if (!sheetEntry) {
    return { columns: [], data: [] };
  }
  const sheetXml = decodeUTF8(sheetEntry.data);

  // Extract rows
  const rows = extractTagContent(sheetXml, "row");
  const grid: (string | number | boolean | null)[][] = [];

  for (const rowXml of rows) {
    const cells = extractTagContent(rowXml, "c");
    const rowData: (string | number | boolean | null)[] = [];

    for (const cellXml of cells) {
      const ref = getAttr(cellXml, "r") ?? "";
      const type = getAttr(cellXml, "t");
      const valueStr = getInnerText(cellXml, "v");

      const [colIdx] = parseCellRef(ref);

      // Ensure array is large enough
      while (rowData.length <= colIdx) {
        rowData.push(null);
      }

      if (type === "s" && valueStr !== undefined) {
        // Shared string
        const idx = parseInt(valueStr, 10);
        rowData[colIdx] = sharedStrings[idx] ?? "";
      } else if (type === "inlineStr") {
        // Inline string: <c t="inlineStr"><is><t>value</t></is></c>
        const inline = getInnerText(cellXml, "t");
        rowData[colIdx] = inline !== undefined ? unescapeXml(inline) : null;
      } else if (type === "str" && valueStr !== undefined) {
        // Formula string result stored directly in <v>
        rowData[colIdx] = unescapeXml(valueStr);
      } else if (type === "b" && valueStr !== undefined) {
        rowData[colIdx] = valueStr === "1";
      } else if (valueStr !== undefined && valueStr !== "") {
        const num = Number(valueStr);
        rowData[colIdx] = Number.isNaN(num) ? unescapeXml(valueStr) : num;
      } else {
        rowData[colIdx] = null;
      }
    }

    grid.push(rowData);
  }

  if (grid.length === 0) {
    return { columns: [], data: [] };
  }

  // Determine max column count
  let maxCols = 0;
  for (const row of grid) {
    if (row.length > maxCols) maxCols = row.length;
  }

  // Extract headers and data
  let columns: string[];
  let dataRows: (string | number | boolean | null)[][];

  if (header && grid.length > 0) {
    columns = (grid[0] ?? []).map((h, i) => (h === null || h === "" ? `Column${i}` : String(h)));
    // Pad column names
    while (columns.length < maxCols) {
      columns.push(`Column${columns.length}`);
    }
    dataRows = grid.slice(1);
  } else {
    columns = Array.from({ length: maxCols }, (_, i) => `Column${i}`);
    dataRows = grid;
  }

  const data: Record<string, string | number | boolean | null>[] = [];
  for (const row of dataRows) {
    const obj: Record<string, string | number | boolean | null> = {};
    for (let c = 0; c < columns.length; c++) {
      const col = columns[c]!;
      obj[col] = c < row.length ? (row[c] ?? null) : null;
    }
    data.push(obj);
  }

  return { columns, data };
}

// ─── XLSX Writer ─────────────────────────────────────────────────────────────

/** Options for writing an XLSX file. */
export type XlsxWriteOptions = {
  /** Sheet name. Default: "Sheet1". */
  readonly sheetName?: string;
};

/**
 * Write DataFrame-like data to an XLSX buffer.
 *
 * @param columns - Column names (header row)
 * @param data - Array of row objects (column name → value)
 * @param options - Write options
 * @returns Uint8Array containing the .xlsx file
 */
export function writeXlsx(
  columns: readonly string[],
  data: readonly Record<string, string | number | boolean | null | undefined>[],
  options: XlsxWriteOptions = {}
): Uint8Array {
  const sheetName = options.sheetName ?? "Sheet1";

  // Build shared strings table
  const sharedStrings: string[] = [];
  const sharedStringMap = new Map<string, number>();

  function getSSI(s: string): number {
    const existing = sharedStringMap.get(s);
    if (existing !== undefined) return existing;
    const idx = sharedStrings.length;
    sharedStrings.push(s);
    sharedStringMap.set(s, idx);
    return idx;
  }

  // Pre-register header strings
  for (const col of columns) {
    getSSI(col);
  }

  // Build sheet XML rows
  const sheetRows: string[] = [];
  const lastCol = colLetter(columns.length - 1);

  // Header row
  let headerCells = "";
  for (let c = 0; c < columns.length; c++) {
    const ref = `${colLetter(c)}1`;
    const ssi = getSSI(columns[c]!);
    headerCells += `<c r="${ref}" t="s"><v>${ssi}</v></c>`;
  }
  sheetRows.push(`<row r="1">${headerCells}</row>`);

  // Data rows
  for (let r = 0; r < data.length; r++) {
    const row = data[r]!;
    const rowNum = r + 2;
    let cells = "";
    for (let c = 0; c < columns.length; c++) {
      const ref = `${colLetter(c)}${rowNum}`;
      const value = row[columns[c]!];

      if (value === null || value === undefined) {
        // Empty cells are omitted; readers report them as null.
      } else if (typeof value === "boolean") {
        cells += `<c r="${ref}" t="b"><v>${value ? 1 : 0}</v></c>`;
      } else if (typeof value === "number") {
        cells += `<c r="${ref}"><v>${value}</v></c>`;
      } else {
        const ssi = getSSI(String(value));
        cells += `<c r="${ref}" t="s"><v>${ssi}</v></c>`;
      }
    }
    sheetRows.push(`<row r="${rowNum}">${cells}</row>`);
  }

  const lastRow = data.length + 1;
  const dimension = `A1:${lastCol}${lastRow}`;

  // Build XML files
  const contentTypes = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>
</Types>`;

  const rels = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>`;

  const workbookRels = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/>
</Relationships>`;

  const workbook = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="${escapeXml(sheetName)}" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>`;

  const sheetXml = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <dimension ref="${dimension}"/>
  <sheetData>
${sheetRows.join("\n")}
  </sheetData>
</worksheet>`;

  let ssiXml = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="${sharedStrings.length}" uniqueCount="${sharedStrings.length}">`;
  for (const s of sharedStrings) {
    ssiXml += `<si><t>${escapeXml(s)}</t></si>`;
  }
  ssiXml += "</sst>";

  const files: ZipEntry[] = [
    { name: "[Content_Types].xml", data: encodeUTF8(contentTypes) },
    { name: "_rels/.rels", data: encodeUTF8(rels) },
    { name: "xl/_rels/workbook.xml.rels", data: encodeUTF8(workbookRels) },
    { name: "xl/workbook.xml", data: encodeUTF8(workbook) },
    { name: "xl/worksheets/sheet1.xml", data: encodeUTF8(sheetXml) },
    { name: "xl/sharedStrings.xml", data: encodeUTF8(ssiXml) },
  ];

  return createZip(files);
}
