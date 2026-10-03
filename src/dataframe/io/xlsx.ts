/**
 * Zero-dependency Excel (.xlsx) reader and writer for Deepbox DataFrames.
 *
 * XLSX files are ZIP archives containing Office Open XML spreadsheet files.
 * This module implements a minimal ZIP reader/writer and XML parser/serializer
 * to handle common spreadsheet data without any external dependencies.
 *
 * Supports:
 * - Reading: cell values (strings, numbers, booleans) from any sheet, chosen by
 *   name or position; shared, inline and rich-text strings; sparse rows and
 *   columns; stored and deflate-compressed archives (deflate needs Node.js)
 * - Writing: valid .xlsx files with a single worksheet
 *
 * Limitations:
 * - One sheet per call (no multi-sheet writing)
 * - No formula evaluation (cached formula results are read)
 * - No cell formatting/styling
 * - No merged cells
 * - No date type: date cells are numbers (Excel serial dates) on read, and `Date`
 *   values are written as ISO 8601 text
 *
 * @module dataframe/io/xlsx
 * @see {@link https://deepbox.dev/docs/dataframe-overview | Deepbox documentation}
 */

import { DataValidationError, InvalidParameterError } from "../../core";

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

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder("utf-8");

function encodeUTF8(str: string): Uint8Array {
  return textEncoder.encode(str);
}

function decodeUTF8(data: Uint8Array): string {
  return textDecoder.decode(data);
}

function writeUint16LE(value: number): Uint8Array {
  return new Uint8Array([value & 0xff, (value >> 8) & 0xff]);
}

function writeUint32LE(value: number): Uint8Array {
  return new Uint8Array([
    value & 0xff,
    (value >> 8) & 0xff,
    (value >> 16) & 0xff,
    (value >>> 24) & 0xff,
  ]);
}

function readUint16LE(data: Uint8Array, offset: number): number {
  return data[offset]! | (data[offset + 1]! << 8);
}

function readUint32LE(data: Uint8Array, offset: number): number {
  return (
    data[offset]! +
    data[offset + 1]! * 0x100 +
    data[offset + 2]! * 0x10000 +
    data[offset + 3]! * 0x1000000
  );
}

function concatBytes(arrays: readonly Uint8Array[]): Uint8Array {
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

/** MS-DOS date for 1980-01-01 (the zip epoch); a zero date is invalid in some tools. */
const DOS_DATE_EPOCH = 0x0021;
const ZIP_LIMIT = 0xffffffff;

function createZip(entries: ZipEntry[]): Uint8Array {
  const parts: Uint8Array[] = [];
  const centralDir: Uint8Array[] = [];
  let localOffset = 0;

  for (const entry of entries) {
    const nameBytes = encodeUTF8(entry.name);
    const crc = crc32(entry.data);
    const size = entry.data.length;
    if (size >= ZIP_LIMIT) {
      throw new DataValidationError(
        `writeXlsx: "${entry.name}" is too large for a zip archive without zip64 support`
      );
    }

    // Local file header
    const localHeader = concatBytes([
      new Uint8Array([0x50, 0x4b, 0x03, 0x04]), // signature
      writeUint16LE(20), // version needed
      writeUint16LE(0x0800), // flags: UTF-8 names
      writeUint16LE(0), // compression: stored
      writeUint16LE(0), // mod time
      writeUint16LE(DOS_DATE_EPOCH), // mod date
      writeUint32LE(crc),
      writeUint32LE(size),
      writeUint32LE(size),
      writeUint16LE(nameBytes.length),
      writeUint16LE(0), // extra field length
      nameBytes,
    ]);

    parts.push(localHeader);
    parts.push(entry.data);

    // Central directory entry
    const cdEntry = concatBytes([
      new Uint8Array([0x50, 0x4b, 0x01, 0x02]),
      writeUint16LE(20), // version made by
      writeUint16LE(20), // version needed
      writeUint16LE(0x0800), // flags: UTF-8 names
      writeUint16LE(0), // compression
      writeUint16LE(0), // mod time
      writeUint16LE(DOS_DATE_EPOCH), // mod date
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
      nameBytes,
    ]);
    centralDir.push(cdEntry);

    localOffset += localHeader.length + entry.data.length;
    if (localOffset >= ZIP_LIMIT) {
      throw new DataValidationError("writeXlsx: the archive is too large without zip64 support");
    }
  }

  let cdSize = 0;
  for (const cd of centralDir) cdSize += cd.length;

  // End of central directory
  const eocd = concatBytes([
    new Uint8Array([0x50, 0x4b, 0x05, 0x06]),
    writeUint16LE(0), // disk number
    writeUint16LE(0), // central dir disk
    writeUint16LE(entries.length),
    writeUint16LE(entries.length),
    writeUint32LE(cdSize),
    writeUint32LE(localOffset),
    writeUint16LE(0), // comment length
  ]);

  return concatBytes([...parts, ...centralDir, eocd]);
}

/** Default cap on a single decompressed zip entry (guards against zip bombs). */
const DEFAULT_MAX_ENTRY_BYTES = 256 * 1024 * 1024;

/**
 * Inflate a raw-deflate stream using Node's zlib (accessed via
 * `process.getBuiltinModule` so browser bundles stay import-free).
 */
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
      "readXlsx: this file uses deflate-compressed zip entries, which require a Node.js runtime " +
        "(20.16+ or 22.3+) to read"
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
    throw new DataValidationError(`readXlsx: zip entry "${entryName}" is corrupt`, {
      cause: err,
    });
  }
}

type ZipDirectoryEntry = {
  name: string;
  method: number;
  compressedSize: number;
  localOffset: number;
};

/** A parsed zip central directory; entries are decompressed lazily on request. */
class ZipArchive {
  private readonly entries: ZipDirectoryEntry[] = [];

  constructor(
    private readonly data: Uint8Array,
    private readonly maxEntryBytes: number
  ) {
    const notZip = (reason: string): DataValidationError =>
      new DataValidationError(`readXlsx: not a valid .xlsx file (${reason})`);

    // Find the End of Central Directory record (it may be followed by a comment).
    let eocdOffset = -1;
    const lowest = Math.max(0, data.length - 22 - 0xffff);
    for (let i = data.length - 22; i >= lowest; i--) {
      if (
        data[i] === 0x50 &&
        data[i + 1] === 0x4b &&
        data[i + 2] === 0x05 &&
        data[i + 3] === 0x06
      ) {
        eocdOffset = i;
        break;
      }
    }
    if (eocdOffset < 0) throw notZip("no zip directory found");

    const numEntries = readUint16LE(data, eocdOffset + 10);
    const cdOffset = readUint32LE(data, eocdOffset + 16);
    if (cdOffset === ZIP_LIMIT) throw notZip("zip64 archives are not supported");

    let pos = cdOffset;
    for (let e = 0; e < numEntries; e++) {
      if (
        pos + 46 > data.length ||
        data[pos] !== 0x50 ||
        data[pos + 1] !== 0x4b ||
        data[pos + 2] !== 0x01 ||
        data[pos + 3] !== 0x02
      ) {
        throw notZip("corrupt zip directory");
      }

      const flags = readUint16LE(data, pos + 8);
      const method = readUint16LE(data, pos + 10);
      const compressedSize = readUint32LE(data, pos + 20);
      const nameLen = readUint16LE(data, pos + 28);
      const extraLen = readUint16LE(data, pos + 30);
      const commentLen = readUint16LE(data, pos + 32);
      const localOffset = readUint32LE(data, pos + 42);
      if (pos + 46 + nameLen > data.length) throw notZip("corrupt zip directory");

      const name = decodeUTF8(data.subarray(pos + 46, pos + 46 + nameLen));
      if ((flags & 1) !== 0) {
        throw new DataValidationError(
          `readXlsx: zip entry "${name}" is encrypted; password-protected files are not supported`
        );
      }
      this.entries.push({ name, method, compressedSize, localOffset });
      pos += 46 + nameLen + extraLen + commentLen;
    }
  }

  names(): string[] {
    return this.entries.map((e) => e.name);
  }

  has(name: string): boolean {
    return this.entries.some((e) => e.name === name);
  }

  /** Decompress and return an entry, or `undefined` when it is not in the archive. */
  read(name: string): Uint8Array | undefined {
    const entry = this.entries.find((e) => e.name === name);
    if (!entry) return undefined;
    const data = this.data;
    const { method, compressedSize, localOffset } = entry;

    if (
      localOffset + 30 > data.length ||
      data[localOffset] !== 0x50 ||
      data[localOffset + 1] !== 0x4b ||
      data[localOffset + 2] !== 0x03 ||
      data[localOffset + 3] !== 0x04
    ) {
      throw new DataValidationError(`readXlsx: zip entry "${name}" has a corrupt local header`);
    }
    if (compressedSize > this.maxEntryBytes) {
      throw new DataValidationError(
        `readXlsx: zip entry "${name}" exceeds the ${this.maxEntryBytes}-byte limit; ` +
          "raise maxEntryBytes explicitly if the file is trusted"
      );
    }
    const localNameLen = readUint16LE(data, localOffset + 26);
    const localExtraLen = readUint16LE(data, localOffset + 28);
    const dataStart = localOffset + 30 + localNameLen + localExtraLen;
    if (dataStart + compressedSize > data.length) {
      throw new DataValidationError(`readXlsx: zip entry "${name}" is truncated`);
    }
    const rawData = data.subarray(dataStart, dataStart + compressedSize);

    if (method === 0) return rawData;
    if (method === 8) return inflateRaw(rawData, this.maxEntryBytes, name);
    throw new DataValidationError(
      `readXlsx: unsupported zip compression method ${method} for entry "${name}"`
    );
  }
}

// ─── XML Utilities ───────────────────────────────────────────────────────────

/** Characters XML 1.0 cannot carry; Excel stores them as `_xHHHH_` escapes. */
// biome-ignore lint/suspicious/noControlCharactersInRegex: matching the control range is the point
const XML_ILLEGAL_CHARS = /[\u0000-\u0008\u000B-\u001F￾￿]/g;

/**
 * Escape text for an XML element or attribute. Characters XML cannot carry
 * (control characters, including a literal CR which parsers would normalise)
 * use Excel's `_xHHHH_` escape, and text that already looks like such an
 * escape has its underscore escaped so it reads back unchanged.
 */
function escapeXml(s: string): string {
  return s
    .replace(/_x(?=[0-9A-Fa-f]{4}_)/g, "_x005F_x")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(
      XML_ILLEGAL_CHARS,
      (c) => `_x${c.charCodeAt(0).toString(16).toUpperCase().padStart(4, "0")}_`
    );
}

/** Decode the standard XML entities and numeric character references in one pass. */
function unescapeXml(s: string): string {
  if (!s.includes("&")) return s;
  return s.replace(
    /&(?:#x([0-9a-fA-F]+)|#(\d+)|(lt|gt|amp|quot|apos));/g,
    (match, hex: string | undefined, dec: string | undefined, name: string | undefined) => {
      if (name !== undefined) {
        switch (name) {
          case "lt":
            return "<";
          case "gt":
            return ">";
          case "amp":
            return "&";
          case "quot":
            return '"';
          default:
            return "'";
        }
      }
      const code = hex !== undefined ? parseInt(hex, 16) : Number(dec);
      return code <= 0x10ffff ? String.fromCodePoint(code) : match;
    }
  );
}

/** Decode Excel's `_xHHHH_` escapes (applied after XML unescaping). */
function decodeXstring(s: string): string {
  if (!s.includes("_x")) return s;
  return s.replace(/_x([0-9A-Fa-f]{4})_/g, (_, hex: string) =>
    String.fromCharCode(parseInt(hex, 16))
  );
}

function decodeText(raw: string): string {
  return decodeXstring(unescapeXml(raw));
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

/**
 * Parse the column part of a cell reference (e.g. "AB12" gives 27). Returns -1
 * for malformed references and `MAX_COLUMNS` for columns beyond Excel's limit.
 */
function parseColumnRef(ref: string): number {
  const match = /^([A-Za-z]+)\d*$/.exec(ref);
  if (!match) return -1;
  const letters = match[1]!.toUpperCase();
  let col = 0;
  for (let i = 0; i < letters.length; i++) {
    col = col * 26 + (letters.charCodeAt(i) - 64);
    if (col > MAX_COLUMNS) return MAX_COLUMNS; // out of range; the caller rejects it
  }
  return col - 1;
}

/** Excel's worksheet limits (XFD columns, 1,048,576 rows). */
const MAX_COLUMNS = 16384;
const MAX_ROWS = 1048576;
/** Excel's cap on the characters in one cell. */
const MAX_CELL_CHARS = 32767;

/** Read the attributes of an XML start tag body (the text between the tag name and `>`). */
function parseAttributes(attrs: string): Record<string, string> {
  const out: Record<string, string> = Object.create(null) as Record<string, string>;
  const re = /([\w:.-]+)\s*=\s*(?:"([^"]*)"|'([^']*)')/g;
  let m = re.exec(attrs);
  while (m) {
    out[m[1]!] = m[2] ?? m[3] ?? "";
    m = re.exec(attrs);
  }
  return out;
}

/**
 * Concatenate the text runs (`<t>` elements) of a shared or inline string,
 * covering plain, rich-text and `xml:space="preserve"` forms. Phonetic
 * guides (`<rPh>`) are not part of the cell text.
 */
function stringItemText(body: string): string {
  const source = body.includes("<rPh") ? body.replace(/<rPh\b[\s\S]*?<\/rPh>/g, "") : body;
  let text = "";
  const re = /<t\b[^>]*?(?:\/>|>([\s\S]*?)<\/t>)/g;
  let m = re.exec(source);
  while (m) {
    if (m[1] !== undefined) text += m[1];
    m = re.exec(source);
  }
  return decodeText(text);
}

// ─── XLSX Reader ─────────────────────────────────────────────────────────────

/** A cell value as returned by {@link readXlsx}. */
export type XlsxCell = string | number | boolean | null;

/** Result of {@link readXlsx}: the header names and one record per row. */
export type XlsxReadResult = {
  columns: string[];
  data: Record<string, XlsxCell>[];
};

/** Options for reading an XLSX file. */
export type XlsxReadOptions = {
  /**
   * Sheet to read: a sheet name, or a 0-based position in workbook order.
   * Default: the first sheet.
   */
  readonly sheet?: string | number;
  /** Whether the first row contains headers. Default: true. */
  readonly header?: boolean;
  /**
   * Maximum decompressed size per zip entry in bytes (guards against
   * decompression bombs in untrusted files). Default: 256 MiB.
   */
  readonly maxEntryBytes?: number;
};

/** Resolve a workbook-relationship target to a zip entry name. */
function resolvePartName(target: string): string {
  const parts: string[] = [];
  const full = target.startsWith("/") ? target.slice(1) : `xl/${target}`;
  for (const seg of full.split("/")) {
    if (seg === "..") parts.pop();
    else if (seg !== "." && seg !== "") parts.push(seg);
  }
  return parts.join("/");
}

/**
 * Locate the worksheet XML for `sheet` (name or position; the workbook's
 * first sheet when omitted) by resolving workbook.xml sheet names through the
 * workbook relationships. Falls back to the first `xl/worksheets/sheet*`
 * entry for files without workbook metadata.
 */
function resolveSheetXml(zip: ZipArchive, sheet?: string | number): Uint8Array | undefined {
  // Only inflated when the workbook metadata cannot name a sheet.
  const readFallback = (): Uint8Array | undefined => {
    const fallbackName = zip.names().find((n) => n.startsWith("xl/worksheets/sheet"));
    return fallbackName === undefined ? undefined : zip.read(fallbackName);
  };

  const workbook = zip.read("xl/workbook.xml");
  if (!workbook) {
    if (sheet !== undefined) {
      throw new DataValidationError(
        `readXlsx: sheet ${JSON.stringify(sheet)} not found (file has no workbook metadata)`
      );
    }
    return readFallback();
  }

  const workbookXml = decodeUTF8(workbook);
  const sheets: { name: string; rid: string }[] = [];
  const sheetTagRe = /<sheet\b([^>]*?)\/?>/g;
  let m = sheetTagRe.exec(workbookXml);
  while (m) {
    const attrs = parseAttributes(m[1]!);
    const name = attrs["name"];
    const ridKey = Object.keys(attrs).find((k) => k === "id" || k.endsWith(":id"));
    const rid = ridKey === undefined ? undefined : attrs[ridKey];
    if (name !== undefined && rid !== undefined) {
      sheets.push({ name: decodeText(name), rid });
    }
    m = sheetTagRe.exec(workbookXml);
  }

  const relsXml = zip.read("xl/_rels/workbook.xml.rels");
  const ridToTarget = new Map<string, string>();
  if (relsXml) {
    const text = decodeUTF8(relsXml);
    const relRe = /<Relationship\b([^>]*?)\/?>/g;
    let rm = relRe.exec(text);
    while (rm) {
      const attrs = parseAttributes(rm[1]!);
      const id = attrs["Id"];
      const target = attrs["Target"];
      if (id !== undefined && target !== undefined) ridToTarget.set(id, unescapeXml(target));
      rm = relRe.exec(text);
    }
  }

  const load = (rid: string, label: string): Uint8Array => {
    const target = ridToTarget.get(rid);
    const partName = target === undefined ? undefined : resolvePartName(target);
    const part = partName === undefined ? undefined : zip.read(partName);
    if (!part) {
      throw new DataValidationError(`readXlsx: the worksheet part for sheet ${label} is missing`);
    }
    return part;
  };

  if (sheet === undefined) {
    const first = sheets[0];
    return first ? load(first.rid, JSON.stringify(first.name)) : readFallback();
  }

  const available = sheets.map((sh) => sh.name).join(", ") || "(none)";
  if (typeof sheet === "number") {
    const wantedByIndex = Number.isInteger(sheet) && sheet >= 0 ? sheets[sheet] : undefined;
    if (!wantedByIndex) {
      throw new DataValidationError(
        `readXlsx: sheet index ${sheet} is out of range; available sheets: ${available}`
      );
    }
    return load(wantedByIndex.rid, JSON.stringify(wantedByIndex.name));
  }

  const wanted = sheets.find((sh) => sh.name === sheet);
  if (!wanted) {
    throw new DataValidationError(
      `readXlsx: sheet "${sheet}" not found; available sheets: ${available}`
    );
  }
  return load(wanted.rid, JSON.stringify(wanted.name));
}

/** Zero-based index of the last row of a range reference such as "A1:C7", or -1 when invalid. */
function lastRowOfRange(ref: string): number {
  const last = ref.split(":").pop() ?? "";
  const match = /(\d+)$/.exec(last);
  return match ? parseInt(match[1]!, 10) - 1 : -1;
}

const NUMBER_RE = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/;
const VALUE_RE = /<v(?:\s[^>]*)?>([\s\S]*?)<\/v>/;
const INLINE_RE = /<is>([\s\S]*?)<\/is>/;

function parseCellValue(
  type: string | undefined,
  body: string | undefined,
  sharedStrings: readonly string[]
): XlsxCell {
  if (body === undefined) return null;

  if (type === "inlineStr") {
    const inline = INLINE_RE.exec(body);
    return inline ? stringItemText(inline[1]!) : null;
  }

  const raw = VALUE_RE.exec(body)?.[1];
  if (raw === undefined) return null;

  switch (type) {
    case "s": {
      const idx = parseInt(raw, 10);
      const str = sharedStrings[idx];
      if (str === undefined) {
        throw new DataValidationError(`readXlsx: shared string index ${raw} is out of range`);
      }
      return str;
    }
    case "str":
    case "d":
      return decodeText(raw);
    case "b":
      return raw === "1" || raw === "true";
    case "e":
      // Formula errors (#DIV/0!, #N/A, ...) carry no value.
      return null;
    default: {
      const trimmed = raw.trim();
      if (trimmed === "") return null;
      return NUMBER_RE.test(trimmed) ? Number(trimmed) : decodeText(raw);
    }
  }
}

/** Make header names unique the way pandas does: `a`, `a.1`, `a.2`. */
function dedupeHeaders(names: string[]): string[] {
  const used = new Set(names);
  const seen = new Map<string, number>();
  const first = new Set<string>();
  return names.map((name) => {
    if (!first.has(name)) {
      first.add(name);
      return name;
    }
    let k = seen.get(name) ?? 0;
    let candidate: string;
    do {
      k++;
      candidate = `${name}.${k}`;
    } while (used.has(candidate));
    seen.set(name, k);
    used.add(candidate);
    return candidate;
  });
}

/** Set `obj[key]` without letting a `"__proto__"` column rewrite the prototype. */
function setOwn(obj: Record<string, XlsxCell>, key: string, value: XlsxCell): void {
  if (key === "__proto__") {
    Object.defineProperty(obj, key, {
      value,
      enumerable: true,
      writable: true,
      configurable: true,
    });
  } else {
    obj[key] = value;
  }
}

/**
 * Read cell data from an XLSX buffer.
 *
 * Returns an object with `columns` (header names) and `data`
 * (array of objects mapping column names to values).
 *
 * Empty cells are `null`, formula cells give their cached result (formula
 * errors become `null`), and numeric cells are numbers (dates are Excel serial
 * numbers). Rows keep their sheet positions, so blank rows inside the data stay
 * as rows of `null`; blank rows before the data are dropped. Blank rows after the
 * data are dropped too, except plain empty rows inside the sheet's declared
 * dimension, which is how {@link writeXlsx} stores all-null rows. Trailing empty
 * columns are dropped. Blank or missing header cells are named `Column<i>` and
 * repeated headers are suffixed `.1`, `.2`, ... like pandas.
 *
 * @param buffer - The raw .xlsx file as a Uint8Array
 * @param options - Read options
 * @returns Parsed spreadsheet data
 * @throws {DataValidationError} If the buffer is not a valid .xlsx file or the sheet does not exist
 * @throws {InvalidParameterError} If `maxEntryBytes` is not a positive number
 */
export function readXlsx(buffer: Uint8Array, options: XlsxReadOptions = {}): XlsxReadResult {
  if (!(buffer instanceof Uint8Array)) {
    throw new DataValidationError("readXlsx: buffer must be a Uint8Array");
  }
  const { header = true } = options;
  const maxEntryBytes = options.maxEntryBytes ?? DEFAULT_MAX_ENTRY_BYTES;
  if (!(maxEntryBytes > 0) || !Number.isFinite(maxEntryBytes)) {
    throw new InvalidParameterError(
      "readXlsx: maxEntryBytes must be a positive number",
      "maxEntryBytes",
      options.maxEntryBytes
    );
  }
  const zip = new ZipArchive(buffer, maxEntryBytes);

  // Extract shared strings (self-closing <si/> items still occupy an index).
  const sharedStrings: string[] = [];
  const ssBytes = zip.read("xl/sharedStrings.xml");
  if (ssBytes) {
    const ssXml = decodeUTF8(ssBytes);
    const siRe = /<si\b[^>]*?(?:\/>|>([\s\S]*?)<\/si>)/g;
    let m = siRe.exec(ssXml);
    while (m) {
      sharedStrings.push(m[1] === undefined ? "" : stringItemText(m[1]));
      m = siRe.exec(ssXml);
    }
  }

  // Resolve the requested sheet via workbook.xml + its relationships.
  const sheetBytes = resolveSheetXml(zip, options.sheet);
  if (!sheetBytes) {
    return { columns: [], data: [] };
  }
  const sheetXml = decodeUTF8(sheetBytes);

  // Extract rows by their sheet position (empty rows are not stored in the file).
  const rowMap = new Map<number, XlsxCell[]>();
  // Rows that carry cell or row formatting (Excel keeps those for blank styled rows).
  const formattedRows = new Set<number>();
  const rowRe = /<row\b([^>]*?)(?:\/>|>([\s\S]*?)<\/row>)/g;
  let rowCounter = -1;
  let rm = rowRe.exec(sheetXml);
  while (rm) {
    const rowAttrs = parseAttributes(rm[1]!);
    const rowNum = rowAttrs["r"] !== undefined ? parseInt(rowAttrs["r"], 10) - 1 : rowCounter + 1;
    if (!(rowNum >= 0) || rowNum >= MAX_ROWS) {
      throw new DataValidationError(`readXlsx: invalid row reference "${rowAttrs["r"] ?? ""}"`);
    }
    rowCounter = rowNum;

    const cells: XlsxCell[] = rowMap.get(rowNum) ?? [];
    const rowXml = rm[2];
    const bare =
      (rowXml === undefined || rowXml.trim() === "") &&
      Object.keys(rowAttrs).every((k) => k === "r" || k === "spans");
    if (!bare) formattedRows.add(rowNum);
    if (rowXml !== undefined) {
      const cellRe = /<c\b([^>]*?)(?:\/>|>([\s\S]*?)<\/c>)/g;
      let colCounter = -1;
      let cm = cellRe.exec(rowXml);
      while (cm) {
        const attrs = parseAttributes(cm[1]!);
        let colIdx = attrs["r"] !== undefined ? parseColumnRef(attrs["r"]) : -1;
        if (colIdx < 0) colIdx = colCounter + 1;
        if (colIdx >= MAX_COLUMNS) {
          throw new DataValidationError(`readXlsx: invalid cell reference "${attrs["r"] ?? ""}"`);
        }
        colCounter = colIdx;

        const value = parseCellValue(attrs["t"], cm[2], sharedStrings);
        if (value !== null) {
          while (cells.length <= colIdx) cells.push(null);
          cells[colIdx] = value;
        }
        cm = cellRe.exec(rowXml);
      }
    }
    rowMap.set(rowNum, cells);
    rm = rowRe.exec(sheetXml);
  }

  // Rows that hold at least one value delimit the used range.
  const dimension = /<dimension\b[^>]*?\bref\s*=\s*"([^"]*)"/.exec(sheetXml)?.[1];
  const dimensionLastRow = dimension ? lastRowOfRange(dimension) : -1;
  const filled = [...rowMap.entries()]
    .filter(([, cells]) => cells.some((v) => v !== null))
    .sort((a, b) => a[0] - b[0]);
  const firstFilled = filled[0];
  const lastFilled = filled[filled.length - 1];
  if (!firstFilled || !lastFilled) {
    return { columns: [], data: [] };
  }

  // Blank rows after the last value are kept only when they are plain empty <row> elements
  // inside the declared dimension (writeXlsx writes all-null rows that way, so they survive a
  // round trip). Formatted leftovers, which Excel and other tools leave behind, are dropped.
  let lastRow = lastFilled[0];
  for (const r of rowMap.keys()) {
    if (r > lastRow && r <= dimensionLastRow && !formattedRows.has(r)) lastRow = r;
  }

  const grid: XlsxCell[][] = [];
  let maxCols = 0;
  for (let r = firstFilled[0]; r <= lastRow; r++) {
    const cells = rowMap.get(r) ?? [];
    grid.push(cells);
    if (cells.length > maxCols) maxCols = cells.length;
  }

  // Extract headers and data
  let columns: string[];
  let dataRows: XlsxCell[][];

  if (header) {
    const names: string[] = [];
    const headerCells = grid[0] ?? [];
    for (let i = 0; i < maxCols; i++) {
      const h = headerCells[i];
      names.push(h === null || h === undefined || h === "" ? `Column${i}` : String(h));
    }
    columns = dedupeHeaders(names);
    dataRows = grid.slice(1);
  } else {
    columns = Array.from({ length: maxCols }, (_, i) => `Column${i}`);
    dataRows = grid;
  }

  const data: Record<string, string | number | boolean | null>[] = new Array(dataRows.length);
  for (let r = 0; r < dataRows.length; r++) {
    const row = dataRows[r]!;
    const obj: Record<string, XlsxCell> = {};
    for (let c = 0; c < columns.length; c++) setOwn(obj, columns[c]!, row[c] ?? null);
    data[r] = obj;
  }

  return { columns, data };
}

// ─── XLSX Writer ─────────────────────────────────────────────────────────────

/** Options for writing an XLSX file. */
export type XlsxWriteOptions = {
  /**
   * Sheet name. Default: "Sheet1". Excel limits names to 31 characters and
   * forbids `\ / ? * [ ] :` and a leading or trailing apostrophe.
   */
  readonly sheetName?: string;
};

function validateSheetName(name: string): void {
  // biome-ignore lint/suspicious/noControlCharactersInRegex: control characters are rejected on purpose
  const illegal = /[\\/?*[\]:\u0000-\u001F]/;
  if (
    typeof name !== "string" ||
    name.length === 0 ||
    name.length > 31 ||
    illegal.test(name) ||
    name.startsWith("'") ||
    name.endsWith("'")
  ) {
    throw new InvalidParameterError(
      "writeXlsx: sheetName must be 1-31 characters, must not contain \\ / ? * [ ] : or control " +
        "characters, and must not start or end with an apostrophe",
      "sheetName",
      name
    );
  }
}

/** Text element for a string cell, preserving leading/trailing whitespace. */
function textElement(s: string): string {
  if (s.length > MAX_CELL_CHARS) {
    throw new DataValidationError(
      `writeXlsx: a cell holds ${s.length} characters; Excel allows at most ${MAX_CELL_CHARS}`
    );
  }
  const preserve = s !== s.trim() ? ' xml:space="preserve"' : "";
  return `<t${preserve}>${escapeXml(s)}</t>`;
}

/**
 * Write DataFrame-like data to an XLSX buffer.
 *
 * Value mapping: numbers become numeric cells (`NaN` and `null`/`undefined`
 * become empty cells, `Infinity`/`-Infinity` are written as the text
 * `"inf"`/`"-inf"` since Excel has no such numbers), booleans become boolean
 * cells, safe-integer `bigint`s become numbers, `Date`s become ISO 8601 text
 * (invalid dates are empty), and everything else is written as text.
 *
 * @param columns - Column names (header row); must be unique
 * @param data - Array of row objects (column name to value)
 * @param options - Write options
 * @returns Uint8Array containing the .xlsx file
 * @throws {DataValidationError} If column names repeat, exceed Excel's sheet limits, or a text cell is longer than 32767 characters
 * @throws {InvalidParameterError} If `sheetName` is not a valid Excel sheet name
 */
export function writeXlsx(
  columns: readonly string[],
  data: readonly Record<string, unknown>[],
  options: XlsxWriteOptions = {}
): Uint8Array {
  const sheetName = options.sheetName ?? "Sheet1";
  validateSheetName(sheetName);

  if (columns.length > MAX_COLUMNS) {
    throw new DataValidationError(
      `writeXlsx: ${columns.length} columns exceed Excel's limit of ${MAX_COLUMNS}`
    );
  }
  if (data.length + 1 > MAX_ROWS) {
    throw new DataValidationError(
      `writeXlsx: ${data.length + 1} rows (including the header) exceed Excel's limit of ${MAX_ROWS}`
    );
  }
  const seen = new Set<string>();
  for (const col of columns) {
    if (typeof col !== "string") {
      throw new DataValidationError("writeXlsx: column names must be strings");
    }
    if (seen.has(col)) {
      throw new DataValidationError(`writeXlsx: duplicate column name "${col}"`);
    }
    seen.add(col);
  }

  // Build shared strings table
  const sharedStrings: string[] = [];
  const sharedStringMap = new Map<string, number>();
  let stringRefs = 0;

  const getSSI = (s: string): number => {
    stringRefs++;
    const existing = sharedStringMap.get(s);
    if (existing !== undefined) return existing;
    const idx = sharedStrings.length;
    sharedStrings.push(s);
    sharedStringMap.set(s, idx);
    return idx;
  };

  // Build sheet XML rows
  const sheetRows: string[] = [];
  const letters = columns.map((_, c) => colLetter(c));

  // Header row
  let headerCells = "";
  for (let c = 0; c < columns.length; c++) {
    headerCells += `<c r="${letters[c]}1" t="s"><v>${getSSI(columns[c]!)}</v></c>`;
  }
  sheetRows.push(`<row r="1">${headerCells}</row>`);

  // Data rows
  for (let r = 0; r < data.length; r++) {
    const row = data[r];
    if (typeof row !== "object" || row === null) {
      throw new DataValidationError(`writeXlsx: row ${r} is not an object`);
    }
    const rowNum = r + 2;
    let cells = "";
    for (let c = 0; c < columns.length; c++) {
      const name = columns[c]!;
      const value = Object.hasOwn(row, name) ? row[name] : undefined;
      const ref = `${letters[c]}${rowNum}`;

      if (value === null || value === undefined) {
        // Empty cells are omitted; readers report them as null.
      } else if (typeof value === "boolean") {
        cells += `<c r="${ref}" t="b"><v>${value ? 1 : 0}</v></c>`;
      } else if (typeof value === "number") {
        if (Number.isNaN(value)) {
          // NaN marks a missing value: leave the cell empty.
        } else if (Number.isFinite(value)) {
          cells += `<c r="${ref}"><v>${value}</v></c>`;
        } else {
          cells += `<c r="${ref}" t="s"><v>${getSSI(value > 0 ? "inf" : "-inf")}</v></c>`;
        }
      } else if (typeof value === "bigint") {
        if (value >= BigInt(Number.MIN_SAFE_INTEGER) && value <= BigInt(Number.MAX_SAFE_INTEGER)) {
          cells += `<c r="${ref}"><v>${value}</v></c>`;
        } else {
          cells += `<c r="${ref}" t="s"><v>${getSSI(String(value))}</v></c>`;
        }
      } else if (value instanceof Date) {
        if (!Number.isNaN(value.getTime())) {
          cells += `<c r="${ref}" t="s"><v>${getSSI(value.toISOString())}</v></c>`;
        }
      } else {
        cells += `<c r="${ref}" t="s"><v>${getSSI(String(value))}</v></c>`;
      }
    }
    sheetRows.push(`<row r="${rowNum}">${cells}</row>`);
  }

  const lastRow = data.length + 1;
  const dimension = columns.length === 0 ? "A1" : `A1:${letters[columns.length - 1]}${lastRow}`;

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

  const ssiParts: string[] = [
    `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="${stringRefs}" uniqueCount="${sharedStrings.length}">`,
  ];
  for (const s of sharedStrings) ssiParts.push(`<si>${textElement(s)}</si>`);
  ssiParts.push("</sst>");

  const files: ZipEntry[] = [
    { name: "[Content_Types].xml", data: encodeUTF8(contentTypes) },
    { name: "_rels/.rels", data: encodeUTF8(rels) },
    { name: "xl/_rels/workbook.xml.rels", data: encodeUTF8(workbookRels) },
    { name: "xl/workbook.xml", data: encodeUTF8(workbook) },
    { name: "xl/worksheets/sheet1.xml", data: encodeUTF8(sheetXml) },
    { name: "xl/sharedStrings.xml", data: encodeUTF8(ssiParts.join("")) },
  ];

  return createZip(files);
}
