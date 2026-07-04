/**
 * Zero-dependency Parquet reader and writer for Deepbox DataFrames.
 *
 * Implements a spec-compliant subset of Apache Parquet (verified against
 * pyarrow) supporting:
 * - PLAIN encoding for all primitive types (booleans bit-packed per spec)
 * - Flat schemas (no nested/repeated fields)
 * - BOOLEAN, INT32, INT64, FLOAT, DOUBLE, BYTE_ARRAY (UTF8 string) types
 * - Nullable columns (OPTIONAL repetition with RLE definition levels)
 * - Uncompressed data pages (NONE compression), format version 1
 *
 * The reader consumes files this writer produces as well as other writers'
 * uncompressed PLAIN v1 files; dictionary-encoded or compressed files throw
 * a descriptive error instead of returning wrong data.
 *
 * The Parquet format uses Thrift-encoded metadata. This module includes
 * a minimal Thrift Compact Protocol encoder/decoder.
 *
 * @module dataframe/io/parquet
 * @see {@link https://deepbox.dev/docs/dataframe-overview | Deepbox documentation}
 */

import { DataValidationError } from "../../core";

// ─── Parquet Constants ───────────────────────────────────────────────────────

const PARQUET_MAGIC = new Uint8Array([0x50, 0x41, 0x52, 0x31]); // "PAR1"

enum ParquetType {
  BOOLEAN = 0,
  INT32 = 1,
  INT64 = 2,
  FLOAT = 4,
  DOUBLE = 5,
  BYTE_ARRAY = 6,
}

enum Encoding {
  PLAIN = 0,
  RLE = 3,
}

enum CompressionCodec {
  UNCOMPRESSED = 0,
}

enum PageType {
  DATA_PAGE = 0,
  DICTIONARY_PAGE = 2,
}

enum FieldRepetitionType {
  REQUIRED = 0,
  OPTIONAL = 1,
}

/** ConvertedType.UTF8 — annotates BYTE_ARRAY columns as strings. */
const CONVERTED_TYPE_UTF8 = 0;

// Thrift Compact Protocol type codes.
const TC_I32 = 5;
const TC_I64 = 6;
const TC_BINARY = 8;
const TC_LIST = 9;
const TC_STRUCT = 12;

// ─── Thrift Compact Protocol (minimal) ───────────────────────────────────────

class ThriftWriter {
  private parts: Uint8Array[] = [];
  private lastFieldId = 0;

  writeFieldBegin(type: number, id: number): void {
    const delta = id - this.lastFieldId;
    if (delta > 0 && delta <= 15) {
      this.writeByte((delta << 4) | type);
    } else {
      this.writeByte(type);
      this.writeI16(id);
    }
    this.lastFieldId = id;
  }

  writeFieldStop(): void {
    this.writeByte(0);
  }

  writeStructBegin(): void {
    this.lastFieldId = 0;
  }

  writeByte(value: number): void {
    this.parts.push(new Uint8Array([value & 0xff]));
  }

  writeI16(value: number): void {
    this.writeVarint(this.zigzag16(value));
  }

  writeI32(value: number): void {
    this.writeVarint(this.zigzag32(value));
  }

  writeI64(value: bigint): void {
    this.writeVarintBig(this.zigzag64(value));
  }

  writeString(value: string): void {
    const bytes = new TextEncoder().encode(value);
    this.writeVarint(bytes.length);
    this.parts.push(bytes);
  }

  /** Append pre-encoded struct bytes verbatim (no length prefix). */
  writeStructBytes(value: Uint8Array): void {
    this.parts.push(value);
  }

  writeListBegin(elemType: number, size: number): void {
    if (size <= 14) {
      this.writeByte((size << 4) | elemType);
    } else {
      this.writeByte(0xf0 | elemType);
      this.writeVarint(size);
    }
  }

  private writeVarint(value: number): void {
    let v = value >>> 0;
    while (v > 0x7f) {
      this.parts.push(new Uint8Array([(v & 0x7f) | 0x80]));
      v >>>= 7;
    }
    this.parts.push(new Uint8Array([v & 0x7f]));
  }

  private writeVarintBig(value: bigint): void {
    let v = value & 0xffffffffffffffffn;
    while (v > 0x7fn) {
      this.parts.push(new Uint8Array([Number(v & 0x7fn) | 0x80]));
      v >>= 7n;
    }
    this.parts.push(new Uint8Array([Number(v & 0x7fn)]));
  }

  private zigzag16(n: number): number {
    return ((n << 1) ^ (n >> 15)) >>> 0;
  }

  private zigzag32(n: number): number {
    return ((n << 1) ^ (n >> 31)) >>> 0;
  }

  private zigzag64(n: bigint): bigint {
    return ((n << 1n) ^ (n >> 63n)) & 0xffffffffffffffffn;
  }

  toBytes(): Uint8Array {
    let size = 0;
    for (const p of this.parts) size += p.length;
    const result = new Uint8Array(size);
    let offset = 0;
    for (const p of this.parts) {
      result.set(p, offset);
      offset += p.length;
    }
    return result;
  }
}

class ThriftReader {
  private pos = 0;
  private lastFieldId = 0;
  private readonly fieldIdStack: number[] = [];
  private readonly data: Uint8Array;

  constructor(data: Uint8Array) {
    this.data = data;
  }

  get position(): number {
    return this.pos;
  }

  readFieldBegin(): { type: number; id: number } {
    const byte = this.readByte();
    if (byte === 0) return { type: 0, id: 0 }; // STOP

    const delta = (byte >> 4) & 0x0f;
    const type = byte & 0x0f;

    if (delta !== 0) {
      this.lastFieldId += delta;
    } else {
      this.lastFieldId = this.readI16();
    }

    return { type, id: this.lastFieldId };
  }

  readByte(): number {
    return this.data[this.pos++]!;
  }

  readI16(): number {
    return this.unzigzag32(this.readVarint());
  }

  readI32(): number {
    return this.unzigzag32(this.readVarint());
  }

  readI64(): bigint {
    return this.unzigzag64(this.readVarintBig());
  }

  readString(): string {
    const len = this.readVarint();
    const bytes = this.data.subarray(this.pos, this.pos + len);
    this.pos += len;
    return new TextDecoder().decode(bytes);
  }

  readBinary(): Uint8Array {
    const len = this.readVarint();
    const bytes = this.data.slice(this.pos, this.pos + len);
    this.pos += len;
    return bytes;
  }

  readListBegin(): { elemType: number; size: number } {
    const byte = this.readByte();
    const size = (byte >> 4) & 0x0f;
    const elemType = byte & 0x0f;
    if (size === 0x0f) {
      return { elemType, size: this.readVarint() };
    }
    return { elemType, size };
  }

  /** Enter a nested struct: nested field ids start from their own context. */
  enterStruct(): void {
    this.fieldIdStack.push(this.lastFieldId);
    this.lastFieldId = 0;
  }

  /** Leave a nested struct, restoring the parent's field-id context. */
  exitStruct(): void {
    this.lastFieldId = this.fieldIdStack.pop() ?? 0;
  }

  skipField(type: number): void {
    switch (type) {
      case 1: // BOOL_TRUE / BOOL_FALSE
      case 2:
        break;
      case 3: // I8
        this.pos++;
        break;
      case 4: // I16
      case 5: // I32
        this.readVarint();
        break;
      case 6: // I64
        this.readVarintBig();
        break;
      case 7: // DOUBLE
        this.pos += 8;
        break;
      case 8: // BINARY / STRING
        this.readBinary();
        break;
      case 9: // LIST
      case 10: {
        // SET
        const list = this.readListBegin();
        for (let i = 0; i < list.size; i++) this.skipField(list.elemType);
        break;
      }
      case 12: {
        // STRUCT
        this.enterStruct();
        let field = this.readFieldBegin();
        while (field.type !== 0) {
          this.skipField(field.type);
          field = this.readFieldBegin();
        }
        this.exitStruct();
        break;
      }
      default:
        break;
    }
  }

  resetFieldTracking(): void {
    this.lastFieldId = 0;
  }

  private readVarint(): number {
    let result = 0;
    let shift = 0;
    let byte: number;
    do {
      byte = this.data[this.pos++]!;
      result |= (byte & 0x7f) << shift;
      shift += 7;
    } while ((byte & 0x80) !== 0);
    return result >>> 0;
  }

  private readVarintBig(): bigint {
    let result = 0n;
    let shift = 0n;
    let byte: number;
    do {
      byte = this.data[this.pos++]!;
      result |= BigInt(byte & 0x7f) << shift;
      shift += 7n;
    } while ((byte & 0x80) !== 0);
    return result;
  }

  private unzigzag32(n: number): number {
    return ((n >>> 1) ^ -(n & 1)) | 0;
  }

  private unzigzag64(n: bigint): bigint {
    return (n >> 1n) ^ -(n & 1n);
  }
}

// ─── Value encoding ──────────────────────────────────────────────────────────

type ParquetValue = string | number | boolean | bigint | null | undefined;

const INT32_MIN = -2147483648;
const INT32_MAX = 2147483647;

/**
 * Infer the Parquet physical type of a column by scanning every value, so a
 * column like `[1, 2, 1.5]` becomes DOUBLE instead of silently truncating.
 */
function inferColumnType(values: readonly ParquetValue[]): ParquetType {
  let sawValue = false;
  let allBoolean = true;
  let allBigInt = true;
  let allNumber = true;
  let allInteger = true;
  let fitsInt32 = true;

  for (const v of values) {
    if (v === null || v === undefined) continue;
    sawValue = true;
    if (typeof v !== "boolean") allBoolean = false;
    if (typeof v !== "bigint") allBigInt = false;
    if (typeof v === "number") {
      if (!Number.isInteger(v)) {
        allInteger = false;
      } else if (v < INT32_MIN || v > INT32_MAX) {
        fitsInt32 = false;
      }
    } else {
      allNumber = false;
    }
    if (!allBoolean && !allBigInt && !allNumber) return ParquetType.BYTE_ARRAY;
  }

  if (!sawValue) return ParquetType.BYTE_ARRAY;
  if (allBoolean) return ParquetType.BOOLEAN;
  if (allBigInt) return ParquetType.INT64;
  if (!allNumber) return ParquetType.BYTE_ARRAY;
  if (!allInteger) return ParquetType.DOUBLE;
  return fitsInt32 ? ParquetType.INT32 : ParquetType.INT64;
}

/** PLAIN-encode non-null values (booleans bit-packed LSB-first per spec). */
function encodeValuesPlain(values: readonly ParquetValue[], ptype: ParquetType): Uint8Array {
  const nonNull: (string | number | boolean | bigint)[] = [];
  for (const v of values) {
    if (v !== null && v !== undefined) nonNull.push(v);
  }

  switch (ptype) {
    case ParquetType.BOOLEAN: {
      const out = new Uint8Array(Math.ceil(nonNull.length / 8));
      for (let i = 0; i < nonNull.length; i++) {
        if (nonNull[i]) out[i >> 3]! |= 1 << (i & 7);
      }
      return out;
    }
    case ParquetType.INT32: {
      const out = new Uint8Array(nonNull.length * 4);
      const view = new DataView(out.buffer);
      for (let i = 0; i < nonNull.length; i++) {
        view.setInt32(i * 4, Number(nonNull[i]), true);
      }
      return out;
    }
    case ParquetType.INT64: {
      const out = new Uint8Array(nonNull.length * 8);
      const view = new DataView(out.buffer);
      for (let i = 0; i < nonNull.length; i++) {
        const v = nonNull[i];
        view.setBigInt64(i * 8, typeof v === "bigint" ? v : BigInt(Math.trunc(Number(v))), true);
      }
      return out;
    }
    case ParquetType.FLOAT: {
      const out = new Uint8Array(nonNull.length * 4);
      const view = new DataView(out.buffer);
      for (let i = 0; i < nonNull.length; i++) {
        view.setFloat32(i * 4, Number(nonNull[i]), true);
      }
      return out;
    }
    case ParquetType.DOUBLE: {
      const out = new Uint8Array(nonNull.length * 8);
      const view = new DataView(out.buffer);
      for (let i = 0; i < nonNull.length; i++) {
        view.setFloat64(i * 8, Number(nonNull[i]), true);
      }
      return out;
    }
    case ParquetType.BYTE_ARRAY: {
      const parts: Uint8Array[] = [];
      const encoder = new TextEncoder();
      for (const v of nonNull) {
        const bytes = encoder.encode(String(v));
        const lenBuf = new Uint8Array(4);
        new DataView(lenBuf.buffer).setInt32(0, bytes.length, true);
        parts.push(lenBuf, bytes);
      }
      return concat(...parts);
    }
  }
}

/**
 * Encode definition levels (bit width 1) as an RLE/bit-packed hybrid run,
 * prefixed with the 4-byte little-endian length required in data pages v1.
 */
function encodeDefinitionLevels(values: readonly ParquetValue[]): Uint8Array {
  const n = values.length;
  const groups = Math.ceil(n / 8);
  const packed = new Uint8Array(groups);
  for (let i = 0; i < n; i++) {
    if (values[i] !== null && values[i] !== undefined) {
      packed[i >> 3]! |= 1 << (i & 7);
    }
  }
  // Bit-packed run header: varint((numGroups << 1) | 1)
  const headerParts: number[] = [];
  let h = ((groups << 1) | 1) >>> 0;
  while (h > 0x7f) {
    headerParts.push((h & 0x7f) | 0x80);
    h >>>= 7;
  }
  headerParts.push(h);

  const body = concat(new Uint8Array(headerParts), packed);
  const out = new Uint8Array(4 + body.length);
  new DataView(out.buffer).setInt32(0, body.length, true);
  out.set(body, 4);
  return out;
}

/** Decode `count` 1-bit levels from an RLE/bit-packed hybrid buffer. */
function decodeDefinitionLevels(data: Uint8Array, count: number): Uint8Array {
  const out = new Uint8Array(count);
  let pos = 0;
  let produced = 0;

  const readVarint = (): number => {
    let result = 0;
    let shift = 0;
    let byte: number;
    do {
      byte = data[pos++]!;
      result |= (byte & 0x7f) << shift;
      shift += 7;
    } while ((byte & 0x80) !== 0);
    return result >>> 0;
  };

  while (produced < count && pos < data.length) {
    const header = readVarint();
    if (header & 1) {
      // Bit-packed run: (header >> 1) groups of 8 values.
      const groups = header >>> 1;
      for (let g = 0; g < groups && produced < count; g++) {
        const byte = data[pos++] ?? 0;
        for (let b = 0; b < 8 && produced < count; b++) {
          out[produced++] = (byte >> b) & 1;
        }
      }
    } else {
      // RLE run: (header >> 1) copies of a 1-byte value (bit width 1).
      const runLength = header >>> 1;
      const value = data[pos++] ?? 0;
      for (let i = 0; i < runLength && produced < count; i++) {
        out[produced++] = value & 1;
      }
    }
  }
  return out;
}

// ─── Metadata encoding ───────────────────────────────────────────────────────

function writePageHeader(dataSize: number, numValues: number, hasDefLevels: boolean): Uint8Array {
  const tw = new ThriftWriter();
  tw.writeStructBegin();
  tw.writeFieldBegin(TC_I32, 1); // type
  tw.writeI32(PageType.DATA_PAGE);
  tw.writeFieldBegin(TC_I32, 2); // uncompressed_page_size
  tw.writeI32(dataSize);
  tw.writeFieldBegin(TC_I32, 3); // compressed_page_size
  tw.writeI32(dataSize);
  tw.writeFieldBegin(TC_STRUCT, 5); // data_page_header
  {
    const dph = new ThriftWriter();
    dph.writeStructBegin();
    dph.writeFieldBegin(TC_I32, 1); // num_values (incl. nulls)
    dph.writeI32(numValues);
    dph.writeFieldBegin(TC_I32, 2); // encoding
    dph.writeI32(Encoding.PLAIN);
    dph.writeFieldBegin(TC_I32, 3); // definition_level_encoding
    dph.writeI32(hasDefLevels ? Encoding.RLE : Encoding.PLAIN);
    dph.writeFieldBegin(TC_I32, 4); // repetition_level_encoding
    dph.writeI32(Encoding.RLE);
    dph.writeFieldStop();
    tw.writeStructBytes(dph.toBytes());
  }
  tw.writeFieldStop();
  return tw.toBytes();
}

function writeColumnMetaData(
  ptype: ParquetType,
  name: string,
  numValues: number,
  pageOffset: number,
  totalSize: number,
  optional: boolean
): Uint8Array {
  const tw = new ThriftWriter();
  tw.writeStructBegin();
  tw.writeFieldBegin(TC_I32, 1); // type
  tw.writeI32(ptype);
  tw.writeFieldBegin(TC_LIST, 2); // encodings
  tw.writeListBegin(TC_I32, optional ? 2 : 1);
  tw.writeI32(Encoding.PLAIN);
  if (optional) tw.writeI32(Encoding.RLE);
  tw.writeFieldBegin(TC_LIST, 3); // path_in_schema
  tw.writeListBegin(TC_BINARY, 1);
  tw.writeString(name);
  tw.writeFieldBegin(TC_I32, 4); // codec
  tw.writeI32(CompressionCodec.UNCOMPRESSED);
  tw.writeFieldBegin(TC_I64, 5); // num_values
  tw.writeI64(BigInt(numValues));
  tw.writeFieldBegin(TC_I64, 6); // total_uncompressed_size
  tw.writeI64(BigInt(totalSize));
  tw.writeFieldBegin(TC_I64, 7); // total_compressed_size
  tw.writeI64(BigInt(totalSize));
  tw.writeFieldBegin(TC_I64, 9); // data_page_offset
  tw.writeI64(BigInt(pageOffset));
  tw.writeFieldStop();
  return tw.toBytes();
}

// ─── Parquet Writer ──────────────────────────────────────────────────────────

/** Options for writing a Parquet file. */
export type ParquetWriteOptions = {
  /** Created-by string. Default: "deepbox" */
  readonly createdBy?: string;
};

/**
 * Write DataFrame-like data to a Parquet buffer.
 *
 * Columns containing `null`/`undefined` are written as OPTIONAL with RLE
 * definition levels, so nulls survive the round-trip (also when read by
 * external tools such as pyarrow/pandas).
 *
 * @param columns - Column names
 * @param data - Array of row objects
 * @param options - Write options
 * @returns Uint8Array containing the .parquet file
 */
export function writeParquet(
  columns: readonly string[],
  data: readonly Record<string, unknown>[],
  options: ParquetWriteOptions = {}
): Uint8Array {
  const numRows = data.length;

  const columnValues: ParquetValue[][] = columns.map((col) =>
    data.map((row) => row[col] as ParquetValue)
  );
  const ptypes = columnValues.map((values) => inferColumnType(values));
  const optional = columnValues.map((values) => values.some((v) => v === null || v === undefined));

  // Encode column chunks sequentially after the 4-byte magic.
  const columnChunks: Uint8Array[] = [];
  const columnOffsets: number[] = [];
  let currentOffset = 4;

  for (let c = 0; c < columns.length; c++) {
    const values = columnValues[c]!;
    const encoded = encodeValuesPlain(values, ptypes[c]!);
    const pageData = optional[c] ? concat(encodeDefinitionLevels(values), encoded) : encoded;
    const pageHeader = writePageHeader(pageData.length, numRows, optional[c]!);

    columnOffsets.push(currentOffset);
    const chunk = concat(pageHeader, pageData);
    columnChunks.push(chunk);
    currentOffset += chunk.length;
  }

  // RowGroup
  const rgTw = new ThriftWriter();
  rgTw.writeStructBegin();
  rgTw.writeFieldBegin(TC_LIST, 1); // columns: list<ColumnChunk>
  rgTw.writeListBegin(TC_STRUCT, columns.length);
  for (let c = 0; c < columns.length; c++) {
    const cc = new ThriftWriter();
    cc.writeStructBegin();
    cc.writeFieldBegin(TC_I64, 2); // file_offset
    cc.writeI64(BigInt(columnOffsets[c]!));
    cc.writeFieldBegin(TC_STRUCT, 3); // meta_data
    cc.writeStructBytes(
      writeColumnMetaData(
        ptypes[c]!,
        columns[c]!,
        numRows,
        columnOffsets[c]!,
        columnChunks[c]!.length,
        optional[c]!
      )
    );
    cc.writeFieldStop();
    rgTw.writeStructBytes(cc.toBytes());
  }
  let totalColSize = 0;
  for (const ch of columnChunks) totalColSize += ch.length;
  rgTw.writeFieldBegin(TC_I64, 2); // total_byte_size
  rgTw.writeI64(BigInt(totalColSize));
  rgTw.writeFieldBegin(TC_I64, 3); // num_rows
  rgTw.writeI64(BigInt(numRows));
  rgTw.writeFieldStop();

  // FileMetaData
  const fmTw = new ThriftWriter();
  fmTw.writeStructBegin();
  fmTw.writeFieldBegin(TC_I32, 1); // version
  fmTw.writeI32(1);
  fmTw.writeFieldBegin(TC_LIST, 2); // schema
  fmTw.writeListBegin(TC_STRUCT, columns.length + 1);
  {
    const root = new ThriftWriter();
    root.writeStructBegin();
    root.writeFieldBegin(TC_BINARY, 4); // name
    root.writeString("schema");
    root.writeFieldBegin(TC_I32, 5); // num_children
    root.writeI32(columns.length);
    root.writeFieldStop();
    fmTw.writeStructBytes(root.toBytes());
  }
  for (let c = 0; c < columns.length; c++) {
    const el = new ThriftWriter();
    el.writeStructBegin();
    el.writeFieldBegin(TC_I32, 1); // type
    el.writeI32(ptypes[c]!);
    el.writeFieldBegin(TC_I32, 3); // repetition_type
    el.writeI32(optional[c] ? FieldRepetitionType.OPTIONAL : FieldRepetitionType.REQUIRED);
    el.writeFieldBegin(TC_BINARY, 4); // name
    el.writeString(columns[c]!);
    if (ptypes[c] === ParquetType.BYTE_ARRAY) {
      el.writeFieldBegin(TC_I32, 6); // converted_type = UTF8
      el.writeI32(CONVERTED_TYPE_UTF8);
    }
    el.writeFieldStop();
    fmTw.writeStructBytes(el.toBytes());
  }
  fmTw.writeFieldBegin(TC_I64, 3); // num_rows
  fmTw.writeI64(BigInt(numRows));
  fmTw.writeFieldBegin(TC_LIST, 4); // row_groups
  fmTw.writeListBegin(TC_STRUCT, 1);
  fmTw.writeStructBytes(rgTw.toBytes());
  fmTw.writeFieldBegin(TC_BINARY, 6); // created_by
  fmTw.writeString(options.createdBy ?? "deepbox");
  fmTw.writeFieldStop();

  const metaBytes = fmTw.toBytes();
  const metaLen = new Uint8Array(4);
  new DataView(metaLen.buffer).setInt32(0, metaBytes.length, true);

  return concat(PARQUET_MAGIC, ...columnChunks, metaBytes, metaLen, PARQUET_MAGIC);
}

// ─── Parquet Reader ──────────────────────────────────────────────────────────

/** Options for reading a Parquet file. */
export type ParquetReadOptions = {
  /** Columns to read. Default: all columns. */
  readonly columns?: readonly string[];
};

type SchemaColumn = {
  name: string;
  type: ParquetType;
  optional: boolean;
  utf8: boolean;
};

type ColumnChunkInfo = {
  pageOffset: number;
  numValues: number;
  codec: number;
};

function parseSchema(reader: ThriftReader, size: number): SchemaColumn[] {
  const cols: SchemaColumn[] = [];
  for (let i = 0; i < size; i++) {
    reader.enterStruct();
    let elemName = "";
    let elemType = -1;
    let repetition = FieldRepetitionType.REQUIRED;
    let converted = -1;
    let numChildren = -1;
    let sf = reader.readFieldBegin();
    while (sf.type !== 0) {
      switch (sf.id) {
        case 1:
          elemType = reader.readI32();
          break;
        case 3:
          repetition = reader.readI32();
          break;
        case 4:
          elemName = reader.readString();
          break;
        case 5:
          numChildren = reader.readI32();
          break;
        case 6:
          converted = reader.readI32();
          break;
        default:
          reader.skipField(sf.type);
      }
      sf = reader.readFieldBegin();
    }
    reader.exitStruct();
    // Skip group elements (e.g. the root, which has num_children).
    if (numChildren < 0 && elemName) {
      cols.push({
        name: elemName,
        type: elemType >= 0 ? (elemType as ParquetType) : ParquetType.BYTE_ARRAY,
        optional: repetition === FieldRepetitionType.OPTIONAL,
        utf8: converted === CONVERTED_TYPE_UTF8,
      });
    }
  }
  return cols;
}

function parseColumnChunk(reader: ThriftReader): ColumnChunkInfo {
  reader.enterStruct();
  let pageOffset = -1;
  let numValues = 0;
  let codec = CompressionCodec.UNCOMPRESSED;
  let f = reader.readFieldBegin();
  while (f.type !== 0) {
    if (f.id === 3 && f.type === TC_STRUCT) {
      // meta_data
      reader.enterStruct();
      let mf = reader.readFieldBegin();
      while (mf.type !== 0) {
        switch (mf.id) {
          case 4:
            codec = reader.readI32();
            break;
          case 5:
            numValues = Number(reader.readI64());
            break;
          case 9:
            pageOffset = Number(reader.readI64());
            break;
          case 11: // dictionary_page_offset — page data starts there instead
            pageOffset = Number(reader.readI64());
            break;
          default:
            reader.skipField(mf.type);
        }
        mf = reader.readFieldBegin();
      }
      reader.exitStruct();
    } else {
      reader.skipField(f.type);
    }
    f = reader.readFieldBegin();
  }
  reader.exitStruct();
  return { pageOffset, numValues, codec };
}

/** Decode `count` PLAIN values of `ptype` starting at `pos`. */
function decodeValuesPlain(
  buffer: Uint8Array,
  pos: number,
  count: number,
  ptype: ParquetType
): unknown[] {
  const values: unknown[] = [];
  let readPos = pos;
  let bitPos = 0;
  for (let r = 0; r < count; r++) {
    switch (ptype) {
      case ParquetType.BOOLEAN: {
        const byte = buffer[readPos + (bitPos >> 3)] ?? 0;
        values.push(((byte >> (bitPos & 7)) & 1) !== 0);
        bitPos++;
        break;
      }
      case ParquetType.INT32: {
        const v = new DataView(buffer.buffer, buffer.byteOffset + readPos, 4);
        values.push(v.getInt32(0, true));
        readPos += 4;
        break;
      }
      case ParquetType.INT64: {
        const v = new DataView(buffer.buffer, buffer.byteOffset + readPos, 8);
        values.push(Number(v.getBigInt64(0, true)));
        readPos += 8;
        break;
      }
      case ParquetType.FLOAT: {
        const v = new DataView(buffer.buffer, buffer.byteOffset + readPos, 4);
        values.push(v.getFloat32(0, true));
        readPos += 4;
        break;
      }
      case ParquetType.DOUBLE: {
        const v = new DataView(buffer.buffer, buffer.byteOffset + readPos, 8);
        values.push(v.getFloat64(0, true));
        readPos += 8;
        break;
      }
      case ParquetType.BYTE_ARRAY: {
        const strLen = new DataView(buffer.buffer, buffer.byteOffset + readPos, 4).getInt32(
          0,
          true
        );
        readPos += 4;
        const strBytes = buffer.subarray(readPos, readPos + strLen);
        values.push(new TextDecoder().decode(strBytes));
        readPos += strLen;
        break;
      }
      default:
        throw new DataValidationError(`readParquet: unsupported physical type ${ptype}`);
    }
  }
  return values;
}

/**
 * Read column data from a Parquet buffer.
 *
 * Supports uncompressed PLAIN-encoded data pages (v1) with flat schemas —
 * the format this module writes. Dictionary-encoded or compressed files
 * throw a descriptive error instead of returning wrong data.
 *
 * @param buffer - The raw .parquet file as a Uint8Array
 * @param options - Read options (`columns` selects a subset)
 * @returns Parsed data with columns and rows
 */
export function readParquet(
  buffer: Uint8Array,
  options: ParquetReadOptions = {}
): { columns: string[]; data: Record<string, unknown>[] } {
  if (buffer.length < 12) {
    return { columns: [], data: [] };
  }

  const headerMagic = buffer.subarray(0, 4);
  const footerMagic = buffer.subarray(buffer.length - 4);
  if (!arrayEquals(headerMagic, PARQUET_MAGIC) || !arrayEquals(footerMagic, PARQUET_MAGIC)) {
    return { columns: [], data: [] };
  }

  const metaLen = new DataView(buffer.buffer, buffer.byteOffset + buffer.length - 8, 4).getInt32(
    0,
    true
  );
  if (metaLen <= 0 || metaLen > buffer.length - 12) {
    return { columns: [], data: [] };
  }

  const metaStart = buffer.length - 8 - metaLen;
  const reader = new ThriftReader(buffer.subarray(metaStart, metaStart + metaLen));
  reader.resetFieldTracking();

  let schema: SchemaColumn[] = [];
  const chunks: ColumnChunkInfo[] = [];
  let numRows = 0;

  let field = reader.readFieldBegin();
  while (field.type !== 0) {
    switch (field.id) {
      case 1:
        reader.readI32(); // version
        break;
      case 2: {
        const list = reader.readListBegin();
        schema = parseSchema(reader, list.size);
        break;
      }
      case 3:
        numRows = Number(reader.readI64());
        break;
      case 4: {
        // row_groups: list<RowGroup>
        const groups = reader.readListBegin();
        for (let g = 0; g < groups.size; g++) {
          reader.enterStruct();
          let rf = reader.readFieldBegin();
          while (rf.type !== 0) {
            if (rf.id === 1 && rf.type === TC_LIST) {
              const cols = reader.readListBegin();
              for (let c = 0; c < cols.size; c++) {
                chunks.push(parseColumnChunk(reader));
              }
            } else {
              reader.skipField(rf.type);
            }
            rf = reader.readFieldBegin();
          }
          reader.exitStruct();
        }
        break;
      }
      default:
        reader.skipField(field.type);
    }
    field = reader.readFieldBegin();
  }

  if (chunks.length > schema.length) {
    throw new DataValidationError(
      "readParquet: multiple row groups are not supported by this reader"
    );
  }

  const wanted = options.columns ? new Set(options.columns) : null;
  const columnData = new Map<string, unknown[]>();

  for (let c = 0; c < schema.length; c++) {
    const col = schema[c]!;
    const chunk = chunks[c];
    if (!chunk) continue;
    if (wanted && !wanted.has(col.name)) continue;

    if (chunk.codec !== CompressionCodec.UNCOMPRESSED) {
      throw new DataValidationError(
        `readParquet: compressed column "${col.name}" is not supported (codec ${chunk.codec}); ` +
          "write with compression disabled"
      );
    }

    // Parse the page header at the chunk's data offset.
    const pageReader = new ThriftReader(buffer.subarray(chunk.pageOffset));
    pageReader.resetFieldTracking();
    let pageType = -1;
    let encoding = Encoding.PLAIN;
    let pageNumValues = chunk.numValues;
    let pf = pageReader.readFieldBegin();
    while (pf.type !== 0) {
      switch (pf.id) {
        case 1:
          pageType = pageReader.readI32();
          break;
        case 5: {
          pageReader.enterStruct();
          let df = pageReader.readFieldBegin();
          while (df.type !== 0) {
            if (df.id === 1) pageNumValues = pageReader.readI32();
            else if (df.id === 2) encoding = pageReader.readI32();
            else pageReader.skipField(df.type);
            df = pageReader.readFieldBegin();
          }
          pageReader.exitStruct();
          break;
        }
        default:
          pageReader.skipField(pf.type);
      }
      pf = pageReader.readFieldBegin();
    }

    if (pageType !== PageType.DATA_PAGE || encoding !== Encoding.PLAIN) {
      throw new DataValidationError(
        `readParquet: column "${col.name}" uses an unsupported page type or encoding ` +
          "(only uncompressed PLAIN data pages v1 are supported)"
      );
    }

    let dataPos = chunk.pageOffset + pageReader.position;

    let defLevels: Uint8Array | null = null;
    if (col.optional) {
      const levelLen = new DataView(buffer.buffer, buffer.byteOffset + dataPos, 4).getInt32(
        0,
        true
      );
      defLevels = decodeDefinitionLevels(
        buffer.subarray(dataPos + 4, dataPos + 4 + levelLen),
        pageNumValues
      );
      dataPos += 4 + levelLen;
    }

    const nonNullCount = defLevels
      ? defLevels.reduce((acc, level) => acc + level, 0)
      : pageNumValues;
    const raw = decodeValuesPlain(buffer, dataPos, nonNullCount, col.type);

    const values: unknown[] = [];
    if (defLevels) {
      let next = 0;
      for (let r = 0; r < pageNumValues; r++) {
        values.push(defLevels[r] ? raw[next++] : null);
      }
    } else {
      values.push(...raw);
    }
    columnData.set(col.name, values);
  }

  const outColumns = schema.map((s) => s.name).filter((n) => !wanted || wanted.has(n));
  const data: Record<string, unknown>[] = [];
  for (let r = 0; r < numRows; r++) {
    const row: Record<string, unknown> = {};
    for (const name of outColumns) {
      row[name] = columnData.get(name)?.[r] ?? null;
    }
    data.push(row);
  }

  return { columns: outColumns, data };
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function concat(...arrays: Uint8Array[]): Uint8Array {
  let size = 0;
  for (const a of arrays) size += a.length;
  const result = new Uint8Array(size);
  let offset = 0;
  for (const a of arrays) {
    result.set(a, offset);
    offset += a.length;
  }
  return result;
}

function arrayEquals(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}
