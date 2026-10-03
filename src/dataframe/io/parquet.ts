/**
 * Zero-dependency Parquet reader and writer for Deepbox DataFrames.
 *
 * Implements a spec-compliant subset of Apache Parquet (verified against
 * pyarrow).
 *
 * Writer (always the same, simple layout):
 * - Flat schemas, a single row group, one PLAIN-encoded data page per column
 * - BOOLEAN (bit-packed), INT32, INT64, DOUBLE, BYTE_ARRAY (UTF8 string) and
 *   INT64 TIMESTAMP_MILLIS (for `Date` columns)
 * - Nullable columns (OPTIONAL repetition with RLE definition levels)
 * - Uncompressed, format version 1
 *
 * Reader:
 * - Flat schemas with any number of row groups and data pages
 * - Data pages v1 and v2, PLAIN, dictionary (PLAIN_DICTIONARY / RLE_DICTIONARY)
 *   and RLE (boolean) encodings
 * - UNCOMPRESSED, SNAPPY and (on Node.js) GZIP column chunks
 * - Logical types: strings, dates, timestamps (ms/us/ns, returned as `Date`),
 *   unsigned integers
 * - Anything else (nested/repeated fields, DECIMAL, INT96, FIXED_LEN_BYTE_ARRAY,
 *   delta encodings, ZSTD/LZ4/Brotli) throws a descriptive error instead of
 *   returning wrong data.
 *
 * The Parquet format uses Thrift-encoded metadata. This module includes
 * a minimal Thrift Compact Protocol encoder/decoder.
 *
 * @module dataframe/io/parquet
 * @see {@link https://deepbox.dev/docs/dataframe-overview | Deepbox documentation}
 */

import { DataValidationError, InvalidParameterError } from "../../core";

// ─── Parquet Constants ───────────────────────────────────────────────────────

const PARQUET_MAGIC = new Uint8Array([0x50, 0x41, 0x52, 0x31]); // "PAR1"

enum ParquetType {
  BOOLEAN = 0,
  INT32 = 1,
  INT64 = 2,
  INT96 = 3,
  FLOAT = 4,
  DOUBLE = 5,
  BYTE_ARRAY = 6,
  FIXED_LEN_BYTE_ARRAY = 7,
}

enum Encoding {
  PLAIN = 0,
  PLAIN_DICTIONARY = 2,
  RLE = 3,
  RLE_DICTIONARY = 8,
}

enum CompressionCodec {
  UNCOMPRESSED = 0,
  SNAPPY = 1,
  GZIP = 2,
}

enum PageType {
  DATA_PAGE = 0,
  INDEX_PAGE = 1,
  DICTIONARY_PAGE = 2,
  DATA_PAGE_V2 = 3,
}

enum FieldRepetitionType {
  REQUIRED = 0,
  OPTIONAL = 1,
  REPEATED = 2,
}

// ConvertedType values used by this module.
const CT_UTF8 = 0;
const CT_ENUM = 4;
const CT_DECIMAL = 5;
const CT_DATE = 6;
const CT_TIMESTAMP_MILLIS = 9;
const CT_TIMESTAMP_MICROS = 10;
const CT_UINT_32 = 13;
const CT_UINT_64 = 14;
const CT_JSON = 19;
const CT_INTERVAL = 21;

// LogicalType union member ids.
const LT_STRING = 1;
const LT_ENUM = 4;
const LT_DECIMAL = 5;
const LT_DATE = 6;
const LT_TIMESTAMP = 8;
const LT_INTEGER = 10;
const LT_JSON = 12;
const LT_UUID = 14;

// Thrift Compact Protocol type codes.
const TC_BOOL_TRUE = 1;
const TC_BOOL_FALSE = 2;
const TC_I8 = 3;
const TC_I16 = 4;
const TC_I32 = 5;
const TC_I64 = 6;
const TC_DOUBLE = 7;
const TC_BINARY = 8;
const TC_LIST = 9;
const TC_SET = 10;
const TC_MAP = 11;
const TC_STRUCT = 12;

const MAX_THRIFT_DEPTH = 64;
const TWO_32 = 4294967296;

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder("utf-8");

function truncated(what: string): DataValidationError {
  return new DataValidationError(`readParquet: unexpected end of data while reading ${what}`);
}

// ─── Byte helpers ────────────────────────────────────────────────────────────

/** Growable byte buffer used by the Thrift writer. */
class ByteSink {
  private buf = new Uint8Array(256);
  private len = 0;

  private reserve(extra: number): void {
    const need = this.len + extra;
    if (need <= this.buf.length) return;
    let cap = this.buf.length * 2;
    while (cap < need) cap *= 2;
    const next = new Uint8Array(cap);
    next.set(this.buf.subarray(0, this.len));
    this.buf = next;
  }

  byte(value: number): void {
    this.reserve(1);
    this.buf[this.len++] = value & 0xff;
  }

  bytes(value: Uint8Array): void {
    this.reserve(value.length);
    this.buf.set(value, this.len);
    this.len += value.length;
  }

  toBytes(): Uint8Array {
    return this.buf.slice(0, this.len);
  }
}

function concat(arrays: readonly Uint8Array[]): Uint8Array {
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

/** Assign `obj[key] = value` without letting a `"__proto__"` column rewrite the prototype. */
function setOwn(obj: Record<string, unknown>, key: string, value: unknown): void {
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

// ─── Thrift Compact Protocol (minimal) ───────────────────────────────────────

class ThriftWriter {
  private readonly sink = new ByteSink();
  private lastFieldId = 0;

  writeFieldBegin(type: number, id: number): void {
    const delta = id - this.lastFieldId;
    if (delta > 0 && delta <= 15) {
      this.sink.byte((delta << 4) | type);
    } else {
      this.sink.byte(type);
      this.writeI32(id);
    }
    this.lastFieldId = id;
  }

  writeFieldStop(): void {
    this.sink.byte(0);
  }

  writeStructBegin(): void {
    this.lastFieldId = 0;
  }

  writeI32(value: number): void {
    this.writeVarint(((value << 1) ^ (value >> 31)) >>> 0);
  }

  writeI64(value: bigint): void {
    let v = ((value << 1n) ^ (value >> 63n)) & 0xffffffffffffffffn;
    while (v > 0x7fn) {
      this.sink.byte(Number(v & 0x7fn) | 0x80);
      v >>= 7n;
    }
    this.sink.byte(Number(v));
  }

  writeString(value: string): void {
    const bytes = textEncoder.encode(value);
    this.writeVarint(bytes.length);
    this.sink.bytes(bytes);
  }

  /** Append pre-encoded struct bytes verbatim (no length prefix). */
  writeStructBytes(value: Uint8Array): void {
    this.sink.bytes(value);
  }

  writeListBegin(elemType: number, size: number): void {
    if (size <= 14) {
      this.sink.byte((size << 4) | elemType);
    } else {
      this.sink.byte(0xf0 | elemType);
      this.writeVarint(size);
    }
  }

  private writeVarint(value: number): void {
    let v = value >>> 0;
    while (v > 0x7f) {
      this.sink.byte((v & 0x7f) | 0x80);
      v >>>= 7;
    }
    this.sink.byte(v);
  }

  toBytes(): Uint8Array {
    return this.sink.toBytes();
  }
}

class ThriftReader {
  private pos: number;
  private lastFieldId = 0;
  private readonly fieldIdStack: number[] = [];
  private depth = 0;
  private readonly data: Uint8Array;

  constructor(data: Uint8Array, start = 0) {
    this.data = data;
    this.pos = start;
  }

  get position(): number {
    return this.pos;
  }

  private need(n: number): void {
    if (n < 0 || this.pos + n > this.data.length) throw truncated("Thrift metadata");
  }

  readFieldBegin(): { type: number; id: number } {
    const byte = this.readByte();
    if (byte === 0) return { type: 0, id: 0 }; // STOP

    const delta = (byte >> 4) & 0x0f;
    const type = byte & 0x0f;

    if (delta !== 0) {
      this.lastFieldId += delta;
    } else {
      this.lastFieldId = this.readI32();
    }

    return { type, id: this.lastFieldId };
  }

  readByte(): number {
    this.need(1);
    return this.data[this.pos++]!;
  }

  readI32(): number {
    const n = this.readVarint();
    return ((n >>> 1) ^ -(n & 1)) | 0;
  }

  readI64(): bigint {
    const n = this.readVarintBig();
    return (n >> 1n) ^ -(n & 1n);
  }

  readString(): string {
    const len = this.readVarint();
    this.need(len);
    const text = textDecoder.decode(this.data.subarray(this.pos, this.pos + len));
    this.pos += len;
    return text;
  }

  readListBegin(): { elemType: number; size: number } {
    const byte = this.readByte();
    let size = (byte >> 4) & 0x0f;
    const elemType = byte & 0x0f;
    if (size === 0x0f) size = this.readVarint();
    // Every list element occupies at least one byte, which bounds corrupt sizes.
    if (size > this.data.length - this.pos) throw truncated("Thrift list");
    return { elemType, size };
  }

  /** Enter a nested struct: nested field ids start from their own context. */
  enterStruct(): void {
    if (++this.depth > MAX_THRIFT_DEPTH) {
      throw new DataValidationError("readParquet: metadata is nested too deeply");
    }
    this.fieldIdStack.push(this.lastFieldId);
    this.lastFieldId = 0;
  }

  /** Leave a nested struct, restoring the parent's field-id context. */
  exitStruct(): void {
    this.depth--;
    this.lastFieldId = this.fieldIdStack.pop() ?? 0;
  }

  /** Skip a field of the given Thrift type. `inContainer` marks list/map elements (bools take a byte there). */
  skipField(type: number, inContainer = false): void {
    switch (type) {
      case TC_BOOL_TRUE:
      case TC_BOOL_FALSE:
        if (inContainer) this.readByte();
        break;
      case TC_I8:
        this.readByte();
        break;
      case TC_I16:
      case TC_I32:
        this.readVarint();
        break;
      case TC_I64:
        this.readVarintBig();
        break;
      case TC_DOUBLE:
        this.need(8);
        this.pos += 8;
        break;
      case TC_BINARY: {
        const len = this.readVarint();
        this.need(len);
        this.pos += len;
        break;
      }
      case TC_LIST:
      case TC_SET: {
        const list = this.readListBegin();
        for (let i = 0; i < list.size; i++) this.skipField(list.elemType, true);
        break;
      }
      case TC_MAP: {
        const size = this.readVarint();
        if (size > 0) {
          const kv = this.readByte();
          for (let i = 0; i < size; i++) {
            this.skipField((kv >> 4) & 0x0f, true);
            this.skipField(kv & 0x0f, true);
          }
        }
        break;
      }
      case TC_STRUCT: {
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
        throw new DataValidationError(`readParquet: unknown Thrift field type ${type}`);
    }
  }

  private readVarint(): number {
    let result = 0;
    let shift = 0;
    for (;;) {
      const byte = this.readByte();
      result += (byte & 0x7f) * 2 ** shift;
      if ((byte & 0x80) === 0) break;
      shift += 7;
      if (shift > 28) throw new DataValidationError("readParquet: malformed Thrift varint");
    }
    return result >>> 0;
  }

  private readVarintBig(): bigint {
    let result = 0n;
    let shift = 0n;
    for (;;) {
      const byte = this.readByte();
      result |= BigInt(byte & 0x7f) << shift;
      if ((byte & 0x80) === 0) break;
      shift += 7n;
      if (shift > 63n) throw new DataValidationError("readParquet: malformed Thrift varint");
    }
    return result;
  }
}

// ─── Value encoding ──────────────────────────────────────────────────────────

type ParquetValue = string | number | boolean | bigint | Date | null | undefined;

/** How a column is stored in the file beyond its physical type. */
type ColumnKind = "plain" | "string" | "timestampMs";

const INT32_MIN = -2147483648;
const INT32_MAX = 2147483647;
const INT64_MIN_NUM = -(2 ** 63);
const INT64_MAX_NUM = 2 ** 63; // exclusive bound
const INT64_MIN = -(2n ** 63n);
const INT64_MAX = 2n ** 63n - 1n;

type ColumnPlan = {
  readonly ptype: ParquetType;
  readonly kind: ColumnKind;
  readonly optional: boolean;
  /** Per-row values with `null` for missing entries (invalid dates count as missing). */
  readonly values: readonly ParquetValue[];
};

/**
 * Infer the Parquet physical type of a column by scanning every value, so a
 * column like `[1, 2, 1.5]` becomes DOUBLE instead of silently truncating.
 *
 * - all booleans: BOOLEAN; all `Date`s: INT64 timestamp (ms)
 * - integers (numbers or bigints): INT32 when every value fits, else INT64
 * - any non-integer or out-of-INT64-range number: DOUBLE
 * - everything else (strings, mixed types): BYTE_ARRAY holding UTF-8 text
 */
function planColumn(name: string, rows: readonly Record<string, unknown>[]): ColumnPlan {
  const values: ParquetValue[] = new Array(rows.length);
  let optional = false;
  let sawValue = false;
  let allBoolean = true;
  let allDate = true;
  let nonNullCount = 0;
  let numeric = 0; // numbers and bigints
  let allInteger = true; // every number is an integer within the INT64 range (and not -0)
  let fitsInt32 = true;
  let sawBigint = false;

  for (let i = 0; i < rows.length; i++) {
    const row = rows[i]!;
    let v = Object.hasOwn(row, name) ? (row[name] as ParquetValue) : null;
    if (v instanceof Date && Number.isNaN(v.getTime())) v = null;
    if (v === undefined) v = null;
    values[i] = v;
    if (v === null) {
      optional = true;
      continue;
    }
    sawValue = true;
    nonNullCount++;
    if (typeof v !== "boolean") allBoolean = false;
    if (!(v instanceof Date)) allDate = false;
    if (typeof v === "number") {
      numeric++;
      if (!Number.isInteger(v) || Object.is(v, -0) || v < INT64_MIN_NUM || v >= INT64_MAX_NUM) {
        allInteger = false;
      } else if (v < INT32_MIN || v > INT32_MAX) {
        fitsInt32 = false;
      }
    } else if (typeof v === "bigint") {
      numeric++;
      sawBigint = true;
      if (v < INT64_MIN || v > INT64_MAX) {
        throw new DataValidationError(
          `writeParquet: column "${name}" holds a bigint outside the INT64 range`
        );
      }
      fitsInt32 = false;
    }
  }

  const make = (ptype: ParquetType, kind: ColumnKind): ColumnPlan => ({
    ptype,
    kind,
    optional,
    values,
  });

  if (!sawValue) return make(ParquetType.BYTE_ARRAY, "string");
  if (allBoolean) return make(ParquetType.BOOLEAN, "plain");
  if (allDate) return make(ParquetType.INT64, "timestampMs");
  if (numeric === nonNullCount) {
    if (!sawBigint) {
      if (!allInteger) return make(ParquetType.DOUBLE, "plain");
      return make(fitsInt32 ? ParquetType.INT32 : ParquetType.INT64, "plain");
    }
    // bigints mixed with numbers: only exact when every number is an integer.
    if (allInteger) return make(ParquetType.INT64, "plain");
  }
  return make(ParquetType.BYTE_ARRAY, "string");
}

/** PLAIN-encode non-null values (booleans bit-packed LSB-first per spec). */
function encodeValuesPlain(plan: ColumnPlan): Uint8Array {
  const nonNull: Exclude<ParquetValue, null | undefined>[] = [];
  for (const v of plan.values) {
    if (v !== null && v !== undefined) nonNull.push(v);
  }
  const n = nonNull.length;

  switch (plan.ptype) {
    case ParquetType.BOOLEAN: {
      const out = new Uint8Array(Math.ceil(n / 8));
      for (let i = 0; i < n; i++) {
        if (nonNull[i]) out[i >> 3]! |= 1 << (i & 7);
      }
      return out;
    }
    case ParquetType.INT32: {
      const out = new Uint8Array(n * 4);
      const view = new DataView(out.buffer);
      for (let i = 0; i < n; i++) view.setInt32(i * 4, Number(nonNull[i]), true);
      return out;
    }
    case ParquetType.INT64: {
      const out = new Uint8Array(n * 8);
      const view = new DataView(out.buffer);
      for (let i = 0; i < n; i++) {
        const v = nonNull[i];
        const p = i * 8;
        if (typeof v === "bigint") {
          view.setBigInt64(p, v, true);
        } else {
          const num = v instanceof Date ? v.getTime() : Number(v);
          if (Math.abs(num) < 2 ** 53) {
            // Exact two-word split avoids a BigInt allocation per value.
            const hi = Math.floor(num / TWO_32);
            view.setUint32(p, num - hi * TWO_32, true);
            view.setInt32(p + 4, hi, true);
          } else {
            view.setBigInt64(p, BigInt(num), true);
          }
        }
      }
      return out;
    }
    case ParquetType.DOUBLE: {
      const out = new Uint8Array(n * 8);
      const view = new DataView(out.buffer);
      for (let i = 0; i < n; i++) view.setFloat64(i * 8, Number(nonNull[i]), true);
      return out;
    }
    case ParquetType.BYTE_ARRAY: {
      const encoded: Uint8Array[] = new Array(n);
      let total = 0;
      for (let i = 0; i < n; i++) {
        const v = nonNull[i];
        const bytes = textEncoder.encode(v instanceof Date ? v.toISOString() : String(v));
        encoded[i] = bytes;
        total += 4 + bytes.length;
      }
      const out = new Uint8Array(total);
      const view = new DataView(out.buffer);
      let pos = 0;
      for (const bytes of encoded) {
        view.setUint32(pos, bytes.length, true);
        out.set(bytes, pos + 4);
        pos += 4 + bytes.length;
      }
      return out;
    }
    default:
      throw new DataValidationError(`writeParquet: unsupported physical type ${plan.ptype}`);
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
  const header: number[] = [];
  let h = groups * 2 + 1;
  while (h > 0x7f) {
    header.push((h % 128) | 0x80);
    h = Math.floor(h / 128);
  }
  header.push(h);

  const bodyLength = header.length + packed.length;
  const out = new Uint8Array(4 + bodyLength);
  new DataView(out.buffer).setInt32(0, bodyLength, true);
  out.set(header, 4);
  out.set(packed, 4 + header.length);
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
 * Column types are inferred from every value of the column:
 * booleans become BOOLEAN, `Date`s become TIMESTAMP_MILLIS, integers become
 * INT32 (or INT64 when any value exceeds the 32-bit range or is a `bigint`),
 * other numbers become DOUBLE, and everything else (strings, mixed columns)
 * is stored as UTF-8 text. Invalid dates are written as nulls.
 *
 * Columns containing `null`/`undefined` are written as OPTIONAL with RLE
 * definition levels, so nulls survive the round-trip (also when read by
 * external tools such as pyarrow/pandas). `NaN` is a regular DOUBLE value,
 * not a null.
 *
 * @param columns - Column names (must be unique)
 * @param data - Array of row objects
 * @param options - Write options
 * @returns Uint8Array containing the .parquet file
 * @throws {DataValidationError} If column names repeat, a row is not an object, a `bigint`
 *   exceeds the INT64 range, or a column chunk is larger than 2 GiB
 */
export function writeParquet(
  columns: readonly string[],
  data: readonly Record<string, unknown>[],
  options: ParquetWriteOptions = {}
): Uint8Array {
  const numRows = data.length;

  const seen = new Set<string>();
  for (const col of columns) {
    if (typeof col !== "string") {
      throw new DataValidationError("writeParquet: column names must be strings");
    }
    if (seen.has(col)) {
      throw new DataValidationError(`writeParquet: duplicate column name "${col}"`);
    }
    seen.add(col);
  }
  for (let r = 0; r < numRows; r++) {
    const row = data[r];
    if (typeof row !== "object" || row === null) {
      throw new DataValidationError(`writeParquet: row ${r} is not an object`);
    }
  }

  const plans = columns.map((col) => planColumn(col, data));

  // Encode column chunks sequentially after the 4-byte magic.
  const columnChunks: Uint8Array[] = [];
  const columnOffsets: number[] = [];
  let currentOffset = 4;

  for (let c = 0; c < columns.length; c++) {
    const plan = plans[c]!;
    const encoded = encodeValuesPlain(plan);
    const pageData = plan.optional
      ? concat([encodeDefinitionLevels(plan.values), encoded])
      : encoded;
    if (pageData.length > INT32_MAX) {
      throw new DataValidationError(
        `writeParquet: column "${columns[c]}" is too large for a single data page`
      );
    }
    const pageHeader = writePageHeader(pageData.length, numRows, plan.optional);

    columnOffsets.push(currentOffset);
    const chunk = concat([pageHeader, pageData]);
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
        plans[c]!.ptype,
        columns[c]!,
        numRows,
        columnOffsets[c]!,
        columnChunks[c]!.length,
        plans[c]!.optional
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
    const plan = plans[c]!;
    const el = new ThriftWriter();
    el.writeStructBegin();
    el.writeFieldBegin(TC_I32, 1); // type
    el.writeI32(plan.ptype);
    el.writeFieldBegin(TC_I32, 3); // repetition_type
    el.writeI32(plan.optional ? FieldRepetitionType.OPTIONAL : FieldRepetitionType.REQUIRED);
    el.writeFieldBegin(TC_BINARY, 4); // name
    el.writeString(columns[c]!);
    if (plan.kind === "string" || plan.kind === "timestampMs") {
      el.writeFieldBegin(TC_I32, 6); // converted_type
      el.writeI32(plan.kind === "string" ? CT_UTF8 : CT_TIMESTAMP_MILLIS);
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

  return concat([PARQUET_MAGIC, ...columnChunks, metaBytes, metaLen, PARQUET_MAGIC]);
}

// ─── Parquet Reader ──────────────────────────────────────────────────────────

/** Options for reading a Parquet file. */
export type ParquetReadOptions = {
  /**
   * Columns to read, returned in the order given. Default: all columns in
   * file order. Unknown names throw.
   */
  readonly columns?: readonly string[];
};

/** Result of {@link readParquet}: the column names and one record per row. */
export type ParquetReadResult = {
  columns: string[];
  data: Record<string, unknown>[];
};

/** How a stored value is turned into a JavaScript value. */
type ReadKind =
  | "plain"
  | "string"
  | "date"
  | "timestampMs"
  | "timestampUs"
  | "timestampNs"
  | "uint32"
  | "uint64"
  | "unsupported";

type LogicalType = { id: number; unit?: number; bits?: number; signed?: boolean };

type SchemaElement = {
  name: string;
  type: number;
  repetition: number;
  converted: number;
  numChildren: number;
  logical: LogicalType | undefined;
};

type SchemaColumn = {
  name: string;
  type: ParquetType;
  optional: boolean;
  kind: ReadKind;
  /** Why the column cannot be decoded, when `kind` is "unsupported". */
  unsupported?: string;
};

type ColumnChunkInfo = {
  codec: number;
  numValues: number;
  dataPageOffset: number;
  dictionaryPageOffset: number;
  externalFile: boolean;
};

type RowGroupInfo = { numRows: number; chunks: ColumnChunkInfo[] };

type FileMetadata = {
  schema: SchemaElement[];
  numRows: number;
  rowGroups: RowGroupInfo[];
};

function readBoolField(type: number): boolean {
  return type === TC_BOOL_TRUE;
}

function parseLogicalType(reader: ThriftReader): LogicalType | undefined {
  reader.enterStruct();
  let result: LogicalType | undefined;
  let f = reader.readFieldBegin();
  while (f.type !== 0) {
    if (f.type === TC_STRUCT) {
      const logical: LogicalType = { id: f.id };
      reader.enterStruct();
      let inner = reader.readFieldBegin();
      while (inner.type !== 0) {
        if (f.id === LT_TIMESTAMP && inner.id === 2 && inner.type === TC_STRUCT) {
          // TimeUnit union: 1 = MILLIS, 2 = MICROS, 3 = NANOS
          reader.enterStruct();
          const unit = reader.readFieldBegin();
          if (unit.type !== 0) {
            logical.unit = unit.id;
            reader.skipField(unit.type);
            let rest = reader.readFieldBegin();
            while (rest.type !== 0) {
              reader.skipField(rest.type);
              rest = reader.readFieldBegin();
            }
          }
          reader.exitStruct();
        } else if (f.id === LT_INTEGER && inner.id === 1 && inner.type === TC_I8) {
          logical.bits = reader.readByte();
        } else if (f.id === LT_INTEGER && inner.id === 2 && inner.type <= TC_BOOL_FALSE) {
          logical.signed = readBoolField(inner.type);
        } else {
          reader.skipField(inner.type);
        }
        inner = reader.readFieldBegin();
      }
      reader.exitStruct();
      result = logical;
    } else {
      reader.skipField(f.type);
    }
    f = reader.readFieldBegin();
  }
  reader.exitStruct();
  return result;
}

function parseSchema(reader: ThriftReader, size: number): SchemaElement[] {
  const elements: SchemaElement[] = [];
  for (let i = 0; i < size; i++) {
    reader.enterStruct();
    const el: SchemaElement = {
      name: "",
      type: -1,
      repetition: FieldRepetitionType.REQUIRED,
      converted: -1,
      numChildren: -1,
      logical: undefined,
    };
    let sf = reader.readFieldBegin();
    while (sf.type !== 0) {
      switch (sf.id) {
        case 1:
          el.type = reader.readI32();
          break;
        case 3:
          el.repetition = reader.readI32();
          break;
        case 4:
          el.name = reader.readString();
          break;
        case 5:
          el.numChildren = reader.readI32();
          break;
        case 6:
          el.converted = reader.readI32();
          break;
        case 10:
          if (sf.type === TC_STRUCT) el.logical = parseLogicalType(reader);
          else reader.skipField(sf.type);
          break;
        default:
          reader.skipField(sf.type);
      }
      sf = reader.readFieldBegin();
    }
    reader.exitStruct();
    elements.push(el);
  }
  return elements;
}

function parseColumnChunk(reader: ThriftReader): ColumnChunkInfo {
  reader.enterStruct();
  const info: ColumnChunkInfo = {
    codec: CompressionCodec.UNCOMPRESSED,
    numValues: 0,
    dataPageOffset: -1,
    dictionaryPageOffset: -1,
    externalFile: false,
  };
  let f = reader.readFieldBegin();
  while (f.type !== 0) {
    if (f.id === 1 && f.type === TC_BINARY) {
      info.externalFile = reader.readString().length > 0; // file_path
    } else if (f.id === 3 && f.type === TC_STRUCT) {
      // meta_data
      reader.enterStruct();
      let mf = reader.readFieldBegin();
      while (mf.type !== 0) {
        switch (mf.id) {
          case 4:
            info.codec = reader.readI32();
            break;
          case 5:
            info.numValues = Number(reader.readI64());
            break;
          case 9:
            info.dataPageOffset = Number(reader.readI64());
            break;
          case 11:
            info.dictionaryPageOffset = Number(reader.readI64());
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
  return info;
}

function parseFileMetadata(footer: Uint8Array): FileMetadata {
  const reader = new ThriftReader(footer);
  const meta: FileMetadata = { schema: [], numRows: 0, rowGroups: [] };

  let field = reader.readFieldBegin();
  while (field.type !== 0) {
    switch (field.id) {
      case 2: {
        const list = reader.readListBegin();
        meta.schema = parseSchema(reader, list.size);
        break;
      }
      case 3:
        meta.numRows = Number(reader.readI64());
        break;
      case 4: {
        // row_groups: list<RowGroup>
        const groups = reader.readListBegin();
        for (let g = 0; g < groups.size; g++) {
          reader.enterStruct();
          const group: RowGroupInfo = { numRows: 0, chunks: [] };
          let rf = reader.readFieldBegin();
          while (rf.type !== 0) {
            if (rf.id === 1 && rf.type === TC_LIST) {
              const cols = reader.readListBegin();
              for (let c = 0; c < cols.size; c++) group.chunks.push(parseColumnChunk(reader));
            } else if (rf.id === 3 && rf.type === TC_I64) {
              group.numRows = Number(reader.readI64());
            } else {
              reader.skipField(rf.type);
            }
            rf = reader.readFieldBegin();
          }
          reader.exitStruct();
          meta.rowGroups.push(group);
        }
        break;
      }
      default:
        reader.skipField(field.type);
    }
    field = reader.readFieldBegin();
  }
  return meta;
}

/** Turn the raw schema list into flat leaf columns, rejecting nested/repeated schemas. */
function flatColumns(schema: readonly SchemaElement[]): SchemaColumn[] {
  const root = schema[0];
  if (!root || root.numChildren < 0) {
    throw new DataValidationError("readParquet: file has no schema");
  }
  if (root.numChildren !== schema.length - 1) {
    throw new DataValidationError(
      "readParquet: nested schemas (structs, lists, maps) are not supported; " +
        "only flat tables can be read"
    );
  }
  const cols: SchemaColumn[] = [];
  for (let i = 1; i < schema.length; i++) {
    const el = schema[i]!;
    if (el.numChildren >= 0 || el.type < 0) {
      throw new DataValidationError(
        `readParquet: nested column "${el.name}" is not supported; only flat tables can be read`
      );
    }
    if (el.repetition === FieldRepetitionType.REPEATED) {
      throw new DataValidationError(
        `readParquet: repeated column "${el.name}" is not supported; only flat tables can be read`
      );
    }
    const type = el.type as ParquetType;
    const { kind, unsupported } = columnKind(el, type);
    const col: SchemaColumn = {
      name: el.name,
      type,
      optional: el.repetition === FieldRepetitionType.OPTIONAL,
      kind,
    };
    if (unsupported !== undefined) col.unsupported = unsupported;
    cols.push(col);
  }
  return cols;
}

function columnKind(
  el: SchemaElement,
  type: ParquetType
): { kind: ReadKind; unsupported?: string } {
  const lt = el.logical;
  const ct = el.converted;
  const isBytes = type === ParquetType.BYTE_ARRAY;

  if (lt?.id === LT_DECIMAL || ct === CT_DECIMAL) {
    return { kind: "unsupported", unsupported: "DECIMAL values" };
  }
  if (ct === CT_INTERVAL) return { kind: "unsupported", unsupported: "INTERVAL values" };
  if (lt?.id === LT_UUID) return { kind: "unsupported", unsupported: "UUID values" };

  if (lt?.id === LT_STRING || lt?.id === LT_ENUM || lt?.id === LT_JSON) {
    return { kind: isBytes ? "string" : "plain" };
  }
  if (lt?.id === LT_DATE) return { kind: type === ParquetType.INT32 ? "date" : "plain" };
  if (lt?.id === LT_TIMESTAMP && type === ParquetType.INT64) {
    if (lt.unit === 1) return { kind: "timestampMs" };
    if (lt.unit === 2) return { kind: "timestampUs" };
    if (lt.unit === 3) return { kind: "timestampNs" };
  }
  if (lt?.id === LT_INTEGER && lt.signed === false) {
    if (lt.bits === 32 && type === ParquetType.INT32) return { kind: "uint32" };
    if (lt.bits === 64 && type === ParquetType.INT64) return { kind: "uint64" };
  }
  if (lt) return { kind: "plain" };

  if (ct === CT_UTF8 || ct === CT_ENUM || ct === CT_JSON) {
    return { kind: isBytes ? "string" : "plain" };
  }
  if (ct === CT_DATE && type === ParquetType.INT32) return { kind: "date" };
  if (ct === CT_TIMESTAMP_MILLIS && type === ParquetType.INT64) return { kind: "timestampMs" };
  if (ct === CT_TIMESTAMP_MICROS && type === ParquetType.INT64) return { kind: "timestampUs" };
  if (ct === CT_UINT_32 && type === ParquetType.INT32) return { kind: "uint32" };
  if (ct === CT_UINT_64 && type === ParquetType.INT64) return { kind: "uint64" };
  return { kind: "plain" };
}

const PHYSICAL_NAMES: Record<number, string> = {
  [ParquetType.INT96]: "INT96",
  [ParquetType.FIXED_LEN_BYTE_ARRAY]: "FIXED_LEN_BYTE_ARRAY",
};

const CODEC_NAMES: Record<number, string> = {
  1: "SNAPPY",
  2: "GZIP",
  3: "LZO",
  4: "BROTLI",
  5: "LZ4",
  6: "ZSTD",
  7: "LZ4_RAW",
};

const ENCODING_NAMES: Record<number, string> = {
  0: "PLAIN",
  2: "PLAIN_DICTIONARY",
  3: "RLE",
  4: "BIT_PACKED",
  5: "DELTA_BINARY_PACKED",
  6: "DELTA_LENGTH_BYTE_ARRAY",
  7: "DELTA_BYTE_ARRAY",
  8: "RLE_DICTIONARY",
  9: "BYTE_STREAM_SPLIT",
};

// ─── Decompression ───────────────────────────────────────────────────────────

/** Decode a raw Snappy block (the format Parquet uses for SNAPPY pages). */
function snappyDecompress(src: Uint8Array, expectedLength: number, what: string): Uint8Array {
  let pos = 0;
  let length = 0;
  let shift = 0;
  for (;;) {
    if (pos >= src.length) throw truncated(what);
    const b = src[pos++]!;
    length += (b & 0x7f) * 2 ** shift;
    if ((b & 0x80) === 0) break;
    shift += 7;
    if (shift > 28) throw new DataValidationError(`readParquet: corrupt Snappy data in ${what}`);
  }
  if (length !== expectedLength) {
    throw new DataValidationError(`readParquet: corrupt Snappy data in ${what} (size mismatch)`);
  }
  const out = new Uint8Array(length);
  let o = 0;
  const bad = (): DataValidationError =>
    new DataValidationError(`readParquet: corrupt Snappy data in ${what}`);

  while (pos < src.length) {
    const tag = src[pos++]!;
    const kind = tag & 3;
    if (kind === 0) {
      let litLen = tag >> 2;
      if (litLen >= 60) {
        const extra = litLen - 59;
        if (pos + extra > src.length) throw bad();
        litLen = 0;
        for (let i = 0; i < extra; i++) litLen += src[pos + i]! * 2 ** (8 * i);
        pos += extra;
      }
      litLen += 1;
      if (pos + litLen > src.length || o + litLen > length) throw bad();
      out.set(src.subarray(pos, pos + litLen), o);
      pos += litLen;
      o += litLen;
    } else {
      let copyLen: number;
      let offset: number;
      if (kind === 1) {
        if (pos + 1 > src.length) throw bad();
        copyLen = ((tag >> 2) & 7) + 4;
        offset = ((tag >> 5) << 8) | src[pos]!;
        pos += 1;
      } else if (kind === 2) {
        if (pos + 2 > src.length) throw bad();
        copyLen = (tag >> 2) + 1;
        offset = src[pos]! | (src[pos + 1]! << 8);
        pos += 2;
      } else {
        if (pos + 4 > src.length) throw bad();
        copyLen = (tag >> 2) + 1;
        offset = src[pos]! + src[pos + 1]! * 256 + src[pos + 2]! * 65536 + src[pos + 3]! * 16777216;
        pos += 4;
      }
      if (offset === 0 || offset > o || o + copyLen > length) throw bad();
      // Byte-wise copy: the ranges may overlap (run-length style copies).
      for (let i = 0; i < copyLen; i++) out[o + i] = out[o - offset + i]!;
      o += copyLen;
    }
  }
  if (o !== length) throw bad();
  return out;
}

function gunzip(src: Uint8Array, expectedLength: number, what: string): Uint8Array {
  const proc = (
    globalThis as {
      process?: { getBuiltinModule?: (id: string) => unknown };
    }
  ).process;
  const zlib = proc?.getBuiltinModule?.("node:zlib") as
    | { gunzipSync?: (buf: Uint8Array, opts?: { maxOutputLength?: number }) => Uint8Array }
    | undefined;
  if (!zlib?.gunzipSync) {
    throw new DataValidationError(
      "readParquet: GZIP-compressed columns require a Node.js runtime (20.16+ or 22.3+)"
    );
  }
  let out: Uint8Array;
  try {
    out = new Uint8Array(zlib.gunzipSync(src, { maxOutputLength: Math.max(expectedLength, 1) }));
  } catch (err) {
    throw new DataValidationError(`readParquet: corrupt GZIP data in ${what}`, { cause: err });
  }
  if (out.length !== expectedLength) {
    throw new DataValidationError(`readParquet: corrupt GZIP data in ${what} (size mismatch)`);
  }
  return out;
}

function decompress(
  src: Uint8Array,
  codec: number,
  expectedLength: number,
  what: string
): Uint8Array {
  switch (codec) {
    case CompressionCodec.UNCOMPRESSED:
      return src;
    case CompressionCodec.SNAPPY:
      return snappyDecompress(src, expectedLength, what);
    case CompressionCodec.GZIP:
      return gunzip(src, expectedLength, what);
    default:
      throw new DataValidationError(
        `readParquet: ${what} uses the unsupported ${CODEC_NAMES[codec] ?? `codec ${codec}`} ` +
          "compression (supported: UNCOMPRESSED, SNAPPY, GZIP)"
      );
  }
}

// ─── Value decoding ──────────────────────────────────────────────────────────

/**
 * Decode `count` values from an RLE / bit-packed hybrid stream in
 * `data[start, end)`. Throws if the stream ends before `count` values.
 */
function decodeRleHybrid(
  data: Uint8Array,
  start: number,
  end: number,
  bitWidth: number,
  count: number,
  what: string
): Uint32Array {
  const out = new Uint32Array(count);
  const valueBytes = (bitWidth + 7) >> 3;
  let pos = start;
  let produced = 0;

  while (produced < count) {
    let header = 0;
    let shift = 0;
    for (;;) {
      if (pos >= end) throw truncated(what);
      const b = data[pos++]!;
      header += (b & 0x7f) * 2 ** shift;
      if ((b & 0x80) === 0) break;
      shift += 7;
      if (shift > 28) throw new DataValidationError(`readParquet: corrupt levels in ${what}`);
    }

    if (header % 2 === 1) {
      // Bit-packed run: groups of 8 values, `bitWidth` bits per value, LSB first.
      const groups = (header - 1) / 2;
      const take = Math.min(groups * 8, count - produced);
      if (bitWidth > 0) {
        if (pos + Math.ceil((take * bitWidth) / 8) > end) throw truncated(what);
        if (bitWidth === 1) {
          for (let i = 0; i < take; i++) out[produced + i] = (data[pos + (i >> 3)]! >> (i & 7)) & 1;
        } else {
          let bitPos = 0;
          for (let i = 0; i < take; i++) {
            let v = 0;
            let got = 0;
            while (got < bitWidth) {
              const off = bitPos & 7;
              const n = Math.min(8 - off, bitWidth - got);
              v += ((data[pos + (bitPos >> 3)]! >> off) & ((1 << n) - 1)) * 2 ** got;
              got += n;
              bitPos += n;
            }
            out[produced + i] = v;
          }
        }
      }
      produced += take;
      pos += groups * bitWidth;
    } else {
      const run = header / 2;
      if (pos + valueBytes > end) throw truncated(what);
      let v = 0;
      for (let i = 0; i < valueBytes; i++) v += data[pos + i]! * 2 ** (8 * i);
      pos += valueBytes;
      const take = Math.min(run, count - produced);
      if (v !== 0) out.fill(v, produced, produced + take);
      produced += take;
    }
  }
  return out;
}

/** Decode `count` PLAIN values of `ptype` starting at `start`. */
function decodeValuesPlain(
  buf: Uint8Array,
  start: number,
  count: number,
  col: SchemaColumn
): unknown[] {
  const ptype = col.type;
  const what = `column "${col.name}"`;
  if (count < 0 || count > (buf.length - start) * 8) throw truncated(what);
  const out: unknown[] = new Array(count);
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);

  const checkBytes = (n: number): void => {
    if (start + n > buf.length) throw truncated(what);
  };

  switch (ptype) {
    case ParquetType.BOOLEAN:
      checkBytes(Math.ceil(count / 8));
      for (let i = 0; i < count; i++) {
        out[i] = ((buf[start + (i >> 3)]! >> (i & 7)) & 1) === 1;
      }
      return out;
    case ParquetType.INT32:
      checkBytes(count * 4);
      for (let i = 0; i < count; i++) out[i] = view.getInt32(start + i * 4, true);
      return out;
    case ParquetType.INT64:
      checkBytes(count * 8);
      for (let i = 0; i < count; i++) {
        const p = start + i * 8;
        const hi = view.getInt32(p + 4, true);
        // |value| <= 2^53 is exact as a double; larger values stay bigint.
        out[i] =
          hi >= -2097152 && hi < 2097152
            ? hi * TWO_32 + view.getUint32(p, true)
            : view.getBigInt64(p, true);
      }
      return out;
    case ParquetType.FLOAT:
      checkBytes(count * 4);
      for (let i = 0; i < count; i++) out[i] = view.getFloat32(start + i * 4, true);
      return out;
    case ParquetType.DOUBLE:
      checkBytes(count * 8);
      for (let i = 0; i < count; i++) out[i] = view.getFloat64(start + i * 8, true);
      return out;
    case ParquetType.BYTE_ARRAY: {
      let pos = start;
      const asString = col.kind === "string";
      for (let i = 0; i < count; i++) {
        if (pos + 4 > buf.length) throw truncated(what);
        const len = view.getUint32(pos, true);
        pos += 4;
        if (pos + len > buf.length) throw truncated(what);
        out[i] = asString
          ? textDecoder.decode(buf.subarray(pos, pos + len))
          : buf.slice(pos, pos + len);
        pos += len;
      }
      return out;
    }
    default:
      throw new DataValidationError(
        `readParquet: ${what} has the unsupported physical type ` +
          `${PHYSICAL_NAMES[ptype] ?? ptype}`
      );
  }
}

type PageHeader = {
  type: number;
  uncompressedSize: number;
  compressedSize: number;
  numValues: number;
  encoding: number;
  defLevelsLength: number;
  repLevelsLength: number;
  isCompressed: boolean;
  end: number;
};

function readPageHeader(buffer: Uint8Array, offset: number): PageHeader {
  const reader = new ThriftReader(buffer, offset);
  const h: PageHeader = {
    type: -1,
    uncompressedSize: 0,
    compressedSize: 0,
    numValues: 0,
    encoding: Encoding.PLAIN,
    defLevelsLength: 0,
    repLevelsLength: 0,
    isCompressed: true,
    end: offset,
  };
  let pf = reader.readFieldBegin();
  while (pf.type !== 0) {
    switch (pf.id) {
      case 1:
        h.type = reader.readI32();
        break;
      case 2:
        h.uncompressedSize = reader.readI32();
        break;
      case 3:
        h.compressedSize = reader.readI32();
        break;
      case 5: // data_page_header
      case 7: // dictionary_page_header
      case 8: {
        // data_page_header_v2
        if (pf.type !== TC_STRUCT) {
          reader.skipField(pf.type);
          break;
        }
        const kind = pf.id;
        reader.enterStruct();
        let df = reader.readFieldBegin();
        while (df.type !== 0) {
          if (df.id === 1 && df.type === TC_I32) h.numValues = reader.readI32();
          else if (kind === 8 && df.id === 4 && df.type === TC_I32) h.encoding = reader.readI32();
          else if (kind !== 8 && df.id === 2 && df.type === TC_I32) h.encoding = reader.readI32();
          else if (kind === 8 && df.id === 5 && df.type === TC_I32)
            h.defLevelsLength = reader.readI32();
          else if (kind === 8 && df.id === 6 && df.type === TC_I32)
            h.repLevelsLength = reader.readI32();
          else if (kind === 8 && df.id === 7 && df.type <= TC_BOOL_FALSE)
            h.isCompressed = readBoolField(df.type);
          else reader.skipField(df.type);
          df = reader.readFieldBegin();
        }
        reader.exitStruct();
        break;
      }
      default:
        reader.skipField(pf.type);
    }
    pf = reader.readFieldBegin();
  }
  h.end = reader.position;
  if (h.compressedSize < 0 || h.uncompressedSize < 0 || h.numValues < 0) {
    throw new DataValidationError("readParquet: corrupt page header");
  }
  return h;
}

function encodingName(encoding: number): string {
  return ENCODING_NAMES[encoding] ?? `encoding ${encoding}`;
}

/** Decode the values of one data page into `numValues` entries (nulls included). */
function decodeDataPage(
  header: PageHeader,
  raw: Uint8Array,
  codec: number,
  col: SchemaColumn,
  dictionary: readonly unknown[] | undefined
): unknown[] {
  const what = `column "${col.name}"`;
  const numValues = header.numValues;
  let payload: Uint8Array;
  let levels: Uint32Array | undefined;
  let pos = 0;

  if (header.type === PageType.DATA_PAGE_V2) {
    const levelBytes = header.repLevelsLength + header.defLevelsLength;
    if (levelBytes > raw.length) throw truncated(what);
    if (col.optional) {
      levels = decodeRleHybrid(
        raw,
        header.repLevelsLength,
        levelBytes,
        1,
        numValues,
        `${what} definition levels`
      );
    }
    const body = raw.subarray(levelBytes);
    payload =
      codec !== CompressionCodec.UNCOMPRESSED && header.isCompressed
        ? decompress(body, codec, header.uncompressedSize - levelBytes, what)
        : body;
  } else {
    payload = decompress(raw, codec, header.uncompressedSize, what);
    if (col.optional) {
      if (payload.length < 4) throw truncated(what);
      const levelLen = new DataView(
        payload.buffer,
        payload.byteOffset,
        payload.byteLength
      ).getUint32(0, true);
      pos = 4;
      if (pos + levelLen > payload.length) throw truncated(what);
      levels = decodeRleHybrid(
        payload,
        pos,
        pos + levelLen,
        1,
        numValues,
        `${what} definition levels`
      );
      pos += levelLen;
    }
  }

  let nonNull = numValues;
  if (levels) {
    nonNull = 0;
    for (let i = 0; i < levels.length; i++) {
      const level = levels[i]!;
      // A flat optional column only has definition levels 0 (null) and 1 (present).
      if (level > 1) {
        throw new DataValidationError(`readParquet: corrupt definition levels in ${what}`);
      }
      nonNull += level;
    }
  }

  let decoded: unknown[];
  switch (header.encoding) {
    case Encoding.PLAIN:
      decoded = decodeValuesPlain(payload, pos, nonNull, col);
      break;
    case Encoding.PLAIN_DICTIONARY:
    case Encoding.RLE_DICTIONARY: {
      if (!dictionary) {
        throw new DataValidationError(
          `readParquet: ${what} has a dictionary-encoded page without a dictionary`
        );
      }
      if (pos >= payload.length && nonNull > 0) throw truncated(what);
      const bitWidth = payload[pos] ?? 0;
      if (bitWidth > 32)
        throw new DataValidationError(`readParquet: corrupt dictionary indices in ${what}`);
      const indices = decodeRleHybrid(payload, pos + 1, payload.length, bitWidth, nonNull, what);
      decoded = new Array(nonNull);
      for (let i = 0; i < nonNull; i++) {
        const idx = indices[i]!;
        if (idx >= dictionary.length) {
          throw new DataValidationError(`readParquet: dictionary index out of range in ${what}`);
        }
        decoded[i] = dictionary[idx];
      }
      break;
    }
    case Encoding.RLE: {
      if (col.type !== ParquetType.BOOLEAN) {
        throw new DataValidationError(
          `readParquet: ${what} uses the unsupported RLE encoding for a non-boolean column`
        );
      }
      if (pos + 4 > payload.length) throw truncated(what);
      const len = new DataView(payload.buffer, payload.byteOffset, payload.byteLength).getUint32(
        pos,
        true
      );
      if (pos + 4 + len > payload.length) throw truncated(what);
      const bits = decodeRleHybrid(payload, pos + 4, pos + 4 + len, 1, nonNull, what);
      decoded = new Array(nonNull);
      for (let i = 0; i < nonNull; i++) decoded[i] = bits[i] === 1;
      break;
    }
    default:
      throw new DataValidationError(
        `readParquet: ${what} uses the unsupported ${encodingName(header.encoding)} encoding ` +
          "(supported: PLAIN, dictionary, RLE for booleans)"
      );
  }

  if (!levels) return decoded;
  const values: unknown[] = new Array(numValues);
  let next = 0;
  for (let i = 0; i < numValues; i++) values[i] = levels[i] ? decoded[next++] : null;
  return values;
}

/** Floor division that works for numbers and bigints and returns a number. */
function floorDiv(v: unknown, divisor: number): number {
  if (typeof v === "bigint") {
    const d = BigInt(divisor);
    let q = v / d;
    if (v % d < 0n) q -= 1n;
    return Number(q);
  }
  return Math.floor((v as number) / divisor);
}

/** Convert physical values into their logical JavaScript form, in place. */
function applyLogicalType(values: unknown[], col: SchemaColumn): void {
  switch (col.kind) {
    case "date":
      for (let i = 0; i < values.length; i++) {
        const v = values[i];
        if (v !== null) values[i] = new Date((v as number) * 86400000);
      }
      break;
    case "timestampMs":
      for (let i = 0; i < values.length; i++) {
        const v = values[i];
        if (v !== null) values[i] = new Date(Number(v));
      }
      break;
    case "timestampUs":
      for (let i = 0; i < values.length; i++) {
        const v = values[i];
        if (v !== null) values[i] = new Date(floorDiv(v, 1000));
      }
      break;
    case "timestampNs":
      for (let i = 0; i < values.length; i++) {
        const v = values[i];
        if (v !== null) values[i] = new Date(floorDiv(v, 1000000));
      }
      break;
    case "uint32":
      for (let i = 0; i < values.length; i++) {
        const v = values[i];
        if (v !== null) values[i] = (v as number) >>> 0;
      }
      break;
    case "uint64":
      for (let i = 0; i < values.length; i++) {
        const v = values[i];
        if (v === null) continue;
        if (typeof v === "number" && v >= 0) continue;
        const u = BigInt.asUintN(64, BigInt(v as number | bigint));
        values[i] = u <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(u) : u;
      }
      break;
    default:
      break;
  }
}

function decodeColumnChunk(
  buffer: Uint8Array,
  limit: number,
  chunk: ColumnChunkInfo,
  col: SchemaColumn
): unknown[] {
  const what = `column "${col.name}"`;
  if (col.unsupported !== undefined) {
    throw new DataValidationError(
      `readParquet: ${what} holds ${col.unsupported}, which is not supported`
    );
  }
  if (chunk.externalFile) {
    throw new DataValidationError(`readParquet: ${what} is stored in an external file`);
  }
  if (chunk.codec !== CompressionCodec.UNCOMPRESSED && !CODEC_NAMES[chunk.codec]) {
    throw new DataValidationError(
      `readParquet: ${what} uses an unknown compression codec ${chunk.codec}`
    );
  }

  if (chunk.numValues === 0) return [];

  let pos = chunk.dataPageOffset;
  if (chunk.dictionaryPageOffset > 0 && (pos < 0 || chunk.dictionaryPageOffset < pos)) {
    pos = chunk.dictionaryPageOffset;
  }
  if (pos < 4 || pos >= limit) {
    throw new DataValidationError(`readParquet: ${what} has an invalid page offset`);
  }

  const out: unknown[] = [];
  let dictionary: unknown[] | undefined;

  while (out.length < chunk.numValues) {
    if (pos >= limit) throw truncated(what);
    const header = readPageHeader(buffer, pos);
    const start = header.end;
    const end = start + header.compressedSize;
    if (end > limit) throw truncated(what);
    pos = end;
    const raw = buffer.subarray(start, end);

    if (header.type === PageType.DICTIONARY_PAGE) {
      if (header.encoding !== Encoding.PLAIN && header.encoding !== Encoding.PLAIN_DICTIONARY) {
        throw new DataValidationError(
          `readParquet: ${what} has a dictionary page with the unsupported ` +
            `${encodingName(header.encoding)} encoding`
        );
      }
      const payload = decompress(raw, chunk.codec, header.uncompressedSize, what);
      dictionary = decodeValuesPlain(payload, 0, header.numValues, col);
    } else if (header.type === PageType.INDEX_PAGE) {
      // Index pages carry no data.
    } else if (header.type === PageType.DATA_PAGE || header.type === PageType.DATA_PAGE_V2) {
      if (header.numValues > chunk.numValues - out.length) {
        throw new DataValidationError(
          `readParquet: ${what} holds more values than its metadata declares`
        );
      }
      const page = decodeDataPage(header, raw, chunk.codec, col, dictionary);
      for (let i = 0; i < page.length; i++) out.push(page[i]);
    } else {
      throw new DataValidationError(`readParquet: ${what} has an unknown page type ${header.type}`);
    }
  }

  if (out.length !== chunk.numValues) {
    throw new DataValidationError(
      `readParquet: ${what} holds more values than its metadata declares`
    );
  }
  applyLogicalType(out, col);
  return out;
}

/**
 * Read column data from a Parquet buffer.
 *
 * Supports flat schemas with any number of row groups and data pages (v1 and
 * v2), PLAIN / dictionary / RLE-boolean encodings and UNCOMPRESSED, SNAPPY or
 * GZIP (Node.js only) compression, which covers the defaults of pyarrow and
 * pandas. Anything outside that subset (nested columns, DECIMAL, INT96,
 * delta encodings, ZSTD/LZ4/Brotli) throws a descriptive error instead of
 * returning wrong data.
 *
 * Value mapping: BOOLEAN to `boolean`, INT32/FLOAT/DOUBLE to `number`, INT64
 * to `number` (or `bigint` when the value is beyond +/-2^53), strings to
 * `string`, unannotated BYTE_ARRAY to `Uint8Array`, DATE and TIMESTAMP
 * columns to `Date` (nanoseconds are truncated to milliseconds), unsigned
 * integers to non-negative values, nulls to `null`.
 *
 * @param buffer - The raw .parquet file as a Uint8Array
 * @param options - Read options (`columns` selects and orders a subset)
 * @returns Parsed data with columns and rows
 * @throws {DataValidationError} If the buffer is not a valid or supported Parquet file
 * @throws {InvalidParameterError} If `options.columns` names a column the file does not have
 */
export function readParquet(
  buffer: Uint8Array,
  options: ParquetReadOptions = {}
): ParquetReadResult {
  if (!(buffer instanceof Uint8Array)) {
    throw new DataValidationError("readParquet: buffer must be a Uint8Array");
  }
  if (buffer.length < 12) {
    throw new DataValidationError("readParquet: not a Parquet file (buffer is too short)");
  }
  if (
    !arrayEquals(buffer.subarray(0, 4), PARQUET_MAGIC) ||
    !arrayEquals(buffer.subarray(buffer.length - 4), PARQUET_MAGIC)
  ) {
    throw new DataValidationError('readParquet: not a Parquet file (missing "PAR1" magic bytes)');
  }

  const metaLen = new DataView(buffer.buffer, buffer.byteOffset + buffer.length - 8, 4).getUint32(
    0,
    true
  );
  if (metaLen === 0 || metaLen > buffer.length - 12) {
    throw new DataValidationError("readParquet: corrupt Parquet footer length");
  }
  const metaStart = buffer.length - 8 - metaLen;
  const meta = parseFileMetadata(buffer.subarray(metaStart, metaStart + metaLen));
  const schema = flatColumns(meta.schema);

  let selected: number[];
  if (options.columns) {
    const index = new Map<string, number>();
    schema.forEach((col, i) => {
      if (!index.has(col.name)) index.set(col.name, i);
    });
    const missing = options.columns.filter((name) => !index.has(name));
    if (missing.length > 0) {
      throw new InvalidParameterError(
        `readParquet: unknown column(s) ${missing.map((m) => `"${m}"`).join(", ")}; ` +
          `available columns: ${schema.map((s) => s.name).join(", ") || "(none)"}`,
        "columns",
        options.columns
      );
    }
    selected = [...new Set(options.columns)].map((name) => index.get(name)!);
  } else {
    selected = schema.map((_, i) => i);
  }

  let numRows = 0;
  for (const group of meta.rowGroups) {
    if (group.chunks.length !== schema.length) {
      throw new DataValidationError(
        `readParquet: a row group has ${group.chunks.length} column chunks but the schema has ${schema.length} columns`
      );
    }
    numRows += group.numRows;
  }
  if (numRows !== meta.numRows) {
    throw new DataValidationError(
      `readParquet: row group sizes (${numRows}) disagree with the file row count (${meta.numRows})`
    );
  }

  const columnData: unknown[][] = selected.map(() => []);
  for (const group of meta.rowGroups) {
    for (let s = 0; s < selected.length; s++) {
      const col = schema[selected[s]!]!;
      const values = decodeColumnChunk(buffer, metaStart, group.chunks[selected[s]!]!, col);
      if (values.length !== group.numRows) {
        throw new DataValidationError(
          `readParquet: column "${col.name}" has ${values.length} values but its row group has ${group.numRows} rows`
        );
      }
      const target = columnData[s]!;
      for (let i = 0; i < values.length; i++) target.push(values[i]);
    }
  }

  const outColumns = selected.map((i) => schema[i]!.name);
  const data: Record<string, unknown>[] = new Array(numRows);
  for (let r = 0; r < numRows; r++) {
    const row: Record<string, unknown> = {};
    for (let s = 0; s < outColumns.length; s++) setOwn(row, outColumns[s]!, columnData[s]![r]);
    data[r] = row;
  }

  return { columns: outColumns, data };
}
