import { describe, expect, it } from "vitest";
import { DataValidationError, InvalidParameterError } from "../../src/core/errors/index";
import { readParquet, readXlsx, writeParquet, writeXlsx } from "../../src/dataframe/index";
import { createKey, isValidNumber } from "../../src/dataframe/utils";

// Fixtures below were written by pyarrow 24 (store_schema=False) from the same
// 12-row table, so the reader is checked against files produced by another
// implementation:
//   i: int64 (null when i % 4 == 0)   s: "k<i % 3>" (null when i % 5 == 0)
//   b: i % 3 == 0                      u: uint32 4000000000 + i
//   ts: timestamp[ns] 2020-01-01 + i s d: date32 2020-01-01 + i days
const fromB64 = (b64: string) => new Uint8Array(Buffer.from(b64, "base64"));

const FIXTURES: Record<string, string> = {
  snappy_dict:
    "UEFSMRUEFZABFWBMFRIVABIAAEgEAQAJAQACCQcEAAMNCAAFDQgABg0IAAcNCAAJDQg8CgAAAAAAAAALAAAAAAAAABUAFSIVJiwV" +
    "GBUQFQYVBhwYCAsAAAAAAAAAGAgBAAAAAAAAABYGKAgLAAAAAAAAABgIAQAAAAAAAAAREQAAABFAAwAAAAXuDgQFEDJUdggAAAAV" +
    "BBUkFShMFQYVABIAABJEAgAAAGsxAgAAAGsyAgAAAGswFQAVGhUeLBUYFRAVBhUGHDYGKAJrMhgCazAREQAAAA0wAwAAAAXeCwIF" +
    "JJIBABUAFRAVFCwVGBUAFQYVBhwYAQEYAQAWACgBARgBABERAAAACBwCAAAAGAFJAhUEFWAVZEwVGBUAEgAAMLwAKGvuAShr7gIo" +
    "a+4DKGvuBChr7gUoa+4GKGvuByhr7ggoa+4JKGvuCihr7gsoa+4VABUgFSQsFRgVEBUGFQYcNgAoBAsoa+4YBAAoa+4REQAAABA8" +
    "AgAAABgBBAUQMlR2mLoAABUEFcABFaQBTBUYFQASAABgLAAAirk1muUVAMok9QUIDJS/MDYBEAheWmwFCAgo9acFCAjyj+MFCAy8" +
    "Kh83ASAIhsVaBQgIUGCWBQgIGvvRBQg45JUNOJrlFQCuMEk4muUVFQAVIBUkLBUYFRAVBhUGHBgIAK4wSTia5RUYCAAAirk1muUV" +
    "FgAoCACuMEk4muUVGAgAAIq5NZrlFRERAAAAEDwCAAAAGAEEBRAyVHaYugAAFQQVYBVkTBUYFQASAAAwvFZHAABXRwAAWEcAAFlH" +
    "AABaRwAAW0cAAFxHAABdRwAAXkcAAF9HAABgRwAAYUcAABUAFSAVJCwVGBUQFQYVBhwYBGFHAAAYBFZHAAAWACgEYUcAABgEVkcA" +
    "ABERAAAAEDwCAAAAGAEEBRAyVHaYugAAFQQZfDUAGAZzY2hlbWEVDAAVBCUCGAFpABUMJQIYAXMlAEwcAAAAFQAlAhgBYgAVAiUC" +
    "GAF1JRpMrBMgEgAAABUEJQIYAnRzbIwSHDwAAAAAABUCJQIYAWQlDExsAAAAFhgZHBlsJgAcFQQZNQAGEBkYAWkVAhYYFs4CFqIC" +
    "JoYBJggcGAgLAAAAAAAAABgIAQAAAAAAAAAWBigICwAAAAAAAAAYCAEAAAAAAAAAEREAGSwVBBUAFQIAFQAVEBUCADwpBhkmBhIA" +
    "AAAmABwVDBk1AAYQGRgBcxUCFhgWmAEWoAEm7gImqgIcNgYoAmsyGAJrMBERABksFQQVABUCABUAFRAVAgA8FiQZBhkmBhIAAAAm" +
    "ABwVABklBgAZGAFiFQIWGBZWFlomygM8GAEBGAEAFgAoAQEYAQAREQAZHBUAFQAVAgA8KQYZJgAYAAAAJgAcFQIZNQAGEBkYAXUV" +
    "AhYYFuIBFuoBJqQFJqQEHDYAKAQLKGvuGAQAKGvuEREAGSwVBBUAFQIAFQAVEBUCADwpBhkmABgAAAAmABwVBBk1AAYQGRgCdHMV" +
    "AhYYFv4CFuYCJtIHJo4GHBgIAK4wSTia5RUYCAAAirk1muUVFgAoCACuMEk4muUVGAgAAIq5NZrlFRERABksFQQVABUCABUAFRAV" +
    "AgA8KQYZJgAYAAAAJgAcFQIZNQAGEBkYAWQVAhYYFvoBFoICJvQJJvQIHBgEYUcAABgEVkcAABYAKARhRwAAGARWRwAAEREAGSwV" +
    "BBUAFQIAFQAVEBUCADwpBhkmABgAAAAWlgsWGCYIFu4KACggcGFycXVldC1jcHAtYXJyb3cgdmVyc2lvbiAyNC4wLjAZbBwAABwA" +
    "ABwAABwAABwAABwAAACZAgAAUEFSMQ==",
  gzip:
    "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",
  v2:
    "UEFSMRUGFZYBFZYBXBUYFQYVGBUAFQYVABIcGAgLAAAAAAAAABgIAQAAAAAAAAAWBigICwAAAAAAAAAYCAEAAAAAAAAAEREAAAAF" +
    "7g4BAAAAAAAAAAIAAAAAAAAAAwAAAAAAAAAFAAAAAAAAAAYAAAAAAAAABwAAAAAAAAAJAAAAAAAAAAoAAAAAAAAACwAAAAAAAAAV" +
    "BhVyFXJcFRgVBhUYFQAVBhUAEhw2BigCazIYAmswEREAAAAF3gsCAAAAazECAAAAazICAAAAazACAAAAazECAAAAazACAAAAazEC" +
    "AAAAazICAAAAazACAAAAazIVBhUSFRJcFRgVABUYFQYVBBUAEhwYAQEYAQAWACgBARgBABERAAAAGAEDAAAABUkCFQYVZBVkXBUY" +
    "FQAVGBUAFQQVABIcNgAoBAsoa+4YBAAoa+4REQAAABgBAChr7gEoa+4CKGvuAyhr7gQoa+4FKGvuBihr7gcoa+4IKGvuCShr7goo" +
    "a+4LKGvuFQYVxAEVxAFcFRgVABUYFQAVBBUAEhwYCACuMEk4muUVGAgAAIq5NZrlFRYAKAgArjBJOJrlFRgIAACKuTWa5RUREQAA" +
    "ABgBAACKuTWa5RUAyiT1NZrlFQCUvzA2muUVAF5abDaa5RUAKPWnNprlFQDyj+M2muUVALwqHzea5RUAhsVaN5rlFQBQYJY3muUV" +
    "ABr70Tea5RUA5JUNOJrlFQCuMEk4muUVFQYVZBVkXBUYFQAVGBUAFQQVABIcGARhRwAAGARWRwAAFgAoBGFHAAAYBFZHAAAREQAA" +
    "ABgBVkcAAFdHAABYRwAAWUcAAFpHAABbRwAAXEcAAF1HAABeRwAAX0cAAGBHAABhRwAAFQQZfDUAGAZzY2hlbWEVDAAVBCUCGAFp" +
    "ABUMJQIYAXMlAEwcAAAAFQAlAhgBYgAVAiUCGAF1JRpMrBMgEgAAABUEJQIYAnRzbIwSHDwAAAAAABUCJQIYAWQlDExsAAAAFhgZ" +
    "HBlsJgAcFQQZJQYAGRgBaRUAFhgWogIWogImCDwYCAsAAAAAAAAAGAgBAAAAAAAAABYGKAgLAAAAAAAAABgIAQAAAAAAAAAREQAZ" +
    "HBUAFQAVAgA8KQYZJgYSAAAAJgAcFQwZJQYAGRgBcxUAFhgWugEWugEmqgI8NgYoAmsyGAJrMBERABkcFQAVABUCADwWJBkGGSYG" +
    "EgAAACYAHBUAGRUGGRgBYhUAFhgWYhZiJuQDPBgBARgBABYAKAEBGAEAEREAGRwVABUGFQIAPCkGGSYAGAAAACYAHBUCGSUGABkY" +
    "AXUVABYYFrQBFrQBJsYEPDYAKAQLKGvuGAQAKGvuEREAGRwVABUAFQIAPCkGGSYAGAAAACYAHBUEGSUGABkYAnRzFQAWGBbQAhbQ" +
    "Aib6BTwYCACuMEk4muUVGAgAAIq5NZrlFRYAKAgArjBJOJrlFRgIAACKuTWa5RUREQAZHBUAFQAVAgA8KQYZJgAYAAAAJgAcFQIZ" +
    "JQYAGRgBZBUAFhgWzAEWzAEmygg8GARhRwAAGARWRwAAFgAoBGFHAAAYBFZHAAAREQAZHBUAFQAVAgA8KQYZJgAYAAAAFo4KFhgm" +
    "CBaOCgAoIHBhcnF1ZXQtY3BwLWFycm93IHZlcnNpb24gMjQuMC4wGWwcAAAcAAAcAAAcAAAcAAAcAAAAYQIAAFBBUjE=",
  multi:
    "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",
  zstd:
    "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",
  nested:
    "UEFSMRUEFTAVLkwVBhUAEgAAGAQBAAkBPAIAAAAAAAAAAwAAAAAAAAAVABUgFSQsFQYVEBUGFQYcGAgDAAAAAAAAABgIAQAAAAAA" +
    "AAAWACgIAwAAAAAAAAAYCAEAAAAAAAAAEREAAAAQPAIAAAADAgIAAAAGAwIDJAAVBBlMNQAYBnNjaGVtYRUCADUCGAFsFQIVBkw8" +
    "AAAANQQYBGxpc3QVAgAVBCUCGAdlbGVtZW50ABYEGRwZHCYAHBUEGTUABhAZOAFsBGxpc3QHZWxlbWVudBUCFgYW6gEW7AEmUiYI" +
    "HBgIAwAAAAAAAAAYCAEAAAAAAAAAFgAoCAMAAAAAAAAAGAgBAAAAAAAAABERABksFQQVABUCABUAFRAVAgA8KSYEAhlGAAAABgAA" +
    "ABbqARYEJggW7AEAKCBwYXJxdWV0LWNwcC1hcnJvdyB2ZXJzaW9uIDI0LjAuMBkcHAAAAOYAAABQQVIx",
};

const EXPECTED_ROWS = Array.from({ length: 12 }, (_, i) => ({
  i: i % 4 ? i : null,
  s: i % 5 ? `k${i % 3}` : null,
  b: i % 3 === 0,
  u: 4000000000 + i,
  ts: new Date(Date.UTC(2020, 0, 1, 0, 0, i)),
  d: new Date(Date.UTC(2020, 0, 1 + i)),
}));

describe("parquet reader: files from other writers", () => {
  it.each([
    "snappy_dict",
    "gzip",
    "v2",
    "multi",
  ])("reads the %s pyarrow file (codecs, dictionary, v2 pages, row groups, logical types)", (name) => {
    const out = readParquet(fromB64(FIXTURES[name]!));
    expect(out.columns).toEqual(["i", "s", "b", "u", "ts", "d"]);
    expect(out.data).toEqual(EXPECTED_ROWS);
  });

  it("returns unsigned 32-bit integers without wrapping to negatives", () => {
    const out = readParquet(fromB64(FIXTURES.snappy_dict!), { columns: ["u"] });
    expect(out.data[0]?.u).toBe(4000000000);
  });

  it("rejects unsupported compression and nested schemas instead of returning garbage", () => {
    expect(() => readParquet(fromB64(FIXTURES.zstd!))).toThrow(DataValidationError);
    expect(() => readParquet(fromB64(FIXTURES.zstd!))).toThrow(/ZSTD/);
    expect(() => readParquet(fromB64(FIXTURES.nested!))).toThrow(/nested/);
  });

  it("does not decode unselected columns of an unsupported codec", () => {
    // Schema is flat and valid; the zstd chunks only matter when selected.
    expect(() => readParquet(fromB64(FIXTURES.zstd!), { columns: [] })).not.toThrow();
  });

  it("returns columns in the requested order and rejects unknown names", () => {
    const out = readParquet(fromB64(FIXTURES.gzip!), { columns: ["d", "i"] });
    expect(out.columns).toEqual(["d", "i"]);
    expect(Object.keys(out.data[0]!)).toEqual(["d", "i"]);
    expect(() => readParquet(fromB64(FIXTURES.gzip!), { columns: ["nope"] })).toThrow(
      InvalidParameterError
    );
    expect(() => readParquet(fromB64(FIXTURES.gzip!), { columns: ["nope"] })).toThrow(
      /available columns: i, s, b, u, ts, d/
    );
  });
});

describe("parquet reader: invalid input", () => {
  it("throws a typed error for buffers that are not Parquet files", () => {
    expect(() => readParquet(new Uint8Array(0))).toThrow(DataValidationError);
    expect(() => readParquet(new Uint8Array(4))).toThrow(DataValidationError);
    expect(() => readParquet(new Uint8Array(64))).toThrow(/PAR1/);
  });

  it("throws on a corrupt footer length and on truncated files", () => {
    const good = writeParquet(["a"], [{ a: 1 }, { a: 2 }]);
    const badLen = good.slice();
    new DataView(badLen.buffer).setUint32(badLen.length - 8, 0x7fffffff, true);
    expect(() => readParquet(badLen)).toThrow(DataValidationError);
    expect(() => readParquet(good.subarray(0, good.length - 6))).toThrow(DataValidationError);
  });

  it("throws instead of silently padding with nulls when a page holds fewer values than declared", () => {
    const good = writeParquet(["a"], [{ a: 1 }, { a: 2 }, { a: 3 }]);
    const broken = good.slice();
    // Page header of the single REQUIRED INT32 column: type, sizes (12 bytes), then the data
    // page header whose num_values is the zigzag varint 6 (= 3 values). Make it claim 63.
    const idx = Buffer.from(broken).indexOf(
      Buffer.from([0x15, 0x00, 0x15, 0x18, 0x15, 0x18, 0x2c, 0x15, 0x06])
    );
    expect(idx).toBeGreaterThan(0);
    broken[idx + 8] = 0x7e;
    expect(() => readParquet(broken)).toThrow(DataValidationError);
  });

  it("rejects definition levels above 1 instead of reading values from the wrong rows", () => {
    const good = writeParquet(["a"], [{ a: 1 }, { a: null }]);
    const broken = good.slice();
    // Levels block: length 2, bit-packed header (one group), packed bits 0b01.
    const idx = Buffer.from(broken).indexOf(Buffer.from([0x02, 0x00, 0x00, 0x00, 0x03, 0x01]));
    expect(idx).toBeGreaterThan(0);
    broken[idx + 4] = 0x10; // RLE run of 8 ...
    broken[idx + 5] = 0x05; // ... of the level value 5
    expect(() => readParquet(broken)).toThrow(/definition levels/);
  });

  it("skips boolean elements of unknown list fields in the footer", () => {
    // Compact protocol stores list<bool> elements as one byte each; the old skipper
    // consumed none of them and mis-parsed everything after the list.
    const good = writeParquet(["a", "b"], [{ a: 1, b: "x" }]);
    const footerLen = new DataView(good.buffer, good.byteOffset).getUint32(good.length - 8, true);
    const footerStart = good.length - 8 - footerLen;
    const footerBody = good.subarray(footerStart, good.length - 9); // without the stop byte
    expect(good[good.length - 9]).toBe(0); // FileMetaData stop
    // field 20 (delta 14 from field 6), type LIST, 3 elements of type BOOL_TRUE
    const extra = Uint8Array.from([0xe9, 0x31, 1, 2, 1]);
    const newFooter = new Uint8Array(footerBody.length + extra.length + 1);
    newFooter.set(footerBody, 0);
    newFooter.set(extra, footerBody.length);
    const out = new Uint8Array(footerStart + newFooter.length + 8);
    out.set(good.subarray(0, footerStart), 0);
    out.set(newFooter, footerStart);
    new DataView(out.buffer).setUint32(footerStart + newFooter.length, newFooter.length, true);
    out.set([0x50, 0x41, 0x52, 0x31], footerStart + newFooter.length + 4);
    expect(readParquet(out).data).toEqual([{ a: 1, b: "x" }]);
  });
});

describe("parquet writer", () => {
  it("round-trips Date columns as TIMESTAMP_MILLIS and treats invalid dates as null", () => {
    const rows = [
      { t: new Date(Date.UTC(2020, 0, 2, 3, 4, 5, 678)) },
      { t: new Date(-1000) },
      { t: new Date(Number.NaN) },
    ];
    const out = readParquet(writeParquet(["t"], rows));
    expect(out.data.map((r) => (r.t as Date | null)?.getTime() ?? null)).toEqual([
      Date.UTC(2020, 0, 2, 3, 4, 5, 678),
      -1000,
      null,
    ]);
  });

  it("writes integers outside the int64 range as DOUBLE instead of wrapping", () => {
    // 1e30 used to be sent through BigInt.asIntN-style wrapping and came back as garbage.
    const out = readParquet(writeParquet(["v"], [{ v: 1e30 }, { v: -1e30 }, { v: 5 }]));
    expect(out.data.map((r) => r.v)).toEqual([1e30, -1e30, 5]);
  });

  it("keeps bigint precision beyond 2^53 and mixes bigints with integer numbers", () => {
    const rows = [{ v: 2n ** 62n }, { v: 7 }, { v: -(2n ** 63n) }];
    const out = readParquet(writeParquet(["v"], rows));
    expect(out.data.map((r) => r.v)).toEqual([2n ** 62n, 7, -(2n ** 63n)]);
  });

  it("rejects bigints outside the INT64 range", () => {
    expect(() => writeParquet(["v"], [{ v: 2n ** 63n }])).toThrow(DataValidationError);
    expect(() => writeParquet(["v"], [{ v: 2n ** 63n }])).toThrow(/INT64 range/);
  });

  it("keeps negative zero (a column with -0 is DOUBLE, not INT32)", () => {
    const out = readParquet(writeParquet(["v"], [{ v: -0 }, { v: 3 }]));
    expect(Object.is(out.data[0]?.v, -0)).toBe(true);
    expect(out.data[1]?.v).toBe(3);
  });

  it("writes NaN as a DOUBLE value, not as a null", () => {
    const out = readParquet(writeParquet(["v"], [{ v: Number.NaN }, { v: 1 }, { v: null }]));
    expect(out.data[0]?.v).toBeNaN();
    expect(out.data[2]?.v).toBeNull();
  });

  it("falls back to text for mixed columns and stringifies dates as ISO 8601", () => {
    const out = readParquet(
      writeParquet(
        ["m", "n"],
        [
          { m: 1, n: new Date(Date.UTC(2020, 0, 1)) },
          { m: "a", n: 1 },
          { m: true, n: null },
        ]
      )
    );
    expect(out.data.map((r) => r.m)).toEqual(["1", "a", "true"]);
    expect(out.data.map((r) => r.n)).toEqual(["2020-01-01T00:00:00.000Z", "1", null]);
  });

  it("rejects duplicate column names and non-object rows", () => {
    expect(() => writeParquet(["a", "a"], [{ a: 1 }])).toThrow(/duplicate column name "a"/);
    expect(() => writeParquet(["a"], [null as unknown as Record<string, unknown>])).toThrow(
      DataValidationError
    );
  });

  it("ignores inherited properties and handles special column names", () => {
    // row["toString"] used to resolve to Object.prototype.toString and be written as text.
    const out = readParquet(
      writeParquet(["toString", "__proto__", "constructor"], [{ toString: "x" }, {}])
    );
    expect(out.columns).toEqual(["toString", "__proto__", "constructor"]);
    expect(out.data[0]?.toString).toBe("x");
    expect(out.data[1]?.toString).toBeNull();
    expect(Object.getOwnPropertyDescriptor(out.data[0]!, "__proto__")?.value).toBeNull();
    expect(Object.getPrototypeOf(out.data[0]!)).toBe(Object.prototype);
  });

  it("round-trips more than 8 rows with sparse nulls (definition-level tail)", () => {
    const rows = Array.from({ length: 21 }, (_, i) => ({ v: i % 5 === 0 ? null : i }));
    expect(readParquet(writeParquet(["v"], rows)).data).toEqual(rows);
  });

  it("writes a zero-column file and a zero-row file", () => {
    expect(readParquet(writeParquet([], [])).data).toEqual([]);
    const out = readParquet(writeParquet(["a", "b"], []));
    expect(out.columns).toEqual(["a", "b"]);
    expect(out.data).toEqual([]);
  });
});

// ─── XLSX ────────────────────────────────────────────────────────────────────

type ZipSpec = { name: string; data: string; method?: number };

/** Build a stored zip archive (the reader does not check CRCs). */
function buildZip(entries: ZipSpec[]): Uint8Array {
  const enc = new TextEncoder();
  const chunks: Buffer[] = [];
  const central: Buffer[] = [];
  let offset = 0;
  for (const e of entries) {
    const name = Buffer.from(e.name);
    const body = Buffer.from(enc.encode(e.data));
    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0);
    local.writeUInt16LE(20, 4);
    local.writeUInt16LE(e.method ?? 0, 8);
    local.writeUInt32LE(body.length, 18);
    local.writeUInt32LE(body.length, 22);
    local.writeUInt16LE(name.length, 26);
    chunks.push(local, name, body);
    const cd = Buffer.alloc(46);
    cd.writeUInt32LE(0x02014b50, 0);
    cd.writeUInt16LE(20, 4);
    cd.writeUInt16LE(20, 6);
    cd.writeUInt16LE(e.method ?? 0, 10);
    cd.writeUInt32LE(body.length, 20);
    cd.writeUInt32LE(body.length, 24);
    cd.writeUInt16LE(name.length, 28);
    cd.writeUInt32LE(offset, 42);
    central.push(cd, name);
    offset += local.length + name.length + body.length;
  }
  const cdBuf = Buffer.concat(central);
  const eocd = Buffer.alloc(22);
  eocd.writeUInt32LE(0x06054b50, 0);
  eocd.writeUInt16LE(entries.length, 8);
  eocd.writeUInt16LE(entries.length, 10);
  eocd.writeUInt32LE(cdBuf.length, 12);
  eocd.writeUInt32LE(offset, 16);
  return new Uint8Array(Buffer.concat([...chunks, cdBuf, eocd]));
}

const NS = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"';

/** A workbook with the given worksheet bodies (<sheetData> content) and optional shared strings. */
function workbookZip(
  sheets: { name: string; rows: string }[],
  sharedStringsXml?: string,
  extra: ZipSpec[] = []
): Uint8Array {
  const entries: ZipSpec[] = [
    {
      name: "xl/workbook.xml",
      data: `<workbook ${NS} xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>${sheets
        .map((s, i) => `<sheet name="${s.name}" sheetId="${i + 1}" r:id="rId${i + 1}"/>`)
        .join("")}</sheets></workbook>`,
    },
    {
      name: "xl/_rels/workbook.xml.rels",
      data: `<Relationships>${sheets
        .map(
          (_, i) =>
            `<Relationship Id="rId${i + 1}" Type="worksheet" Target="worksheets/sheet${i + 1}.xml"/>`
        )
        .join("")}</Relationships>`,
    },
    ...sheets.map((s, i) => ({
      name: `xl/worksheets/sheet${i + 1}.xml`,
      data: `<worksheet ${NS}><sheetData>${s.rows}</sheetData></worksheet>`,
    })),
    ...extra,
  ];
  if (sharedStringsXml !== undefined) {
    entries.push({ name: "xl/sharedStrings.xml", data: `<sst ${NS}>${sharedStringsXml}</sst>` });
  }
  return buildZip(entries);
}

describe("xlsx reader: real-world sheet XML", () => {
  it("does not merge self-closing styled cells into the next cell", () => {
    // Excel writes empty formatted cells as <c r="A1" s="1"/>; the old tag scanner ran
    // from such a tag to the next </c> and returned B1's value for A1.
    const buf = workbookZip([
      {
        name: "S",
        rows:
          '<row r="1"><c r="A1" t="inlineStr"><is><t>x</t></is></c><c r="B1" t="inlineStr"><is><t>y</t></is></c></row>' +
          '<row r="2"><c r="A2" s="1"/><c r="B2"><v>5</v></c></row>',
      },
    ]);
    expect(readXlsx(buf)).toEqual({ columns: ["x", "y"], data: [{ x: null, y: 5 }] });
  });

  it("reads xml:space=preserve, rich-text runs and skips phonetic text", () => {
    const buf = workbookZip(
      [
        {
          name: "S",
          rows:
            '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>' +
            '<row r="2"><c r="A2" t="s"><v>2</v></c><c r="B2" t="s"><v>3</v></c></row>',
        },
      ],
      "<si><t>h1</t></si><si><t>h2</t></si>" +
        '<si><t xml:space="preserve">  lead and trail  </t></si>' +
        '<si><r><t>Hel</t></r><r><rPr><b/></rPr><t xml:space="preserve">lo </t></r><r><t>world</t></r>' +
        '<rPh sb="0" eb="1"><t>PHONETIC</t></rPh></si>'
    );
    expect(readXlsx(buf).data).toEqual([{ h1: "  lead and trail  ", h2: "Hello world" }]);
  });

  it("drops formatted blank rows after the data even when the dimension covers them", () => {
    // Excel and openpyxl leave styled empty rows behind; pandas does not return them.
    const buf = buildZip([
      {
        name: "xl/worksheets/sheet1.xml",
        data:
          `<worksheet ${NS}><dimension ref="A1:A6"/><sheetData>` +
          '<row r="1"><c r="A1" t="inlineStr"><is><t>h</t></is></c></row>' +
          '<row r="2"><c r="A2"><v>1</v></c></row>' +
          '<row r="3"/><row r="4"><c r="A4" s="1"/></row>' +
          '<row r="5" s="2" customFormat="1"/><row r="6" ht="30" customHeight="1"/>' +
          "</sheetData></worksheet>",
      },
    ]);
    // Row 3 is a plain empty row inside the dimension, which is how writeXlsx stores an
    // all-null row, so it stays. Rows 4-6 carry formatting and are dropped.
    expect(readXlsx(buf).data).toEqual([{ h: 1 }, { h: null }]);
  });

  it("keeps trailing all-null rows written by writeXlsx", () => {
    const rows = [{ a: 1 }, { a: null }, { a: null }];
    expect(readXlsx(writeXlsx(["a"], rows)).data).toEqual(rows);
  });

  it("matches sheet names that contain text looking like an Excel escape", () => {
    const buf = writeXlsx(["a"], [{ a: 1 }], { sheetName: "_x0041_ sheet" });
    expect(readXlsx(buf, { sheet: "_x0041_ sheet" }).data).toEqual([{ a: 1 }]);
  });

  it("keeps blank rows inside the data and drops leading and trailing empty rows", () => {
    const buf = workbookZip([
      {
        name: "S",
        rows:
          '<row r="1"><c r="A1" s="2"/></row>' +
          '<row r="3"><c r="A3" t="inlineStr"><is><t>h</t></is></c></row>' +
          '<row r="4"><c r="A4"><v>1</v></c></row>' +
          '<row r="7"><c r="A7"><v>2</v></c></row>' +
          '<row r="9"><c r="A9" s="1"/></row><row r="10" spans="1:1"/>',
      },
    ]);
    expect(readXlsx(buf).data).toEqual([{ h: 1 }, { h: null }, { h: null }, { h: 2 }]);
  });

  it("places cells without an r attribute sequentially instead of stacking them in column A", () => {
    const buf = workbookZip([
      {
        name: "S",
        rows:
          '<row><c t="inlineStr"><is><t>a</t></is></c><c t="inlineStr"><is><t>b</t></is></c></row>' +
          "<row><c><v>1</v></c><c><v>2</v></c></row>",
      },
    ]);
    expect(readXlsx(buf)).toEqual({ columns: ["a", "b"], data: [{ a: 1, b: 2 }] });
  });

  it("suffixes repeated headers like pandas instead of overwriting the first column", () => {
    // pandas.read_excel on headers [a, a, a.1, a, b, b] gives [a, a.2, a.1, a.3, b, b.1].
    const header = ["a", "a", "a.1", "a", "b", "b"]
      .map((h, i) => `<c r="${"ABCDEF"[i]}1" t="inlineStr"><is><t>${h}</t></is></c>`)
      .join("");
    const body = [1, 2, 3, 4, 5, 6]
      .map((v, i) => `<c r="${"ABCDEF"[i]}2"><v>${v}</v></c>`)
      .join("");
    const buf = workbookZip([
      { name: "S", rows: `<row r="1">${header}</row><row r="2">${body}</row>` },
    ]);
    const out = readXlsx(buf);
    expect(out.columns).toEqual(["a", "a.2", "a.1", "a.3", "b", "b.1"]);
    expect(out.data).toEqual([{ a: 1, "a.2": 2, "a.1": 3, "a.3": 4, b: 5, "b.1": 6 }]);
  });

  it("maps formula errors to null and reads cached formula strings and booleans", () => {
    const buf = workbookZip([
      {
        name: "S",
        rows:
          '<row r="1"><c r="A1" t="inlineStr"><is><t>e</t></is></c><c r="B1" t="inlineStr"><is><t>f</t></is></c><c r="C1" t="inlineStr"><is><t>g</t></is></c></row>' +
          '<row r="2"><c r="A2" t="e"><f>1/0</f><v>#DIV/0!</v></c><c r="B2" t="str"><f>"a"&amp;"b"</f><v>ab</v></c><c r="C2" t="b"><v>1</v></c></row>',
      },
    ]);
    expect(readXlsx(buf).data).toEqual([{ e: null, f: "ab", g: true }]);
  });

  it("decodes hex and decimal character references and Excel _xHHHH_ escapes", () => {
    const buf = workbookZip(
      [
        {
          name: "S",
          rows: '<row r="1"><c r="A1" t="s"><v>0</v></c></row><row r="2"><c r="A2" t="s"><v>1</v></c></row>',
        },
      ],
      "<si><t>h</t></si><si><t>&#x41;&#66;_x0043_&#1114112;_x005F_x0044_</t></si>"
    );
    // &#1114112; is beyond U+10FFFF and stays as written instead of throwing a RangeError.
    expect(readXlsx(buf).data).toEqual([{ h: "ABC&#1114112;_x0044_" }]);
  });

  it("does not turn hex-looking or non-decimal numeric text into numbers", () => {
    const buf = workbookZip([
      {
        name: "S",
        rows:
          '<row r="1"><c r="A1" t="inlineStr"><is><t>h</t></is></c></row>' +
          '<row r="2"><c r="A2"><v>0x10</v></c></row><row r="3"><c r="A3"><v>1.5E-3</v></c></row>' +
          '<row r="4"><c r="A4"><v>.5</v></c></row>',
      },
    ]);
    expect(readXlsx(buf).data).toEqual([{ h: "0x10" }, { h: 0.0015 }, { h: 0.5 }]);
  });

  it("selects sheets by name or position and reports bad selections", () => {
    const rowsOf = (v: number) =>
      `<row r="1"><c r="A1" t="inlineStr"><is><t>v</t></is></c></row><row r="2"><c r="A2"><v>${v}</v></c></row>`;
    const buf = workbookZip([
      { name: "One", rows: rowsOf(1) },
      { name: "Two", rows: rowsOf(2) },
    ]);
    expect(readXlsx(buf).data).toEqual([{ v: 1 }]);
    expect(readXlsx(buf, { sheet: 1 }).data).toEqual([{ v: 2 }]);
    expect(readXlsx(buf, { sheet: "Two" }).data).toEqual([{ v: 2 }]);
    expect(() => readXlsx(buf, { sheet: 2 })).toThrow(/out of range; available sheets: One, Two/);
    expect(() => readXlsx(buf, { sheet: -1 })).toThrow(DataValidationError);
    expect(() => readXlsx(buf, { sheet: "Three" })).toThrow(DataValidationError);
  });

  it("fails loudly when the named sheet's part is missing instead of reading another sheet", () => {
    const entries: ZipSpec[] = [
      {
        name: "xl/workbook.xml",
        data: `<workbook ${NS} xmlns:r="x"><sheets><sheet name="A" sheetId="1" r:id="rId1"/><sheet name="B" sheetId="2" r:id="rId9"/></sheets></workbook>`,
      },
      {
        name: "xl/_rels/workbook.xml.rels",
        data: '<Relationships><Relationship Id="rId1" Type="w" Target="worksheets/sheet1.xml"/></Relationships>',
      },
      { name: "xl/worksheets/sheet1.xml", data: `<worksheet ${NS}><sheetData/></worksheet>` },
    ];
    expect(() => readXlsx(buildZip(entries), { sheet: "B" })).toThrow(/worksheet part/);
  });

  it("resolves absolute and parent-relative worksheet targets", () => {
    const entries: ZipSpec[] = [
      {
        name: "xl/workbook.xml",
        data: `<workbook ${NS} xmlns:r="x"><sheets><sheet name="A" sheetId="1" r:id="rId1"/></sheets></workbook>`,
      },
      {
        name: "xl/_rels/workbook.xml.rels",
        data: '<Relationships><Relationship Id="rId1" Type="w" Target="/xl/worksheets/s.xml"/></Relationships>',
      },
      {
        name: "xl/worksheets/s.xml",
        data: `<worksheet ${NS}><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>q</t></is></c></row><row r="2"><c r="A2"><v>3</v></c></row></sheetData></worksheet>`,
      },
    ];
    expect(readXlsx(buildZip(entries)).data).toEqual([{ q: 3 }]);
    entries[1] = {
      name: "xl/_rels/workbook.xml.rels",
      data: '<Relationships><Relationship Id="rId1" Type="w" Target="../xl/worksheets/s.xml"/></Relationships>',
    };
    expect(readXlsx(buildZip(entries)).data).toEqual([{ q: 3 }]);
  });

  it("only decompresses the parts it needs", () => {
    // A part with an unsupported compression method is irrelevant for reading cells.
    const buf = workbookZip(
      [
        {
          name: "S",
          rows: '<row r="1"><c r="A1" t="inlineStr"><is><t>h</t></is></c></row><row r="2"><c r="A2"><v>1</v></c></row>',
        },
      ],
      undefined,
      [{ name: "docProps/thumbnail.jpeg", data: "zz", method: 99 }]
    );
    expect(readXlsx(buf).data).toEqual([{ h: 1 }]);
  });

  it("reads columns with __proto__ headers as own properties", () => {
    const buf = workbookZip([
      {
        name: "S",
        rows: '<row r="1"><c r="A1" t="inlineStr"><is><t>__proto__</t></is></c></row><row r="2"><c r="A2"><v>1</v></c></row>',
      },
    ]);
    const row = readXlsx(buf).data[0]!;
    expect(Object.getOwnPropertyDescriptor(row, "__proto__")?.value).toBe(1);
    expect(Object.getPrototypeOf(row)).toBe(Object.prototype);
  });

  it("returns empty results for a valid zip that has no worksheet", () => {
    expect(readXlsx(buildZip([{ name: "a.txt", data: "x" }]))).toEqual({ columns: [], data: [] });
  });
});

describe("xlsx reader: invalid input", () => {
  it("throws a typed error for non-zip, truncated and corrupt archives", () => {
    expect(() => readXlsx(new Uint8Array(0))).toThrow(DataValidationError);
    expect(() => readXlsx(new Uint8Array(64))).toThrow(/not a valid \.xlsx file/);
    const good = writeXlsx(["a"], [{ a: 1 }]);
    expect(() => readXlsx(good.subarray(0, good.length - 30))).toThrow(DataValidationError);
    const badLocal = good.slice();
    // Break the local header signature of xl/workbook.xml, which the reader needs.
    const raw = Buffer.from(badLocal);
    let workbookHeader = -1;
    for (let at = raw.indexOf("PK\x03\x04"); at >= 0; at = raw.indexOf("PK\x03\x04", at + 1)) {
      if (raw.toString("latin1", at + 30, at + 45) === "xl/workbook.xml") workbookHeader = at;
    }
    expect(workbookHeader).toBeGreaterThan(0);
    badLocal[workbookHeader] = 0;
    expect(() => readXlsx(badLocal)).toThrow(/corrupt local header/);
  });

  it("validates maxEntryBytes", () => {
    const good = writeXlsx(["a"], [{ a: 1 }]);
    expect(() => readXlsx(good, { maxEntryBytes: 0 })).toThrow(InvalidParameterError);
    expect(() => readXlsx(good, { maxEntryBytes: Number.NaN })).toThrow(InvalidParameterError);
  });

  it("rejects out-of-range cell and row references", () => {
    const wide = workbookZip([{ name: "S", rows: '<row r="1"><c r="ZZZZ1"><v>1</v></c></row>' }]);
    expect(() => readXlsx(wide)).toThrow(/invalid cell reference/);
    const tall = workbookZip([{ name: "S", rows: '<row r="99999999"><c r="A1"/></row>' }]);
    expect(() => readXlsx(tall)).toThrow(/invalid row reference/);
  });

  it("rejects an out-of-range shared string index", () => {
    const buf = workbookZip(
      [{ name: "S", rows: '<row r="1"><c r="A1" t="s"><v>5</v></c></row>' }],
      "<si><t>only</t></si>"
    );
    expect(() => readXlsx(buf)).toThrow(/shared string index 5/);
  });
});

describe("xlsx writer", () => {
  it("writes NaN as an empty cell and infinities as text (they are not valid cell numbers)", () => {
    // "NaN"/"Infinity" in a <v> element make Excel report a corrupt file.
    const out = readXlsx(
      writeXlsx(
        ["v"],
        [{ v: Number.NaN }, { v: Number.POSITIVE_INFINITY }, { v: Number.NEGATIVE_INFINITY }]
      )
    );
    expect(out.data.map((r) => r.v)).toEqual([null, "inf", "-inf"]);
  });

  it("writes dates as ISO text, bigints as numbers when safe and text otherwise", () => {
    const out = readXlsx(
      writeXlsx(
        ["d", "n"],
        [
          { d: new Date(Date.UTC(2020, 0, 2, 3, 4, 5, 6)), n: 5n },
          { d: new Date(Number.NaN), n: 2n ** 60n },
        ]
      )
    );
    expect(out.data).toEqual([
      { d: "2020-01-02T03:04:05.006Z", n: 5 },
      { d: null, n: "1152921504606846976" },
    ]);
  });

  it("round-trips control characters, CR and text that looks like an escape", () => {
    const tricky = ["a\u0001b", "line1\r\nline2", "_x0041_ stays", "tab\there", "￾"];
    const out = readXlsx(
      writeXlsx(
        ["t"],
        tricky.map((t) => ({ t }))
      )
    );
    expect(out.data.map((r) => r.t)).toEqual(tricky);
  });

  it("marks leading and trailing whitespace as significant", () => {
    const buf = writeXlsx(["t"], [{ t: "  x " }, { t: "y" }]);
    const raw = Buffer.from(buf).toString("latin1");
    expect(raw).toContain('<t xml:space="preserve">  x </t>');
    expect(raw).toContain("<t>y</t>");
    expect(readXlsx(buf).data).toEqual([{ t: "  x " }, { t: "y" }]);
  });

  it("reports the real number of string references in the shared string table", () => {
    // 1 header + 3 text cells, of which "a" appears twice: count=4, uniqueCount=3.
    const buf = writeXlsx(["h"], [{ h: "a" }, { h: "a" }, { h: "b" }]);
    expect(Buffer.from(buf).toString("latin1")).toContain('count="4" uniqueCount="3"');
  });

  it("stamps entries with a valid zip date", () => {
    const buf = writeXlsx(["a"], [{ a: 1 }]);
    // Local header: mod time at byte 10, mod date at byte 12 (0x0021 = 1980-01-01).
    expect(buf[12]! | (buf[13]! << 8)).toBe(0x0021);
  });

  it("validates sheet names against Excel's rules", () => {
    for (const bad of [
      "",
      "a/b",
      "a:b",
      "x?",
      "y*",
      "[z]",
      "'q",
      "q'",
      "a\u0001",
      "x".repeat(32),
    ]) {
      expect(() => writeXlsx(["a"], [], { sheetName: bad })).toThrow(InvalidParameterError);
    }
    expect(() => writeXlsx(["a"], [], { sheetName: "x".repeat(31) })).not.toThrow();
    const buf = writeXlsx(["a"], [{ a: 1 }], { sheetName: "R&D <1>" });
    expect(readXlsx(buf, { sheet: "R&D <1>" }).data).toEqual([{ a: 1 }]);
  });

  it("rejects duplicate columns, oversized cells and non-object rows", () => {
    expect(() => writeXlsx(["a", "a"], [])).toThrow(/duplicate column name "a"/);
    expect(() => writeXlsx(["a"], [{ a: "x".repeat(32768) }])).toThrow(/32767/);
    expect(() => writeXlsx(["a"], [{ a: "x".repeat(32767) }])).not.toThrow();
    expect(() => writeXlsx(["a"], [null as unknown as Record<string, unknown>])).toThrow(
      DataValidationError
    );
  });

  it("writes a valid sheet when there are no columns or rows", () => {
    // The dimension used to be "A1:1", which is not a valid reference.
    const noCols = writeXlsx([], [{}]);
    expect(Buffer.from(noCols).toString("latin1")).toContain('<dimension ref="A1"/>');
    expect(readXlsx(noCols)).toEqual({ columns: [], data: [] });
    const noRows = writeXlsx(["a", "b"], []);
    expect(readXlsx(noRows)).toEqual({ columns: ["a", "b"], data: [] });
  });

  it("does not read inherited properties as cell values", () => {
    const out = readXlsx(writeXlsx(["constructor"], [{}]));
    expect(out.data).toEqual([{ constructor: null }]);
  });

  it("round-trips header-only and sparse data through the new reader", () => {
    const rows = [
      { a: 1, b: null, c: "x" },
      { a: null, b: null, c: null },
      { a: 3, b: 4, c: null },
    ];
    expect(readXlsx(writeXlsx(["a", "b", "c"], rows)).data).toEqual(rows);
  });
});

// ─── utils ───────────────────────────────────────────────────────────────────

describe("createKey", () => {
  it("does not collide for composite keys whose parts contain separators", () => {
    // Both used to produce "[s:a,s:b]".
    expect(createKey(["a,s:b"])).not.toBe(createKey(["a", "b"]));
    expect(createKey(["a", "b,c"])).not.toBe(createKey(["a,s:b", "c"]));
    expect(createKey({ "a:s:b": 1 })).not.toBe(createKey({ a: "b" }));
    expect(createKey([["x"], "y"])).not.toBe(createKey(["[s:x]", "y"]));
  });

  it("separates types and treats equal values alike", () => {
    const keys = [1, "1", 1n, true, null, undefined, Number.NaN, Infinity, -Infinity].map(
      createKey
    );
    expect(new Set(keys).size).toBe(keys.length);
    expect(createKey(0)).toBe(createKey(-0));
    expect(createKey(Number.NaN)).toBe(createKey(Number.NaN));
    expect(createKey({ a: 1, b: 2 })).toBe(createKey({ b: 2, a: 1 }));
  });

  it("keys dates by timestamp and distinguishes maps, sets and typed arrays", () => {
    expect(createKey(new Date(5))).toBe(createKey(new Date(5)));
    expect(createKey(new Date(5))).not.toBe(createKey(new Date(6)));
    // These all used to hash to "{}".
    expect(createKey(new Map([["a", 1]]))).not.toBe(createKey(new Map([["a", 2]])));
    expect(createKey(new Map([["a", 1]]))).toBe(createKey(new Map([["a", 1]])));
    expect(createKey(new Set([1, 2]))).toBe(createKey(new Set([2, 1])));
    expect(createKey(new Set([1, 2]))).not.toBe(createKey(new Set([1, 3])));
    expect(createKey(new Float64Array([1, 2]))).not.toBe(createKey(new Float64Array([1, 3])));
    expect(createKey(/a/g)).not.toBe(createKey(/a/i));
  });

  it("keys circular structures instead of overflowing the stack", () => {
    const a: Record<string, unknown> = { n: 1 };
    a.self = a;
    expect(() => createKey(a)).not.toThrow();
    const shared = { z: 1 };
    // The same object twice is not a cycle.
    expect(createKey([shared, shared])).toBe(createKey([{ z: 1 }, { z: 1 }]));
  });
});

describe("isValidNumber", () => {
  it("accepts only finite numbers", () => {
    expect(isValidNumber(1.5)).toBe(true);
    expect(isValidNumber(0)).toBe(true);
    for (const v of [Number.NaN, Infinity, -Infinity, "1", null, undefined, 1n, true]) {
      expect(isValidNumber(v)).toBe(false);
    }
  });
});
