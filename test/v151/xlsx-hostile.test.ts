import { describe, expect, it } from "vitest";
import { DataValidationError } from "../../src/core/errors/index";
import { readXlsx } from "../../src/dataframe/index";

// Builds a single-entry stored (uncompressed) zip holding one worksheet part.
function u32(v: number): number[] {
  return [v & 255, (v >> 8) & 255, (v >> 16) & 255, (v >>> 24) & 255];
}
function u16(v: number): number[] {
  return [v & 255, (v >> 8) & 255];
}
function zipSheet(xml: string): Uint8Array {
  const data = new TextEncoder().encode(xml);
  const name = new TextEncoder().encode("xl/worksheets/sheet1.xml");
  const lh = [
    0x50,
    0x4b,
    3,
    4,
    ...u16(20),
    ...u16(0),
    ...u16(0),
    ...u16(0),
    ...u16(0x21),
    ...u32(0),
    ...u32(data.length),
    ...u32(data.length),
    ...u16(name.length),
    ...u16(0),
    ...name,
  ];
  const cd = [
    0x50,
    0x4b,
    1,
    2,
    ...u16(20),
    ...u16(20),
    ...u16(0),
    ...u16(0),
    ...u16(0),
    ...u16(0x21),
    ...u32(0),
    ...u32(data.length),
    ...u32(data.length),
    ...u16(name.length),
    ...u16(0),
    ...u16(0),
    ...u16(0),
    ...u16(0),
    ...u32(0),
    ...u32(0),
    ...name,
  ];
  const eocd = [
    0x50,
    0x4b,
    5,
    6,
    0,
    0,
    0,
    0,
    ...u16(1),
    ...u16(1),
    ...u32(cd.length),
    ...u32(lh.length + data.length),
    0,
    0,
  ];
  const out = new Uint8Array(lh.length + data.length + cd.length + eocd.length);
  out.set(lh, 0);
  out.set(data, lh.length);
  out.set(cd, lh.length + data.length);
  out.set(eocd, lh.length + data.length + cd.length);
  return out;
}
const sheet = (rows: string) =>
  `<?xml version="1.0"?><worksheet><sheetData>${rows}</sheetData></worksheet>`;

describe("readXlsx hostile input (DF-1: sparse cells and a cell cap)", () => {
  it("rejects a sheet whose rows x columns exceeds the cell cap without padding rows", () => {
    let rows = "";
    for (let r = 1; r <= 20000; r++) rows += `<row r="${r}"><c r="XFD${r}"><v>1</v></c></row>`;
    const file = zipSheet(sheet(rows));
    expect(() => readXlsx(file, { header: false })).toThrow(DataValidationError);
    expect(() => readXlsx(file, { header: false })).toThrow(/cells/);
  });

  it("honors a raised maxCells and a lowered maxCells", () => {
    const rows = `<row r="1"><c r="A1"><v>1</v></c><c r="C1"><v>2</v></c></row><row r="2"><c r="B2"><v>3</v></c></row>`;
    const file = zipSheet(sheet(rows));
    const ok = readXlsx(file, { header: false });
    expect(ok.columns).toEqual(["Column0", "Column1", "Column2"]);
    expect(ok.data).toEqual([
      { Column0: 1, Column1: null, Column2: 2 },
      { Column0: null, Column1: 3, Column2: null },
    ]);
    expect(() => readXlsx(file, { header: false, maxCells: 5 })).toThrow(DataValidationError);
    expect(() => readXlsx(file, { header: false, maxCells: 0 })).toThrow();
  });

  it("still reads a modest sheet whose only cell sits in the last column", () => {
    const file = zipSheet(sheet(`<row r="1"><c r="XFD1"><v>7</v></c></row>`));
    const out = readXlsx(file, { header: false });
    expect(out.columns.length).toBe(16384);
    expect(out.data[0]?.["Column16383"]).toBe(7);
  });
});

describe("readXlsx hostile input (DF-2: linear tag scanning)", () => {
  it("throws DataValidationError for unterminated row tags", () => {
    let rows = "";
    for (let r = 1; r <= 20000; r++) rows += `<row r="${r}">`;
    expect(() => readXlsx(zipSheet(sheet(rows)), { header: false })).toThrow(DataValidationError);
  });

  it("throws DataValidationError for unterminated cell tags", () => {
    let cells = "";
    for (let c = 0; c < 20000; c++) cells += `<c r="A1"><v>1</v>`;
    expect(() => readXlsx(zipSheet(sheet(`<row r="1">${cells}</row>`)), { header: false })).toThrow(
      DataValidationError
    );
  });

  it("throws DataValidationError for an unterminated value element", () => {
    const file = zipSheet(sheet(`<row r="1"><c r="A1"><v>1</c></row>`));
    expect(() => readXlsx(file, { header: false })).toThrow(DataValidationError);
  });

  it("does not choke on a very long attribute run", () => {
    const junk = "a".repeat(200000);
    const file = zipSheet(sheet(`<row r="1" ${junk}><c r="A1"><v>1</v></c></row>`));
    expect(readXlsx(file, { header: false }).data).toEqual([{ Column0: 1 }]);
  });
});
