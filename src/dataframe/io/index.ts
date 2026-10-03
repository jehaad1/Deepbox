/**
 * DataFrame IO format handlers.
 *
 * @module dataframe/io
 * @see {@link https://deepbox.dev/docs/dataframe-overview | Deepbox documentation}
 */

export type { ParquetReadOptions, ParquetReadResult, ParquetWriteOptions } from "./parquet";
export { readParquet, writeParquet } from "./parquet";
export type { XlsxCell, XlsxReadOptions, XlsxReadResult, XlsxWriteOptions } from "./xlsx";
export { readXlsx, writeXlsx } from "./xlsx";
