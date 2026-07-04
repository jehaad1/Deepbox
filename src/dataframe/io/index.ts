/**
 * DataFrame IO format handlers.
 *
 * @module dataframe/io
 * @see {@link https://deepbox.dev/docs/dataframe-overview | Deepbox documentation}
 */

export type { ParquetReadOptions, ParquetWriteOptions } from "./parquet";
export { readParquet, writeParquet } from "./parquet";
export type { XlsxReadOptions, XlsxWriteOptions } from "./xlsx";
export { readXlsx, writeXlsx } from "./xlsx";
