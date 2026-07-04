/**
 * @see {@link https://deepbox.dev/docs/dataframe-overview | Deepbox documentation}
 */

export { Categorical } from "./Categorical";
export { DataFrame, DataFrameGroupBy } from "./DataFrame";
export {
  DateTimeAccessor,
  date_range,
  timedelta,
  to_datetime,
} from "./DateTimeAccessor";
export type {
  ParquetReadOptions,
  ParquetWriteOptions,
  XlsxReadOptions,
  XlsxWriteOptions,
} from "./io/index";
// IO: Excel and Parquet
export { readParquet, readXlsx, writeParquet, writeXlsx } from "./io/index";
export { MultiIndex } from "./MultiIndex";
export { PlotAccessor } from "./PlotAccessor";
export { Series } from "./Series";
export { StringAccessor } from "./StringAccessor";
export {
  type CellStyle,
  StyleAccessor,
  type StyleFunction,
} from "./StyleAccessor";
export type {
  AggregateFunction,
  DataFrameData,
  DataFrameOptions,
  DataValue,
  SeriesOptions,
} from "./types";
