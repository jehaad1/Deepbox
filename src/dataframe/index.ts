/**
 * @see {@link https://deepbox.dev/docs/dataframe-overview | Deepbox documentation}
 */

export { Categorical } from "./Categorical";
export { DataFrame, DataFrameGroupBy, EWM, Expanding, Rolling } from "./DataFrame";
export {
  type DateFreq,
  type DateRoundFreq,
  DateTimeAccessor,
  date_range,
  dateRange,
  timedelta,
  to_datetime,
  toDatetime,
} from "./DateTimeAccessor";
export type {
  ParquetReadOptions,
  ParquetReadResult,
  ParquetWriteOptions,
  XlsxCell,
  XlsxReadOptions,
  XlsxReadResult,
  XlsxWriteOptions,
} from "./io/index";
// IO: Excel and Parquet
export { readParquet, readXlsx, writeParquet, writeXlsx } from "./io/index";
export { MultiIndex } from "./MultiIndex";
export { PlotAccessor, type PlotBaseOptions } from "./PlotAccessor";
export { Series } from "./Series";
export { StringAccessor } from "./StringAccessor";
export {
  type CellStyle,
  StyleAccessor,
  type StyleFunction,
} from "./StyleAccessor";
export type {
  AggregateFunction,
  ConcatOptions,
  CorrelationMethod,
  CorrOptions,
  DataFrameData,
  DataFrameOptions,
  DataValue,
  FillMethod,
  FillnaMethodOptions,
  FillOptions,
  GroupByOptions,
  NamedAggregation,
  RollingOptions,
  SampleOptions,
  SeriesOptions,
  ValueCountsOptions,
} from "./types";
