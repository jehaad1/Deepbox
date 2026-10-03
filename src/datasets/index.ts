/**
 * @see {@link https://deepbox.dev/docs/datasets-builtin | Deepbox documentation}
 */

export type { CollateFn, DataLoaderOptions, StreamingDataLoaderOptions } from "./DataLoader";
export { DataLoader } from "./DataLoader";
export {
  makeBiclusters,
  makeBlobs,
  makeCheckerboard,
  makeCircles,
  makeClassification,
  makeFriedman1,
  makeFriedman2,
  makeFriedman3,
  makeGaussianQuantiles,
  makeLowRankMatrix,
  makeMoons,
  makeRegression,
  makeSCurve,
  makeSPDMatrix,
  makeSparseUncorrelated,
  makeSwissRoll,
} from "./generators";
export type { ImageDataset, ImageFetchOptions } from "./image";
export { fetchCIFAR10, fetchMNIST } from "./image";
// Kaggle integration
export type {
  KaggleCredentials,
  KaggleDatasetInfo,
  KaggleDownloadResult,
  KaggleFetchOptions,
} from "./kaggle";
export {
  fetchKaggleDataset,
  fetchKaggleDatasetInfo,
  listKaggleFiles,
  readKaggleCredentials,
  searchKaggleDatasets,
} from "./kaggle";
export type { Dataset, DatasetLoadOptions } from "./loaders";
export {
  loadBreastCancer,
  loadConcentricRings,
  loadCropYield,
  loadCustomerSegments,
  loadDiabetes,
  loadDigits,
  loadEnergyEfficiency,
  loadFitnessScores,
  loadFlowersExtended,
  loadFruitQuality,
  loadGaussianIslands,
  loadHousingMini,
  loadIris,
  loadLeafShapes,
  loadLinnerud,
  loadMoonsMulti,
  loadPerfectlySeparable,
  loadPlantGrowth,
  loadSeedMorphology,
  loadSensorStates,
  loadSpiralArms,
  loadStudentPerformance,
  loadTrafficConditions,
  loadWeatherOutcomes,
  loadWine,
} from "./loaders";
export type { FetchCSVDatasetOptions, RemoteDataset } from "./remote";
export { fetchCSVDataset, parseCSV } from "./remote";
export type { Sampler } from "./samplers";
export {
  SequentialSampler,
  SubsetRandomSampler,
  WeightedRandomSampler,
} from "./samplers";
export type {
  AsyncFactory,
  Batch,
  StreamCollateFn,
  StreamSample,
  SyncFactory,
} from "./streaming";
export {
  asyncIterableDataset,
  defaultCollate,
  iterableDataset,
  StreamingDataset,
} from "./streaming";
export type { TextDataset, TextFetchOptions } from "./text";
export { fetch20Newsgroups, fetchIMDB } from "./text";
export { filterDataset, mapDataset, randomSplit, Subset } from "./transforms";
