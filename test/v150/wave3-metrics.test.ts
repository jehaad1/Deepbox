import { describe, expect, it } from "vitest";
import {
  DataValidationError,
  DTypeError,
  InvalidParameterError,
  ShapeError,
} from "../../src/core/errors";
import {
  type AveragedMetricOptions,
  accuracy,
  classificationReport,
  confusionMatrix,
  explainedVarianceScore,
  f1Score,
  fbetaScore,
  jaccardScore,
  logLoss,
  mae,
  mape,
  matthewsCorrcoef,
  meanAbsolutePercentageError,
  meanSquaredLogError,
  mse,
  precision,
  r2Score,
  recall,
  rmse,
  rocAucScore,
} from "../../src/metrics";
import { tensor } from "../../src/ndarray";
import { Tensor } from "../../src/ndarray/tensor";

// Expected values come from scikit-learn 1.8 (see the fixture generator notes in the
// commit): datasets are small seeded random samples, every case lists the options passed to
// both libraries.
type Dataset = {
  yTrue: Array<number | string>;
  yPred: Array<number | string>;
  proba?: number[][];
  score?: number[];
  prob?: number[];
  w: number[];
  classes?: Array<number | string>;
};
type Expected = number | string | Expected[];
type Case = {
  fn: string;
  ds: string;
  opts: Record<string, unknown>;
  expected: Expected;
  yPredOverride?: number[];
  yTrueOverride?: number[];
  score?: boolean;
  prob?: boolean;
  subset?: number[];
  weightSubset?: boolean;
};
const FIXTURES = JSON.parse(
  '{"datasets":{"mc3":{"yTrue":[2,1,0,2,1,1,1,2,0,1,1,2,0,2,2,0,0,1,0,0,1,2,0,0,0,2,0,2,0,2],"yPred":[2,1,0,2,1,1,1,2,0,1,1,2,0,2,2,1,0,1,0,0,1,0,0,1,0,2,0,1,0,2],"proba":[[0.21,0.17,0.62],[0.16,0.6,0.24],[0.53,0.26,0.21],[0.2,0.19,0.61],[0.18,0.59,0.23],[0.22,0.6,0.18],[0.21,0.62,0.17],[0.23,0.16,0.61],[0.54,0.3,0.16],[0.17,0.62,0.21],[0.17,0.62,0.21],[0.23,0.17,0.6],[0.59,0.19,0.22],[0.22,0.22,0.56],[0.29,0.14,0.57],[0.65,0.13,0.22],[0.64,0.15,0.21],[0.2,0.61,0.19],[0.66,0.17,0.17],[0.66,0.14,0.2],[0.15,0.61,0.24],[0.14,0.27,0.59],[0.59,0.19,0.22],[0.59,0.18,0.23],[0.61,0.22,0.17],[0.22,0.16,0.62],[0.57,0.23,0.2],[0.25,0.14,0.61],[0.59,0.2,0.21],[0.21,0.21,0.58]],"w":[2.0,1.19,0.58,2.27,1.37,0.73,2.61,1.73,1.49,1.2,1.38,2.13,2.19,1.97,1.71,2.7,2.09,1.15,1.29,2.95,2.1,0.56,2.85,0.85,2.33,1.76,2.16,0.54,1.13,0.72],"classes":[0,1,2]},"mc4s":{"yTrue":["ant","bee","bee","dog","bee","bee","dog","bee","dog","bee","dog","dog","cat","bee","ant","bee","bee","bee","cat","dog","ant","ant","bee","dog","cat","ant","ant","cat","bee","bee","bee","cat","dog","bee","cat","cat"],"yPred":["ant","bee","bee","cat","ant","bee","dog","bee","dog","ant","bee","dog","cat","bee","cat","cat","bee","bee","cat","dog","cat","ant","bee","dog","cat","cat","dog","cat","bee","bee","bee","cat","dog","bee","bee","cat"],"proba":[[0.5,0.23,0.17,0.1],[0.16,0.58,0.12,0.14],[0.15,0.55,0.1,0.2],[0.14,0.14,0.17,0.55],[0.13,0.55,0.11,0.21],[0.16,0.58,0.11,0.15],[0.17,0.11,0.16,0.56],[0.18,0.54,0.09,0.19],[0.15,0.12,0.14,0.59],[0.17,0.55,0.13,0.15],[0.17,0.12,0.13,0.58],[0.16,0.15,0.08,0.61],[0.16,0.13,0.51,0.2],[0.12,0.61,0.14,0.13],[0.57,0.15,0.15,0.13],[0.12,0.55,0.15,0.18],[0.18,0.52,0.13,0.17],[0.16,0.55,0.13,0.16],[0.17,0.16,0.55,0.12],[0.17,0.09,0.16,0.58],[0.58,0.11,0.12,0.19],[0.53,0.14,0.18,0.15],[0.13,0.54,0.17,0.16],[0.15,0.13,0.16,0.56],[0.19,0.16,0.57,0.08],[0.5,0.11,0.2,0.19],[0.53,0.22,0.13,0.12],[0.16,0.17,0.53,0.14],[0.14,0.54,0.22,0.1],[0.09,0.56,0.21,0.14],[0.09,0.58,0.19,0.14],[0.17,0.14,0.51,0.18],[0.16,0.12,0.14,0.58],[0.13,0.56,0.1,0.21],[0.14,0.16,0.53,0.17],[0.22,0.13,0.56,0.09]],"w":[3.0,2.86,2.37,0.59,1.37,0.91,2.87,0.98,2.09,2.6,1.43,0.69,1.23,1.92,2.23,2.39,1.17,1.98,2.04,1.73,1.66,1.33,1.25,1.96,1.67,2.13,1.0,2.95,1.43,1.19,0.99,1.17,0.56,2.38,1.65,2.54],"classes":["ant","bee","cat","dog"]},"mcSparse":{"yTrue":[9,2,9,2,5,2,2,2,5,5,9,2,5,5,2,5,5,5,2,5,5,2,9,5,5,5,2],"yPred":[9,2,9,2,2,2,2,2,5,5,9,2,9,5,9,5,2,5,2,5,5,2,9,5,5,2,2],"proba":[[0.27,0.18,0.55],[0.61,0.16,0.23],[0.19,0.2,0.61],[0.61,0.18,0.21],[0.27,0.51,0.22],[0.6,0.24,0.16],[0.59,0.19,0.22],[0.57,0.23,0.2],[0.24,0.59,0.17],[0.21,0.59,0.2],[0.2,0.17,0.63],[0.6,0.12,0.28],[0.22,0.59,0.19],[0.12,0.63,0.25],[0.62,0.14,0.24],[0.21,0.56,0.23],[0.18,0.61,0.21],[0.19,0.58,0.23],[0.58,0.15,0.27],[0.24,0.61,0.15],[0.25,0.61,0.14],[0.55,0.21,0.24],[0.19,0.17,0.64],[0.16,0.6,0.24],[0.22,0.54,0.24],[0.19,0.58,0.23],[0.57,0.22,0.21]],"w":[1.13,1.51,2.23,1.65,2.13,1.48,0.73,0.65,1.4,2.29,1.53,2.85,1.79,1.4,2.2,1.25,0.86,1.05,2.65,1.89,1.37,1.03,1.58,1.82,1.42,1.96,1.29],"classes":[2,5,9]},"bin":{"yTrue":[1,0,0,1,1,1,1,0,0,0,0,1,1,0,1,1,0,0,0,0,1,0,1,0],"yPred":[1,0,0,0,1,0,1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,0,0,0],"score":[0.6,0.3,0.9,0.8,0.6,0.5,0.6,0.3,0.4,-0.1,0.3,0.6,0.4,0.3,0.8,0.7,0.8,0.2,-0.2,0.8,0.8,0.6,0.6,0.3],"prob":[0.6,0.3,0.9,0.8,0.6,0.5,0.6,0.3,0.4,0.02,0.3,0.6,0.4,0.3,0.8,0.7,0.8,0.2,0.02,0.8,0.8,0.6,0.6,0.3],"w":[2.09,2.04,1.02,2.77,2.93,1.64,2.72,1.91,2.62,2.59,2.53,1.96,2.22,1.14,0.67,2.12,1.58,1.41,2.5,2.67,1.26,2.12,0.61,2.36]},"reg":{"yTrue":[5.915,6.86,3.78,3.612,11.427,5.821,5.808,5.523,5.043,0.652,6.637,5.91,5.136,6.366,1.999,10.43,5.276,6.003,4.019,7.763],"yPred":[7.187,6.982,2.941,5.502,11.477,-0.324,6.919,6.424,3.783,-0.69,6.026,4.502,3.651,2.621,4.024,10.852,4.233,3.932,2.542,6.23],"w":[1.21,0.79,0.54,1.48,0.68,2.5,2.31,1.87,2.97,2.38,1.6,0.54,1.36,2.5,0.6,1.17,1.79,2.16,0.86,0.65]},"regPos":{"yTrue":[0.56,3.295,8.208,7.271,8.857,1.251,6.505,8.022,4.204,3.462,0.013,3.325,6.971,6.178,1.46,3.315,9.287,3.783,8.205,9.399],"yPred":[2.585,0.156,9.648,8.66,13.272,6.907,6.165,9.68,0.33,0.617,1.6,3.665,4.864,4.61,0.886,2.641,9.608,7.098,5.423,8.954],"w":[1.21,0.68,1.44,1.59,1.38,0.63,1.22,1.14,2.99,2.47,1.66,1.99,2.13,1.29,1.28,1.9,0.9,2.09,0.85,1.16]}},"cases":[{"fn":"precision","ds":"bin","opts":{"average":"binary","zeroDivision":0},"expected":0.6666666666666666},{"fn":"precision","ds":"bin","opts":{"average":"binary","zeroDivision":0,"sampleWeight":"w"},"expected":0.6505091649694501},{"fn":"precision","ds":"bin","opts":{"average":"micro","zeroDivision":0},"expected":0.7083333333333334},{"fn":"precision","ds":"bin","opts":{"average":"micro","zeroDivision":0,"sampleWeight":"w"},"expected":0.7135636057287279},{"fn":"precision","ds":"bin","opts":{"average":"macro","zeroDivision":0},"expected":0.7083333333333333},{"fn":"precision","ds":"bin","opts":{"average":"macro","zeroDivision":0,"sampleWeight":"w"},"expected":0.7157909976613496},{"fn":"precision","ds":"bin","opts":{"average":"weighted","zeroDivision":0},"expected":0.7118055555555555},{"fn":"precision","ds":"bin","opts":{"average":"weighted","zeroDivision":0,"sampleWeight":"w"},"expected":0.7233531307659292},{"fn":"precision","ds":"bin","opts":{"average":null,"zeroDivision":0},"expected":[0.75,0.6666666666666666]},{"fn":"precision","ds":"bin","opts":{"average":null,"zeroDivision":0,"sampleWeight":"w"},"expected":[0.781072830353249,0.6505091649694501]},{"fn":"precision","ds":"bin","opts":{"average":"binary","zeroDivision":0},"expected":0.0,"yPredOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"fn":"precision","ds":"bin","opts":{"average":"binary","zeroDivision":0},"expected":0.0,"yPredOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"yTrueOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"fn":"precision","ds":"bin","opts":{"average":"binary","zeroDivision":1},"expected":1.0,"yPredOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"fn":"precision","ds":"bin","opts":{"average":"binary","zeroDivision":1},"expected":1.0,"yPredOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"yTrueOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"fn":"precision","ds":"bin","opts":{"average":"binary","zeroDivision":"NaN"},"expected":"NaN","yPredOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"fn":"precision","ds":"bin","opts":{"average":"binary","zeroDivision":"NaN"},"expected":"NaN","yPredOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"yTrueOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"fn":"precision","ds":"mc3","opts":{"average":"micro","zeroDivision":0},"expected":0.8666666666666667},{"fn":"precision","ds":"mc3","opts":{"average":"micro","zeroDivision":0,"sampleWeight":"w"},"expected":0.9064950733963403},{"fn":"precision","ds":"mc3","opts":{"average":"macro","zeroDivision":0},"expected":0.8787878787878788},{"fn":"precision","ds":"mc3","opts":{"average":"macro","zeroDivision":0,"sampleWeight":"w"},"expected":0.9043080647773349},{"fn":"precision","ds":"mc3","opts":{"average":"weighted","zeroDivision":0},"expected":0.8909090909090909},{"fn":"precision","ds":"mc3","opts":{"average":"weighted","zeroDivision":0,"sampleWeight":"w"},"expected":0.9260418366072538},{"fn":"precision","ds":"mc3","opts":{"average":null,"zeroDivision":0},"expected":[0.9090909090909091,0.7272727272727273,1.0]},{"fn":"precision","ds":"mc3","opts":{"average":null,"zeroDivision":0,"sampleWeight":"w"},"expected":[0.9714576962283384,0.7414664981036663,1.0]},{"fn":"precision","ds":"mc3","opts":{"average":"micro","labels":[2,0],"zeroDivision":0},"expected":0.9473684210526315},{"fn":"precision","ds":"mc3","opts":{"average":"macro","labels":[2,0],"zeroDivision":0},"expected":0.9545454545454546},{"fn":"precision","ds":"mc3","opts":{"average":"weighted","labels":[2,0],"zeroDivision":0},"expected":0.9504132231404957},{"fn":"precision","ds":"mc3","opts":{"average":null,"labels":[2,0],"zeroDivision":1,"sampleWeight":"w"},"expected":[1.0,0.9714576962283384]},{"fn":"precision","ds":"mc3","opts":{"average":"micro","labels":[2,0],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.9834856974343851},{"fn":"precision","ds":"mc3","opts":{"average":"macro","labels":[2,0],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.9857288481141693},{"fn":"precision","ds":"mc3","opts":{"average":"weighted","labels":[2,0],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.9830173292558614},{"fn":"precision","ds":"mc3","opts":{"average":null,"labels":[2,0],"zeroDivision":"NaN"},"expected":[1.0,0.9090909090909091]},{"fn":"precision","ds":"mc3","opts":{"average":null,"labels":[1,99],"zeroDivision":0,"sampleWeight":"w"},"expected":[0.7414664981036663,0.0]},{"fn":"precision","ds":"mc3","opts":{"average":"micro","labels":[1,99],"zeroDivision":1,"sampleWeight":"w"},"expected":0.7414664981036663},{"fn":"precision","ds":"mc3","opts":{"average":"macro","labels":[1,99],"zeroDivision":1,"sampleWeight":"w"},"expected":0.8707332490518331},{"fn":"precision","ds":"mc3","opts":{"average":"weighted","labels":[1,99],"zeroDivision":1,"sampleWeight":"w"},"expected":0.7414664981036663},{"fn":"precision","ds":"mc3","opts":{"average":null,"labels":[1,99],"zeroDivision":1},"expected":[0.7272727272727273,1.0]},{"fn":"precision","ds":"mc3","opts":{"average":"micro","labels":[1,99],"zeroDivision":"NaN"},"expected":0.7272727272727273},{"fn":"precision","ds":"mc3","opts":{"average":"macro","labels":[1,99],"zeroDivision":"NaN"},"expected":0.7272727272727273},{"fn":"precision","ds":"mc3","opts":{"average":"weighted","labels":[1,99],"zeroDivision":"NaN"},"expected":0.7272727272727273},{"fn":"precision","ds":"mc3","opts":{"average":"micro","labels":[99,2,0],"zeroDivision":0,"sampleWeight":"w"},"expected":0.9834856974343851},{"fn":"precision","ds":"mc3","opts":{"average":"macro","labels":[99,2,0],"zeroDivision":0,"sampleWeight":"w"},"expected":0.6571525654094462},{"fn":"precision","ds":"mc3","opts":{"average":"weighted","labels":[99,2,0],"zeroDivision":0,"sampleWeight":"w"},"expected":0.9830173292558614},{"fn":"precision","ds":"mc3","opts":{"average":null,"labels":[99,2,0],"zeroDivision":0},"expected":[0.0,1.0,0.9090909090909091]},{"fn":"precision","ds":"mc3","opts":{"average":"micro","labels":[99,2,0],"zeroDivision":1},"expected":0.9473684210526315},{"fn":"precision","ds":"mc3","opts":{"average":"macro","labels":[99,2,0],"zeroDivision":1},"expected":0.9696969696969697},{"fn":"precision","ds":"mc3","opts":{"average":"weighted","labels":[99,2,0],"zeroDivision":1},"expected":0.9504132231404957},{"fn":"precision","ds":"mc3","opts":{"average":"micro","labels":[99,2,0],"zeroDivision":"NaN"},"expected":0.9473684210526315},{"fn":"precision","ds":"mc3","opts":{"average":"micro","labels":[99,2,0],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.9834856974343851},{"fn":"precision","ds":"mc3","opts":{"average":"macro","labels":[99,2,0],"zeroDivision":"NaN"},"expected":0.9545454545454546},{"fn":"precision","ds":"mc3","opts":{"average":"macro","labels":[99,2,0],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.9857288481141693},{"fn":"precision","ds":"mc3","opts":{"average":"weighted","labels":[99,2,0],"zeroDivision":"NaN"},"expected":0.9504132231404957},{"fn":"precision","ds":"mc3","opts":{"average":"weighted","labels":[99,2,0],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.9830173292558614},{"fn":"precision","ds":"mc3","opts":{"average":null,"labels":[99,2,0],"zeroDivision":"NaN"},"expected":["NaN",1.0,0.9090909090909091]},{"fn":"precision","ds":"mc3","opts":{"average":null,"labels":[99,2,0],"zeroDivision":"NaN","sampleWeight":"w"},"expected":["NaN",1.0,0.9714576962283384]},{"fn":"precision","ds":"mc4s","opts":{"average":"micro","zeroDivision":0},"expected":0.7222222222222222},{"fn":"precision","ds":"mc4s","opts":{"average":"micro","zeroDivision":0,"sampleWeight":"w"},"expected":0.7263681592039801},{"fn":"precision","ds":"mc4s","opts":{"average":"macro","zeroDivision":0},"expected":0.689935064935065},{"fn":"precision","ds":"mc4s","opts":{"average":"macro","zeroDivision":0,"sampleWeight":"w"},"expected":0.7140555868569793},{"fn":"precision","ds":"mc4s","opts":{"average":"weighted","zeroDivision":0},"expected":0.737012987012987},{"fn":"precision","ds":"mc4s","opts":{"average":"weighted","zeroDivision":0,"sampleWeight":"w"},"expected":0.7457861587960067},{"fn":"precision","ds":"mc4s","opts":{"average":null,"zeroDivision":0},"expected":[0.5,0.8571428571428571,0.5454545454545454,0.8571428571428571]},{"fn":"precision","ds":"mc4s","opts":{"average":null,"zeroDivision":0,"sampleWeight":"w"},"expected":[0.5216867469879518,0.8631719235895159,0.5631067961165047,0.908256880733945]},{"fn":"precision","ds":"mc4s","opts":{"average":"micro","labels":["cat","ant"],"zeroDivision":0},"expected":0.5333333333333333},{"fn":"precision","ds":"mc4s","opts":{"average":"macro","labels":["cat","ant"],"zeroDivision":0},"expected":0.5227272727272727},{"fn":"precision","ds":"mc4s","opts":{"average":"weighted","labels":["cat","ant"],"zeroDivision":0},"expected":0.5244755244755245},{"fn":"precision","ds":"mc4s","opts":{"average":null,"labels":["cat","ant"],"zeroDivision":1,"sampleWeight":"w"},"expected":[0.5631067961165047,0.5216867469879518]},{"fn":"precision","ds":"mc4s","opts":{"average":"micro","labels":["cat","ant"],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.5512110726643598},{"fn":"precision","ds":"mc4s","opts":{"average":"macro","labels":["cat","ant"],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.5423967715522282},{"fn":"precision","ds":"mc4s","opts":{"average":"weighted","labels":["cat","ant"],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.5439963262949976},{"fn":"precision","ds":"mc4s","opts":{"average":null,"labels":["cat","ant"],"zeroDivision":"NaN"},"expected":[0.5454545454545454,0.5]},{"fn":"precision","ds":"mc4s","opts":{"average":null,"labels":["bee","zzz"],"zeroDivision":0,"sampleWeight":"w"},"expected":[0.8631719235895159,0.0]},{"fn":"precision","ds":"mc4s","opts":{"average":"micro","labels":["bee","zzz"],"zeroDivision":1,"sampleWeight":"w"},"expected":0.8631719235895159},{"fn":"precision","ds":"mc4s","opts":{"average":"macro","labels":["bee","zzz"],"zeroDivision":1,"sampleWeight":"w"},"expected":0.9315859617947579},{"fn":"precision","ds":"mc4s","opts":{"average":"weighted","labels":["bee","zzz"],"zeroDivision":1,"sampleWeight":"w"},"expected":0.8631719235895159},{"fn":"precision","ds":"mc4s","opts":{"average":null,"labels":["bee","zzz"],"zeroDivision":1},"expected":[0.8571428571428571,1.0]},{"fn":"precision","ds":"mc4s","opts":{"average":"micro","labels":["bee","zzz"],"zeroDivision":"NaN"},"expected":0.8571428571428571},{"fn":"precision","ds":"mc4s","opts":{"average":"macro","labels":["bee","zzz"],"zeroDivision":"NaN"},"expected":0.8571428571428571},{"fn":"precision","ds":"mc4s","opts":{"average":"weighted","labels":["bee","zzz"],"zeroDivision":"NaN"},"expected":0.8571428571428571},{"fn":"precision","ds":"mc4s","opts":{"average":"micro","labels":["zzz","cat","ant"],"zeroDivision":0,"sampleWeight":"w"},"expected":0.5512110726643598},{"fn":"precision","ds":"mc4s","opts":{"average":"macro","labels":["zzz","cat","ant"],"zeroDivision":0,"sampleWeight":"w"},"expected":0.36159784770148545},{"fn":"precision","ds":"mc4s","opts":{"average":"weighted","labels":["zzz","cat","ant"],"zeroDivision":0,"sampleWeight":"w"},"expected":0.5439963262949976},{"fn":"precision","ds":"mc4s","opts":{"average":null,"labels":["zzz","cat","ant"],"zeroDivision":0},"expected":[0.0,0.5454545454545454,0.5]},{"fn":"precision","ds":"mc4s","opts":{"average":"micro","labels":["zzz","cat","ant"],"zeroDivision":1},"expected":0.5333333333333333},{"fn":"precision","ds":"mc4s","opts":{"average":"macro","labels":["zzz","cat","ant"],"zeroDivision":1},"expected":0.6818181818181818},{"fn":"precision","ds":"mc4s","opts":{"average":"weighted","labels":["zzz","cat","ant"],"zeroDivision":1},"expected":0.5244755244755245},{"fn":"precision","ds":"mc4s","opts":{"average":"micro","labels":["zzz","cat","ant"],"zeroDivision":"NaN"},"expected":0.5333333333333333},{"fn":"precision","ds":"mc4s","opts":{"average":"micro","labels":["zzz","cat","ant"],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.5512110726643598},{"fn":"precision","ds":"mc4s","opts":{"average":"macro","labels":["zzz","cat","ant"],"zeroDivision":"NaN"},"expected":0.5227272727272727},{"fn":"precision","ds":"mc4s","opts":{"average":"macro","labels":["zzz","cat","ant"],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.5423967715522282},{"fn":"precision","ds":"mc4s","opts":{"average":"weighted","labels":["zzz","cat","ant"],"zeroDivision":"NaN"},"expected":0.5244755244755245},{"fn":"precision","ds":"mc4s","opts":{"average":"weighted","labels":["zzz","cat","ant"],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.5439963262949976},{"fn":"precision","ds":"mc4s","opts":{"average":null,"labels":["zzz","cat","ant"],"zeroDivision":"NaN"},"expected":["NaN",0.5454545454545454,0.5]},{"fn":"precision","ds":"mc4s","opts":{"average":null,"labels":["zzz","cat","ant"],"zeroDivision":"NaN","sampleWeight":"w"},"expected":["NaN",0.5631067961165047,0.5216867469879518]},{"fn":"precision","ds":"mcSparse","opts":{"average":"micro","zeroDivision":0},"expected":0.8148148148148148},{"fn":"precision","ds":"mcSparse","opts":{"average":"micro","zeroDivision":0,"sampleWeight":"w"},"expected":0.7927677329624477},{"fn":"precision","ds":"mcSparse","opts":{"average":"macro","zeroDivision":0},"expected":0.8055555555555555},{"fn":"precision","ds":"mcSparse","opts":{"average":"macro","zeroDivision":0,"sampleWeight":"w"},"expected":0.7850362820628929},{"fn":"precision","ds":"mcSparse","opts":{"average":"weighted","zeroDivision":0},"expected":0.8580246913580247},{"fn":"precision","ds":"mcSparse","opts":{"average":"weighted","zeroDivision":0,"sampleWeight":"w"},"expected":0.84484127457179},{"fn":"precision","ds":"mcSparse","opts":{"average":null,"zeroDivision":0},"expected":[0.75,1.0,0.6666666666666666]},{"fn":"precision","ds":"mcSparse","opts":{"average":null,"zeroDivision":0,"sampleWeight":"w"},"expected":[0.736562001064396,1.0,0.618546845124283]},{"fn":"precision","ds":"mcSparse","opts":{"average":"micro","labels":[9,2],"zeroDivision":0},"expected":0.7222222222222222},{"fn":"precision","ds":"mcSparse","opts":{"average":"macro","labels":[9,2],"zeroDivision":0},"expected":0.7083333333333333},{"fn":"precision","ds":"mcSparse","opts":{"average":"weighted","labels":[9,2],"zeroDivision":0},"expected":0.7261904761904762},{"fn":"precision","ds":"mcSparse","opts":{"average":null,"labels":[9,2],"zeroDivision":1,"sampleWeight":"w"},"expected":[0.618546845124283,0.736562001064396]},{"fn":"precision","ds":"mcSparse","opts":{"average":"micro","labels":[9,2],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.6943589743589743},{"fn":"precision","ds":"mcSparse","opts":{"average":"macro","labels":[9,2],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.6775544230943396},{"fn":"precision","ds":"mcSparse","opts":{"average":"weighted","labels":[9,2],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.7026411632619735},{"fn":"precision","ds":"mcSparse","opts":{"average":null,"labels":[9,2],"zeroDivision":"NaN"},"expected":[0.6666666666666666,0.75]},{"fn":"precision","ds":"mcSparse","opts":{"average":null,"labels":[5,99],"zeroDivision":0,"sampleWeight":"w"},"expected":[1.0,0.0]},{"fn":"precision","ds":"mcSparse","opts":{"average":"micro","labels":[5,99],"zeroDivision":1,"sampleWeight":"w"},"expected":1.0},{"fn":"precision","ds":"mcSparse","opts":{"average":"macro","labels":[5,99],"zeroDivision":1,"sampleWeight":"w"},"expected":1.0},{"fn":"precision","ds":"mcSparse","opts":{"average":"weighted","labels":[5,99],"zeroDivision":1,"sampleWeight":"w"},"expected":1.0},{"fn":"precision","ds":"mcSparse","opts":{"average":null,"labels":[5,99],"zeroDivision":1},"expected":[1.0,1.0]},{"fn":"precision","ds":"mcSparse","opts":{"average":"micro","labels":[5,99],"zeroDivision":"NaN"},"expected":1.0},{"fn":"precision","ds":"mcSparse","opts":{"average":"macro","labels":[5,99],"zeroDivision":"NaN"},"expected":1.0},{"fn":"precision","ds":"mcSparse","opts":{"average":"weighted","labels":[5,99],"zeroDivision":"NaN"},"expected":1.0},{"fn":"precision","ds":"mcSparse","opts":{"average":"micro","labels":[99,9,2],"zeroDivision":0,"sampleWeight":"w"},"expected":0.6943589743589743},{"fn":"precision","ds":"mcSparse","opts":{"average":"macro","labels":[99,9,2],"zeroDivision":0,"sampleWeight":"w"},"expected":0.4517029487295597},{"fn":"precision","ds":"mcSparse","opts":{"average":"weighted","labels":[99,9,2],"zeroDivision":0,"sampleWeight":"w"},"expected":0.7026411632619735},{"fn":"precision","ds":"mcSparse","opts":{"average":null,"labels":[99,9,2],"zeroDivision":0},"expected":[0.0,0.6666666666666666,0.75]},{"fn":"precision","ds":"mcSparse","opts":{"average":"micro","labels":[99,9,2],"zeroDivision":1},"expected":0.7222222222222222},{"fn":"precision","ds":"mcSparse","opts":{"average":"macro","labels":[99,9,2],"zeroDivision":1},"expected":0.8055555555555555},{"fn":"precision","ds":"mcSparse","opts":{"average":"weighted","labels":[99,9,2],"zeroDivision":1},"expected":0.7261904761904762},{"fn":"precision","ds":"mcSparse","opts":{"average":"micro","labels":[99,9,2],"zeroDivision":"NaN"},"expected":0.7222222222222222},{"fn":"precision","ds":"mcSparse","opts":{"average":"micro","labels":[99,9,2],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.6943589743589743},{"fn":"precision","ds":"mcSparse","opts":{"average":"macro","labels":[99,9,2],"zeroDivision":"NaN"},"expected":0.7083333333333333},{"fn":"precision","ds":"mcSparse","opts":{"average":"macro","labels":[99,9,2],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.6775544230943396},{"fn":"precision","ds":"mcSparse","opts":{"average":"weighted","labels":[99,9,2],"zeroDivision":"NaN"},"expected":0.7261904761904762},{"fn":"precision","ds":"mcSparse","opts":{"average":"weighted","labels":[99,9,2],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.7026411632619735},{"fn":"precision","ds":"mcSparse","opts":{"average":null,"labels":[99,9,2],"zeroDivision":"NaN"},"expected":["NaN",0.6666666666666666,0.75]},{"fn":"precision","ds":"mcSparse","opts":{"average":null,"labels":[99,9,2],"zeroDivision":"NaN","sampleWeight":"w"},"expected":["NaN",0.618546845124283,0.736562001064396]},{"fn":"recall","ds":"bin","opts":{"average":"binary","zeroDivision":0},"expected":0.7272727272727273},{"fn":"recall","ds":"bin","opts":{"average":"binary","zeroDivision":0,"sampleWeight":"w"},"expected":0.7608384945212006},{"fn":"recall","ds":"bin","opts":{"average":"micro","zeroDivision":0},"expected":0.7083333333333334},{"fn":"recall","ds":"bin","opts":{"average":"micro","zeroDivision":0,"sampleWeight":"w"},"expected":0.7135636057287279},{"fn":"recall","ds":"bin","opts":{"average":"macro","zeroDivision":0},"expected":0.7097902097902098},{"fn":"recall","ds":"bin","opts":{"average":"macro","zeroDivision":0,"sampleWeight":"w"},"expected":0.7184713423908382},{"fn":"recall","ds":"bin","opts":{"average":"weighted","zeroDivision":0},"expected":0.7083333333333334},{"fn":"recall","ds":"bin","opts":{"average":"weighted","zeroDivision":0,"sampleWeight":"w"},"expected":0.7135636057287279},{"fn":"recall","ds":"bin","opts":{"average":null,"zeroDivision":0},"expected":[0.6923076923076923,0.7272727272727273]},{"fn":"recall","ds":"bin","opts":{"average":null,"zeroDivision":0,"sampleWeight":"w"},"expected":[0.6761041902604757,0.7608384945212006]},{"fn":"recall","ds":"bin","opts":{"average":"binary","zeroDivision":0},"expected":0.0,"yPredOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"fn":"recall","ds":"bin","opts":{"average":"binary","zeroDivision":0},"expected":0.0,"yPredOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"yTrueOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"fn":"recall","ds":"bin","opts":{"average":"binary","zeroDivision":1},"expected":0.0,"yPredOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"fn":"recall","ds":"bin","opts":{"average":"binary","zeroDivision":1},"expected":1.0,"yPredOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"yTrueOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"fn":"recall","ds":"bin","opts":{"average":"binary","zeroDivision":"NaN"},"expected":0.0,"yPredOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"fn":"recall","ds":"bin","opts":{"average":"binary","zeroDivision":"NaN"},"expected":"NaN","yPredOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"yTrueOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"fn":"recall","ds":"mc3","opts":{"average":"micro","zeroDivision":0},"expected":0.8666666666666667},{"fn":"recall","ds":"mc3","opts":{"average":"micro","zeroDivision":0,"sampleWeight":"w"},"expected":0.9064950733963402},{"fn":"recall","ds":"mc3","opts":{"average":"macro","zeroDivision":0},"expected":0.8777777777777779},{"fn":"recall","ds":"mc3","opts":{"average":"macro","zeroDivision":0,"sampleWeight":"w"},"expected":0.923838281251422},{"fn":"recall","ds":"mc3","opts":{"average":"weighted","zeroDivision":0},"expected":0.8666666666666667},{"fn":"recall","ds":"mc3","opts":{"average":"weighted","zeroDivision":0,"sampleWeight":"w"},"expected":0.9064950733963402},{"fn":"recall","ds":"mc3","opts":{"average":null,"zeroDivision":0},"expected":[0.8333333333333334,1.0,0.8]},{"fn":"recall","ds":"mc3","opts":{"average":null,"zeroDivision":0,"sampleWeight":"w"},"expected":[0.8429898275099513,1.0,0.9285250162443145]},{"fn":"recall","ds":"mc3","opts":{"average":"micro","labels":[2,0],"zeroDivision":0},"expected":0.8181818181818182},{"fn":"recall","ds":"mc3","opts":{"average":"macro","labels":[2,0],"zeroDivision":0},"expected":0.8166666666666667},{"fn":"recall","ds":"mc3","opts":{"average":"weighted","labels":[2,0],"zeroDivision":0},"expected":0.8181818181818182},{"fn":"recall","ds":"mc3","opts":{"average":null,"labels":[2,0],"zeroDivision":1,"sampleWeight":"w"},"expected":[0.9285250162443145,0.8429898275099513]},{"fn":"recall","ds":"mc3","opts":{"average":"micro","labels":[2,0],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.8776315789473683},{"fn":"recall","ds":"mc3","opts":{"average":"macro","labels":[2,0],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.8857574218771329},{"fn":"recall","ds":"mc3","opts":{"average":"weighted","labels":[2,0],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.8776315789473683},{"fn":"recall","ds":"mc3","opts":{"average":null,"labels":[2,0],"zeroDivision":"NaN"},"expected":[0.8,0.8333333333333334]},{"fn":"recall","ds":"mc3","opts":{"average":null,"labels":[1,99],"zeroDivision":0,"sampleWeight":"w"},"expected":[1.0,0.0]},{"fn":"recall","ds":"mc3","opts":{"average":"micro","labels":[1,99],"zeroDivision":1,"sampleWeight":"w"},"expected":1.0},{"fn":"recall","ds":"mc3","opts":{"average":"macro","labels":[1,99],"zeroDivision":1,"sampleWeight":"w"},"expected":1.0},{"fn":"recall","ds":"mc3","opts":{"average":"weighted","labels":[1,99],"zeroDivision":1,"sampleWeight":"w"},"expected":1.0},{"fn":"recall","ds":"mc3","opts":{"average":null,"labels":[1,99],"zeroDivision":1},"expected":[1.0,1.0]},{"fn":"recall","ds":"mc3","opts":{"average":"micro","labels":[1,99],"zeroDivision":"NaN"},"expected":1.0},{"fn":"recall","ds":"mc3","opts":{"average":"macro","labels":[1,99],"zeroDivision":"NaN"},"expected":1.0},{"fn":"recall","ds":"mc3","opts":{"average":"weighted","labels":[1,99],"zeroDivision":"NaN"},"expected":1.0},{"fn":"recall","ds":"mc3","opts":{"average":"micro","labels":[99,2,0],"zeroDivision":0,"sampleWeight":"w"},"expected":0.8776315789473683},{"fn":"recall","ds":"mc3","opts":{"average":"macro","labels":[99,2,0],"zeroDivision":0,"sampleWeight":"w"},"expected":0.5905049479180886},{"fn":"recall","ds":"mc3","opts":{"average":"weighted","labels":[99,2,0],"zeroDivision":0,"sampleWeight":"w"},"expected":0.8776315789473683},{"fn":"recall","ds":"mc3","opts":{"average":null,"labels":[99,2,0],"zeroDivision":0},"expected":[0.0,0.8,0.8333333333333334]},{"fn":"recall","ds":"mc3","opts":{"average":"micro","labels":[99,2,0],"zeroDivision":1},"expected":0.8181818181818182},{"fn":"recall","ds":"mc3","opts":{"average":"macro","labels":[99,2,0],"zeroDivision":1},"expected":0.8777777777777778},{"fn":"recall","ds":"mc3","opts":{"average":"weighted","labels":[99,2,0],"zeroDivision":1},"expected":0.8181818181818182},{"fn":"recall","ds":"mc3","opts":{"average":"micro","labels":[99,2,0],"zeroDivision":"NaN"},"expected":0.8181818181818182},{"fn":"recall","ds":"mc3","opts":{"average":"micro","labels":[99,2,0],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.8776315789473683},{"fn":"recall","ds":"mc3","opts":{"average":"macro","labels":[99,2,0],"zeroDivision":"NaN"},"expected":0.8166666666666667},{"fn":"recall","ds":"mc3","opts":{"average":"macro","labels":[99,2,0],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.8857574218771329},{"fn":"recall","ds":"mc3","opts":{"average":"weighted","labels":[99,2,0],"zeroDivision":"NaN"},"expected":0.8181818181818182},{"fn":"recall","ds":"mc3","opts":{"average":"weighted","labels":[99,2,0],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.8776315789473683},{"fn":"recall","ds":"mc3","opts":{"average":null,"labels":[99,2,0],"zeroDivision":"NaN"},"expected":["NaN",0.8,0.8333333333333334]},{"fn":"recall","ds":"mc3","opts":{"average":null,"labels":[99,2,0],"zeroDivision":"NaN","sampleWeight":"w"},"expected":["NaN",0.9285250162443145,0.8429898275099513]},{"fn":"recall","ds":"mc4s","opts":{"average":"micro","zeroDivision":0},"expected":0.7222222222222222},{"fn":"recall","ds":"mc4s","opts":{"average":"micro","zeroDivision":0,"sampleWeight":"w"},"expected":0.72636815920398},{"fn":"recall","ds":"mc4s","opts":{"average":"macro","zeroDivision":0},"expected":0.6851190476190476},{"fn":"recall","ds":"mc4s","opts":{"average":"macro","zeroDivision":0,"sampleWeight":"w"},"expected":0.7102247990310002},{"fn":"recall","ds":"mc4s","opts":{"average":"weighted","zeroDivision":0},"expected":0.7222222222222222},{"fn":"recall","ds":"mc4s","opts":{"average":"weighted","zeroDivision":0,"sampleWeight":"w"},"expected":0.72636815920398},{"fn":"recall","ds":"mc4s","opts":{"average":null,"zeroDivision":0},"expected":[0.3333333333333333,0.8,0.8571428571428571,0.75]},{"fn":"recall","ds":"mc4s","opts":{"average":null,"zeroDivision":0,"sampleWeight":"w"},"expected":[0.38149779735682815,0.7533927879022877,0.8754716981132074,0.8305369127516778]},{"fn":"recall","ds":"mc4s","opts":{"average":"micro","labels":["cat","ant"],"zeroDivision":0},"expected":0.6153846153846154},{"fn":"recall","ds":"mc4s","opts":{"average":"macro","labels":["cat","ant"],"zeroDivision":0},"expected":0.5952380952380952},{"fn":"recall","ds":"mc4s","opts":{"average":"weighted","labels":["cat","ant"],"zeroDivision":0},"expected":0.6153846153846154},{"fn":"recall","ds":"mc4s","opts":{"average":null,"labels":["cat","ant"],"zeroDivision":1,"sampleWeight":"w"},"expected":[0.8754716981132074,0.38149779735682815]},{"fn":"recall","ds":"mc4s","opts":{"average":"micro","labels":["cat","ant"],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.647560975609756},{"fn":"recall","ds":"mc4s","opts":{"average":"macro","labels":["cat","ant"],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.6284847477350177},{"fn":"recall","ds":"mc4s","opts":{"average":"weighted","labels":["cat","ant"],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.647560975609756},{"fn":"recall","ds":"mc4s","opts":{"average":null,"labels":["cat","ant"],"zeroDivision":"NaN"},"expected":[0.8571428571428571,0.3333333333333333]},{"fn":"recall","ds":"mc4s","opts":{"average":null,"labels":["bee","zzz"],"zeroDivision":0,"sampleWeight":"w"},"expected":[0.7533927879022877,0.0]},{"fn":"recall","ds":"mc4s","opts":{"average":"micro","labels":["bee","zzz"],"zeroDivision":1,"sampleWeight":"w"},"expected":0.7533927879022877},{"fn":"recall","ds":"mc4s","opts":{"average":"macro","labels":["bee","zzz"],"zeroDivision":1,"sampleWeight":"w"},"expected":0.8766963939511438},{"fn":"recall","ds":"mc4s","opts":{"average":"weighted","labels":["bee","zzz"],"zeroDivision":1,"sampleWeight":"w"},"expected":0.7533927879022877},{"fn":"recall","ds":"mc4s","opts":{"average":null,"labels":["bee","zzz"],"zeroDivision":1},"expected":[0.8,1.0]},{"fn":"recall","ds":"mc4s","opts":{"average":"micro","labels":["bee","zzz"],"zeroDivision":"NaN"},"expected":0.8},{"fn":"recall","ds":"mc4s","opts":{"average":"macro","labels":["bee","zzz"],"zeroDivision":"NaN"},"expected":0.8},{"fn":"recall","ds":"mc4s","opts":{"average":"weighted","labels":["bee","zzz"],"zeroDivision":"NaN"},"expected":0.8},{"fn":"recall","ds":"mc4s","opts":{"average":"micro","labels":["zzz","cat","ant"],"zeroDivision":0,"sampleWeight":"w"},"expected":0.647560975609756},{"fn":"recall","ds":"mc4s","opts":{"average":"macro","labels":["zzz","cat","ant"],"zeroDivision":0,"sampleWeight":"w"},"expected":0.41898983182334515},{"fn":"recall","ds":"mc4s","opts":{"average":"weighted","labels":["zzz","cat","ant"],"zeroDivision":0,"sampleWeight":"w"},"expected":0.647560975609756},{"fn":"recall","ds":"mc4s","opts":{"average":null,"labels":["zzz","cat","ant"],"zeroDivision":0},"expected":[0.0,0.8571428571428571,0.3333333333333333]},{"fn":"recall","ds":"mc4s","opts":{"average":"micro","labels":["zzz","cat","ant"],"zeroDivision":1},"expected":0.6153846153846154},{"fn":"recall","ds":"mc4s","opts":{"average":"macro","labels":["zzz","cat","ant"],"zeroDivision":1},"expected":0.7301587301587302},{"fn":"recall","ds":"mc4s","opts":{"average":"weighted","labels":["zzz","cat","ant"],"zeroDivision":1},"expected":0.6153846153846154},{"fn":"recall","ds":"mc4s","opts":{"average":"micro","labels":["zzz","cat","ant"],"zeroDivision":"NaN"},"expected":0.6153846153846154},{"fn":"recall","ds":"mc4s","opts":{"average":"micro","labels":["zzz","cat","ant"],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.647560975609756},{"fn":"recall","ds":"mc4s","opts":{"average":"macro","labels":["zzz","cat","ant"],"zeroDivision":"NaN"},"expected":0.5952380952380952},{"fn":"recall","ds":"mc4s","opts":{"average":"macro","labels":["zzz","cat","ant"],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.6284847477350177},{"fn":"recall","ds":"mc4s","opts":{"average":"weighted","labels":["zzz","cat","ant"],"zeroDivision":"NaN"},"expected":0.6153846153846154},{"fn":"recall","ds":"mc4s","opts":{"average":"weighted","labels":["zzz","cat","ant"],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.647560975609756},{"fn":"recall","ds":"mc4s","opts":{"average":null,"labels":["zzz","cat","ant"],"zeroDivision":"NaN"},"expected":["NaN",0.8571428571428571,0.3333333333333333]},{"fn":"recall","ds":"mc4s","opts":{"average":null,"labels":["zzz","cat","ant"],"zeroDivision":"NaN","sampleWeight":"w"},"expected":["NaN",0.8754716981132074,0.38149779735682815]},{"fn":"recall","ds":"mcSparse","opts":{"average":"micro","zeroDivision":0},"expected":0.8148148148148148},{"fn":"recall","ds":"mcSparse","opts":{"average":"micro","zeroDivision":0,"sampleWeight":"w"},"expected":0.7927677329624477},{"fn":"recall","ds":"mcSparse","opts":{"average":"macro","zeroDivision":0},"expected":0.8641025641025641},{"fn":"recall","ds":"mcSparse","opts":{"average":"macro","zeroDivision":0,"sampleWeight":"w"},"expected":0.8453780720278798},{"fn":"recall","ds":"mcSparse","opts":{"average":"weighted","zeroDivision":0},"expected":0.8148148148148148},{"fn":"recall","ds":"mcSparse","opts":{"average":"weighted","zeroDivision":0,"sampleWeight":"w"},"expected":0.7927677329624477},{"fn":"recall","ds":"mcSparse","opts":{"average":null,"zeroDivision":0},"expected":[0.9,0.6923076923076923,1.0]},{"fn":"recall","ds":"mcSparse","opts":{"average":null,"zeroDivision":0,"sampleWeight":"w"},"expected":[0.8628428927680798,0.6732913233155597,1.0]},{"fn":"recall","ds":"mcSparse","opts":{"average":"micro","labels":[9,2],"zeroDivision":0},"expected":0.9285714285714286},{"fn":"recall","ds":"mcSparse","opts":{"average":"macro","labels":[9,2],"zeroDivision":0},"expected":0.95},{"fn":"recall","ds":"mcSparse","opts":{"average":"weighted","labels":[9,2],"zeroDivision":0},"expected":0.9285714285714286},{"fn":"recall","ds":"mcSparse","opts":{"average":null,"labels":[9,2],"zeroDivision":1,"sampleWeight":"w"},"expected":[1.0,0.8628428927680798]},{"fn":"recall","ds":"mcSparse","opts":{"average":"micro","labels":[9,2],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.902265659706797},{"fn":"recall","ds":"mcSparse","opts":{"average":"macro","labels":[9,2],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.9314214463840399},{"fn":"recall","ds":"mcSparse","opts":{"average":"weighted","labels":[9,2],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.902265659706797},{"fn":"recall","ds":"mcSparse","opts":{"average":null,"labels":[9,2],"zeroDivision":"NaN"},"expected":[1.0,0.9]},{"fn":"recall","ds":"mcSparse","opts":{"average":null,"labels":[5,99],"zeroDivision":0,"sampleWeight":"w"},"expected":[0.6732913233155597,0.0]},{"fn":"recall","ds":"mcSparse","opts":{"average":"micro","labels":[5,99],"zeroDivision":1,"sampleWeight":"w"},"expected":0.6732913233155597},{"fn":"recall","ds":"mcSparse","opts":{"average":"macro","labels":[5,99],"zeroDivision":1,"sampleWeight":"w"},"expected":0.8366456616577799},{"fn":"recall","ds":"mcSparse","opts":{"average":"weighted","labels":[5,99],"zeroDivision":1,"sampleWeight":"w"},"expected":0.6732913233155597},{"fn":"recall","ds":"mcSparse","opts":{"average":null,"labels":[5,99],"zeroDivision":1},"expected":[0.6923076923076923,1.0]},{"fn":"recall","ds":"mcSparse","opts":{"average":"micro","labels":[5,99],"zeroDivision":"NaN"},"expected":0.6923076923076923},{"fn":"recall","ds":"mcSparse","opts":{"average":"macro","labels":[5,99],"zeroDivision":"NaN"},"expected":0.6923076923076923},{"fn":"recall","ds":"mcSparse","opts":{"average":"weighted","labels":[5,99],"zeroDivision":"NaN"},"expected":0.6923076923076923},{"fn":"recall","ds":"mcSparse","opts":{"average":"micro","labels":[99,9,2],"zeroDivision":0,"sampleWeight":"w"},"expected":0.902265659706797},{"fn":"recall","ds":"mcSparse","opts":{"average":"macro","labels":[99,9,2],"zeroDivision":0,"sampleWeight":"w"},"expected":0.6209476309226932},{"fn":"recall","ds":"mcSparse","opts":{"average":"weighted","labels":[99,9,2],"zeroDivision":0,"sampleWeight":"w"},"expected":0.902265659706797},{"fn":"recall","ds":"mcSparse","opts":{"average":null,"labels":[99,9,2],"zeroDivision":0},"expected":[0.0,1.0,0.9]},{"fn":"recall","ds":"mcSparse","opts":{"average":"micro","labels":[99,9,2],"zeroDivision":1},"expected":0.9285714285714286},{"fn":"recall","ds":"mcSparse","opts":{"average":"macro","labels":[99,9,2],"zeroDivision":1},"expected":0.9666666666666667},{"fn":"recall","ds":"mcSparse","opts":{"average":"weighted","labels":[99,9,2],"zeroDivision":1},"expected":0.9285714285714286},{"fn":"recall","ds":"mcSparse","opts":{"average":"micro","labels":[99,9,2],"zeroDivision":"NaN"},"expected":0.9285714285714286},{"fn":"recall","ds":"mcSparse","opts":{"average":"micro","labels":[99,9,2],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.902265659706797},{"fn":"recall","ds":"mcSparse","opts":{"average":"macro","labels":[99,9,2],"zeroDivision":"NaN"},"expected":0.95},{"fn":"recall","ds":"mcSparse","opts":{"average":"macro","labels":[99,9,2],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.9314214463840399},{"fn":"recall","ds":"mcSparse","opts":{"average":"weighted","labels":[99,9,2],"zeroDivision":"NaN"},"expected":0.9285714285714286},{"fn":"recall","ds":"mcSparse","opts":{"average":"weighted","labels":[99,9,2],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.902265659706797},{"fn":"recall","ds":"mcSparse","opts":{"average":null,"labels":[99,9,2],"zeroDivision":"NaN"},"expected":["NaN",1.0,0.9]},{"fn":"recall","ds":"mcSparse","opts":{"average":null,"labels":[99,9,2],"zeroDivision":"NaN","sampleWeight":"w"},"expected":["NaN",1.0,0.8628428927680798]},{"fn":"f1Score","ds":"bin","opts":{"average":"binary","zeroDivision":0},"expected":0.6956521739130435},{"fn":"f1Score","ds":"bin","opts":{"average":"binary","zeroDivision":0,"sampleWeight":"w"},"expected":0.7013614404918752},{"fn":"f1Score","ds":"bin","opts":{"average":"micro","zeroDivision":0},"expected":0.7083333333333334},{"fn":"f1Score","ds":"bin","opts":{"average":"micro","zeroDivision":0,"sampleWeight":"w"},"expected":0.7135636057287279},{"fn":"f1Score","ds":"bin","opts":{"average":"macro","zeroDivision":0},"expected":0.7078260869565217},{"fn":"f1Score","ds":"bin","opts":{"average":"macro","zeroDivision":0,"sampleWeight":"w"},"expected":0.7130846053127122},{"fn":"f1Score","ds":"bin","opts":{"average":"weighted","zeroDivision":0},"expected":0.7088405797101448},{"fn":"f1Score","ds":"bin","opts":{"average":"weighted","zeroDivision":0,"sampleWeight":"w"},"expected":0.7144425961828595},{"fn":"f1Score","ds":"bin","opts":{"average":null,"zeroDivision":0},"expected":[0.72,0.6956521739130435]},{"fn":"f1Score","ds":"bin","opts":{"average":null,"zeroDivision":0,"sampleWeight":"w"},"expected":[0.7248077701335491,0.7013614404918752]},{"fn":"f1Score","ds":"bin","opts":{"average":"binary","zeroDivision":0},"expected":0.0,"yPredOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"fn":"f1Score","ds":"bin","opts":{"average":"binary","zeroDivision":0},"expected":0.0,"yPredOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"yTrueOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"fn":"f1Score","ds":"bin","opts":{"average":"binary","zeroDivision":1},"expected":0.0,"yPredOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"fn":"f1Score","ds":"bin","opts":{"average":"binary","zeroDivision":1},"expected":1.0,"yPredOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"yTrueOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"fn":"f1Score","ds":"bin","opts":{"average":"binary","zeroDivision":"NaN"},"expected":0.0,"yPredOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"fn":"f1Score","ds":"bin","opts":{"average":"binary","zeroDivision":"NaN"},"expected":"NaN","yPredOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"yTrueOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"fn":"f1Score","ds":"mc3","opts":{"average":"micro","zeroDivision":0},"expected":0.8666666666666667},{"fn":"f1Score","ds":"mc3","opts":{"average":"micro","zeroDivision":0,"sampleWeight":"w"},"expected":0.9064950733963402},{"fn":"f1Score","ds":"mc3","opts":{"average":"macro","zeroDivision":0},"expected":0.8668531231460292},{"fn":"f1Score","ds":"mc3","opts":{"average":"macro","zeroDivision":0,"sampleWeight":"w"},"expected":0.905718825997779},{"fn":"f1Score","ds":"mc3","opts":{"average":"weighted","zeroDivision":0},"expected":0.86868378676159},{"fn":"f1Score","ds":"mc3","opts":{"average":"weighted","zeroDivision":0,"sampleWeight":"w"},"expected":0.9092642577814075},{"fn":"f1Score","ds":"mc3","opts":{"average":null,"zeroDivision":0},"expected":[0.8695652173913043,0.8421052631578947,0.8888888888888888]},{"fn":"f1Score","ds":"mc3","opts":{"average":null,"zeroDivision":0,"sampleWeight":"w"},"expected":[0.9026758228747336,0.8515426497277677,0.9629380053908355]},{"fn":"f1Score","ds":"mc3","opts":{"average":"micro","labels":[2,0],"zeroDivision":0},"expected":0.8780487804878049},{"fn":"f1Score","ds":"mc3","opts":{"average":"macro","labels":[2,0],"zeroDivision":0},"expected":0.8792270531400965},{"fn":"f1Score","ds":"mc3","opts":{"average":"weighted","labels":[2,0],"zeroDivision":0},"expected":0.8783487044356609},{"fn":"f1Score","ds":"mc3","opts":{"average":null,"labels":[2,0],"zeroDivision":1,"sampleWeight":"w"},"expected":[0.9629380053908355,0.9026758228747336]},{"fn":"f1Score","ds":"mc3","opts":{"average":"micro","labels":[2,0],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.9275483242942566},{"fn":"f1Score","ds":"mc3","opts":{"average":"macro","labels":[2,0],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.9328069141327846},{"fn":"f1Score","ds":"mc3","opts":{"average":"weighted","labels":[2,0],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.9270820067937549},{"fn":"f1Score","ds":"mc3","opts":{"average":null,"labels":[2,0],"zeroDivision":"NaN"},"expected":[0.8888888888888888,0.8695652173913043]},{"fn":"f1Score","ds":"mc3","opts":{"average":null,"labels":[1,99],"zeroDivision":0,"sampleWeight":"w"},"expected":[0.8515426497277677,0.0]},{"fn":"f1Score","ds":"mc3","opts":{"average":"micro","labels":[1,99],"zeroDivision":1,"sampleWeight":"w"},"expected":0.8515426497277677},{"fn":"f1Score","ds":"mc3","opts":{"average":"macro","labels":[1,99],"zeroDivision":1,"sampleWeight":"w"},"expected":0.9257713248638839},{"fn":"f1Score","ds":"mc3","opts":{"average":"weighted","labels":[1,99],"zeroDivision":1,"sampleWeight":"w"},"expected":0.8515426497277676},{"fn":"f1Score","ds":"mc3","opts":{"average":null,"labels":[1,99],"zeroDivision":1},"expected":[0.8421052631578947,1.0]},{"fn":"f1Score","ds":"mc3","opts":{"average":"micro","labels":[1,99],"zeroDivision":"NaN"},"expected":0.8421052631578947},{"fn":"f1Score","ds":"mc3","opts":{"average":"macro","labels":[1,99],"zeroDivision":"NaN"},"expected":0.8421052631578947},{"fn":"f1Score","ds":"mc3","opts":{"average":"weighted","labels":[1,99],"zeroDivision":"NaN"},"expected":0.8421052631578947},{"fn":"f1Score","ds":"mc3","opts":{"average":"micro","labels":[99,2,0],"zeroDivision":0,"sampleWeight":"w"},"expected":0.9275483242942566},{"fn":"f1Score","ds":"mc3","opts":{"average":"macro","labels":[99,2,0],"zeroDivision":0,"sampleWeight":"w"},"expected":0.6218712760885231},{"fn":"f1Score","ds":"mc3","opts":{"average":"weighted","labels":[99,2,0],"zeroDivision":0,"sampleWeight":"w"},"expected":0.9270820067937549},{"fn":"f1Score","ds":"mc3","opts":{"average":null,"labels":[99,2,0],"zeroDivision":0},"expected":[0.0,0.8888888888888888,0.8695652173913043]},{"fn":"f1Score","ds":"mc3","opts":{"average":"micro","labels":[99,2,0],"zeroDivision":1},"expected":0.8780487804878049},{"fn":"f1Score","ds":"mc3","opts":{"average":"macro","labels":[99,2,0],"zeroDivision":1},"expected":0.9194847020933977},{"fn":"f1Score","ds":"mc3","opts":{"average":"weighted","labels":[99,2,0],"zeroDivision":1},"expected":0.8783487044356609},{"fn":"f1Score","ds":"mc3","opts":{"average":"micro","labels":[99,2,0],"zeroDivision":"NaN"},"expected":0.8780487804878049},{"fn":"f1Score","ds":"mc3","opts":{"average":"micro","labels":[99,2,0],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.9275483242942566},{"fn":"f1Score","ds":"mc3","opts":{"average":"macro","labels":[99,2,0],"zeroDivision":"NaN"},"expected":0.8792270531400965},{"fn":"f1Score","ds":"mc3","opts":{"average":"macro","labels":[99,2,0],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.9328069141327846},{"fn":"f1Score","ds":"mc3","opts":{"average":"weighted","labels":[99,2,0],"zeroDivision":"NaN"},"expected":0.8783487044356609},{"fn":"f1Score","ds":"mc3","opts":{"average":"weighted","labels":[99,2,0],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.9270820067937549},{"fn":"f1Score","ds":"mc3","opts":{"average":null,"labels":[99,2,0],"zeroDivision":"NaN"},"expected":["NaN",0.8888888888888888,0.8695652173913043]},{"fn":"f1Score","ds":"mc3","opts":{"average":null,"labels":[99,2,0],"zeroDivision":"NaN","sampleWeight":"w"},"expected":["NaN",0.9629380053908355,0.9026758228747336]},{"fn":"f1Score","ds":"mc4s","opts":{"average":"micro","zeroDivision":0},"expected":0.7222222222222222},{"fn":"f1Score","ds":"mc4s","opts":{"average":"micro","zeroDivision":0,"sampleWeight":"w"},"expected":0.72636815920398},{"fn":"f1Score","ds":"mc4s","opts":{"average":"macro","zeroDivision":0},"expected":0.6735632183908047},{"fn":"f1Score","ds":"mc4s","opts":{"average":"macro","zeroDivision":0,"sampleWeight":"w"},"expected":0.6995759856938368},{"fn":"f1Score","ds":"mc4s","opts":{"average":"weighted","zeroDivision":0},"expected":0.7189016602809706},{"fn":"f1Score","ds":"mc4s","opts":{"average":"weighted","zeroDivision":0,"sampleWeight":"w"},"expected":0.725008888373727},{"fn":"f1Score","ds":"mc4s","opts":{"average":null,"zeroDivision":0},"expected":[0.4,0.8275862068965517,0.6666666666666666,0.8]},{"fn":"f1Score","ds":"mc4s","opts":{"average":null,"zeroDivision":0,"sampleWeight":"w"},"expected":[0.44071246819338417,0.8045548654244307,0.6853766617429836,0.8676599474145487]},{"fn":"f1Score","ds":"mc4s","opts":{"average":"micro","labels":["cat","ant"],"zeroDivision":0},"expected":0.5714285714285714},{"fn":"f1Score","ds":"mc4s","opts":{"average":"macro","labels":["cat","ant"],"zeroDivision":0},"expected":0.5333333333333333},{"fn":"f1Score","ds":"mc4s","opts":{"average":"weighted","labels":["cat","ant"],"zeroDivision":0},"expected":0.5435897435897435},{"fn":"f1Score","ds":"mc4s","opts":{"average":null,"labels":["cat","ant"],"zeroDivision":1,"sampleWeight":"w"},"expected":[0.6853766617429836,0.44071246819338417]},{"fn":"f1Score","ds":"mc4s","opts":{"average":"micro","labels":["cat","ant"],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.5955140186915887},{"fn":"f1Score","ds":"mc4s","opts":{"average":"macro","labels":["cat","ant"],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.5630445649681839},{"fn":"f1Score","ds":"mc4s","opts":{"average":"weighted","labels":["cat","ant"],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.5724929789467254},{"fn":"f1Score","ds":"mc4s","opts":{"average":null,"labels":["cat","ant"],"zeroDivision":"NaN"},"expected":[0.6666666666666666,0.4]},{"fn":"f1Score","ds":"mc4s","opts":{"average":null,"labels":["bee","zzz"],"zeroDivision":0,"sampleWeight":"w"},"expected":[0.8045548654244307,0.0]},{"fn":"f1Score","ds":"mc4s","opts":{"average":"micro","labels":["bee","zzz"],"zeroDivision":1,"sampleWeight":"w"},"expected":0.8045548654244307},{"fn":"f1Score","ds":"mc4s","opts":{"average":"macro","labels":["bee","zzz"],"zeroDivision":1,"sampleWeight":"w"},"expected":0.9022774327122154},{"fn":"f1Score","ds":"mc4s","opts":{"average":"weighted","labels":["bee","zzz"],"zeroDivision":1,"sampleWeight":"w"},"expected":0.8045548654244307},{"fn":"f1Score","ds":"mc4s","opts":{"average":null,"labels":["bee","zzz"],"zeroDivision":1},"expected":[0.8275862068965517,1.0]},{"fn":"f1Score","ds":"mc4s","opts":{"average":"micro","labels":["bee","zzz"],"zeroDivision":"NaN"},"expected":0.8275862068965517},{"fn":"f1Score","ds":"mc4s","opts":{"average":"macro","labels":["bee","zzz"],"zeroDivision":"NaN"},"expected":0.8275862068965517},{"fn":"f1Score","ds":"mc4s","opts":{"average":"weighted","labels":["bee","zzz"],"zeroDivision":"NaN"},"expected":0.8275862068965517},{"fn":"f1Score","ds":"mc4s","opts":{"average":"micro","labels":["zzz","cat","ant"],"zeroDivision":0,"sampleWeight":"w"},"expected":0.5955140186915887},{"fn":"f1Score","ds":"mc4s","opts":{"average":"macro","labels":["zzz","cat","ant"],"zeroDivision":0,"sampleWeight":"w"},"expected":0.3753630433121226},{"fn":"f1Score","ds":"mc4s","opts":{"average":"weighted","labels":["zzz","cat","ant"],"zeroDivision":0,"sampleWeight":"w"},"expected":0.5724929789467254},{"fn":"f1Score","ds":"mc4s","opts":{"average":null,"labels":["zzz","cat","ant"],"zeroDivision":0},"expected":[0.0,0.6666666666666666,0.4]},{"fn":"f1Score","ds":"mc4s","opts":{"average":"micro","labels":["zzz","cat","ant"],"zeroDivision":1},"expected":0.5714285714285714},{"fn":"f1Score","ds":"mc4s","opts":{"average":"macro","labels":["zzz","cat","ant"],"zeroDivision":1},"expected":0.6888888888888888},{"fn":"f1Score","ds":"mc4s","opts":{"average":"weighted","labels":["zzz","cat","ant"],"zeroDivision":1},"expected":0.5435897435897435},{"fn":"f1Score","ds":"mc4s","opts":{"average":"micro","labels":["zzz","cat","ant"],"zeroDivision":"NaN"},"expected":0.5714285714285714},{"fn":"f1Score","ds":"mc4s","opts":{"average":"micro","labels":["zzz","cat","ant"],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.5955140186915887},{"fn":"f1Score","ds":"mc4s","opts":{"average":"macro","labels":["zzz","cat","ant"],"zeroDivision":"NaN"},"expected":0.5333333333333333},{"fn":"f1Score","ds":"mc4s","opts":{"average":"macro","labels":["zzz","cat","ant"],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.5630445649681839},{"fn":"f1Score","ds":"mc4s","opts":{"average":"weighted","labels":["zzz","cat","ant"],"zeroDivision":"NaN"},"expected":0.5435897435897435},{"fn":"f1Score","ds":"mc4s","opts":{"average":"weighted","labels":["zzz","cat","ant"],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.5724929789467254},{"fn":"f1Score","ds":"mc4s","opts":{"average":null,"labels":["zzz","cat","ant"],"zeroDivision":"NaN"},"expected":["NaN",0.6666666666666666,0.4]},{"fn":"f1Score","ds":"mc4s","opts":{"average":null,"labels":["zzz","cat","ant"],"zeroDivision":"NaN","sampleWeight":"w"},"expected":["NaN",0.6853766617429836,0.44071246819338417]},{"fn":"f1Score","ds":"mcSparse","opts":{"average":"micro","zeroDivision":0},"expected":0.8148148148148148},{"fn":"f1Score","ds":"mcSparse","opts":{"average":"micro","zeroDivision":0,"sampleWeight":"w"},"expected":0.7927677329624477},{"fn":"f1Score","ds":"mcSparse","opts":{"average":"macro","zeroDivision":0},"expected":0.8121212121212121},{"fn":"f1Score","ds":"mcSparse","opts":{"average":"macro","zeroDivision":0,"sampleWeight":"w"},"expected":0.787930584214767},{"fn":"f1Score","ds":"mcSparse","opts":{"average":"weighted","zeroDivision":0},"expected":0.8154882154882155},{"fn":"f1Score","ds":"mcSparse","opts":{"average":"weighted","zeroDivision":0,"sampleWeight":"w"},"expected":0.7949570822586056},{"fn":"f1Score","ds":"mcSparse","opts":{"average":null,"zeroDivision":0},"expected":[0.8181818181818182,0.8181818181818182,0.8]},{"fn":"f1Score","ds":"mcSparse","opts":{"average":null,"zeroDivision":0,"sampleWeight":"w"},"expected":[0.794717197817973,0.8047508690614136,0.7643236857649144]},{"fn":"f1Score","ds":"mcSparse","opts":{"average":"micro","labels":[9,2],"zeroDivision":0},"expected":0.8125},{"fn":"f1Score","ds":"mcSparse","opts":{"average":"macro","labels":[9,2],"zeroDivision":0},"expected":0.8090909090909091},{"fn":"f1Score","ds":"mcSparse","opts":{"average":"weighted","labels":[9,2],"zeroDivision":0},"expected":0.812987012987013},{"fn":"f1Score","ds":"mcSparse","opts":{"average":null,"labels":[9,2],"zeroDivision":1,"sampleWeight":"w"},"expected":[0.7643236857649144,0.794717197817973]},{"fn":"f1Score","ds":"mcSparse","opts":{"average":"micro","labels":[9,2],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.7847758887171561},{"fn":"f1Score","ds":"mcSparse","opts":{"average":"macro","labels":[9,2],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.7795204417914436},{"fn":"f1Score","ds":"mcSparse","opts":{"average":"weighted","labels":[9,2],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.7859812572145394},{"fn":"f1Score","ds":"mcSparse","opts":{"average":null,"labels":[9,2],"zeroDivision":"NaN"},"expected":[0.8,0.8181818181818182]},{"fn":"f1Score","ds":"mcSparse","opts":{"average":null,"labels":[5,99],"zeroDivision":0,"sampleWeight":"w"},"expected":[0.8047508690614136,0.0]},{"fn":"f1Score","ds":"mcSparse","opts":{"average":"micro","labels":[5,99],"zeroDivision":1,"sampleWeight":"w"},"expected":0.8047508690614136},{"fn":"f1Score","ds":"mcSparse","opts":{"average":"macro","labels":[5,99],"zeroDivision":1,"sampleWeight":"w"},"expected":0.9023754345307068},{"fn":"f1Score","ds":"mcSparse","opts":{"average":"weighted","labels":[5,99],"zeroDivision":1,"sampleWeight":"w"},"expected":0.8047508690614136},{"fn":"f1Score","ds":"mcSparse","opts":{"average":null,"labels":[5,99],"zeroDivision":1},"expected":[0.8181818181818182,1.0]},{"fn":"f1Score","ds":"mcSparse","opts":{"average":"micro","labels":[5,99],"zeroDivision":"NaN"},"expected":0.8181818181818182},{"fn":"f1Score","ds":"mcSparse","opts":{"average":"macro","labels":[5,99],"zeroDivision":"NaN"},"expected":0.8181818181818182},{"fn":"f1Score","ds":"mcSparse","opts":{"average":"weighted","labels":[5,99],"zeroDivision":"NaN"},"expected":0.8181818181818182},{"fn":"f1Score","ds":"mcSparse","opts":{"average":"micro","labels":[99,9,2],"zeroDivision":0,"sampleWeight":"w"},"expected":0.7847758887171561},{"fn":"f1Score","ds":"mcSparse","opts":{"average":"macro","labels":[99,9,2],"zeroDivision":0,"sampleWeight":"w"},"expected":0.5196802945276291},{"fn":"f1Score","ds":"mcSparse","opts":{"average":"weighted","labels":[99,9,2],"zeroDivision":0,"sampleWeight":"w"},"expected":0.7859812572145394},{"fn":"f1Score","ds":"mcSparse","opts":{"average":null,"labels":[99,9,2],"zeroDivision":0},"expected":[0.0,0.8,0.8181818181818182]},{"fn":"f1Score","ds":"mcSparse","opts":{"average":"micro","labels":[99,9,2],"zeroDivision":1},"expected":0.8125},{"fn":"f1Score","ds":"mcSparse","opts":{"average":"macro","labels":[99,9,2],"zeroDivision":1},"expected":0.8727272727272727},{"fn":"f1Score","ds":"mcSparse","opts":{"average":"weighted","labels":[99,9,2],"zeroDivision":1},"expected":0.812987012987013},{"fn":"f1Score","ds":"mcSparse","opts":{"average":"micro","labels":[99,9,2],"zeroDivision":"NaN"},"expected":0.8125},{"fn":"f1Score","ds":"mcSparse","opts":{"average":"micro","labels":[99,9,2],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.7847758887171561},{"fn":"f1Score","ds":"mcSparse","opts":{"average":"macro","labels":[99,9,2],"zeroDivision":"NaN"},"expected":0.8090909090909091},{"fn":"f1Score","ds":"mcSparse","opts":{"average":"macro","labels":[99,9,2],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.7795204417914436},{"fn":"f1Score","ds":"mcSparse","opts":{"average":"weighted","labels":[99,9,2],"zeroDivision":"NaN"},"expected":0.812987012987013},{"fn":"f1Score","ds":"mcSparse","opts":{"average":"weighted","labels":[99,9,2],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.7859812572145394},{"fn":"f1Score","ds":"mcSparse","opts":{"average":null,"labels":[99,9,2],"zeroDivision":"NaN"},"expected":["NaN",0.8,0.8181818181818182]},{"fn":"f1Score","ds":"mcSparse","opts":{"average":null,"labels":[99,9,2],"zeroDivision":"NaN","sampleWeight":"w"},"expected":["NaN",0.7643236857649144,0.794717197817973]},{"fn":"fbeta05","ds":"bin","opts":{"average":"binary","zeroDivision":0},"expected":0.6779661016949152},{"fn":"fbeta05","ds":"bin","opts":{"average":"binary","zeroDivision":0,"sampleWeight":"w"},"expected":0.6699387532511117},{"fn":"fbeta05","ds":"bin","opts":{"average":"micro","zeroDivision":0},"expected":0.7083333333333334},{"fn":"fbeta05","ds":"bin","opts":{"average":"micro","zeroDivision":0,"sampleWeight":"w"},"expected":0.7135636057287278},{"fn":"fbeta05","ds":"bin","opts":{"average":"macro","zeroDivision":0},"expected":0.7078355098638511},{"fn":"fbeta05","ds":"bin","opts":{"average":"macro","zeroDivision":0,"sampleWeight":"w"},"expected":0.7137444379570845},{"fn":"fbeta05","ds":"bin","opts":{"average":"weighted","zeroDivision":0},"expected":0.7103246272112624},{"fn":"fbeta05","ds":"bin","opts":{"average":"weighted","zeroDivision":0,"sampleWeight":"w"},"expected":0.7188188117119886},{"fn":"fbeta05","ds":"bin","opts":{"average":null,"zeroDivision":0},"expected":[0.7377049180327869,0.6779661016949152]},{"fn":"fbeta05","ds":"bin","opts":{"average":null,"zeroDivision":0,"sampleWeight":"w"},"expected":[0.7575501226630573,0.6699387532511117]},{"fn":"fbeta05","ds":"bin","opts":{"average":"binary","zeroDivision":0},"expected":0.0,"yPredOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"fn":"fbeta05","ds":"bin","opts":{"average":"binary","zeroDivision":0},"expected":0.0,"yPredOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"yTrueOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"fn":"fbeta05","ds":"bin","opts":{"average":"binary","zeroDivision":1},"expected":0.0,"yPredOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"fn":"fbeta05","ds":"bin","opts":{"average":"binary","zeroDivision":1},"expected":1.0,"yPredOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"yTrueOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"fn":"fbeta05","ds":"bin","opts":{"average":"binary","zeroDivision":"NaN"},"expected":0.0,"yPredOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"fn":"fbeta05","ds":"bin","opts":{"average":"binary","zeroDivision":"NaN"},"expected":"NaN","yPredOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"yTrueOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"fn":"fbeta05","ds":"mc3","opts":{"average":"micro","zeroDivision":0},"expected":0.8666666666666667},{"fn":"fbeta05","ds":"mc3","opts":{"average":"micro","zeroDivision":0,"sampleWeight":"w"},"expected":0.9064950733963403},{"fn":"fbeta05","ds":"mc3","opts":{"average":"macro","zeroDivision":0},"expected":0.8714896214896215},{"fn":"fbeta05","ds":"mc3","opts":{"average":"macro","zeroDivision":0,"sampleWeight":"w"},"expected":0.9031526983458468},{"fn":"fbeta05","ds":"mc3","opts":{"average":"weighted","zeroDivision":0},"expected":0.8797313797313798},{"fn":"fbeta05","ds":"mc3","opts":{"average":"weighted","zeroDivision":0,"sampleWeight":"w"},"expected":0.9178220617412353},{"fn":"fbeta05","ds":"mc3","opts":{"average":null,"zeroDivision":0},"expected":[0.8928571428571429,0.7692307692307693,0.9523809523809523]},{"fn":"fbeta05","ds":"mc3","opts":{"average":null,"zeroDivision":0,"sampleWeight":"w"},"expected":[0.942724305074686,0.7818957472337021,0.9848380427291522]},{"fn":"fbeta05","ds":"mc3","opts":{"average":"micro","labels":[2,0],"zeroDivision":0},"expected":0.9183673469387755},{"fn":"fbeta05","ds":"mc3","opts":{"average":"macro","labels":[2,0],"zeroDivision":0},"expected":0.9226190476190477},{"fn":"fbeta05","ds":"mc3","opts":{"average":"weighted","labels":[2,0],"zeroDivision":0},"expected":0.91991341991342},{"fn":"fbeta05","ds":"mc3","opts":{"average":null,"labels":[2,0],"zeroDivision":1,"sampleWeight":"w"},"expected":[0.9848380427291522,0.942724305074686]},{"fn":"fbeta05","ds":"mc3","opts":{"average":"micro","labels":[2,0],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.9603202027182676},{"fn":"fbeta05","ds":"mc3","opts":{"average":"macro","labels":[2,0],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.9637811739019191},{"fn":"fbeta05","ds":"mc3","opts":{"average":"weighted","labels":[2,0],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.9597803688247449},{"fn":"fbeta05","ds":"mc3","opts":{"average":null,"labels":[2,0],"zeroDivision":"NaN"},"expected":[0.9523809523809523,0.8928571428571429]},{"fn":"fbeta05","ds":"mc3","opts":{"average":null,"labels":[1,99],"zeroDivision":0,"sampleWeight":"w"},"expected":[0.7818957472337021,0.0]},{"fn":"fbeta05","ds":"mc3","opts":{"average":"micro","labels":[1,99],"zeroDivision":1,"sampleWeight":"w"},"expected":0.7818957472337021},{"fn":"fbeta05","ds":"mc3","opts":{"average":"macro","labels":[1,99],"zeroDivision":1,"sampleWeight":"w"},"expected":0.890947873616851},{"fn":"fbeta05","ds":"mc3","opts":{"average":"weighted","labels":[1,99],"zeroDivision":1,"sampleWeight":"w"},"expected":0.7818957472337023},{"fn":"fbeta05","ds":"mc3","opts":{"average":null,"labels":[1,99],"zeroDivision":1},"expected":[0.7692307692307693,1.0]},{"fn":"fbeta05","ds":"mc3","opts":{"average":"micro","labels":[1,99],"zeroDivision":"NaN"},"expected":0.7692307692307693},{"fn":"fbeta05","ds":"mc3","opts":{"average":"macro","labels":[1,99],"zeroDivision":"NaN"},"expected":0.7692307692307693},{"fn":"fbeta05","ds":"mc3","opts":{"average":"weighted","labels":[1,99],"zeroDivision":"NaN"},"expected":0.7692307692307693},{"fn":"fbeta05","ds":"mc3","opts":{"average":"micro","labels":[99,2,0],"zeroDivision":0,"sampleWeight":"w"},"expected":0.9603202027182676},{"fn":"fbeta05","ds":"mc3","opts":{"average":"macro","labels":[99,2,0],"zeroDivision":0,"sampleWeight":"w"},"expected":0.6425207826012794},{"fn":"fbeta05","ds":"mc3","opts":{"average":"weighted","labels":[99,2,0],"zeroDivision":0,"sampleWeight":"w"},"expected":0.9597803688247449},{"fn":"fbeta05","ds":"mc3","opts":{"average":null,"labels":[99,2,0],"zeroDivision":0},"expected":[0.0,0.9523809523809523,0.8928571428571429]},{"fn":"fbeta05","ds":"mc3","opts":{"average":"micro","labels":[99,2,0],"zeroDivision":1},"expected":0.9183673469387755},{"fn":"fbeta05","ds":"mc3","opts":{"average":"macro","labels":[99,2,0],"zeroDivision":1},"expected":0.9484126984126985},{"fn":"fbeta05","ds":"mc3","opts":{"average":"weighted","labels":[99,2,0],"zeroDivision":1},"expected":0.91991341991342},{"fn":"fbeta05","ds":"mc3","opts":{"average":"micro","labels":[99,2,0],"zeroDivision":"NaN"},"expected":0.9183673469387755},{"fn":"fbeta05","ds":"mc3","opts":{"average":"micro","labels":[99,2,0],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.9603202027182676},{"fn":"fbeta05","ds":"mc3","opts":{"average":"macro","labels":[99,2,0],"zeroDivision":"NaN"},"expected":0.9226190476190477},{"fn":"fbeta05","ds":"mc3","opts":{"average":"macro","labels":[99,2,0],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.9637811739019191},{"fn":"fbeta05","ds":"mc3","opts":{"average":"weighted","labels":[99,2,0],"zeroDivision":"NaN"},"expected":0.91991341991342},{"fn":"fbeta05","ds":"mc3","opts":{"average":"weighted","labels":[99,2,0],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.9597803688247449},{"fn":"fbeta05","ds":"mc3","opts":{"average":null,"labels":[99,2,0],"zeroDivision":"NaN"},"expected":["NaN",0.9523809523809523,0.8928571428571429]},{"fn":"fbeta05","ds":"mc3","opts":{"average":null,"labels":[99,2,0],"zeroDivision":"NaN","sampleWeight":"w"},"expected":["NaN",0.9848380427291522,0.942724305074686]},{"fn":"fbeta05","ds":"mc4s","opts":{"average":"micro","zeroDivision":0},"expected":0.7222222222222222},{"fn":"fbeta05","ds":"mc4s","opts":{"average":"micro","zeroDivision":0,"sampleWeight":"w"},"expected":0.7263681592039801},{"fn":"fbeta05","ds":"mc4s","opts":{"average":"macro","zeroDivision":0},"expected":0.6802961261329116},{"fn":"fbeta05","ds":"mc4s","opts":{"average":"macro","zeroDivision":0,"sampleWeight":"w"},"expected":0.7056620035556402},{"fn":"fbeta05","ds":"mc4s","opts":{"average":"weighted","zeroDivision":0},"expected":0.7274345219664192},{"fn":"fbeta05","ds":"mc4s","opts":{"average":"weighted","zeroDivision":0,"sampleWeight":"w"},"expected":0.7351727899941538},{"fn":"fbeta05","ds":"mc4s","opts":{"average":null,"zeroDivision":0},"expected":[0.45454545454545453,0.8450704225352113,0.5882352941176471,0.8333333333333334]},{"fn":"fbeta05","ds":"mc4s","opts":{"average":null,"zeroDivision":0,"sampleWeight":"w"},"expected":[0.48597081930415253,0.8387291720625055,0.6063774176685832,0.8915706051873198]},{"fn":"fbeta05","ds":"mc4s","opts":{"average":"micro","labels":["cat","ant"],"zeroDivision":0},"expected":0.547945205479452},{"fn":"fbeta05","ds":"mc4s","opts":{"average":"macro","labels":["cat","ant"],"zeroDivision":0},"expected":0.5213903743315508},{"fn":"fbeta05","ds":"mc4s","opts":{"average":"weighted","labels":["cat","ant"],"zeroDivision":0},"expected":0.5265322912381737},{"fn":"fbeta05","ds":"mc4s","opts":{"average":null,"labels":["cat","ant"],"zeroDivision":1,"sampleWeight":"w"},"expected":[0.6063774176685832,0.48597081930415253]},{"fn":"fbeta05","ds":"mc4s","opts":{"average":"micro","labels":["cat","ant"],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.5681169757489299},{"fn":"fbeta05","ds":"mc4s","opts":{"average":"macro","labels":["cat","ant"],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.5461741184863679},{"fn":"fbeta05","ds":"mc4s","opts":{"average":"weighted","labels":["cat","ant"],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.5508239667971894},{"fn":"fbeta05","ds":"mc4s","opts":{"average":null,"labels":["cat","ant"],"zeroDivision":"NaN"},"expected":[0.5882352941176471,0.45454545454545453]},{"fn":"fbeta05","ds":"mc4s","opts":{"average":null,"labels":["bee","zzz"],"zeroDivision":0,"sampleWeight":"w"},"expected":[0.8387291720625055,0.0]},{"fn":"fbeta05","ds":"mc4s","opts":{"average":"micro","labels":["bee","zzz"],"zeroDivision":1,"sampleWeight":"w"},"expected":0.8387291720625055},{"fn":"fbeta05","ds":"mc4s","opts":{"average":"macro","labels":["bee","zzz"],"zeroDivision":1,"sampleWeight":"w"},"expected":0.9193645860312527},{"fn":"fbeta05","ds":"mc4s","opts":{"average":"weighted","labels":["bee","zzz"],"zeroDivision":1,"sampleWeight":"w"},"expected":0.8387291720625055},{"fn":"fbeta05","ds":"mc4s","opts":{"average":null,"labels":["bee","zzz"],"zeroDivision":1},"expected":[0.8450704225352113,1.0]},{"fn":"fbeta05","ds":"mc4s","opts":{"average":"micro","labels":["bee","zzz"],"zeroDivision":"NaN"},"expected":0.8450704225352113},{"fn":"fbeta05","ds":"mc4s","opts":{"average":"macro","labels":["bee","zzz"],"zeroDivision":"NaN"},"expected":0.8450704225352113},{"fn":"fbeta05","ds":"mc4s","opts":{"average":"weighted","labels":["bee","zzz"],"zeroDivision":"NaN"},"expected":0.8450704225352113},{"fn":"fbeta05","ds":"mc4s","opts":{"average":"micro","labels":["zzz","cat","ant"],"zeroDivision":0,"sampleWeight":"w"},"expected":0.5681169757489299},{"fn":"fbeta05","ds":"mc4s","opts":{"average":"macro","labels":["zzz","cat","ant"],"zeroDivision":0,"sampleWeight":"w"},"expected":0.3641160789909119},{"fn":"fbeta05","ds":"mc4s","opts":{"average":"weighted","labels":["zzz","cat","ant"],"zeroDivision":0,"sampleWeight":"w"},"expected":0.5508239667971894},{"fn":"fbeta05","ds":"mc4s","opts":{"average":null,"labels":["zzz","cat","ant"],"zeroDivision":0},"expected":[0.0,0.5882352941176471,0.45454545454545453]},{"fn":"fbeta05","ds":"mc4s","opts":{"average":"micro","labels":["zzz","cat","ant"],"zeroDivision":1},"expected":0.547945205479452},{"fn":"fbeta05","ds":"mc4s","opts":{"average":"macro","labels":["zzz","cat","ant"],"zeroDivision":1},"expected":0.680926916221034},{"fn":"fbeta05","ds":"mc4s","opts":{"average":"weighted","labels":["zzz","cat","ant"],"zeroDivision":1},"expected":0.5265322912381737},{"fn":"fbeta05","ds":"mc4s","opts":{"average":"micro","labels":["zzz","cat","ant"],"zeroDivision":"NaN"},"expected":0.547945205479452},{"fn":"fbeta05","ds":"mc4s","opts":{"average":"micro","labels":["zzz","cat","ant"],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.5681169757489299},{"fn":"fbeta05","ds":"mc4s","opts":{"average":"macro","labels":["zzz","cat","ant"],"zeroDivision":"NaN"},"expected":0.5213903743315508},{"fn":"fbeta05","ds":"mc4s","opts":{"average":"macro","labels":["zzz","cat","ant"],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.5461741184863679},{"fn":"fbeta05","ds":"mc4s","opts":{"average":"weighted","labels":["zzz","cat","ant"],"zeroDivision":"NaN"},"expected":0.5265322912381737},{"fn":"fbeta05","ds":"mc4s","opts":{"average":"weighted","labels":["zzz","cat","ant"],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.5508239667971894},{"fn":"fbeta05","ds":"mc4s","opts":{"average":null,"labels":["zzz","cat","ant"],"zeroDivision":"NaN"},"expected":["NaN",0.5882352941176471,0.45454545454545453]},{"fn":"fbeta05","ds":"mc4s","opts":{"average":null,"labels":["zzz","cat","ant"],"zeroDivision":"NaN","sampleWeight":"w"},"expected":["NaN",0.6063774176685832,0.48597081930415253]},{"fn":"fbeta2","ds":"bin","opts":{"average":"binary","zeroDivision":0},"expected":0.7142857142857143},{"fn":"fbeta2","ds":"bin","opts":{"average":"binary","zeroDivision":0,"sampleWeight":"w"},"expected":0.7358768777071238},{"fn":"fbeta2","ds":"bin","opts":{"average":"micro","zeroDivision":0},"expected":0.7083333333333334},{"fn":"fbeta2","ds":"bin","opts":{"average":"micro","zeroDivision":0,"sampleWeight":"w"},"expected":0.7135636057287278},{"fn":"fbeta2","ds":"bin","opts":{"average":"macro","zeroDivision":0},"expected":0.7087053571428572},{"fn":"fbeta2","ds":"bin","opts":{"average":"macro","zeroDivision":0,"sampleWeight":"w"},"expected":0.7153276854979874},{"fn":"fbeta2","ds":"bin","opts":{"average":"weighted","zeroDivision":0},"expected":0.7082403273809524},{"fn":"fbeta2","ds":"bin","opts":{"average":"weighted","zeroDivision":0,"sampleWeight":"w"},"expected":0.7129473030811749},{"fn":"fbeta2","ds":"bin","opts":{"average":null,"zeroDivision":0},"expected":[0.703125,0.7142857142857143]},{"fn":"fbeta2","ds":"bin","opts":{"average":null,"zeroDivision":0,"sampleWeight":"w"},"expected":[0.694778493288851,0.7358768777071238]},{"fn":"fbeta2","ds":"bin","opts":{"average":"binary","zeroDivision":0},"expected":0.0,"yPredOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"fn":"fbeta2","ds":"bin","opts":{"average":"binary","zeroDivision":0},"expected":0.0,"yPredOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"yTrueOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"fn":"fbeta2","ds":"bin","opts":{"average":"binary","zeroDivision":1},"expected":0.0,"yPredOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"fn":"fbeta2","ds":"bin","opts":{"average":"binary","zeroDivision":1},"expected":1.0,"yPredOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"yTrueOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"fn":"fbeta2","ds":"bin","opts":{"average":"binary","zeroDivision":"NaN"},"expected":0.0,"yPredOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"fn":"fbeta2","ds":"bin","opts":{"average":"binary","zeroDivision":"NaN"},"expected":"NaN","yPredOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"yTrueOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"fn":"fbeta2","ds":"mc3","opts":{"average":"micro","zeroDivision":0},"expected":0.8666666666666667},{"fn":"fbeta2","ds":"mc3","opts":{"average":"micro","zeroDivision":0,"sampleWeight":"w"},"expected":0.9064950733963402},{"fn":"fbeta2","ds":"mc3","opts":{"average":"macro","zeroDivision":0},"expected":0.8703411728638374},{"fn":"fbeta2","ds":"mc3","opts":{"average":"macro","zeroDivision":0,"sampleWeight":"w"},"expected":0.9142308105329895},{"fn":"fbeta2","ds":"mc3","opts":{"average":"weighted","zeroDivision":0},"expected":0.8648228441291115},{"fn":"fbeta2","ds":"mc3","opts":{"average":"weighted","zeroDivision":0,"sampleWeight":"w"},"expected":0.9056980924518796},{"fn":"fbeta2","ds":"mc3","opts":{"average":null,"zeroDivision":0},"expected":[0.847457627118644,0.9302325581395349,0.8333333333333334]},{"fn":"fbeta2","ds":"mc3","opts":{"average":null,"zeroDivision":0,"sampleWeight":"w"},"expected":[0.8658913320007269,0.9348103283391777,0.9419907712590638]},{"fn":"fbeta2","ds":"mc3","opts":{"average":"micro","labels":[2,0],"zeroDivision":0},"expected":0.8411214953271028},{"fn":"fbeta2","ds":"mc3","opts":{"average":"macro","labels":[2,0],"zeroDivision":0},"expected":0.8403954802259888},{"fn":"fbeta2","ds":"mc3","opts":{"average":"weighted","labels":[2,0],"zeroDivision":0},"expected":0.8410374935798665},{"fn":"fbeta2","ds":"mc3","opts":{"average":null,"labels":[2,0],"zeroDivision":1,"sampleWeight":"w"},"expected":[0.9419907712590638,0.8658913320007269]},{"fn":"fbeta2","ds":"mc3","opts":{"average":"micro","labels":[2,0],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.8969393792695389},{"fn":"fbeta2","ds":"mc3","opts":{"average":"macro","labels":[2,0],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.9039410516298954},{"fn":"fbeta2","ds":"mc3","opts":{"average":"weighted","labels":[2,0],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.8967116049003533},{"fn":"fbeta2","ds":"mc3","opts":{"average":null,"labels":[2,0],"zeroDivision":"NaN"},"expected":[0.8333333333333334,0.847457627118644]},{"fn":"fbeta2","ds":"mc3","opts":{"average":null,"labels":[1,99],"zeroDivision":0,"sampleWeight":"w"},"expected":[0.9348103283391777,0.0]},{"fn":"fbeta2","ds":"mc3","opts":{"average":"micro","labels":[1,99],"zeroDivision":1,"sampleWeight":"w"},"expected":0.9348103283391777},{"fn":"fbeta2","ds":"mc3","opts":{"average":"macro","labels":[1,99],"zeroDivision":1,"sampleWeight":"w"},"expected":0.9674051641695889},{"fn":"fbeta2","ds":"mc3","opts":{"average":"weighted","labels":[1,99],"zeroDivision":1,"sampleWeight":"w"},"expected":0.9348103283391777},{"fn":"fbeta2","ds":"mc3","opts":{"average":null,"labels":[1,99],"zeroDivision":1},"expected":[0.9302325581395349,1.0]},{"fn":"fbeta2","ds":"mc3","opts":{"average":"micro","labels":[1,99],"zeroDivision":"NaN"},"expected":0.9302325581395349},{"fn":"fbeta2","ds":"mc3","opts":{"average":"macro","labels":[1,99],"zeroDivision":"NaN"},"expected":0.9302325581395349},{"fn":"fbeta2","ds":"mc3","opts":{"average":"weighted","labels":[1,99],"zeroDivision":"NaN"},"expected":0.9302325581395349},{"fn":"fbeta2","ds":"mc3","opts":{"average":"micro","labels":[99,2,0],"zeroDivision":0,"sampleWeight":"w"},"expected":0.8969393792695389},{"fn":"fbeta2","ds":"mc3","opts":{"average":"macro","labels":[99,2,0],"zeroDivision":0,"sampleWeight":"w"},"expected":0.6026273677532635},{"fn":"fbeta2","ds":"mc3","opts":{"average":"weighted","labels":[99,2,0],"zeroDivision":0,"sampleWeight":"w"},"expected":0.8967116049003533},{"fn":"fbeta2","ds":"mc3","opts":{"average":null,"labels":[99,2,0],"zeroDivision":0},"expected":[0.0,0.8333333333333334,0.847457627118644]},{"fn":"fbeta2","ds":"mc3","opts":{"average":"micro","labels":[99,2,0],"zeroDivision":1},"expected":0.8411214953271028},{"fn":"fbeta2","ds":"mc3","opts":{"average":"macro","labels":[99,2,0],"zeroDivision":1},"expected":0.8935969868173258},{"fn":"fbeta2","ds":"mc3","opts":{"average":"weighted","labels":[99,2,0],"zeroDivision":1},"expected":0.8410374935798665},{"fn":"fbeta2","ds":"mc3","opts":{"average":"micro","labels":[99,2,0],"zeroDivision":"NaN"},"expected":0.8411214953271028},{"fn":"fbeta2","ds":"mc3","opts":{"average":"micro","labels":[99,2,0],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.8969393792695389},{"fn":"fbeta2","ds":"mc3","opts":{"average":"macro","labels":[99,2,0],"zeroDivision":"NaN"},"expected":0.8403954802259888},{"fn":"fbeta2","ds":"mc3","opts":{"average":"macro","labels":[99,2,0],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.9039410516298954},{"fn":"fbeta2","ds":"mc3","opts":{"average":"weighted","labels":[99,2,0],"zeroDivision":"NaN"},"expected":0.8410374935798665},{"fn":"fbeta2","ds":"mc3","opts":{"average":"weighted","labels":[99,2,0],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.8967116049003533},{"fn":"fbeta2","ds":"mc3","opts":{"average":null,"labels":[99,2,0],"zeroDivision":"NaN"},"expected":["NaN",0.8333333333333334,0.847457627118644]},{"fn":"fbeta2","ds":"mc3","opts":{"average":null,"labels":[99,2,0],"zeroDivision":"NaN","sampleWeight":"w"},"expected":["NaN",0.9419907712590638,0.8658913320007269]},{"fn":"fbeta2","ds":"mc4s","opts":{"average":"micro","zeroDivision":0},"expected":0.7222222222222222},{"fn":"fbeta2","ds":"mc4s","opts":{"average":"micro","zeroDivision":0,"sampleWeight":"w"},"expected":0.72636815920398},{"fn":"fbeta2","ds":"mc4s","opts":{"average":"macro","zeroDivision":0},"expected":0.6766038016038016},{"fn":"fbeta2","ds":"mc4s","opts":{"average":"macro","zeroDivision":0,"sampleWeight":"w"},"expected":0.7023159810908091},{"fn":"fbeta2","ds":"mc4s","opts":{"average":"weighted","zeroDivision":0},"expected":0.7178744678744678},{"fn":"fbeta2","ds":"mc4s","opts":{"average":"weighted","zeroDivision":0,"sampleWeight":"w"},"expected":0.7226289816619796},{"fn":"fbeta2","ds":"mc4s","opts":{"average":null,"zeroDivision":0},"expected":[0.35714285714285715,0.8108108108108109,0.7692307692307693,0.7692307692307693]},{"fn":"fbeta2","ds":"mc4s","opts":{"average":null,"zeroDivision":0,"sampleWeight":"w"},"expected":[0.40316573556797014,0.7730564176016552,0.7880434782608694,0.8449982929327415]},{"fn":"fbeta2","ds":"mc4s","opts":{"average":"micro","labels":["cat","ant"],"zeroDivision":0},"expected":0.5970149253731343},{"fn":"fbeta2","ds":"mc4s","opts":{"average":"macro","labels":["cat","ant"],"zeroDivision":0},"expected":0.5631868131868132},{"fn":"fbeta2","ds":"mc4s","opts":{"average":"weighted","labels":["cat","ant"],"zeroDivision":0},"expected":0.5790363482671175},{"fn":"fbeta2","ds":"mc4s","opts":{"average":null,"labels":["cat","ant"],"zeroDivision":1,"sampleWeight":"w"},"expected":[0.7880434782608694,0.40316573556797014]},{"fn":"fbeta2","ds":"mc4s","opts":{"average":"micro","labels":["cat","ant"],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.6256873527101334},{"fn":"fbeta2","ds":"mc4s","opts":{"average":"macro","labels":["cat","ant"],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.5956046069144197},{"fn":"fbeta2","ds":"mc4s","opts":{"average":"weighted","labels":["cat","ant"],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.6104677717745114},{"fn":"fbeta2","ds":"mc4s","opts":{"average":null,"labels":["cat","ant"],"zeroDivision":"NaN"},"expected":[0.7692307692307693,0.35714285714285715]},{"fn":"fbeta2","ds":"mc4s","opts":{"average":null,"labels":["bee","zzz"],"zeroDivision":0,"sampleWeight":"w"},"expected":[0.7730564176016552,0.0]},{"fn":"fbeta2","ds":"mc4s","opts":{"average":"micro","labels":["bee","zzz"],"zeroDivision":1,"sampleWeight":"w"},"expected":0.7730564176016552},{"fn":"fbeta2","ds":"mc4s","opts":{"average":"macro","labels":["bee","zzz"],"zeroDivision":1,"sampleWeight":"w"},"expected":0.8865282088008276},{"fn":"fbeta2","ds":"mc4s","opts":{"average":"weighted","labels":["bee","zzz"],"zeroDivision":1,"sampleWeight":"w"},"expected":0.7730564176016552},{"fn":"fbeta2","ds":"mc4s","opts":{"average":null,"labels":["bee","zzz"],"zeroDivision":1},"expected":[0.8108108108108109,1.0]},{"fn":"fbeta2","ds":"mc4s","opts":{"average":"micro","labels":["bee","zzz"],"zeroDivision":"NaN"},"expected":0.8108108108108109},{"fn":"fbeta2","ds":"mc4s","opts":{"average":"macro","labels":["bee","zzz"],"zeroDivision":"NaN"},"expected":0.8108108108108109},{"fn":"fbeta2","ds":"mc4s","opts":{"average":"weighted","labels":["bee","zzz"],"zeroDivision":"NaN"},"expected":0.8108108108108109},{"fn":"fbeta2","ds":"mc4s","opts":{"average":"micro","labels":["zzz","cat","ant"],"zeroDivision":0,"sampleWeight":"w"},"expected":0.6256873527101334},{"fn":"fbeta2","ds":"mc4s","opts":{"average":"macro","labels":["zzz","cat","ant"],"zeroDivision":0,"sampleWeight":"w"},"expected":0.3970697379429465},{"fn":"fbeta2","ds":"mc4s","opts":{"average":"weighted","labels":["zzz","cat","ant"],"zeroDivision":0,"sampleWeight":"w"},"expected":0.6104677717745114},{"fn":"fbeta2","ds":"mc4s","opts":{"average":null,"labels":["zzz","cat","ant"],"zeroDivision":0},"expected":[0.0,0.7692307692307693,0.35714285714285715]},{"fn":"fbeta2","ds":"mc4s","opts":{"average":"micro","labels":["zzz","cat","ant"],"zeroDivision":1},"expected":0.5970149253731343},{"fn":"fbeta2","ds":"mc4s","opts":{"average":"macro","labels":["zzz","cat","ant"],"zeroDivision":1},"expected":0.7087912087912088},{"fn":"fbeta2","ds":"mc4s","opts":{"average":"weighted","labels":["zzz","cat","ant"],"zeroDivision":1},"expected":0.5790363482671175},{"fn":"fbeta2","ds":"mc4s","opts":{"average":"micro","labels":["zzz","cat","ant"],"zeroDivision":"NaN"},"expected":0.5970149253731343},{"fn":"fbeta2","ds":"mc4s","opts":{"average":"micro","labels":["zzz","cat","ant"],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.6256873527101334},{"fn":"fbeta2","ds":"mc4s","opts":{"average":"macro","labels":["zzz","cat","ant"],"zeroDivision":"NaN"},"expected":0.5631868131868132},{"fn":"fbeta2","ds":"mc4s","opts":{"average":"macro","labels":["zzz","cat","ant"],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.5956046069144197},{"fn":"fbeta2","ds":"mc4s","opts":{"average":"weighted","labels":["zzz","cat","ant"],"zeroDivision":"NaN"},"expected":0.5790363482671175},{"fn":"fbeta2","ds":"mc4s","opts":{"average":"weighted","labels":["zzz","cat","ant"],"zeroDivision":"NaN","sampleWeight":"w"},"expected":0.6104677717745114},{"fn":"fbeta2","ds":"mc4s","opts":{"average":null,"labels":["zzz","cat","ant"],"zeroDivision":"NaN"},"expected":["NaN",0.7692307692307693,0.35714285714285715]},{"fn":"fbeta2","ds":"mc4s","opts":{"average":null,"labels":["zzz","cat","ant"],"zeroDivision":"NaN","sampleWeight":"w"},"expected":["NaN",0.7880434782608694,0.40316573556797014]},{"fn":"jaccard","ds":"bin","opts":{"average":"binary","zeroDivision":0},"expected":0.5333333333333333},{"fn":"jaccard","ds":"bin","opts":{"average":"binary","zeroDivision":0,"sampleWeight":"w"},"expected":0.5400743997294556},{"fn":"jaccard","ds":"bin","opts":{"average":"micro","zeroDivision":0},"expected":0.5483870967741935},{"fn":"jaccard","ds":"bin","opts":{"average":"micro","zeroDivision":0,"sampleWeight":"w"},"expected":0.5546823837590047},{"fn":"jaccard","ds":"bin","opts":{"average":"macro","zeroDivision":0},"expected":0.5479166666666666},{"fn":"jaccard","ds":"bin","opts":{"average":"macro","zeroDivision":0,"sampleWeight":"w"},"expected":0.5542326933588566},{"fn":"jaccard","ds":"bin","opts":{"average":"weighted","zeroDivision":0},"expected":0.5491319444444445},{"fn":"jaccard","ds":"bin","opts":{"average":"weighted","zeroDivision":0,"sampleWeight":"w"},"expected":0.5558727652830712},{"fn":"jaccard","ds":"bin","opts":{"average":null,"zeroDivision":0},"expected":[0.5625,0.5333333333333333]},{"fn":"jaccard","ds":"bin","opts":{"average":null,"zeroDivision":0,"sampleWeight":"w"},"expected":[0.5683909869882577,0.5400743997294556]},{"fn":"jaccard","ds":"bin","opts":{"average":"binary","zeroDivision":0},"expected":0.0,"yPredOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"fn":"jaccard","ds":"bin","opts":{"average":"binary","zeroDivision":0},"expected":0.0,"yPredOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"yTrueOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"fn":"jaccard","ds":"bin","opts":{"average":"binary","zeroDivision":1},"expected":0.0,"yPredOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"fn":"jaccard","ds":"bin","opts":{"average":"binary","zeroDivision":1},"expected":1.0,"yPredOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"yTrueOverride":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"fn":"jaccard","ds":"mc3","opts":{"average":"micro","zeroDivision":0},"expected":0.7647058823529411},{"fn":"jaccard","ds":"mc3","opts":{"average":"micro","zeroDivision":0,"sampleWeight":"w"},"expected":0.8289812431040824},{"fn":"jaccard","ds":"mc3","opts":{"average":"macro","zeroDivision":0},"expected":0.7655011655011655},{"fn":"jaccard","ds":"mc3","opts":{"average":"macro","zeroDivision":0,"sampleWeight":"w"},"expected":0.8308689884540744},{"fn":"jaccard","ds":"mc3","opts":{"average":"weighted","zeroDivision":0},"expected":0.7682983682983683},{"fn":"jaccard","ds":"mc3","opts":{"average":"weighted","zeroDivision":0,"sampleWeight":"w"},"expected":0.8362505001043239},{"fn":"jaccard","ds":"mc3","opts":{"average":null,"zeroDivision":0},"expected":[0.7692307692307693,0.7272727272727273,0.8]},{"fn":"jaccard","ds":"mc3","opts":{"average":null,"zeroDivision":0,"sampleWeight":"w"},"expected":[0.8226154510142426,0.7414664981036663,0.9285250162443145]},{"fn":"jaccard","ds":"mc3","opts":{"average":"micro","labels":[2,0],"zeroDivision":0},"expected":0.782608695652174},{"fn":"jaccard","ds":"mc3","opts":{"average":"macro","labels":[2,0],"zeroDivision":0},"expected":0.7846153846153847},{"fn":"jaccard","ds":"mc3","opts":{"average":"weighted","labels":[2,0],"zeroDivision":0},"expected":0.7832167832167833},{"fn":"jaccard","ds":"mc3","opts":{"average":null,"labels":[2,0],"zeroDivision":1,"sampleWeight":"w"},"expected":[0.9285250162443145,0.8226154510142426]},{"fn":"jaccard","ds":"mc3","opts":{"average":null,"labels":[1,99],"zeroDivision":0,"sampleWeight":"w"},"expected":[0.7414664981036663,0.0]},{"fn":"jaccard","ds":"mc3","opts":{"average":"micro","labels":[1,99],"zeroDivision":1,"sampleWeight":"w"},"expected":0.7414664981036663},{"fn":"jaccard","ds":"mc3","opts":{"average":"macro","labels":[1,99],"zeroDivision":1,"sampleWeight":"w"},"expected":0.8707332490518331},{"fn":"jaccard","ds":"mc3","opts":{"average":"weighted","labels":[1,99],"zeroDivision":1,"sampleWeight":"w"},"expected":0.7414664981036663},{"fn":"jaccard","ds":"mc3","opts":{"average":null,"labels":[1,99],"zeroDivision":1},"expected":[0.7272727272727273,1.0]},{"fn":"jaccard","ds":"mc3","opts":{"average":"micro","labels":[99,2,0],"zeroDivision":0,"sampleWeight":"w"},"expected":0.8648858921161825},{"fn":"jaccard","ds":"mc3","opts":{"average":"macro","labels":[99,2,0],"zeroDivision":0,"sampleWeight":"w"},"expected":0.5837134890861857},{"fn":"jaccard","ds":"mc3","opts":{"average":"weighted","labels":[99,2,0],"zeroDivision":0,"sampleWeight":"w"},"expected":0.8655088249324216},{"fn":"jaccard","ds":"mc3","opts":{"average":null,"labels":[99,2,0],"zeroDivision":0},"expected":[0.0,0.8,0.7692307692307693]},{"fn":"jaccard","ds":"mc3","opts":{"average":"micro","labels":[99,2,0],"zeroDivision":1},"expected":0.782608695652174},{"fn":"jaccard","ds":"mc3","opts":{"average":"macro","labels":[99,2,0],"zeroDivision":1},"expected":0.8564102564102565},{"fn":"jaccard","ds":"mc3","opts":{"average":"weighted","labels":[99,2,0],"zeroDivision":1},"expected":0.7832167832167833},{"fn":"jaccard","ds":"mc4s","opts":{"average":"micro","zeroDivision":0},"expected":0.5652173913043478},{"fn":"jaccard","ds":"mc4s","opts":{"average":"micro","zeroDivision":0,"sampleWeight":"w"},"expected":0.5703125},{"fn":"jaccard","ds":"mc4s","opts":{"average":"macro","zeroDivision":0},"expected":0.5306372549019608},{"fn":"jaccard","ds":"mc4s","opts":{"average":"macro","zeroDivision":0,"sampleWeight":"w"},"expected":0.5608140582324379},{"fn":"jaccard","ds":"mc4s","opts":{"average":"weighted","zeroDivision":0},"expected":0.5811546840958606},{"fn":"jaccard","ds":"mc4s","opts":{"average":"weighted","zeroDivision":0,"sampleWeight":"w"},"expected":0.5874923741333818},{"fn":"jaccard","ds":"mc4s","opts":{"average":null,"zeroDivision":0},"expected":[0.25,0.7058823529411765,0.5,0.6666666666666666]},{"fn":"jaccard","ds":"mc4s","opts":{"average":null,"zeroDivision":0,"sampleWeight":"w"},"expected":[0.2826370757180156,0.6730169726359543,0.5213483146067414,0.7662538699690403]},{"fn":"jaccard","ds":"mc4s","opts":{"average":"micro","labels":["cat","ant"],"zeroDivision":0},"expected":0.4},{"fn":"jaccard","ds":"mc4s","opts":{"average":"macro","labels":["cat","ant"],"zeroDivision":0},"expected":0.375},{"fn":"jaccard","ds":"mc4s","opts":{"average":"weighted","labels":["cat","ant"],"zeroDivision":0},"expected":0.38461538461538464},{"fn":"jaccard","ds":"mc4s","opts":{"average":null,"labels":["cat","ant"],"zeroDivision":1,"sampleWeight":"w"},"expected":[0.5213483146067414,0.2826370757180156]},{"fn":"jaccard","ds":"mc4s","opts":{"average":null,"labels":["bee","zzz"],"zeroDivision":0,"sampleWeight":"w"},"expected":[0.6730169726359543,0.0]},{"fn":"jaccard","ds":"mc4s","opts":{"average":"micro","labels":["bee","zzz"],"zeroDivision":1,"sampleWeight":"w"},"expected":0.6730169726359543},{"fn":"jaccard","ds":"mc4s","opts":{"average":"macro","labels":["bee","zzz"],"zeroDivision":1,"sampleWeight":"w"},"expected":0.8365084863179771},{"fn":"jaccard","ds":"mc4s","opts":{"average":"weighted","labels":["bee","zzz"],"zeroDivision":1,"sampleWeight":"w"},"expected":0.6730169726359543},{"fn":"jaccard","ds":"mc4s","opts":{"average":null,"labels":["bee","zzz"],"zeroDivision":1},"expected":[0.7058823529411765,1.0]},{"fn":"jaccard","ds":"mc4s","opts":{"average":"micro","labels":["zzz","cat","ant"],"zeroDivision":0,"sampleWeight":"w"},"expected":0.4240085174341228},{"fn":"jaccard","ds":"mc4s","opts":{"average":"macro","labels":["zzz","cat","ant"],"zeroDivision":0,"sampleWeight":"w"},"expected":0.2679951301082523},{"fn":"jaccard","ds":"mc4s","opts":{"average":"weighted","labels":["zzz","cat","ant"],"zeroDivision":0,"sampleWeight":"w"},"expected":0.4112112186153984},{"fn":"jaccard","ds":"mc4s","opts":{"average":null,"labels":["zzz","cat","ant"],"zeroDivision":0},"expected":[0.0,0.5,0.25]},{"fn":"jaccard","ds":"mc4s","opts":{"average":"micro","labels":["zzz","cat","ant"],"zeroDivision":1},"expected":0.4},{"fn":"jaccard","ds":"mc4s","opts":{"average":"macro","labels":["zzz","cat","ant"],"zeroDivision":1},"expected":0.5833333333333334},{"fn":"jaccard","ds":"mc4s","opts":{"average":"weighted","labels":["zzz","cat","ant"],"zeroDivision":1},"expected":0.38461538461538464},{"fn":"rocAucScore","ds":"mc3","opts":{"multiClass":"ovr","average":"macro"},"expected":1.0},{"fn":"rocAucScore","ds":"mc3","opts":{"multiClass":"ovr","average":"macro","sampleWeight":"w"},"expected":1.0},{"fn":"rocAucScore","ds":"mc3","opts":{"multiClass":"ovr","average":"weighted"},"expected":1.0},{"fn":"rocAucScore","ds":"mc3","opts":{"multiClass":"ovr","average":"weighted","sampleWeight":"w"},"expected":1.0},{"fn":"rocAucScore","ds":"mc3","opts":{"multiClass":"ovr","average":"micro"},"expected":1.0},{"fn":"rocAucScore","ds":"mc3","opts":{"multiClass":"ovr","average":"micro","sampleWeight":"w"},"expected":1.0},{"fn":"rocAucScore","ds":"mc3","opts":{"multiClass":"ovr","average":null},"expected":[1.0,1.0,1.0]},{"fn":"rocAucScore","ds":"mc3","opts":{"multiClass":"ovr","average":null,"sampleWeight":"w"},"expected":[1.0,1.0,1.0]},{"fn":"rocAucScore","ds":"mc3","opts":{"multiClass":"ovo","average":"macro"},"expected":1.0},{"fn":"rocAucScore","ds":"mc3","opts":{"multiClass":"ovo","average":"weighted"},"expected":1.0},{"fn":"rocAucScore","ds":"mc3","opts":{"multiClass":"ovr","labels":[0,1,2]},"expected":1.0},{"fn":"rocAucScore","ds":"mc4s","opts":{"multiClass":"ovr","average":"macro"},"expected":1.0},{"fn":"rocAucScore","ds":"mc4s","opts":{"multiClass":"ovr","average":"macro","sampleWeight":"w"},"expected":1.0},{"fn":"rocAucScore","ds":"mc4s","opts":{"multiClass":"ovr","average":"weighted"},"expected":1.0},{"fn":"rocAucScore","ds":"mc4s","opts":{"multiClass":"ovr","average":"weighted","sampleWeight":"w"},"expected":1.0},{"fn":"rocAucScore","ds":"mc4s","opts":{"multiClass":"ovr","average":"micro"},"expected":1.0},{"fn":"rocAucScore","ds":"mc4s","opts":{"multiClass":"ovr","average":"micro","sampleWeight":"w"},"expected":1.0},{"fn":"rocAucScore","ds":"mc4s","opts":{"multiClass":"ovr","average":null},"expected":[1.0,1.0,1.0,1.0]},{"fn":"rocAucScore","ds":"mc4s","opts":{"multiClass":"ovr","average":null,"sampleWeight":"w"},"expected":[1.0,1.0,1.0,1.0]},{"fn":"rocAucScore","ds":"mc4s","opts":{"multiClass":"ovo","average":"macro"},"expected":1.0},{"fn":"rocAucScore","ds":"mc4s","opts":{"multiClass":"ovo","average":"weighted"},"expected":1.0},{"fn":"rocAucScore","ds":"mc4s","opts":{"multiClass":"ovr","labels":["ant","bee","cat","dog"]},"expected":1.0},{"fn":"rocAucScore","ds":"mcSparse","opts":{"multiClass":"ovr","average":"macro"},"expected":1.0},{"fn":"rocAucScore","ds":"mcSparse","opts":{"multiClass":"ovr","average":"macro","sampleWeight":"w"},"expected":1.0},{"fn":"rocAucScore","ds":"mcSparse","opts":{"multiClass":"ovr","average":"weighted"},"expected":1.0},{"fn":"rocAucScore","ds":"mcSparse","opts":{"multiClass":"ovr","average":"weighted","sampleWeight":"w"},"expected":1.0},{"fn":"rocAucScore","ds":"mcSparse","opts":{"multiClass":"ovr","average":"micro"},"expected":1.0},{"fn":"rocAucScore","ds":"mcSparse","opts":{"multiClass":"ovr","average":"micro","sampleWeight":"w"},"expected":1.0},{"fn":"rocAucScore","ds":"mcSparse","opts":{"multiClass":"ovr","average":null},"expected":[1.0,1.0,1.0]},{"fn":"rocAucScore","ds":"mcSparse","opts":{"multiClass":"ovr","average":null,"sampleWeight":"w"},"expected":[1.0,1.0,1.0]},{"fn":"rocAucScore","ds":"mcSparse","opts":{"multiClass":"ovo","average":"macro"},"expected":1.0},{"fn":"rocAucScore","ds":"mcSparse","opts":{"multiClass":"ovo","average":"weighted"},"expected":1.0},{"fn":"rocAucScore","ds":"mcSparse","opts":{"multiClass":"ovr","labels":[2,5,9]},"expected":1.0},{"fn":"rocAucScore","ds":"bin","opts":{},"expected":0.7552447552447553,"score":true},{"fn":"rocAucScore","ds":"bin","opts":{"sampleWeight":"w"},"expected":0.7794167925153019,"score":true},{"fn":"logLoss","ds":"mc3","opts":{"normalize":true},"expected":0.5087393028300523},{"fn":"logLoss","ds":"mc3","opts":{"normalize":false},"expected":15.26217908490157},{"fn":"logLoss","ds":"mc3","opts":{"normalize":true,"sampleWeight":"w"},"expected":0.5012203574182638},{"fn":"logLoss","ds":"mc3","opts":{"normalize":false,"sampleWeight":"w"},"expected":24.925688374410257},{"fn":"logLoss","ds":"mc3","opts":{"labels":[0,1,2]},"expected":0.5049047781437356,"subset":[1,2,4,5,6,8,9,10,12,15,16,17,18,19,20,22,23,24,26,28]},{"fn":"logLoss","ds":"mc3","opts":{"labels":[0,1,2],"sampleWeight":"w"},"expected":0.4951941233438824,"subset":[1,2,4,5,6,8,9,10,12,15,16,17,18,19,20,22,23,24,26,28],"weightSubset":true},{"fn":"logLoss","ds":"mc4s","opts":{"normalize":true},"expected":0.592013510409283},{"fn":"logLoss","ds":"mc4s","opts":{"normalize":false},"expected":21.312486374734185},{"fn":"logLoss","ds":"mc4s","opts":{"normalize":true,"sampleWeight":"w"},"expected":0.5941090792498323},{"fn":"logLoss","ds":"mc4s","opts":{"normalize":false,"sampleWeight":"w"},"expected":37.01893672805706},{"fn":"logLoss","ds":"mc4s","opts":{"labels":["ant","bee","cat","dog"]},"expected":0.603532206901804,"subset":[0,1,2,4,5,7,9,12,13,14,15,16,17,18,20,21,22,24,25,26,27,28,29,30,31,33,34,35]},{"fn":"logLoss","ds":"mc4s","opts":{"labels":["ant","bee","cat","dog"],"sampleWeight":"w"},"expected":0.6032047545126444,"subset":[0,1,2,4,5,7,9,12,13,14,15,16,17,18,20,21,22,24,25,26,27,28,29,30,31,33,34,35],"weightSubset":true},{"fn":"logLoss","ds":"mcSparse","opts":{"normalize":true},"expected":0.5289119313097979},{"fn":"logLoss","ds":"mcSparse","opts":{"normalize":false},"expected":14.280622145364541},{"fn":"logLoss","ds":"mcSparse","opts":{"normalize":true,"sampleWeight":"w"},"expected":0.5268541258031071},{"fn":"logLoss","ds":"mcSparse","opts":{"normalize":false,"sampleWeight":"w"},"expected":22.72848698714605},{"fn":"logLoss","ds":"mcSparse","opts":{"labels":[2,5,9]},"expected":0.5339202721986591,"subset":[1,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26]},{"fn":"logLoss","ds":"mcSparse","opts":{"labels":[2,5,9],"sampleWeight":"w"},"expected":0.532822538950357,"subset":[1,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26],"weightSubset":true},{"fn":"logLoss","ds":"bin","opts":{},"expected":0.5910488578454466,"prob":true},{"fn":"logLoss","ds":"bin","opts":{"labels":[0,1]},"expected":0.5910488578454466,"prob":true},{"fn":"logLoss","ds":"bin","opts":{"sampleWeight":"w"},"expected":0.5621144079797792,"prob":true},{"fn":"logLoss","ds":"bin","opts":{"sampleWeight":"w","labels":[0,1]},"expected":0.5621144079797792,"prob":true},{"fn":"matthewsCorrcoef","ds":"mc3","opts":{},"expected":0.8094446585163529},{"fn":"matthewsCorrcoef","ds":"mc3","opts":{"sampleWeight":"w"},"expected":0.8648510940128149},{"fn":"accuracy","ds":"mc3","opts":{},"expected":0.8666666666666667},{"fn":"accuracy","ds":"mc3","opts":{"sampleWeight":"w"},"expected":0.9064950733963403},{"fn":"confusionMatrix","ds":"mc3","opts":{},"expected":[[10.0,2.0,0.0],[0.0,8.0,0.0],[1.0,1.0,8.0]]},{"fn":"confusionMatrix","ds":"mc3","opts":{"sampleWeight":"w"},"expected":[[19.06,3.5500000000000003,0.0],[0.0,11.73,0.0],[0.56,0.54,14.29]]},{"fn":"confusionMatrix","ds":"mc3","opts":{"normalize":"true"},"expected":[[0.8333333333333334,0.16666666666666666,0.0],[0.0,1.0,0.0],[0.1,0.1,0.8]]},{"fn":"confusionMatrix","ds":"mc3","opts":{"sampleWeight":"w","normalize":"true"},"expected":[[0.8429898275099513,0.15701017249004867,0.0],[0.0,1.0,0.0],[0.0363872644574399,0.03508771929824562,0.9285250162443145]]},{"fn":"confusionMatrix","ds":"mc3","opts":{"normalize":"pred"},"expected":[[0.9090909090909091,0.18181818181818182,0.0],[0.0,0.7272727272727273,0.0],[0.09090909090909091,0.09090909090909091,1.0]]},{"fn":"confusionMatrix","ds":"mc3","opts":{"sampleWeight":"w","normalize":"pred"},"expected":[[0.9714576962283384,0.22439949431099876,0.0],[0.0,0.7414664981036663,0.0],[0.028542303771661576,0.03413400758533502,1.0]]},{"fn":"confusionMatrix","ds":"mc3","opts":{"normalize":"all"},"expected":[[0.3333333333333333,0.06666666666666667,0.0],[0.0,0.26666666666666666,0.0],[0.03333333333333333,0.03333333333333333,0.26666666666666666]]},{"fn":"confusionMatrix","ds":"mc3","opts":{"sampleWeight":"w","normalize":"all"},"expected":[[0.38326965614317315,0.07138548160064348,0.0],[0.0,0.23587371807761917,0.0],[0.01126080836517193,0.01085863663784436,0.287351699175548]]},{"fn":"matthewsCorrcoef","ds":"mc4s","opts":{},"expected":0.6198315921657315},{"fn":"matthewsCorrcoef","ds":"mc4s","opts":{"sampleWeight":"w"},"expected":0.6296749358310231},{"fn":"accuracy","ds":"mc4s","opts":{},"expected":0.7222222222222222},{"fn":"accuracy","ds":"mc4s","opts":{"sampleWeight":"w"},"expected":0.7263681592039801},{"fn":"confusionMatrix","ds":"mc4s","opts":{},"expected":[[2.0,0.0,3.0,1.0],[2.0,12.0,1.0,0.0],[0.0,1.0,6.0,0.0],[0.0,1.0,1.0,6.0]]},{"fn":"confusionMatrix","ds":"mc4s","opts":{"sampleWeight":"w"},"expected":[[4.33,0.0,6.02,1.0],[3.97,19.43,2.39,0.0],[0.0,1.65,11.599999999999998,0.0],[0.0,1.43,0.59,9.9]]},{"fn":"confusionMatrix","ds":"mc4s","opts":{"normalize":"true"},"expected":[[0.3333333333333333,0.0,0.5,0.16666666666666666],[0.13333333333333333,0.8,0.06666666666666667,0.0],[0.0,0.14285714285714285,0.8571428571428571,0.0],[0.0,0.125,0.125,0.75]]},{"fn":"confusionMatrix","ds":"mc4s","opts":{"sampleWeight":"w","normalize":"true"},"expected":[[0.3814977973568282,0.0,0.5303964757709251,0.0881057268722467],[0.15393563396665375,0.7533927879022877,0.09267157813105856,0.0],[0.0,0.12452830188679247,0.8754716981132075,0.0],[0.0,0.11996644295302013,0.04949664429530201,0.8305369127516778]]},{"fn":"confusionMatrix","ds":"mc4s","opts":{"normalize":"pred"},"expected":[[0.5,0.0,0.2727272727272727,0.14285714285714285],[0.5,0.8571428571428571,0.09090909090909091,0.0],[0.0,0.07142857142857142,0.5454545454545454,0.0],[0.0,0.07142857142857142,0.09090909090909091,0.8571428571428571]]},{"fn":"confusionMatrix","ds":"mc4s","opts":{"sampleWeight":"w","normalize":"pred"},"expected":[[0.5216867469879518,0.0,0.2922330097087379,0.09174311926605504],[0.4783132530120482,0.8631719235895159,0.11601941747572818,0.0],[0.0,0.07330075521990227,0.5631067961165048,0.0],[0.0,0.06352732119058196,0.028640776699029126,0.908256880733945]]},{"fn":"confusionMatrix","ds":"mc4s","opts":{"normalize":"all"},"expected":[[0.05555555555555555,0.0,0.08333333333333333,0.027777777777777776],[0.05555555555555555,0.3333333333333333,0.027777777777777776,0.0],[0.0,0.027777777777777776,0.16666666666666666,0.0],[0.0,0.027777777777777776,0.027777777777777776,0.16666666666666666]]},{"fn":"confusionMatrix","ds":"mc4s","opts":{"sampleWeight":"w","normalize":"all"},"expected":[[0.06949125341036752,0.0,0.09661370566522226,0.016048788316482106],[0.06371368961643396,0.3118279569892473,0.038356604076392235,0.0],[0.0,0.02648050072219547,0.1861659444711924,0.0],[0.0,0.02294976729256941,0.009468785106724442,0.15888300433317284]]},{"fn":"matthewsCorrcoef","ds":"mcSparse","opts":{},"expected":0.7305161505085608},{"fn":"matthewsCorrcoef","ds":"mcSparse","opts":{"sampleWeight":"w"},"expected":0.7003746898210392},{"fn":"accuracy","ds":"mcSparse","opts":{},"expected":0.8148148148148148},{"fn":"accuracy","ds":"mcSparse","opts":{"sampleWeight":"w"},"expected":0.7927677329624477},{"fn":"confusionMatrix","ds":"mcSparse","opts":{},"expected":[[9.0,0.0,1.0],[3.0,9.0,1.0],[0.0,0.0,4.0]]},{"fn":"confusionMatrix","ds":"mcSparse","opts":{"sampleWeight":"w"},"expected":[[13.84,0.0,2.2],[4.949999999999999,13.889999999999999,1.79],[0.0,0.0,6.47]]},{"fn":"confusionMatrix","ds":"mcSparse","opts":{"normalize":"true"},"expected":[[0.9,0.0,0.1],[0.23076923076923078,0.6923076923076923,0.07692307692307693],[0.0,0.0,1.0]]},{"fn":"confusionMatrix","ds":"mcSparse","opts":{"sampleWeight":"w","normalize":"true"},"expected":[[0.8628428927680798,0.0,0.1371571072319202],[0.2399418322830829,0.67329132331556,0.08676684440135726],[0.0,0.0,1.0]]},{"fn":"confusionMatrix","ds":"mcSparse","opts":{"normalize":"pred"},"expected":[[0.75,0.0,0.16666666666666666],[0.25,1.0,0.16666666666666666],[0.0,0.0,0.6666666666666666]]},{"fn":"confusionMatrix","ds":"mcSparse","opts":{"sampleWeight":"w","normalize":"pred"},"expected":[[0.736562001064396,0.0,0.21032504780114722],[0.26343799893560405,1.0,0.17112810707456977],[0.0,0.0,0.6185468451242829]]},{"fn":"confusionMatrix","ds":"mcSparse","opts":{"normalize":"all"},"expected":[[0.3333333333333333,0.0,0.037037037037037035],[0.1111111111111111,0.3333333333333333,0.037037037037037035],[0.0,0.0,0.14814814814814814]]},{"fn":"confusionMatrix","ds":"mcSparse","opts":{"sampleWeight":"w","normalize":"all"},"expected":[[0.32081594807603153,0.0,0.05099675475197033],[0.11474269819193322,0.3219749652294854,0.041492814093648585],[0.0,0.0,0.1499768196569309]]},{"fn":"matthewsCorrcoef","ds":"bin","opts":{},"expected":0.4181210050035454},{"fn":"matthewsCorrcoef","ds":"bin","opts":{"sampleWeight":"w"},"expected":0.43425406819019313},{"fn":"accuracy","ds":"bin","opts":{},"expected":0.7083333333333334},{"fn":"accuracy","ds":"bin","opts":{"sampleWeight":"w"},"expected":0.7135636057287277},{"fn":"confusionMatrix","ds":"bin","opts":{},"expected":[[9.0,4.0],[3.0,8.0]]},{"fn":"confusionMatrix","ds":"bin","opts":{"sampleWeight":"w"},"expected":[[17.91,8.58],[5.0200000000000005,15.97]]},{"fn":"confusionMatrix","ds":"bin","opts":{"normalize":"true"},"expected":[[0.6923076923076923,0.3076923076923077],[0.2727272727272727,0.7272727272727273]]},{"fn":"confusionMatrix","ds":"bin","opts":{"sampleWeight":"w","normalize":"true"},"expected":[[0.6761041902604756,0.3238958097395243],[0.23916150547879944,0.7608384945212006]]},{"fn":"confusionMatrix","ds":"bin","opts":{"normalize":"pred"},"expected":[[0.75,0.3333333333333333],[0.25,0.6666666666666666]]},{"fn":"confusionMatrix","ds":"bin","opts":{"sampleWeight":"w","normalize":"pred"},"expected":[[0.781072830353249,0.3494908350305499],[0.218927169646751,0.6505091649694501]]},{"fn":"confusionMatrix","ds":"bin","opts":{"normalize":"all"},"expected":[[0.375,0.16666666666666666],[0.125,0.3333333333333333]]},{"fn":"confusionMatrix","ds":"bin","opts":{"sampleWeight":"w","normalize":"all"},"expected":[[0.37721145745577084,0.18070766638584665],[0.10572872788542545,0.336352148272957]]},{"fn":"mse","ds":"reg","opts":{},"expected":4.0940176},{"fn":"rmse","ds":"reg","opts":{},"expected":2.0233678854820245},{"fn":"mae","ds":"reg","opts":{},"expected":1.5376},{"fn":"r2Score","ds":"reg","opts":{},"expected":0.2774119797059891},{"fn":"explainedVarianceScore","ds":"reg","opts":{},"expected":0.3789019589764152},{"fn":"meanAbsolutePercentageError","ds":"reg","opts":{},"expected":0.403474137376752},{"fn":"mse","ds":"reg","opts":{"sampleWeight":"w"},"expected":5.75517087917223},{"fn":"rmse","ds":"reg","opts":{"sampleWeight":"w"},"expected":2.398993722203589},{"fn":"mae","ds":"reg","opts":{"sampleWeight":"w"},"expected":1.8326255006675571},{"fn":"r2Score","ds":"reg","opts":{"sampleWeight":"w"},"expected":-0.274625201904521},{"fn":"explainedVarianceScore","ds":"reg","opts":{"sampleWeight":"w"},"expected":0.011480438446666463},{"fn":"meanAbsolutePercentageError","ds":"reg","opts":{"sampleWeight":"w"},"expected":0.4852084133731338},{"fn":"mse","ds":"regPos","opts":{},"expected":6.237585100000001},{"fn":"rmse","ds":"regPos","opts":{},"expected":2.497515785735898},{"fn":"mae","ds":"regPos","opts":{},"expected":2.0246999999999997},{"fn":"r2Score","ds":"regPos","opts":{},"expected":0.30135191720316956},{"fn":"explainedVarianceScore","ds":"regPos","opts":{},"expected":0.305391085205591},{"fn":"meanAbsolutePercentageError","ds":"regPos","opts":{},"expected":6.829302640192476},{"fn":"meanSquaredLogError","ds":"regPos","opts":{},"expected":0.43279738920434785},{"fn":"mse","ds":"regPos","opts":{"sampleWeight":"w"},"expected":6.036711271666666},{"fn":"rmse","ds":"regPos","opts":{"sampleWeight":"w"},"expected":2.4569719720962766},{"fn":"mae","ds":"regPos","opts":{"sampleWeight":"w"},"expected":2.0457769999999997},{"fn":"r2Score","ds":"regPos","opts":{"sampleWeight":"w"},"expected":0.20677553353999512},{"fn":"explainedVarianceScore","ds":"regPos","opts":{"sampleWeight":"w"},"expected":0.20883946003040155},{"fn":"meanAbsolutePercentageError","ds":"regPos","opts":{"sampleWeight":"w"},"expected":7.370597940705086},{"fn":"meanSquaredLogError","ds":"regPos","opts":{"sampleWeight":"w"},"expected":0.4669740656783984}]}'
) as { datasets: Record<string, Dataset>; cases: Case[] };

const f64 = { dtype: "float64" } as const;

function labelsTensor(values: ReadonlyArray<number | string>): Tensor {
  return typeof values[0] === "string"
    ? tensor(values as string[])
    : tensor(values as number[], f64);
}

function decode(value: Expected): number | number[] | number[][] {
  if (Array.isArray(value)) return value.map(decode) as number[] | number[][];
  if (value === "NaN") return Number.NaN;
  if (value === "Infinity") return Number.POSITIVE_INFINITY;
  if (value === "-Infinity") return Number.NEGATIVE_INFINITY;
  return value as number;
}

function expectClose(actual: unknown, expected: unknown, label: string): void {
  if (Array.isArray(expected)) {
    expect(Array.isArray(actual), label).toBe(true);
    const a = actual as unknown[];
    expect(a.length, label).toBe(expected.length);
    expected.forEach((e, i) => {
      expectClose(a[i], e, `${label}[${i}]`);
    });
    return;
  }
  const e = expected as number;
  const a = actual as number;
  if (Number.isNaN(e)) {
    expect(Number.isNaN(a), `${label}: expected NaN, got ${a}`).toBe(true);
    return;
  }
  const tolerance = 1e-12 + 1e-9 * Math.abs(e);
  expect(Math.abs(a - e) <= tolerance, `${label}: got ${a}, expected ${e}`).toBe(true);
}

function tensorToRows(t: Tensor): number[][] {
  const [rows, cols] = t.shape as [number, number];
  const data = t.data as Float64Array;
  return Array.from({ length: rows }, (_, r) =>
    Array.from({ length: cols }, (_, c) => data[t.offset + r * cols + c] as number)
  );
}

function runCase(c: Case): unknown {
  const d = FIXTURES.datasets[c.ds] as Dataset;
  let yTrueValues = c.yTrueOverride ?? d.yTrue;
  let yPredValues = c.yPredOverride ?? d.yPred;
  let weights = d.w;
  let proba = d.proba;
  if (c.subset !== undefined) {
    const pick = <T>(values: T[]): T[] => (c.subset as number[]).map((i) => values[i] as T);
    yTrueValues = pick(yTrueValues);
    yPredValues = pick(yPredValues);
    if (proba !== undefined) proba = pick(proba);
    if (c.weightSubset === true) weights = pick(weights);
  }
  const opts: Record<string, unknown> = { ...c.opts };
  if (opts.sampleWeight === "w") opts.sampleWeight = weights;
  if (opts.zeroDivision === "NaN") opts.zeroDivision = Number.NaN;
  if (opts.average === undefined) delete opts.average;

  const yTrue = labelsTensor(yTrueValues);
  const yPred = labelsTensor(yPredValues);
  const asOptions = opts as AveragedMetricOptions;
  switch (c.fn) {
    case "precision":
      return precision(yTrue, yPred, asOptions);
    case "recall":
      return recall(yTrue, yPred, asOptions);
    case "f1Score":
      return f1Score(yTrue, yPred, asOptions);
    case "fbeta05":
      return fbetaScore(yTrue, yPred, 0.5, asOptions);
    case "fbeta2":
      return fbetaScore(yTrue, yPred, 2, asOptions);
    case "jaccard":
      return jaccardScore(yTrue, yPred, asOptions);
    case "accuracy":
      return accuracy(yTrue, yPred, opts);
    case "matthewsCorrcoef":
      return matthewsCorrcoef(yTrue, yPred, opts);
    case "confusionMatrix":
      return tensorToRows(confusionMatrix(yTrue, yPred, opts));
    case "rocAucScore": {
      const scores =
        c.score === true ? tensor(d.score as number[], f64) : tensor(proba as number[][], f64);
      return rocAucScore(yTrue, scores, opts);
    }
    case "logLoss": {
      const probs =
        c.prob === true ? tensor(d.prob as number[], f64) : tensor(proba as number[][], f64);
      return logLoss(yTrue, probs, opts);
    }
    default: {
      const regTrue = tensor(d.yTrue as number[], f64);
      const regPred = tensor(d.yPred as number[], f64);
      const fn = {
        mse,
        rmse,
        mae,
        r2Score,
        explainedVarianceScore,
        meanAbsolutePercentageError,
        meanSquaredLogError,
      }[c.fn];
      if (fn === undefined) throw new Error(`unknown function ${c.fn}`);
      return fn(regTrue, regPred, opts);
    }
  }
}

describe("v1.5.0 wave 3 metrics: scikit-learn parity on random data", () => {
  const groups = new Map<string, Case[]>();
  for (const c of FIXTURES.cases) {
    const list = groups.get(c.fn) ?? [];
    list.push(c);
    groups.set(c.fn, list);
  }
  for (const [fn, cases] of groups) {
    it(`${fn}: ${cases.length} cases match scikit-learn`, () => {
      cases.forEach((c, i) => {
        const label = `${fn}#${i} ${c.ds} ${JSON.stringify(c.opts)}`;
        expectClose(runCase(c), decode(c.expected), label);
      });
    });
  }
});

function strided1D(logical: number[], stride: number, offset: number): Tensor {
  const buf = new Float64Array(offset + logical.length * stride + 1).fill(-99);
  logical.forEach((v, i) => {
    buf[offset + i * stride] = v;
  });
  return Tensor.fromTypedArray({
    data: buf,
    shape: [logical.length],
    dtype: "float64",
    device: "cpu",
    offset,
    strides: [stride],
  });
}

/** An [n, k] matrix stored column-major, i.e. a transposed view. */
function columnMajor(rows: number[][]): Tensor {
  const n = rows.length;
  const k = (rows[0] as number[]).length;
  const buf = new Float64Array(n * k);
  rows.forEach((row, i) => {
    row.forEach((v, c) => {
      buf[c * n + i] = v;
    });
  });
  return Tensor.fromTypedArray({
    data: buf,
    shape: [n, k],
    dtype: "float64",
    device: "cpu",
    strides: [1, n],
  });
}

function int64(values: bigint[]): Tensor {
  return tensor(BigInt64Array.from(values));
}

const PROBA = [
  [0.8, 0.1, 0.1],
  [0.2, 0.5, 0.3],
  [0.1, 0.2, 0.7],
  [0.3, 0.3, 0.4],
];
const LABELS = [0, 1, 2, 2];

describe("v1.5.0 wave 3 metrics: multiclass rocAucScore", () => {
  it("matches scikit-learn on a small hand case", () => {
    expect(
      rocAucScore(tensor(LABELS), tensor(PROBA, f64), { multiClass: "ovo", average: "weighted" })
    ).toBeCloseTo(1.0, 12);
    expect(rocAucScore(tensor(LABELS), tensor(PROBA, f64))).toBeCloseTo(
      rocAucScore(tensor(LABELS), tensor(PROBA, f64), { multiClass: "ovr", average: "macro" }),
      15
    );
  });

  it("per-class AUCs: average null on ovr", () => {
    const perClass = rocAucScore(tensor(LABELS), tensor(PROBA, f64), { average: null });
    expect(perClass).toHaveLength(3);
    expect(perClass[0]).toBe(1);
  });

  it("reads strided score matrices and labels like dense ones", () => {
    const dense = rocAucScore(tensor(LABELS), tensor(PROBA, f64), { multiClass: "ovo" });
    const view = rocAucScore(strided1D(LABELS, 3, 2), columnMajor(PROBA), { multiClass: "ovo" });
    expect(view).toBe(dense);
  });

  it("accepts a column-vector score as the binary case and a two column matrix", () => {
    const yTrue = tensor([0, 0, 1, 1]);
    expect(rocAucScore(yTrue, tensor([[0.1], [0.4], [0.35], [0.8]], f64))).toBe(0.75);
    const two = tensor(
      [
        [0.9, 0.1],
        [0.6, 0.4],
        [0.65, 0.35],
        [0.2, 0.8],
      ],
      f64
    );
    expect(rocAucScore(yTrue, two)).toBeCloseTo(0.75, 12);
    expect(rocAucScore(yTrue, tensor([0.1, 0.4, 0.35, 0.8], f64), { average: null })).toEqual([
      0.75,
    ]);
  });

  it("supports int64 labels", () => {
    const a = rocAucScore(int64([0n, 1n, 2n, 2n]), tensor(PROBA, f64));
    expect(a).toBe(rocAucScore(tensor(LABELS), tensor(PROBA, f64)));
  });

  it("returns 0.5 for empty input", () => {
    expect(rocAucScore(tensor([]), tensor([]))).toBe(0.5);
    expect(rocAucScore(tensor([]), tensor([], f64).reshape([0, 3]))).toBe(0.5);
    expect(rocAucScore(tensor([]), tensor([], f64).reshape([0, 3]), { average: null })).toEqual([
      0.5, 0.5, 0.5,
    ]);
  });

  it("validates the score matrix and the options", () => {
    const y = tensor(LABELS);
    const p = tensor(PROBA, f64);
    expect(() => rocAucScore(y, tensor([[0.5, 0.2, 0.1], ...PROBA.slice(1)], f64))).toThrow(
      InvalidParameterError
    );
    expect(() => rocAucScore(y, p, { labels: [0, 1] })).toThrow(/3 columns/);
    expect(() => rocAucScore(y, p, { labels: [2, 1, 0] })).toThrow(/ascending/);
    expect(() => rocAucScore(y, p, { labels: [0, 1, 3] })).toThrow(/not in labels/);
    expect(() => rocAucScore(tensor([0, 1, 1, 1]), p)).toThrow(/distinct classes/);
    expect(() => rocAucScore(y, p, { multiClass: "ovo", average: null })).toThrow(
      InvalidParameterError
    );
    expect(() => rocAucScore(y, p, { multiClass: "ovo", average: "micro" })).toThrow(
      InvalidParameterError
    );
    expect(() => rocAucScore(y, p, { multiClass: "ovo", sampleWeight: [1, 1, 1, 1] })).toThrow(
      /ovo/
    );
    expect(() => rocAucScore(y, p, { multiClass: "bad" as never })).toThrow(/multiClass/);
    expect(() => rocAucScore(y, p, { average: "samples" as never })).toThrow(/average/);
    expect(() => rocAucScore(tensor([0, 1, 2]), p)).toThrow(ShapeError);
    expect(() => rocAucScore(y, tensor([[[0.5]]], f64))).toThrow(ShapeError);
    expect(() => rocAucScore(y, tensor(PROBA.map((r) => r.map(String))))).toThrow(DTypeError);
    expect(() => rocAucScore(y, tensor([[Number.NaN, 0.5, 0.5], ...PROBA.slice(1)], f64))).toThrow(
      DataValidationError
    );
  });

  it("ovr throws for a class without positives or negatives, ovo needs two classes", () => {
    expect(() =>
      rocAucScore(
        tensor([0, 1, 1, 0]),
        tensor(
          [
            [0.5, 0.3, 0.2],
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
            [0.6, 0.2, 0.2],
          ],
          f64
        )
      )
    ).toThrow(InvalidParameterError);
    const p = tensor(
      [
        [0.5, 0.3, 0.2],
        [0.1, 0.8, 0.1],
      ],
      f64
    );
    expect(() => rocAucScore(tensor([1, 1]), p, { labels: [0, 1, 2], multiClass: "ovo" })).toThrow(
      /at least two classes/
    );
  });

  it("weights binary scores like scikit-learn (negative and zero weights)", () => {
    const y = tensor([0, 0, 1, 1, 1]);
    const s = tensor([0.1, 0.4, 0.35, 0.8, 0.4], f64);
    // sklearn: roc_auc_score(y, s, sample_weight=[1, 2, 3, 4, 0]) = 0.7142857142857143
    expect(rocAucScore(y, s, { sampleWeight: [1, 2, 3, 4, 0] })).toBeCloseTo(
      0.7142857142857143,
      14
    );
    expect(rocAucScore(y, s)).toBe(0.75);
    expect(() => rocAucScore(y, s, { sampleWeight: [1, 2] })).toThrow(ShapeError);
  });
});

describe("v1.5.0 wave 3 metrics: multiclass logLoss", () => {
  it("2-D probabilities, string labels and the labels option", () => {
    const y = tensor(["b", "a", "c"]);
    const p = tensor(
      [
        [0.1, 0.8, 0.1],
        [0.7, 0.2, 0.1],
        [0.2, 0.2, 0.6],
      ],
      f64
    );
    // sklearn: log_loss(["b","a","c"], p) = -(ln .8 + ln .7 + ln .6) / 3
    const expected = -(Math.log(0.8) + Math.log(0.7) + Math.log(0.6)) / 3;
    expect(logLoss(y, p)).toBeCloseTo(expected, 12);
    expect(logLoss(y, p, { labels: ["a", "b", "c"] })).toBeCloseTo(expected, 12);
    expect(logLoss(y, p, { normalize: false })).toBeCloseTo(expected * 3, 12);
  });

  it("int64 labels and strided probability matrices", () => {
    const p = tensor(PROBA, f64);
    const dense = logLoss(tensor(LABELS), p);
    expect(logLoss(int64([0n, 1n, 2n, 2n]), p)).toBe(dense);
    expect(logLoss(strided1D(LABELS, 2, 1), columnMajor(PROBA))).toBe(dense);
  });

  it("clips at the dtype epsilon like scikit-learn", () => {
    const wrong = [
      [0, 1],
      [1, 0],
    ];
    // sklearn: 15.942385152878742 for float32 input and 36.04365338911715 for float64
    expect(logLoss(tensor([0, 1]), tensor(wrong, { dtype: "float32" }))).toBeCloseTo(
      15.942385152878742,
      9
    );
    expect(logLoss(tensor([0, 1]), tensor(wrong, f64))).toBeCloseTo(36.04365338911715, 9);
  });

  it("a two-column matrix is the binary case", () => {
    const two = tensor(
      [
        [0.9, 0.1],
        [0.2, 0.8],
        [0.3, 0.7],
      ],
      f64
    );
    const one = tensor([0.1, 0.8, 0.7], f64);
    const y = tensor([0, 1, 1]);
    expect(logLoss(y, two)).toBeCloseTo(0.22839300363692283, 12);
    expect(logLoss(y, one)).toBeCloseTo(0.22839300363692283, 12);
  });

  it("empty input and validation", () => {
    expect(logLoss(tensor([]), tensor([], f64).reshape([0, 3]))).toBe(0);
    const p = tensor(PROBA, f64);
    expect(() => logLoss(tensor([0, 1, 1, 1]), p)).toThrow(/distinct classes/);
    expect(logLoss(tensor([0, 1, 1, 1]), p, { labels: [0, 1, 2] })).toBeGreaterThan(0);
    expect(() => logLoss(tensor(LABELS), p, { labels: [0, 1] })).toThrow(InvalidParameterError);
    expect(() => logLoss(tensor(LABELS), p, { labels: [2, 1, 0] })).toThrow(/ascending/);
    expect(() => logLoss(tensor([0, 1, 5, 2]), p, { labels: [0, 1, 2] })).toThrow(/not in labels/);
    expect(() => logLoss(tensor([0, 1, 1, 0]), tensor([0.1, 0.2, 1.2, 0.3], f64))).toThrow(
      /range \[0, 1\]/
    );
    expect(() => logLoss(tensor(LABELS), tensor([[1.2, 0, 0], ...PROBA.slice(1)], f64))).toThrow(
      /range \[0, 1\]/
    );
    expect(() =>
      logLoss(tensor([0, 1, 1]), tensor([0.2, 0.9, 0.5], f64), { labels: [0, 1, 2] })
    ).toThrow(InvalidParameterError);
    expect(() => logLoss(tensor(LABELS), p, { normalize: 1 as never })).toThrow(/normalize/);
    expect(() => logLoss(tensor([0, 1]), tensor([[0.5, 0.5]], f64))).toThrow(ShapeError);
    expect(() =>
      logLoss(tensor(LABELS), tensor([[Number.NaN, 0.1, 0.1], ...PROBA.slice(1)], f64))
    ).toThrow(DataValidationError);
  });

  it("sampleWeight must have a non-zero sum", () => {
    const p = tensor(PROBA, f64);
    expect(() => logLoss(tensor(LABELS), p, { sampleWeight: [0, 0, 0, 0] })).toThrow(
      InvalidParameterError
    );
    expect(logLoss(tensor(LABELS), p, { sampleWeight: [0, 0, 0, 0], normalize: false })).toBe(0);
    expect(() => logLoss(tensor(LABELS), p, { sampleWeight: [1, 1] })).toThrow(ShapeError);
    expect(() => logLoss(tensor(LABELS), p, { sampleWeight: [1, Number.NaN, 1, 1] })).toThrow(
      DataValidationError
    );
  });
});

describe("v1.5.0 wave 3 metrics: jaccardScore and matthewsCorrcoef", () => {
  it("binary behavior is unchanged", () => {
    const yTrue = tensor([0, 1, 1, 0, 1]);
    const yPred = tensor([0, 1, 0, 0, 1]);
    expect(jaccardScore(yTrue, yPred)).toBeCloseTo(2 / 3, 15);
    expect(jaccardScore(tensor([0, 0]), tensor([0, 0]))).toBe(1);
    expect(jaccardScore(tensor([]), tensor([]))).toBe(1);
    expect(jaccardScore(tensor([]), tensor([]), null)).toEqual([]);
    expect(matthewsCorrcoef(yTrue, yPred)).toBeCloseTo(0.6666666666666666, 12);
    expect(matthewsCorrcoef(tensor([0, 0, 0]), tensor([0, 0, 0]))).toBe(0);
    expect(matthewsCorrcoef(tensor([]), tensor([]))).toBe(0);
    expect(() => jaccardScore(tensor([2, 2, 3]), tensor([2, 3, 3]))).toThrow(/binary/);
  });

  it("jaccard zeroDivision is honoured", () => {
    expect(jaccardScore(tensor([0, 0]), tensor([0, 0]), { zeroDivision: 0 })).toBe(0);
    expect(jaccardScore(tensor([0, 0]), tensor([0, 0]), { zeroDivision: "warn" })).toBe(0);
    expect(jaccardScore(tensor([0, 0]), tensor([0, 0]), { zeroDivision: Number.NaN })).toBeNaN();
    expect(
      jaccardScore(tensor([0, 1, 2, 2]), tensor([0, 1, 2, 2]), {
        average: null,
        labels: [3],
      })
    ).toEqual([1]);
    expect(
      jaccardScore(tensor([0, 1, 2, 2]), tensor([0, 1, 2, 2]), {
        average: null,
        labels: [3],
        zeroDivision: 0,
      })
    ).toEqual([0]);
  });

  it("multiclass mcc works for strings and int64 labels", () => {
    // sklearn: matthews_corrcoef(['a','b','c','a'], ['a','b','b','a']) = 0.6708203932499369
    expect(
      matthewsCorrcoef(tensor(["a", "b", "c", "a"]), tensor(["a", "b", "b", "a"]))
    ).toBeCloseTo(0.6708203932499369, 12);
    expect(matthewsCorrcoef(int64([0n, 1n, 2n, 0n]), int64([0n, 1n, 1n, 0n]))).toBeCloseTo(
      0.6708203932499369,
      12
    );
  });

  it("reads strided labels", () => {
    const yTrue = strided1D([0, 1, 2, 2, 1, 0], 3, 2);
    const yPred = strided1D([0, 1, 2, 1, 1, 2], 2, 1);
    expect(matthewsCorrcoef(yTrue, yPred)).toBeCloseTo(0.5222329678670935, 12);
    expect(jaccardScore(yTrue, yPred, "macro")).toBeCloseTo(
      jaccardScore(tensor([0, 1, 2, 2, 1, 0]), tensor([0, 1, 2, 1, 1, 2]), "macro"),
      15
    );
  });

  it("mcc rejects mismatched sizes and NaN labels", () => {
    expect(() => matthewsCorrcoef(tensor([0, 1]), tensor([0]))).toThrow(ShapeError);
    expect(() => matthewsCorrcoef(tensor([0, Number.NaN]), tensor([0, 1]))).toThrow(
      DataValidationError
    );
    expect(() => matthewsCorrcoef(tensor([0, 1]), tensor([0, 1]), { sampleWeight: [1] })).toThrow(
      ShapeError
    );
  });
});

describe("v1.5.0 wave 3 metrics: options object for averaged metrics", () => {
  const yTrue = tensor([0, 1, 2, 2, 1, 0]);
  const yPred = tensor([0, 2, 2, 2, 1, 1]);

  it("matches the positional average", () => {
    for (const average of ["micro", "macro", "weighted"] as const) {
      expect(precision(yTrue, yPred, { average })).toBe(precision(yTrue, yPred, average));
      expect(recall(yTrue, yPred, { average })).toBe(recall(yTrue, yPred, average));
      expect(f1Score(yTrue, yPred, { average })).toBe(f1Score(yTrue, yPred, average));
      expect(fbetaScore(yTrue, yPred, 2, { average })).toBe(fbetaScore(yTrue, yPred, 2, average));
      expect(jaccardScore(yTrue, yPred, { average })).toBe(jaccardScore(yTrue, yPred, average));
    }
    expect(precision(yTrue, yPred, { average: null })).toEqual(precision(yTrue, yPred, null));
    expect(precision(yTrue, yPred, {})).toBe(precision(yTrue, yPred));
  });

  it("labels without average scores the listed classes, weighted by support", () => {
    // sklearn: precision_score(y, p, labels=[2], average="weighted") = 0.6666666666666666
    expect(precision(yTrue, yPred, { labels: [2] })).toBeCloseTo(2 / 3, 12);
    expect(precision(yTrue, yPred, { labels: [2, 0], average: null })).toEqual([2 / 3, 1]);
  });

  it("empty input returns the zero-division value", () => {
    const empty = tensor([]);
    expect(precision(empty, empty, { zeroDivision: 1 })).toBe(1);
    expect(precision(empty, empty, { average: null })).toEqual([]);
    expect(precision(empty, empty, { average: "macro" })).toBe(0);
    expect(f1Score(empty, empty, { labels: [0, 1], average: null, zeroDivision: 1 })).toEqual([
      1, 1,
    ]);
  });

  it("validates zeroDivision, labels and sample weights", () => {
    expect(() => precision(yTrue, yPred, { zeroDivision: 2 })).toThrow(/zeroDivision/);
    expect(() => precision(yTrue, yPred, { labels: [] })).toThrow(InvalidParameterError);
    expect(() => precision(yTrue, yPred, { labels: [0, 0] })).toThrow(/unique/);
    expect(() => precision(yTrue, yPred, { labels: ["a"] })).toThrow(DTypeError);
    expect(() => precision(yTrue, yPred, { sampleWeight: [1, 2] })).toThrow(ShapeError);
    expect(() => precision(yTrue, yPred, { average: "nope" as never })).toThrow(
      /Invalid average parameter/
    );
    expect(() =>
      precision(tensor([0, 1]), tensor([0, 1]), {
        average: "binary",
        sampleWeight: [1, Number.POSITIVE_INFINITY],
      })
    ).toThrow(DataValidationError);
  });

  it("accepts tensors and typed arrays as sample weights", () => {
    const w = [1, 2, 1, 1, 1, 1];
    const expected = precision(yTrue, yPred, { average: "macro", sampleWeight: w });
    expect(precision(yTrue, yPred, { average: "macro", sampleWeight: tensor(w, f64) })).toBe(
      expected
    );
    expect(precision(yTrue, yPred, { average: "macro", sampleWeight: new Float64Array(w) })).toBe(
      expected
    );
    expect(precision(yTrue, yPred, { average: "macro", sampleWeight: strided1D(w, 2, 1) })).toBe(
      expected
    );
    expect(
      precision(yTrue, yPred, { average: "macro", sampleWeight: tensor(w.map((v) => [v])) })
    ).toBe(expected);
  });

  it("zero total weight gives the zero-division value for ratio metrics", () => {
    // sklearn: precision_score([1, 2], [1, 2], sample_weight=[0, 0], average="macro") = 0.0
    expect(
      precision(tensor([1, 2]), tensor([1, 2]), { average: "macro", sampleWeight: [0, 0] })
    ).toBe(0);
    expect(
      precision(tensor([1, 2]), tensor([1, 2]), {
        average: "weighted",
        sampleWeight: [0, 0],
        zeroDivision: 1,
      })
    ).toBe(1);
  });

  it("works with strided label views", () => {
    const t = strided1D([0, 1, 2, 2, 1, 0], 3, 2);
    const p = strided1D([0, 2, 2, 2, 1, 1], 2, 1);
    expect(f1Score(t, p, { average: "macro" })).toBe(f1Score(yTrue, yPred, "macro"));
  });
});

describe("v1.5.0 wave 3 metrics: sampleWeight on accuracy, confusionMatrix and regression", () => {
  it("accuracy handles negative weights, zero-sum weights and 0-d inputs", () => {
    // sklearn: accuracy_score([0,1,1,0,1,0], [0,1,0,0,1,1], sample_weight=[1,-1,2,1,1,1]) = 0.4
    expect(
      accuracy(tensor([0, 1, 1, 0, 1, 0]), tensor([0, 1, 0, 0, 1, 1]), {
        sampleWeight: [1, -1, 2, 1, 1, 1],
      })
    ).toBeCloseTo(0.4, 12);
    expect(() => accuracy(tensor([0, 1]), tensor([0, 1]), { sampleWeight: [1, -1] })).toThrow(
      /sum to zero/
    );
    expect(accuracy(tensor([]), tensor([]), { sampleWeight: [] })).toBe(0);
    expect(accuracy(tensor(1), tensor(1), { sampleWeight: tensor(2) })).toBe(1);
  });

  it("confusionMatrix weights compose with labels", () => {
    // sklearn: confusion_matrix([1,2,3,1], [1,3,3,2], sample_weight=[1,2,3,4])
    const cm = confusionMatrix(tensor([1, 2, 3, 1]), tensor([1, 3, 3, 2]), {
      sampleWeight: [1, 2, 3, 4],
    });
    expect(Array.from(cm.data as Float64Array)).toEqual([1, 4, 0, 0, 0, 2, 0, 0, 3]);
    const subset = confusionMatrix(tensor([1, 2, 3, 1]), tensor([1, 3, 3, 2]), {
      labels: [3, 1],
      sampleWeight: [1, 2, 3, 4],
    });
    expect(Array.from(subset.data as Float64Array)).toEqual([3, 0, 0, 1]);
  });

  it("regression metrics with weights: negative weights and constant targets", () => {
    const t = tensor([1, 2, 3], f64);
    const p = tensor([1.5, 2, 2], f64);
    // sklearn with sample_weight=[1, -1, 3]
    expect(mse(t, p, { sampleWeight: [1, -1, 3] })).toBeCloseTo(1.0833333333333333, 12);
    expect(mae(t, p, { sampleWeight: [1, -1, 3] })).toBeCloseTo(1.1666666666666667, 12);
    expect(rmse(t, p, { sampleWeight: [1, -1, 3] })).toBeCloseTo(Math.sqrt(1.0833333333333333), 12);
    // sklearn: r2_score([1,2,3,4], [1.5,2,2,5], sample_weight=[0,2,1,1]) = 0.2727272727272727
    expect(
      r2Score(tensor([1, 2, 3, 4], f64), tensor([1.5, 2, 2, 5], f64), {
        sampleWeight: [0, 2, 1, 1],
      })
    ).toBeCloseTo(0.2727272727272727, 12);
    // constant targets among the weighted samples: perfect fit 1, otherwise 0
    const ct = tensor([2, 2, 2, 9], f64);
    expect(r2Score(ct, tensor([2, 2, 2, 5], f64), { sampleWeight: [1, 1, 2, 0] })).toBe(1);
    expect(r2Score(ct, tensor([2, 2, 3, 5], f64), { sampleWeight: [1, 1, 2, 0] })).toBe(0);
    expect(
      explainedVarianceScore(ct, tensor([2, 2, 2, 5], f64), { sampleWeight: [1, 1, 2, 0] })
    ).toBe(1);
  });

  it("weights are validated for every regression metric", () => {
    const t = tensor([1, 2, 3], f64);
    const p = tensor([1, 2, 4], f64);
    for (const fn of [
      mse,
      rmse,
      mae,
      r2Score,
      explainedVarianceScore,
      meanAbsolutePercentageError,
      meanSquaredLogError,
    ]) {
      expect(() => fn(t, p, { sampleWeight: [1, 1] })).toThrow(ShapeError);
      expect(() => fn(t, p, { sampleWeight: [0, 0, 0] })).toThrow(InvalidParameterError);
      expect(() => fn(t, p, { sampleWeight: [1, Number.NaN, 1] })).toThrow(DataValidationError);
    }
    expect(mse(tensor([], f64), tensor([], f64), { sampleWeight: [] })).toBe(0);
    expect(() => r2Score(tensor([], f64), tensor([], f64), { sampleWeight: [] })).toThrow(
      InvalidParameterError
    );
  });

  it("reads strided inputs and weights", () => {
    const t = strided1D([1, 2, 3, 4], 2, 1);
    const p = strided1D([1.5, 2, 2, 5], 3, 2);
    const w = strided1D([1, 2, 3, 4], 2, 0);
    const dense = mse(tensor([1, 2, 3, 4], f64), tensor([1.5, 2, 2, 5], f64), {
      sampleWeight: [1, 2, 3, 4],
    });
    expect(mse(t, p, { sampleWeight: w })).toBe(dense);
  });

  it("msle weights match scikit-learn and negatives are still rejected", () => {
    const t = tensor([3, 5, 2.5, 7], f64);
    const p = tensor([2.5, 5, 4, 8], f64);
    expect(meanSquaredLogError(t, p, { sampleWeight: [1, 2, 3, 4] })).toBeCloseTo(
      0.04549730536710696,
      12
    );
    expect(() => meanSquaredLogError(tensor([-1, 1], f64), tensor([1, 1], f64))).toThrow(
      InvalidParameterError
    );
  });
});

describe("v1.5.0 wave 3 metrics: meanAbsolutePercentageError", () => {
  it("is a fraction with scikit-learn's epsilon guard, mape stays a percentage", () => {
    const t = tensor([3, -0.5, 2, 7], f64);
    const p = tensor([2.5, 0, 2, 8], f64);
    // sklearn: mean_absolute_percentage_error = 0.3273809523809524
    expect(meanAbsolutePercentageError(t, p)).toBeCloseTo(0.3273809523809524, 14);
    expect(mape(t, p)).toBeCloseTo(32.73809523809524, 12);
    // zero targets are not skipped: the term is |p - t| / eps
    const zeroT = tensor([0, 2, -4], f64);
    const zeroP = tensor([1, 2, -3], f64);
    expect(meanAbsolutePercentageError(zeroT, zeroP)).toBeCloseTo(1501199875790165.2, -2);
    expect(
      meanAbsolutePercentageError(tensor([0, 1], f64), tensor([1, 1], f64), {
        sampleWeight: [0, 1],
      })
    ).toBe(0);
  });

  it("handles empty input and rejects bad dtypes", () => {
    expect(meanAbsolutePercentageError(tensor([], f64), tensor([], f64))).toBe(0);
    expect(() => meanAbsolutePercentageError(tensor(["a"]), tensor([1], f64))).toThrow(DTypeError);
    expect(() => meanAbsolutePercentageError(tensor([Number.NaN], f64), tensor([1], f64))).toThrow(
      DataValidationError
    );
    expect(() => meanAbsolutePercentageError(tensor([1, 2], f64), tensor([1], f64))).toThrow(
      ShapeError
    );
  });
});

describe("v1.5.0 wave 3 metrics: int64 labels mixed with numeric labels", () => {
  it("confusionMatrix, accuracy and classificationReport accept the mix", () => {
    const big = int64([0n, 1n, 1n, 0n, 1n]);
    const num = tensor([0, 1, 0, 0, 1]);
    expect(Array.from(confusionMatrix(big, num).data as Float64Array)).toEqual([2, 0, 1, 2]);
    expect(Array.from(confusionMatrix(num, big).data as Float64Array)).toEqual([2, 1, 0, 2]);
    expect(accuracy(big, num)).toBe(0.8);
    expect(classificationReport(big, num)).toBe(classificationReport(tensor([0, 1, 1, 0, 1]), num));
    expect(classificationReport(num, big)).toContain("Weighted Avg");
    expect(f1Score(big, num)).toBeCloseTo(0.8, 15);
    expect(f1Score(int64([0n, 1n, 2n, 2n]), tensor([0, 2, 2, 2]), "macro")).toBeCloseTo(
      f1Score(tensor([0, 1, 2, 2]), tensor([0, 2, 2, 2]), "macro"),
      15
    );
  });

  it("bool labels and the labels option work with the mix", () => {
    const cm = confusionMatrix(int64([1n, 2n, 3n]), tensor([1, 3, 3]), { labels: [3n, 1] });
    expect(Array.from(cm.data as Float64Array)).toEqual([1, 0, 0, 1]);
    expect(confusionMatrix(int64([1n, 2n]), tensor([true, false])).shape).toEqual([3, 3]);
  });

  it("rejects non-integer numeric labels and unsafe int64 values", () => {
    expect(() => confusionMatrix(int64([1n]), tensor([1.5]))).toThrow(DTypeError);
    expect(() => accuracy(int64([2n ** 60n]), tensor([1]))).toThrow(/safe integer/);
    expect(() => confusionMatrix(int64([1n]), tensor([1]), { labels: [2n ** 60n] })).toThrow(
      /safe integer/
    );
    expect(() => confusionMatrix(int64([1n]), tensor(["1"]))).toThrow(DTypeError);
  });

  it("int64 against int64 still keeps bigint labels exact", () => {
    const big = 2n ** 60n;
    const cm = confusionMatrix(int64([big, big + 1n]), int64([big, big]));
    expect(Array.from(cm.data as Float64Array)).toEqual([1, 0, 1, 0]);
  });
});

describe("reviewer additions", () => {
  it("classificationReport shows 0 (not NaN) for a class with an undefined score", () => {
    // Class 0 is predicted once but never true: its recall has a zero denominator.
    const report = classificationReport(tensor([1, 1]), tensor([0, 1]));
    expect(report).not.toContain("NaN");
    expect(report).toContain("Macro Avg     0.5000      0.2500      0.3333      2");
    expect(report).toContain("Weighted Avg  1.0000      0.5000      0.6667      2");
  });

  it("jaccardScore averages over both classes of a binary problem like scikit-learn", () => {
    // scikit-learn: jaccard_score([1, 1], [0, 1], average="macro") == 0.25
    expect(jaccardScore(tensor([1, 1]), tensor([0, 1]), "macro")).toBeCloseTo(0.25, 12);
    expect(jaccardScore(tensor([1, 1]), tensor([0, 1]), "micro")).toBeCloseTo(1 / 3, 12);
    expect(jaccardScore(tensor([1, 1]), tensor([0, 1]), null)).toEqual([0, 0.5]);
    expect(jaccardScore(tensor([1, 1]), tensor([0, 1]))).toBe(0.5);
  });
});
