/**
 * Wave 3 tests for the ml, datasets and plot groups of v1.5.0:
 *
 * - tree estimators: sampleWeight, classWeight, minImpurityDecrease, maxLeafNodes, ccpAlpha
 *   (DecisionTree, RandomForest, ExtraTrees, GradientBoosting),
 * - Ridge with alpha = 0 on rank-deficient data,
 * - fetch20Newsgroups / fetchIMDB with the official archives (fetch is mocked, no network),
 * - calculateWhiskers clamping.
 *
 * Reference values come from scikit-learn 1.8 (random_state=0, min_samples_leaf=6; the data set was chosen so that
 * the results do not depend on how scikit-learn breaks ties between equal splits), numpy 2.4 (`lstsq` for the minimum-norm solutions) and matplotlib's
 * `boxplot_stats`. The data set is 60 rows, 3 features, 3 classes with weights in [0.2, 3].
 */
import { gzipSync } from "node:zlib";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  DataValidationError,
  DeepboxError,
  InvalidParameterError,
  ShapeError,
} from "../../src/core";
import { fetch20Newsgroups, fetchIMDB } from "../../src/datasets";
import {
  GradientBoostingClassifier,
  GradientBoostingRegressor,
} from "../../src/ml/ensemble/GradientBoosting";
import { Ridge } from "../../src/ml/linear/Ridge";
import { DecisionTreeClassifier, DecisionTreeRegressor } from "../../src/ml/tree/DecisionTree";
import { ExtraTreesClassifier, ExtraTreesRegressor } from "../../src/ml/tree/ExtraTrees";
import { RandomForestClassifier, RandomForestRegressor } from "../../src/ml/tree/RandomForest";
import { type Tensor, tensor } from "../../src/ndarray";
import { calculateQuartiles, calculateWhiskers } from "../../src/plot/utils/statistics";

afterEach(() => {
  vi.unstubAllGlobals();
});

type ClassifierFix = { leaves: number; depth: number; proba: number[][]; imp: number[] };
type RegressorFix = { leaves: number; depth: number; pred: number[]; imp: number[] };

type Fixtures = {
  X: number[][];
  Xt: number[][];
  yc: number[];
  yb: number[];
  yr: number[];
  sw: number[];
  tc: {
    sw_gini: ClassifierFix;
    sw_entropy: ClassifierFix;
    balanced: ClassifierFix;
    cwmap: ClassifierFix;
    mid: ClassifierFix;
    mln: ClassifierFix;
    mln_depth: ClassifierFix;
    ccp: ClassifierFix & { alpha: number };
    ccp_entropy: ClassifierFix & { alpha: number };
  };
  tr: {
    sw: RegressorFix;
    mid: RegressorFix;
    mln: RegressorFix;
    ccp: RegressorFix & { alpha: number };
  };
  rfc: { proba: number[][] };
  rfr: { pred: number[] };
  rfc_growth: { proba: number[][] };
  gb: {
    reg_squared_error: number[];
    reg_huber: number[];
    reg_quantile: number[];
    stump_squared_error: number[];
    stump_absolute_error: number[];
    stump_huber: number[];
    stump_quantile: number[];
    stump_clf: number[];
    reg_growth: number[];
    clf: number[];
    clf_growth: number[];
  };
  ridge: {
    X: number[][];
    y: number[];
    coef: number[];
    intercept: number;
    wideX: number[][];
    wideY: number[];
    wideCoef: number[];
    wideIntercept: number;
  };
};

const FIX: Fixtures = JSON.parse(
  `{"X":[[0.126,-0.132,0.64],[0.105,-0.536,0.362],[1.304,0.947,-0.704],[-1.265,-0.623,0.041],[-2.325,-0.219,-1.246],[-0.732,-0.544,-0.316],[0.412,1.043,-0.129],[1.366,-0.665,0.352],[0.903,0.094,-0.743],[-0.922,-0.458,0.22],[-1.01,-0.209,-0.159],[0.541,0.215,0.355],[-0.654,-0.13,0.784],[1.493,-1.259,1.514],[1.346,0.781,0.264],[-0.314,1.458,1.96],[1.802,1.315,0.357],[-1.208,-0.004,0.656],[-1.288,0.395,0.43],[0.696,-1.184,-0.662],[-0.436,-1.17,1.739],[-0.496,0.329,-0.259],[1.583,1.32,0.633],[-2.204,0.052,0.684],[1.004,-0.618,1.822],[-1.32,-0.662,0.935],[0.049,2.002,0.189],[-0.633,-0.378,-1.091],[-1.278,0.63,0.581],[1.295,-0.755,1.689],[-0.287,1.574,-0.433],[-0.735,0.25,1.031],[0.161,-0.586,-1.341],[-1.402,0.503,0.99],[-0.164,-1.074,0.873],[-1.28,-0.713,0.621],[-2.25,0.386,-0.582],[0.109,-0.076,0.202],[0.694,-0.758,1.421],[0.726,0.844,1.165],[0.788,0.844,0.076],[-1.427,-0.135,-0.77],[-1.423,0.258,-0.569],[-1.03,-1.043,0.268],[0.359,1.322,-0.014],[1.042,1.402,1.15],[-2.365,1.229,0.34],[0.424,0.371,0.383],[0.319,-0.359,-1.902],[-0.109,-0.804,1.08],[-0.289,0.083,-0.85],[-0.511,-0.012,-1.485],[0.301,-0.106,-1.186],[-2.398,0.513,-0.298],[-0.53,-0.236,1.816],[-0.05,0.087,-1.487],[1.647,0.917,1.067],[0.048,0.917,0.371],[0.613,-0.152,-1.474],[1.029,-1.935,-0.24]],"yc":[0,0,1,0,0,0,1,1,1,0,0,0,1,2,1,2,1,0,0,1,1,0,1,0,2,1,1,0,0,2,1,2,0,1,1,0,0,0,1,2,1,0,0,0,1,2,0,1,0,1,0,0,1,0,1,0,2,1,0,0],"yr":[0.38,-0.772,3.813,-3.624,-4.546,-2.29,1.58,2.527,1.896,-3.004,-2.365,1.487,-1.245,2.126,3.933,-0.148,3.893,-2.375,-2.115,1.397,-1.802,-0.516,3.327,-4.408,1.062,-3.379,-0.845,-2.008,-2.029,1.344,0.246,-0.678,-0.834,-2.361,-1.459,-3.556,-3.792,-0.157,0.004,2.872,2.705,-3.233,-2.419,-3.089,0.314,2.45,-4.42,1.223,-0.212,-0.998,-0.764,-1.476,0.584,-3.714,-1.802,0.242,4.172,1.152,0.548,2.974],"yb":[0,0,1,0,0,0,1,1,1,0,0,0,1,1,1,1,1,0,0,1,1,0,1,0,1,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,1,0,0,0,1,1,0,1,0,1,0,0,1,0,1,0,1,1,0,0],"sw":[2.513,1.69,2.477,2.992,1.182,0.679,1.297,2.309,1.43,1.847,0.557,2.233,0.984,0.734,2.616,1.78,1.557,2.717,0.441,2.149,1.118,0.691,2.089,1.216,1.124,2.842,0.758,1.634,0.267,0.657,2.674,2.41,1.759,0.823,1.762,0.234,2.196,2.207,2.009,1.912,0.406,0.89,1.808,1.304,2.978,2.786,0.626,1.852,2.149,0.582,1.075,2.205,2.723,1.157,0.869,2.501,1.838,1.534,0.917,0.403],"Xt":[[0.219,0.845,0.993],[-1.375,1.998,0.947],[-0.379,-0.819,-0.969],[0.123,-0.648,-0.765],[0.811,0.365,-0.395],[0.734,1.367,-1.094],[-0.603,0.943,0.719],[0.227,1.162,-1.088]],"tc":{"sw_gini":{"leaves":7,"depth":5,"proba":[[0.03292657269093204,0.9670734273090679,0.0],[0.03292657269093204,0.9670734273090679,0.0],[0.17640643460874308,0.823593565391257,0.0],[0.17640643460874308,0.823593565391257,0.0],[0.4687721160651097,0.5312278839348904,0.0],[0.03292657269093204,0.9670734273090679,0.0],[0.03292657269093204,0.9670734273090679,0.0],[0.03292657269093204,0.9670734273090679,0.0]],"imp":[0.1572305020660816,0.5130643143343515,0.32970518359956696]},"sw_entropy":{"leaves":7,"depth":5,"proba":[[0.03292657269093204,0.9670734273090679,0.0],[0.03292657269093204,0.9670734273090679,0.0],[0.17640643460874308,0.823593565391257,0.0],[0.17640643460874308,0.823593565391257,0.0],[0.4687721160651097,0.5312278839348904,0.0],[0.03292657269093204,0.9670734273090679,0.0],[0.03292657269093204,0.9670734273090679,0.0],[0.03292657269093204,0.9670734273090679,0.0]],"imp":[0.14938519663111266,0.3512666851769634,0.49934811819192376]},"balanced":{"leaves":6,"depth":4,"proba":[[0.07348242811501597,0.926517571884984,0.0],[0.07348242811501597,0.926517571884984,0.0],[1.0,0.0,0.0],[1.0,0.0,0.0],[0.38818565400843885,0.6118143459915611,0.0],[0.07348242811501597,0.926517571884984,0.0],[0.07348242811501597,0.926517571884984,0.0],[0.07348242811501597,0.926517571884984,0.0]],"imp":[0.18608611995808902,0.1940356775151877,0.6198782025267233]},"cwmap":{"leaves":8,"depth":5,"proba":[[0.0,0.7912860154602951,0.20871398453970483],[0.20801717252396162,0.731829073482428,0.06015375399361022],[0.009873941858898188,0.9730855455781646,0.017040512562937194],[0.009873941858898188,0.9730855455781646,0.017040512562937194],[0.0,1.0,0.0],[0.0,1.0,0.0],[0.0,0.7912860154602951,0.20871398453970483],[0.0,1.0,0.0]],"imp":[0.28480426473409054,0.12834881655858896,0.5868469187073205]},"mid":{"leaves":6,"depth":4,"proba":[[0.03292657269093204,0.9670734273090679,0.0],[0.03292657269093204,0.9670734273090679,0.0],[0.17640643460874308,0.823593565391257,0.0],[0.17640643460874308,0.823593565391257,0.0],[0.4687721160651097,0.5312278839348904,0.0],[0.03292657269093204,0.9670734273090679,0.0],[0.03292657269093204,0.9670734273090679,0.0],[0.03292657269093204,0.9670734273090679,0.0]],"imp":[0.1595201143950485,0.5205356278785845,0.3199442577263671]},"mln":{"leaves":5,"depth":4,"proba":[[0.03292657269093204,0.9670734273090679,0.0],[0.03292657269093204,0.9670734273090679,0.0],[0.17640643460874306,0.8235935653912569,0.0],[0.17640643460874306,0.8235935653912569,0.0],[0.46877211606510977,0.5312278839348904,0.0],[0.03292657269093204,0.9670734273090679,0.0],[0.03292657269093204,0.9670734273090679,0.0],[0.03292657269093204,0.9670734273090679,0.0]],"imp":[0.10547252360711058,0.5540090007551695,0.3405184756377199]},"mln_depth":{"leaves":4,"depth":2,"proba":[[0.0,0.8,0.2],[0.0,0.8,0.2],[0.8484848484848485,0.15151515151515152,0.0],[0.8484848484848485,0.15151515151515152,0.0],[0.8484848484848485,0.15151515151515152,0.0],[0.09090909090909091,0.9090909090909091,0.0],[0.09090909090909091,0.9090909090909091,0.0],[0.09090909090909091,0.9090909090909091,0.0]],"imp":[0.21241786241115734,0.4190693308300925,0.36851280675875014]},"ccp":{"alpha":0.06160960229561859,"leaves":4,"depth":3,"proba":[[0.03292657269093204,0.9670734273090679,0.0],[0.03292657269093204,0.9670734273090679,0.0],[0.17640643460874308,0.823593565391257,0.0],[0.17640643460874308,0.823593565391257,0.0],[0.8349635576212106,0.16503644237878945,0.0],[0.03292657269093204,0.9670734273090679,0.0],[0.03292657269093204,0.9670734273090679,0.0],[0.03292657269093204,0.9670734273090679,0.0]],"imp":[0.0,0.6193314519406005,0.3806685480593995]},"ccp_entropy":{"alpha":0.11354700259362394,"leaves":4,"depth":3,"proba":[[0.03292657269093204,0.9670734273090679,0.0],[0.03292657269093204,0.9670734273090679,0.0],[0.17640643460874308,0.823593565391257,0.0],[0.17640643460874308,0.823593565391257,0.0],[0.8349635576212106,0.16503644237878945,0.0],[0.03292657269093204,0.9670734273090679,0.0],[0.03292657269093204,0.9670734273090679,0.0],[0.03292657269093204,0.9670734273090679,0.0]],"imp":[0.0,0.43338805459530877,0.5666119454046912]}},"tr":{"sw":{"leaves":8,"depth":4,"pred":[0.19110197476686783,-2.931781644770086,-1.4057367327667611,-0.5187879483500717,2.0101885233588708,3.4327123907914032,-1.4057367327667611,0.19110197476686783],"imp":[0.9742754879552326,0.025724512044767563,0.0]},"mid":{"leaves":8,"depth":4,"pred":[0.19110197476686783,-2.931781644770086,-1.4057367327667611,-0.5187879483500717,2.0101885233588708,3.4327123907914032,-1.4057367327667611,0.19110197476686783],"imp":[0.9742754879552326,0.025724512044767563,0.0]},"mln":{"leaves":6,"depth":3,"pred":[-0.06763601185288476,-3.264993592796225,-1.4057367327667611,-0.06763601185288476,2.0101885233588703,3.4327123907914032,-1.4057367327667611,-0.06763601185288476],"imp":[0.9806550497512535,0.019344950248746422,0.0]},"ccp":{"alpha":0.18361657872780382,"leaves":4,"depth":2,"pred":[0.23522796484594555,-3.264993592796225,-1.4057367327667611,0.23522796484594555,3.0087827021219447,3.0087827021219447,-1.4057367327667611,0.23522796484594555],"imp":[1.0,0.0,0.0]}},"rfc":{"proba":[[0.026293298660316447,0.9737067013396835,0.0],[0.026293298660316447,0.9737067013396835,0.0],[0.1452083489189796,0.8547916510810204,0.0],[0.1452083489189796,0.8547916510810204,0.0],[0.4117160211065394,0.5882839788934606,0.0],[0.026293298660316447,0.9737067013396835,0.0],[0.026293298660316447,0.9737067013396835,0.0],[0.026293298660316447,0.9737067013396835,0.0]]},"rfr":{"pred":[0.19110197476686783,-2.9317816447700857,-1.4057367327667611,-0.5187879483500717,2.0101885233588703,3.4327123907914032,-1.4057367327667611,0.19110197476686783]},"rfc_growth":{"proba":[[0.03292657269093204,0.9670734273090679,0.0],[0.03292657269093204,0.9670734273090679,0.0],[0.17640643460874306,0.8235935653912568,0.0],[0.17640643460874306,0.8235935653912568,0.0],[0.8349635576212107,0.16503644237878945,0.0],[0.03292657269093204,0.9670734273090679,0.0],[0.03292657269093204,0.9670734273090679,0.0],[0.03292657269093204,0.9670734273090679,0.0]]},"gb":{"reg_squared_error":[0.253336495176757,-2.1567912978903774,-1.360231230834521,-0.39095274840095706,1.3538476652749398,2.0040191139296586,-0.7262719248627157,0.253336495176757],"reg_huber":[0.14019423574227102,-2.1271398192711053,-1.3182805455246536,-0.39502030405018507,1.3799731692670314,2.0431906125252066,-0.6933510347642188,0.14019423574227102],"reg_quantile":[1.827192768,1.827192768,1.827192768,1.827192768,2.42827008,2.88529152,1.827192768,1.827192768],"stump_squared_error":[-0.1459231262280784,-0.1459231262280784,-0.1459231262280784,-0.1459231262280784,-0.1459231262280784,-0.1459231262280784,-0.1459231262280784,-0.1459231262280784],"stump_absolute_error":[0.004,0.004,0.004,0.004,0.004,0.004,0.004,0.004],"stump_huber":[-0.12934864534755952,-0.12934864534755952,-0.12934864534755952,-0.12934864534755952,-0.12934864534755952,-0.12934864534755952,-0.12934864534755952,-0.12934864534755952],"stump_quantile":[3.327,3.327,3.327,3.327,3.327,3.327,3.327,3.327],"stump_clf":[0.5577341361157521,0.5577341361157521,0.5577341361157521,0.5577341361157521,0.5577341361157521,0.5577341361157521,0.5577341361157521,0.5577341361157521],"reg_growth":[-0.020795075799907053,-2.404796283994289,-1.0982622739172059,-0.25496667765272646,1.5356541997604862,1.8241439307983196,-0.8640906720643864,-0.020795075799907053],"clf":[0.8677196652913047,0.8677196652913047,0.17514850115335936,0.17514850115335936,0.855428226679299,0.8873375500261564,0.77185971871962,0.77185971871962],"clf_growth":[0.8710324407108646,0.8710324407108646,0.17239382871146047,0.17239382871146047,0.8100668921410312,0.8478549067088815,0.7777072652852355,0.7777072652852355]},"ridge":{"X":[[0.034,1.36,1.225,1.394],[-0.51,-0.298,-0.527,-0.808],[0.57,-0.056,0.747,0.514],[-1.847,1.567,-0.096,-0.28],[0.68,-0.137,-0.379,0.543],[0.463,0.825,-0.203,1.288],[-0.153,0.686,-0.87,0.533],[-1.514,0.395,-0.671,-1.119],[-1.92,-0.814,-0.468,-2.734],[-1.193,-1.492,0.037,-2.685],[0.897,-0.233,-0.744,0.664],[0.385,0.717,-0.3,1.102]],"y":[0.545,1.043,-0.207,-0.814,0.348,0.248,1.099,-1.285,-0.662,-0.838,-1.734,0.126],"coef":[0.0805213163087564,0.07076152895398817,0.0835815054666843,0.15128284526274466],"intercept":-0.12919379345798893,"wideX":[[0.528,-0.739,1.386,0.822,0.627,0.402],[0.956,-1.332,0.614,0.603,-1.768,0.347],[-0.25,0.782,-0.439,-0.018,0.343,-0.876],[0.599,-0.105,0.492,-0.522,1.086,0.605]],"wideY":[-0.178,0.632,1.26,1.791],"wideCoef":[0.26336545909846354,0.14396907007661533,-0.5694098188999698,-0.9725767778414545,-0.062395828833507555,0.35206295514545694],"wideIntercept":1.2755891774737913}}`
) as Fixtures;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

const flat = (t: Tensor): number[] => Array.from(t.data as ArrayLike<number>);
const mat = (rows: readonly (readonly number[])[]): Tensor =>
  tensor(
    rows.map((r) => [...r]),
    { dtype: "float64" }
  );
const vec = (values: readonly number[]): Tensor => tensor([...values], { dtype: "float64" });
const ivec = (values: readonly number[]): Tensor => tensor([...values], { dtype: "int32" });

function expectClose(actual: readonly number[], expected: readonly number[], tol = 1e-9): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(Math.abs((actual[i] as number) - (expected[i] as number))).toBeLessThan(tol);
  }
}

const X = mat(FIX.X);
const Xt = mat(FIX.Xt);
const yc = ivec(FIX.yc);
const yb = ivec(FIX.yb);
const yr = vec(FIX.yr);
const sw = vec(FIX.sw);

function expectClassifier(
  model: DecisionTreeClassifier,
  fix: { leaves: number; depth: number; proba: number[][]; imp: number[] }
): void {
  expect(model.getNLeaves()).toBe(fix.leaves);
  expect(model.getDepth()).toBe(fix.depth);
  expectClose(flat(model.predictProba(Xt)), fix.proba.flat());
  expectClose(flat(model.featureImportances), fix.imp, 1e-9);
}

function expectRegressor(
  model: DecisionTreeRegressor,
  fix: { leaves: number; depth: number; pred: number[]; imp: number[] }
): void {
  expect(model.getNLeaves()).toBe(fix.leaves);
  expect(model.getDepth()).toBe(fix.depth);
  expectClose(flat(model.predict(Xt)), fix.pred);
  expectClose(flat(model.featureImportances), fix.imp, 1e-9);
}

const treeBase = { maxDepth: Number.POSITIVE_INFINITY, minSamplesLeaf: 6 };

// ---------------------------------------------------------------------------
// DecisionTreeClassifier
// ---------------------------------------------------------------------------

describe("DecisionTreeClassifier sample weights, class weights and growth limits", () => {
  it("matches scikit-learn with sampleWeight (gini and entropy)", () => {
    expectClassifier(new DecisionTreeClassifier(treeBase).fit(X, yc, sw), FIX.tc.sw_gini);
    expectClassifier(
      new DecisionTreeClassifier({ ...treeBase, criterion: "entropy" }).fit(X, yc, sw),
      FIX.tc.sw_entropy
    );
  });

  it("matches scikit-learn with classWeight 'balanced' and with a weight map", () => {
    expectClassifier(
      new DecisionTreeClassifier({ ...treeBase, classWeight: "balanced" }).fit(X, yc),
      FIX.tc.balanced
    );
    expectClassifier(
      new DecisionTreeClassifier({ ...treeBase, classWeight: { 0: 1, 1: 4, 2: 0.5 } }).fit(
        X,
        yc,
        sw
      ),
      FIX.tc.cwmap
    );
  });

  it("classWeight is the same as multiplying the sample weights by hand", () => {
    const counts = [0, 0, 0];
    for (const label of FIX.yc) counts[label] = (counts[label] as number) + 1;
    const weights = FIX.yc.map((label, i) => {
      const classWeight = FIX.yc.length / (3 * (counts[label] as number));
      return classWeight * (FIX.sw[i] as number);
    });
    const byHand = new DecisionTreeClassifier(treeBase).fit(X, yc, vec(weights));
    const viaOption = new DecisionTreeClassifier({ ...treeBase, classWeight: "balanced" }).fit(
      X,
      yc,
      sw
    );
    expectClose(flat(viaOption.predictProba(Xt)), flat(byHand.predictProba(Xt)), 1e-12);
  });

  it("minImpurityDecrease matches scikit-learn", () => {
    expectClassifier(
      new DecisionTreeClassifier({ ...treeBase, minImpurityDecrease: 0.01 }).fit(X, yc, sw),
      FIX.tc.mid
    );
  });

  it("maxLeafNodes grows best-first like scikit-learn", () => {
    expectClassifier(
      new DecisionTreeClassifier({ ...treeBase, maxLeafNodes: 5 }).fit(X, yc, sw),
      FIX.tc.mln
    );
    expectClassifier(
      new DecisionTreeClassifier({
        minSamplesLeaf: 6,
        maxDepth: 2,
        maxLeafNodes: 6,
      }).fit(X, yc),
      FIX.tc.mln_depth
    );
  });

  it("ccpAlpha prunes like scikit-learn (gini and entropy)", () => {
    expectClassifier(
      new DecisionTreeClassifier({ ...treeBase, ccpAlpha: FIX.tc.ccp.alpha }).fit(X, yc, sw),
      FIX.tc.ccp
    );
    expectClassifier(
      new DecisionTreeClassifier({
        ...treeBase,
        criterion: "entropy",
        ccpAlpha: FIX.tc.ccp_entropy.alpha,
      }).fit(X, yc, sw),
      FIX.tc.ccp_entropy
    );
  });

  it("ccpAlpha shrinks the tree monotonically and a huge value leaves a single leaf", () => {
    let previous = Number.POSITIVE_INFINITY;
    for (const alpha of [0, 0.002, 0.01, 0.03, 0.1, 10]) {
      const model = new DecisionTreeClassifier({ ...treeBase, ccpAlpha: alpha }).fit(X, yc, sw);
      expect(model.getNLeaves()).toBeLessThanOrEqual(previous);
      previous = model.getNLeaves();
    }
    expect(previous).toBe(1);
    const stump = new DecisionTreeClassifier({ ccpAlpha: 10 }).fit(X, yc, sw);
    expect(flat(stump.featureImportances)).toEqual([0, 0, 0]);
  });

  it("maxLeafNodes bounds the number of leaves; 2 gives a single split", () => {
    for (const limit of [2, 3, 4, 9]) {
      const model = new DecisionTreeClassifier({
        maxDepth: Number.POSITIVE_INFINITY,
        maxLeafNodes: limit,
      }).fit(X, yc);
      expect(model.getNLeaves()).toBeLessThanOrEqual(limit);
    }
    const stump = new DecisionTreeClassifier({ maxLeafNodes: 2 }).fit(X, yc);
    expect(stump.getNLeaves()).toBe(2);
    expect(stump.getDepth()).toBe(1);
  });

  it("a weight of zero removes the sample, an integer weight repeats it", () => {
    const keep = FIX.sw.map((_, i) => (i % 3 === 0 ? 0 : 1));
    const rows = FIX.X.filter((_, i) => keep[i] === 1);
    const labels = FIX.yc.filter((_, i) => keep[i] === 1);
    const zeroed = new DecisionTreeClassifier({ minSamplesLeaf: 2 }).fit(X, yc, vec(keep));
    const dropped = new DecisionTreeClassifier({ minSamplesLeaf: 2 }).fit(mat(rows), ivec(labels));
    expectClose(flat(zeroed.predictProba(Xt)), flat(dropped.predictProba(Xt)), 1e-12);
    expect(zeroed.getNLeaves()).toBe(dropped.getNLeaves());

    const counts = FIX.sw.map((_, i) => 1 + (i % 3));
    const repeatedRows: number[][] = [];
    const repeatedLabels: number[] = [];
    FIX.X.forEach((row, i) => {
      for (let k = 0; k < (counts[i] as number); k++) {
        repeatedRows.push(row);
        repeatedLabels.push(FIX.yc[i] as number);
      }
    });
    const weighted = new DecisionTreeClassifier().fit(X, yc, vec(counts));
    const repeated = new DecisionTreeClassifier().fit(mat(repeatedRows), ivec(repeatedLabels));
    expectClose(flat(weighted.predictProba(Xt)), flat(repeated.predictProba(Xt)), 1e-12);
  });

  it("scaling all weights does not change the tree", () => {
    const a = new DecisionTreeClassifier(treeBase).fit(X, yc, sw);
    const scaled = vec(FIX.sw.map((w) => w * 1000));
    const b = new DecisionTreeClassifier(treeBase).fit(X, yc, scaled);
    expectClose(flat(a.predictProba(Xt)), flat(b.predictProba(Xt)), 1e-12);
    expect(a.getNLeaves()).toBe(b.getNLeaves());
    // minImpurityDecrease is relative to the total weight, so it is scale free as well.
    const c = new DecisionTreeClassifier({ ...treeBase, minImpurityDecrease: 0.01 }).fit(X, yc, sw);
    const d = new DecisionTreeClassifier({ ...treeBase, minImpurityDecrease: 0.01 }).fit(
      X,
      yc,
      scaled
    );
    expect(c.getNLeaves()).toBe(d.getNLeaves());
  });

  it("accepts integer and float32 weights and works on a single sample", () => {
    const ints = tensor(
      FIX.sw.map((w) => Math.max(1, Math.round(w))),
      { dtype: "int32" }
    );
    const model = new DecisionTreeClassifier().fit(X, yc, ints);
    expect(model.getNLeaves()).toBeGreaterThan(1);
    const f32 = tensor(
      FIX.sw.map((w) => w),
      { dtype: "float32" }
    );
    expect(new DecisionTreeClassifier().fit(X, yc, f32).getNLeaves()).toBeGreaterThan(1);
    const one = new DecisionTreeClassifier().fit(mat([[1, 2]]), ivec([3]), vec([2.5]));
    expect(flat(one.predict(mat([[0, 0]])))).toEqual([3]);
    expect(flat(one.predictProba(mat([[0, 0]])))).toEqual([1]);
  });

  it("rejects invalid sample weights", () => {
    const make = () => new DecisionTreeClassifier();
    expect(() => make().fit(X, yc, vec([1, 2, 3]))).toThrow(ShapeError);
    expect(() => make().fit(X, yc, mat([FIX.sw]))).toThrow(ShapeError);
    expect(() => make().fit(X, yc, vec(FIX.sw.map((_, i) => (i === 3 ? -1 : 1))))).toThrow(
      DataValidationError
    );
    expect(() => make().fit(X, yc, vec(FIX.sw.map((_, i) => (i === 3 ? Number.NaN : 1))))).toThrow(
      DataValidationError
    );
    expect(() =>
      make().fit(X, yc, vec(FIX.sw.map((_, i) => (i === 3 ? Number.POSITIVE_INFINITY : 1))))
    ).toThrow(DataValidationError);
    expect(() => make().fit(X, yc, vec(FIX.sw.map(() => 0)))).toThrow(DataValidationError);
  });

  it("validates classWeight, minImpurityDecrease, maxLeafNodes and ccpAlpha", () => {
    // @ts-expect-error invalid class weight on purpose
    expect(() => new DecisionTreeClassifier({ classWeight: "balanced_subsample" })).toThrow(
      InvalidParameterError
    );
    // @ts-expect-error invalid class weight on purpose
    expect(() => new DecisionTreeClassifier({ classWeight: "bal" })).toThrow(InvalidParameterError);
    expect(() => new DecisionTreeClassifier({ classWeight: { 0: -1 } })).toThrow(
      InvalidParameterError
    );
    expect(() => new DecisionTreeClassifier({ classWeight: { 0: Number.NaN } })).toThrow(
      InvalidParameterError
    );
    expect(() => new DecisionTreeClassifier({ classWeight: { a: 1 } as never })).toThrow(
      InvalidParameterError
    );
    expect(() => new DecisionTreeClassifier({ minImpurityDecrease: -0.1 })).toThrow(
      InvalidParameterError
    );
    expect(() => new DecisionTreeClassifier({ minImpurityDecrease: Number.NaN })).toThrow(
      InvalidParameterError
    );
    expect(() => new DecisionTreeClassifier({ maxLeafNodes: 1 })).toThrow(InvalidParameterError);
    expect(() => new DecisionTreeClassifier({ maxLeafNodes: 2.5 })).toThrow(InvalidParameterError);
    expect(() => new DecisionTreeClassifier({ ccpAlpha: -1 })).toThrow(InvalidParameterError);
    expect(() => new DecisionTreeClassifier({ ccpAlpha: Number.POSITIVE_INFINITY })).toThrow(
      InvalidParameterError
    );
    // scikit-learn raises a ValueError for a class that is not in y
    const model = new DecisionTreeClassifier({ classWeight: { 0: 1, 7: 2 } });
    expect(() => model.fit(X, yc)).toThrow(InvalidParameterError);
  });

  it("exposes the new parameters through getParams, setParams and clone", () => {
    const model = new DecisionTreeClassifier({
      minImpurityDecrease: 0.1,
      maxLeafNodes: 6,
      ccpAlpha: 0.2,
      classWeight: { 1: 3 },
    });
    expect(model.getParams()).toMatchObject({
      minImpurityDecrease: 0.1,
      maxLeafNodes: 6,
      ccpAlpha: 0.2,
      classWeight: { 1: 3 },
    });
    const params = model.getParams();
    (params["classWeight"] as Record<number, number>)[1] = 99;
    expect(model.getParams()["classWeight"]).toEqual({ 1: 3 });
    const copy = model.clone();
    expect(copy.getParams()).toEqual(model.getParams());
    model.setParams({ maxLeafNodes: undefined, classWeight: "balanced", ccpAlpha: 0 });
    expect(model.getParams()).toMatchObject({
      maxLeafNodes: undefined,
      classWeight: "balanced",
      ccpAlpha: 0,
    });
    expect(() => model.setParams({ ccpAlpha: -1, maxLeafNodes: 4 })).toThrow(InvalidParameterError);
    expect(model.getParams()["maxLeafNodes"]).toBeUndefined();
  });

  it("keeps working without the new options (default behavior unchanged)", () => {
    const model = new DecisionTreeClassifier({ maxDepth: Number.POSITIVE_INFINITY }).fit(X, yc);
    expect(model.score(X, yc)).toBe(1);
    expect(new DecisionTreeClassifier().getParams()).toMatchObject({
      minImpurityDecrease: 0,
      ccpAlpha: 0,
    });
  });
});

// ---------------------------------------------------------------------------
// DecisionTreeRegressor
// ---------------------------------------------------------------------------

describe("DecisionTreeRegressor sample weights and growth limits", () => {
  it("matches scikit-learn with sampleWeight", () => {
    expectRegressor(new DecisionTreeRegressor(treeBase).fit(X, yr, sw), FIX.tr.sw);
  });

  it("minImpurityDecrease, maxLeafNodes and ccpAlpha match scikit-learn", () => {
    expectRegressor(
      new DecisionTreeRegressor({ ...treeBase, minImpurityDecrease: 0.02 }).fit(X, yr, sw),
      FIX.tr.mid
    );
    expectRegressor(
      new DecisionTreeRegressor({ ...treeBase, maxLeafNodes: 6 }).fit(X, yr, sw),
      FIX.tr.mln
    );
    expectRegressor(
      new DecisionTreeRegressor({ ...treeBase, ccpAlpha: FIX.tr.ccp.alpha }).fit(X, yr, sw),
      FIX.tr.ccp
    );
  });

  it("a zero weight drops a sample and weights act like repeated rows", () => {
    const keep = FIX.sw.map((_, i) => (i % 4 === 0 ? 0 : 1));
    const zeroed = new DecisionTreeRegressor({ minSamplesLeaf: 2 }).fit(X, yr, vec(keep));
    const dropped = new DecisionTreeRegressor({ minSamplesLeaf: 2 }).fit(
      mat(FIX.X.filter((_, i) => keep[i] === 1)),
      vec(FIX.yr.filter((_, i) => keep[i] === 1))
    );
    expectClose(flat(zeroed.predict(Xt)), flat(dropped.predict(Xt)), 1e-12);

    const counts = FIX.sw.map((_, i) => 1 + (i % 2));
    const rows: number[][] = [];
    const targets: number[] = [];
    FIX.X.forEach((row, i) => {
      for (let k = 0; k < (counts[i] as number); k++) {
        rows.push(row);
        targets.push(FIX.yr[i] as number);
      }
    });
    const weighted = new DecisionTreeRegressor().fit(X, yr, vec(counts));
    const repeated = new DecisionTreeRegressor().fit(mat(rows), vec(targets));
    expectClose(flat(weighted.predict(Xt)), flat(repeated.predict(Xt)), 1e-9);
  });

  it("a single leaf predicts the weighted mean", () => {
    const model = new DecisionTreeRegressor({ maxLeafNodes: 2, ccpAlpha: 1e9 }).fit(X, yr, sw);
    let num = 0;
    let den = 0;
    FIX.yr.forEach((value, i) => {
      num += value * (FIX.sw[i] as number);
      den += FIX.sw[i] as number;
    });
    expect(model.getNLeaves()).toBe(1);
    expect(flat(model.predict(Xt))[0]).toBeCloseTo(num / den, 12);
  });

  it("rejects invalid weights and options", () => {
    expect(() => new DecisionTreeRegressor().fit(X, yr, vec([1]))).toThrow(ShapeError);
    expect(() => new DecisionTreeRegressor().fit(X, yr, vec(FIX.sw.map(() => 0)))).toThrow(
      DataValidationError
    );
    expect(() => new DecisionTreeRegressor({ maxLeafNodes: 0 })).toThrow(InvalidParameterError);
    expect(() => new DecisionTreeRegressor({ ccpAlpha: -0.5 })).toThrow(InvalidParameterError);
    expect(() => new DecisionTreeRegressor({ minImpurityDecrease: -1 })).toThrow(
      InvalidParameterError
    );
    const model = new DecisionTreeRegressor({ maxLeafNodes: 3 });
    expect(model.getParams()).toMatchObject({ maxLeafNodes: 3, ccpAlpha: 0 });
    expect(model.clone().getParams()).toEqual(model.getParams());
    model.setParams({ ccpAlpha: 0.5, minImpurityDecrease: 0.1 });
    expect(model.getParams()).toMatchObject({ ccpAlpha: 0.5, minImpurityDecrease: 0.1 });
    expect(() => model.setParams({ ccpAlpha: -1 })).toThrow(InvalidParameterError);
  });

  it("handles constant targets and best-first growth on a pure node", () => {
    const flatY = vec(FIX.yr.map(() => 2));
    const model = new DecisionTreeRegressor({ maxLeafNodes: 4, ccpAlpha: 0.1 }).fit(X, flatY, sw);
    expect(model.getNLeaves()).toBe(1);
    expect(flat(model.predict(Xt))[0]).toBeCloseTo(2, 12);
  });
});

// ---------------------------------------------------------------------------
// Forests
// ---------------------------------------------------------------------------

describe("RandomForest and ExtraTrees weights and growth limits", () => {
  const forestBase = {
    nEstimators: 3,
    bootstrap: false,
    maxFeatures: null,
    minSamplesLeaf: 6,
    maxDepth: Number.POSITIVE_INFINITY,
    randomState: 0,
  } as const;

  it("RandomForestClassifier matches scikit-learn (balanced weights and sampleWeight)", () => {
    const model = new RandomForestClassifier({ ...forestBase, classWeight: "balanced" }).fit(
      X,
      yc,
      sw
    );
    expectClose(flat(model.predictProba(Xt)), FIX.rfc.proba.flat());
  });

  it("RandomForestClassifier passes the growth limits to the trees like scikit-learn", () => {
    const model = new RandomForestClassifier({
      ...forestBase,
      maxLeafNodes: 4,
      minImpurityDecrease: 0.005,
    }).fit(X, yc, sw);
    expectClose(flat(model.predictProba(Xt)), FIX.rfc_growth.proba.flat());
  });

  it("RandomForestRegressor matches scikit-learn with sampleWeight", () => {
    const model = new RandomForestRegressor(forestBase).fit(X, yr, sw);
    expectClose(flat(model.predict(Xt)), FIX.rfr.pred);
  });

  it("balanced_subsample equals balanced when there is no bootstrap", () => {
    const a = new RandomForestClassifier({ ...forestBase, classWeight: "balanced" }).fit(X, yc);
    const b = new RandomForestClassifier({ ...forestBase, classWeight: "balanced_subsample" }).fit(
      X,
      yc
    );
    expectClose(flat(a.predictProba(Xt)), flat(b.predictProba(Xt)), 1e-12);
    const e = new ExtraTreesClassifier({ nEstimators: 5, randomState: 3, classWeight: "balanced" });
    const f = new ExtraTreesClassifier({
      nEstimators: 5,
      randomState: 3,
      classWeight: "balanced_subsample",
    });
    expectClose(flat(e.fit(X, yc).predictProba(Xt)), flat(f.fit(X, yc).predictProba(Xt)), 1e-12);
  });

  it("balanced_subsample with bootstrap is reproducible and differs from no class weight", () => {
    const skewed = ivec(FIX.yc.map((label) => (label === 2 ? 1 : 0)));
    const options = { nEstimators: 12, maxDepth: 2, randomState: 5 } as const;
    const a = new RandomForestClassifier({ ...options, classWeight: "balanced_subsample" });
    const b = new RandomForestClassifier({ ...options, classWeight: "balanced_subsample" });
    const none = new RandomForestClassifier(options);
    expectClose(
      flat(a.fit(X, skewed).predictProba(X)),
      flat(b.fit(X, skewed).predictProba(X)),
      1e-12
    );
    const balanced = flat(a.predictProba(X));
    const plain = flat(none.fit(X, skewed).predictProba(X));
    // balancing raises the mean probability of the rare class
    const mean = (values: number[]) => values.reduce((s, v) => s + v, 0) / values.length;
    expect(mean(balanced.filter((_, i) => i % 2 === 1))).toBeGreaterThan(
      mean(plain.filter((_, i) => i % 2 === 1))
    );
  });

  it("sample weights shift bootstrap forests and zero-weight rows are ignored", () => {
    const rare = new RandomForestRegressor({ nEstimators: 8, maxDepth: 3, randomState: 2 });
    const heavy = vec(FIX.sw.map((_, i) => ((FIX.yr[i] as number) > 1 ? 50 : 1)));
    const light = vec(FIX.sw.map((_, i) => ((FIX.yr[i] as number) > 1 ? 1 : 50)));
    const meanOf = (weights: Tensor) =>
      flat(rare.fit(X, yr, weights).predict(X)).reduce((s, v) => s + v, 0) / FIX.yr.length;
    expect(meanOf(heavy)).toBeGreaterThan(meanOf(light));

    // rows with weight 0 never reach a tree: the same forest results from fitting on a copy of
    // the data in which they have a different target
    const keep = FIX.sw.map((_, i) => (i < 20 ? 0 : 1));
    const changed = vec(FIX.yr.map((v, i) => (i < 20 ? v + 1000 : v)));
    const a = new RandomForestRegressor({
      ...forestBase,
      nEstimators: 2,
    }).fit(X, yr, vec(keep));
    const b = new RandomForestRegressor({
      ...forestBase,
      nEstimators: 2,
    }).fit(X, changed, vec(keep));
    expectClose(flat(a.predict(Xt)), flat(b.predict(Xt)), 1e-9);
  });

  it("forest growth options validate, appear in getParams and survive clone", () => {
    const forest = new RandomForestClassifier({
      maxLeafNodes: 8,
      ccpAlpha: 0.01,
      minImpurityDecrease: 0.001,
      classWeight: "balanced_subsample",
    });
    expect(forest.getParams()).toMatchObject({
      maxLeafNodes: 8,
      ccpAlpha: 0.01,
      minImpurityDecrease: 0.001,
      classWeight: "balanced_subsample",
    });
    expect(forest.clone().getParams()).toEqual(forest.getParams());
    expect(() => new RandomForestClassifier({ maxLeafNodes: 1 })).toThrow(InvalidParameterError);
    expect(() => new RandomForestClassifier({ classWeight: "nope" as never })).toThrow(
      InvalidParameterError
    );
    expect(() => forest.setParams({ ccpAlpha: -1 })).toThrow(InvalidParameterError);
    expect(() => new RandomForestRegressor({ minImpurityDecrease: -1 })).toThrow(
      InvalidParameterError
    );
    expect(() =>
      new RandomForestClassifier({ classWeight: { 9: 2 }, nEstimators: 2 }).fit(X, yc)
    ).toThrow(InvalidParameterError);
    expect(() => new RandomForestRegressor().fit(X, yr, vec([1, 2]))).toThrow(ShapeError);
    expect(() => new RandomForestClassifier().fit(X, yc, vec(FIX.sw.map(() => 0)))).toThrow(
      DataValidationError
    );
  });

  it("RandomForestRegressor.setParams accepts and rejects the growth options", () => {
    const forest = new RandomForestRegressor();
    forest.setParams({ maxLeafNodes: 5, ccpAlpha: 0.2, minImpurityDecrease: 0.3 });
    expect(forest.getParams()).toMatchObject({
      maxLeafNodes: 5,
      ccpAlpha: 0.2,
      minImpurityDecrease: 0.3,
    });
    expect(() => forest.setParams({ maxLeafNodes: 1 })).toThrow(InvalidParameterError);
  });

  it("ExtraTrees: weights scale freely and heavy classes win", () => {
    const options = { nEstimators: 6, randomState: 4, maxDepth: 4 } as const;
    const a = new ExtraTreesClassifier(options).fit(X, yc, sw);
    const b = new ExtraTreesClassifier(options).fit(X, yc, vec(FIX.sw.map((w) => w * 7)));
    expectClose(flat(a.predictProba(Xt)), flat(b.predictProba(Xt)), 1e-9);

    const only0 = vec(FIX.yc.map((label) => (label === 0 ? 1000 : 0.001)));
    const meanClass0 = (proba: number[]) => {
      let total = 0;
      for (let i = 0; i < 8; i++) total += proba[i * 3] as number;
      return total / 8;
    };
    const heavy = meanClass0(
      flat(new ExtraTreesClassifier(options).fit(X, yc, only0).predictProba(Xt))
    );
    const plain = meanClass0(flat(new ExtraTreesClassifier(options).fit(X, yc).predictProba(Xt)));
    expect(heavy).toBeGreaterThan(plain + 0.2);
  });

  it("ExtraTrees: a stump, a pruned tree and a leaf-limited tree", () => {
    // minImpurityDecrease larger than any gain gives the weighted class prior
    const prior = new ExtraTreesClassifier({
      nEstimators: 3,
      randomState: 1,
      minImpurityDecrease: 10,
    }).fit(X, yc, sw);
    const weights = [0, 0, 0];
    let total = 0;
    FIX.yc.forEach((label, i) => {
      weights[label] = (weights[label] as number) + (FIX.sw[i] as number);
      total += FIX.sw[i] as number;
    });
    expectClose(
      flat(prior.predictProba(mat([FIX.Xt[0] as number[]]))),
      weights.map((w) => w / total),
      1e-12
    );

    const pruned = new ExtraTreesRegressor({ nEstimators: 3, randomState: 1, ccpAlpha: 1e9 }).fit(
      X,
      yr,
      sw
    );
    let num = 0;
    FIX.yr.forEach((value, i) => {
      num += value * (FIX.sw[i] as number);
    });
    expect(flat(pruned.predict(Xt))[0]).toBeCloseTo(num / total, 12);

    const limited = new ExtraTreesRegressor({
      nEstimators: 1,
      randomState: 1,
      maxLeafNodes: 2,
      maxDepth: 10,
    }).fit(X, yr);
    expect(new Set(flat(limited.predict(X))).size).toBe(2);
  });

  it("ExtraTrees: bootstrap with weights, parameters and validation", () => {
    const model = new ExtraTreesRegressor({
      nEstimators: 4,
      bootstrap: true,
      randomState: 9,
      minImpurityDecrease: 0.01,
      maxLeafNodes: 6,
      ccpAlpha: 0.001,
    }).fit(X, yr, sw);
    expect(model.getParams()).toMatchObject({
      minImpurityDecrease: 0.01,
      maxLeafNodes: 6,
      ccpAlpha: 0.001,
    });
    expect(model.clone().getParams()).toEqual(model.getParams());
    expect(flat(model.predict(Xt)).every(Number.isFinite)).toBe(true);
    const classifier = new ExtraTreesClassifier({
      nEstimators: 4,
      bootstrap: true,
      randomState: 9,
      classWeight: "balanced_subsample",
    }).fit(X, yc, sw);
    expect(classifier.getParams()["classWeight"]).toBe("balanced_subsample");
    expect(classifier.clone().getParams()).toEqual(classifier.getParams());
    expect(() => new ExtraTreesClassifier({ classWeight: "x" as never })).toThrow(
      InvalidParameterError
    );
    expect(() => new ExtraTreesClassifier().setParams({ classWeight: { 0: -1 } })).toThrow(
      InvalidParameterError
    );
    expect(() => new ExtraTreesRegressor({ maxLeafNodes: 1 })).toThrow(InvalidParameterError);
    expect(() => new ExtraTreesRegressor().setParams({ ccpAlpha: -1 })).toThrow(
      InvalidParameterError
    );
    expect(() => new ExtraTreesRegressor().fit(X, yr, vec([1]))).toThrow(ShapeError);
    expect(() => new ExtraTreesClassifier({ classWeight: { 5: 1 } }).fit(X, yc)).toThrow(
      InvalidParameterError
    );
    expect(() =>
      new ExtraTreesClassifier({ classWeight: { 0: 0, 1: 0, 2: 0 } }).fit(X, yc)
    ).toThrow(DataValidationError);
  });
});

// ---------------------------------------------------------------------------
// GradientBoosting
// ---------------------------------------------------------------------------

describe("GradientBoosting sample weights and tree growth options", () => {
  const regBase = {
    nEstimators: 6,
    maxDepth: 3,
    learningRate: 0.2,
    minSamplesLeaf: 6,
    randomState: 0,
  };

  it("matches scikit-learn with sampleWeight for the squared, huber and quantile losses", () => {
    for (const [loss, key] of [
      ["ls", "reg_squared_error"],
      ["huber", "reg_huber"],
      ["quantile", "reg_quantile"],
    ] as const) {
      const model = new GradientBoostingRegressor({ ...regBase, loss }).fit(X, yr, sw);
      expectClose(flat(model.predict(Xt)), FIX.gb[key], 1e-9);
    }
  });

  it("matches scikit-learn for every loss when each stage is a single leaf", () => {
    // minSamplesSplit above n_samples gives one-leaf trees, so only the weighted initial value,
    // leaf update and Huber transition point are tested (no split ties).
    const stump = { nEstimators: 5, learningRate: 0.5, minSamplesSplit: 1000, randomState: 0 };
    for (const [loss, key] of [
      ["ls", "stump_squared_error"],
      ["lad", "stump_absolute_error"],
      ["huber", "stump_huber"],
      ["quantile", "stump_quantile"],
    ] as const) {
      const model = new GradientBoostingRegressor({ ...stump, loss }).fit(X, yr, sw);
      expectClose(flat(model.predict(Xt)), FIX.gb[key], 1e-9);
    }
    const classifier = new GradientBoostingClassifier(stump).fit(X, yb, sw);
    const proba = (classifier.predictProba(Xt).toArray() as number[][]).map(
      (row) => row[1] as number
    );
    expectClose(proba, FIX.gb.stump_clf, 1e-9);
  });

  it("matches scikit-learn with tree growth options", () => {
    const model = new GradientBoostingRegressor({
      ...regBase,
      maxLeafNodes: 4,
      minImpurityDecrease: 0.5,
      ccpAlpha: 0.01,
    }).fit(X, yr, sw);
    expectClose(flat(model.predict(Xt)), FIX.gb.reg_growth, 1e-9);
  });

  it("matches scikit-learn for the binary classifier with sampleWeight", () => {
    const base = {
      nEstimators: 6,
      maxDepth: 3,
      learningRate: 0.2,
      minSamplesLeaf: 6,
      randomState: 0,
    };
    const model = new GradientBoostingClassifier(base).fit(X, yb, sw);
    const proba = (model.predictProba(Xt).toArray() as number[][]).map((row) => row[1] as number);
    expectClose(proba, FIX.gb.clf, 1e-9);
    const grown = new GradientBoostingClassifier({
      ...base,
      maxLeafNodes: 4,
      minImpurityDecrease: 0.5,
    }).fit(X, yb, sw);
    const grownProba = (grown.predictProba(Xt).toArray() as number[][]).map(
      (row) => row[1] as number
    );
    expectClose(grownProba, FIX.gb.clf_growth, 1e-9);
  });

  it("a zero weight is the same as removing the row", () => {
    const keep = FIX.sw.map((_, i) => (i % 5 === 0 ? 0 : 1));
    const weights = vec(FIX.sw.map((w, i) => w * (keep[i] as number)));
    const rows = FIX.X.filter((_, i) => keep[i] === 1);
    const kept = (values: readonly number[]) => values.filter((_, i) => keep[i] === 1);
    const kw = FIX.sw.filter((_, i) => keep[i] === 1);
    for (const loss of ["ls", "huber"] as const) {
      const a = new GradientBoostingRegressor({ ...regBase, loss }).fit(X, yr, weights);
      const b = new GradientBoostingRegressor({ ...regBase, loss }).fit(
        mat(rows),
        vec(kept(FIX.yr)),
        vec(kw)
      );
      expectClose(flat(a.predict(Xt)), flat(b.predict(Xt)), 1e-9);
    }
    const c = new GradientBoostingClassifier(regBase).fit(X, yb, weights);
    const d = new GradientBoostingClassifier(regBase).fit(mat(rows), ivec(kept(FIX.yb)), vec(kw));
    expectClose(flat(c.predictProba(Xt)), flat(d.predictProba(Xt)), 1e-9);
  });

  it("weights work with subsample, early stopping and warm start", () => {
    const model = new GradientBoostingRegressor({
      ...regBase,
      nEstimators: 30,
      subsample: 0.7,
      nIterNoChange: 3,
      validationFraction: 0.2,
      loss: "huber",
    }).fit(X, yr, sw);
    expect(model.nEstimatorsFitted).toBeGreaterThan(0);
    expect(flat(model.predict(Xt)).every(Number.isFinite)).toBe(true);
    const warm = new GradientBoostingRegressor({ ...regBase, nEstimators: 3, warmStart: true });
    warm.fit(X, yr, sw);
    warm.setParams({ nEstimators: 6 });
    warm.fit(X, yr, sw);
    const direct = new GradientBoostingRegressor(regBase).fit(X, yr, sw);
    expectClose(flat(warm.predict(Xt)), flat(direct.predict(Xt)), 1e-9);
  });

  it("validates weights and the new options", () => {
    expect(() => new GradientBoostingRegressor().fit(X, yr, vec([1, 2]))).toThrow(ShapeError);
    expect(() => new GradientBoostingRegressor().fit(X, yr, vec(FIX.sw.map(() => -1)))).toThrow(
      DataValidationError
    );
    expect(() => new GradientBoostingClassifier().fit(X, yb, vec([1]))).toThrow(ShapeError);
    // all positives carry zero weight
    expect(() =>
      new GradientBoostingClassifier().fit(
        X,
        yb,
        vec(FIX.sw.map((_, i) => (FIX.yb[i] === 1 ? 0 : 1)))
      )
    ).toThrow(DataValidationError);
    expect(() => new GradientBoostingRegressor({ maxLeafNodes: 1 })).toThrow(InvalidParameterError);
    expect(() => new GradientBoostingClassifier({ ccpAlpha: -1 })).toThrow(InvalidParameterError);
    const model = new GradientBoostingRegressor({ maxLeafNodes: 5, ccpAlpha: 0.1 });
    expect(model.getParams()).toMatchObject({
      maxLeafNodes: 5,
      ccpAlpha: 0.1,
      minImpurityDecrease: 0,
    });
    expect(model.clone().getParams()).toEqual(model.getParams());
    model.setParams({ minImpurityDecrease: 2, maxLeafNodes: undefined });
    expect(model.getParams()).toMatchObject({ minImpurityDecrease: 2, maxLeafNodes: undefined });
    const classifier = new GradientBoostingClassifier({ maxLeafNodes: 3 });
    classifier.setParams({ ccpAlpha: 0.3, minImpurityDecrease: 0.2, maxLeafNodes: 7 });
    expect(classifier.getParams()).toMatchObject({
      ccpAlpha: 0.3,
      minImpurityDecrease: 0.2,
      maxLeafNodes: 7,
    });
    expect(() => classifier.setParams({ maxLeafNodes: 0 })).toThrow(InvalidParameterError);
    expect(classifier.clone().getParams()).toEqual(classifier.getParams());
  });
});

// ---------------------------------------------------------------------------
// Ridge
// ---------------------------------------------------------------------------

describe("Ridge with alpha = 0 on rank-deficient data", () => {
  // numpy.linalg.lstsq on the centered data gives the minimum-norm solution.
  it("falls back to the minimum-norm least-squares solution (auto and cholesky)", () => {
    const Xr = mat(FIX.ridge.X);
    const yRidge = vec(FIX.ridge.y);
    for (const solver of ["auto", "cholesky", "svd"] as const) {
      const model = new Ridge({ alpha: 0, solver }).fit(Xr, yRidge);
      expectClose(flat(model.coef), FIX.ridge.coef, 1e-6);
      expect(model.intercept).toBeCloseTo(FIX.ridge.intercept, 6);
    }
  });

  // Rank 2 by construction, but rounding noise leaves X^T X with a positive pivot that is
  // about 1e-14 of the largest one, so the plain Cholesky factorization "succeeds" with
  // meaningless coefficients unless the pivot is checked against a relative tolerance.
  const NOISY = {
    X: [
      [0.8252355664065646, -0.7959325088778874, 0.5990268901434542],
      [-1.875697183126355, 1.8049171272404962, -1.3642444457570735],
      [0.09373029979380137, -0.10702913287849974, 0.057279474326145746],
      [-1.2623305808934104, 1.179856548518816, -0.9406684619076426],
      [-0.5029667199360411, 0.49444686248607966, -0.35905341061533047],
      [-0.7681770466526463, 0.6886105336834558, -0.591441226180012],
      [-0.29894620481283557, 0.3128616254829944, -0.2011290728236889],
      [1.3884927655908452, -1.307550804565484, 1.0283579857861467],
      [-0.9441420082313249, 0.9088220073001408, -0.6865006266595445],
      [-0.1710541379681839, 0.1260684418093837, -0.14934245064139592],
      [0.3265968281440772, -0.2880446351454325, 0.25451253974488935],
      [-0.3093280460196238, 0.31359286540014686, -0.2146707056489934],
      [-0.3610557335191036, 0.3560481006681909, -0.25703012415567783],
      [-1.1215455919756956, 1.0688394144812883, -0.8224487633243193],
      [-0.5844017339667101, 0.5155451706575818, -0.4553339751687261],
      [-0.9312562800889729, 0.8627835040961035, -0.6988935761445426],
      [-0.013013486435173858, 0.03231788653714894, 0.0033429868131305135],
      [1.1109103741498363, -0.982218638386139, 0.8641372406233303],
      [0.5802916020346202, -0.5766804135728119, 0.410229928872692],
      [-0.5344701167522785, 0.5244628215222528, -0.38215989897176533],
      [-0.5242981055243142, 0.5366920265201366, -0.36051586947803793],
      [0.3160864077373767, -0.3942017090161854, 0.17163857777383185],
      [1.1761906560460313, -1.1492113447687708, 0.8442136187266698],
      [-1.0656053371100862, 1.0054847509583948, -0.7879249982696249],
      [-0.725106050557609, 0.7429424742116855, -0.49814460574363356],
      [1.2396869357267877, -1.1860479931932029, 0.9060951445439032],
    ],
    y: [
      0.5770754169394714, 2.332712025382542, -0.3365255299285121, -0.9148298592004128,
      0.735687023006662, -0.3971572496742207, 0.28911616638337856, 0.13257652253025445,
      1.1683939613256054, -0.6568374851700608, -0.09227952885816246, -2.5521126492129294,
      -0.8901138962794292, 1.5363388115385548, -0.6815059146443607, -0.7014768178924555,
      -1.2179071819849998, 0.579834289685921, 0.17799016337176263, -1.1868668596304217,
      0.6674503740893216, 0.26010587984314715, 0.7540333794321797, -0.6414795084271917,
      1.7380087577355086, 0.40986760593441185,
    ],
    coef: [0.929067783736589, 4.007521317072018, 3.8471183914170797],
  };

  it("falls back to the SVD when X^T X is singular only up to rounding noise", () => {
    for (const solver of ["auto", "cholesky"] as const) {
      const model = new Ridge({ alpha: 0, solver }).fit(mat(NOISY.X), vec(NOISY.y));
      expectClose(flat(model.coef), NOISY.coef, 1e-6);
    }
  });

  it("handles more features than samples", () => {
    const model = new Ridge({ alpha: 0 }).fit(mat(FIX.ridge.wideX), vec(FIX.ridge.wideY));
    expectClose(flat(model.coef), FIX.ridge.wideCoef, 1e-8);
    expect(model.intercept).toBeCloseTo(FIX.ridge.wideIntercept, 8);
    const noIntercept = new Ridge({ alpha: 0, fitIntercept: false, solver: "cholesky" }).fit(
      mat(FIX.ridge.wideX),
      vec(FIX.ridge.wideY)
    );
    const predictions = flat(noIntercept.predict(mat(FIX.ridge.wideX)));
    expectClose(predictions, FIX.ridge.wideY, 1e-8);
  });

  it("leaves well-posed problems on the direct solvers", () => {
    const well = new Ridge({ alpha: 0.5 }).fit(X, yr);
    const viaSvd = new Ridge({ alpha: 0.5, solver: "svd" }).fit(X, yr);
    expectClose(flat(well.coef), flat(viaSvd.coef), 1e-9);
    expect(well.intercept).toBeCloseTo(viaSvd.intercept, 9);
    const exact = new Ridge({ alpha: 0 }).fit(X, yr);
    const lstsq = new Ridge({ alpha: 0, solver: "svd" }).fit(X, yr);
    expectClose(flat(exact.coef), flat(lstsq.coef), 1e-8);
  });

  it("still validates its options", () => {
    expect(() => new Ridge({ alpha: -1 }).fit(X, yr)).toThrow(InvalidParameterError);
  });
});

// ---------------------------------------------------------------------------
// Text datasets
// ---------------------------------------------------------------------------

type TarEntry = { name: string; data?: string | Uint8Array; type?: "0" | "5" | "L" };

function tarHeader(name: string, size: number, type: string): Uint8Array {
  const header = new Uint8Array(512);
  const put = (text: string, at: number) => {
    for (let i = 0; i < text.length; i++) header[at + i] = text.charCodeAt(i);
  };
  put(name.slice(0, 100), 0);
  put("0000644\0", 100);
  put("0000000\0", 108);
  put("0000000\0", 116);
  put(`${size.toString(8).padStart(11, "0")}\0`, 124);
  put("00000000000\0", 136);
  put("        ", 148);
  put(type, 156);
  put("ustar  \0", 257);
  let checksum = 0;
  for (const byte of header) checksum += byte;
  put(`${checksum.toString(8).padStart(6, "0")}\0 `, 148);
  return header;
}

function makeTarGz(entries: readonly TarEntry[]): Uint8Array<ArrayBuffer> {
  const parts: Uint8Array[] = [];
  const push = (name: string, bytes: Uint8Array, type: string) => {
    parts.push(tarHeader(name, bytes.length, type));
    parts.push(bytes);
    const pad = (512 - (bytes.length % 512)) % 512;
    if (pad > 0) parts.push(new Uint8Array(pad));
  };
  for (const entry of entries) {
    const bytes =
      typeof entry.data === "string"
        ? new TextEncoder().encode(entry.data)
        : (entry.data ?? new Uint8Array(0));
    if (entry.name.length > 100) {
      push("././@LongLink", new TextEncoder().encode(`${entry.name}\0`), "L");
      push(entry.name.slice(0, 100), bytes, entry.type ?? "0");
    } else {
      push(entry.name, bytes, entry.type ?? "0");
    }
  }
  parts.push(new Uint8Array(1024));
  const total = new Uint8Array(parts.reduce((s, p) => s + p.length, 0));
  let offset = 0;
  for (const part of parts) {
    total.set(part, offset);
    offset += part.length;
  }
  return new Uint8Array(gzipSync(total));
}

function stubArchive(bytes: Uint8Array<ArrayBuffer>, calls: string[] = []): string[] {
  vi.stubGlobal("fetch", async (url: string) => {
    calls.push(url);
    return new Response(bytes, { status: 200 });
  });
  return calls;
}

const latin1 = (text: string): Uint8Array => Uint8Array.from(text, (ch) => ch.charCodeAt(0));

function newsgroupsArchive(): Uint8Array<ArrayBuffer> {
  const entries: TarEntry[] = [
    { name: "20news-bydate-test/", type: "5" },
    { name: "20news-bydate-test/sci.space/", type: "5" },
  ];
  // Written in a shuffled order on purpose: the loader sorts groups and file names. File names
  // sort as strings, like scikit-learn's load_files: "10" comes before "2".
  const docs: Array<[string, string, string, string | Uint8Array]> = [
    ["train", "sci.space", "2", "orbit 2"],
    ["train", "alt.atheism", "7", "plain text"],
    ["train", "sci.space", "10", "orbit 10"],
    ["train", "alt.atheism", "3", latin1("caf\xe9 \x93quoted\x94")],
    ["test", "sci.space", "5", "test orbit"],
    ["test", "alt.atheism", "5", "test atheism"],
    ["train", "comp.graphics", "1", "pixels"],
  ];
  for (const [split, group, name, data] of docs) {
    entries.push({ name: `20news-bydate-${split}/${group}/${name}`, data });
  }
  return makeTarGz(entries);
}

describe("fetch20Newsgroups with the official archive", () => {
  it("downloads the figshare archive by default and parses it like scikit-learn", async () => {
    const calls = stubArchive(newsgroupsArchive());
    const news = await fetch20Newsgroups({ subset: "train" });
    expect(calls).toEqual(["https://ndownloader.figshare.com/files/5975967"]);
    expect(news.isSynthetic).toBe(false);
    expect(news.classNames).toEqual(["alt.atheism", "comp.graphics", "sci.space"]);
    expect(news.nClasses).toBe(3);
    expect(news.target.dtype).toBe("int32");
    expect(Array.from(news.target.data as Int32Array)).toEqual([0, 0, 1, 2, 2]);
    // within a group the file names sort as strings: "10" < "2"
    expect(news.texts).toEqual([
      "café \u0093quoted\u0094",
      "plain text",
      "pixels",
      "orbit 10",
      "orbit 2",
    ]);
    expect(news.description).toContain("20 Newsgroups");
  });

  it("ignores AppleDouble metadata entries written by macOS tar", async () => {
    stubArchive(
      makeTarGz([
        { name: "20news-bydate-train/alt.atheism/1", data: "doc" },
        { name: "20news-bydate-train/alt.atheism/._1", data: "\u0000\u0005\u0016\u0007" },
        { name: "20news-bydate-train/._alt.atheism", data: "\u0000\u0005\u0016\u0007" },
      ])
    );
    const news = await fetch20Newsgroups({ subset: "train" });
    expect(news.classNames).toEqual(["alt.atheism"]);
    expect(news.texts).toEqual(["doc"]);
  });

  it("supports the test and all subsets", async () => {
    stubArchive(newsgroupsArchive());
    const test = await fetch20Newsgroups({ subset: "test" });
    expect(test.texts).toEqual(["test atheism", "test orbit"]);
    expect(Array.from(test.target.data as Int32Array)).toEqual([0, 2]);
    const all = await fetch20Newsgroups();
    expect(all.texts).toHaveLength(7);
    expect(all.texts.slice(-2)).toEqual(["test atheism", "test orbit"]);
  });

  it("takes the same number of posts from every class with maxSamples", async () => {
    stubArchive(newsgroupsArchive());
    const some = await fetch20Newsgroups({ subset: "train", maxSamples: 3 });
    expect(Array.from(some.target.data as Int32Array)).toEqual([0, 1, 2]);
    const more = await fetch20Newsgroups({ subset: "train", maxSamples: 4 });
    expect(Array.from(more.target.data as Int32Array)).toEqual([0, 0, 1, 2]);
    const everything = await fetch20Newsgroups({ subset: "train", maxSamples: 1000 });
    expect(everything.texts).toHaveLength(5);
  });

  it("reads another archive location with archiveUrl", async () => {
    const calls = stubArchive(newsgroupsArchive());
    await fetch20Newsgroups({ archiveUrl: "https://mirror.example/20news.tar.gz" });
    expect(calls).toEqual(["https://mirror.example/20news.tar.gz"]);
  });

  it("still reads a JSON mirror when baseUrl is given", async () => {
    const calls: string[] = [];
    vi.stubGlobal("fetch", async (url: string) => {
      calls.push(url);
      return new Response(
        JSON.stringify({ data: ["a", "b"], target: [0, 1], target_names: ["x", "y"] }),
        { status: 200 }
      );
    });
    const news = await fetch20Newsgroups({ baseUrl: "https://host/dir", subset: "test" });
    expect(calls).toEqual(["https://host/dir/20newsgroups_test.json"]);
    expect(news.classNames).toEqual(["x", "y"]);
  });

  it("rejects baseUrl together with archiveUrl and empty values", async () => {
    await expect(
      fetch20Newsgroups({ baseUrl: "https://a/", archiveUrl: "https://b/x.tar.gz" })
    ).rejects.toThrow(InvalidParameterError);
    await expect(fetch20Newsgroups({ archiveUrl: "" })).rejects.toThrow(InvalidParameterError);
  });

  it("explains failures and can fall back to synthetic data", async () => {
    vi.stubGlobal(
      "fetch",
      async () => new Response("gone", { status: 404, statusText: "Not Found" })
    );
    await expect(fetch20Newsgroups()).rejects.toThrow(
      /Failed to fetch 20 Newsgroups dataset from https:\/\/ndownloader\.figshare\.com\/files\/5975967: HTTP 404 Not Found/
    );
    const fallback = await fetch20Newsgroups({ allowSyntheticFallback: true, maxSamples: 20 });
    expect(fallback.isSynthetic).toBe(true);
    expect(fallback.texts).toHaveLength(20);
  });

  it("rejects archives that are not gzip or have no newsgroup folders", async () => {
    stubArchive(new Uint8Array([1, 2, 3, 4]));
    await expect(fetch20Newsgroups()).rejects.toThrow(DeepboxError);
    stubArchive(makeTarGz([{ name: "other/file.txt", data: "x" }]));
    await expect(fetch20Newsgroups()).rejects.toThrow(/20news-bydate/);
    stubArchive(makeTarGz([{ name: "20news-bydate-train/a/1", data: "x" }]));
    await expect(fetch20Newsgroups({ subset: "test" })).rejects.toThrow(/no documents/);
  });

  it("finds entries that carry GNU long names", async () => {
    const longGroup = `${"g".repeat(60)}.group`;
    stubArchive(
      makeTarGz([{ name: `20news-bydate-train/${longGroup}/${"1".repeat(40)}`, data: "long" }])
    );
    const news = await fetch20Newsgroups({ subset: "train" });
    expect(news.classNames).toEqual([longGroup]);
    expect(news.texts).toEqual(["long"]);
  });
});

function imdbArchive(): Uint8Array<ArrayBuffer> {
  return makeTarGz([
    { name: "aclImdb/", type: "5" },
    { name: "aclImdb/train/pos/2_8.txt", data: "great <br /> film é" },
    { name: "aclImdb/train/pos/1_9.txt", data: "loved it" },
    { name: "aclImdb/train/neg/0_1.txt", data: "awful" },
    { name: "aclImdb/train/neg/3_2.txt", data: "bad" },
    { name: "aclImdb/train/unsup/0_0.txt", data: "unlabeled" },
    { name: "aclImdb/test/neg/0_3.txt", data: "test neg" },
    { name: "aclImdb/test/pos/0_7.txt", data: "test pos" },
    { name: "aclImdb/train/urls_pos.txt", data: "http://example" },
    { name: "aclImdb/imdb.vocab", data: "word" },
  ]);
}

describe("fetchIMDB with the official archive", () => {
  it("downloads the Stanford archive by default and labels neg 0 and pos 1", async () => {
    const calls = stubArchive(imdbArchive());
    const imdb = await fetchIMDB({ subset: "train" });
    expect(calls).toEqual(["https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"]);
    expect(imdb.isSynthetic).toBe(false);
    expect(imdb.classNames).toEqual(["negative", "positive"]);
    expect(imdb.target.dtype).toBe("int32");
    expect(imdb.texts).toEqual(["awful", "bad", "loved it", "great <br /> film é"]);
    expect(Array.from(imdb.target.data as Int32Array)).toEqual([0, 0, 1, 1]);
    expect(imdb.texts).not.toContain("unlabeled");
  });

  it("supports test and all subsets and a class balanced maxSamples", async () => {
    stubArchive(imdbArchive());
    const test = await fetchIMDB({ subset: "test" });
    expect(test.texts).toEqual(["test neg", "test pos"]);
    const all = await fetchIMDB();
    expect(all.texts).toHaveLength(6);
    expect(Array.from(all.target.data as Int32Array)).toEqual([0, 0, 1, 1, 0, 1]);
    const few = await fetchIMDB({ maxSamples: 2 });
    expect(Array.from(few.target.data as Int32Array)).toEqual([0, 1]);
    expect(few.texts).toEqual(["awful", "loved it"]);
  });

  it("honors archiveUrl, baseUrl and the synthetic fallback", async () => {
    const calls = stubArchive(imdbArchive());
    await fetchIMDB({ archiveUrl: "https://mirror.example/imdb.tar.gz" });
    expect(calls).toEqual(["https://mirror.example/imdb.tar.gz"]);

    vi.stubGlobal("fetch", async (url: string) => {
      calls.push(url);
      return new Response(JSON.stringify({ data: ["a"], target: [1] }), { status: 200 });
    });
    const json = await fetchIMDB({ baseUrl: "https://host" });
    expect(calls.at(-1)).toBe("https://host/imdb_all.json");
    expect(Array.from(json.target.data as Int32Array)).toEqual([1]);

    vi.stubGlobal("fetch", async () => new Response("", { status: 404, statusText: "Not Found" }));
    await expect(fetchIMDB()).rejects.toThrow(/HTTP 404 Not Found/);
    const fallback = await fetchIMDB({ allowSyntheticFallback: true, maxSamples: 4 });
    expect(fallback.isSynthetic).toBe(true);
    expect(fallback.texts).toHaveLength(4);
  });

  it("rejects archives without reviews", async () => {
    stubArchive(makeTarGz([{ name: "aclImdb/train/unsup/0_0.txt", data: "x" }]));
    await expect(fetchIMDB()).rejects.toThrow(/aclImdb/);
    await expect(fetchIMDB({ timeout: -5 })).rejects.toThrow(InvalidParameterError);
  });

  it("passes a timeout signal to the archive download", async () => {
    const seen: Array<AbortSignal | undefined> = [];
    vi.stubGlobal("fetch", async (_url: string, init?: { signal?: AbortSignal }) => {
      seen.push(init?.signal);
      return new Response(imdbArchive(), { status: 200 });
    });
    await fetchIMDB({ timeout: 5000 });
    await fetchIMDB({ timeout: 0 });
    expect(seen[0]).toBeDefined();
    expect(seen[1]).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// plot
// ---------------------------------------------------------------------------

describe("calculateWhiskers keeps the whiskers outside the box", () => {
  // matplotlib.cbook.boxplot_stats gives whislo / whishi
  it("matches matplotlib on uneven data", () => {
    for (const [data, lo, hi] of [
      [[1, 2, 3, 100], 1, 27.25],
      [[1, 2, 3, 4, 5, 100], 1, 5],
      [[0.1, 0.2, 50, 51, 52, 1000], 0.1, 52],
    ] as const) {
      const { q1, q3 } = calculateQuartiles(data);
      const whiskers = calculateWhiskers(data, q1, q3);
      expect(whiskers.lowerWhisker).toBeCloseTo(lo, 12);
      expect(whiskers.upperWhisker).toBeCloseTo(hi, 12);
    }
  });

  it("clamps both ends and collapses when everything is an outlier", () => {
    const low = calculateWhiskers([-1000, 1000, 1001, 1002, 1003], 1000.5, 1002.5);
    expect(low.lowerWhisker).toBeLessThanOrEqual(1000.5);
    expect(low.upperWhisker).toBeGreaterThanOrEqual(1002.5);
    expect(low.outliers).toEqual([-1000]);
    const none = calculateWhiskers([0, 100], 10, 11);
    expect([none.lowerWhisker, none.upperWhisker]).toEqual([10, 11]);
    expect(calculateWhiskers([], 0, 0)).toEqual({ lowerWhisker: 0, upperWhisker: 0, outliers: [] });
    // NaN values are ignored
    const withNaN = calculateWhiskers([1, 2, 3, Number.NaN], 1.5, 2.5);
    expect(withNaN.outliers).toEqual([]);
  });

  it("holds lower <= q1 <= q3 <= upper for random data", () => {
    let state = 12345;
    const next = () => {
      state = (state * 1664525 + 1013904223) % 4294967296;
      return state / 4294967296;
    };
    for (let round = 0; round < 50; round++) {
      const n = 1 + Math.floor(next() * 12);
      const data = Array.from({ length: n }, () => Math.round(next() * 1000) / 10).sort(
        (a, b) => a - b
      );
      if (round % 3 === 0) data.push(1e5);
      const { q1, q3 } = calculateQuartiles(data);
      const { lowerWhisker, upperWhisker } = calculateWhiskers(data, q1, q3);
      expect(lowerWhisker).toBeLessThanOrEqual(q1);
      expect(upperWhisker).toBeGreaterThanOrEqual(q3);
    }
  });
});
