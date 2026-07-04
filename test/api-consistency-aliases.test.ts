import { describe, expect, it } from "vitest";
import * as ndarray from "../src/ndarray";
import {
  atleast_1d,
  atleast1d,
  broadcast_to,
  broadcastTo,
  column_stack,
  columnStack,
  empty_like,
  emptyLike,
  flipLr,
  fliplr,
  flipUd,
  flipud,
  full_like,
  fullLike,
  index_select,
  indexSelect,
  ones_like,
  onesLike,
  tensor,
  zeros_like,
  zerosLike,
} from "../src/ndarray";
import * as nn from "../src/nn";
import {
  clip_grad_norm_,
  clip_grad_value_,
  clipGradNorm,
  clipGradNorm_,
  clipGradValue,
  clipGradValue_,
  kaiming_normal_,
  kaiming_uniform_,
  kaimingNormal,
  kaimingNormal_,
  kaimingUniform,
  kaimingUniform_,
  orthogonal,
  orthogonal_,
  xavier_normal_,
  xavier_uniform_,
  xavierNormal,
  xavierNormal_,
  xavierUniform,
  xavierUniform_,
} from "../src/nn";
import {
  f_classif,
  f_regression,
  fClassif,
  fRegression,
  mutual_info_classif,
  mutual_info_regression,
  mutualInfoClassif,
  mutualInfoRegression,
} from "../src/preprocess";
import {
  chi2_contingency,
  chi2Contingency,
  f_oneway,
  fisher_exact,
  fisherExact,
  fOneway,
  ks_2samp,
  ks2samp,
  kurtosis,
  mean,
  skewness,
  std,
  ttest_ind,
  ttest_rel,
  ttestInd,
  ttestRel,
  variance,
} from "../src/stats";

describe("stats reductions: options-object overloads", () => {
  const t = tensor([
    [1, 2, 3],
    [4, 5, 6],
  ]);

  it("mean accepts an options object equivalent to positional", () => {
    const a = mean(t, 1, true);
    const b = mean(t, { axis: 1, keepdims: true });
    expect(b.shape).toEqual(a.shape);
    expect(Array.from(b.data as Float64Array)).toEqual(Array.from(a.data as Float64Array));
    // Positional still works unchanged
    expect(Array.from(mean(t).data as Float64Array)).toEqual(
      Array.from(mean(t, {}).data as Float64Array)
    );
  });

  it("std/variance options object routes ddof + keepdims", () => {
    const v1 = std(tensor([1, 2, 3, 4, 5]), 0, false, 1);
    const v2 = std(tensor([1, 2, 3, 4, 5]), { ddof: 1 });
    expect((v2.data as Float64Array)[0]).toBeCloseTo((v1.data as Float64Array)[0] as number, 12);

    const w1 = variance(t, 1, true, 1);
    const w2 = variance(t, { axis: 1, keepdims: true, ddof: 1 });
    expect(w2.shape).toEqual(w1.shape);
    expect(Array.from(w2.data as Float64Array)).toEqual(Array.from(w1.data as Float64Array));
  });

  it("skewness/kurtosis options object routes bias/fisher", () => {
    const data = tensor([1, 2, 2, 3, 3, 3, 4, 4, 4, 4]);
    const s1 = skewness(data, undefined, false);
    const s2 = skewness(data, { bias: false });
    expect((s2.data as Float64Array)[0]).toBeCloseTo((s1.data as Float64Array)[0] as number, 12);

    const k1 = kurtosis(data, undefined, false, true);
    const k2 = kurtosis(data, { fisher: false, bias: true });
    expect((k2.data as Float64Array)[0]).toBeCloseTo((k1.data as Float64Array)[0] as number, 12);
  });

  it("options object defaults match positional defaults", () => {
    // bias defaults to true, keepdims to false, ddof to 0
    const a = skewness(tensor([1, 2, 3, 4, 5]));
    const b = skewness(tensor([1, 2, 3, 4, 5]), {});
    expect((b.data as Float64Array)[0]).toBeCloseTo((a.data as Float64Array)[0] as number, 12);
  });
});

describe("naming aliases refer to the same function", () => {
  it("ndarray camelCase aliases are identical references", () => {
    expect(broadcastTo).toBe(broadcast_to);
    expect(columnStack).toBe(column_stack);
    expect(atleast1d).toBe(atleast_1d);
    expect(zerosLike).toBe(zeros_like);
    expect(onesLike).toBe(ones_like);
    expect(emptyLike).toBe(empty_like);
    expect(fullLike).toBe(full_like);
    expect(indexSelect).toBe(index_select);
    expect(flipLr).toBe(fliplr);
    expect(flipUd).toBe(flipud);
    // Also reachable off the module namespace
    expect(ndarray.broadcastTo).toBe(ndarray.broadcast_to);
  });

  it("stats/preprocess camelCase aliases are identical references", () => {
    expect(ttestInd).toBe(ttest_ind);
    expect(ttestRel).toBe(ttest_rel);
    expect(fOneway).toBe(f_oneway);
    expect(chi2Contingency).toBe(chi2_contingency);
    expect(ks2samp).toBe(ks_2samp);
    expect(fisherExact).toBe(fisher_exact);
    expect(fClassif).toBe(f_classif);
    expect(fRegression).toBe(f_regression);
    expect(mutualInfoClassif).toBe(mutual_info_classif);
    expect(mutualInfoRegression).toBe(mutual_info_regression);
  });

  it("nn camelCase aliases are identical references", () => {
    expect(kaimingNormal_).toBe(kaiming_normal_);
    expect(kaimingNormal).toBe(kaiming_normal_);
    expect(kaimingUniform_).toBe(kaiming_uniform_);
    expect(kaimingUniform).toBe(kaiming_uniform_);
    expect(xavierNormal_).toBe(xavier_normal_);
    expect(xavierNormal).toBe(xavier_normal_);
    expect(xavierUniform_).toBe(xavier_uniform_);
    expect(xavierUniform).toBe(xavier_uniform_);
    expect(orthogonal).toBe(orthogonal_);
    expect(clipGradNorm_).toBe(clip_grad_norm_);
    expect(clipGradNorm).toBe(clip_grad_norm_);
    expect(clipGradValue_).toBe(clip_grad_value_);
    expect(clipGradValue).toBe(clip_grad_value_);
    // Single-word init aliases live on the module namespace
    expect(nn.zeros).toBe(nn.zeros_);
    expect(nn.ones).toBe(nn.ones_);
    expect(nn.constant).toBe(nn.constant_);
  });
});

describe("aliases produce identical results", () => {
  it("broadcastTo === broadcast_to output", () => {
    const t = tensor([1, 2, 3]);
    const a = broadcast_to(t, [2, 3]);
    const b = broadcastTo(t, [2, 3]);
    expect(b.shape).toEqual(a.shape);
    expect(Array.from(b.data as Float64Array)).toEqual(Array.from(a.data as Float64Array));
  });

  it("zerosLike fills the same values", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(Array.from(zerosLike(t).data as Float64Array)).toEqual(
      Array.from(zeros_like(t).data as Float64Array)
    );
  });
});
