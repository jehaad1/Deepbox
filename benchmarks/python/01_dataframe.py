"""
Benchmark 01 — DataFrame Operations
Pandas
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from utils import run, create_suite, header, footer

suite = create_suite("dataframe", "Pandas")
header("Benchmark 01 — DataFrame Operations", "Pandas")

# ── Data generators ──────────────────────────────────────

def make_mixed_df(n, seed):
    rng = np.random.RandomState(seed)
    cats = ["A", "B", "C", "D", "E"]
    return pd.DataFrame({
        "name": [f"person_{i}" for i in range(n)],
        "age": rng.randint(18, 78, n),
        "score": rng.rand(n) * 100,
        "category": [cats[i % 5] for i in rng.randint(0, 5, n)],
        "value": rng.rand(n) * 1000,
    })

def make_numeric_df(n, ncols, seed):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({f"col{c}": rng.rand(n) * 100 for c in range(ncols)})

df100 = make_mixed_df(100, 42)
df1k = make_mixed_df(1000, 42)
df10k = make_mixed_df(10000, 42)
df50k = make_mixed_df(50000, 42)
numDf100 = make_numeric_df(100, 5, 42)
numDf1k = make_numeric_df(1000, 5, 42)
numDf10k = make_numeric_df(10000, 5, 42)

# ── Creation ─────────────────────────────────────────────

run(suite, "create", "100 rows", lambda: make_mixed_df(100, 99))
run(suite, "create", "1K rows", lambda: make_mixed_df(1000, 99))
run(suite, "create", "10K rows", lambda: make_mixed_df(10000, 99))
run(suite, "create (numeric)", "100x5", lambda: make_numeric_df(100, 5, 99))
run(suite, "create (numeric)", "1Kx5", lambda: make_numeric_df(1000, 5, 99))
run(suite, "create (numeric)", "10Kx5", lambda: make_numeric_df(10000, 5, 99))

# ── Select ───────────────────────────────────────────────

run(suite, "select (1 col)", "1K rows", lambda: df1k[["score"]])
run(suite, "select (1 col)", "10K rows", lambda: df10k[["score"]])
run(suite, "select (3 cols)", "1K rows", lambda: df1k[["name", "age", "score"]])
run(suite, "select (3 cols)", "10K rows", lambda: df10k[["name", "age", "score"]])

# ── Filter ───────────────────────────────────────────────

run(suite, "filter (numeric >)", "1K rows", lambda: df1k[df1k["age"] > 40])
run(suite, "filter (numeric >)", "10K rows", lambda: df10k[df10k["age"] > 40])
run(suite, "filter (string ==)", "1K rows", lambda: df1k[df1k["category"] == "A"])
run(suite, "filter (string ==)", "10K rows", lambda: df10k[df10k["category"] == "A"])
run(suite, "filter (compound)", "1K rows", lambda: df1k[(df1k["age"] > 30) & (df1k["score"] > 50)])
run(suite, "filter (compound)", "10K rows", lambda: df10k[(df10k["age"] > 30) & (df10k["score"] > 50)])

# ── Sort ─────────────────────────────────────────────────

run(suite, "sort (single col)", "1K rows", lambda: df1k.sort_values("age"))
run(suite, "sort (single col)", "10K rows", lambda: df10k.sort_values("age"))
run(suite, "sort (descending)", "1K rows", lambda: df1k.sort_values("score", ascending=False))
run(suite, "sort (descending)", "10K rows", lambda: df10k.sort_values("score", ascending=False))

# ── GroupBy ──────────────────────────────────────────────

run(suite, "groupBy + sum", "1K rows", lambda: df1k.groupby("category").sum(numeric_only=True))
run(suite, "groupBy + sum", "10K rows", lambda: df10k.groupby("category").sum(numeric_only=True))
run(suite, "groupBy + mean", "1K rows", lambda: df1k.groupby("category").mean(numeric_only=True))
run(suite, "groupBy + mean", "10K rows", lambda: df10k.groupby("category").mean(numeric_only=True))
run(suite, "groupBy + count", "1K rows", lambda: df1k.groupby("category").count())
run(suite, "groupBy + count", "10K rows", lambda: df10k.groupby("category").count())
run(suite, "groupBy + min", "1K rows", lambda: df1k.groupby("category").min(numeric_only=True))
run(suite, "groupBy + min", "10K rows", lambda: df10k.groupby("category").min(numeric_only=True))
run(suite, "groupBy + max", "1K rows", lambda: df1k.groupby("category").max(numeric_only=True))
run(suite, "groupBy + max", "10K rows", lambda: df10k.groupby("category").max(numeric_only=True))

# ── Head / Tail ──────────────────────────────────────────

run(suite, "head(10)", "10K rows", lambda: df10k.head(10))
run(suite, "head(10)", "50K rows", lambda: df50k.head(10))
run(suite, "tail(10)", "10K rows", lambda: df10k.tail(10))
run(suite, "tail(10)", "50K rows", lambda: df50k.tail(10))

# ── Loc / Iloc ───────────────────────────────────────────

run(suite, "iloc(0)", "1K rows", lambda: df1k.iloc[0])
run(suite, "iloc(500)", "1K rows", lambda: df1k.iloc[500])
run(suite, "iloc(0)", "10K rows", lambda: df10k.iloc[0])
run(suite, "iloc(5000)", "10K rows", lambda: df10k.iloc[5000])

# ── Join ─────────────────────────────────────────────────

left_df = pd.DataFrame({"key": [i % 50 for i in range(500)], "val_l": list(range(500))})
right_df = pd.DataFrame({"key": list(range(50)), "val_r": [i * 10 for i in range(50)]})
left_df2k = pd.DataFrame({"key": [i % 100 for i in range(2000)], "val_l": list(range(2000))})
right_df100 = pd.DataFrame({"key": list(range(100)), "val_r": [i * 10 for i in range(100)]})

run(suite, "join (inner)", "500×50", lambda: left_df.merge(right_df, on="key", how="inner"))
run(suite, "join (left)", "500×50", lambda: left_df.merge(right_df, on="key", how="left"))
run(suite, "join (inner)", "2K×100", lambda: left_df2k.merge(right_df100, on="key", how="inner"))
run(suite, "join (left)", "2K×100", lambda: left_df2k.merge(right_df100, on="key", how="left"))

# ── Concat ───────────────────────────────────────────────

run(suite, "concat (axis=0)", "2×1K", lambda: pd.concat([df1k, df1k], axis=0, ignore_index=True))
run(suite, "concat (axis=0)", "2×10K", lambda: pd.concat([df10k, df10k], axis=0, ignore_index=True))

# ── FillNa / DropNa ─────────────────────────────────────

dfna1k = pd.DataFrame({
    "a": [None if i % 5 == 0 else i for i in range(1000)],
    "b": [None if i % 3 == 0 else i * 2 for i in range(1000)],
})
dfna10k = pd.DataFrame({
    "a": [None if i % 5 == 0 else i for i in range(10000)],
    "b": [None if i % 3 == 0 else i * 2 for i in range(10000)],
})

run(suite, "fillna(0)", "1K rows", lambda: dfna1k.fillna(0))
run(suite, "fillna(0)", "10K rows", lambda: dfna10k.fillna(0))
run(suite, "dropna", "1K rows", lambda: dfna1k.dropna())
run(suite, "dropna", "10K rows", lambda: dfna10k.dropna())

# ── Describe ─────────────────────────────────────────────

run(suite, "describe", "100x5", lambda: numDf100.describe())
run(suite, "describe", "1Kx5", lambda: numDf1k.describe())
run(suite, "describe", "10Kx5", lambda: numDf10k.describe())

# ── Correlation ──────────────────────────────────────────

run(suite, "corr", "100x5", lambda: numDf100.corr())
run(suite, "corr", "1Kx5", lambda: numDf1k.corr())

# ── Drop ─────────────────────────────────────────────────

run(suite, "drop (1 col)", "1K rows", lambda: df1k.drop(columns=["value"]))
run(suite, "drop (2 cols)", "10K rows", lambda: df10k.drop(columns=["value", "name"]))

footer(suite, "pandas-dataframe.json")
