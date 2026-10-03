# Deepbox Examples

> **Browse online:** https://deepbox.dev/examples · **Docs:** https://deepbox.dev/docs

This directory contains **50 self-contained examples (00-49)** for Deepbox 1.5.0. They run from tensor basics to runtime tooling, inference workflows, model selection, FFTs and DataFrame workflows. Each example is a single `index.ts` with a `README.md` that explains what it shows.

## Conventions Used in the Examples

- **Training with plain tensors.** Data is a plain `tensor(...)`. When gradient mode is on and a module has trainable parameters, `model.forward(x)` returns a `GradTensor` that tracks the weights, so `loss.backward()` works without wrapping the data. `parameter(...)` is used only for tensors you want gradients for, and `noGrad(() => ...)` turns tracking off for evaluation. See examples 13 and 14.
- **Fluent methods.** `Tensor` and `GradTensor` share one method surface, so `a.add(b).mul(2).sum()` and `loss.item()` work. The functional forms (`add(a, b)`) still work.
- **camelCase names.** Examples use `matrixPower`, `toDatetime`, `ttestInd`, `crossValScore`, `multivariateNormal` and so on. The snake_case names still work in 1.x and are marked deprecated.
- **dtypes.** `tensor()` creates `float32` tensors. Float operations keep the input float dtype, and index results such as `argmax` are `int32`.
- **Typing note.** `Tensor.item()` is typed as `string | number | bigint`, so examples write `Number(t.item())` before calling `toFixed`. Metric functions already return plain numbers.

## Example Catalog

### Foundations

| #   | Example                                      | Modules Used           | Description                                    |
| --- | -------------------------------------------- | ---------------------- | ---------------------------------------------- |
| 00  | [Quick Start](./00-quick-start/)             | ndarray, dataframe, ml | Tensors, DataFrames, and a first ML model          |
| 01  | [Tensor Basics](./01-tensor-basics/)         | ndarray                | Creating and inspecting N-dimensional arrays   |
| 02  | [Tensor Operations](./02-tensor-operations/) | ndarray                | Arithmetic, math, reductions, and broadcasting |
| 03  | [Data Analysis](./03-data-analysis/)         | dataframe, stats, plot | Exploratory tabular workflow with charts       |
| 04  | [DataFrame Basics](./04-dataframe-basics/)   | dataframe              | Selection, filtering, sorting, and reshaping   |
| 05  | [DataFrame GroupBy](./05-dataframe-groupby/) | dataframe              | GroupBy operations and aggregation             |

### Classical ML

| #   | Example                                              | Modules Used                  | Description                                                |
| --- | ---------------------------------------------------- | ----------------------------- | ---------------------------------------------------------- |
| 06  | [ML Pipeline](./06-ml-pipeline/)                     | ml, metrics, preprocess, plot | End-to-end ML pipeline with multiple models                |
| 07  | [Linear Regression](./07-linear-regression/)         | ml, metrics, preprocess, random | Supervised regression workflow                           |
| 08  | [Logistic Regression](./08-logistic-regression/)     | ml, metrics, preprocess       | Binary classification with evaluation                      |
| 09  | [Ridge & Lasso](./09-ridge-lasso/)                   | ml, metrics, preprocess       | Regularized linear models                                  |
| 10  | [Advanced ML Models](./10-advanced-ml-models/)       | ml, metrics, preprocess       | KMeans, KNN, PCA, and Gaussian Naive Bayes                 |
| 11  | [Tree & Ensemble Models](./11-tree-ensemble-models/) | ml, metrics, preprocess       | Trees, forests, boosting, and SVM                          |
| 12  | [Complete Pipeline](./12-complete-pipeline/)         | ml, metrics, preprocess, plot | Full workflow from data split to visualization             |

### Neural Nets & Optimization

| #   | Example                                                    | Modules Used       | Description                                               |
| --- | ---------------------------------------------------------- | ------------------ | --------------------------------------------------------- |
| 13  | [Neural Network Training](./13-neural-network-training/)   | nn, optim, ndarray | Training with plain tensors: models, losses, optimizers   |
| 14  | [Autograd](./14-autograd/)                                 | ndarray, nn        | Reverse-mode differentiation and gradient flow            |
| 15  | [Activation Functions](./15-activation-functions/)         | ndarray, plot      | Thirteen activations compared, with SVG output            |
| 16  | [LR Schedulers](./16-lr-schedulers/)                       | optim, nn          | Learning-rate scheduler patterns                          |
| 27  | [CNN Layers](./27-cnn-layers/)                             | nn, ndarray        | Convolution and pooling layers                            |
| 28  | [RNN, LSTM, GRU](./28-rnn-lstm-gru/)                       | nn, ndarray        | Sequence modeling layers                                  |
| 29  | [Attention & Transformer](./29-attention-transformer/)     | nn, ndarray        | Multihead attention and transformer building blocks       |
| 30  | [Normalization & Dropout](./30-normalization-dropout/)     | nn, ndarray        | BatchNorm, LayerNorm, and dropout families                |
| 32  | [Module System](./32-module-system/)                       | nn, ndarray        | Custom modules, hooks, and parameter/state management     |
| 39  | [Advanced Neural Networks](./39-advanced-neural-networks/) | nn, optim, ndarray | Trainer utilities, containers, initialization, clipping   |
| 40  | [Transformer Architecture](./40-transformer-architecture/) | nn, ndarray        | Encoder/decoder stacks and positional encoding            |
| 41  | [Advanced Optimizers & Schedulers](./41-advanced-optimizers-schedulers/) | optim, nn, ndarray | RAdam, LAMB, LARS, warm restarts, lambda, sequential LR |

### Data, Stats, and Math

| #   | Example                                                    | Modules Used           | Description                                                 |
| --- | ---------------------------------------------------------- | ---------------------- | ----------------------------------------------------------- |
| 17  | [Encoders](./17-preprocessing-encoders/)                   | preprocess, ndarray    | Label, one-hot, ordinal, binarizers, and multilabel tools   |
| 18  | [Scalers](./18-preprocessing-scalers/)                     | preprocess, ndarray    | Standard, MinMax, Robust, MaxAbs, Power, and Quantile       |
| 19  | [Statistics](./19-statistics/)                             | stats, ndarray         | Descriptive statistics and correlation analysis             |
| 20  | [Linear Algebra](./20-linear-algebra/)                     | linalg, ndarray        | SVD, QR, LU, solving, and norms                             |
| 21  | [Random Sampling](./21-random-sampling/)                   | random, ndarray        | Random distributions and seeded sampling                    |
| 22  | [Datasets](./22-datasets/)                                 | datasets               | Built-in datasets and synthetic generators                  |
| 23  | [Cross-Validation](./23-cross-validation/)                 | preprocess, ml, ndarray | KFold, StratifiedKFold, LeaveOneOut, crossValScore         |
| 24  | [Metrics](./24-metrics/)                                   | metrics, ndarray       | Binary and multiclass, probability, regression and clustering metrics |
| 26  | [Sparse Matrices](./26-sparse-matrices/)                   | ndarray                | CSR sparse matrix operations                                |
| 31  | [DataLoader](./31-dataloader/)                             | datasets, ndarray      | Batch iteration with shuffle, seed, and drop-last           |
| 33  | [Advanced DataFrame Features](./33-dataframe-advanced/)    | dataframe              | String/datetime accessors, rolling, expanding, query, eval  |
| 35  | [Advanced Clustering](./35-advanced-clustering/)           | ml, metrics, datasets  | Agglomerative, GMM, spectral, OPTICS, MeanShift, Birch      |
| 37  | [Model Selection & Pipeline](./37-model-selection-pipeline/) | ml, preprocess       | GridSearchCV, RandomizedSearchCV, crossValidate, Pipeline   |
| 38  | [Feature Engineering](./38-feature-engineering/)           | preprocess, ml, metrics | Imputation, polynomial features, discretization, selection |
| 42  | [FFT & Signal Processing](./42-fft-signal-processing/)     | ndarray                | FFT, inverse FFT, rFFT, 2D FFT, and spectral filtering      |
| 43  | [Statistical Tests](./43-statistical-tests/)               | stats, ndarray         | Distributions, t-tests, ANOVA, chi-square, correlations     |
| 48  | [Statistical Inference Playbook](./48-statistical-inference-playbook/) | stats, plot      | Confidence intervals, bootstrap, KDE, multiple testing, power |
| 49  | [Advanced Linear Algebra Toolkit](./49-advanced-linear-algebra/) | linalg, ndarray | Hessenberg, Schur, polar, matrix functions, sparse solvers     |

### Specialized Topics

| #   | Example                                                    | Modules Used            | Description                                                   |
| --- | ---------------------------------------------------------- | ----------------------- | ------------------------------------------------------------- |
| 25  | [Plotting](./25-plotting/)                                 | plot, ndarray           | Core plotting API with SVG outputs                            |
| 34  | [Ensemble & Advanced ML Models](./34-ensemble-advanced-ml/) | ml, metrics, datasets | AdaBoost, Bagging, Voting, Stacking, ExtraTrees, GPR, LDA     |
| 36  | [Kernel SVM & Anomaly Detection](./36-svm-anomaly-detection/) | ml, metrics, preprocess | SVC, SVR, NuSVC, OneClassSVM, IsolationForest, LOF         |
| 44  | [Advanced Visualization](./44-advanced-visualization/)     | plot, ndarray           | Diagnostic plots, feature importance, ROC, residuals          |
| 45  | [Core Runtime Tooling](./45-core-runtime-tooling/)         | core                    | Logger, warnings, backend registry, and serialization         |
| 46  | [Dataset Transforms & Samplers](./46-dataset-transforms-samplers/) | datasets          | Subset, randomSplit, mapDataset, filterDataset, samplers      |
| 47  | [DataFrame IO & Styling](./47-dataframe-io-styling/)       | dataframe               | JSON/XLSX/Parquet round-trips, style rendering, plot accessor |

## Coverage Summary

- `deepbox/core`: runtime configuration, errors, warnings, logging, serialization, backends
- `deepbox/ndarray`: dense/sparse tensors, autograd, FFT, signal and numerical utilities
- `deepbox/linalg`: decompositions, sparse and structured solvers, matrix functions, equations, and special matrices
- `deepbox/dataframe`: accessors, window ops, IO, styling, and plotting integration
- `deepbox/stats`: distributions, tests, confidence intervals, bootstrap, KDE, power analysis, and correlations
- `deepbox/preprocess`: encoders, scalers, imputers, feature engineering, text, and splitters
- `deepbox/ml`: classical ML, ensembles, anomaly detection, calibration, pipelines, model selection
- `deepbox/nn` + `deepbox/optim`: deep learning layers, training utilities, initialization, and schedulers
- `deepbox/random`, `deepbox/datasets`, `deepbox/metrics`, `deepbox/plot`: sampling, data loading, evaluation, and visualization

## Suggested Path

- **Start here**: `00`-`06`
- **Build depth**: `07`-`24`
- **Cover the rest of the library**: `25`-`49`

## License

MIT. See the license file in the parent directory.
