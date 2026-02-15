# Deepbox Examples

> **Browse online:** https://deepbox.dev/examples · **Docs:** https://deepbox.dev/docs

Comprehensive examples demonstrating the full capabilities of Deepbox. Each example is self-contained with its own directory, entry point, and documentation.

## Examples Overview

### Getting Started

| #   | Example                                      | Modules Used           | Description                                    |
| --- | -------------------------------------------- | ---------------------- | ---------------------------------------------- |
| 00  | [Quick Start](./00-quick-start/)             | ndarray, dataframe, ml | Tensors, DataFrames, and ML in under 50 lines  |
| 01  | [Tensor Basics](./01-tensor-basics/)         | ndarray                | Creating and manipulating N-dimensional arrays |
| 02  | [Tensor Operations](./02-tensor-operations/) | ndarray                | Arithmetic, math functions, and reductions     |

### DataFrames

| #   | Example                                      | Modules Used           | Description                                 |
| --- | -------------------------------------------- | ---------------------- | ------------------------------------------- |
| 03  | [Data Analysis](./03-data-analysis/)         | dataframe, stats, plot | Data analysis workflow with visualization   |
| 04  | [DataFrame Basics](./04-dataframe-basics/)   | dataframe              | Creating, selecting, filtering, and sorting |
| 05  | [DataFrame GroupBy](./05-dataframe-groupby/) | dataframe              | GroupBy operations and aggregation          |

### Classical Machine Learning

| #   | Example                                              | Modules Used                  | Description                                            |
| --- | ---------------------------------------------------- | ----------------------------- | ------------------------------------------------------ |
| 06  | [ML Pipeline](./06-ml-pipeline/)                     | ml, metrics, preprocess, plot | End-to-end ML pipeline with multiple models            |
| 07  | [Linear Regression](./07-linear-regression/)         | ml, metrics, preprocess       | Regression with train/test split and evaluation        |
| 08  | [Logistic Regression](./08-logistic-regression/)     | ml, metrics, preprocess       | Binary classification with Iris dataset                |
| 09  | [Ridge & Lasso](./09-ridge-lasso/)                   | ml, metrics, preprocess       | Regularized regression (Ridge L2, Lasso L1)            |
| 10  | [Advanced ML Models](./10-advanced-ml-models/)       | ml, metrics, preprocess       | KMeans, KNN, PCA, Gaussian Naive Bayes                 |
| 11  | [Tree & Ensemble Models](./11-tree-ensemble-models/) | ml, metrics, preprocess       | Decision Trees, Random Forests, Gradient Boosting, SVM |
| 12  | [Complete Pipeline](./12-complete-pipeline/)         | ml, metrics, preprocess, plot | Full ML workflow: load, preprocess, train, evaluate    |

### Neural Networks & Autograd

| #   | Example                                                  | Modules Used       | Description                                      |
| --- | -------------------------------------------------------- | ------------------ | ------------------------------------------------ |
| 13  | [Neural Network Training](./13-neural-network-training/) | nn, optim, ndarray | Sequential models, custom modules, Adam, SGD     |
| 14  | [Autograd](./14-autograd/)                               | ndarray            | Automatic differentiation, backward pass, noGrad |
| 15  | [Activation Functions](./15-activation-functions/)       | ndarray, plot      | ReLU, Sigmoid, GELU, Mish, Swish, ELU comparison |

### Optimization

| #   | Example                              | Modules Used | Description                                            |
| --- | ------------------------------------ | ------------ | ------------------------------------------------------ |
| 16  | [LR Schedulers](./16-lr-schedulers/) | optim, nn    | All 8 schedulers: Step, Cosine, OneCycle, Warmup, etc. |

### Preprocessing

| #   | Example                                  | Modules Used        | Description                                       |
| --- | ---------------------------------------- | ------------------- | ------------------------------------------------- |
| 17  | [Encoders](./17-preprocessing-encoders/) | preprocess, ndarray | Label, OneHot, Ordinal, Binarizer, MultiLabel     |
| 18  | [Scalers](./18-preprocessing-scalers/)   | preprocess, ndarray | Standard, MinMax, Robust, MaxAbs, Power, Quantile |

### Statistics & Linear Algebra

| #   | Example                                | Modules Used    | Description                                  |
| --- | -------------------------------------- | --------------- | -------------------------------------------- |
| 19  | [Statistics](./19-statistics/)         | stats, ndarray  | Descriptive stats, percentiles, correlations |
| 20  | [Linear Algebra](./20-linear-algebra/) | linalg, ndarray | SVD, QR, LU decompositions, solvers, norms   |

### Utilities

| #   | Example                                    | Modules Used        | Description                                        |
| --- | ------------------------------------------ | ------------------- | -------------------------------------------------- |
| 21  | [Random Sampling](./21-random-sampling/)   | random              | 10 distributions: uniform, normal, binomial, etc.  |
| 22  | [Datasets](./22-datasets/)                 | datasets            | 24 built-in datasets + 6 synthetic generators      |
| 23  | [Cross-Validation](./23-cross-validation/) | preprocess, ndarray | KFold, StratifiedKFold, LeaveOneOut                |
| 24  | [Metrics](./24-metrics/)                   | metrics, ndarray    | Classification, regression, and clustering metrics |

### Visualization & Special Topics

| #   | Example                                  | Modules Used  | Description                                         |
| --- | ---------------------------------------- | ------------- | --------------------------------------------------- |
| 25  | [Plotting](./25-plotting/)               | plot, ndarray | Line, scatter, bar, histogram, heatmap (SVG output) |
| 26  | [Sparse Matrices](./26-sparse-matrices/) | ndarray       | CSR sparse matrix operations                        |

### Deep Learning Layers

| #   | Example                                                | Modules Used      | Description                                         |
| --- | ------------------------------------------------------ | ----------------- | --------------------------------------------------- |
| 27  | [CNN Layers](./27-cnn-layers/)                         | nn, ndarray       | Conv1d, Conv2d, MaxPool2d, AvgPool2d                |
| 28  | [RNN, LSTM, GRU](./28-rnn-lstm-gru/)                   | nn, ndarray       | Recurrent layers for sequence modeling              |
| 29  | [Attention & Transformer](./29-attention-transformer/) | nn, ndarray       | MultiheadAttention, TransformerEncoderLayer         |
| 30  | [Normalization & Dropout](./30-normalization-dropout/) | nn, ndarray       | BatchNorm1d, LayerNorm, Dropout                     |
| 31  | [DataLoader](./31-dataloader/)                         | datasets, ndarray | Batch iteration with shuffle and drop-last          |
| 32  | [Module System](./32-module-system/)                   | nn, ndarray       | Custom modules, state dicts, hooks, freeze/unfreeze |

## Features Coverage

Each example demonstrates specific Deepbox modules:

### Core Modules

- **deepbox/ndarray** — Tensor operations, autograd, broadcasting
- **deepbox/linalg** — SVD, eigenvalues, matrix decomposition
- **deepbox/dataframe** — Data manipulation, grouping, filtering
- **deepbox/stats** — Statistical tests, correlations, descriptive stats

### Machine Learning

- **deepbox/ml** — Classical ML (RF, GBM, SVM, KNN, Trees, Regression)
- **deepbox/preprocess** — Scaling, encoding, train/test split, cross-validation
- **deepbox/metrics** — Classification, regression, clustering metrics

### Deep Learning

- **deepbox/nn** — Neural network layers, loss functions, modules
- **deepbox/optim** — Optimizers and learning rate schedulers

### Utilities

- **deepbox/random** — Random sampling from 10+ distributions
- **deepbox/datasets** — 24 built-in datasets + 6 synthetic generators
- **deepbox/plot** — Publication-quality SVG/PNG visualizations

## Learning Path

- **Beginners** — Start with 00–05: tensors, DataFrames, basic operations
- **Intermediate** — Move to 06–18: ML pipelines, neural networks, preprocessing
- **Advanced** — Explore 19–32: statistics, linear algebra, visualization, sparse matrices, deep learning layers

## License

MIT — See LICENSE file in parent directory
