# Neural Network Image Classifier

> **View online:** https://deepbox.dev/projects/02-neural-image-classifier

Trains a small multi-layer perceptron on the 8x8 digits dataset with mini-batch Adam, then reports test accuracy, precision, recall, F1 and a confusion matrix.

## Features

- Model: `Linear`, `ReLU`, `Linear` in a `Sequential`. `src/models.ts` also builds GELU and LeakyReLU variants
- Data pipeline: `loadDigits`, `trainTestSplit`, `StandardScaler`
- Training on plain tensors. The model has trainable parameters and grad mode is on, so `model.forward(x)` returns a `GradTensor` that tracks the weights, and `loss.backward()` fills the gradients. No `parameter(...)` wrapper is needed for the data
- Evaluation under `noGrad()`, where the output is a plain `Tensor`
- Training loss and accuracy curves as SVG

## Training loop

```ts
const optimizer = new Adam(model.parameters(), { lr: 0.001 });

optimizer.zeroGrad();
const output = model.forward(tensor(XBatch, { dtype: "float32" }));
const loss = crossEntropyLoss(output, tensor(yBatch, { dtype: "int32" }));
loss.backward();
optimizer.step();
```

`crossEntropyLoss` takes class indices as targets, not one-hot rows. Its return type is a union, so `index.ts` first checks `output instanceof GradTensor`, which narrows the type to `GradTensor`.

## Deepbox Modules Used

| Module               | Features Used                                                                           |
| -------------------- | --------------------------------------------------------------------------------------- |
| `deepbox/nn`         | `Sequential`, `Linear`, `ReLU`, `GELU`, `LeakyReLU`, `crossEntropyLoss`                 |
| `deepbox/optim`      | `Adam`                                                                                  |
| `deepbox/ndarray`    | `tensor`, `GradTensor`, `noGrad`, and the `argmax(1)`, `toArray()` and `item()` methods |
| `deepbox/metrics`    | `confusionMatrix`, `precision`, `recall`, `f1Score`                                     |
| `deepbox/datasets`   | `loadDigits`                                                                            |
| `deepbox/preprocess` | `trainTestSplit`, `StandardScaler`                                                      |
| `deepbox/random`     | `setSeed`                                                                               |
| `deepbox/plot`       | `Figure`, loss and accuracy SVG plots                                                   |

## Usage

```bash
npm run project:02
```

## Output

- Training progress with loss and accuracy every 5 epochs
- Test metrics and a confusion matrix
- `output/loss-curve.svg`
- `output/accuracy-curve.svg`

## Architecture

```text
02-neural-image-classifier/
├── index.ts              # Main entry point
├── README.md             # This file
├── output/               # Generated SVGs
└── src/
    ├── models.ts         # MLP model definitions
    └── trainer.ts        # Optional helpers (not imported by index.ts): trainStep, evaluateModel, EarlyStopping, LR schedules
```
