# Neural Network Image Classifier

> **View online:** https://deepbox.dev/projects/02-neural-image-classifier

A neural image classification project built with Deepbox, demonstrating an end-to-end MLP workflow for the digits dataset.

## Features

- **Model Architecture**: Multi-layer perceptron (MLP) with dropout and activation layers
- **Data Pipeline**: Dataset loading, train/test split, feature scaling
- **Evaluation Metrics**: Precision, recall, F1-score, confusion matrix
- **Visualizations**: Training loss and accuracy curves

## Deepbox Modules Used

| Module               | Features Used                                |
| -------------------- | -------------------------------------------- |
| `deepbox/nn`         | Sequential model layers and loss computation |
| `deepbox/ndarray`    | Tensor/GradTensor data handling              |
| `deepbox/metrics`    | confusionMatrix, precision, recall, f1Score  |
| `deepbox/datasets`   | loadDigits                                   |
| `deepbox/preprocess` | trainTestSplit, StandardScaler               |
| `deepbox/plot`       | Loss and accuracy SVG plots                  |

## Usage

```bash
npm run project:02
```

## Output

- Training progress with loss/accuracy
- Model evaluation metrics
- Confusion matrix visualization
- Learning curves plot

## Architecture

```
02-neural-image-classifier/
├── index.ts              # Main entry point
├── README.md             # This file
└── src/
    ├── models.ts         # MLP model definitions
    └── trainer.ts        # Training helpers
```
