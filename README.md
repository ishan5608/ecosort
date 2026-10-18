# EcoSort: Recyclable Waste Classification

<One-sentence summary of what your model does and the final result, for example:
"A compact CNN that classifies 96×96 waste images into six classes, with 91.2% validation accuracy
and 290,412 parameters.">

## Results at a glance

| Metric | Value |
|---|---|
| Validation accuracy (best checkpoint) | <xx.x%> |
| Parameters | <count> (limit: 350,000) |
| ONNX file size | <x.xx MB> (limit: 1.50 MB) |
| Best epoch | <n> |
| Training time | <CPU or GPU, minutes> |

## Problem

The organizers asked for a classifier of 96×96 RGB images across six classes (cardboard, glass,
metal, paper, plastic, trash). The model must be exported to ONNX, stay within 350,000 parameters
and 1.5 MB, and reach at least 75% accuracy on a hidden test set.

## Repository structure

```
ecosort/
├── configs/       # one JSON file per experiment
├── src/           # data, model, training, export code
├── logs/          # one JSON log per run
├── models/        # final model.onnx
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python download_data.py
```

## How to reproduce

```bash
python -m src.train --config configs/<best_config>.json
python -m src.export_onnx --checkpoint checkpoints/<run_id>_best.pt
python verify_submission.py models/model.onnx
```

Random seeds are fixed (`SEED = 42`), so these commands should reproduce the reported numbers
on the same hardware. Small differences across CPUs or GPUs are possible.

## Data

- Training images: <n> (per class: <n>)
- Validation split: <n> images, held out from the training folder with a fixed seed
- Preprocessing: scaled to [0, 1], normalized per channel with the training-set mean and std
  (mean = <values>, std = <values>)
- Augmentation (training only): <list, or "none">

The validation split is used for all model and checkpoint decisions. The hidden test set was used
only for the final submission.

## Final model

<Describe the architecture in a short paragraph or a small layer-by-layer list. Explain why you
chose it: what it trades off, and how it fits the size limits.>

Example:
- Conv block 1: Conv2d(3→32, 3×3), BatchNorm, ReLU, MaxPool
- Conv block 2: Conv2d(32→64, 3×3), BatchNorm, ReLU, MaxPool
- ...
- Global average pooling, Linear(→6)

## Training

- Optimizer: <AdamW, lr = 1e-3, weight decay = 1e-4>
- Batch size: <64>
- Max epochs: <50>, early stopping with patience <8> on validation accuracy
- Loss: CrossEntropyLoss
- Checkpoint: the epoch with the highest validation accuracy

## Experiment log

Every run is listed, including the ones that didn't help. Each run has a config file and a JSON
log in `logs/`.

| Run | Change | Params | Val acc | Best epoch | Result and what I learned |
|---|---|---|---|---|---|
| 001 | Basic placeholder model, used to check if the pipeline works. | 5286 | 66.67%| 23| Only meant to test the data loading, training, and logging end to end. Not a real attempt at the task. |
| 002 | Fully connected model (Flatten, 432 hidden units) | 11,946,966 | 91.7% | 11 | Good accuracy, but 34x over the parameter limit and about 48 MB, so it cannot be submitted. |
| 003 | + BatchNorm | <n> | <xx.x%> | <n> | <...> |
| 004 | SGD with momentum instead of AdamW | <n> | <xx.x%> | <n> | Worse; reverted. <Reason.> |

**Notes**
- **001:** Used a very small model (5,286 parameters) to check that data loading, training, and logging work. The accuracy is low because the model is too small for this task. The loss went down and validation accuracy improved to 66.7% by epoch 23, so the pipeline works end to end.
- **002** Used a model with very large number of parameters(11,946,966) and got rejected by verify_submissions.py . When making the model I didn't account for the given restrictions and just made a simple MLP which was used in Pytorch official tutorial documentation. But during the process I learned how flatten works the hard way. Flatten scales a 96x96 image to 9216(i.e 96*96), this just increased the number of parameters used drastically. It also has 432 hidden parameters , I just did 9216/(batch_size) cause I didn't know how to decide ideal number of hidden parameters.  

### Key observations

- <Example: Plastic and glass were the most confused pair, which suggests reflective surfaces
  are hard for the model.>
- <Example: Augmentation reduced the gap between training and validation accuracy from X to Y.>

## What didn't work

<A short paragraph on the approaches you tried that failed, and what you think caused the failure.
Reviewers find this section valuable because it shows how you reasoned.>

## Limitations and next steps

- <Example: The model is small, so it may miss fine texture on paper versus cardboard.>
- <Example: Given more time, I would try a depthwise-separable architecture to fit more depth
  within the parameter budget.>

## Environment

- Python <3.12>, PyTorch <version>, onnxruntime <version>
- Hardware: <CPU model, or GPU model in Colab/Kaggle>