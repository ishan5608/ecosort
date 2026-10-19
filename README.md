# EcoSort: Recyclable Waste Classification

A compact CNN with 94,470 parameters that classifies 96×96 RGB waste images into six classes. It reaches 100% on the local validation split and passes all four pre-submission checks.

## Results at a glance

| Metric | Value |
|---|---|
| Validation accuracy (selected checkpoint) | 100.0% (240 images, epoch 17) |
| Accuracy from `verify_submission.py` | 100.00% (240 images) |
| Parameters | 94,470 in PyTorch (94,026 in the verifier), limit 350,000 |
| ONNX file size | 0.38 MB (393,842 bytes), limit 1.50 MB |
| Best epoch | 17 (training stopped at epoch 25) |
| Average CPU latency | 0.56 ms/image, target ≤ 35 ms |
| Training hardware | CPU only (laptop, no CUDA GPU) |

## Problem

The organizers asked for a classifier of 96×96 RGB images across six classes: cardboard, glass, metal, paper, plastic, and trash. The model must be exported to ONNX, stay within 350,000 parameters and 1.50 MB, and reach at least 75% accuracy on a hidden test set.

## Repository structure

```
ecosort/
├── configs/              # one JSON file per run (settings)
├── src/
│   ├── data.py           # datasets, transforms, DataLoaders
│   ├── model.py          # model definition, build_model()
│   ├── train.py          # training loop, checkpointing, early stopping, logs
│   ├── export_onnx.py    # exports a checkpoint to ONNX
│   └── check_model.py    # PyTorch vs ONNX comparison, per-class accuracy
├── logs/                 # one JSON log per run
├── models/
│   └── model.onnx        # submitted model
├── checkpoints/          # best .pt files (not committed)
├── data/                 # dataset (not committed)
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

Windows (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place `ecosort_dataset_public.zip` in the `data/` folder, then extract it from the project root:

```powershell
python download_data.py
```

This creates `data/public/train/` and `data/public/val/`, each with six class folders.

## How to reproduce

Run these from the project root:

```powershell
python -m src.train --config configs/cnn_v2_norm.json
python -m src.export_onnx --checkpoint checkpoints/cnn_v2_norm_best.pt
python verify_submission.py --model models\model.onnx --val_dir data\public\val
```

Seeds are fixed (seed 42 in the config), so the same code and data should reproduce these numbers on the same hardware. Small differences are possible on other machines. The checkpoint is not stored in the repository, so the training command must run before the export.

## Data

- **Training:** 900 images (150 per class), from `data/public/train`.
- **Validation:** 240 images (40 per class), from `data/public/val`. Used for all model and checkpoint decisions.
- **Hidden test set:** held by the organizers and not available locally. It was not used for any decision.
- **Duplicate check:** an MD5 comparison of file contents found no exact duplicates between the training and validation folders.
- **Preprocessing:** convert to a tensor, scale pixels to [0, 1], then normalize each channel with mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`. These are the same values `verify_submission.py` uses. The same transform is applied to training and validation.
- **Augmentation:** none.

## Final model

`cnn_v2_norm` (seed 42), with three convolutional blocks and a linear classifier:

- Conv block 1: Conv2d(3→32, 3×3, padding 1), BatchNorm2d(32), ReLU, MaxPool2d(2). Output 32×48×48.
- Conv block 2: Conv2d(32→64, 3×3, padding 1), BatchNorm2d(64), ReLU, MaxPool2d(2). Output 64×24×24.
- Conv block 3: Conv2d(64→128, 3×3, padding 1), BatchNorm2d(128), ReLU, MaxPool2d(2). Output 128×12×12.
- Global average pooling to 128 values, then Linear(128→6).

Parameter breakdown: 896 + 64 + 18,496 + 128 + 73,856 + 256 + 774 = 94,470.

I chose this design because convolutional filters reuse the same weights across the whole image, which keeps the parameter count low while still detecting local patterns such as edges and textures. A fully connected model of similar accuracy was far over the limit (see the experiment log).

## Training

- Optimizer: AdamW, learning rate 1e-3, weight decay 1e-4
- Learning rate schedule: none (constant)
- Batch size: 64
- Loss: CrossEntropyLoss
- Maximum epochs: 50, with early stopping after 8 epochs without improvement in validation accuracy
- Seed: 42
- Checkpoint: the epoch with the highest validation accuracy

## Experiment log

Every run is listed, including the ones that didn't help. Each run has a config in `configs/` and a log in `logs/` with the same name.

| Run | Change | Params | Val acc | Best epoch | Result |
|-----|--------|--------|---------|------------|--------|
| baseline | Basic placeholder model, pipeline check | 5,286 | 66.7% | 23 | Only for testing the pipeline |
| mlp_v1 | Fully connected model, 432 hidden units | 11,946,966 | 91.7% | 11 | Good accuracy, but over the parameter limit |
| cnn_v1 | 3 conv blocks (32, 64, 128), BatchNorm, seed 42 | 94,470 | 97.9% | 7 | Best so far, not final yet |
| cnn_v1_seed0 | Same as cnn_v1, seed 0 | 94,470 | 100.0% | 21 | Accuracy swings between epochs |
| cnn_v2_norm | cnn_v1 with ImageNet normalization, seed 42 | 94,470 | 100.0% | 17 | **Selected for submission**, stable from epoch 17 |
| cnn_v2_norm_seed0 | Same as cnn_v2_norm, seed 0 | 94,470 | 99.6% | 6 | Peak at epoch 6, dips afterward. Not selected |
| cnn_v3 | Wider CNN with 32, 128, 256 channels, seed 42 | 335,430 | 99.6% | 14 | Not chosen. Accuracy went up and down a lot in later epochs, so I kept the smaller model |

**Notes**

- **baseline:** I used a very small model (5,286 parameters) to check that data loading, training, and logging work. The accuracy is low because the model is too small for this task. The loss went down and validation accuracy improved to 66.7% by epoch 23, so the pipeline works end to end.
- **mlp_v1:** Used a model with very large number of parameters(11,946,966) and got rejected by verify_submissions.py . When making the model I didn't account for the given restrictions and just made a simple MLP which was used in Pytorch official tutorial documentation. But during the process I learned how flatten works the hard way. Flatten scales a 96x96 image to 27648(i.e 3* 96*96), this just increased the number of parameters used drastically. It also has 432 hidden parameters , I just did 27648/(batch_size) cause I didn't know how to decide ideal number of hidden parameters.  
- **cnn_v1:** My first CNN, with three conv blocks and BatchNorm. It has 94,470 parameters, which is under the 350,000 limit. Validation accuracy reached 97.9% at epoch 7, and training stopped at epoch 15. The validation numbers jumped between epochs, so this result may be partly luck. When I exported it and ran `verify_submission.py`, it scored 25.83% on the same validation images. That is because it was trained without the verifier's normalization, as explained in cnn_v2_norm.
- **cnn_v1_seed0:** The same model with seed 0. It reached 100% at epoch 21, but validation accuracy dropped to 52.5% at epoch 14 and 68.8% at epoch 29. The training curve was unstable.
- **cnn_v2_norm:** My first runs got much lower accuracy in `verify_submission.py` than in my own validation, which meant my training preprocessing didn't match the verifier's. The verifier normalizes with the ImageNet mean and std after scaling to [0, 1]. I added the same normalization to my training and validation transform. Validation accuracy then reached 100% at epoch 17, and the verifier, PyTorch, and ONNX checks all agreed. The checkpoint was chosen on the validation set, so this number is optimistic.
- **cnn_v2_norm_seed0:** The same model with seed 0, to check that the result doesn't depend on one seed. It reached 99.58% at epoch 6, then dipped in later epochs. One validation image is predicted differently by PyTorch and ONNX, which is likely floating-point rounding. Not selected.
- **cnn_v3:** I tested a wider version of the CNN, with 32, 128, and 256 channels in the three conv blocks. It has 335,430 parameters, which is within the limit. It reached 99.6% validation accuracy at epoch 14, but its later epochs were less stable, with validation loss spikes and accuracy dropping as low as 88%. Since the smaller model was more stable and matched its accuracy, I kept the 94,470-parameter model for submission.

## Key observations

- **Normalization mattered most.** The same architecture scored 97.9% in PyTorch but 25.83% in the verifier, because the verifier normalizes its inputs and my training didn't.
- **Glass and plastic were the most confused pair.** In the unnormalized model (cnn_v1), 4 of 40 glass images were predicted as plastic, and 1 plastic image was predicted as glass. Reflective surfaces may explain this, but I haven't tested that.
- **Stability improved with normalization.** In the selected run, validation accuracy stayed between 98.3% and 100% from epoch 17 to epoch 25. Without normalization, it swung between about 50% and 100%.
- **The selected model made no errors on the validation set.** Its confusion matrix is diagonal for all six classes, and PyTorch, ONNX, and the verifier agree.

## What didn't work

- **Fully connected model (mlp_v1):** the first layer alone had millions of weights, and flattening the image discarded its spatial structure. It reached 91.7% but could not be submitted.
- **Training without the verifier's normalization:** the model scored 97.9% in PyTorch and 25.83% in the verifier.
- **Wider CNN (cnn_v3):** it had 3.5 times the parameters of the selected model and was less stable, with no gain in accuracy.
- **Unnormalized CNN runs:** validation accuracy jumped between epochs, which made the best-epoch selection unreliable.

## Limitations and next steps

- **The validation set is saturated.** With 240 images and 100% accuracy, it can no longer separate good models from very good ones.
- **Checkpoint selection is optimistic.** The checkpoint was chosen on the same validation images used to report the score. The hidden test result may be lower.
- **Only two seeds were run** for the selected architecture. Seed 1 was skipped, so the spread across seeds is not measured.
- **Parameter counts differ slightly.** PyTorch counts 94,470 parameters and the verifier counts 94,026. Both are well under the limit.
- **Latency was measured on one laptop CPU** and may differ on other hardware.
- **Next steps:** run more seeds, hold out a separate validation split from the training images for model selection, and try training-only augmentation such as horizontal flips and small crops.

## Environment

- Operating system: Windows
- Python: 3.12.10
- PyTorch and onnxruntime: versions listed in `requirements.txt`
- Hardware: laptop CPU with integrated graphics. All training ran on the CPU.
- The ONNX export prints a version-conversion warning. The file was not modified, and it passes verification.