# DL-assignment-2

Deep Learning assignment, Section B. A compact Inception-style CNN is trained from scratch on 500 CIFAR-10 training images and evaluated on both the clean test set and a shifted test set.

## Setup

- **Dataset:** CIFAR-10 loaded through `torchvision`.
- **`train_small`:** 500 training images (50 per class), split into 400 training images and 100 validation images.
- **`clean_test`:** the original 10,000 CIFAR-10 test images.
- **`shifted_test`:** the same test images with random rotation (±25°), brightness and contrast changes, and a 16×16 black occlusion box (25% of the image).
- Test sets are used for reporting only; hyperparameters are selected using the validation split.

## Seeds

| Purpose | Seed |
|---|---|
| Data selection and split | `SPLIT_SEED = 42` |
| Shifted test set | `SHIFT_SEED = 123` |
| Training runs (mean ± std) | `RUN_SEEDS = [0, 1, 2]` |

## Model and training

- Inception-style CNN trained from scratch. The earlier model description reports 145,242 trainable parameters.
- Previously documented training configuration: AdamW (learning rate `3e-3`, weight decay `5e-4`), cosine learning-rate schedule, 60 epochs, batch size 64, label smoothing 0.1, Mixup (`alpha = 0.2`), and best-validation checkpoint selection.
- Augmentation is applied on the fly to training data only: random crop, horizontal flip, rotation, color jitter, Gaussian noise, and Random Erasing.

## Latest results (mean ± std over 3 seeds)

The following values are transcribed from the latest Kaggle notebook output provided.

| Configuration | Train Acc | Val Acc | Clean Test | Shifted Test |
|---|---:|---:|---:|---:|
| No augmentation | 0.9567 ± 0.0136 | 0.5167 ± 0.0189 | 0.4374 ± 0.0056 | 0.1840 ± 0.0059 |
| Geometric only | 0.6700 ± 0.0569 | 0.4500 ± 0.0141 | 0.3727 ± 0.0013 | 0.2377 ± 0.0083 |
| Photometric only | 0.5792 ± 0.0293 | 0.5300 ± 0.0163 | 0.4179 ± 0.0067 | 0.2240 ± 0.0083 |
| Geometric + photometric | 0.3767 ± 0.0042 | 0.3667 ± 0.0125 | 0.3379 ± 0.0097 | 0.2119 ± 0.0326 |
| + Cutout / Random Erasing | 0.3492 ± 0.0405 | 0.3500 ± 0.0000 | 0.3179 ± 0.0138 | 0.2331 ± 0.0090 |

## Results interpretation

- **Highest clean-test accuracy in the displayed results:** No augmentation, at `0.4374 ± 0.0056`.
- **Highest validation accuracy in the displayed results:** Photometric only, at `0.5300 ± 0.0163`.
- **Highest shifted-test accuracy in the displayed results:** Geometric only, at `0.2377 ± 0.0083`.
- The no-augmentation baseline has the highest training accuracy, while the augmentation configurations shown trade training accuracy for different validation and test outcomes.
- In the displayed results, the Cutout / Random Erasing configuration reaches `0.2331 ± 0.0090` shifted-test accuracy, compared with `0.1840 ± 0.0059` for the baseline.

## How to reproduce

1. Open the notebook in a Kaggle or Colab runtime.
2. Ensure internet access is enabled if CIFAR-10 must be downloaded automatically.
3. Run the notebook cells in order from top to bottom.
4. Training all configurations and seeds may take some time.

## Notes

- The official hidden corruption was not provided. The `shifted_test` set uses the corruptions listed in the assignment setup; Part E4 uses an assumed random-grayscale corruption (`p = 0.5`).
- Validation uses the clean 100-image validation split, so results can vary and may favor weaker augmentation.
- The latest screenshot does not include results for `+ Mixup + label smoothing` or `Final model`. Those values are intentionally omitted rather than copying metrics from the earlier output.
- Parts E, F, and E4 were previously documented as using seed 0; confirm against the notebook if this has changed.
