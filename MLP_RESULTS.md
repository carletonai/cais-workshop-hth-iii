# MLP workshop results

Run `venv/bin/python workshop.py --compare` from this directory. The default seed is 42. Both models run on CPU.

The original validation rows contain 45 survivors and 5 deaths. Always predicting survival earns 90% accuracy on that split. Increasing network width cannot resolve this evaluation problem. The original training also scales each split independently, so the same input value means different things during training and validation.

The updated example uses a reproducible stratified 60/20/20 split and fits scaling on training data only. It log-transforms three skewed measurements. Follow-up duration is excluded because the example assumes predictions are made at the initial assessment. The smaller 11 → 32 → 16 → 2 MLP uses dropout, AdamW, weight decay, and class-weighted loss. Early stopping restores the checkpoint with the lowest validation loss; the test set does not select checkpoints.

## Measured results

Both MLPs below use the corrected split and preprocessing. The previous architecture uses its original SGD learning rate of 0.001 for 50 epochs. This comparison measures the combined architecture and training changes, not architecture alone.

| Model | Test accuracy | Test balanced accuracy | Test death recall |
| --- | ---: | ---: | ---: |
| Always predict the training majority | 68.3% | 50.0% | 0.0% |
| Previous MLP + SGD | 31.7% | 50.0% | 100.0% |
| Updated MLP + AdamW | 66.7% | 62.9% | 52.6% |

The updated model correctly classified 30 of 41 survivors and 10 of 19 deaths. Its validation accuracy was 71.7%, with the best checkpoint at epoch 8. It improved balanced accuracy over the previous setup but did not beat the majority baseline on raw test accuracy. These are results from one small holdout, not evidence of a reliable 90% model. The former 90% score is not directly comparable because the split and available features changed.

Before further model selection, use cross-validation within the development data and keep a fresh final holdout. More nodes alone are unlikely to solve the limits of 299 examples.

Training-only preprocessing follows [scikit-learn's guidance](https://scikit-learn.org/1.8/common_pitfalls.html). Dataset details are available from [UCI](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records).
