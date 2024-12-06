# Deep Fingerprinting (DF)
This repository contains the Tensorflow 2 implementation of the **DF** WF attack as described in the [ACM CCS2018 paper](https://arxiv.org/pdf/1801.02265).

## Major Changes
1. Converted the code from **Tensorflow 1** to **Tensorflow 2**:
2. Changed dataset format:
   - **Old format**: `.pkl` file.
   - **New format**: `.npz` file.
2. Added **Tensorboard** logging.

## Modifying for Other Data Formats
If your dataset is in a different format:
- Update the `'load_data'` method in the `ClosedWorld_DF_NoDef.py` file.