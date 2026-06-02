# Kaggle Inputs To Add

For the current active `birdclef-2026-eos-9.ipynb` configuration:

```python
Model_22 + Model_51 + Model_74
_runSED_once = True
```

add these inputs in Kaggle before running or saving a submission version.

## Required Inputs

1. **BirdCLEF+ 2026 competition data**
   - Add from the competition page: `birdclef-2026`
   - Expected mount:
     - `/kaggle/input/competitions/birdclef-2026`
     - or `/kaggle/input/birdclef-2026`
   - Needed files:
     - `sample_submission.csv`
     - `taxonomy.csv`
     - `train_soundscapes_labels.csv`
     - `train_soundscapes/`
     - `test_soundscapes/`

2. **Perch ONNX for BirdCLEF 2026**
   - Kaggle dataset slug/name:
     - `rishikeshjani/perch-onnx-for-birdclef-2026`
   - Expected mount:
     - `/kaggle/input/datasets/rishikeshjani/perch-onnx-for-birdclef-2026`
   - Needed files:
     - `perch_v2.onnx`
     - `onnxruntime-1.24.4-*.whl`

3. **Google Perch model**
   - Kaggle model:
     - `google/bird-vocalization-classifier`
   - Framework/version:
     - `tensorflow2/perch_v2_cpu/1`
   - Expected mount:
     - `/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1`
   - Needed for:
     - TF fallback
     - Perch labels/assets

4. **TensorFlow wheels**
   - Kaggle notebook output/input:
     - `ashok205/tf-wheels`
   - Expected mount:
     - `/kaggle/input/notebooks/ashok205/tf-wheels/tf_wheels`
   - Needed files:
     - `tensorflow-2.20.0-*.whl`
     - `tensorboard-2.20.0-*.whl`

5. **Pretrained ProtoSSM / ResidualSSM weights**
   - Kaggle dataset slug/name:
     - `hideyukizushi/sgkfk-202604041716`
   - Expected mount:
     - `/kaggle/input/datasets/hideyukizushi/sgkfk-202604041716`
   - Needed files:
     - `train_proto_ssm_single/models/proto_ssm_best.pt`
     - `train_proto_ssm_single/models/proto_ssm_history.json`
     - `ResidualSSM/models/residual_ssm_best.pt`

6. **Perch training cache**
   - Add at least one of these:
     - `jaejohn/perch-meta`
     - notebook output `vyankteshdwivedi/notebook1b25083f0d`
   - Expected mounts:
     - `/kaggle/input/datasets/jaejohn/perch-meta`
     - or `/kaggle/input/notebooks/vyankteshdwivedi/notebook1b25083f0d`
   - Needed files, one pair:
     - `full_perch_meta.parquet` + `full_perch_arrays.npz`
     - or `perch_meta.parquet` + `perch_arrays.npz`

7. **Distilled SED public ONNX folds**
   - Kaggle dataset slug/name:
     - `tuckerarrants/bc2026-distilled-sed-public`
   - Expected mount:
     - searched automatically under `/kaggle/input`
   - Needed files:
     - `sed_fold0.onnx`
     - other `sed_fold*.onnx` files

## Optional / Future Inputs

1. **Extra prediction artifact CSVs**
   - Only needed if you manually enable:
     - `EXTRA_ARTIFACT_BLEND_WEIGHT > 0`
     - `EXTRA_ARTIFACT_CSVS = [...]`
   - CSVs must exactly match `sample_submission.csv` rows and 234 class columns.

2. **Model_1 inputs**
   - Not needed for the current active ensemble.
   - Add only if you enable `Model_1`.
   - Referenced inputs:
     - `tuckerarrants/perch-v2-no-dft-onnx`
     - `tuckerarrants/birdclef-2026-waveform-cache`
     - `tuckerarrants/bc2026-distilled-sed-public`

## Quick Kaggle Check

After adding inputs, run the first cells and confirm the manifest shows `found=True` for all required active inputs:

```text
competition data
sample_submission.csv
taxonomy.csv
onnxruntime import/wheel
Perch ONNX or TF backend
Perch train cache
pretrained ProtoSSM
pretrained ResidualSSM
SED ONNX folds
```

If any required row is `found=False`, add the missing Kaggle input before saving a submission version.
