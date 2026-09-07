# Industrial Image Anomaly Detection with SimCLR and Vision Transformers

<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?color=00E5C3&amp;lines=Self-Supervised+Vision+Transformers;SimCLR+%7C+MVTec+AD+%7C+Anomaly+Detection;Patch-Level+Localization+%7C+Frozen-Feature+Classification;Three-Seed+Reproducibility&amp;center=true&amp;width=900&amp;height=45" alt="SimCLR and Vision Transformers for industrial anomaly detection">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Language-Python-blue" alt="Python">
  <img src="https://img.shields.io/badge/Framework-PyTorch-red" alt="PyTorch">
  <img src="https://img.shields.io/badge/Backbone-ViT--B%2F16-purple" alt="ViT-B/16">
  <img src="https://img.shields.io/badge/SSL-SimCLR-orange" alt="SimCLR">
  <img src="https://img.shields.io/badge/Dataset-MVTec%20AD-green" alt="MVTec AD">
  <img src="https://img.shields.io/badge/Seeds-42%20%7C%20123%20%7C%202026-teal" alt="Three random seeds">
  <img src="https://img.shields.io/badge/Status-Research-orange" alt="Research project">
</p>

<!-- Optional repository badges: replace YOUR_GITHUB_USERNAME, then remove this comment wrapper.
![Repository size](https://img.shields.io/github/repo-size/YOUR_GITHUB_USERNAME/ViT_Anomaly_Detection_SimCLR)
![Last commit](https://img.shields.io/github/last-commit/YOUR_GITHUB_USERNAME/ViT_Anomaly_Detection_SimCLR)
![Contributors](https://img.shields.io/github/contributors/YOUR_GITHUB_USERNAME/ViT_Anomaly_Detection_SimCLR)
-->

This project investigates industrial image anomaly detection using a **Vision Transformer trained from scratch with self-supervised SimCLR**, followed by local patch scoring, anomaly localization, and a separate supervised classification experiment.

The notebooks cover **15 MVTec AD categories** and include fixed-configuration experiments for random seeds **42, 123, and 2026**.

> **Before running:** the three final seed notebooks require an existing, compatible SSL checkpoint for their respective seed. They do not train a new SSL encoder. The reference notebook contains the training workflow.

---

## 1. Project Overview

The implementation contains two distinct evaluation protocols:

| Component | Protocol A — Detection and localization | Protocol B — Supervised classification |
|---|---|---|
| Purpose | Detect normal versus anomalous images and localize defects | Predict category, anomaly status, and defect type |
| Encoder | Self-supervised ViT using official `train/good` data | Same encoder, frozen during probe training |
| Downstream method | Normal reference banks and patch kNN scoring | Supervised linear heads |
| Evaluation data | Full official MVTec test split | Held-out subset of a separate split inside official test |
| Labels used for fitting | Normal training/validation membership; synthetic anomalies for diagnostic audit | Labeled downstream training images, including real defects |
| Interpretation | Normal-only anomaly detection benchmark | Separate supervised representation evaluation |

**Primary detector:** local patch scoring. Global-only scoring and fixed **25% global / 75% local** fusion are secondary ablations.

Multi-class category and defect-type classification belongs to **Protocol B**; Protocol A performs binary anomaly detection across multiple categories.

<!-- Add the image later and remove the comment wrapper:
<p align="center">
  <img src="images/framework_overview.png" width="950" alt="Overview of SimCLR-ViT and the two evaluation protocols">
</p>
-->

## 2. Notebooks

The links below assume the notebooks are stored in the repository's `notebooks/` folder.

| Notebook | Seed | Role |
|---|---:|---|
| [Code1_SEED_42_MVTec.ipynb](notebooks/Code1_SEED_42_MVTec.ipynb) | 42 | Final fixed-aggregation experiment |
| [Code2_SEED_123_MVTec.ipynb](notebooks/Code2_SEED_123_MVTec.ipynb) | 123 | Final fixed-aggregation experiment |
| [Code3_SEED_2026_MVTec.ipynb](notebooks/Code3_SEED_2026_MVTec.ipynb) | 2026 | Final fixed-aggregation experiment |
| [Reference code.ipynb](notebooks/Reference%20code.ipynb) | Configurable | Development workflow, SSL training, and aggregation selection |

The reference notebook uses the development configuration `v5p1_local_primary_rawloc`. The final notebooks use `v5p1_FINAL_FIXED_LOGSUMEXP`, with `logsumexp_t4` fixed before the reruns.

## Python package structure

The modular package under `src/` is extracted from the final seed notebooks
(`v5p1_FINAL_FIXED_LOGSUMEXP`). Notebooks remain the scientific audit trail;
numerical settings are unchanged.

| Module | Role |
|---|---|
| `src/config.py` | Dataclass Protocol A configuration and YAML loading |
| `src/model.py` | `SimCLRv2ViT`, checkpoint load/freeze helpers |
| `src/data.py` | MVTec indexing, splits, SSL vs eval transforms |
| `src/features.py` | Multilayer patch features and fixed QR projection |
| `src/anomaly_scoring.py` | Prototypes, kNN, `logsumexp_t4` aggregation, hybrid ablation |
| `src/calibration.py` | Position MAD calibration and q=0.95 image calibration |
| `src/localization.py` | RAW patch-score → 224×224 maps; visualization limits |
| `src/metrics.py` | Image/pixel metrics and multi-seed sample SD (`ddof=1`) |
| `src/protocol_a.py` | Known-category Protocol A orchestration API |

Example (portable paths; no notebook Drive mounts required):

```bash
pip install -r requirements.txt
python scripts/run_protocol_a.py --config configs/seed42.yaml --data-root /path/to/mvtec-ad
```

Seed wrappers: `scripts/run_seed42.py`, `scripts/run_seed123.py`, `scripts/run_seed2026.py`.

See also [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) and [checkpoints/README.md](checkpoints/README.md).

## 3. Start with the Environment

<details>
<summary><strong>3.1 Google Colab — notebook-native setup</strong></summary>

1. Open the desired notebook in Google Colab.
2. Select a GPU runtime using **Runtime → Change runtime type**.
3. Run the setup cell and authorize Google Drive mounting when prompted.
4. Check the printed dataset path, project root, seed, and device before continuing.

The setup cell installs `timm`, `kagglehub`, `scikit-learn`, `pandas`, and `scipy`, while preserving Colab's existing PyTorch installation. The default configuration requires a CUDA GPU.

</details>

<details>
<summary><strong>3.2 Local Jupyter setup</strong></summary>

Example environment; this is not a pinned reproduction environment:

```bash
conda create -n mvtec-simclr python=3.11 -y
conda activate mvtec-simclr
```

Install a matching CUDA-enabled `torch` and `torchvision` build using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/), then install the notebook dependencies:

```bash
pip install timm kagglehub scikit-learn pandas scipy numpy pillow tqdm matplotlib jupyterlab
```

From your existing repository folder:

```bash
jupyter lab
```

Before executing the setup cell locally, set `use_google_drive: bool = False` and replace the non-Drive `PROJECT_ROOT` assignment with a writable local path, for example:

```python
PROJECT_ROOT = Path.cwd() / CFG.drive_folder
```

The original non-Drive branch still points to `/content`, so changing the flag alone is insufficient for a typical Windows setup. If multiprocessing causes problems, use `num_workers: int = 0` for the local run. Record configuration changes; SSL-related changes may invalidate checkpoint compatibility.

Verify GPU access:

```python
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

</details>

## 4. Dataset

The project uses [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad), which provides normal training images, normal and defective test images, and pixel-level defect annotations.

**Categories:** bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper.

<details>
<summary><strong>4.1 Automatic download</strong></summary>

When `manual_mvtec_root` is `None`, the notebook downloads the dataset using:

```python
import kagglehub
path = kagglehub.dataset_download("ipythonx/mvtec-ad")
print(path)
```

The notebook resolves the dataset root after downloading.

</details>

<details>
<summary><strong>4.2 Use an existing dataset</strong></summary>

Set the field in `Config` before running the setup cell:

```python
manual_mvtec_root: Optional[str] = "/path/to/mvtec-ad"
```

For Windows, use a raw string such as `r"S:\Datasets\mvtec-ad"`.

The root should contain category folders. For example, `bottle/` contains:

| Relative path | Contents |
|---|---|
| `bottle/train/good/` | Normal training images |
| `bottle/test/good/` | Normal test images |
| `bottle/test/<defect_type>/` | Defective test images |
| `bottle/ground_truth/<defect_type>/` | Corresponding defect masks |

Preserve the original category and defect folder names.

</details>

## 5. SSL Training and Checkpoints

<details>
<summary><strong>5.1 Prepare compatible checkpoints</strong></summary>

Use `Reference code.ipynb` to run the SSL training stage for each required seed: **42**, **123**, and **2026**. Keep the SSL configuration aligned with the corresponding final notebook.

The reference notebook can reuse compatible checkpoints or resume training. For a deliberately fresh training experiment, set `resume_ssl=False` and `reuse_compatible_ssl_checkpoint=False`, and use a separate output location to preserve previous runs.

The final notebooks search recursively beneath `PROJECT_ROOT` for:

```text
ssl_best_checkpoint.pt
```

Compatibility checks include the seed, backbone, preprocessing/augmentation settings, projection dimensions, training batch size, learning rates, temperature, and other SSL configuration fields. A weights file without the expected checkpoint structure and configuration metadata is insufficient.

No public checkpoint download URL is supplied here. Make the matching checkpoint available under the project root before executing a final notebook.

</details>

<details>
<summary><strong>5.2 Default SSL configuration</strong></summary>

| Parameter | Value |
|---|---|
| Backbone | `vit_base_patch16_224` |
| External pretrained weights | Disabled: `pretrained=False` |
| Input size | 224 × 224 |
| Projection hidden / output dimensions | 4096 / 256 |
| Maximum epochs / early-stopping patience | 200 / 30 |
| Contrastive batch size | 16 |
| Gradient accumulation | 2 steps |
| Optimizer | AdamW |
| Encoder / projector learning rate | `1e-4` / `3e-4` |
| Encoder / projector weight decay | `1e-2` / `1e-4` |
| Warmup epochs | 10 |
| NT-Xent temperature | 0.10 |
| Gradient clipping | 1.0 |
| Mixed precision | Enabled |

Gradient accumulation changes the optimizer update batch, but contrastive negatives are formed within each micro-batch.

</details>

## 6. Run the Final Experiments

1. Prepare compatible SSL checkpoints for all three seeds under the same project root.
2. Open `Code1_SEED_42_MVTec.ipynb` and verify storage and dataset paths.
3. Execute the notebook cells in order. Protocol A builds reference banks, calibrates scores, and evaluates detection/localization; Protocol B subsequently trains and evaluates frozen-feature classifiers.
4. Repeat with `Code2_SEED_123_MVTec.ipynb` and `Code3_SEED_2026_MVTec.ipynb`.
5. Run the multi-seed aggregation cell after all runs are available. Confirm that its run table includes all three seeds before reporting a three-seed summary.

**Keep the final scientific settings fixed across seeds.** Leave `final_eval_only=True`, `quick_test=False`, and the fixed aggregation unchanged. These notebooks are evaluation-only for the SSL encoder; they still fit downstream components, including Protocol B linear heads.

## 7. Protocol A — Detection and Localization

### Fixed detector configuration

| Setting | Value |
|---|---|
| Primary detector | `local_patch` |
| Transformer blocks | 6, 9, 12 (`patch_layer_indices=(5, 8, 11)`) |
| Patch projection dimension | 256 |
| Patch prototypes per category | 1024 |
| Patch kNN / global kNN | `k=3` / `k=3` |
| Spatial penalty | 0.00 |
| Image aggregation | `logsumexp_t4` |
| Normal calibration quantile | 0.95 |
| Localization map | `raw_patch_knn` |
| Secondary hybrid | 25% global + 75% local |
| AUPRO maximum false-positive rate | 0.30 |
| Bootstrap iterations | 1000 |

Official normal training images are split into **80% SSL training and 20% SSL validation**. The validation pool is subdivided into **30% detector tuning and 70% detector calibration**. In the final notebooks, the synthetic branch audits the already-fixed aggregation rule.

Image-level detection uses position-wise robust patch calibration, fixed aggregation, and normal-only image calibration. Localization uses raw patch kNN distances to form spatial anomaly maps, separately from image-score calibration.

Evaluation includes image ROC-AUC, average precision, threshold-based classification metrics, pixel ROC-AUC, pixel average precision, and AUPRO. Category-level and pooled results should be labeled separately.

> Earlier development results on the official MVTec test set had already been inspected. The final runs are fixed-configuration reproducibility experiments on MVTec; an untouched external benchmark is still needed for confirmatory generalization.

## 8. Protocol B — Frozen-Feature Classification

Protocol B creates a separate supervised downstream split inside the official test set, grouped by `category::defect_type`, with target proportions of **60% training / 20% validation / 20% testing**. Actual group counts follow the notebook's small-group handling.

The encoder remains frozen. Linear heads learn category, binary anomaly status, and defect-type predictions. The default feature mode is `global_plus_patch_stats`, and defect prediction uses a category-constrained hierarchical rule.

| Setting | Value |
|---|---|
| Maximum probe epochs | 200 |
| Batch size | 128 |
| Learning rate | `1e-3` |
| Weight decay | `1e-4` |
| Early-stopping patience | 30 |
| Category / anomaly / defect loss weights | 1.0 / 1.0 / 1.0 |

Report these results as **supervised downstream classification**, separately from the normal-only Protocol A benchmark.

## 9. Results and Generated Files

Each run uses a version tag, seed, and configuration hash in its directory name. With the default Drive configuration, outputs are beneath:

```text
/content/drive/MyDrive/MVTec_SimCLR_ViT_PATCH_PUBLICATION/<run_name>/
```

| Folder | Contents |
|---|---|
| `metadata/` | Dataset indices and split CSV files |
| `cache/` | Cached representations |
| `results/models/` | SSL and downstream checkpoints |
| `results/patch_banks/` | Category-specific patch prototypes |
| `results/tables/` | Predictions, metrics, and training histories |
| `results/figures/` | Generated plots |

<details>
<summary><strong>Key files to inspect</strong></summary>

| File | Purpose |
|---|---|
| `protocolA_official_test_predictions.csv` | Image-level Protocol A predictions |
| `protocolA_primary_per_category_metrics.csv` | Primary detector metrics by category |
| `protocolA_per_category_pixel_metrics.csv` | Localization metrics by category |
| `protocolA_ablation_metrics.csv` | Global/local/hybrid comparison |
| `protocolA_v5p1_publication_macro_summary.csv` | Macro summary |
| `protocolA_primary_bootstrap_95ci.csv` | Bootstrap uncertainty estimates |
| `protocolB_downstream_test_predictions.csv` | Held-out supervised predictions |
| `protocolB_multiclass_summary.csv` | Downstream classification summary |

The aggregation cell saves these files directly under `PROJECT_ROOT`:

- `publication_multiseed_v5p1_FINAL_FIXED_LOGSUMEXP_protocolA_runs.csv`
- `publication_multiseed_v5p1_FINAL_FIXED_LOGSUMEXP_protocolA_summary.csv`

Use the complete final-run exports to populate publication tables. Keep seed variability and bootstrap confidence intervals distinct.

</details>

## 10. Figures and Visual Results

Figures can be added later in an `images/` folder at the repository root. Suggested filenames:

| Image | Content |
|---|---|
| `framework_overview.png` | Overall method and protocol diagram |
| `ssl_training_loss.png` | SSL training and validation loss |
| `protocol_a_score_distribution.png` | Normal/anomalous score distributions |
| `detector_ablation.png` | Global, local, and hybrid comparison |
| `localization_examples.png` | Input images, masks, and anomaly maps |
| `protocol_b_confusion_matrix.png` | Downstream classification results |

<!-- Upload the images, then remove this comment wrapper to display them.
### SSL Training
![SSL training and validation loss](images/ssl_training_loss.png)

### Anomaly Scores
![Protocol A score distributions](images/protocol_a_score_distribution.png)

### Detector Ablation
![Global, local, and hybrid detector comparison](images/detector_ablation.png)

### Localization
![Inputs, ground-truth masks, and anomaly maps](images/localization_examples.png)

### Downstream Classification
![Protocol B confusion matrix](images/protocol_b_confusion_matrix.png)
-->

## 11. Troubleshooting

| Problem | What to check |
|---|---|
| No compatible SSL checkpoint | Confirm the seed, checkpoint configuration, and search location under `PROJECT_ROOT`; generate the checkpoint with the reference workflow if needed. |
| CUDA unavailable | Enable a Colab GPU or install the appropriate CUDA-enabled PyTorch build. |
| GPU out of memory | Close other GPU workloads; reduce evaluation batch size if needed and record the change. Changing SSL batch size creates a different training configuration. |
| Dataset root not found | Set `manual_mvtec_root` to the directory containing all category folders. |
| Drive mount fails locally | Disable Drive and edit the non-Drive `PROJECT_ROOT` assignment before executing setup. |
| Embedding cache mismatch | Regenerate caches through the notebook; preserve strict cache validation. |
| Missing seed in summary | Complete the missing run under the shared project root and rerun aggregation. |
| Images do not display | Match the filename and case exactly, and remove the surrounding HTML comment markers. |

## 12. Reproducibility Notes

- Preserve split manifests, configuration metadata, environment versions, and original checkpoint provenance.
- Keep the primary detector, aggregation, thresholds, and ablation definitions consistent across final runs.
- Use official test labels and masks for evaluation in Protocol A; Protocol B has its own explicitly supervised split.
- Retain all three seed results and report their variability.
- Report runtime measurements with hardware and batch-size details.

## Reproducing paper figures

Standalone scripts under `figure_scripts/` regenerate manuscript Figures 4–7 into `figures/` as:

- **PDF** — manuscript version
- **SVG** — editable vector backup
- **PNG** — 300-dpi preview

```bash
python figure_scripts/fig4_training_and_scores.py
python figure_scripts/fig5_global_local_hybrid.py
python figure_scripts/fig6_localization_examples.py --data-root /path/to/mvtec-ad --checkpoint-root checkpoints
python figure_scripts/fig7_pixel_roc_auc.py
```

Or generate all available figures from archived `outputs/` CSVs:

```bash
python figure_scripts/generate_all.py
```

| Figure | Scope |
|---|---|
| Fig. 4 | Representative seed 42 only |
| Fig. 5 | Three-seed aggregation |
| Fig. 6 | Qualitative seed 42 only (needs MVTec/checkpoint, or a verified final asset) |
| Fig. 7 | Three-seed aggregation |

Figures 4, 5, and 7 are intended to reproduce from CSV tables under `outputs/` without re-running the neural network. Figure 6 uses raw-patch localization and may require the MVTec dataset plus the seed-42 checkpoint.

## 13. References and Citation

- [MVTec AD dataset and original publication information](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
- [PyTorch installation documentation](https://pytorch.org/get-started/locally/)

<!-- Add the finalized paper title, authors, venue/year, DOI or preprint URL, and BibTeX here when available. -->

If you use this repository, please cite the associated paper/software.
Citation metadata are available in [`CITATION.cff`](CITATION.cff).
Acknowledge the MVTec AD dataset when applicable.

## 14. Contributing

Suggestions, bug reports, and reproducibility improvements are welcome. When reporting an issue, include the notebook name, seed, runtime environment, configuration changes, and relevant error message.

<!-- Optional contributors image: replace YOUR_GITHUB_USERNAME, then remove this comment wrapper.
<p align="center">
  <img src="https://contrib.rocks/image?repo=YOUR_GITHUB_USERNAME/ViT_Anomaly_Detection_SimCLR" alt="Project contributors">
</p>
-->

<!-- Add a license section and badge after choosing and committing a LICENSE file. -->
