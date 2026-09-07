# Reproducibility notes

Derived from the executed final seed notebooks:

- `notebooks/Code1_SEED_42_MVTec.ipynb`
- `notebooks/Code2_SEED_123_MVTec.ipynb`
- `notebooks/Code3_SEED_2026_MVTec.ipynb`

## Reviewer / notebook benchmark environment

| Item | Value |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB |
| CUDA runtime | 12.8 |
| PyTorch | 2.11.0+cu128 |

Install a matching CUDA-enabled PyTorch build from the official selector, then:

```bash
pip install -r requirements.txt
```

## Best SSL checkpoint epochs

| Seed | Best SSL epoch |
|-----:|---------------:|
| 42 | 192 |
| 123 | 164 |
| 2026 | 184 |

## Scientific Protocol A freeze (unchanged)

- Backbone: `vit_base_patch16_224`, `pretrained=False`
- Patch layers: blocks 6/9/12 → indices `(5, 8, 11)`
- Projection: 2304 → 256 fixed QR (`seed + 9107`)
- Prototypes: 1024 / category (`MiniBatchKMeans`)
- Patch kNN: `k=3`
- Primary aggregation: `logsumexp_t4` (beta=4)
- Image threshold quantile: `q=0.95`
- Primary detector: `local_patch`
- Localization: **raw** patch kNN distances (no position calibration)
- Hybrid 25/75: secondary ablation only

## Package entry point

```bash
python scripts/run_protocol_a.py --config configs/seed42.yaml
```

Notebooks remain the audit trail and source of truth for the modular `src/` package.
