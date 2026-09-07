# SSL checkpoints (not committed)

This directory stores seed-specific SimCLR / ViT checkpoints used by the
final Protocol A notebooks and by `scripts/run_protocol_a.py`.

Checkpoint files (`*.pt` / `*.pth` / `*.ckpt`) are **gitignored** and must
not be committed to normal Git history.

## Expected layout

```text
checkpoints/
  seed42/ssl_best_checkpoint.pt
  seed123/ssl_best_checkpoint.pt
  seed2026/ssl_best_checkpoint.pt
```

## Best SSL epochs (executed final notebooks)

| Seed | Best SSL epoch |
|-----:|---------------:|
| 42   | 192            |
| 123  | 164            |
| 2026 | 184            |

These epochs were observed in the executed Colab runs on NVIDIA A100-SXM4-80GB
(CUDA 12.8, PyTorch 2.11.0+cu128).

## Download

Public checkpoint download URL will be added after archival release.

Until then, produce compatible checkpoints with `notebooks/Reference code.ipynb`
using the matching seed and SSL configuration, or place existing
`ssl_best_checkpoint.pt` files into the paths above.
