# Data directory

This folder is a placeholder for local MVTec AD data.

**Do not commit MVTec images** to Git. Dataset files under `data/mvtec/` are
gitignored.

## Obtain MVTec AD

Download MVTec AD from the official source:

https://www.mvtec.com/research-teaching/datasets/mvtec-ad

Do not redistribute dataset images automatically from this repository.

## Expected structure

Point `DATA_ROOT` / `data_root` at the directory that contains category folders:

```text
data/mvtec/
    bottle/
        train/good/
        test/good/
        test/<defect_type>/
        ground_truth/<defect_type>/
    cable/
    capsule/
    carpet/
    grid/
    hazelnut/
    leather/
    metal_nut/
    pill/
    screw/
    tile/
    toothbrush/
    transistor/
    wood/
    zipper/
```

Example:

```bash
export DATA_ROOT=/path/to/mvtec-ad
python scripts/run_protocol_a.py --config configs/seed42.yaml --data-root "$DATA_ROOT"
```
