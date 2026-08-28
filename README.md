# MINT

## Structure

- `infer.py` — entry point (single experiment or `--run-all`)
- `mint_engine/` — modules: architecture, models, datasets, features, metrics, etc.
- `checkpoints/` — trained weights per method, per fold
- `out/` — per-fold predictions and logs

## Run

```bash
python infer.py --run-all \
  --ckpt-dir checkpoints --manifest test/test_manifest.csv \
  --prot-dir features/protein --lig-dir features/ligand --out-dir out
```

Single experiment:

```bash
python infer.py --exp <method> --kind pocketnet|mlp \
  --ckpt-dir checkpoints/<method> --manifest test/test_manifest.csv \
  --prot-dir features/protein --lig-dir features/ligand --out-dir out
```

## Results (5-fold mean AUC)

| method | AUC |
|---|---|
| MINT D+A | 0.9320 |
| MINT w/o A | 0.9002 |
| MINT w/o D | 0.9159 |
| MLP D+A | 0.8920 |
| MLP w/o A | 0.8697 |
| MLP w/o D | 0.7903 |

## Requirements

Python 3.9; torch, torch_geometric, sklearn, pandas, numpy.
