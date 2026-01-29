# GEPC‑Diffusion

**GEPC** (Group‑Equivariant Posterior Consistency) is a **training‑free** out‑of‑distribution (OOD) score computed from a **pretrained unconditional diffusion backbone** (OpenAI *improved‑diffusion* style UNet).

This repository provides:

- **Standard image OOD** benchmarks (CIFAR/SVHN/CelebA/DTD/Places/SUN, etc.).
- **SAR (user data)** benchmarks using `torchvision.datasets.ImageFolder`.

> **Backbones in this repo**
> - **OpenAI improved-diffusion (official checkpoints)**: e.g., **LSUN Bedroom 256×256**
> - **Third-party checkpoint trained with the improved-diffusion codebase**: **DiffPath CelebA 32×32**

---

## Repository layout

```
.
├── checkpoints/                # diffusion checkpoints (downloaded or user‑provided)
├── configs/                    # YAML runs
│   ├── gepc_celeba.yaml
│   ├── gepc_cifar10.yaml
│   ├── gepc_sar_256.yaml
│   └── gepc_svhn.yaml
├── gepc/
│   ├── adapters/               # diffusion backbone adapters
│   │   └── improved.py
│   ├── datasets/               # dataset loaders
│   │   └── images.py
│   ├── methods/                # GEPC implementation
│   │   └── gepc.py
│   └── utils/
│       └── metrics.py
├── results/                    # outputs (auto‑created)
├── scripts/
│   ├── bench_gepc_images.py    # standard image benchmarks
│   └── bench_ood_sar.py        # SAR ImageFolder benchmarks
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If you want editable installs:

```bash
pip install -e .
```

---

## Backbone dependency (improved‑diffusion)

The default adapter is `adapter: improved` and expects the *improved‑diffusion* codebase to be importable.

Typical options:

1) **Add improved‑diffusion to `PYTHONPATH`**:

```bash
export PYTHONPATH=/path/to/improved-diffusion:$PYTHONPATH
```

2) Or **vendor/clone it next to this repo** and point `PYTHONPATH` accordingly.

If imports fail, run:

```bash
python -c "import improved_diffusion; print('ok')"
```

---

## Checkpoints

This repo does **not** ship diffusion checkpoints. Download them manually, place them under `./checkpoints/`,
and reference them from YAML with `model_path:`.

> **Which checkpoint for what?**
> - **Standard image OOD (CIFAR/SVHN/CelebA 32×32)**: we use the **DiffPath CelebA 32×32** checkpoint (trained with the improved-diffusion codebase).
> - **SAR 256×256**: we use an **official OpenAI improved-diffusion** checkpoint (e.g., **LSUN Bedroom 256×256**).

### 1) DiffPath CelebA 32×32 (third-party, improved-diffusion compatible)

Download and save it directly into `./checkpoints/`:

```bash
mkdir -p checkpoints
wget -O checkpoints/celeba_ema_0.9999_499999.pt \
  https://huggingface.co/ajrheng/diffpath/resolve/main/celeba_ema_0.9999_499999.pt
```

YAML example (used by standard image configs):

```yaml
adapter: improved
model_path: checkpoints/celeba_ema_0.9999_499999.pt
```

### 2) OpenAI improved-diffusion official checkpoint (LSUN Bedroom 256×256)

Download the LSUN Bedroom checkpoint and save it under the name expected by the provided configs:

```bash
mkdir -p checkpoints
wget -O checkpoints/lsun_uncond_100M_2400K_bs64.pt \
  https://openaipublic.blob.core.windows.net/diffusion/jul-2021/lsun_bedroom.pt
```

YAML example (used by SAR config):

```yaml
adapter: improved
model_path: checkpoints/lsun_uncond_100M_2400K_bs64.pt
```

> **Important:** `improved_args` in the YAML must match the checkpoint architecture (e.g., `num_channels`, `num_res_blocks`, `learn_sigma`, etc.).
> If you swap checkpoints, update `improved_args` accordingly.

---

## Quickstart: standard image OOD

Run a config as‑is:

```bash
python scripts/bench_gepc_images.py --config configs/gepc_cifar10.yaml --verbose
```

Useful overrides:

- `--data_dir` : override `data_root` from YAML
- `--in_dist`  : override the ID dataset name (reuses YAML limits/splits)
- `--out_dist` : evaluate only one OOD dataset from the YAML list
- `--device` / `--seed` / `--strict_determinism`

Example (run only CIFAR10 vs SVHN):

```bash
python scripts/bench_gepc_images.py \
  --config configs/gepc_cifar10.yaml \
  --out_dist svhn \
  --device 0 \
  --seed 1337 \
  --strict_determinism \
  --verbose
```

### Outputs

`bench_gepc_images.py` writes under:

```
results/gepc/<ID_NAME>/
  ├── config_used.yaml
  ├── main_results.json
  └── main_results_flat.json
```

---

## Datasets

Most torchvision datasets can be downloaded automatically when `download: true` in the YAML.

### CelebA manual download (if torchvision download fails)

`torchvision.datasets.CelebA` can fail in some environments (mirror issues / manual acceptance / connectivity).
If it fails:

1) Download CelebA manually from the official source.
2) Organize the folder as expected by torchvision:

```
<data_root>/celeba/
  ├── img_align_celeba/                # images
  ├── list_attr_celeba.txt
  ├── list_eval_partition.txt
  ├── identity_CelebA.txt              # (optional but common)
  ├── list_bbox_celeba.txt             # (optional but common)
  └── list_landmarks_align_celeba.txt  # (optional but common)
```

3) In your YAML, set:

```yaml
eval:
  ood:
    - { name: celeba, split: test, limit: 1000, download: false }
```

---

## Quickstart: SAR (ImageFolder)

SAR chips are loaded via `torchvision.datasets.ImageFolder`. Each split must be a valid ImageFolder with **one class** (e.g. `0`).

Expected layout:

```
./data/sar/HRSID_bg/train/0/*.png
./data/sar/HRSID_bg/test/0/*.png
./data/sar/HRSID_ship/test/0/*.png
```

Run:

```bash
python scripts/bench_ood_sar.py --config configs/gepc_sar_256.yaml --verbose
```

Optional (if supported by your script): save qualitative examples and score dumps:

```bash
python scripts/bench_ood_sar.py \
  --config configs/gepc_sar_256.yaml \
  --qual_dir results/qual_sar \
  --save_scores_npz results/scores_sar \
  --strict_determinism \
  --verbose
```

### Outputs

By default, SAR runs are stored next to the config:

```
configs/results_gepc_sar/<RUN_TAG>/
  ├── config_used.yaml
  └── metrics.json
```

If `--qual_dir` is enabled, the script exports per‑OOD folders containing:

- `*_raw.png` (grayscale SAR)
- `*_gepc.png` (heatmap)
- `*_overlay.png` (overlay)
- `*_map.npy` (raw GEPC map)

---

## Configuration (YAML)

All configs share the same high‑level fields:

- `image_size`: backbone input size (e.g. 32, 64, 256)
- `data_image_size`: dataset resize size (if different from backbone)
- `adapter`: should be `improved`
- `model_path`: path to checkpoint
- `improved_args`: UNet hyper‑params (must match the checkpoint)
- `batch_size`, `device`, `seed`, `strict_determinism`

### Standard images

Standard image configs use:

```yaml
data_root: ./data

eval:
  id_train: { name: cifar10, split: train, limit: 2000, download: true }
  id_test:  { name: cifar10, split: test,  limit: 1000, download: true }
  ood:
    - { name: svhn,     split: test, limit: 1000, download: true }
    - { name: celeba,   split: test, limit: 1000, download: true }
    - { name: cifar100, split: test, limit: 1000, download: true }

gepc:
  # GEPC hyper‑params (t selection, pooling, KDE calibration, etc.)
  ...
```

### SAR

SAR configs use ImageFolder roots:

```yaml
eval:
  id_train: { root: ./data/sar/HRSID_bg/train, limit: 500 }
  id_test:  { root: ./data/sar/HRSID_bg/test,  limit: 100 }
  ood:
    - { name: HRSID_ship, root: ./data/sar/HRSID_ship/test, limit: 100 }

gepc:
  ...
```

---

## Reproducibility tips

- Use `--strict_determinism` (or `strict_determinism: true` in YAML) for the most stable numbers.
- Keep `num_workers: 0` for SAR runs when exporting maps.

---

## Troubleshooting

- **ImportError: improved_diffusion**
  - Add the improved‑diffusion directory to `PYTHONPATH` (see above).

- **CelebA download fails**
  - Download manually and set `download: false` (see CelebA section).

- **CUDA OOM**
  - Reduce `batch_size` in YAML (and/or `gepc.internal_bs`).

---

## License / citation

If you use this repository in academic work, please cite the associated GEPC paper/artifact.
