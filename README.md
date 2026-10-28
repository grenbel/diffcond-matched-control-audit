# Naïve Difficulty Conditioning Fails on ShapeNet-55: A Matched-Control Audit

Code, configurations, logs and result artefacts for the paper by Wenbo Gong, *Transactions on Machine Learning Research* (2026).

- Paper and reviews: https://openreview.net/forum?id=cpe5PHAgoQ
- Checkpoints: https://huggingface.co/spacetime1008/diffcond-matched-control-audit

The paper audits a naïve difficulty-conditioned extension of AdaPoinTr (DiffCond) on ShapeNet-55. The double-digit per-bin CD-ℓ2 gains first observed for it arise when the warm-started extension is compared with a from-scratch baseline, and they disappear under a matched warm-start control. The paper proposes a stratified matched-control protocol; under it, this DiffCond extension underperforms its matched AdaPoinTr control on ShapeNet-55 and shows no large gain in the tested UpTrans analogue.

## Contents

| Path | What |
|---|---|
| `cfgs/` | configurations of the paper's runs and of the released checkpoints (AdaPoinTr, DiffCond, ablations, specialists, UpTrans, PCN) |
| `models/` | `AdaPoinTr_DiffCond.py`, `UpTrans_DiffCond.py` and the PoinTr model files they need; ODGNet's two files are fetched, see `third_party/` |
| `datasets/`, `tools/runner.py`, `tools/builder.py`, `utils/parser.py` | the patched PoinTr training and evaluation pipeline |
| `tools/` | analysis, evaluation, audit and data-preparation scripts |
| `data/results/` | the aggregate artefacts behind the tables of the paper and the supplementary material, and the provenance record of the added-seed ablation run |
| `data/PCN/difficulty_proxy.json` | per-sample difficulty labels for the PCN sanity check |
| `experiments/` | per-seed S/M/H evaluation summaries of the UpTrans runs |
| `logs/` | the 212 retained experiment logs, described in `logs/README.md` |
| `third_party/` | how to obtain ODGNet's two files, with our five-line patch |
| `REPRO.md` | full reproduction notes |
| `supplementary.pdf` | the supplementary material of the paper (camera-ready version) |
| `checkpoints.sha256` | SHA-256 of the 26 checkpoints on Hugging Face |
| `release_manifest.sha256` | SHA-256 of every other file in this repository |

Some log paths are about 160 characters long. On Windows, clone into a short folder or with `git -c core.longpaths=true clone ...`.

## Regenerate the headline numbers without a GPU

Python 3 is enough for the first three scripts; the fourth also needs NumPy. Each script rewrites its output in `data/results/`.

```bash
python tools/aggregate_results.py   # cross-backbone per-seed table (main Table 4, supplementary Table 10)
python tools/paired_stats.py        # paired-seed confidence intervals (main Table 4)
python tools/gap_a_results.py --all-results data/results/all_results.json \
    --logs "data/results/gap_a_provenance/eval_logs/*.log" \
    --train-logs "data/results/gap_a_provenance/train_logs/*.log" \
    --meta-dir data/results/gap_a_provenance/meta \
    --out data/results/gap_a_results.json   # three-seed ablation (main Table 3), strict provenance checks
python tools/crop_path_quantification.py --out data/results/crop_path_quantification.json   # Section 3.3
```

The shipped outputs were written on Windows; rerunning the four scripts there reproduces them byte for byte. On Linux we reran the first three: they give the same content with LF instead of CRLF line endings, and `gap_a_results.json` lists its 16 log paths with `/` instead of `\`, so `git status` marks the four output files as modified while `git diff --ignore-cr-at-eol` shows changes on those 16 lines only.

## Full reproduction

Combine this repository with upstream PoinTr at commit `3f98676aeaabd9c1150dd04cd0a51f504949ab9c`, fetch ODGNet's two files, build the CUDA extensions and train with PoinTr's `main.py`. `REPRO.md` gives every step, the recorded software stack and the GPU budget (about 99 GPU-hours for the cross-backbone pipeline). In short:

```bash
git clone https://github.com/yuxumin/PoinTr.git && cd PoinTr
git checkout 3f98676aeaabd9c1150dd04cd0a51f504949ab9c
# copy this repository's files over the clone, then
python third_party/fetch_odgnet.py
# build the CUDA extensions as in REPRO.md, then for example
python main.py --config cfgs/ShapeNet55_models/AdaPoinTr_baseline_smoke30_diffcrop.yaml \
    --exp_name vanilla_seed42 --seed 42 --val_freq 10 \
    --start_ckpts <path-to>/baseline_300ep/ckpt-best.pth
```

## Checkpoints

The 26 retained AdaPoinTr-family checkpoints of the supplementary manifest (Table 14) are on Hugging Face, byte-identical to that manifest; `checkpoints.sha256` lists their SHA-256.

| File in the Hugging Face repository | Role in the paper | Seed | Epoch | Configuration (`cfgs/ShapeNet55_models/`) | Size (MB) |
|---|---|---|---|---|---|
| `baseline_300ep/ckpt-best.pth` | Baseline (300-epoch from-scratch AdaPoinTr; warm-start source) | – | best | `AdaPoinTr.yaml` | 389.5 |
| `vanilla_ft/seed42/ckpt-epoch-030.pth` | Vanilla-FT (matched control) | 42 | 30 | `AdaPoinTr_baseline_smoke30_diffcrop.yaml` | 389.5 |
| `vanilla_ft/seed43/ckpt-epoch-030.pth` | Vanilla-FT (matched control) | 43 | 30 | `AdaPoinTr_baseline_smoke30_diffcrop.yaml` | 389.5 |
| `vanilla_ft/seed44/ckpt-epoch-030.pth` | Vanilla-FT (matched control) | 44 | 30 | `AdaPoinTr_baseline_smoke30_diffcrop.yaml` | 389.5 |
| `diffcond_full/seed42/ckpt-epoch-030.pth` | DiffCond (full) | 42 | 30 | `AdaPoinTr_DiffCond_smoke30.yaml` | 396.9 |
| `diffcond_full/seed43/ckpt-epoch-030.pth` | DiffCond (full) | 43 | 30 | `AdaPoinTr_DiffCond_smoke30.yaml` | 396.9 |
| `diffcond_full/seed44/ckpt-epoch-030.pth` | DiffCond (full) | 44 | 30 | `AdaPoinTr_DiffCond_smoke30.yaml` | 396.9 |
| `ablation_v1_lambda0/seed42/ckpt-epoch-030.pth` | Ablation v1-λ0 (CE off, ranking on) | 42 | 30 | `AdaPoinTr_DiffCond_v1_lambda0.yaml` | 395.9 |
| `ablation_v1_lambda0/seed43/ckpt-epoch-030.pth` | Ablation v1-λ0 (CE off, ranking on) | 43 | 30 | `AdaPoinTr_DiffCond_v1_lambda0.yaml` | 395.9 |
| `ablation_v1_lambda0/seed44/ckpt-epoch-030.pth` | Ablation v1-λ0 (CE off, ranking on) | 44 | 30 | `AdaPoinTr_DiffCond_v1_lambda0.yaml` | 395.9 |
| `ablation_v2b/seed42/ckpt-epoch-030.pth` | Ablation v2b (CE off, ranking off) | 42 | 30 | `AdaPoinTr_DiffCond_v2b.yaml` | 395.9 |
| `ablation_v2b/seed43/ckpt-epoch-030.pth` | Ablation v2b (CE off, ranking off) | 43 | 30 | `AdaPoinTr_DiffCond_v2b.yaml` | 395.9 |
| `ablation_v2b/seed44/ckpt-epoch-030.pth` | Ablation v2b (CE off, ranking off) | 44 | 30 | `AdaPoinTr_DiffCond_v2b.yaml` | 395.9 |
| `hard_specialist/seed42/ckpt-epoch-030.pth` | Hard-Specialist | 42 | 30 | `AdaPoinTr_hard_specialist.yaml` | 389.5 |
| `hard_specialist/seed43/ckpt-epoch-030.pth` | Hard-Specialist | 43 | 30 | `AdaPoinTr_hard_specialist.yaml` | 389.5 |
| `hard_specialist/seed44/ckpt-epoch-030.pth` | Hard-Specialist | 44 | 30 | `AdaPoinTr_hard_specialist.yaml` | 389.5 |
| `easywide_specialist/seed42/ckpt-epoch-030.pth` | EasyWide-Specialist | 42 | 30 | `AdaPoinTr_easywide_specialist.yaml` | 389.5 |
| `easywide_specialist/seed43/ckpt-epoch-030.pth` | EasyWide-Specialist | 43 | 30 | `AdaPoinTr_easywide_specialist.yaml` | 389.5 |
| `easywide_specialist/seed44/ckpt-epoch-030.pth` | EasyWide-Specialist | 44 | 30 | `AdaPoinTr_easywide_specialist.yaml` | 389.5 |
| `from_scratch_30ep_rerun/seed42/ckpt-best.pth` | From-scratch 30-epoch baseline rerun (Pitfall 2 audit) | 42 | best | `AdaPoinTr_baseline_smoke30.yaml` | 389.5 |
| `from_scratch_30ep_rerun/seed42/ckpt-epoch-029.pth` | From-scratch 30-epoch baseline rerun (Pitfall 2 audit) | 42 | 29 | `AdaPoinTr_baseline_smoke30.yaml` | 389.5 |
| `from_scratch_30ep_rerun/seed42/ckpt-epoch-030.pth` | From-scratch 30-epoch baseline rerun (Pitfall 2 audit) | 42 | 30 | `AdaPoinTr_baseline_smoke30.yaml` | 389.5 |
| `from_scratch_30ep_rerun/seed42/ckpt-last.pth` | From-scratch 30-epoch baseline rerun (Pitfall 2 audit) | 42 | last | `AdaPoinTr_baseline_smoke30.yaml` | 389.5 |
| `non_paper/vanilla_stock_crop/ckpt-epoch-030.pth` | Not used in the paper: early vanilla fine-tune, stock crop path | – | 30 | `AdaPoinTr_baseline_smoke30.yaml` | 389.5 |
| `non_paper/easy_specialist_pilot/ckpt-epoch-030.pth` | Not used in the paper: Easy-Specialist pilot | – | 30 | `AdaPoinTr_easy_specialist.yaml` | 389.5 |
| `non_paper/median_specialist_pilot/ckpt-epoch-030.pth` | Not used in the paper: Median-Specialist pilot | – | 30 | `AdaPoinTr_median_specialist.yaml` | 389.5 |

- DiffCond and the two ablation conditions load into `models/AdaPoinTr_DiffCond.py`; all other checkpoints load into upstream PoinTr's `models/AdaPoinTr.py`. The configuration column names the file each checkpoint was trained with; `REPRO.md` shows how to evaluate a checkpoint on the S/M/H bins.
- Every fine-tuned checkpoint was warm-started from `baseline_300ep/ckpt-best.pth`; `baseline_300ep/` and `from_scratch_30ep_rerun/` were trained from scratch.
- The `metrics` and `best_metrics` stored in each file are training-time validation values under PoinTr's random-crop validation proxy, not the paper's S/M/H results (Section 3.1 of the paper explains the difference).
- The UpTrans and PCN checkpoints were not retained; those runs are documented by their configurations, logs and per-seed evaluation summaries.

## Logs

`logs/` holds the 212 experiment logs of the supplementary log inventory (Table 15); the 11 LaTeX compile logs of that inventory are not experiment logs and are not released. `logs/README.md` maps each inventory row to its folder and explains the run names; `logs/REDACTIONS.md` lists the four lines in which a tool name was removed.

## Licences

Code: MIT (`LICENSE`), including the PoinTr files it builds on but not the few ODGNet lines named in `LICENSES.md`. Logs and our result artefacts: CC BY 4.0. Checkpoints: CC BY-NC 4.0. ShapeNet- and PCN-derived content stays under ShapeNet's terms of use. Details in `LICENSES.md` and `THIRD_PARTY_NOTICES.md`.

## Citation

```bibtex
@article{gong2026naive,
  title   = {Na{\"i}ve Difficulty Conditioning Fails on {ShapeNet-55}: A Matched-Control Audit},
  author  = {Gong, Wenbo},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://openreview.net/forum?id=cpe5PHAgoQ}
}
```

## Acknowledgements

This work builds on PoinTr and AdaPoinTr (Yu et al.), uses UpTrans from ODGNet (Cai et al.), and uses the ShapeNet and PCN datasets.
