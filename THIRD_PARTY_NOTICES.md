# Third-party notices

## PoinTr

This repository builds on PoinTr (https://github.com/yuxumin/PoinTr), used at commit `3f98676aeaabd9c1150dd04cd0a51f504949ab9c` under the MIT licence, Copyright (c) 2021 Xumin Yu; the full text is in `LICENSE`.

- PoinTr files included unchanged: `cfgs/dataset_configs/PCN.yaml`, `cfgs/dataset_configs/ShapeNet-55.yaml`, `datasets/ShapeNet55Dataset.py`, `datasets/__init__.py`, `datasets/build.py`, `datasets/io.py`, `models/AdaPoinTr.py`, `models/build.py`.
- PoinTr files included with our changes: `cfgs/ShapeNet55_models/AdaPoinTr.yaml`, `datasets/PCNDataset.py`, `datasets/data_transforms.py`, `models/__init__.py`, `requirements.txt`, `tools/builder.py`, `tools/runner.py`, `utils/parser.py`, and `cfgs/dataset_configs/PCN_npy.yaml` (a copy of PoinTr's `cfgs/dataset_configs/PCN.yaml` that reads `.npy` files).
- Our files that build on PoinTr code: `models/AdaPoinTr_DiffCond.py` (imports and extends AdaPoinTr) and the AdaPoinTr configuration files added in `cfgs/ShapeNet55_models/`.

The rest of the PoinTr framework (for example `main.py`, the other files in `utils/`, and `extensions/`) is not included; `REPRO.md` explains how to combine this repository with a PoinTr checkout at the commit above.

## GRNet

PoinTr's dataset code in `datasets/data_transforms.py` and `datasets/PCNDataset.py` is adapted from GRNet (https://github.com/hzxie/GRNet), MIT licence, Copyright (c) 2020 Haozhe Xie. The permission notice is the MIT text reproduced in `LICENSE`.

## ODGNet

The UpTrans experiments use `models/pointnet.py` and `models/UpTrans.py` from ODGNet (https://github.com/corecai163/ODGNet; Cai et al., AAAI 2024). The ODGNet repository states no licence, so neither file is included; `third_party/ODGNET.md` and `third_party/fetch_odgnet.py` obtain them from ODGNet commit `032bca98f89cc3ea4733b331ff91468df1bea0f9` and apply our five-line change. `models/UpTrans_DiffCond.py` is our code: it subclasses ODGNet's UpTrans classes and repeats 14 short statements from them in overridden methods. Those statements, and the five ODGNet lines that the patch replaces (it carries no other ODGNet lines), remain ODGNet's; the MIT licence of this repository does not cover them. The UpTrans configuration files (`cfgs/*/UpTrans*.yaml`) follow ODGNet's settings, with our changes noted in their comments.

## Datasets

ShapeNet (https://huggingface.co/ShapeNet; terms of use: https://huggingface.co/spaces/ShapeNet/README/blob/main/TERMS.md) and PCN (https://github.com/wentaoyuan/pcn, whose data are generated from ShapeNetCore.v1) are not included. `LICENSES.md` explains how files with values derived from them are covered.
