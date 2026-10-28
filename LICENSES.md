# Licences

Different parts of this release carry different licences.

| What | Where | Licence |
|---|---|---|
| Our code, scripts and patch | `models/AdaPoinTr_DiffCond.py`, `models/UpTrans_DiffCond.py`, `tools/` (except `tools/builder.py` and `tools/runner.py`), `third_party/`, the configuration files we added in `cfgs/` | MIT (`LICENSE`) for the code we wrote; see the ODGNet note below |
| PoinTr files, unchanged or changed by us | listed in `THIRD_PARTY_NOTICES.md` | MIT (`LICENSE`), with PoinTr's copyright notice |
| ODGNet's `models/pointnet.py` and `models/UpTrans.py` | not included | obtained from ODGNet, which states no licence (`third_party/ODGNET.md`) |
| Logs | `logs/` | CC BY 4.0 |
| Results and documentation we wrote | `data/results/`, `experiments/`, `README.md`, `REPRO.md`, `supplementary.pdf` | CC BY 4.0 |
| Per-sample difficulty labels for PCN samples | `data/PCN/difficulty_proxy.json` (keyed by ShapeNet model ids; PCN data are generated from ShapeNetCore.v1) | ShapeNet's terms of use (non-commercial research and education); no other rights are granted |
| Model checkpoints | Hugging Face repository linked in `README.md` | CC BY-NC 4.0; ShapeNet's terms of use apply as well |

ODGNet note: the 14 short statements of ODGNet's classes that `models/UpTrans_DiffCond.py` repeats (`THIRD_PARTY_NOTICES.md`) and the five ODGNet lines that `third_party/odgnet_uptrans.patch` replaces remain ODGNet's; this release grants no licence for them.

CC BY 4.0 covers our own text and measurements in the logs and results. Dataset-derived content in them, such as ShapeNet model ids, stays under the dataset's terms; the model ids are identifiers only, and nothing in this release licenses ShapeNet or PCN data.

CC BY 4.0: https://creativecommons.org/licenses/by/4.0/ · CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/ · ShapeNet terms of use: https://huggingface.co/spaces/ShapeNet/README/blob/main/TERMS.md
