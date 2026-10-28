# Provenance record of the added-seed ablation run

This directory is the provenance record of the three-seed ablation added
after review (main paper Table 3; seeds 43 and 44 for v1-lambda0 and v2b,
trained under the matched-control recipe with the same warm-start
checkpoint, 30-epoch schedule and crop path as the seed-42 runs).

## Contents

- `meta/M0_PASS.json` -- reproduction gate: before any training, the
  retained seed-42 v2b checkpoint was re-evaluated on the Hard bin and the
  result compared sample by sample with the original April log; the file
  also carries the dataset acceptance record, the code attestation of the
  working tree and the live environment snapshot.
- `meta/ACCEPTED.json` -- dataset acceptance record (official ShapeNet-55
  archive size / MD5 / SHA-256, fresh-unpack comparison, per-file manifest
  digest).
- `meta/code_attestation.json` -- SHA-256 of every source file of the
  authors' working tree at attestation time. The working tree contains
  files that are not part of this repository (pilot configurations and
  models outside the paper); their names appear here because the
  attestation covers the whole tree. Of the code and configuration files
  of this repository, 41 appear in the attestation: 6 equal the attested
  files and 35 are release copies whose SHA-256 differs from the attested
  value. Against the attested versions, the configuration files differ
  only in comments and line endings, except that the two PCN model
  configurations (`cfgs/PCN_models/UpTrans.yaml`,
  `cfgs/PCN_models/UpTrans_DiffCond.yaml`) take their dataset paths from
  `cfgs/dataset_configs/PCN_npy.yaml`, the `.npy` copy of the PCN data that
  the reported PCN runs read, which gives the configuration those runs
  logged (`REPRO.md`). The model, dataset and training code (`models/`,
  `datasets/`, `tools/runner.py`, `tools/builder.py`) differs only in line
  endings, comments, docstrings and messages, except that
  `models/__init__.py` no longer imports a development model absent from
  this repository. The analysis scripts additionally rename arguments,
  variables and result keys and change default paths and pinned digests
  to match this repository (for example, `tools/gap_a_results.py` reads
  the `vanilla_ft` / `diffcond_oracle` entries of this repository's
  `all_results.json` and pins that file's SHA-256), and
  `tools/cache_pcn_to_npy.py` takes as arguments the data paths and the
  worker count that the attested version fixed. The supplementary bundle
  reviewed with the paper held 39 attested files, 36 of them anonymised
  copies; this repository leaves out ODGNet's `models/pointnet.py` and
  `models/UpTrans.py`, which `third_party/fetch_odgnet.py` rebuilds and
  checks against their attested SHA-256, and adds `utils/parser.py` and
  the configurations `cfgs/ShapeNet55_models/AdaPoinTr_easy_specialist.yaml`
  and `AdaPoinTr_median_specialist.yaml`, all three equal to their attested
  versions, and `tools/cache_pcn_to_npy.py`.
- `meta/gap_a_env_manifest.json` -- Python / PyTorch / CUDA / cuDNN stack,
  compiled extensions and package list of the training host.
- `meta/train_meta_<cell>_seed<S>.json`, `meta/train_end_<cell>_seed<S>.json`
  -- per-run start and end sidecars (configuration digest, warm-start
  checkpoint digest at launch and at end, final checkpoint digest).
- `meta/M1_PASS.json` -- seal of the four trained checkpoints. The training
  host became unavailable during sealing, so the checkpoints were sealed
  and evaluated on a second host with the same GPU model and the same
  Python / PyTorch / CUDA / cuDNN stack and compiled extensions but a
  different driver and kernel version. The eleven recorded host
  differences (`accepted_differences`) are listed explicitly; the M0
  reproduction was repeated on the second host before evaluation.
- `meta/eval_meta_<cell>_seed<S>_<bin>.json` -- per-evaluation sidecars
  (checkpoint digest, log digest, bin, inference mode).
- `meta/M2_PASS.json` -- seal of the twelve evaluations.
- `eval_logs/` -- the twelve official S/M/H evaluation logs (2 cells x 2
  seeds x 3 bins); their SHA-256 digests equal the sidecar values.
- `train_logs/` -- the four training logs; their digests equal the values
  sealed in `M1_PASS.json`.

## Regenerating the matrix

From the repository root:

```bash
python tools/gap_a_results.py \
  --all-results data/results/all_results.json \
  --logs "data/results/gap_a_provenance/eval_logs/*.log" \
  --train-logs "data/results/gap_a_provenance/train_logs/*.log" \
  --meta-dir data/results/gap_a_provenance/meta \
  --out data/results/gap_a_results.json
```

The strict mode verifies every digest in the chain (log <-> sidecar <->
sentinel <-> checkpoint) and aborts on any mismatch; the output records the
digest of every input. The seed-42 cells of the matrix come from the
retained `all_results.json` (its per-run keys are the anonymised names used
throughout this repository).

The Windows-side pull record and the M0 console logs are not included;
they contain only local paths and console output and add nothing to the
chain above. Paths of the form `/root/...` in the sentinels are the
training host's working directories.
