# Logs

This folder holds the 212 experiment logs of the retained-log inventory in the supplementary material (Table 15). The inventory counts 223 files; the other 11 are LaTeX compile logs of paper drafts, which are not experiment logs and are not released. Four lines were edited to remove tool names (`REDACTIONS.md`); every other byte is as retained. The archive kept some logs in two places, so 48 pairs of files here are byte-identical copies (164 distinct logs; see the end of this file).

## Where each inventory row is

| Inventory row (supplementary Table 15) | Files | Location |
|---|---|---|
| AdaPoinTr training logs for Baseline, Vanilla-FT and DiffCond at seed 42 and for two non-paper pilots; batch logs with the per-epoch training and evaluation output of the seed-43/44 Vanilla-FT and DiffCond runs, the specialist runs and the seed-42 ablation runs; official S/M/H evaluation logs; epoch-0 and routing sanity logs | 16 | top level of `logs/` |
| Per-run logs: DiffCond seed-43/44 evaluations (7), Baseline training (1), Vanilla-FT seed-42 training (1), EasyWide-Specialist seed-43/44 training (2) and evaluations (6) | 17 | `remote_exp_logs/` |
| Orchestration and smoke-test logs, including the Vanilla-FT Hard-subset evaluation and the predictor-calibration run | 9 | `remote_logs/` |
| UpTrans training logs (Vanilla and DiffCond, seeds 42 to 45), PCN training logs (2 models × 2 seeds) and the baseline main log | 13 | `remote_archive_20260501/train_logs/` |
| UpTrans official S/M/H evaluation logs (2 models × 4 seeds × 3 bins) | 24 | `remote_archive_20260501/eval_logs/` |
| Development audit of 2026-04-16: epoch-0 checkpoint construction and probe, routing sanity check, batch step logs | 7 | `remote_archive_20260501/audit_20260416/` |
| Remote orchestration logs (9) and pre-project pilot logs, non-paper (9) | 18 | `remote_archive_20260501/orchestration/` (9), `remote_archive_20260501/autodl_tmp_backup/` (9) |
| From-scratch baseline rerun (Pitfall 2 audit): training (3) and per-bin evaluation (3) | 6 | `from_scratch_baseline_rerun_20260518/` |
| Added-seed ablation run: training (4), evaluation (12), reproduction gate (3), launcher and dataset-verification logs (39) | 58 | `gap_a_formal_20260910_final/` (`train_logs/` 4, `eval_logs/` 12, `m0/` 3, `launcher_logs/` 39) |
| Earlier pull of the same added-seed run (superseded copies) | 35 | `gap_a_formal_20260910_run1_clone_pull/` |
| Non-paper PCN pilot logs | 8 | `pcn_easr_backup/` |
| UpTrans stage master log | 1 | `b1_stage3_final/` |
| LaTeX compile logs of paper drafts | 11 | not released |

The training and evaluation logs of the added-seed ablation run also ship, with their sentinels and sidecars, in `data/results/gap_a_provenance/`.

## Names in file and folder names

| Name | Meaning |
|---|---|
| `R100`, `baseline_sn55` | Baseline: the 300-epoch from-scratch AdaPoinTr, warm-start source of every fine-tune |
| `R200C`, `r200c` | Vanilla-FT, the matched control |
| `R300`, `r300` | DiffCond (full) |
| `R200B`, `r200b` | an early vanilla warm-start fine-tune with the stock crop path; not used in the paper |
| `v1_lambda0`, `v2b` | ablation conditions v1-λ0 (CE off, ranking on) and v2b (CE off, ranking off) |
| `hard_specialist`, `easywide_specialist` | Hard-Specialist and EasyWide-Specialist (data-side probe) |
| `easy_specialist`, `median_specialist` | pilots not used in the paper |
| `smoke30` | the 30-epoch fine-tuning configurations (a historical name) |
| `diffcrop` | the shared DiffCond crop pipeline (`training_crop_mode: diffcond`) |
| `phase2B` | seeds 43 and 44 of Vanilla-FT and DiffCond |
| `phase2_5`, `phase2_6` | specialist runs: Hard-Specialist seeds 43 and 44 and EasyWide-Specialist seed 42; EasyWide-Specialist seeds 43 and 44 |
| `phase2_D` | the Easy and Median specialist pilots, not used in the paper |
| `b1` | the UpTrans cross-backbone runs (ShapeNet-55 seeds 42 to 45, PCN seeds 42 and 43) |
| `gap_a` | the added-seed ablation run (v1-λ0 and v2b at seeds 43 and 44): `m0` reproduction gate, `m1` training, `eval-only` evaluation, `rehearsal` small-scale dry run |
| `baseline_smoke30_rerun` | the from-scratch 30-epoch baseline rerun of the Pitfall 2 audit |
| `easr` | a pre-project idea; these pilot logs are not used in the paper |

## Byte-identical copies

- 34 of the 35 files in `gap_a_formal_20260910_run1_clone_pull/` equal files in `gap_a_formal_20260910_final/`; the exception is `root_gap_a_chain_20260910.log`.
- The 8 files in `pcn_easr_backup/` equal the 8 `experiments_easr__*` files in `remote_archive_20260501/autodl_tmp_backup/`.
- At the top level, `batch_final.log`, `batch_step34_v2.log`, `step3_r300_ep0.log`, `step4_dtr_sanity.log`, `phase2B_watcher.log` and `phase2_5_watcher.log` equal, in that order, `remote_logs/batch_final.log`, `remote_archive_20260501/audit_20260416/batch_step34_v2.log`, `remote_archive_20260501/audit_20260416/step3_r300_ep0.log`, `remote_archive_20260501/audit_20260416/step4_dtr_sanity.log`, `remote_archive_20260501/orchestration/d_to_b_watcher.log` and `remote_archive_20260501/orchestration/b_to_p25_watcher.log`.

`release_manifest.sha256` at the repository root lists the SHA-256 of every other file in the repository.
