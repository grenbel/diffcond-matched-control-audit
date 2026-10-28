#!/usr/bin/env python3
"""Added-seed ablation results: the 3-seed four-condition ablation matrix and paired statistics, with a closed
evidence chain.

Inputs (strict/formal mode, all mandatory):
  --all-results  all_results.json (retained seed-42 cells, 3-seed Vanilla-FT and DiffCond Oracle)
  --logs         the 12 eval logs pulled as eval_<cell>_seed<S>_<mode>.log (cell in v1_lambda0/v2b,
                 S in 43/44, mode in easy/median/hard); identity is verified INSIDE each log
                 (args.config / mode / difficulty_source oracle / deterministic / exp_name / ckpts)
  --train-logs   the 4 training logs pulled as train_<cell>_seed<S>.log (identity from args.config /
                 exp_name / seed / start_ckpts; zero "NaN/Inf loss" skips; final "[Training] EPOCH: 30")
  --meta-dir     the pulled gap_a_logs metadata: M0_PASS.json, M1_PASS.json and the 12
                 eval_meta_<cell>_seed<S>_<mode>.json sidecars written by the launcher; every eval log's
                 SHA-256 must equal its sidecar, every sidecar's checkpoint SHA-256 must equal the M1
                 sentinel's, every training log's SHA-256 must equal the M1 sentinel's
Every formal statistic is asserted to pair seeds [42, 43, 44] (n=3, df=2). Duplicates, conflicts,
non-finite values, off-target logs or any identity/hash mismatch abort with a non-zero exit.
--allow-partial is a labelled diagnostic mode (never for reported numbers). The output JSON records the
SHA-256 of every input file.

Delta convention = main paper Tables 2-3 / tools/paired_stats.py: CD metrics (lower is better)
delta% = (vanilla - candidate) / vanilla * 100; F-Score delta% = (candidate - vanilla) / vanilla
* 100. "Mean" = per-seed mean over the three bins, then paired (raw-derived, as in Table 2).
Diagnostic: --allow-partial without logs reproduces the main comparison's DiffCond row (-4.18/-3.09/-5.48/-4.56 %)."""
import argparse
import glob
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gap_a_digests import ACCEPTABLE_DIFFS, _gpus_same_hardware as gpus_same_hardware   # the single whitelist of acceptable host differences

MODES = ["easy", "median", "hard"]
METRICS = ["F-Score", "CDL1", "CDL2"]
CELLS = {  # display name -> (all_results seed-42 key pattern, log cell token)
    "DiffCond (full)": ("diffcond_oracle_seed{s}", None),
    "v1-lambda0 (rank on, CE off)": ("v1_lambda0", "v1_lambda0"),
    "v2b (rank off, CE off)": ("v2b", "v2b"),
}
LOG_CELLS = ["v1_lambda0", "v2b"]
NEW_SEEDS = [43, 44]
VANILLA_KEY = "vanilla_ft_seed{s}"
SEEDS = [42, 43, 44]
T_CRIT_95 = {1: 12.706204736432095, 2: 4.302652729911275, 3: 3.182446305284263}
EVAL_NAME_RE = re.compile(r"^eval_(v1_lambda0|v2b)_seed(\d+)_(easy|median|hard)\.log$")
TRAIN_NAME_RE = re.compile(r"^train_(v1_lambda0|v2b)_seed(\d+)\.log$")
NAMESPACES = {   # must mirror tools/gap_a_digests.py
    "formal": {"cfg_dir": "cfgs/ShapeNet55_models", "exp_prefix": "diffcond_", "final_epoch": 30},
    "rehearsal": {"cfg_dir": "gap_a_rehearsal/rehearsal_cfgs", "exp_prefix": "rehearsal_diffcond_", "final_epoch": 0},
}
NS = NAMESPACES["formal"]
NUM = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[-+]?nan|[-+]?inf)"
OVERALL_RE = re.compile(r"Overall\s+" + r"\s+".join([NUM] * 4), re.IGNORECASE)
ARGS_RE = re.compile(r"args\.(\w+) : (.*)")
NAN_SKIP_RE = re.compile(r"NaN/Inf loss")
def final_epoch_re():
    return re.compile(r"\[Training\] EPOCH: " + str(NS["final_epoch"]) + r" EpochTime")
R100_TAIL = "baseline_sn55/ckpt-best.pth"
# The retained results file all_results.json (anonymised per-run keys; its DiffCond rows reproduce the main
# comparison). Strict mode refuses any other content.
ALL_RESULTS_SHA256_PINNED = "3208df86ba1809c2ee566abb721dfae1ab211988b97d71f5fec9a85fb39b5393"
R100_PINNED_SHA256 = "f1acb03baa1f80c5c4eeb361d9b23a0193e9891d31df1a0de0b4de37bee534b3"   # must mirror tools/gap_a_digests.py


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return Path(path).read_text(encoding="utf-8", errors="replace")


def log_args(text):
    return {k: v.strip() for k, v in ARGS_RE.findall(text)}


def cfg_of(cell):
    return f"{NS['cfg_dir']}/AdaPoinTr_DiffCond_{cell}.yaml"


def parse_eval_log(path):
    """((cell, seed, mode, metrics, ckpt, sha256), None) or (None, reason); identity checked inside the log."""
    m = EVAL_NAME_RE.match(Path(path).name)
    if not m:
        return None, "basename is not eval_<cell>_seed<S>_<mode>.log"
    cell, seed, mode = m.group(1), int(m.group(2)), m.group(3)
    text = read(path)
    a = log_args(text)
    ident = {"config": a.get("config") == cfg_of(cell), "mode": a.get("mode") == mode,
             "difficulty_source": a.get("difficulty_source") == "oracle", "deterministic": a.get("deterministic") == "True",
             "exp_name": a.get("exp_name") == f"test_{NS['exp_prefix']}{cell}_seed{seed}_eval_{mode}_{mode}",
             "ckpts": (a.get("ckpts") or "").endswith(f"/{NS['exp_prefix']}{cell}_seed{seed}/ckpt-epoch-{NS['final_epoch']:03d}.pth")}
    bad = [k for k, ok in ident.items() if not ok]
    if bad:
        return None, f"log-internal identity mismatch: {bad} (args {[a.get(k) for k in bad]})"
    hits = OVERALL_RE.findall(text)
    if not hits:
        return None, "no Overall line"
    vals = [float(x) for x in hits[-1]]
    if not all(math.isfinite(v) for v in vals):
        return None, f"non-finite Overall {hits[-1]}"
    f, l1, l2, emd = vals
    return (cell, seed, mode, {"F-Score": f, "CDL1": l1, "CDL2": l2, "EMD": emd}, a["ckpts"], list(hits[-1]), sha256_file(path)), None


def parse_train_log(path):
    m = TRAIN_NAME_RE.match(Path(path).name)
    if not m:
        return None, "basename is not train_<cell>_seed<S>.log"
    cell, seed = m.group(1), int(m.group(2))
    text = read(path)
    a = log_args(text)
    ident = {"config": a.get("config") == cfg_of(cell), "exp_name": a.get("exp_name") == f"{NS['exp_prefix']}{cell}_seed{seed}",
             "seed": a.get("seed") == str(seed), "start_ckpts": (a.get("start_ckpts") or "").endswith(R100_TAIL)}
    bad = [k for k, ok in ident.items() if not ok]
    if bad:
        return None, f"log-internal identity mismatch: {bad} (args {[a.get(k) for k in bad]})"
    n = len(NAN_SKIP_RE.findall(text))
    if n:
        return None, f"{n} NaN/Inf batch skips (the seed-42 runs had none)"
    if not final_epoch_re().search(text):
        return None, f"final '[Training] EPOCH: {NS['final_epoch']}' line missing"
    return (cell, seed, sha256_file(path)), None


def collect(paths, parser, target_keys, what):
    """Parse every file; return ({key: record}, problems). Duplicates/conflicts/off-target are problems."""
    results, problems = {}, []
    for p in paths:
        r, why = parser(p)
        if r is None:
            problems.append(f"{p}: {why}")
            continue
        key = r[:3] if what == "eval" else r[:2]
        if key not in target_keys:
            problems.append(f"{p}: not an added-seed ablation target {key}")
            continue
        if key in results:
            same = results[key][3] == r[3] if what == "eval" else results[key][2] == r[2]
            problems.append(f"{p}: {'duplicates' if same else 'conflicts with'} {results[key][-2]}")
            continue
        results[key] = r + (str(p),)
    missing = [f"missing {what} log for {k}" for k in sorted(target_keys - set(results))]
    return results, problems + missing


def check_all_results(per):
    """Schema + finiteness of every value this analysis reads; Vanilla denominators non-zero."""
    problems = []
    keys = [VANILLA_KEY.format(s=s) for s in SEEDS] + ["diffcond_oracle_seed42", "diffcond_oracle_seed43", "diffcond_oracle_seed44", "v1_lambda0", "v2b"]
    for k in keys:
        if k not in per:
            problems.append(f"all_results: missing {k}")
            continue
        for m in MODES:
            for met in METRICS:
                v = per[k].get(m, {}).get(met)
                if not isinstance(v, (int, float)) or not math.isfinite(v):
                    problems.append(f"all_results: {k}.{m}.{met} missing or non-finite ({v!r})")
                elif k.startswith("vanilla_ft") and v == 0:
                    problems.append(f"all_results: Vanilla {k}.{m}.{met} is zero (division by zero)")
    return problems


def check_meta(meta_dir, evals, trains):
    """Sidecars + sentinels: hashes must tie every log to the checkpoint the launcher evaluated, and the
    M2 seal must bind the M0/M1 sentinels and exactly the 12 sidecars."""
    problems, meta = [], {}
    md = Path(meta_dir)
    for name in ("M0_PASS.json", "M1_PASS.json", "M2_PASS.json"):
        if not (md / name).is_file():
            problems.append(f"meta: {name} missing")
    if problems:
        return problems, meta
    m1 = json.loads(read(md / "M1_PASS.json"))
    m2 = json.loads(read(md / "M2_PASS.json"))
    for name in ("M0_PASS.json", "M1_PASS.json", "M2_PASS.json"):
        meta[name] = sha256_file(md / name)
    if meta["M0_PASS.json"] != m1.get("m0_sentinel_sha256"):
        problems.append("meta: M0_PASS.json hash differs from the one recorded in M1_PASS.json")
    if meta["M0_PASS.json"] != m2.get("m0_sentinel_sha256") or meta["M1_PASS.json"] != m2.get("m1_sentinel_sha256"):
        problems.append("meta: M2_PASS.json does not bind these M0/M1 sentinel files")
    if sorted(m2.get("evals", {})) != sorted(f"{c}_seed{s}_{m}" for c in LOG_CELLS for s in NEW_SEEDS for m in MODES):
        problems.append(f"meta: M2_PASS.json seals {len(m2.get('evals', {}))} evals, not exactly the 12 target evals")
    m0 = json.loads(read(md / "M0_PASS.json"))
    att = md / "code_attestation.json"
    bound = m0.get("snapshot", {}).get("attestation", {}).get("file_sha256")
    bound_m1 = m1.get("attestation_at_seal", {}).get("file_sha256")   # re-issued attestation when M1 accepted host differences
    if not att.is_file():
        problems.append("meta: code_attestation.json missing")
    elif sha256_file(att) not in (bound, bound_m1):
        problems.append("meta: code_attestation.json hash differs from the one bound in M0_PASS.json / M1_PASS.json")
    else:
        meta["code_attestation.json"] = sha256_file(att)
    # explicitly accepted host differences (M1 sealed on another host of the same GPU model): every path must be
    # in the whitelist of tools/gap_a_digests.py, and a GPU inventory may differ in nothing but the UUID
    accepted = m1.get("accepted_differences", [])
    for x in accepted:
        if x.get("path") not in ACCEPTABLE_DIFFS:
            problems.append(f"meta: M1_PASS.json accepted a non-whitelisted difference {x.get('path')}")
        elif x["path"] == "/environment/gpus" and not gpus_same_hardware(x.get("sentinel"), x.get("now")):
            problems.append(f"meta: M1_PASS.json accepted a GPU inventory that differs in more than the UUID: {x}")
    if set(m1.get("accepted_paths", [])) != {x.get("path") for x in accepted}:
        problems.append("meta: M1_PASS.json accepted_paths and accepted_differences disagree")
    meta["accepted_differences"] = accepted
    meta["sealed_on_m0_host"] = m1.get("sealed_on_m0_host", not accepted)
    for (cell, seed), (_, _, sha, path) in trains.items():
        run = m1.get("runs", {}).get(f"{cell}_seed{seed}")
        if run is None:
            problems.append(f"meta: {cell} seed{seed} not in M1_PASS.json")
            continue
        if run["train_log_sha256"] != sha:
            problems.append(f"meta: training log {path} hash != M1_PASS.json")
        tm = md / f"train_meta_{cell}_seed{seed}.json"
        if not tm.is_file():
            problems.append(f"meta: train_meta_{cell}_seed{seed}.json missing")
            continue
        if sha256_file(tm) != run.get("train_meta_sha256"):
            problems.append(f"meta: {tm.name}: hash differs from M1_PASS.json")
        t = json.loads(read(tm))
        ident = {"cell": t.get("cell") == cell, "seed": t.get("seed") == seed, "r100 path": t.get("r100") == run.get("start_ckpt"),
                 "r100 hash": t.get("r100_sha256_at_launch") == run.get("start_ckpt_sha256"), "config path": t.get("config") == cfg_of(cell),
                 "config hash": t.get("config_sha256") == run.get("config_sha256"), "M0 hash": t.get("m0_sentinel_sha256") == meta["M0_PASS.json"]}
        badf = [k for k, ok in ident.items() if not ok]
        if badf:
            problems.append(f"meta: {tm.name}: identity mismatch {badf}")
        meta[tm.name] = sha256_file(tm)
        if run.get("start_ckpt_sha256") != R100_PINNED_SHA256:
            problems.append(f"meta: {cell} seed{seed}: warm-start hash in M1_PASS.json is not the pinned R100")
        te = md / f"train_end_{cell}_seed{seed}.json"
        if not te.is_file():
            problems.append(f"meta: train_end_{cell}_seed{seed}.json missing")
            continue
        if sha256_file(te) != run.get("train_end_sha256"):
            problems.append(f"meta: {te.name}: hash differs from M1_PASS.json")
        e = json.loads(read(te))
        ident_e = {"cell": e.get("cell") == cell, "seed": e.get("seed") == seed, "ckpt path": e.get("ckpt") == run.get("ckpt"),
                   "ckpt hash": e.get("ckpt_sha256") == run.get("ckpt_sha256"), "r100 hash": e.get("r100_sha256_at_end") == R100_PINNED_SHA256,
                   "config hash": e.get("config_sha256") == run.get("config_sha256"), "M0 hash": e.get("m0_sentinel_sha256") == meta["M0_PASS.json"]}
        bade = [k for k, ok in ident_e.items() if not ok]
        if bade:
            problems.append(f"meta: {te.name}: identity mismatch {bade}")
        meta[te.name] = sha256_file(te)
    for (cell, seed, mode), (_, _, _, _, ckpt, overall, sha, path) in evals.items():
        sc = md / f"eval_meta_{cell}_seed{seed}_{mode}.json"
        if not sc.is_file():
            problems.append(f"meta: sidecar {sc.name} missing")
            continue
        s = json.loads(read(sc))
        run = m1.get("runs", {}).get(f"{cell}_seed{seed}", {})
        if s.get("log_sha256") != sha:
            problems.append(f"meta: {sc.name}: eval log hash differs from the pulled log {path}")
        if s.get("ckpt_sha256") != run.get("ckpt_sha256"):
            problems.append(f"meta: {sc.name}: checkpoint hash differs from M1_PASS.json")
        if s.get("ckpt") != ckpt:
            problems.append(f"meta: {sc.name}: checkpoint path differs from the log's args.ckpts")
        if s.get("overall") != overall:
            problems.append(f"meta: {sc.name}: recorded Overall differs from the log")
        if s.get("m1_sentinel_sha256") != meta["M1_PASS.json"]:
            problems.append(f"meta: {sc.name}: M1 sentinel hash differs")
        meta[sc.name] = sha256_file(sc)
        sealed = m2.get("evals", {}).get(f"{cell}_seed{seed}_{mode}", {})
        if sealed.get("sidecar_sha256") != meta[sc.name] or sealed.get("log_sha256") != sha:
            problems.append(f"meta: {sc.name}: not the sidecar/log sealed in M2_PASS.json")
    meta["ckpt_sha256"] = {k: v["ckpt_sha256"] for k, v in m1.get("runs", {}).items()}
    return problems, meta


def load_inputs(a, strict):
    per = json.load(open(a.all_results, encoding="utf-8"))["per_variant"]
    problems = check_all_results(per) if strict else []
    ar_sha = sha256_file(a.all_results)
    if strict and ar_sha != ALL_RESULTS_SHA256_PINNED:
        problems.append(f"all_results: sha256 {ar_sha} is not the pinned retained-results file {ALL_RESULTS_SHA256_PINNED}")
    table, vanilla = {}, {}
    for s in SEEDS:
        if VANILLA_KEY.format(s=s) in per:
            vanilla[s] = per[VANILLA_KEY.format(s=s)]
    for disp, (key_pat, _) in CELLS.items():
        table[disp] = {}
        for s in SEEDS:
            k = key_pat.format(s=s) if "{s}" in key_pat else (key_pat if s == 42 else None)
            if k and k in per:
                table[disp][s] = dict(per[k])
    eval_keys = {(c, s, m) for c in LOG_CELLS for s in NEW_SEEDS for m in MODES}
    train_keys = {(c, s) for c in LOG_CELLS for s in NEW_SEEDS}
    evals, p1 = collect(a.eval_logs, parse_eval_log, eval_keys, "eval")
    trains, p2 = collect(a.train_log_files, parse_train_log, train_keys, "train")
    problems += p1 + p2
    meta = {}
    if strict or a.meta_dir:
        if not a.meta_dir:
            problems.append("--meta-dir is mandatory in strict mode")
        else:
            p3, meta = check_meta(a.meta_dir, evals, trains)
            problems += p3
    if strict and problems:
        print("STRICT MODE FAILURES:", file=sys.stderr)
        for e in problems:
            print("  " + e, file=sys.stderr)
        sys.exit(2)
    for e in problems:
        print("WARN (partial mode): " + e, file=sys.stderr)
    for (cell, seed, mode), rec in evals.items():
        disp = next(d for d, (_, tok) in CELLS.items() if tok == cell)
        table[disp].setdefault(seed, {})[mode] = rec[3]
    return vanilla, table, evals, trains, meta


def delta_pct(metric, van, cand):
    if metric == "F-Score":
        return (cand - van) / van * 100.0
    return (van - cand) / van * 100.0


def paired_stats(deltas):
    n = len(deltas)
    mean = statistics.fmean(deltas)
    if n < 2:
        return {"n": n, "mean": mean, "sd": None, "ci95": None}
    sd = statistics.stdev(deltas)
    half = T_CRIT_95[n - 1] * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "ci95": [mean - half, mean + half]}


def analyse(vanilla, table):
    out = {}
    for disp, seeds in table.items():
        out[disp] = {}
        for metric in METRICS:
            out[disp][metric] = {}
            for mode in MODES + ["Mean"]:
                deltas, used = [], []
                for s in sorted(seeds):
                    if s not in vanilla:
                        continue
                    if mode == "Mean":
                        if any(m not in seeds[s] for m in MODES):
                            continue
                        v = statistics.fmean(vanilla[s][m][metric] for m in MODES)
                        c = statistics.fmean(seeds[s][m][metric] for m in MODES)
                    else:
                        if mode not in seeds[s]:
                            continue
                        v, c = vanilla[s][mode][metric], seeds[s][mode][metric]
                    deltas.append(delta_pct(metric, v, c))
                    used.append(s)
                if deltas:
                    st = paired_stats(deltas)
                    st["seeds"] = used
                    st["per_seed_delta_pct"] = deltas
                    out[disp][metric][mode] = st
    return out


def assert_formal(stats):
    bad = [f"{d} / {m} / {mode}: seeds {st['seeds'] if st else None}"
           for d, metrics in stats.items() for m, modes in metrics.items() for mode in MODES + ["Mean"]
           for st in [modes.get(mode)] if st is None or st["seeds"] != SEEDS or st["n"] != 3]
    if bad:
        print("FORMAL-STAT ASSERTION FAILED (every statistic must pair seeds 42, 43, 44):", file=sys.stderr)
        for b in bad:
            print("  " + b, file=sys.stderr)
        sys.exit(2)


def fmt(st):
    if st is None:
        return "n/a"
    s = f"{st['mean']:+.2f}%"
    if st["ci95"]:
        s += f" [{st['ci95'][0]:+.2f}, {st['ci95'][1]:+.2f}]"
    return s + f" (n={st['n']})"


def markdown(vanilla, table, stats):
    lines = ["| Candidate (vs Vanilla-FT, paired per seed) | Simple | Moderate | Hard | Mean |", "|---|---|---|---|---|"]
    for disp in table:
        lines.append(f"| {disp} | " + " | ".join(fmt(stats[disp]["CDL2"].get(m)) for m in MODES + ["Mean"]) + " |")
    lines += ["", "Raw CD-l2 per bin, mean +/- SD over seeds (Vanilla-FT first):", "| Row | seeds | Simple | Moderate | Hard |", "|---|---|---|---|---|"]

    def msd(vals):
        return f"{vals[0]:.3f}" if len(vals) == 1 else f"{statistics.fmean(vals):.3f} +/- {statistics.stdev(vals):.3f}"
    vs = sorted(vanilla)
    lines.append("| Vanilla-FT | " + ",".join(map(str, vs)) + " | " + " | ".join(msd([vanilla[s][m]["CDL2"] for s in vs]) for m in MODES) + " |")
    for disp, seeds in table.items():
        ss = sorted(s for s in seeds if all(m in seeds[s] for m in MODES))
        if ss:
            lines.append(f"| {disp} | " + ",".join(map(str, ss)) + " | " + " | ".join(msd([seeds[s][m]["CDL2"] for s in ss]) for m in MODES) + " |")
    return "\n".join(lines)


def expand(patterns):
    out = []
    for pat in patterns:
        out.extend(sorted(glob.glob(pat)) or [pat])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--all-results", required=True)
    ap.add_argument("--logs", nargs="*", default=[], help="the 12 eval logs (files or globs)")
    ap.add_argument("--train-logs", nargs="*", default=[], help="the 4 training logs (files or globs)")
    ap.add_argument("--meta-dir", default=None, help="pulled gap_a_logs metadata (M0_PASS.json, M1_PASS.json, eval_meta_*.json)")
    ap.add_argument("--allow-partial", action="store_true", help="diagnostic mode: report whatever exists")
    ap.add_argument("--namespace", choices=sorted(NAMESPACES), default="formal", help="formal run or the tiny dress rehearsal (numbers meaningless)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    global NS
    NS = NAMESPACES[a.namespace]
    if a.namespace == "rehearsal" and a.out and Path(a.out).resolve().as_posix().find("/results/") >= 0:
        sys.exit("refusing to write rehearsal output under results/ (the rehearsal numbers are meaningless)")
    strict = not a.allow_partial
    a.eval_logs, a.train_log_files = expand(a.logs), expand(a.train_logs)
    vanilla, table, evals, trains, meta = load_inputs(a, strict)
    stats = analyse(vanilla, table)
    if strict:
        assert_formal(stats)
    accepted = meta.get("accepted_differences", []) if isinstance(meta, dict) else []
    host_note = "" if not accepted else (f" (M1 sealed on another host of the same GPU model; {len(accepted)} recorded host differences: "
                                         + ", ".join(x["path"] for x in accepted) + ")")
    res = {"mode": ("strict" if strict else "partial (diagnostic; not for reported numbers)") + host_note + ("" if a.namespace == "formal" else " REHEARSAL (pipeline test; numbers meaningless)"),
           "namespace": a.namespace,
           "inputs": {"all_results": a.all_results, "all_results_sha256": sha256_file(a.all_results),
                      "all_results_sha256_pinned": ALL_RESULTS_SHA256_PINNED,
                      "eval_logs": {f"{c}_seed{s}_{m}": {"path": r[-1], "sha256": r[-2]} for (c, s, m), r in sorted(evals.items())},
                      "train_logs": {f"{c}_seed{s}": {"path": r[-1], "sha256": r[-2]} for (c, s), r in sorted(trains.items())},
                      "meta": meta},
           "convention": ("delta% = (vanilla - candidate)/vanilla*100 for CD (lower better), (candidate - vanilla)/vanilla*100 "
                          "for F-Score; Mean = per-seed mean over bins then paired; CI = Student-t 95%, df = n-1"),
           "vanilla": {str(s): v for s, v in vanilla.items()},
           "cells": {d: {str(s): v for s, v in seeds.items()} for d, seeds in table.items()},
           "paired_stats": stats}
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"mode: {res['mode']}; vanilla seeds: {sorted(vanilla)}; eval logs: {len(evals)}; train logs: {len(trains)}")
    print(markdown(vanilla, table, stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
