#!/usr/bin/env python3
"""Added-seed ablation state digests: bind the verified instance state to M0, M0 to M1, M1 to the evals, and the
evals to a final M2 seal. Run from the repo root on the instance.

  snapshot [PARAMS]                                 live JSON snapshot (no dataset/archive re-hash)
  verify-state [PARAMS]                             full re-hash + expected-state checks, no sentinel
  write-m0 PATH --m0a-log L --m0b-log L --m0c-log L [PARAMS]
  check-m0 PATH [--accept P ...] [PARAMS]           exit 3 if the live state differs
  record-train-start OUT --cell C --seed S [PARAMS] warm-start hash at launch time
  write-m1 PATH --m0 M0 --train-log cell:seed:LOG ... [--accept P ...] [PARAMS]
      --accept names a host-identity difference (ACCEPTABLE_DIFFS) that the seal may record instead of
      refusing; used once, 2026-09-10, to seal a training finished on a host whose GPU then failed
  check-m1 PATH --m0 M0 [PARAMS]
  record-eval OUT --cell C --seed S --mode M --ckpt P --log L --m1 M1
  write-m2 PATH --m0 M0 --m1 M1 [PARAMS]            after the 12 evals: final state + sidecar seal
  check-m2 PATH --m0 M0 --m1 M1 [PARAMS]
PARAMS (passed by the launcher from its actual values, bound into every sentinel):
  --r100 P --v2b42 P --cells "v1_lambda0 v2b" --seeds "43 44" --parallel N --threads NT --gpus "0"

Expected state is PINNED here, not read from the instance: the official dataset manifest digest, the
archive digest, the split-list digests, the Pointnet2 commit and the pip freeze list. The dataset is
fully re-hashed (52,470 files) and the archive re-hashed at every write/check; the code tree,
compiled modules, environment and parameters are recomputed live and deep-compared."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

DS = os.environ.get("DS", "/root/data/ShapeNet55-34")
LINK = "data/ShapeNet55-34/shapenet_pc"
SPLIT_DIR = "data/ShapeNet55-34/ShapeNet-55"
MANIFEST_SHA256_PINNED = "20ab2c57367a49a9e8815bd622952d0f18b95b6310c65ce7cd6d6786054fa0aa"   # official archive manifest, ingest 2026-09-09 16:04
ARCHIVE_SHA256_PINNED = "70774799990cbdc02bb7bfe9c78af940d418f404c58d9093e548222e2f01d0c1"
TRAIN_SHA256_PINNED = "dc3f73b9095ff1827633808330807e42d3d2d256fb03ace4ddb0baec69347153"     # data/ShapeNet55-34/ShapeNet-55/train.txt, CR-stripped
TEST_SHA256_PINNED = "dd18af90b8de5246b91c6bc457370434a19e7322d1627ea0916605ec511ba918"      # test.txt
POINTNET2_COMMIT_PINNED = "b5ceb6d9ca0467ea34beb81023f96ee82228f626"
R100_PINNED_PATH = "experiments/AdaPoinTr/ShapeNet55_models/baseline_sn55/ckpt-best.pth"          # the warm-start of every run
R100_PINNED_SHA256 = "f1acb03baa1f80c5c4eeb361d9b23a0193e9891d31df1a0de0b4de37bee534b3"          # local backup R100_baseline_300ep/ckpt-best.pth
V2B42_PINNED_SHA256 = "3f8c329a186a1f204eeefe0027cce0ee6eb250db68146c5657684bd5725b6d28"         # retained seed-42 v2b ckpt-epoch-030.pth
POINTNET2_CHECKOUT = "/root/Pointnet2_PyTorch"
FREEZE_EXPECTED = "tools/gap_a_freeze_expected.txt"
CODE_ATTESTATION = "code_attestation.json"
CODE_ROOTS = ["main.py", "tools", "utils", "datasets", "models", "cfgs", "extensions"]
CODE_EXT = (".py", ".sh", ".yaml", ".yml", ".json", ".txt", ".log", ".cu", ".cpp", ".cc", ".h", ".hpp", ".cuh")
SKIP_DIRS = {"__pycache__", "build", ".git", ".ipynb_checkpoints"}
GENERATED_OK = {"cfgs/ShapeNet55_models/AdaPoinTr_DiffCond_v2b_gapA_smoke0.yaml"}   # written by M0, bound by the sentinel, exempt from attestation
COMPILED_MODULES = ["pointnet2_ops", "pointnet2_ops._ext", "chamfer", "emd", "gridding", "gridding_distance",
                    "cubic_feature_sampling", "torch._C"]
REF_LOG = "tools/gap_a_ref/v2b_seed42_eval_hard.log"
MODES = ["easy", "median", "hard"]
M0A_EXPECTED = ["0.340", "16.835", "1.550", "0.000"]
OVERALL_RE = re.compile(r"Overall\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")
ARGS_RE = re.compile(r"args\.(\w+) : (.*)")
VOLATILE = ("checked_at", "written_at", "report", "snapshot_at", "recorded_at")


def sha_file(path, strip_cr=False):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        if strip_cr:
            h.update(fh.read().replace(b"\r\n", b"\n"))
        else:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def sha_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.stdout.strip()


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def dump(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


def require(cond, msg):
    if not cond:
        sys.exit(f"REFUSED: {msg}")


def read_text(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def last_overall(path):
    hits = OVERALL_RE.findall(read_text(path))
    return list(hits[-1]) if hits else None


def log_args(path):
    return {k: v.strip() for k, v in ARGS_RE.findall(read_text(path))}


NAMESPACES = {
    "formal": {"cfg_dir": "cfgs/ShapeNet55_models", "cfg_parent": "ShapeNet55_models", "exp_prefix": "diffcond_",
               "final_epoch": 30, "logdir": "gap_a_logs"},
    # dress rehearsal on 96/16 clouds: same code paths, separate experiment directories and sentinels
    "rehearsal": {"cfg_dir": "gap_a_rehearsal/rehearsal_cfgs", "cfg_parent": "rehearsal_cfgs", "exp_prefix": "rehearsal_diffcond_",
                  "final_epoch": 0, "logdir": "gap_a_logs_rehearsal"},
}
NS = NAMESPACES["formal"]


def set_namespace(name):
    global NS
    NS = NAMESPACES[name]


def ckpt_path(cell, seed):
    return (f"experiments/AdaPoinTr_DiffCond_{cell}/{NS['cfg_parent']}/{NS['exp_prefix']}{cell}_seed{seed}/"
            f"ckpt-epoch-{NS['final_epoch']:03d}.pth")


def cfg_path(cell):
    return f"{NS['cfg_dir']}/AdaPoinTr_DiffCond_{cell}.yaml"


def train_exp(cell, seed):
    return f"{NS['exp_prefix']}{cell}_seed{seed}"


def eval_exp(cell, seed, mode):
    return f"test_{NS['exp_prefix']}{cell}_seed{seed}_eval_{mode}_{mode}"


def smoke_cfg():
    return f"{NS['cfg_dir']}/AdaPoinTr_DiffCond_v2b_gapA_smoke0.yaml"


def smoke_ckpt():
    return f"experiments/AdaPoinTr_DiffCond_v2b_gapA_smoke0/{NS['cfg_parent']}/gapA_smoke_v2b_seed43/ckpt-last.pth"


def code_tree():
    files = {}
    for root in CODE_ROOTS:
        if os.path.isfile(root):
            files[root] = sha_file(root, strip_cr=True)
            continue
        for r, ds, fs in os.walk(root):
            ds[:] = sorted(d for d in ds if d not in SKIP_DIRS and not d.endswith(".egg-info"))
            for f in sorted(fs):
                if f.endswith(CODE_EXT):
                    p = os.path.join(r, f).replace(os.sep, "/")
                    files[p] = sha_file(p, strip_cr=True)
    lines = "".join(f"{files[p]}  {p}\n" for p in sorted(files))
    attest_lines = "".join(f"{files[p]}  {p}\n" for p in sorted(files) if p not in GENERATED_OK)
    return {"files": len(files), "tree_sha256": sha_text(lines), "attest_tree_sha256": sha_text(attest_lines), "per_file": files}


def compiled_modules():
    out = {}
    for name in COMPILED_MODULES:
        try:
            mod = __import__(name, fromlist=["_"])
            f = getattr(mod, "__file__", None)
            entry = {"file": f}
            if f and f.endswith((".so", ".pyd")):
                entry["sha256"] = sha_file(f)
            elif f:
                d = os.path.dirname(f)
                items = sorted(x for x in os.listdir(d) if x.endswith((".py", ".so", ".pyd")))
                entry["members_sha256"] = {x: sha_file(os.path.join(d, x)) for x in items}
            out[name] = entry
        except Exception as ex:
            out[name] = {"import_error": f"{type(ex).__name__}: {ex}"}
    return out


def split_state():
    t = sha_file(os.path.join(SPLIT_DIR, "train.txt"), strip_cr=True)
    e = sha_file(os.path.join(SPLIT_DIR, "test.txt"), strip_cr=True)
    return {"train_sha256": t, "train_pinned_ok": t == TRAIN_SHA256_PINNED,
            "test_sha256": e, "test_pinned_ok": e == TEST_SHA256_PINNED}


def dataset_state(full_verify):
    acc = load(os.path.join(DS, "ACCEPTED.json"))
    man_path = os.path.join(DS, "official_manifest_sha256.txt")
    mf = sha_file(man_path)
    d = {"manifest_file_sha256": mf, "manifest_pinned_ok": mf == MANIFEST_SHA256_PINNED,
         "manifest_matches_declared": mf == acc["manifest_sha256"],
         "archive_path": acc["archive"]["path"], "archive_declared_pinned_ok": acc["archive"]["sha256"] == ARCHIVE_SHA256_PINNED,
         "accepted_dir": acc["accepted_dir"], "link_target": os.path.realpath(LINK),
         "link_ok": os.path.realpath(LINK) == os.path.realpath(acc["accepted_dir"]), "splits": split_state()}
    d["ok"] = (d["manifest_pinned_ok"] and d["manifest_matches_declared"] and d["archive_declared_pinned_ok"] and d["link_ok"]
               and d["splits"]["train_pinned_ok"] and d["splits"]["test_pinned_ok"])
    if full_verify:
        if os.path.isfile(acc["archive"]["path"]):
            ah = sha_file(acc["archive"]["path"])
            d["archive_rehash"] = {"sha256": ah, "pinned_ok": ah == ARCHIVE_SHA256_PINNED}
        else:
            d["archive_rehash"] = {"pinned_ok": False, "error": "official archive missing: keep ShapeNet55.zip on the instance"}
        os.makedirs("gap_a_logs", exist_ok=True)
        rep = f"gap_a_logs/dataset_full_verify_{time.strftime('%Y%m%d_%H%M%S')}.json"
        cmd = [sys.executable, "tools/gap_a_verify_official.py", "--pc", LINK, "--split-dir", SPLIT_DIR,
               "--ref-dir", os.path.join(DS, "reference_npy"), "--mode", "full", "--manifest-expect", man_path, "--report", rep]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        verdict = "PASS" if r.returncode == 0 else "FAIL"
        d["full_verify"] = {"verdict": verdict, "report": rep, "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        if verdict != "PASS":
            d["full_verify"]["tail"] = (r.stdout + r.stderr)[-800:]
        d["ok"] = d["ok"] and d["archive_rehash"]["pinned_ok"] and verdict == "PASS"
    return d


def freeze_state():
    live = sorted(l.strip() for l in sh("pip freeze 2>/dev/null").splitlines() if l.strip())
    exp = None
    if os.path.isfile(FREEZE_EXPECTED):
        exp = sorted(l.strip() for l in read_text(FREEZE_EXPECTED).splitlines() if l.strip() and not l.startswith("#"))
    extra = [l for l in live if exp is None or l not in exp][:10]
    missing = [l for l in (exp or []) if l not in live][:10]
    return {"expected_file": FREEZE_EXPECTED, "expected_sha256": sha_file(FREEZE_EXPECTED, strip_cr=True) if exp is not None else None,
            "matches_expected": exp is not None and live == exp, "live_count": len(live),
            "unexpected_live": extra, "missing_from_live": missing, "live": live}


def pointnet2_state():
    head = sh(f"git -C {POINTNET2_CHECKOUT} rev-parse HEAD 2>/dev/null")
    st = {"checkout": POINTNET2_CHECKOUT, "head": head, "commit_pinned_ok": head == POINTNET2_COMMIT_PINNED}
    try:
        import pointnet2_ops
        inst = os.path.dirname(pointnet2_ops.__file__)
        src = os.path.join(POINTNET2_CHECKOUT, "pointnet2_ops_lib", "pointnet2_ops")
        py = sorted(f for f in os.listdir(src) if f.endswith(".py")) if os.path.isdir(src) else []
        same = bool(py) and all(os.path.isfile(os.path.join(inst, f)) and sha_file(os.path.join(inst, f)) == sha_file(os.path.join(src, f)) for f in py)
        so = sorted(f for f in os.listdir(inst) if f.endswith(".so"))
        st.update(installed=inst, installed_matches_checkout=same, so={f: sha_file(os.path.join(inst, f)) for f in so},
                  checkout_source_sha256=sha_text("".join(f"{sha_file(os.path.join(src, f), strip_cr=True)}  {f}\n" for f in py)) if py else None)
    except Exception as ex:
        st["error"] = f"{type(ex).__name__}: {ex}"
    st["ok"] = bool(st["commit_pinned_ok"] and st.get("installed_matches_checkout") is True and st.get("so"))
    return st


def environment():
    import platform
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "cpu_quota": sh("cat /sys/fs/cgroup/cpu.max 2>/dev/null"),
           "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1"),
           "nvcc": sh("nvcc --version 2>/dev/null | tail -1")}
    try:
        import torch
        env.update(torch=torch.__version__, torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                   cuda_available=torch.cuda.is_available())
    except Exception as ex:
        env["torch_error"] = f"{type(ex).__name__}: {ex}"
    # physical inventory, independent of the CUDA_VISIBLE_DEVICES subset a stage happens to use
    inv = sh("nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader 2>/dev/null")
    env["gpus"] = [] if ("No devices" in inv or "failed" in inv) else [l.strip() for l in inv.splitlines() if l.strip()]
    env["freeze"] = freeze_state()
    env["pointnet2"] = pointnet2_state()
    if os.path.isfile("gap_a_env_manifest.json"):
        m = load("gap_a_env_manifest.json")
        env["build_vars"] = {"TORCH_CUDA_ARCH_LIST": m.get("TORCH_CUDA_ARCH_LIST"), "MAX_JOBS": m.get("MAX_JOBS")}
    env["ok"] = env["freeze"]["matches_expected"] and env["pointnet2"]["ok"]
    env["env_sha256"] = sha_text(canonical(env))
    return env


def params_state(a):
    cells, seeds = a.cells.split(), [int(x) for x in a.seeds.split()]
    return {"r100": a.r100, "r100_sha256": sha_file(a.r100) if os.path.isfile(a.r100) else None,
            "v2b42": a.v2b42, "v2b42_sha256": sha_file(a.v2b42) if os.path.isfile(a.v2b42) else None,
            "cells": cells, "seeds": seeds,
            "configs": {c: {"path": cfg_path(c), "sha256": sha_file(cfg_path(c), strip_cr=True) if os.path.isfile(cfg_path(c)) else None} for c in cells},
            "reference_log_sha256": sha_file(REF_LOG, strip_cr=True) if os.path.isfile(REF_LOG) else None}


def schedule_state(a):
    """Stage scheduling (recorded, never part of the cross-stage scientific comparison)."""
    return {"parallel": a.parallel, "threads": a.threads, "gpus": a.gpus, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def attestation_state(code):
    """code_attestation.json is written by the push step from the files it actually uploaded; the live
    tree (minus the M0-generated smoke config) must reproduce its digest."""
    if not os.path.isfile(CODE_ATTESTATION):
        return {"present": False, "ok": False}
    att = load(CODE_ATTESTATION)
    return {"present": True, "file_sha256": sha_file(CODE_ATTESTATION), "head": att.get("head"),
            "declared_tree_sha256": att.get("attest_tree_sha256"), "live_attest_tree_sha256": code["attest_tree_sha256"],
            "ok": att.get("attest_tree_sha256") == code["attest_tree_sha256"]}


def snapshot(a, full_verify):
    code = code_tree()
    return {"snapshot_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "cwd": os.getcwd(), "dataset": dataset_state(full_verify),
            "code": code, "attestation": attestation_state(code), "compiled": compiled_modules(),
            "environment": environment(), "params": params_state(a), "schedule": schedule_state(a), "namespace": a.namespace}


def strip_volatile(obj):
    if isinstance(obj, dict):
        return {k: strip_volatile(v) for k, v in obj.items() if k not in VOLATILE and k != "schedule"}
    if isinstance(obj, list):
        return [strip_volatile(x) for x in obj]
    return obj


def diff_items(a, b, path=""):
    """Structured comparison: [(path, sentinel_value, live_value)] for every leaf that differs."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            out += diff_items(a.get(k, "<absent>"), b.get(k, "<absent>"), f"{path}/{k}")
    elif a != b:
        out.append((path, a, b))
    return out


def diff(a, b, path=""):
    return [f"{p}: sentinel {str(x)[:90]!r} != now {str(y)[:90]!r}" for p, x, y in diff_items(a, b, path)]


# Differences between the M0 host and a later host that MAY be accepted, but only when named explicitly
# (--accept PATH) at seal time (user decision 2026-09-10: same GPU model + same software = same state).
# Every accepted difference is recorded verbatim in the M1 sentinel and the same list is re-applied,
# never widened, by every later check. Anything else (dataset, compiled extensions, freeze lock,
# torch / driver, scientific code, params) can never be accepted.
ACCEPTABLE_DIFFS = {
    "/environment/cpu_quota": "host identity: cgroup cpu quota",
    "/environment/gpus": "host identity: GPU UUID (same GPU name and memory required)",
    "/environment/driver": "host software: NVIDIA kernel driver version (torch / CUDA runtime / cudnn are still compared)",
    "/environment/platform": "host software: kernel / platform string",
    "/environment/env_sha256": "derived digest of the environment block",
    "/code/per_file/tools/gap_a_digests.py": "provenance tool revised for the host change",
    "/code/per_file/tools/gap_a_results.py": "provenance tool revised for the host change",
    "/code/per_file/tools/gap_a_remote.py": "provenance tool revised for the host change",
    "/code/tree_sha256": "code-tree digest (follows the tool files above)",
    "/code/attest_tree_sha256": "code-tree digest (follows the tool files above)",
    "/attestation/file_sha256": "code_attestation.json re-issued for the revised tools",
    "/attestation/declared_tree_sha256": "code_attestation.json re-issued for the revised tools",
    "/attestation/live_attest_tree_sha256": "code-tree digest (follows the tool files above)",
}
_GPU_UUID = re.compile(r"GPU-[0-9a-fA-F-]+")


def _gpus_same_hardware(a, b):
    """True when two nvidia-smi inventories differ only in the GPU UUID field."""
    def norm(lst):
        return [", ".join(t for t in (x.strip() for x in str(g).split(",")) if not _GPU_UUID.fullmatch(t)) for g in lst]
    return isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and norm(a) == norm(b)


def split_accepted(items, accept):
    """Partition diff items into (remaining, accepted). Refuses an --accept path outside the whitelist
    or a GPU inventory that differs in more than the UUID."""
    accept = list(accept or [])
    for path in accept:
        require(path in ACCEPTABLE_DIFFS, f"--accept {path}: not an acceptable difference (allowed: {sorted(ACCEPTABLE_DIFFS)})")
    remaining, accepted = [], []
    for path, old, new in items:
        if path in accept:
            if path == "/environment/gpus":
                require(_gpus_same_hardware(old, new), f"/environment/gpus differs in more than the UUID: {old!r} vs {new!r}")
            accepted.append({"path": path, "why": ACCEPTABLE_DIFFS[path], "sentinel": old, "now": new})
        else:
            remaining.append((path, old, new))
    return remaining, accepted


def state_problems(now):
    """Expected-state failures that refuse a write regardless of any sentinel."""
    p = []
    if not now["dataset"]["ok"]:
        p.append("dataset: " + canonical({k: v for k, v in now["dataset"].items() if k not in ("splits",)})[:300])
    if not now["environment"]["ok"]:
        p.append(f"environment: freeze matches expected={now['environment']['freeze']['matches_expected']} "
                 f"(unexpected {now['environment']['freeze']['unexpected_live']}, missing {now['environment']['freeze']['missing_from_live']}); "
                 f"pointnet2 ok={now['environment']['pointnet2']['ok']} head={now['environment']['pointnet2']['head']}")
    errs = [k for k, v in now["compiled"].items() if "import_error" in v]
    if errs:
        p.append(f"compiled modules failed to import: {errs}")
    if not now["attestation"]["ok"]:
        p.append(f"code attestation: {now['attestation']}")
    pr = now["params"]
    if not (pr["r100_sha256"] and pr["v2b42_sha256"] and pr["reference_log_sha256"] and all(v["sha256"] for v in pr["configs"].values())):
        p.append("params: R100 / v2b42 / reference log / config missing")
    if pr["r100"] != R100_PINNED_PATH or pr["r100_sha256"] != R100_PINNED_SHA256:
        p.append(f"params: warm-start checkpoint is not the pinned R100 (path {pr['r100']}, sha256 {pr['r100_sha256']})")
    if NS is NAMESPACES["formal"] and pr["v2b42_sha256"] != V2B42_PINNED_SHA256:
        p.append(f"params: retained seed-42 v2b checkpoint is not the pinned one (sha256 {pr['v2b42_sha256']})")
    return p


def m0_log_problems(a):
    """Identity of the three M0 logs, checked from their contents (not from the launcher's word).
    M0a (the retained seed-42 reproduction on the full test split) exists only in the formal namespace."""
    bad = []
    if NS is NAMESPACES["formal"]:
        m0a = log_args(a.m0a_log)
        for k, v in {"config": cfg_path("v2b"), "ckpts": a.v2b42, "mode": "hard", "difficulty_source": "oracle", "deterministic": "True",
                     "exp_name": "test_diffcond_v2b_seed42_repro_eval_hard_hard"}.items():
            if m0a.get(k) != v:
                bad.append(f"M0a args.{k}={m0a.get(k)!r} != {v!r}")
        r = subprocess.run([sys.executable, "tools/gap_a_compare_eval_logs.py", REF_LOG, a.m0a_log, "--require-overall", "--min-samples", "52"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            bad.append("M0a does not reproduce the retained seed-42 hard eval: " + (r.stdout.strip().splitlines() or ["?"])[-1])
    SMOKE_CFG, SMOKE_CKPT = smoke_cfg(), smoke_ckpt()
    m0b = log_args(a.m0b_log)
    for k, v in {"config": SMOKE_CFG, "exp_name": "gapA_smoke_v2b_seed43", "seed": "43", "start_ckpts": a.r100}.items():
        if m0b.get(k) != v:
            bad.append(f"M0b args.{k}={m0b.get(k)!r} != {v!r}")
    tb = read_text(a.m0b_log)
    if re.findall(r"NaN/Inf loss", tb):
        bad.append("M0b has NaN/Inf batch skips")
    if not re.search(r"\[Training\] EPOCH: 0 EpochTime", tb):
        bad.append("M0b lacks the '[Training] EPOCH: 0' line")
    m0c = log_args(a.m0c_log)
    for k, v in {"config": SMOKE_CFG, "ckpts": SMOKE_CKPT, "mode": "hard", "difficulty_source": "oracle", "deterministic": "True",
                 "exp_name": "test_gapA_smoke_v2b_seed43_eval_hard_hard"}.items():
        if m0c.get(k) != v:
            bad.append(f"M0c args.{k}={m0c.get(k)!r} != {v!r}")
    if last_overall(a.m0c_log) is None:
        bad.append("M0c has no Overall line")
    return bad


def cmd_write_m0(a):
    formal = NS is NAMESPACES["formal"]
    require(formal or a.m0a_log is None, "M0a is only defined in the formal namespace")
    require((not formal) or a.m0a_log, "the formal M0 sentinel needs --m0a-log")
    for lg in ([a.m0a_log] if formal else []) + [a.m0b_log, a.m0c_log]:
        require(os.path.isfile(lg), f"M0 log missing: {lg}")
    m0a = last_overall(a.m0a_log) if formal else None
    require((not formal) or m0a == M0A_EXPECTED, f"M0a Overall {m0a} != {M0A_EXPECTED}")
    bad = m0_log_problems(a)
    require(not bad, "M0 logs failed semantic validation:" + chr(10) + "  " + (chr(10) + "  ").join(bad))
    m0c = last_overall(a.m0c_log)
    snap = snapshot(a, full_verify=True)
    problems = state_problems(snap)
    require(not problems, "expected state not met:" + chr(10) + "  " + (chr(10) + "  ").join(problems))
    require(snap["environment"].get("cuda_available") is True, "CUDA not available: the M0 sentinel must be written in GPU mode")
    logs = {"m0b": a.m0b_log, "m0c": a.m0c_log}
    if formal:
        logs["m0a"] = a.m0a_log
    dump(a.path, {"written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "namespace": a.namespace, "m0a_overall": m0a,
                  "m0c_overall_smoke_not_a_result": m0c, "logs": {k: {"path": v, "sha256": sha_file(v)} for k, v in logs.items()}, "snapshot": snap})
    print("M0 sentinel written:", a.path)
    return 0


def check_m0_against(a, sentinel_path, accept=()):
    """Live state vs the M0 snapshot. `accept` = explicitly accepted host-identity paths (see
    ACCEPTABLE_DIFFS); returns (problems, live_snapshot, sentinel, accepted_differences)."""
    s = load(sentinel_path)
    now = snapshot(a, full_verify=True)
    remaining, accepted = split_accepted(diff_items(strip_volatile(s["snapshot"]), strip_volatile(now)), accept)
    d = [f"{p}: sentinel {str(x)[:90]!r} != now {str(y)[:90]!r}" for p, x, y in remaining]
    d += ["expected state: " + p for p in state_problems(now)]
    for k, v in s.get("logs", {}).items():
        if not os.path.isfile(v["path"]) or sha_file(v["path"]) != v["sha256"]:
            d.append(f"M0 log {k} ({v['path']}) missing or changed")
    return d, now, s, accepted


def print_accepted(accepted):
    if accepted:
        print(f"ACCEPTED host differences ({len(accepted)}, recorded in the sentinel):")
        for x in accepted:
            print(f"  {x['path']}: sentinel {str(x['sentinel'])[:90]!r} -> now {str(x['now'])[:90]!r}  [{x['why']}]")


def cmd_check_m0(a):
    d, now, s, accepted = check_m0_against(a, a.path, a.accept)
    print_accepted(accepted)
    if d:
        print("STATE CHANGED since M0 (refusing):")
        for x in d:
            print("  " + x)
        return 3
    print(f"state matches the M0 sentinel written {s['written_at']} (dataset fully re-hashed + archive re-hashed: "
          f"{now['dataset']['full_verify']['verdict']}; {now['code']['files']} source files; attestation ok; "
          f"freeze == expected; pointnet2 {now['environment']['pointnet2']['head'][:12]}; accepted host differences: {len(accepted)})")
    return 0


def cmd_record_train_start(a):
    require(os.path.isfile(a.m0), "M0 sentinel missing")
    require(os.path.isfile(a.r100), f"warm-start checkpoint missing: {a.r100}")
    dump(a.out, {"cell": a.cell, "seed": a.seed, "r100": a.r100, "r100_sha256_at_launch": sha_file(a.r100),
                 "config": cfg_path(a.cell), "config_sha256": sha_file(cfg_path(a.cell), strip_cr=True),
                 "m0_sentinel_sha256": sha_file(a.m0), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    print("train start recorded:", a.out)
    return 0


def cmd_record_train_end(a):
    """Receipt written immediately after a training process exits: binds cell/seed to the bytes of the
    final checkpoint at that moment, so a later swap between runs cannot be sealed by write-m1."""
    ck = ckpt_path(a.cell, a.seed)
    require(os.path.isfile(ck), f"final checkpoint missing right after training: {ck}")
    require(os.path.isfile(a.m0), "M0 sentinel missing")
    dump(a.out, {"cell": a.cell, "seed": a.seed, "ckpt": ck, "ckpt_sha256": sha_file(ck), "ckpt_bytes": os.path.getsize(ck),
                 "r100": a.r100, "r100_sha256_at_end": sha_file(a.r100), "config": cfg_path(a.cell),
                 "config_sha256": sha_file(cfg_path(a.cell), strip_cr=True), "m0_sentinel_sha256": sha_file(a.m0),
                 "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    print("train end recorded:", a.out)
    return 0


def train_log_problems(path, cell, seed, r100):
    args = log_args(path)
    text = read_text(path)
    want = {"config": cfg_path(cell), "exp_name": train_exp(cell, seed), "seed": str(seed), "start_ckpts": r100}
    bad = [f"args.{k}={args.get(k)!r} != {v!r}" for k, v in want.items() if args.get(k) != v]
    n = len(re.findall(r"NaN/Inf loss", text))
    if n:
        bad.append(f"{n} NaN/Inf batch skips")
    if not re.search(r"\[Training\] EPOCH: " + str(NS["final_epoch"]) + r" EpochTime", text):
        bad.append(f"final '[Training] EPOCH: {NS['final_epoch']}' line missing")
    return bad


def cmd_write_m1(a):
    d, now, m0, accepted = check_m0_against(a, a.m0, a.accept)
    print_accepted(accepted)
    require(not d, "state differs from the M0 sentinel:" + chr(10) + "  " + (chr(10) + "  ").join(d))
    logs = {}
    for spec in a.train_log or []:
        cell, seed, path = spec.split(":", 2)
        logs[(cell, int(seed))] = path
    m0_sha = sha_file(a.m0)
    runs = {}
    for cell in now["params"]["cells"]:
        for seed in now["params"]["seeds"]:
            p = ckpt_path(cell, seed)
            require(os.path.isfile(p), f"missing checkpoint {p}")
            require((cell, seed) in logs and os.path.isfile(logs[(cell, seed)]), f"missing training log for {cell} seed {seed}")
            bad = train_log_problems(logs[(cell, seed)], cell, seed, now["params"]["r100"])
            require(not bad, f"training log {logs[(cell, seed)]}: {bad}")
            meta_p = f"{NS['logdir']}/train_meta_{cell}_seed{seed}.json"
            require(os.path.isfile(meta_p), f"missing train-start sidecar {meta_p}")
            meta = load(meta_p)
            require(meta["cell"] == cell and meta["seed"] == seed and meta["r100"] == now["params"]["r100"] and meta["config"] == cfg_path(cell)
                    and meta["r100_sha256_at_launch"] == now["params"]["r100_sha256"] and meta["m0_sentinel_sha256"] == m0_sha
                    and meta["config_sha256"] == now["params"]["configs"][cell]["sha256"],
                    f"train-start sidecar {meta_p} does not match the M0 state (cell / seed / warm-start path+hash / config path+hash / M0 hash)")
            end_p = f"{NS['logdir']}/train_end_{cell}_seed{seed}.json"
            require(os.path.isfile(end_p), f"missing train-end receipt {end_p}")
            end = load(end_p)
            ck_now = sha_file(p)
            require(end["cell"] == cell and end["seed"] == seed and end["ckpt"] == p and end["ckpt_sha256"] == ck_now
                    and end["r100_sha256_at_end"] == now["params"]["r100_sha256"] and end["config_sha256"] == now["params"]["configs"][cell]["sha256"]
                    and end["m0_sentinel_sha256"] == m0_sha,
                    f"train-end receipt {end_p} does not match the checkpoint now at {p} (swapped or replaced since training ended?)")
            runs[f"{cell}_seed{seed}"] = {"ckpt": p, "ckpt_sha256": ck_now, "ckpt_bytes": os.path.getsize(p),
                                        "train_log": logs[(cell, seed)], "train_log_sha256": sha_file(logs[(cell, seed)]),
                                        "train_meta": meta_p, "train_meta_sha256": sha_file(meta_p),
                                        "train_end": end_p, "train_end_sha256": sha_file(end_p),
                                        "config_sha256": now["params"]["configs"][cell]["sha256"], "seed": seed,
                                        "start_ckpt": now["params"]["r100"], "start_ckpt_sha256": now["params"]["r100_sha256"]}
    dump(a.path, {"written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "m0_sentinel": a.m0, "m0_sentinel_sha256": m0_sha,
                  "runs": runs, "params": now["params"], "schedule": now["schedule"],
                  # host-identity acceptance (empty when sealed on the M0 host itself); re-applied, never widened, by later checks
                  "accepted_paths": sorted(a.accept or []), "accepted_differences": accepted,
                  "sealed_on_m0_host": not accepted,
                  "host_at_seal": {"cpu_quota": now["environment"].get("cpu_quota"), "gpus": now["environment"].get("gpus")},
                  "attestation_at_seal": now["attestation"]})
    print("M1 sentinel written:", a.path, "runs:", sorted(runs), "accepted host differences:", len(accepted))
    return 0


def check_m1_core(a, m1_path):
    s = load(m1_path)
    bad = [] if sha_file(a.m0) == s["m0_sentinel_sha256"] else ["M0 sentinel file changed since M1"]
    for k, r in s["runs"].items():
        for what, path, want in (("checkpoint", r["ckpt"], r["ckpt_sha256"]), ("training log", r["train_log"], r["train_log_sha256"]),
                                 ("train-start sidecar", r["train_meta"], r["train_meta_sha256"]),
                                 ("train-end receipt", r["train_end"], r["train_end_sha256"])):
            if not os.path.isfile(path) or sha_file(path) != want:
                bad.append(f"{k}: {what} missing or changed ({path})")
    d, _, _, _ = check_m0_against(a, a.m0, s.get("accepted_paths", []))   # exactly the list sealed into M1, nothing more
    return bad + d, s


def cmd_check_m1(a):
    bad, s = check_m1_core(a, a.path)
    if bad:
        print("M1 STATE CHANGED (refusing):")
        for x in bad:
            print("  " + x)
        return 3
    print(f"M1 sentinel verified ({len(s['runs'])} runs unchanged; M0 state re-verified)")
    return 0


def cmd_record_eval(a):
    m1 = load(a.m1)
    run = m1["runs"].get(f"{a.cell}_seed{a.seed}")
    require(run is not None, f"{a.cell} seed {a.seed} is not in the M1 sentinel")
    require(os.path.realpath(a.ckpt) == os.path.realpath(run["ckpt"]), f"eval ckpt {a.ckpt} is not the M1 checkpoint {run['ckpt']}")
    csha = sha_file(a.ckpt)
    require(csha == run["ckpt_sha256"], "checkpoint hash differs from the M1 sentinel")
    args = log_args(a.log)
    ov = last_overall(a.log)
    checks = {"args.ckpts": args.get("ckpts") == a.ckpt, "args.mode": args.get("mode") == a.mode,
              "args.difficulty_source": args.get("difficulty_source") == "oracle", "args.deterministic": args.get("deterministic") == "True",
              "args.config": args.get("config") == cfg_path(a.cell),
              "args.exp_name": args.get("exp_name") == eval_exp(a.cell, a.seed, a.mode), "Overall": ov is not None}
    require(all(checks.values()), f"eval log identity failed: {checks}")
    dump(a.out, {"cell": a.cell, "seed": a.seed, "mode": a.mode, "ckpt": a.ckpt, "ckpt_sha256": csha, "log": a.log,
                 "log_sha256": sha_file(a.log), "overall": ov, "args": args, "m1_sentinel_sha256": sha_file(a.m1),
                 "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    print("eval recorded:", a.out, ov)
    return 0


def sidecar_paths(params):
    return {(c, s, m): f"{NS['logdir']}/eval_meta_{c}_seed{s}_{m}.json" for c in params["cells"] for s in params["seeds"] for m in MODES}


def cmd_write_m2(a):
    bad, _ = check_m1_core(a, a.m1)
    require(not bad, "M1 check failed; M2 not written:" + chr(10) + "  " + (chr(10) + "  ").join(bad))
    m1 = load(a.m1)
    m1_sha = sha_file(a.m1)
    side = {}
    for (c, s, m), p in sidecar_paths(m1["params"]).items():
        require(os.path.isfile(p), f"missing eval sidecar {p}")
        sc = load(p)
        run = m1["runs"][f"{c}_seed{s}"]
        require(sc["m1_sentinel_sha256"] == m1_sha and sc["ckpt_sha256"] == run["ckpt_sha256"] and sc["cell"] == c and sc["seed"] == s and sc["mode"] == m,
                f"sidecar {p} is not bound to this M1 sentinel / checkpoint")
        require(os.path.isfile(sc["log"]) and sha_file(sc["log"]) == sc["log_sha256"], f"eval log of {p} missing or changed")
        side[f"{c}_seed{s}_{m}"] = {"sidecar": p, "sidecar_sha256": sha_file(p), "log": sc["log"], "log_sha256": sc["log_sha256"], "overall": sc["overall"]}
    dump(a.path, {"written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "m0_sentinel_sha256": sha_file(a.m0), "m1_sentinel_sha256": m1_sha,
                  "evals": side, "n_evals": len(side)})
    print("M2 sentinel written:", a.path, "evals:", len(side))
    return 0


def cmd_check_m2(a):
    s = load(a.path)
    bad = []
    if sha_file(a.m0) != s["m0_sentinel_sha256"]:
        bad.append("M0 sentinel file changed since M2")
    if sha_file(a.m1) != s["m1_sentinel_sha256"]:
        bad.append("M1 sentinel file changed since M2")
    for k, e in s["evals"].items():
        if not os.path.isfile(e["sidecar"]) or sha_file(e["sidecar"]) != e["sidecar_sha256"]:
            bad.append(f"{k}: sidecar missing or changed")
        if not os.path.isfile(e["log"]) or sha_file(e["log"]) != e["log_sha256"]:
            bad.append(f"{k}: eval log missing or changed")
    more, _ = check_m1_core(a, a.m1)
    bad += more
    if bad:
        print("M2 STATE CHANGED (refusing):")
        for x in bad:
            print("  " + x)
        return 3
    print(f"M2 seal verified ({len(s['evals'])} evals; M1 and M0 states re-verified)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--r100", default="experiments/AdaPoinTr/ShapeNet55_models/baseline_sn55/ckpt-best.pth")
    ap.add_argument("--v2b42", default="experiments/AdaPoinTr_DiffCond_v2b/ShapeNet55_models/diffcond_v2b/ckpt-epoch-030.pth")
    ap.add_argument("--cells", default="v1_lambda0 v2b")
    ap.add_argument("--seeds", default="43 44")
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--gpus", default="0", help="CUDA_VISIBLE_DEVICES the launcher exported")
    ap.add_argument("--namespace", choices=sorted(NAMESPACES), default="formal", help="formal run or the tiny dress rehearsal")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("snapshot")
    sub.add_parser("verify-state", help="full dataset + archive re-hash, imports, expected environment; no sentinel")
    p = sub.add_parser("write-m0"); p.add_argument("path"); p.add_argument("--m0a-log", default=None); p.add_argument("--m0b-log", required=True); p.add_argument("--m0c-log", required=True)
    p = sub.add_parser("check-m0"); p.add_argument("path"); p.add_argument("--accept", action="append", help="host-identity path to accept (see ACCEPTABLE_DIFFS)")
    p = sub.add_parser("record-train-start"); p.add_argument("out"); p.add_argument("--cell", required=True); p.add_argument("--seed", type=int, required=True); p.add_argument("--m0", required=True)
    p = sub.add_parser("record-train-end"); p.add_argument("out"); p.add_argument("--cell", required=True); p.add_argument("--seed", type=int, required=True); p.add_argument("--m0", required=True)
    p = sub.add_parser("write-m1"); p.add_argument("path"); p.add_argument("--m0", required=True); p.add_argument("--train-log", action="append", help="cell:seed:path")
    p.add_argument("--accept", action="append", help="host-identity path to accept and record (see ACCEPTABLE_DIFFS); repeatable")
    p = sub.add_parser("check-m1"); p.add_argument("path"); p.add_argument("--m0", required=True)
    p = sub.add_parser("record-eval"); p.add_argument("out"); p.add_argument("--cell", required=True); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--mode", required=True); p.add_argument("--ckpt", required=True); p.add_argument("--log", required=True); p.add_argument("--m1", required=True)
    for name in ("write-m2", "check-m2"):
        p = sub.add_parser(name); p.add_argument("path"); p.add_argument("--m0", required=True); p.add_argument("--m1", required=True)
    a = ap.parse_args()
    set_namespace(a.namespace)
    if a.cmd == "snapshot":
        print(json.dumps(snapshot(a, full_verify=False), indent=2))
        return 0
    if a.cmd == "verify-state":
        now = snapshot(a, full_verify=True)
        problems = state_problems(now)
        env = {k: v for k, v in now["environment"].items() if k != "freeze"}
        env["freeze"] = {k: v for k, v in now["environment"]["freeze"].items() if k != "live"}
        print(json.dumps({"dataset": now["dataset"], "code_files": now["code"]["files"], "attest_tree_sha256": now["code"]["attest_tree_sha256"],
                          "attestation": now["attestation"], "compiled_import_errors": [k for k, v in now["compiled"].items() if "import_error" in v],
                          "environment": env, "params": now["params"], "problems": problems}, indent=2))
        return 0 if not problems else 3
    return {"write-m0": cmd_write_m0, "check-m0": cmd_check_m0, "record-train-start": cmd_record_train_start, "record-train-end": cmd_record_train_end, "write-m1": cmd_write_m1,
            "check-m1": cmd_check_m1, "record-eval": cmd_record_eval, "write-m2": cmd_write_m2, "check-m2": cmd_check_m2}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
