"""
Static audit of the DiffCond fine-tuned checkpoint + matched Vanilla-FT checkpoint
against the published Baseline checkpoint.

Purpose: verify
  1. diff_cond_query_ranking.diff_delta last-layer weight == 0 (dead branch)
  2. diff_cond_proj zero-or-near-zero
  3. diff_cond_embedding three-class cosine structure (collapse or not)
  4. shared-backbone drift norm: Vanilla-FT and DiffCond vs Baseline
  5. diff_cond_gate output statistics (requires forward pass, not just weights)
     --> skip in static audit, do in separate forced-mode probe

Run with:
  python tools/ckpt_static_audit.py \
    --baseline experiments/<baseline-run>/<pretrained-checkpoint>.pth \
    --diffcond experiments/<diffcond-run>/ckpt-epoch-030.pth \
    --vanilla_ft experiments/<vanilla-ft-run>/ckpt-epoch-030.pth

Outputs JSON to stdout + human-readable summary.
"""
import argparse
import json
import sys

import torch


def load_state(path):
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    # Typical keys: 'base_model', 'model', 'state_dict', etc.
    if isinstance(obj, dict):
        for key in ("base_model", "model", "state_dict"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    # fallthrough: assume it IS a state dict
    return obj


def strip_prefix(sd, prefix="module."):
    out = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
        else:
            out[k] = v
    return out


def norm(t):
    return float(torch.linalg.norm(t.float()).item())


def shared_backbone_drift(ref_sd, cand_sd):
    """L2 norm of delta for keys present in BOTH, EXCLUDING diff_cond_* / diff_predictor /
    base_model.diff_cond_* etc. Returns total drift + per-module breakdown."""
    per_module = {}
    total_sq = 0.0
    total_params = 0
    skipped = []
    for k, v_cand in cand_sd.items():
        # Skip DiffCond-specific keys
        if "diff_cond_" in k or "diff_predictor" in k:
            skipped.append(k)
            continue
        if k not in ref_sd:
            skipped.append(k + " (not in ref)")
            continue
        v_ref = ref_sd[k]
        if v_ref.shape != v_cand.shape:
            skipped.append(k + f" (shape mismatch ref={tuple(v_ref.shape)} cand={tuple(v_cand.shape)})")
            continue
        delta = (v_cand.float() - v_ref.float())
        d_sq = float((delta ** 2).sum().item())
        n_params = delta.numel()
        total_sq += d_sq
        total_params += n_params
        top = k.split(".")[0] if "." in k else k
        if top not in per_module:
            per_module[top] = {"delta_sq": 0.0, "n_params": 0, "ref_sq": 0.0}
        per_module[top]["delta_sq"] += d_sq
        per_module[top]["n_params"] += n_params
        per_module[top]["ref_sq"] += float((v_ref.float() ** 2).sum().item())
    total_drift = total_sq ** 0.5
    total_ref = sum(m["ref_sq"] for m in per_module.values()) ** 0.5
    return {
        "total_delta_norm": total_drift,
        "total_ref_norm": total_ref,
        "relative_drift": total_drift / (total_ref + 1e-12),
        "total_params_compared": total_params,
        "per_module": {
            k: {
                "delta_norm": m["delta_sq"] ** 0.5,
                "ref_norm": m["ref_sq"] ** 0.5,
                "relative": (m["delta_sq"] ** 0.5) / (m["ref_sq"] ** 0.5 + 1e-12),
                "n_params": m["n_params"],
            }
            for k, m in per_module.items()
        },
        "skipped_examples": skipped[:10],
    }


def diffcond_static(sd):
    """Extract DiffCond-specific static stats from a state dict."""
    stats = {}
    # diff_delta last layer
    # structure: diff_cond_query_ranking.diff_delta.{0,2,...}
    # last Linear is index -1 but in nn.Sequential stored as e.g. '2.weight'
    dd_keys = sorted([k for k in sd if "diff_cond_query_ranking.diff_delta" in k])
    stats["diff_delta_keys"] = dd_keys
    if dd_keys:
        # last Linear weight is typically the last ".weight" key
        weight_keys = [k for k in dd_keys if k.endswith(".weight")]
        bias_keys = [k for k in dd_keys if k.endswith(".bias")]
        if weight_keys:
            last_w = sd[weight_keys[-1]]
            stats["diff_delta_last_weight_norm"] = norm(last_w)
            stats["diff_delta_last_weight_abs_max"] = float(last_w.abs().max().item())
            stats["diff_delta_last_weight_is_zero"] = bool((last_w == 0).all().item())
            stats["diff_delta_last_weight_shape"] = tuple(last_w.shape)
            stats["diff_delta_last_weight_key"] = weight_keys[-1]
        if bias_keys:
            last_b = sd[bias_keys[-1]]
            stats["diff_delta_last_bias_norm"] = norm(last_b)
            stats["diff_delta_last_bias_is_zero"] = bool((last_b == 0).all().item())
    # diff_cond_proj
    proj_w_keys = [k for k in sd if "diff_cond_proj.weight" in k]
    proj_b_keys = [k for k in sd if "diff_cond_proj.bias" in k]
    if proj_w_keys:
        w = sd[proj_w_keys[0]]
        stats["diff_cond_proj_weight_norm"] = norm(w)
        stats["diff_cond_proj_weight_abs_max"] = float(w.abs().max().item())
        stats["diff_cond_proj_weight_is_zero"] = bool((w == 0).all().item())
    if proj_b_keys:
        b = sd[proj_b_keys[0]]
        stats["diff_cond_proj_bias_norm"] = norm(b)
        stats["diff_cond_proj_bias_is_zero"] = bool((b == 0).all().item())
    # diff_cond_gate (weights + bias stats)
    gate_w_keys = [k for k in sd if "diff_cond_gate" in k and k.endswith(".weight")]
    gate_b_keys = [k for k in sd if "diff_cond_gate" in k and k.endswith(".bias")]
    if gate_w_keys:
        w = sd[gate_w_keys[0]]
        stats["diff_cond_gate_weight_norm"] = norm(w)
        stats["diff_cond_gate_weight_std"] = float(w.float().std().item())
    if gate_b_keys:
        b = sd[gate_b_keys[0]]
        stats["diff_cond_gate_bias_norm"] = norm(b)
        stats["diff_cond_gate_bias_mean"] = float(b.float().mean().item())
    # diff_cond_embedding
    emb_keys = [k for k in sd if "diff_cond_embedding.embed.weight" in k]
    if emb_keys:
        E = sd[emb_keys[0]].float()  # (3, dim)
        stats["diff_cond_embedding_shape"] = tuple(E.shape)
        stats["diff_cond_embedding_rowwise_norms"] = [norm(E[i]) for i in range(E.shape[0])]
        # pairwise cosine
        cos = {}
        for i in range(E.shape[0]):
            for j in range(i + 1, E.shape[0]):
                num = float((E[i] @ E[j]).item())
                den = (norm(E[i]) * norm(E[j])) + 1e-12
                cos[f"{i}-{j}"] = num / den
        stats["diff_cond_embedding_pairwise_cosine"] = cos
    # diff_predictor
    pred_w_keys = sorted([k for k in sd if "diff_predictor" in k and k.endswith(".weight")])
    if pred_w_keys:
        stats["diff_predictor_weight_keys"] = pred_w_keys
        stats["diff_predictor_last_weight_norm"] = norm(sd[pred_w_keys[-1]])
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--diffcond", required=True)
    ap.add_argument("--vanilla_ft", required=True)
    ap.add_argument("--out", default=None, help="Write JSON here")
    args = ap.parse_args()

    print("[load] Baseline ...", file=sys.stderr)
    sd_baseline = strip_prefix(load_state(args.baseline))
    print("[load] DiffCond ...", file=sys.stderr)
    sd_diffcond = strip_prefix(load_state(args.diffcond))
    print("[load] Vanilla-FT ...", file=sys.stderr)
    sd_vanilla_ft = strip_prefix(load_state(args.vanilla_ft))

    report = {
        "diffcond_static": diffcond_static(sd_diffcond),
        "diffcond_shared_drift_vs_baseline": shared_backbone_drift(sd_baseline, sd_diffcond),
        "vanilla_ft_shared_drift_vs_baseline": shared_backbone_drift(sd_baseline, sd_vanilla_ft),
        "sd_key_counts": {
            "baseline": len(sd_baseline),
            "diffcond": len(sd_diffcond),
            "vanilla_ft": len(sd_vanilla_ft),
        },
    }

    # Human summary
    print("=" * 72)
    print("STATIC CKPT AUDIT  (Diagnostic for shared-backbone drift, ranking-delta dead branch, and gate/embedding statistics)")
    print("=" * 72)
    dc = report["diffcond_static"]
    print(f"\n[DiffCond DiffCond static]")
    print(f"  diff_delta last-layer weight norm : {dc.get('diff_delta_last_weight_norm')}")
    print(f"  diff_delta last-layer weight |max| : {dc.get('diff_delta_last_weight_abs_max')}")
    print(f"  diff_delta last-layer weight ==0?   : {dc.get('diff_delta_last_weight_is_zero')}  (expected: True)")
    print(f"  diff_delta last-layer weight shape : {dc.get('diff_delta_last_weight_shape')}")
    print(f"  diff_delta last-layer bias ==0?     : {dc.get('diff_delta_last_bias_is_zero')}")
    print(f"  diff_cond_proj weight norm         : {dc.get('diff_cond_proj_weight_norm')}")
    print(f"  diff_cond_proj weight |max|        : {dc.get('diff_cond_proj_weight_abs_max')}")
    print(f"  diff_cond_proj weight ==0?          : {dc.get('diff_cond_proj_weight_is_zero')}")
    print(f"  diff_cond_proj bias norm            : {dc.get('diff_cond_proj_bias_norm')}")
    print(f"  diff_cond_gate weight norm          : {dc.get('diff_cond_gate_weight_norm')}")
    print(f"  diff_cond_gate bias mean            : {dc.get('diff_cond_gate_bias_mean')}")
    print(f"  diff_cond_embedding shape           : {dc.get('diff_cond_embedding_shape')}")
    print(f"  embedding row norms                 : {dc.get('diff_cond_embedding_rowwise_norms')}")
    print(f"  embedding pairwise cos              : {dc.get('diff_cond_embedding_pairwise_cosine')}")

    drift_diffcond = report["diffcond_shared_drift_vs_baseline"]
    drift_vanilla = report["vanilla_ft_shared_drift_vs_baseline"]
    print(f"\n[Shared-backbone drift vs Baseline checkpoint]")
    print(f"  DiffCond   total_delta_norm={drift_diffcond['total_delta_norm']:.4f}  relative={drift_diffcond['relative_drift']:.4f}")
    print(f"  Vanilla-FT total_delta_norm={drift_vanilla['total_delta_norm']:.4f}  relative={drift_vanilla['relative_drift']:.4f}")

    print(f"\n[Per-module relative drift   (DiffCond vs Vanilla-FT, %)]")
    keys = sorted(set(drift_diffcond["per_module"].keys()) | set(drift_vanilla["per_module"].keys()))
    print(f"  {'module':<30s} {'DiffCond_rel':>13s} {'VanillaFT_rel':>14s} {'diff':>10s}")
    for k in keys:
        rd = drift_diffcond["per_module"].get(k, {}).get("relative", 0) * 100
        rv = drift_vanilla["per_module"].get(k, {}).get("relative", 0) * 100
        print(f"  {k:<30s} {rd:>13.3f} {rv:>14.3f} {rd - rv:>+10.3f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n[wrote] {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
