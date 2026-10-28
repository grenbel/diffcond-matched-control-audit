#!/usr/bin/env python3
"""Distribution-level quantification of the Pitfall-3 crop-path mismatch.

Simulates the two training-time crop samplers as implemented in the patched
PoinTr pipeline and quantifies how their num_crop distributions differ. No
point-cloud data is required: the two paths share the identical geometric crop
routine (misc.seprate_point_cloud viewpoint + argsort) and an FPS-to-2048 step
(Path A inside seprate_point_cloud, Path B outside), so every path-induced
difference is driven by the num_crop sampling law and its batch granularity,
which is what this script measures.

Transcribed sampler logic (verified against source 2026-08-21; re-verified
2026-09-08):

Path A -- stock vanilla "list-crop" (tools/runner.py:217 + utils/misc.py:165-167,
195-197): every sample in the batch independently draws
    num_crop ~ random.randint(npoints/4, 3*npoints/4)     # inclusive ends
then crops and FPS-to-2048 *inside* seprate_point_cloud.

Path B -- DiffCond shared "int-crop+FPS" (tools/runner.py:186-196 +
utils/misc.py:168-169, 198-200): once per batch (per iteration)
    ratio ~ random.uniform(0.25, 0.75)
    num_crop = int(npoints * ratio)                        # floor
shared by all samples in the batch; FPS-to-2048 applied *outside*.

Difficulty bins follow runner.py:201-206 (ratio < 0.375 -> Simple, < 0.625 ->
Moderate, else Hard). With num_crop = floor(npoints * ratio) and npoints
divisible by 8 this is equivalent to thresholding num_crop at 0.375*npoints and
0.625*npoints, which is how bins are assigned here. Path A never produces a
difficulty label during training: its bin statistics are a post-hoc binning of
num_crop by the same thresholds, used only to describe batch composition.

Two kinds of numbers are reported and kept separate:
  analytic  -- exact values under the ideal continuous-uniform model of the two
               samplers (these are the values to quote in formal text);
  simulated -- Monte Carlo realisations (NumPy's Generator reproduces the
               distribution law; it is not a bit-exact transcription of the
               CPython `random` module used in training).
The built-in self-check compares every simulated statistic with its analytic
reference using an estimator-specific standard error (block means for
means/SDs/variances, Bernoulli SEs for proportions), adds deterministic
assertions on support, bin structure and the KS distance, and exits non-zero
on any failure.

Endpoint note: under the ideal model Path B never reaches 3*npoints/4 (ratio
exactly 0.75 has probability zero). Both CPython's random.uniform (training)
and NumPy's Generator.uniform (this simulation) document that the upper
endpoint can be returned through floating-point rounding, with probability of
order 2^-53; the Path-B maximum hi-1 is therefore an almost-sure bound, not a
hard one. Support checks below allow hi; "observed max == hi-1" is reported as a
reproducibility check of the default run, not as a sampler-law invariant.

Review history: an adversarial code review on 2026-08-21 returned NEEDS_FIXES (wrong
closed-form within-batch SD reference; tolerances not estimator-specific;
endpoint wording; simulated/analytic values mixed in output; npoints
divisibility unchecked). Round 2 (2026-09-08) found the self-check still blind
to several quoted analytic fields (RMS within-batch SD, B within-batch SD,
homogeneous probability) and a tautological composition-sum check; this version
adds exact-rational recomputation of the homogeneous probability, an
independent two-bin probability, direct checks of the quoted fields, small-batch
empirical validation of the composition formulas (batch sizes where the events
are observable), and enforces 2 <= blocks <= iters. Round 3 (2026-09-09) showed
that a wrong analytic variance (+2000) or wrong bin counts ([1022, 2052, 1023])
still passed, because every check derived from the same analytic() fields and
the composition sum is an algebraic identity; this version (v4) adds oracle(): an
independent rational recomputation (literal counts, variances by summation over
the support, batch composition by a 2^3-state dynamic programme, exact KS) that
every quoted analytic field must match to 1e-9, and relabels the composition-sum
assertion as a low-level invariant. Round 4 (2026-09-09) found two analytic
fields still outside the oracle (B_homogeneous_batch_prob, B_support_almost_sure)
and that the oracle's within-batch variance reused the (b-1)/b closed form; v5
adds both fields (plus B_within_batch_std and A_support) to the comparison, prints
the B homogeneous value from the checked field, and derives the within-batch
variance from directly summed moments E[X], E[X^2]. Round 5 found the three
standard-deviation fields compared only through their squares (a negative RMS
passed); v6 compares each of them directly with the positive root.
"""

import argparse
import json
import math
import sys
from fractions import Fraction

import numpy as np


def simulate(npoints, batch_size, iters, seed):
    """Draw num_crop matrices of shape (iters, batch_size) for both paths."""
    rng = np.random.default_rng(seed)
    lo, hi = npoints // 4, 3 * npoints // 4          # 2048, 6144 for 8192

    # Path A: per-sample inclusive randint over [lo, hi]
    a = rng.integers(lo, hi + 1, size=(iters, batch_size))
    # Path B: per-batch floor(npoints * U(0.25, 0.75)), broadcast to the batch.
    # rng.uniform nominally samples [0.25, 0.75); NumPy documents that high may be
    # returned through rounding, exactly as CPython's random.uniform does.
    b_ratio = rng.uniform(0.25, 0.75, size=iters)
    b_scalar = np.floor(npoints * b_ratio).astype(np.int64)
    b = np.repeat(b_scalar[:, None], batch_size, axis=1)
    return a, b


def bin_of(num_crop, npoints):
    """Post-hoc difficulty bin (0=S, 1=M, 2=H) per runner.py:201-206 thresholds."""
    t1, t2 = 3 * npoints // 8, 5 * npoints // 8      # 3072, 5120 for 8192
    return np.where(num_crop < t1, 0, np.where(num_crop < t2, 1, 2))


def analytic(npoints, batch_size):
    """Exact reference values under the ideal continuous-uniform model."""
    lo, hi = npoints // 4, 3 * npoints // 4
    n_a = hi - lo + 1                     # Path A support {lo..hi}: 4097 values
    n_b = n_a - 1                         # Path B support {lo..hi-1}: 4096 values
    t1, t2 = 3 * npoints // 8, 5 * npoints // 8
    bs = batch_size

    a_mean, a_var = (lo + hi) / 2.0, (n_a ** 2 - 1) / 12.0
    b_mean, b_var = (lo + hi - 1) / 2.0, (n_b ** 2 - 1) / 12.0
    # The population variance (ddof=0) of bs iid draws has expectation
    # (bs-1)/bs * var; its square root is the RMS within-batch SD.
    a_wb_var = (bs - 1) / bs * a_var

    # Exact bin counts on each support.
    a_counts = [t1 - lo, t2 - t1, hi - t2 + 1]            # 1024 / 2048 / 1025
    b_counts = [t1 - lo, t2 - t1, (hi - 1) - t2 + 1]      # 1024 / 2048 / 1024
    a_p = [c / n_a for c in a_counts]
    b_p = [c / n_b for c in b_counts]
    # Batch composition for Path A (Path B batches are constant by construction).
    # Three routes that must agree, checked non-trivially in self_check():
    #   homogeneous  = sum_k p_k^bs (also recomputed with exact rational arithmetic),
    #   exactly two  = pairwise inclusion-exclusion over bin pairs,
    #   all three    = inclusion-exclusion over "bin k absent" events.
    a_homog = sum(p ** bs for p in a_p)
    a_homog_exact = float(sum(Fraction(c, n_a) ** bs for c in a_counts))
    a_two = sum((a_p[i] + a_p[j]) ** bs - a_p[i] ** bs - a_p[j] ** bs
                for i, j in ((0, 1), (0, 2), (1, 2)))
    a_three = 1.0 - sum((1.0 - p) ** bs for p in a_p) + a_homog

    # KS distance between the two marginal pmfs, computed from the pmfs directly.
    support = np.arange(lo, hi + 1)
    pmf_a = np.full(n_a, 1.0 / n_a)
    pmf_b = np.where(support <= hi - 1, 1.0 / n_b, 0.0)
    ks_exact = float(np.abs(np.cumsum(pmf_a) - np.cumsum(pmf_b)).max())

    return {
        "A_mean": a_mean, "A_std": math.sqrt(a_var),
        "B_mean": b_mean, "B_std": math.sqrt(b_var),
        "A_within_batch_var_expected": a_wb_var,
        "A_within_batch_std_rms": math.sqrt(a_wb_var),
        "B_within_batch_std": 0.0,
        "A_bin_probs_S_M_H": a_p, "B_bin_probs_S_M_H": b_p,
        "A_homogeneous_batch_prob": a_homog,
        "A_homogeneous_batch_prob_exact_rational": a_homog_exact,
        "A_two_bin_batch_prob": a_two,
        "A_three_bin_batch_prob": a_three,
        "B_homogeneous_batch_prob": 1.0,
        "A_support": [lo, hi],
        "B_support_almost_sure": [lo, hi - 1],
        "ks_marginal_exact": ks_exact,
        "ks_marginal_closed_form": 1.0 / n_a,
    }


def block_se(values):
    """Standard error of a statistic from its per-block replicates."""
    v = np.asarray(values, dtype=float)
    return float(v.std(ddof=1) / math.sqrt(len(v)))


def batch_stats(mat, npoints, nblocks):
    """Simulated statistics for an (iters, batch) num_crop matrix, with block SEs."""
    within_var = mat.var(axis=1, ddof=0)
    bins = bin_of(mat, npoints)
    present = np.stack([(bins == k).any(axis=1) for k in range(3)], axis=1)
    n_bins = present.sum(axis=1)
    blocks = np.array_split(mat, nblocks, axis=0)
    return {
        "marginal_mean": float(mat.mean()),
        "marginal_std": float(mat.std(ddof=0)),
        "observed_min": int(mat.min()),
        "observed_max": int(mat.max()),
        "quantiles_p5_p25_p50_p75_p95": [
            float(q) for q in np.quantile(mat, [0.05, 0.25, 0.5, 0.75, 0.95])
        ],
        "within_batch_std_mean": float(np.sqrt(within_var).mean()),
        "within_batch_std_rms": float(math.sqrt(within_var.mean())),
        "within_batch_var_mean": float(within_var.mean()),
        "homogeneous_batch_frac": float((n_bins == 1).mean()),
        "two_bin_batch_frac": float((n_bins == 2).mean()),
        "three_bin_batch_frac": float((n_bins == 3).mean()),
        "homogeneous_batch_count": int((n_bins == 1).sum()),
        "bin_occupancy_S_M_H": [
            float(x) for x in np.bincount(bins.ravel(), minlength=3) / bins.size
        ],
        "_block_se": {
            "marginal_mean": block_se([c.mean() for c in blocks]),
            "marginal_std": block_se([c.std(ddof=0) for c in blocks]),
            "within_batch_var_mean": block_se(
                [c.var(axis=1, ddof=0).mean() for c in blocks]),
        },
    }


def ks_observed(a, b, npoints):
    """Two-sample KS statistic between the simulated marginals (noise-dominated)."""
    support = np.arange(npoints // 4, 3 * npoints // 4 + 2)
    cdf_a = np.searchsorted(np.sort(a.ravel()), support, side="right") / a.size
    cdf_b = np.searchsorted(np.sort(b.ravel()), support, side="right") / b.size
    return float(np.abs(cdf_a - cdf_b).max())


def oracle(npoints, batch_size):
    """Independent exact oracle for every analytic field quoted in the paper, built from
    npoints and batch_size alone with rational arithmetic and DIFFERENT routes than analytic():
    literal bin counts (npoints/8, npoints/4, npoints/8 + 1), variances by direct summation over
    the support, batch composition by a 2^3-state dynamic programme over the draws, KS by exact
    CDF differences. analytic() must reproduce these to 1e-9 relative; an error in analytic()'s
    counts, variance or inclusion-exclusion therefore fails the self-check instead of propagating."""
    lo, hi = npoints // 4, 3 * npoints // 4
    n_a, n_b = hi - lo + 1, hi - lo
    counts_a = [npoints // 8, npoints // 4, npoints // 8 + 1]
    counts_b = [npoints // 8, npoints // 4, npoints // 8]
    assert sum(counts_a) == n_a and sum(counts_b) == n_b
    support_a = range(lo, hi + 1)
    support_b = range(lo, hi)

    def moments(support):
        n = len(support)
        mean = Fraction(sum(support), n)
        ex2 = Fraction(sum(k * k for k in support), n)
        var = Fraction(sum((k - mean) ** 2 for k in support), n)
        return mean, ex2, var
    mean_a, ex2_a, var_a = moments(support_a)
    mean_b, ex2_b, var_b = moments(support_b)
    # expected within-batch population variance E[(1/b) sum X_i^2 - Xbar^2] = E[X^2] - E[Xbar^2],
    # with E[Xbar^2] = Var(Xbar) + E[X]^2 and Var(Xbar) = Var(X)/b: computed from the summed moments,
    # not from analytic()'s (b-1)/b closed form
    wb_var = ex2_a - (var_a / batch_size + mean_a * mean_a)
    p_a = [Fraction(c, n_a) for c in counts_a]
    p_b = [Fraction(c, n_b) for c in counts_b]
    # batch composition: probability of each set of bins seen after batch_size draws
    dist = {0: Fraction(1)}
    for _ in range(batch_size):
        nxt = {}
        for mask, pr in dist.items():
            for k in range(3):
                nxt[mask | (1 << k)] = nxt.get(mask | (1 << k), Fraction(0)) + pr * p_a[k]
        dist = nxt
    homog = sum(pr for m, pr in dist.items() if bin(m).count("1") == 1)
    two = sum(pr for m, pr in dist.items() if bin(m).count("1") == 2)
    three = dist.get(7, Fraction(0))
    # KS: max |F_A(k) - F_B(k)| over the support, exact
    ks = max(abs(Fraction(i + 1, n_a) - Fraction(min(i + 1, n_b), n_b)) for i in range(n_a))
    return {"counts_a": counts_a, "counts_b": counts_b, "mean_a": mean_a, "var_a": var_a,
            "mean_b": mean_b, "var_b": var_b, "wb_var": wb_var,
            "p_a": p_a, "p_b": p_b, "homog": homog, "two": two, "three": three, "ks": ks,
            "b_homog": Fraction(1), "b_wb_var": Fraction(0),          # Path B: one value per batch
            "support_a": [lo, hi], "support_b": [lo, hi - 1]}


def self_check(A, B, ref, npoints, batch_size, iters):
    """Estimator-specific 4-SE tolerances, deterministic assertions, and a field-by-field
    comparison of analytic() against the independent rational oracle()."""
    n_a_eff = iters * batch_size     # Path A: iid values
    n_b_eff = iters                  # Path B: one independent value per iteration
    lo, hi = npoints // 4, 3 * npoints // 4

    def bern_se(p, n):
        return math.sqrt(p * (1.0 - p) / n)

    stat, det = [], []

    def add(name, got, want, tol):
        stat.append({"check": name, "simulated": got, "analytic": want,
                     "tolerance": tol, "pass": bool(abs(got - want) <= tol)})

    def assert_(name, ok, detail, repro=False):
        det.append({"check": name, "pass": bool(ok), "detail": detail, "repro": repro})

    se_a, se_b = A["_block_se"], B["_block_se"]
    add("A marginal mean", A["marginal_mean"], ref["A_mean"], 4 * se_a["marginal_mean"])
    add("A marginal std", A["marginal_std"], ref["A_std"], 4 * se_a["marginal_std"])
    add("B marginal mean", B["marginal_mean"], ref["B_mean"], 4 * se_b["marginal_mean"])
    add("B marginal std", B["marginal_std"], ref["B_std"], 4 * se_b["marginal_std"])
    add("A within-batch var (mean over batches)", A["within_batch_var_mean"],
        ref["A_within_batch_var_expected"], 4 * se_a["within_batch_var_mean"])
    assert_("A_within_batch_std_rms^2 equals A_within_batch_var_expected",
            abs(ref["A_within_batch_std_rms"] ** 2 - ref["A_within_batch_var_expected"])
            <= 1e-9 * ref["A_within_batch_var_expected"],
            f"{ref['A_within_batch_std_rms']:.6f}^2 vs {ref['A_within_batch_var_expected']:.6f}")
    assert_("A homogeneous prob: float formula equals exact-rational recomputation",
            abs(ref["A_homogeneous_batch_prob"] - ref["A_homogeneous_batch_prob_exact_rational"])
            <= 1e-9 * ref["A_homogeneous_batch_prob_exact_rational"],
            f"{ref['A_homogeneous_batch_prob']:.6e} vs {ref['A_homogeneous_batch_prob_exact_rational']:.6e}")
    add("B within-batch std vs analytic field", B["within_batch_std_mean"], ref["B_within_batch_std"], 0.0)
    for k, lab in enumerate("SMH"):
        pa, pb = ref["A_bin_probs_S_M_H"][k], ref["B_bin_probs_S_M_H"][k]
        add(f"A bin occupancy {lab}", A["bin_occupancy_S_M_H"][k], pa, 4 * bern_se(pa, n_a_eff))
        add(f"B bin occupancy {lab}", B["bin_occupancy_S_M_H"][k], pb, 4 * bern_se(pb, n_b_eff))
    # Expected number of difficulty-homogeneous Path-A batches is iters * ~3.5e-15.
    add("A homogeneous batch count (expected ~0, allow <=1)", A["homogeneous_batch_count"], 0, 1)
    p3 = ref["A_three_bin_batch_prob"]
    add("A three-bin batch frac", A["three_bin_batch_frac"], p3, 4 * bern_se(p3, iters))
    add("B homogeneous batch frac (exact 1)", B["homogeneous_batch_frac"], 1.0, 0.0)
    add("B three-bin batch frac (exact 0)", B["three_bin_batch_frac"], 0.0, 0.0)

    assert_("A observed support within [lo, hi]",
            lo <= A["observed_min"] and A["observed_max"] <= hi,
            f"observed [{A['observed_min']}, {A['observed_max']}] vs [{lo}, {hi}]")
    assert_("A attains both endpoints (miss prob ~exp(-iters*bs/n_a))",
            A["observed_min"] == lo and A["observed_max"] == hi,
            f"observed [{A['observed_min']}, {A['observed_max']}]", repro=True)
    assert_("B observed support within [lo, hi] (hi only via endpoint rounding)",
            lo <= B["observed_min"] and B["observed_max"] <= hi,
            f"observed [{B['observed_min']}, {B['observed_max']}]")
    assert_("B observed max == hi-1 (reproducibility check of the default run; the law "
            "allows hi with probability ~2^-53)",
            B["observed_max"] == hi - 1, f"observed max {B['observed_max']} vs {hi - 1}", repro=True)
    assert_("exact KS from pmfs equals closed form 1/n_a",
            abs(ref["ks_marginal_exact"] - ref["ks_marginal_closed_form"]) < 1e-12,
            f"{ref['ks_marginal_exact']:.9f} vs {ref['ks_marginal_closed_form']:.9f}")
    assert_("bin probabilities sum to 1 (A and B)",
            abs(sum(ref["A_bin_probs_S_M_H"]) - 1) < 1e-12
            and abs(sum(ref["B_bin_probs_S_M_H"]) - 1) < 1e-12,
            f"{sum(ref['A_bin_probs_S_M_H'])}, {sum(ref['B_bin_probs_S_M_H'])}")
    assert_("A batch-composition probabilities sum to 1 (low-level invariant only: the three "
            "closed forms satisfy this identity algebraically, so it is not an independent check)",
            abs(ref["A_homogeneous_batch_prob"] + ref["A_two_bin_batch_prob"]
                + ref["A_three_bin_batch_prob"] - 1) < 1e-12,
            "sum_k p^bs + pairwise inclusion-exclusion + all-three inclusion-exclusion")
    # Independent oracle: every quoted analytic field must agree with a rational recomputation
    # that does not share analytic()'s counts, closed forms or inclusion-exclusion.
    orc = oracle(npoints, batch_size)

    def close(got, want, rel=1e-9):
        want = float(want)
        return abs(got - want) <= rel * max(abs(want), 1e-300)
    pairs = [
        ("A_mean", ref["A_mean"], orc["mean_a"]), ("B_mean", ref["B_mean"], orc["mean_b"]),
        ("A_std", ref["A_std"], math.sqrt(float(orc["var_a"]))), ("B_std", ref["B_std"], math.sqrt(float(orc["var_b"]))),
        ("A_within_batch_var_expected", ref["A_within_batch_var_expected"], orc["wb_var"]),
        ("A_within_batch_std_rms", ref["A_within_batch_std_rms"], math.sqrt(float(orc["wb_var"]))),
        ("A_homogeneous_batch_prob", ref["A_homogeneous_batch_prob"], orc["homog"]),
        ("A_homogeneous_batch_prob_exact_rational", ref["A_homogeneous_batch_prob_exact_rational"], orc["homog"]),
        ("A_two_bin_batch_prob", ref["A_two_bin_batch_prob"], orc["two"]),
        ("A_three_bin_batch_prob", ref["A_three_bin_batch_prob"], orc["three"]),
        ("ks_marginal_exact", ref["ks_marginal_exact"], orc["ks"]),
        ("ks_marginal_closed_form", ref["ks_marginal_closed_form"], orc["ks"]),
        ("B_homogeneous_batch_prob", ref["B_homogeneous_batch_prob"], orc["b_homog"]),
        ("B_within_batch_std", ref["B_within_batch_std"], orc["b_wb_var"]),
    ]
    for k, lab in enumerate("SMH"):
        pairs.append((f"A_bin_probs_S_M_H[{lab}]", ref["A_bin_probs_S_M_H"][k], orc["p_a"][k]))
        pairs.append((f"B_bin_probs_S_M_H[{lab}]", ref["B_bin_probs_S_M_H"][k], orc["p_b"][k]))
    for name, got, want in pairs:
        assert_(f"oracle: {name}", close(got, want), f"analytic {got!r} vs oracle {float(want)!r}")
    assert_("oracle: A_support and B_support_almost_sure",
            ref["A_support"] == orc["support_a"] and ref["B_support_almost_sure"] == orc["support_b"],
            f"analytic {ref['A_support']} / {ref['B_support_almost_sure']} vs oracle {orc['support_a']} / {orc['support_b']}")
    assert_("oracle: A support size and literal bin counts",
            ref["A_support"] == [npoints // 4, 3 * npoints // 4]
            and [round(p * (ref["A_support"][1] - ref["A_support"][0] + 1)) for p in ref["A_bin_probs_S_M_H"]] == orc["counts_a"],
            f"counts from analytic probs vs literal {orc['counts_a']}")
    return stat, det


def small_batch_checks(npoints, seed, iters_aux=20000):
    """Validate the batch-composition formulas at batch sizes where the events are
    observable: bs=2 (homogeneous, two-bin) and bs=3 (homogeneous, two-bin,
    three-bin). The default bs=48 homogeneous probability (~3.5e-15) cannot be
    observed directly; this exercises the same analytic() code path where it can."""
    out = []

    def bern_se(p, n):
        return math.sqrt(p * (1.0 - p) / n)

    plan = ((2, ["A_homogeneous_batch_prob", "A_two_bin_batch_prob"]),
            (3, ["A_homogeneous_batch_prob", "A_two_bin_batch_prob", "A_three_bin_batch_prob"]))
    for bs, keys in plan:
        a, _ = simulate(npoints, bs, iters_aux, seed + bs)
        bins = bin_of(a, npoints)
        n_bins = np.stack([(bins == k).any(axis=1) for k in range(3)], axis=1).sum(axis=1)
        ref = analytic(npoints, bs)
        observed = {"A_homogeneous_batch_prob": float((n_bins == 1).mean()),
                    "A_two_bin_batch_prob": float((n_bins == 2).mean()),
                    "A_three_bin_batch_prob": float((n_bins == 3).mean())}
        for k in keys:
            p, tol = ref[k], 4 * bern_se(ref[k], iters_aux)
            out.append({"check": f"small-batch validation bs={bs}: {k}", "simulated": observed[k],
                        "analytic": p, "tolerance": tol, "pass": bool(abs(observed[k] - p) <= tol)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npoints", type=int, default=8192,
                    help="N_POINTS from cfgs/dataset_configs/ShapeNet-55.yaml; must be divisible by 8")
    ap.add_argument("--batch-size", type=int, default=48,
                    help="total_bs from the fine-tune configs (verified 48)")
    ap.add_argument("--iters", type=int, default=50000,
                    help="simulated training iterations (batches)")
    ap.add_argument("--blocks", type=int, default=50,
                    help="number of contiguous blocks for block-mean standard errors")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=str, default=None, help="output JSON path")
    args = ap.parse_args()
    if args.npoints % 8 != 0:
        sys.exit("npoints must be divisible by 8 (bin thresholds and the Path-B "
                 "equiprobable integer support rely on it)")
    if args.batch_size < 2 or not (2 <= args.blocks <= args.iters):
        sys.exit("need batch_size >= 2 and 2 <= blocks <= iters "
                 "(block-mean standard errors require at least two blocks)")

    a, b = simulate(args.npoints, args.batch_size, args.iters, args.seed)
    ref = analytic(args.npoints, args.batch_size)
    A = batch_stats(a, args.npoints, args.blocks)
    B = batch_stats(b, args.npoints, args.blocks)
    ks_obs = ks_observed(a, b, args.npoints)
    stat, det = self_check(A, B, ref, args.npoints, args.batch_size, args.iters)
    stat += small_batch_checks(args.npoints, args.seed)
    # Endpoint-attainment checks describe the recorded default run; they are enforced
    # only for that run (other settings report them without failing).
    strict_repro = (args.npoints, args.batch_size, args.iters, args.seed) == (8192, 48, 50000, 2026)
    failures = ([c["check"] for c in stat if not c["pass"]]
                + [c["check"] for c in det if not c["pass"] and (strict_repro or not c["repro"])])
    n_support_a = ref["A_support"][1] - ref["A_support"][0] + 1

    res = {
        "provenance": {
            "script": "tools/crop_path_quantification.py",
            "sampler_sources": [
                "tools/runner.py:217 + utils/misc.py:165-167,195-197 (Path A)",
                "tools/runner.py:186-196 + utils/misc.py:168-169,198-200 (Path B)",
            ],
            "bin_boundary_source": "tools/runner.py:201-206 (ratio 0.375 / 0.625)",
            "npoints": args.npoints, "batch_size": args.batch_size,
            "iters": args.iters, "seed": args.seed, "blocks": args.blocks,
            "note": ("Both paths share the identical geometric crop routine and an "
                     "FPS-to-2048 step (Path A inside seprate_point_cloud, Path B "
                     "outside); differences quantified here stem from the num_crop "
                     "sampling law and its batch granularity. 'analytic' values are "
                     "exact under the ideal continuous-uniform model and are the ones "
                     "to quote; 'simulated' values are Monte Carlo realisations. Bin "
                     "statistics for Path A are post-hoc binning by the S/M/H "
                     "thresholds (the vanilla path produces no difficulty label)."),
        },
        "analytic": ref,
        "simulated": {
            "path_A_list_crop": A,
            "path_B_int_crop": B,
            "ks_marginal_observed": ks_obs,
            "ks_note": ("Observed two-sample KS between the simulated marginals is "
                        f"dominated by Monte Carlo noise (Path B has only {args.iters} "
                        "independent draws, so the noise scale is ~1/sqrt(iters) = "
                        f"{1 / math.sqrt(args.iters):.4f}); the exact KS distance between "
                        f"the two laws is 1/{n_support_a} = "
                        f"{ref['ks_marginal_closed_form']:.6f}."),
        },
        "self_check": {"passed": not failures, "statistical_checks": stat,
                       "deterministic_checks": det, "failures": failures,
                       "reproducibility_checks_enforced": strict_repro},
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(json.dumps(res, indent=2) + "\n")

    lo, hi = ref["A_support"]
    print("=== Crop-path mismatch quantification (distribution level) ===")
    print(f"npoints={args.npoints} batch_size={args.batch_size} | ANALYTIC = exact under "
          f"the ideal model (quote these) | SIMULATED = Monte Carlo, iters={args.iters}, "
          f"seed={args.seed}")
    print("ANALYTIC")
    print(f"  marginal num_crop      A mean {ref['A_mean']:.1f} sd {ref['A_std']:.2f} "
          f"(support {lo}..{hi})   B mean {ref['B_mean']:.1f} sd {ref['B_std']:.2f} "
          f"(support {lo}..{hi - 1} almost surely)")
    print(f"  KS distance (laws)     1/{n_support_a} = {ref['ks_marginal_closed_form']:.6f}")
    print(f"  within-batch sd        A rms {ref['A_within_batch_std_rms']:.2f} "
          f"(= sqrt({args.batch_size - 1}/{args.batch_size}) * sd)   B 0 (shared scalar)")
    pa, pb = ref["A_bin_probs_S_M_H"], ref["B_bin_probs_S_M_H"]
    print(f"  post-hoc S/M/H probs   A {pa[0]:.5f}/{pa[1]:.5f}/{pa[2]:.5f}   "
          f"B {pb[0]:.5f}/{pb[1]:.5f}/{pb[2]:.5f}")
    print(f"  P(batch homogeneous)   A {ref['A_homogeneous_batch_prob']:.3e}   B {ref['B_homogeneous_batch_prob']:g}")
    print(f"  P(batch spans 3 bins)  A {ref['A_three_bin_batch_prob']:.7f}   B 0")
    print("SIMULATED")
    print(f"  marginal mean/sd       A {A['marginal_mean']:.1f}/{A['marginal_std']:.2f} "
          f"obs [{A['observed_min']}, {A['observed_max']}]   B {B['marginal_mean']:.1f}/"
          f"{B['marginal_std']:.2f} obs [{B['observed_min']}, {B['observed_max']}]")
    print(f"  within-batch sd        A mean {A['within_batch_std_mean']:.2f} rms "
          f"{A['within_batch_std_rms']:.2f}   B {B['within_batch_std_mean']:.1f}")
    print(f"  homogeneous batches    A {A['homogeneous_batch_count']}/{args.iters}   "
          f"B {B['homogeneous_batch_count']}/{args.iters}; A three-bin frac "
          f"{A['three_bin_batch_frac']:.5f}")
    print(f"  observed KS            {ks_obs:.6f} (noise-dominated, see ks_note)")
    n_ok = sum(c["pass"] for c in stat + det)
    n_repro = sum(1 for c in det if c["repro"])
    print(f"SELF-CHECK: {'PASS' if not failures else 'FAIL'} ({n_ok}/{len(stat) + len(det)} "
          f"checks: {len(stat)} statistical incl. small-batch validation + {len(det)} deterministic, "
          f"of which {n_repro} reproducibility checks {'enforced' if strict_repro else 'reported only'})")
    for c in stat + det:
        if not c["pass"]:
            print("  FAIL:", c)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
