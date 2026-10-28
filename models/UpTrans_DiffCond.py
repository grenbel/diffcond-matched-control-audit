#! /usr/bin/python3
# -*- coding: utf-8 -*-
################################################################
# UpTrans_DiffCond -- difficulty-conditioned extension of UpTrans (ODGNet AAAI 2024)
#
# Cross-backbone validation accompanying the paper. Preserves UpTrans vanilla
# reconstruction forward path bit-exactly at t=0 (after warm-start from
# pretrained UpTrans ckpt + zero-init of DiffCond output layers).
#
# Key design points (main paper Sec. 5.5 describes the role-level analogue):
#   - Injection 1: `DifficultyAwareFeaRefine` adds a zero-init key delta
#     to the Dictionary attention keys. Analog to AdaPoinTr_DiffCond's
#     per-query ranking delta; mechanism differs (soft attention vs hard
#     argsort+gather), see main paper Sec. 5.5.
#   - Injection 2: `DiffAwareUNet` applies a FiLM (gamma, beta both
#     zero-init) on `u0_fea` after `ps_0 + mlp_1`. Analog to
#     AdaPoinTr_DiffCond's gated decoder injection; also mechanism-level
#     different (FiLM is feature-independent; gated injection is
#     feature-dependent).
#   - Injection 3: `difficulty_predictor` MLP on global feat + CE aux
#     loss (lambda_ce = 0.1, literal match to AdaPoinTr_DiffCond form; CE is
#     sub-dominant vs UpTrans's x1e3-scaled loss_all).
#   - Shared: `DifficultyEmbedding(3, 256)`, random-init (NOT zero-init).
#
# This released model is oracle-only at inference. `difficulty='predict'` string sentinel
# (as used by AdaPoinTr_DiffCond) is NOT supported here -- it would require
# splitting the encoder to break predictor-feeds-encoder cyclic dep (main
# paper Sec. 5.5 reports Oracle inference only for UpTrans).
################################################################

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .build import MODELS
from .UpTrans import UpTrans, UNet, PostProcess
from .pointnet import PointNet_SA_Module_KNN, MLP_Res, Transformer as UpTransTransformer


# -------------------------------------------------------------------
# Shared constants (structurally identical to AdaPoinTr_DiffCond's DIFF_LEVELS;
# kept local here so runner's `from models.AdaPoinTr_DiffCond import DIFF_LEVELS`
# continues to work while UpTrans_DiffCond users can also read from here).
# -------------------------------------------------------------------
DIFF_LEVELS = {'easy': 0, 'median': 1, 'hard': 2}
NUM_DIFF_LEVELS = 3


# ===================================================================
# DifficultyEmbedding (shared primitive)
# ===================================================================
class DifficultyEmbedding(nn.Module):
    def __init__(self, num_levels=NUM_DIFF_LEVELS, embed_dim=256):
        super().__init__()
        self.embed = nn.Embedding(num_levels, embed_dim)

    def forward(self, difficulty):
        """difficulty: (B,) long tensor -> (B, embed_dim)."""
        return self.embed(difficulty)


# ===================================================================
# DifficultyAwareFeaRefine  (Injection 1: Dictionary key delta)
# Drop-in replacement for UpTrans.FeaRefine that injects a
# difficulty-conditioned bias into the attention keys.
#
# At t=0 (after zero-init of diff_key_delta's output layer),
# K_delta == 0  =>  K = K_base  =>  scores = vanilla scores
# =>  refine output bit-exact equals vanilla FeaRefine output.
# ===================================================================
class DifficultyAwareFeaRefine(nn.Module):
    def __init__(self, dim=512, hidd_dim=64, num_dicts=128, diff_embed_dim=256):
        super().__init__()
        self.num_dicts = num_dicts
        self.hidd_dim = hidd_dim

        # Vanilla FeaRefine params (structurally identical; weights may load
        # from pretrained UpTrans FeaRefine via the builder.load_model patch).
        self.dict_embed = nn.Embedding(dim, num_dicts)
        self.query = nn.Conv1d(dim, hidd_dim, 1)
        self.key = nn.Conv1d(dim, hidd_dim, 1)

        # NEW: difficulty-conditioned key delta. Output reshaped to
        # (B, hidd_dim, num_dicts) so it can be added to K_base directly.
        self.diff_key_delta = nn.Sequential(
            nn.Linear(diff_embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, hidd_dim * num_dicts),
        )
        # Zero-init the final layer so K_delta == 0 at t=0.
        # (diff_embed + MLP body remain random-init; only the output layer
        # is zeroed. This is the minimum needed for bit-exact reconstruction.)
        nn.init.zeros_(self.diff_key_delta[-1].weight)
        nn.init.zeros_(self.diff_key_delta[-1].bias)

    def forward(self, feat, diff_emb):
        """
        Args:
            feat:     (B, C, N)  -- per-seed features to refine
            diff_emb: (B, diff_embed_dim) -- difficulty embedding
        Returns:
            refined:  (B, C, N)
            cons:     (num_dicts, num_dicts) -- orthogonality constraint matrix
        """
        B, C, N = feat.size()
        dict_fea_w = self.dict_embed.weight  # (C, num_dicts)
        # Normalised feature_memory -> cons matrix for orth loss
        feature_memory = F.normalize(dict_fea_w, dim=0)
        cons = feature_memory.transpose(0, 1) @ feature_memory  # (num_dicts, num_dicts)

        dict_fea = dict_fea_w.unsqueeze(0).repeat(B, 1, 1)       # (B, C, num_dicts)

        q = self.query(feat)                                      # (B, hidd, N)
        k_base = self.key(dict_fea)                               # (B, hidd, num_dicts)
        # NEW: difficulty-conditioned key bias. At t=0 this is exactly 0.
        k_delta = self.diff_key_delta(diff_emb).reshape(
            B, self.hidd_dim, self.num_dicts
        )                                                         # (B, hidd, num_dicts)
        k = k_base + k_delta

        d_k = k.size(1)
        scores = torch.matmul(q.transpose(-2, -1), k) / math.sqrt(d_k)   # (B, N, num_dicts)
        scores = torch.softmax(scores, dim=-1)
        output = torch.matmul(scores, dict_fea.transpose(-2, -1))         # (B, N, C)

        return (output.transpose(-2, -1) + feat) / 2, cons


# ===================================================================
# DiffAwareUNet  (Injection 2: u0_fea FiLM + replaces both FeaRefine
#                 with DifficultyAware versions)
# Inherits from UNet so all vanilla submodules (sa_module_1..4,
# transformer_1/2, ps_0/1/2, mlp_1/3) are created by super().__init__()
# and can be loaded bit-exactly from a pretrained UpTrans ckpt.
# Only refine_1 / refine_2 are swapped (same param-name prefix so they
# still load the pretrained dict_embed/query/key weights) and FiLM
# modules are added fresh.
# ===================================================================
class DiffAwareUNet(UNet):
    def __init__(self, dim_feat=384, num_seeds=1024, seed_fea=128,
                 num_dicts=256, diff_embed_dim=256):
        super().__init__(dim_feat, num_seeds, seed_fea, num_dicts)
        # Replace vanilla FeaRefine instances set up by UNet.__init__ with
        # their difficulty-aware variants. The base-param names
        # (dict_embed / query / key) match, so a strict=False load from a
        # pretrained UpTrans ckpt will populate them correctly; only
        # diff_key_delta stays at init (its output layer is zeroed, so
        # the refine output matches vanilla bit-exactly at t=0).
        self.refine_1 = DifficultyAwareFeaRefine(
            dim=256, hidd_dim=64, num_dicts=num_dicts,
            diff_embed_dim=diff_embed_dim,
        )
        self.refine_2 = DifficultyAwareFeaRefine(
            dim=128, hidd_dim=64, num_dicts=num_dicts // 2,
            diff_embed_dim=diff_embed_dim,
        )

        # NEW: FiLM (Injection 2) applied to u0_fea after ps_0 + mlp_1.
        # u0_fea shape is (B, 256, num_seeds//16) -- for num_seeds=1024 that is
        # (B, 256, 64). gamma & beta output dim matches u0_fea's channel dim
        # (256) and broadcasts over the seed-unit axis.
        self.film_gamma = nn.Sequential(
            nn.Linear(diff_embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )
        self.film_beta = nn.Sequential(
            nn.Linear(diff_embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )
        # Zero-init output layers so at t=0: gamma=0, beta=0 => u0_fea unchanged.
        for m in [self.film_gamma[-1], self.film_beta[-1]]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, point_cloud, diff_emb):
        """
        Args:
            point_cloud: (B, 3, N_in) -- partial input
            diff_emb:    (B, diff_embed_dim) -- difficulty embedding
        Returns:
            (l4_fea, u2_xyz, u2_fea, cons1, cons2)  -- same tuple shape as vanilla UNet
        """
        # --- Encoder (unchanged; no encoder-side conditioning) ---
        l0_xyz = point_cloud
        l0_fea = point_cloud
        l1_xyz, l1_fea, idx1 = self.sa_module_1(l0_xyz, l0_fea)
        l1_fea = self.transformer_1(l1_fea, l1_xyz)
        l2_xyz, l2_fea, idx2 = self.sa_module_2(l1_xyz, l1_fea)
        l2_fea = self.transformer_2(l2_fea, l2_xyz)
        l4_xyz, l4_fea = self.sa_module_4(l2_xyz, l2_fea)

        # --- Decoder: seed-feature generation with FiLM on u0_fea ---
        u0_fea = self.ps_0(l4_fea)                                  # (B, 256, num_seeds//16)
        u0_fea = self.mlp_1(
            torch.cat([u0_fea, l4_fea.repeat((1, 1, u0_fea.size(2)))], 1)
        )

        # Injection 2: FiLM. At t=0, gamma == beta == 0 => identity.
        gamma = self.film_gamma(diff_emb).unsqueeze(-1)             # (B, 256, 1)
        beta = self.film_beta(diff_emb).unsqueeze(-1)               # (B, 256, 1)
        u0_fea = (1 + gamma) * u0_fea + beta                         # broadcast over seed units

        # --- Remaining decoder; refine_1 / refine_2 now take diff_emb ---
        u1_fea = self.ps_1(u0_fea)
        u1_fea, cons1 = self.refine_1(u1_fea, diff_emb)              # Injection 1a
        u1_fea = torch.cat([l2_fea, u1_fea], dim=2)

        u2_fea = self.ps_2(u1_fea)
        u2_fea, cons2 = self.refine_2(u2_fea, diff_emb)              # Injection 1b
        u2_xyz = self.mlp_3(u2_fea)

        u2_xyz = torch.cat([l1_xyz, u2_xyz], dim=2)
        u2_fea = torch.cat([l1_fea, u2_fea], dim=2)

        return l4_fea, u2_xyz, u2_fea, cons1, cons2


# ===================================================================
# UpTrans_DiffCond  (main registered model class)
# Inherits from UpTrans; replaces feat_extractor with DiffAwareUNet and
# adds difficulty_embedding + difficulty_predictor (Injection 3). The
# decoder (PostProcess) + loss_func are inherited unchanged.
# ===================================================================
@MODELS.register_module()
class UpTrans_DiffCond(UpTrans):
    def __init__(self, config, **kwargs):
        # Parent __init__ creates vanilla feat_extractor (UNet), decoder
        # (PostProcess), loss_func. We replace feat_extractor below; the
        # old vanilla UNet becomes garbage immediately.
        super().__init__(config, **kwargs)

        diff_embed_dim = getattr(config, 'diff_embed_dim', 256)

        # Replace the vanilla UNet with the DiffCond variant.
        self.feat_extractor = DiffAwareUNet(
            dim_feat=config.dim_feat,
            num_seeds=config.num_seeds,
            seed_fea=config.seed_fea,
            num_dicts=config.num_dicts,
            diff_embed_dim=diff_embed_dim,
        )

        # Shared difficulty embedding (Injection shared primitive).
        self.difficulty_embedding = DifficultyEmbedding(
            num_levels=NUM_DIFF_LEVELS, embed_dim=diff_embed_dim,
        )

        # Injection 3: difficulty predictor -- MLP on global feature (l4_fea),
        # outputs logits over NUM_DIFF_LEVELS. Trained with CE aux loss.
        # NOT zero-initialised (random-init is intentional -- CE term is part
        # of the DiffCond design, not an initialization-free control).
        self.difficulty_predictor = nn.Sequential(
            nn.Linear(config.dim_feat, 128),
            nn.GELU(),
            nn.Linear(128, NUM_DIFF_LEVELS),
        )

        # CE weight (matches AdaPoinTr_DiffCond default).
        self.lambda_ce = getattr(config, 'lambda_ce', 0.1)

    def forward(self, partial_point_cloud, difficulty=None):
        """Oracle-only forward. the released model -- predict-mode NOT supported.

        Args:
            partial_point_cloud: (B, N_in, 3)
            difficulty: (B,) long tensor of difficulty level indices
                        (0=easy, 1=median, 2=hard). If None, defaults to
                        median (used by runner's default-validation path
                        as a diagnostic-only fallback).
                        The string sentinel 'predict' used by
                        AdaPoinTr_DiffCond is NOT supported here -- raises
                        NotImplementedError. Rationale: the predictor
                        takes l4_fea (produced inside feat_extractor) so
                        it cannot run before diff_emb is needed by
                        feat_extractor itself. Splitting the encoder to
                        fix this is left to future work.
        Returns:
            7-tuple (pcd, p3, p1, p2, cons1, cons2, diff_logits) -- same
            coarse/fine/orth structure as vanilla UpTrans plus the
            predictor's logits for CE aux loss.
        """
        if difficulty is None:
            B = partial_point_cloud.size(0)
            difficulty = torch.full(
                (B,), DIFF_LEVELS['median'],
                device=partial_point_cloud.device, dtype=torch.long,
            )
        elif isinstance(difficulty, str):
            raise NotImplementedError(
                "UpTrans_DiffCond does not support difficulty='predict' "
                "(oracle-only at inference). Pass a (B,) long tensor of "
                "level indices instead (main paper Sec. 5.5 reports Oracle inference only)."
            )

        diff_emb = self.difficulty_embedding(difficulty)               # (B, diff_embed_dim)
        in_pcd = partial_point_cloud.permute(0, 2, 1).contiguous()     # (B, 3, N_in)

        partial_shape_code, xyz, fea, cons1, cons2 = self.feat_extractor(in_pcd, diff_emb)
        pcd, p1, p2, p3 = self.decoder(xyz, fea)

        # Predictor takes the encoder-output global feature. Non-cyclic
        # because we use the oracle diff_emb (not predictor output) upstream.
        diff_logits = self.difficulty_predictor(partial_shape_code.squeeze(-1))   # (B, 3)

        return pcd, p3, p1, p2, cons1, cons2, diff_logits

    def get_loss(self, ret, gt, epoch=None,
                 diff_label=None, difficulty_labels=None):
        """
        Accepts both `diff_label` (native name used by runner's UpTrans
        branch) and `difficulty_labels` (name used by AdaPoinTr_DiffCond's
        signature) so the function works under either calling convention.
        `epoch` is accepted for signature parity with AdaPoinTr_DiffCond's
        get_loss(ret, gt, epoch, ...); UpTrans has no schedule-dependent
        loss terms so it is ignored.
        """
        Pc, P3, P1, P2, cons1, cons2, diff_logits = ret
        if diff_label is None:
            diff_label = difficulty_labels

        # Delegate reconstruction loss (4-CD sum x 1e3) to parent UpTrans.
        base_ret = (Pc, P3, P1, P2, cons1, cons2)
        loss_all, cd0, cd3 = super().get_loss(base_ret, gt)

        # CE aux loss (Injection 3). Literal form: lambda_ce * CE (no x1e3).
        # Matches AdaPoinTr_DiffCond.py:373 convention. UpTrans's loss_all is
        # already x1e3-scaled by super().get_loss(); with lambda_ce=0.1 and
        # typical CE~0.5, CE contributes ~0.05 to a ~1000-scale loss_all --
        # sub-dominant but still informative.
        if diff_label is not None:
            loss_ce = F.cross_entropy(diff_logits, diff_label)
            loss_all = loss_all + self.lambda_ce * loss_ce

        return loss_all, cd0, cd3

    def get_constrain(self, ret):
        """Orth-constraint loss is unchanged -- just slice off diff_logits
        before delegating to parent UpTrans.get_constrain which expects the
        6-tuple return shape."""
        base_ret = ret[:6]
        return super().get_constrain(base_ret)
