##############################################################
# AdaPoinTr with Difficulty Conditioning (DiffCond)
#
# Design notes:
#   - Per-query delta replaces constant bias in the ranking residual
#   - Predictor: auxiliary head for predict-eval path (oracle-trained,
#     used at predict-time inference)
#   - difficulty='predict' path fully implemented
#   - Zero-init guarantees verified
#   - Compatible with builder.load_model strict=False + remap
##############################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from extensions.chamfer_dist import ChamferDistanceL1
from .build import MODELS
from models.Transformer_utils import *
from utils import misc

from .AdaPoinTr import (
    PointTransformerEncoderEntry,
    PointTransformerDecoderEntry,
    DGCNN_Grouper,
    SimpleEncoder,
    Fold,
    SimpleRebuildFCLayer,
)

# ---------------------------------------------------------------------------
DIFF_LEVELS = {'easy': 0, 'median': 1, 'hard': 2}
NUM_DIFF_LEVELS = 3


# ===================================================================
# DifficultyEmbedding
# ===================================================================
class DifficultyEmbedding(nn.Module):
    def __init__(self, num_levels=NUM_DIFF_LEVELS, embed_dim=256):
        super().__init__()
        self.embed = nn.Embedding(num_levels, embed_dim)

    def forward(self, difficulty):
        """difficulty: (B,) int -> (B, 1, C)"""
        return self.embed(difficulty).unsqueeze(1)


# ===================================================================
# DifficultyAwareQueryRanking -- C1 fixed (per-query delta)
# ===================================================================
class DifficultyAwareQueryRanking(nn.Module):
    """score = sigmoid(base_scorer(xyz) + delta(xyz, diff_emb))

    base_scorer: same arch as original query_ranking (loads pretrained weights)
    delta: per-query residual conditioned on (xyz, diff_emb), zero-init -> baseline at init
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        # Same arch as original query_ranking (without Sigmoid)
        self.base_scorer = nn.Sequential(
            nn.Linear(3, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, 1),
        )
        # Per-query residual: depends on BOTH xyz and diff_emb -> changes ranking
        self.diff_delta = nn.Sequential(
            nn.Linear(3 + embed_dim, 128), nn.GELU(),
            nn.Linear(128, 1),  # zero-init last layer
        )

    def forward(self, coarse_xyz, diff_emb):
        """
        coarse_xyz: (B, N, 3)
        diff_emb:   (B, 1, C) -- broadcast to (B, N, C)
        Returns:    (B, N, 1) scores in [0,1]
        """
        base = self.base_scorer(coarse_xyz)                              # (B, N, 1)
        diff_broadcast = diff_emb.expand(-1, coarse_xyz.shape[1], -1)    # (B, N, C)
        delta = self.diff_delta(
            torch.cat([coarse_xyz, diff_broadcast], dim=-1))             # (B, N, 1)
        return torch.sigmoid(base + delta)


# ===================================================================
# DiffCondPCTransformer
# ===================================================================
class DiffCondPCTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        encoder_config = config.encoder_config
        decoder_config = config.decoder_config
        self.center_num = getattr(config, 'center_num', [512, 128])
        self.encoder_type = config.encoder_type
        assert self.encoder_type in ['graph', 'pn']

        in_chans = 3
        self.num_query = query_num = config.num_query
        global_feature_dim = config.global_feature_dim

        # v2 ablation controls (2026-04-16)
        # use_vanilla_ranking=True -> skip diff_cond_query_ranking, use vanilla query_ranking
        #   (fixes argsort gradient block identified in v1 static audit: diff_delta dead)
        # ce_lambda is read by the outer AdaPoinTr_DiffCond.get_loss(), not here.
        self.use_vanilla_ranking = getattr(config, 'use_vanilla_ranking', False)

        # ---- Encoder (identical to AdaPoinTr) ----
        if self.encoder_type == 'graph':
            self.grouper = DGCNN_Grouper(k=16)
        else:
            self.grouper = SimpleEncoder(k=32, embed_dims=512)
        self.pos_embed = nn.Sequential(
            nn.Linear(in_chans, 128), nn.GELU(),
            nn.Linear(128, encoder_config.embed_dim),
        )
        self.input_proj = nn.Sequential(
            nn.Linear(self.grouper.num_features, 512), nn.GELU(),
            nn.Linear(512, encoder_config.embed_dim),
        )
        self.encoder = PointTransformerEncoderEntry(encoder_config)
        self.increase_dim = nn.Sequential(
            nn.Linear(encoder_config.embed_dim, 1024), nn.GELU(),
            nn.Linear(1024, global_feature_dim),
        )

        # ---- Query generation (identical to AdaPoinTr) ----
        self.coarse_pred = nn.Sequential(
            nn.Linear(global_feature_dim, 1024), nn.GELU(),
            nn.Linear(1024, 3 * query_num),
        )
        self.mlp_query = nn.Sequential(
            nn.Linear(global_feature_dim + 3, 1024), nn.GELU(),
            nn.Linear(1024, 1024), nn.GELU(),
            nn.Linear(1024, decoder_config.embed_dim),
        )
        if decoder_config.embed_dim == encoder_config.embed_dim:
            self.mem_link = nn.Identity()
        else:
            self.mem_link = nn.Linear(encoder_config.embed_dim, decoder_config.embed_dim)

        # ---- Decoder (identical to AdaPoinTr) ----
        self.decoder = PointTransformerDecoderEntry(decoder_config)

        # ---- Original query_ranking (kept for weight loading compatibility) ----
        self.query_ranking = nn.Sequential(
            nn.Linear(3, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, 1), nn.Sigmoid(),
        )

        # ===========================================================
        #  NEW: difficulty conditioning (diff_cond_ prefix)
        # ===========================================================
        dec_dim = decoder_config.embed_dim

        # (a) Difficulty embedding
        self.diff_cond_embedding = DifficultyEmbedding(NUM_DIFF_LEVELS, dec_dim)

        # (b) Difficulty-aware query ranking (C1 fixed)
        self.diff_cond_query_ranking = DifficultyAwareQueryRanking(dec_dim)

        # (c) Gated injection into decoder queries
        self.diff_cond_gate = nn.Sequential(
            nn.Linear(dec_dim * 2, dec_dim), nn.Sigmoid(),
        )
        self.diff_cond_proj = nn.Linear(dec_dim, dec_dim)

        # (d) Difficulty predictor (auxiliary head)
        self.diff_predictor = nn.Sequential(
            nn.Linear(global_feature_dim, 128), nn.GELU(),
            nn.Linear(128, NUM_DIFF_LEVELS),
        )

        self.apply(self._init_weights)

        # Zero-init new conditioning paths AFTER _init_weights
        nn.init.zeros_(self.diff_cond_proj.weight)
        nn.init.zeros_(self.diff_cond_proj.bias)
        nn.init.zeros_(self.diff_cond_gate[0].bias)
        nn.init.zeros_(self.diff_cond_query_ranking.diff_delta[-1].weight)
        nn.init.zeros_(self.diff_cond_query_ranking.diff_delta[-1].bias)
        nn.init.normal_(self.diff_cond_embedding.embed.weight, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _resolve_difficulty(self, bs, device, global_feature, difficulty):
        """Resolve difficulty input to (B, 1, C) embedding + (B,3) pred logits.

        difficulty: None -> median(1)
                    int  -> broadcast to all samples
                    Tensor(B,) -> per-sample
                    'predict' -> use predictor head
        Returns: (diff_emb, diff_pred_logits)
            diff_emb: (B, 1, C)
            diff_pred_logits: (B, 3) or None
        """
        diff_pred_logits = None

        if difficulty == 'predict':
            # Predict from encoder features
            diff_pred_logits = self.diff_predictor(global_feature)  # (B, 3)
            diff_idx = diff_pred_logits.argmax(dim=-1)             # (B,)
            diff_emb = self.diff_cond_embedding(diff_idx)
            return diff_emb, diff_pred_logits

        # Oracle / default path
        if difficulty is None:
            diff_idx = torch.ones(bs, dtype=torch.long, device=device)  # median
        elif isinstance(difficulty, int):
            diff_idx = torch.full((bs,), difficulty, dtype=torch.long, device=device)
        else:
            diff_idx = difficulty.long().to(device)

        diff_emb = self.diff_cond_embedding(diff_idx)

        # Always compute predictor logits during training (for CE loss)
        if self.training:
            diff_pred_logits = self.diff_predictor(global_feature)

        return diff_emb, diff_pred_logits

    def forward(self, xyz, difficulty=None):
        """
        Args:
            xyz:        (B, N, 3) partial point cloud
            difficulty: None | int | Tensor(B,) | 'predict'
        Returns:
            training: (q, coarse, denoise_length, diff_pred_logits)
            eval:     (q, coarse, 0, diff_pred_logits_or_None)
        """
        bs = xyz.size(0)

        # --- Encoder (unchanged) ---
        coor, f = self.grouper(xyz, self.center_num)
        pe = self.pos_embed(coor)
        x = self.input_proj(f)
        x = self.encoder(x + pe, coor)

        global_feature = self.increase_dim(x)
        global_feature = torch.max(global_feature, dim=1)[0]  # (B, G)

        coarse = self.coarse_pred(global_feature).reshape(bs, -1, 3)
        coarse_inp = misc.fps(xyz, self.num_query // 2)
        coarse = torch.cat([coarse, coarse_inp], dim=1)

        mem = self.mem_link(x)

        # --- Difficulty embedding ---
        diff_emb, diff_pred_logits = self._resolve_difficulty(
            bs, xyz.device, global_feature, difficulty)

        # --- Query selection ---
        if self.use_vanilla_ranking:
            # v2: use vanilla query_ranking (loaded from the published-baseline warm-start).
            # No difficulty conditioning in the selection path -- avoids the
            # argsort/gather gradient-block that killed diff_delta in v1.
            query_ranking = self.query_ranking(coarse)
        else:
            # v1: difficulty-conditioned ranking (per-query delta on base_scorer).
            query_ranking = self.diff_cond_query_ranking(coarse, diff_emb)
        idx = torch.argsort(query_ranking, dim=1, descending=True)
        coarse = torch.gather(
            coarse, 1,
            idx[:, :self.num_query].expand(-1, -1, coarse.size(-1)))

        if self.training:
            picked_points = misc.fps(xyz, 64)
            picked_points = misc.jitter_points(picked_points)
            coarse = torch.cat([coarse, picked_points], dim=1)
            denoise_length = 64

            q = self.mlp_query(torch.cat([
                global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1),
                coarse], dim=-1))
            q = self._inject_difficulty(q, diff_emb)
            q = self.decoder(q=q, v=mem, q_pos=coarse, v_pos=coor,
                             denoise_length=denoise_length)
            return q, coarse, denoise_length, diff_pred_logits
        else:
            q = self.mlp_query(torch.cat([
                global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1),
                coarse], dim=-1))
            q = self._inject_difficulty(q, diff_emb)
            q = self.decoder(q=q, v=mem, q_pos=coarse, v_pos=coor)
            return q, coarse, 0, diff_pred_logits

    def _inject_difficulty(self, q, diff_emb):
        """q' = q + gate * proj(diff_emb)  -- zero-init proj -> q'=q at init"""
        diff_broadcast = diff_emb.expand_as(q)
        gate = self.diff_cond_gate(torch.cat([q, diff_broadcast], dim=-1))
        proj = self.diff_cond_proj(diff_broadcast)
        return q + gate * proj


# ===================================================================
# AdaPoinTr_DiffCond -- top-level model
# ===================================================================
@MODELS.register_module()
class AdaPoinTr_DiffCond(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.trans_dim = config.decoder_config.embed_dim
        self.num_query = config.num_query
        self.num_points = getattr(config, 'num_points', None)
        self.decoder_type = config.decoder_type
        assert self.decoder_type in ['fold', 'fc']

        self.fold_step = 8
        self.ce_lambda = getattr(config, 'ce_lambda', 0.1)  # v2 ablation: 0 to disable CE
        self.base_model = DiffCondPCTransformer(config)

        if self.decoder_type == 'fold':
            self.factor = self.fold_step ** 2
            self.decode_head = Fold(self.trans_dim, step=self.fold_step, hidden_dim=256)
        else:
            if self.num_points is not None:
                self.factor = self.num_points // self.num_query
                assert self.num_points % self.num_query == 0
                self.decode_head = SimpleRebuildFCLayer(
                    self.trans_dim * 2, step=self.num_points // self.num_query)
            else:
                self.factor = self.fold_step ** 2
                self.decode_head = SimpleRebuildFCLayer(
                    self.trans_dim * 2, step=self.fold_step ** 2)

        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1),
        )
        self.reduce_map = nn.Linear(self.trans_dim + 1027, self.trans_dim)
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func = ChamferDistanceL1()

    def get_loss(self, ret, gt, epoch=1, difficulty_labels=None):
        """Loss with optional difficulty prediction CE.

        ret: (pred_coarse, denoised_coarse, denoised_fine, pred_fine, diff_pred_logits)
        """
        pred_coarse, denoised_coarse, denoised_fine, pred_fine = ret[:4]
        diff_pred_logits = ret[4] if len(ret) > 4 else None

        assert pred_fine.size(1) == gt.size(1)

        # Denoise loss
        idx = knn_point(self.factor, gt, denoised_coarse)
        denoised_target = index_points(gt, idx)
        denoised_target = denoised_target.reshape(gt.size(0), -1, 3)
        assert denoised_target.size(1) == denoised_fine.size(1)
        loss_denoised = self.loss_func(denoised_fine, denoised_target) * 0.5

        # Recon loss
        loss_coarse = self.loss_func(pred_coarse, gt)
        loss_fine = self.loss_func(pred_fine, gt)
        loss_recon = loss_coarse + loss_fine

        # Difficulty prediction CE loss (auxiliary, lambda configurable via ce_lambda)
        # NOTE: this is added to loss_recon, so runner's "DenseLoss" includes CE.
        # When comparing train curves with baseline, account for this offset.
        # v2 ablation: ce_lambda=0 disables CE entirely (predictor still exists
        # but gets no gradient signal; oracle-only inference).
        if diff_pred_logits is not None and difficulty_labels is not None and self.ce_lambda > 0:
            loss_diff = F.cross_entropy(diff_pred_logits, difficulty_labels) * self.ce_lambda
            loss_recon = loss_recon + loss_diff

        return loss_denoised, loss_recon

    def forward(self, xyz, difficulty=None):
        q, coarse_point_cloud, denoise_length, diff_pred_logits = \
            self.base_model(xyz, difficulty=difficulty)

        B, M, C = q.shape

        global_feature = self.increase_dim(
            q.transpose(1, 2)).transpose(1, 2)
        global_feature = torch.max(global_feature, dim=1)[0]

        rebuild_feature = torch.cat([
            global_feature.unsqueeze(-2).expand(-1, M, -1),
            q, coarse_point_cloud], dim=-1)

        if self.decoder_type == 'fold':
            rebuild_feature = self.reduce_map(rebuild_feature.reshape(B * M, -1))
            relative_xyz = self.decode_head(rebuild_feature).reshape(B, M, 3, -1)
            rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-1)).transpose(2, 3)
        else:
            rebuild_feature = self.reduce_map(rebuild_feature)
            relative_xyz = self.decode_head(rebuild_feature)
            rebuild_points = relative_xyz + coarse_point_cloud.unsqueeze(-2)

        if self.training:
            pred_fine = rebuild_points[:, :-denoise_length].reshape(B, -1, 3).contiguous()
            pred_coarse = coarse_point_cloud[:, :-denoise_length].contiguous()
            denoised_fine = rebuild_points[:, -denoise_length:].reshape(B, -1, 3).contiguous()
            denoised_coarse = coarse_point_cloud[:, -denoise_length:].contiguous()

            assert pred_fine.size(1) == self.num_query * self.factor
            assert pred_coarse.size(1) == self.num_query

            return (pred_coarse, denoised_coarse, denoised_fine, pred_fine, diff_pred_logits)
        else:
            assert denoise_length == 0
            rebuild_points = rebuild_points.reshape(B, -1, 3).contiguous()
            assert rebuild_points.size(1) == self.num_query * self.factor
            assert coarse_point_cloud.size(1) == self.num_query

            # NOTE: eval return must be (coarse, dense) to match AdaPoinTr interface.
            # diff_pred_logits stored as attribute for optional access.
            self._last_diff_pred_logits = diff_pred_logits
            return (coarse_point_cloud, rebuild_points)
