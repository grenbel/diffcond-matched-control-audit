import torch
import torch.nn as nn
import os
import json
import random as _random
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2

# Protocol constants -- shared by aligned training, aligned validation, and official test
CROP_RATIO = {'easy': 1/4, 'median': 1/2, 'hard': 3/4}
VIEWPOINT_CHOICES = [
    torch.Tensor([1,1,1]),   torch.Tensor([1,1,-1]),
    torch.Tensor([1,-1,1]),  torch.Tensor([-1,1,1]),
    torch.Tensor([-1,-1,1]), torch.Tensor([-1,1,-1]),
    torch.Tensor([1,-1,-1]), torch.Tensor([-1,-1,-1]),
]

# Cross-backbone constants: DiffCond model family + UpTrans family.
# Shared by run_net (training), validate (aligned + default), test_net.
# Cross-backbone setup: main paper Sec. 5.5; run commands: REPRO.md.
DIFFCOND_NAMES = {'AdaPoinTr_DiffCond', 'UpTrans_DiffCond'}
UPTRANS_NAMES  = {'UpTrans',            'UpTrans_DiffCond'}

def pick_outputs(ret, model_name):
    """Return (coarse_points, dense_points) for both model families.

    UpTrans family:     ret = (Pc, P3, P1, P2, cons1, cons2[, diff_logits])
                        dense output is P3 at ret[1] (final upsample).
    AdaPoinTr family:   ret = (..., dense_last) -- dense is the last element.
    """
    if model_name in UPTRANS_NAMES:
        return ret[0], ret[1]
    return ret[0], ret[-1]


def pcn_proxy_valid(data):
    """True iff data[2] is a tensor with all entries >= 0 (PCN proxy loaded)."""
    return (len(data) > 2
            and isinstance(data[2], torch.Tensor)
            and bool((data[2] >= 0).all()))


def assert_pcn_diffcond_proxy(config, dataset_name, data):
    """Guard: fail-hard for DiffCond on PCN when the proxy
    difficulty is missing or invalid (i.e., any -1 sentinel in the batch).
    Without this, missing-proxy silently degenerates the experiment to the
    rejected median-constant regime (option (b) of the error message below),
    contaminating the reported cross-dataset DiffCond claim.

    Override: set `allow_pcn_median_fallback: true` in the yaml config to
    opt into median-constant fallback (useful only for deliberate ablations,
    NEVER for paper cross-backbone runs).
    """
    if (config.model.NAME in DIFFCOND_NAMES
            and dataset_name == 'PCN'
            and not pcn_proxy_valid(data)
            and not getattr(config, 'allow_pcn_median_fallback', False)):
        raise RuntimeError(
            f"PCN proxy difficulty missing/invalid for {config.model.NAME} on "
            f"PCN. Either (a) run tools/compute_pcn_proxy_difficulty.py and "
            f"re-train (recommended; the reported PCN runs use this proxy), or "
            f"(b) explicitly set `allow_pcn_median_fallback: true` in the "
            f"yaml config to opt into the rejected median-constant regime."
        )

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # from IPython import embed; embed()
    
    # parameter setting
    start_epoch = 0
    best_metrics = None
    metrics = None

    # resume ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    # print model info
    print_log('Trainable_parameters:', logger = logger)
    print_log('=' * 25, logger = logger)
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger = logger)
    
    print_log('Untrainable_parameters:', logger = logger)
    print_log('=' * 25, logger = logger)
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer = builder.build_optimizer(base_model, config)
    
    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()


    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)
    scheduler = builder.build_scheduler(base_model, optimizer, config, last_epoch=start_epoch-1)

    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['SparseLoss', 'DenseLoss'])

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train._base_.N_POINTS
            dataset_name = config.dataset.train._base_.NAME
            # Initialize diff_tensor before any dataset branch so non-ShapeNet paths
            # (PCN / Completion3D / Projected_ShapeNet) don't trigger UnboundLocalError
            # at the unconditional read below.
            diff_tensor = None
            if dataset_name == 'PCN' or dataset_name == 'Completion3D' or dataset_name == 'Projected_ShapeNet':
                partial = data[0].cuda()
                gt = data[1].cuda()
                if config.dataset.train._base_.CARS:
                    if idx == 0:
                        print_log('padding while KITTI training', logger=logger)
                    partial = misc.random_dropping(partial, epoch) # specially for KITTI finetune
                # Stage-2.5: pick up proxy difficulty for PCN DiffCond (option (a), the
                # nearest-neighbour proxy). Dataset returns data[2] as a per-sample
                # diff_label tensor (or int -1 sentinel when no proxy is available
                # for that sample). Validation guard: fail-hard here if proxy
                # is missing for DiffCond+PCN (unless yaml explicitly opts in
                # to median fallback via `allow_pcn_median_fallback: true`).
                assert_pcn_diffcond_proxy(config, dataset_name, data)
                if (dataset_name == 'PCN'
                        and config.model.NAME in DIFFCOND_NAMES
                        and pcn_proxy_valid(data)):
                    diff_tensor = data[2].to(gt.device).long()

            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                is_diffcond_model = (config.model.NAME in DIFFCOND_NAMES)
                # matched-vanilla control: vanilla model + DiffCond crop path
                use_diffcond_crop = is_diffcond_model or (
                    getattr(config, 'training_crop_mode', '') == 'diffcond')
                # diff_tensor pre-initialized outside dataset branches (see above)

                if use_diffcond_crop:
                    # Crop ratio range: default [0.25, 0.75] (uniform, same marginal as original).
                    # Override with training_crop_range in config for stratified specialist ft:
                    #   [0.625, 0.75] = hard specialist
                    #   [0.375, 0.625] = median specialist
                    #   [0.25, 0.375] = easy specialist
                    crop_lo, crop_hi = getattr(config, 'training_crop_range', [0.25, 0.75])
                    crop_ratio_val = _random.uniform(crop_lo, crop_hi)
                    num_crop = int(npoints * crop_ratio_val)
                    partial, _ = misc.seprate_point_cloud(gt, npoints, num_crop, fixed_points=None)
                    partial = misc.fps(partial, 2048)
                    partial = partial.cuda()
                    # Only DiffCond model uses the difficulty label.
                    # Vanilla model with training_crop_mode=diffcond (matched-vanilla) leaves diff_tensor=None.
                    if is_diffcond_model:
                        if crop_ratio_val < 0.375:
                            diff_level = 0  # easy
                        elif crop_ratio_val < 0.625:
                            diff_level = 1  # median
                        else:
                            diff_level = 2  # hard
                        diff_tensor = torch.full((gt.shape[0],), diff_level,
                                                 dtype=torch.long, device=gt.device)
                elif getattr(config, 'aligned_training', False):
                    diff = _random.choice(['easy', 'median', 'hard'])
                    num_crop = int(npoints * CROP_RATIO[diff])
                    partial, _ = misc.seprate_point_cloud(gt, npoints, num_crop,
                                    fixed_points=VIEWPOINT_CHOICES)
                    partial = misc.fps(partial, 2048)
                    partial = partial.cuda()
                else:
                    partial, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
                    partial = partial.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            num_iter += 1

            if diff_tensor is not None:
                ret = base_model(partial, difficulty=diff_tensor)
            else:
                ret = base_model(partial)

            # Branch by backbone family: UpTrans returns (loss_all, cd0, cd3) + has
            # separate get_constrain; AdaPoinTr returns (sparse_loss, dense_loss).
            is_uptrans = (config.model.NAME in UPTRANS_NAMES)
            if is_uptrans:
                # UpTrans family: 4-CD reconstruction + separate orth constraint.
                # Signature split between vanilla and DiffCond variants:
                #   vanilla  UpTrans.get_loss(ret, gt)                    -- 2 args only
                #   UpTrans_DiffCond.get_loss(ret, gt, epoch=, diff_label=, difficulty_labels=)
                if config.model.NAME == 'UpTrans_DiffCond':
                    if diff_tensor is not None:
                        loss_pcd, cd0, cd3 = base_model.module.get_loss(
                            ret, gt, epoch=epoch, diff_label=diff_tensor)
                    else:
                        # PCN fallback path (option (b), median-constant): no per-sample
                        # difficulty, CE term skipped inside get_loss.
                        loss_pcd, cd0, cd3 = base_model.module.get_loss(ret, gt, epoch=epoch)
                else:
                    # Vanilla UpTrans (no DiffCond additions, no epoch param).
                    loss_pcd, cd0, cd3 = base_model.module.get_loss(ret, gt)
                orth1, orth2 = base_model.module.get_constrain(ret)
                orth_weight = getattr(config.loss, 'orth_weight', 1.0)
                _loss = loss_pcd + orth_weight * (orth1 + orth2)
                # Parallel meter names for the AverageMeter below.
                sparse_loss, dense_loss = cd0, cd3
            else:
                if diff_tensor is not None:
                    sparse_loss, dense_loss = base_model.module.get_loss(ret, gt, epoch, difficulty_labels=diff_tensor)
                else:
                    sparse_loss, dense_loss = base_model.module.get_loss(ret, gt, epoch)
                _loss = sparse_loss + dense_loss

            # NaN/Inf guard (2026-04-16): skip batch if loss explodes.
            # Prevents a single bad sample from poisoning all weights.
            # CRITICAL: must free the computational graph (del + empty_cache)
            # since backward() won't be called to release intermediates.
            if torch.isnan(_loss) or torch.isinf(_loss):
                if logger:
                    print_log(f'WARNING: NaN/Inf loss at epoch {epoch} batch {idx}, '
                              f'skipping gradient update', logger=logger)
                del _loss, sparse_loss, dense_loss, ret
                torch.cuda.empty_cache()
                base_model.zero_grad()
                num_iter = 0
                continue

            _loss.backward()

            # forward
            if num_iter == config.step_per_update:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), getattr(config, 'grad_norm_clip', 10), norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                sparse_loss = dist_utils.reduce_tensor(sparse_loss, args)
                dense_loss = dist_utils.reduce_tensor(dense_loss, args)
                losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])
            else:
                losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])


            if args.distributed:
                torch.cuda.synchronize()

            n_itr = epoch * n_batches + idx
            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Sparse', sparse_loss.item() * 1000, n_itr)
                train_writer.add_scalar('Loss/Batch/Dense', dense_loss.item() * 1000, n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 100 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)

            if config.scheduler.type == 'GradualWarmup':
                if n_itr < config.scheduler.kwargs_2.total_epoch:
                    scheduler.step()

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step()
        else:
            scheduler.step()
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Sparse', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Dense', losses.avg(1), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)

        if epoch % args.val_freq == 0:
            # Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger=logger)

            # Save ckeckpoints
            if  metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)      
        # Save milestone checkpoints for mid-training S/M/H evaluation (C0/C1/C2)
        _milestone_epochs = getattr(config, 'milestone_epochs', [])
        if (config.max_epoch - epoch) < 2 or epoch in _milestone_epochs:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)     
    if train_writer is not None and val_writer is not None:
        train_writer.close()
        val_writer.close()

def validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1

    interval =  n_samples // 10

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            npoints = config.dataset.val._base_.N_POINTS
            dataset_name = config.dataset.val._base_.NAME
            if dataset_name == 'PCN' or dataset_name == 'Completion3D' or dataset_name == 'Projected_ShapeNet':
                partial = data[0].cuda()
                gt = data[1].cuda()
            elif dataset_name == 'ShapeNet':
                gt = data.cuda()

                # ---- Aligned validation: 8 fixed viewpoints x specified difficulty ----
                if getattr(config, 'aligned_validation', False):
                    val_mode = getattr(config, 'val_mode', 'hard')
                    num_crop = int(npoints * CROP_RATIO[val_mode])
                    # For DiffCond model, build a difficulty tensor matching val_mode
                    # so the model actually conditions on the requested difficulty
                    # instead of silently defaulting to median.
                    val_is_diffcond = (config.model.NAME in DIFFCOND_NAMES)
                    val_diff_tensor = None
                    if val_is_diffcond:
                        # UpTrans_DiffCond and AdaPoinTr_DiffCond both expose
                        # an identical DIFF_LEVELS mapping ({'easy':0,'median':1,'hard':2});
                        # import from AdaPoinTr_DiffCond to keep the existing
                        # call site unchanged.
                        from models.AdaPoinTr_DiffCond import DIFF_LEVELS as _VAL_DL
                        val_diff_tensor = torch.full((gt.shape[0],), _VAL_DL[val_mode],
                                                     dtype=torch.long, device=gt.device)
                    vp_losses = []
                    vp_metrics = []
                    for fp in VIEWPOINT_CHOICES:
                        p, _ = misc.seprate_point_cloud(gt, npoints, num_crop, fixed_points=fp)
                        p = misc.fps(p, 2048)
                        if val_diff_tensor is not None:
                            r = base_model(p, difficulty=val_diff_tensor)
                        else:
                            r = base_model(p)
                        c_pts, d_pts = pick_outputs(r, config.model.NAME)
                        sl1 = ChamferDisL1(c_pts, gt).item() * 1000
                        sl2 = ChamferDisL2(c_pts, gt).item() * 1000
                        dl1 = ChamferDisL1(d_pts, gt).item() * 1000
                        dl2 = ChamferDisL2(d_pts, gt).item() * 1000
                        vp_losses.append([sl1, sl2, dl1, dl2])
                        # 2026-04-16 HOTFIX: EMD makes eval 10x slower (~4h/eval vs ~25min).
                        # Reverting to require_emd=False for ep-0 diagnostic speed.
                        # For Phase 2.6 EMD rerun (paper table), flip back to True and
                        # budget ~40h of GPU (9 evals x 4.4h). See tools/AUDIT_REPORT.md.
                        m = [x.item() for x in Metrics.get(d_pts, gt, require_emd=False)]
                        vp_metrics.append(m)
                    # Average over 8 viewpoints
                    n_vp = len(VIEWPOINT_CHOICES)
                    avg_losses = [sum(l[i] for l in vp_losses) / n_vp for i in range(4)]
                    avg_metrics = [sum(m[i] for m in vp_metrics) / n_vp for i in range(len(vp_metrics[0]))]
                    test_losses.update(avg_losses)
                    _metrics = avg_metrics
                    for _taxonomy_id in taxonomy_ids:
                        if _taxonomy_id not in category_metrics:
                            category_metrics[_taxonomy_id] = AverageMeter(Metrics.names())
                        category_metrics[_taxonomy_id].update(_metrics)
                    if (idx+1) % (n_samples // 10) == 0:
                        print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                                    (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in avg_losses],
                                    ['%.4f' % m for m in _metrics]), logger=logger)
                    continue  # skip default path below
                # ---- End aligned validation ----

                partial, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
                partial = partial.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            # Default validation path. For PCN DiffCond, prefer the per-sample
            # proxy difficulty (data[2] from PCNDataset) so that ckpt-best
            # selection uses the SAME conditioning regime as training + test.
            # Validation-conditioning guard: without this, validation would use median while
            # train/test use proxy, contaminating ckpt-best selection.
            # Validation guard: fail-hard if DiffCond+PCN proxy missing.
            # For ShapeNet: random-crop range [1/4, 3/4] above does NOT yield a
            # single per-sample difficulty -> fall back to median (diagnostic
            # only; ShapeNet-55 cross-backbone reported numbers come from aligned-validation above
            # or test-time oracle in test_net).
            assert_pcn_diffcond_proxy(config, dataset_name, data)
            default_diff_tensor = None
            if config.model.NAME in DIFFCOND_NAMES:
                from models.AdaPoinTr_DiffCond import DIFF_LEVELS as _VAL_DL
                if dataset_name == 'PCN' and pcn_proxy_valid(data):
                    default_diff_tensor = data[2].to(gt.device).long()
                else:
                    default_diff_tensor = torch.full(
                        (gt.shape[0],), _VAL_DL['median'],
                        dtype=torch.long, device=gt.device)
            if default_diff_tensor is not None:
                ret = base_model(partial, difficulty=default_diff_tensor)
            else:
                ret = base_model(partial)
            coarse_points, dense_points = pick_outputs(ret, config.model.NAME)

            sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
            sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
            dense_loss_l1 =  ChamferDisL1(dense_points, gt)
            dense_loss_l2 =  ChamferDisL2(dense_points, gt)

            if args.distributed:
                sparse_loss_l1 = dist_utils.reduce_tensor(sparse_loss_l1, args)
                sparse_loss_l2 = dist_utils.reduce_tensor(sparse_loss_l2, args)
                dense_loss_l1 = dist_utils.reduce_tensor(dense_loss_l1, args)
                dense_loss_l2 = dist_utils.reduce_tensor(dense_loss_l2, args)

            test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

            # 2026-04-16 HOTFIX: revert to require_emd=False for speed (EMD is 10x slower).
            # Flip back to True for Phase 2.6 EMD rerun session (paper table).
            _metrics = Metrics.get(dense_points, gt, require_emd=False)
            if args.distributed:
                _metrics = [dist_utils.reduce_tensor(_metric, args).item() for _metric in _metrics]
            else:
                _metrics = [_metric.item() for _metric in _metrics]

            for _taxonomy_id in taxonomy_ids:
                if _taxonomy_id not in category_metrics:
                    category_metrics[_taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[_taxonomy_id].update(_metrics)


            # if val_writer is not None and idx % 200 == 0:
            #     input_pc = partial.squeeze().detach().cpu().numpy()
            #     input_pc = misc.get_ptcloud_img(input_pc)
            #     val_writer.add_image('Model%02d/Input'% idx , input_pc, epoch, dataformats='HWC')

            #     sparse = coarse_points.squeeze().cpu().numpy()
            #     sparse_img = misc.get_ptcloud_img(sparse)
            #     val_writer.add_image('Model%02d/Sparse' % idx, sparse_img, epoch, dataformats='HWC')

            #     dense = dense_points.squeeze().cpu().numpy()
            #     dense_img = misc.get_ptcloud_img(dense)
            #     val_writer.add_image('Model%02d/Dense' % idx, dense_img, epoch, dataformats='HWC')
                
            #     gt_ptcloud = gt.squeeze().cpu().numpy()
            #     gt_ptcloud_img = misc.get_ptcloud_img(gt_ptcloud)
            #     val_writer.add_image('Model%02d/DenseGT' % idx, gt_ptcloud_img, epoch, dataformats='HWC')
        
            if (idx+1) % interval == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                            (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)
        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()
     
    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)

    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall\t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Loss/Epoch/Sparse', test_losses.avg(0), epoch)
        val_writer.add_scalar('Loss/Epoch/Dense', test_losses.avg(2), epoch)
        for i, metric in enumerate(test_metrics.items):
            val_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch)

    return Metrics(config.consider_metric, test_metrics.avg())


# Legacy alias -- test_net() references this; new code uses CROP_RATIO at module top
crop_ratio = CROP_RATIO

def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
 
    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger = logger)
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP    
    if args.distributed:
        raise NotImplementedError()

    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger=logger)

def test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger = None):

    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            npoints = config.dataset.test._base_.N_POINTS
            dataset_name = config.dataset.test._base_.NAME
            if dataset_name == 'PCN' or dataset_name == 'Projected_ShapeNet':
                partial = data[0].cuda()
                gt = data[1].cuda()

                # Stage-2.5: for PCN DiffCond prefer the per-sample proxy difficulty
                # from PCNDataset (data[2]); fall back to median-constant only when
                # proxy is absent (-1 sentinel or not loaded; reachable only with
                # allow_pcn_median_fallback). For UpTrans family we also need
                # pick_outputs (dense is ret[1], not ret[-1]).
                # Validation guard: fail-hard if DiffCond+PCN proxy missing.
                assert_pcn_diffcond_proxy(config, dataset_name, data)
                pcn_diff_tensor = None
                if config.model.NAME in DIFFCOND_NAMES:
                    from models.AdaPoinTr_DiffCond import DIFF_LEVELS as _DL
                    if pcn_proxy_valid(data):
                        pcn_diff_tensor = data[2].to(gt.device).long()
                    else:
                        pcn_diff_tensor = torch.full((gt.shape[0],), _DL['median'],
                                                     dtype=torch.long, device=gt.device)
                if pcn_diff_tensor is not None:
                    ret = base_model(partial, difficulty=pcn_diff_tensor)
                else:
                    ret = base_model(partial)
                coarse_points, dense_points = pick_outputs(ret, config.model.NAME)

                sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
                dense_loss_l1 =  ChamferDisL1(dense_points, gt)
                dense_loss_l2 =  ChamferDisL2(dense_points, gt)

                test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

                _metrics = Metrics.get(dense_points, gt, require_emd=True)
                # test_metrics.update(_metrics)

                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)

            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                choice = [torch.Tensor([1,1,1]),torch.Tensor([1,1,-1]),torch.Tensor([1,-1,1]),torch.Tensor([-1,1,1]),
                            torch.Tensor([-1,-1,1]),torch.Tensor([-1,1,-1]), torch.Tensor([1,-1,-1]),torch.Tensor([-1,-1,-1])]
                num_crop = int(npoints * crop_ratio[args.mode])
                is_diffcond = (config.model.NAME in DIFFCOND_NAMES)
                diff_src = getattr(args, 'difficulty_source', 'oracle')
                # Fail fast BEFORE any per-view crop/FPS work or GPU forward:
                # UpTrans_DiffCond is oracle-only (main paper Sec. 5.5).
                if config.model.NAME == 'UpTrans_DiffCond' and diff_src == 'predict':
                    raise ValueError(
                        "UpTrans_DiffCond does not support --difficulty_source predict "
                        "in cross-backbone scope (predictor input depends on feat_extractor output; "
                        "would require splitting encoder)."
                    )
                for item in choice:
                    partial, _ = misc.seprate_point_cloud(gt, npoints, num_crop, fixed_points = item)
                    # NOTE: subsample the input
                    partial = misc.fps(partial, 2048)
                    if is_diffcond:
                        if diff_src == 'predict':
                            # UpTrans_DiffCond path already rejected above (fail-fast);
                            # this branch only fires for AdaPoinTr_DiffCond.
                            ret = base_model(partial, difficulty='predict')
                        else:  # oracle (default)
                            from models.AdaPoinTr_DiffCond import DIFF_LEVELS as _DL
                            diff_t = torch.full((gt.shape[0],), _DL[args.mode],
                                                dtype=torch.long, device=gt.device)
                            ret = base_model(partial, difficulty=diff_t)
                    else:
                        ret = base_model(partial)
                    coarse_points, dense_points = pick_outputs(ret, config.model.NAME)

                    sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                    sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
                    dense_loss_l1 =  ChamferDisL1(dense_points, gt)
                    dense_loss_l2 =  ChamferDisL2(dense_points, gt)

                    test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

                    # 2026-04-16 HOTFIX: revert to require_emd=False for speed.
                    # Flip back to True for Phase 2.6 EMD rerun session (paper table).
                    _metrics = Metrics.get(dense_points, gt, require_emd=False)



                    if taxonomy_id not in category_metrics:
                        category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                    category_metrics[taxonomy_id].update(_metrics)
            elif dataset_name == 'KITTI':
                partial = data.cuda()
                ret = base_model(partial)
                dense_points = ret[-1]
                target_path = os.path.join(args.experiment_path, 'vis_result')
                if not os.path.exists(target_path):
                    os.mkdir(target_path)
                misc.visualize_KITTI(
                    os.path.join(target_path, f'{model_id}_{idx:03d}'),
                    [partial[0].cpu(), dense_points[0].cpu()]
                )
                continue
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            if (idx+1) % 200 == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                            (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)
        if dataset_name == 'KITTI':
            return
        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[TEST] Metrics = %s' % (['%.4f' % m for m in test_metrics.avg()]), logger=logger)

     

    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)


    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall \t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)
    return 
