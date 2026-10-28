import os, sys
# online package
import torch
# optimizer
import torch.optim as optim
from timm.scheduler import CosineLRScheduler
# dataloader
from datasets import build_dataset_from_cfg
from models import build_model_from_cfg
# utils
from utils.logger import *
from utils.misc import *

def dataset_builder(args, config):
    dataset = build_dataset_from_cfg(config._base_, config.others)
    shuffle = config.others.subset == 'train'
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle = shuffle)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = config.others.bs if shuffle else 1,
                                            num_workers = int(args.num_workers),
                                            drop_last = config.others.subset == 'train',
                                            worker_init_fn = worker_init_fn,
                                            sampler = sampler)
    else:
        sampler = None
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.others.bs if shuffle else 1,
                                                shuffle = shuffle, 
                                                drop_last = config.others.subset == 'train',
                                                num_workers = int(args.num_workers),
                                                worker_init_fn=worker_init_fn)
    return sampler, dataloader

def model_builder(config):
    model = build_model_from_cfg(config)
    return model

def build_optimizer(base_model, config):
    opti_config = config.optimizer
    if opti_config.type == 'AdamW':
        def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
            decay = []
            no_decay = []
            for name, param in model.module.named_parameters():
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': weight_decay}]
        param_groups = add_weight_decay(base_model, weight_decay=opti_config.kwargs.weight_decay)
        optimizer = optim.AdamW(param_groups, **opti_config.kwargs)
    elif opti_config.type == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, base_model.parameters()), **opti_config.kwargs)
    elif opti_config.type == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, base_model.parameters()), **opti_config.kwargs)
    else:
        raise NotImplementedError()

    return optimizer

def build_scheduler(base_model, optimizer, config, last_epoch=-1):
    sche_config = config.scheduler
    if sche_config.type == 'LambdaLR':
        scheduler = build_lambda_sche(optimizer, sche_config.kwargs, last_epoch=last_epoch)  # misc.py
    elif sche_config.type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, last_epoch=last_epoch, **sche_config.kwargs)
    elif sche_config.type == 'GradualWarmup':
        scheduler_steplr = torch.optim.lr_scheduler.StepLR(optimizer, last_epoch=last_epoch, **sche_config.kwargs_1)
        scheduler = GradualWarmupScheduler(optimizer, after_scheduler=scheduler_steplr, **sche_config.kwargs_2)
    elif sche_config.type == 'CosLR':
        scheduler = CosineLRScheduler(optimizer,
                t_initial=sche_config.kwargs.t_max,
                lr_min=sche_config.kwargs.min_lr,
                warmup_t=sche_config.kwargs.initial_epochs,
                t_in_epochs=True)
    else:
        raise NotImplementedError()
    
    if config.get('bnmscheduler') is not None:
        bnsche_config = config.bnmscheduler
        if bnsche_config.type == 'Lambda':
            bnscheduler = build_lambda_bnsche(base_model, bnsche_config.kwargs)  # misc.py
        scheduler = [scheduler, bnscheduler]
    
    return scheduler

def resume_model(base_model, args, logger = None):
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger = logger)
        return 0, 0
    print_log(f'[RESUME INFO] Loading model weights from {ckpt_path}...', logger = logger )

    # load state dict
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
    state_dict = torch.load(ckpt_path, map_location=map_location)
    # parameter resume of base model
    # if args.local_rank == 0:
    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    base_model.load_state_dict(base_ckpt)

    # parameter
    start_epoch = state_dict['epoch'] + 1
    best_metrics = state_dict['best_metrics']
    if not isinstance(best_metrics, dict):
        best_metrics = best_metrics.state_dict()
    # print(best_metrics)

    print_log(f'[RESUME INFO] resume ckpts @ {start_epoch - 1} epoch( best_metrics = {str(best_metrics):s})', logger = logger)
    return start_epoch, best_metrics

def resume_optimizer(optimizer, args, logger = None):
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger = logger)
        return 0, 0, 0
    print_log(f'[RESUME INFO] Loading optimizer from {ckpt_path}...', logger = logger )
    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # optimizer
    optimizer.load_state_dict(state_dict['optimizer'])

def save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, prefix, args, logger = None):
    if args.local_rank == 0:
        torch.save({
                    'base_model' : base_model.module.state_dict() if args.distributed else base_model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'epoch' : epoch,
                    'metrics' : metrics.state_dict() if metrics is not None else dict(),
                    'best_metrics' : best_metrics.state_dict() if best_metrics is not None else dict(),
                    }, os.path.join(args.experiment_path, prefix + '.pth'))
        print_log(f"Save checkpoint at {os.path.join(args.experiment_path, prefix + '.pth')}", logger = logger)

def load_model(base_model, ckpt_path, logger = None):
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    print_log(f'Loading weights from {ckpt_path}...', logger = logger )

    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # parameter resume of base model
    if state_dict.get('model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['model'].items()}
    elif state_dict.get('base_model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    else:
        raise RuntimeError('mismatch of ckpt weight')
    # --- DiffCond special loading: strict=False + query_ranking remap ---
    model_cls = base_model.__class__.__name__
    if model_cls == 'AdaPoinTr_DiffCond':
        import torch.nn as _nn
        has_diffcond = any('diff_cond_' in k for k in base_ckpt)

        if not has_diffcond:
            # Source is vanilla AdaPoinTr: remap query_ranking -> base_scorer
            extra = {}
            for k, v in list(base_ckpt.items()):
                if k.startswith('base_model.query_ranking.'):
                    suffix = k[len('base_model.query_ranking.'):]
                    if suffix.startswith(('0.', '2.', '4.')):
                        extra[f'base_model.diff_cond_query_ranking.base_scorer.{suffix}'] = v
            base_ckpt.update(extra)

        missing, unexpected = base_model.load_state_dict(base_ckpt, strict=False)

        # Zero-init new conditioning params (only when loading from vanilla)
        if not has_diffcond:
            pctr = base_model.base_model
            _nn.init.zeros_(pctr.diff_cond_proj.weight)
            _nn.init.zeros_(pctr.diff_cond_proj.bias)
            _nn.init.zeros_(pctr.diff_cond_gate[0].bias)
            _nn.init.zeros_(pctr.diff_cond_query_ranking.diff_delta[-1].weight)
            _nn.init.zeros_(pctr.diff_cond_query_ranking.diff_delta[-1].bias)

        EXPECTED_NEW = ('diff_cond_', 'diff_predictor')
        if logger:
            non_expected = [k for k in missing if not any(p in k for p in EXPECTED_NEW)]
            print_log(f'DiffCond load: missing={len(missing)}, unexpected={len(unexpected)}',
                      logger=logger)
            if non_expected:
                print_log(f'WARNING non-expected missing: {non_expected}', logger=logger)
    elif model_cls == 'UpTrans_DiffCond':
        # Cross-backbone DiffCond. Warm-start from a vanilla UpTrans ckpt:
        # all base submodule names match (feat_extractor.sa_module_* / transformer_* /
        # ps_* / mlp_* / refine_{1,2}.{dict_embed,query,key} / decoder.*) so
        # strict=False load populates them, and the new DiffCond params
        # (difficulty_embedding, difficulty_predictor, refine_{1,2}.diff_key_delta,
        # feat_extractor.film_{gamma,beta}) stay at init. Output-affecting
        # new layers get zero-init'd so reconstruction forward at t=0 is
        # bit-exact to vanilla -- see models/UpTrans_DiffCond.py docstrings.
        import torch.nn as _nn

        has_diffcond = any(
            ('difficulty_' in k) or ('film_' in k) or ('diff_key_delta' in k)
            for k in base_ckpt
        )

        missing, unexpected = base_model.load_state_dict(base_ckpt, strict=False)

        # Zero-init output-affecting DiffCond layers only when loading from vanilla.
        # (embedding + predictor stay at random-init; this is intentional -- CE aux
        # loss is part of the tested DiffCond design, not an init-free control.)
        if not has_diffcond:
            fx = base_model.feat_extractor
            for m in [fx.film_gamma[-1], fx.film_beta[-1]]:
                _nn.init.zeros_(m.weight)
                _nn.init.zeros_(m.bias)
            for refine in [fx.refine_1, fx.refine_2]:
                _nn.init.zeros_(refine.diff_key_delta[-1].weight)
                _nn.init.zeros_(refine.diff_key_delta[-1].bias)

        EXPECTED_NEW = (
            'difficulty_embedding', 'difficulty_predictor',
            'diff_key_delta', 'film_gamma', 'film_beta',
        )
        non_expected = [k for k in missing if not any(p in k for p in EXPECTED_NEW)]
        if logger:
            print_log(
                f'UpTrans_DiffCond load: missing={len(missing)}, '
                f'unexpected={len(unexpected)}',
                logger=logger,
            )
        # If base-model keys failed to load, the bit-exact-at-t=0 invariant is
        # violated -- fail hard. Without this check, training would proceed
        # silently with random-init weights in the "missed" base submodules.
        if non_expected:
            if logger:
                print_log(f'FATAL non-expected missing base keys: {non_expected}',
                          logger=logger)
            raise RuntimeError(
                f'UpTrans_DiffCond warm-start failed: {len(non_expected)} base '
                f'model keys are missing (expected to load from pretrained UpTrans). '
                f'Examples: {non_expected[:5]}. Aborting to preserve bit-exact '
                f't=0 equivalence protocol.'
            )
    else:
        base_model.load_state_dict(base_ckpt)

    epoch = -1
    if state_dict.get('epoch') is not None:
        epoch = state_dict['epoch']
    if state_dict.get('metrics') is not None:
        metrics = state_dict['metrics']
        if not isinstance(metrics, dict):
            metrics = metrics.state_dict()
    else:
        metrics = 'No Metrics'
    print_log(f'ckpts @ {epoch} epoch( performance = {str(metrics):s})', logger = logger)
    return 