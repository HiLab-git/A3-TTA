from statistics import mode
from copy import deepcopy
import torch
import torch.optim as optim
from torchvision import transforms
from robustbench.data import dataset_all, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler,Scale_img,Scale_imglab,scale,convert_2d,scal_spacing
from robustbench.utils import load_model,setup_source
from robustbench.utils import clean_accuracy as accuracy
from robustbench.metrics import dice_eval,assd_eval, hd95_eval
import tent
import norm
import a3_tta
from utils.sam import SAM
from conf import cfg, load_cfg_fom_args
import math

def _param_stats_from_iter(params_iter):
    """return (#tensors, #numel) for an iterable of tensors"""
    n_tensors = 0
    n_numel = 0
    for p in params_iter:
        n_tensors += 1
        n_numel += p.numel()
    return n_tensors, n_numel

def _model_param_summary(model):
    total_tensors, total_numel = _param_stats_from_iter(model.parameters())
    train_tensors, train_numel = _param_stats_from_iter(
        (p for p in model.parameters() if p.requires_grad)
    )
    return {
        "total_tensors": total_tensors,
        "total_numel": total_numel,
        "train_tensors": train_tensors,
        "train_numel": train_numel,
    }

def _fmt_count(n):  
    return f"{n:,}"


def setup_norm(model):
    """Set up test-time normalization adaptation.

    Adapt by normalizing features with test batch statistics.
    The statistics are measured independently for each batch;
    no running average or other cross-batch estimation is used.
    """
    norm_model = norm.Norm(model)
    stats, stat_names = norm.collect_stats(model)
    print(stat_names)
   
    return norm_model


def setup_tent(model):
    """Set up TENT adaptation (BN-only by design)."""
    model = tent.configure_model(model)                
    params, param_names = tent.collect_params(model)   

    tent_tensor_cnt, tent_numel = _param_stats_from_iter(params)
    summ = _model_param_summary(model)

    print("\n[TENT] parameter summary")
    print(f"  total params (numel):     {_fmt_count(summ['total_numel'])} "
          f"| tensors: {_fmt_count(summ['total_tensors'])}")
    print(f"  trainable (requires_grad): {_fmt_count(summ['train_numel'])} "
          f"| tensors: {_fmt_count(summ['train_tensors'])}")
    print(f"  collected by TENT:         {_fmt_count(tent_numel)} "
          f"| tensors: {_fmt_count(tent_tensor_cnt)}")
    if isinstance(param_names, (list, tuple)) and len(param_names) > 0:
        preview = 10
        print(f"  trainable names preview ({min(preview, len(param_names))}/{len(param_names)}):")
        for n in param_names[:preview]:
            print("   -", n)

    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    return model, tent_model




def setup_a3_tta(model):
    anchor_model = deepcopy(model)
    model.train()
    anchor_model.eval()

    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg.OPTIM.LR, betas=(0.5, 0.999))
    cotta_model = a3_tta.TTA(model, anchor_model, optimizer)

    summ = _model_param_summary(model)
    print("\n[a3-tta / CoTTA-style] parameter summary")
    print(f"  total params (numel):     {_fmt_count(summ['total_numel'])} "
          f"| tensors: {_fmt_count(summ['total_tensors'])}")
    print(f"  trainable (requires_grad): {_fmt_count(summ['train_numel'])} "
          f"| tensors: {_fmt_count(summ['train_tensors'])}")

    if summ['train_numel'] < 0.9 * summ['total_numel']:
        print("  ⚠️  Trainable fraction < 90% — some layers may be frozen unexpectedly.")

    return cotta_model



def setup_optimizer(params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=cfg.OPTIM.LR,
                   momentum=cfg.OPTIM.MOMENTUM,
                   dampening=cfg.OPTIM.DAMPENING,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError