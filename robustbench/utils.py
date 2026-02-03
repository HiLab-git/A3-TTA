import argparse
import dataclasses
import json
import math
import os
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Union
import requests
import torch
from torch import nn
from robustbench.seg_net.unet2d import UNet2D
from robustbench.seg_net.unet2d5 import UNet2D5
from robustbench.seg_net.unet3d import UNet3D
from torch.nn import init
from unet import UNet
from robustbench.losses import DiceLoss
def get_params(network_name,dataset):
    if 'mms' in dataset:
        class_num = 4
    elif 'fb' in dataset:
        class_num = 2
    elif 'prostate' in dataset:
        class_num = 2
    elif 'city' in dataset:
        class_num = 20
    else:
        raise "undifined dataset!!!"
    params = {'in_chns':1,
              'ft_chns':[16, 32, 64, 128, 256],
              'dropout_p':  [0, 0.5, 0.3, 0.4, 0.5],
              'n_classes': class_num,
              'bilinear': True,
              'deep_supervise': False,
              'lr': 0.0001,
              'up_mode': 'upsample'}
    if network_name == 'unet3d':
        params['trilinear'] = True
    return params

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
            
            

def add_substr_to_state_dict(state_dict, substr):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[substr + k] = v
    return new_state_dict
def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    # logger.info(f"model for evaluation: %s", model)
    return model

def load_model(network_name = 'unet',
               checkpoint_dir = None,
               dataset = 'mms',
               adaptation_method = 'source'):
    """Loads a model from the model_zoo.

     The model is trained on the given ``dataset``, for the given ``threat_model``.

    :param model_name: The name used in the model zoo.
    :param model_dir: The base directory where the models are saved.
    :param dataset: The dataset on which the model is trained.
    :param threat_model: The threat model for which the model is trained.
    :param norm: Deprecated argument that can be used in place of ``threat_model``. If specified, it
      overrides ``threat_model``

    :return: A ready-to-used trained model.
    """
    params = get_params(network_name,dataset)
    if network_name == 'unet':
        model = UNet(params)
    elif network_name == 'unet2d':
        model = UNet2D(params)
    elif network_name == 'unet3d':
        model = UNet3D(params)
    elif network_name == 'unet2d5':
        model = UNet2D5(params)
    elif network_name == 'unet2d':
        model = UNet2D(params)
    else:
        raise "undifined network!!!"
    if adaptation_method == 'source':
        init_weights(model)
    else:
        model.load_state_dict(torch.load(checkpoint_dir,map_location='cpu'),strict=True)

    return model.eval()


import torch
from collections import OrderedDict

def smart_load_state_dict(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')

    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
            sd = ckpt['state_dict']
        elif 'model' in ckpt and isinstance(ckpt['model'], dict):
            sd = ckpt['model']
        elif 'model_state_dict' in ckpt and isinstance(ckpt['model_state_dict'], dict):
            sd = ckpt['model_state_dict']
        else:
            sd = ckpt
    else:
        sd = ckpt

    def strip_prefix(k):
        if k.startswith('model.'):
            return k[len('model.'):]
        if k.startswith('module.'):
            return k[len('module.'):]
        return k

    sd = {strip_prefix(k): v for k, v in sd.items()}

    model_sd = model.state_dict()
    filtered = OrderedDict()
    dropped = []
    for k, v in sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            filtered[k] = v
        else:
            dropped.append(k)

    missing = [k for k in model_sd.keys() if k not in filtered]
    unexpected = [k for k in filtered.keys() if k not in model_sd]  # 理论上为空

    print(f"[smart_load] loadable: {len(filtered)}  dropped: {len(dropped)}  missing: {len(missing)}")
    if missing:
        print(f"[smart_load] missing (show 10): {missing[:10]}")
    if dropped:
        print(f"[smart_load] dropped (show 10): {dropped[:10]}")

    model.load_state_dict(filtered, strict=False)  # 允许剩余少量未对齐的头部参数
    return model


def clean_accuracy(class_num: int,
                   model: nn.Module,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   grad = False,
                   batch_size: int = 100,
                   device: torch.device = None):
    if device is None:
        device = x.device
    loss = DiceLoss(class_num).to(device=device)
    acc = 0.
    with torch.no_grad():
        output = model(x)
        seg_loss = loss(output,y,weight=None,softmax=True)


    return seg_loss.item()


@dataclasses.dataclass
class ModelInfo:
    link: Optional[str] = None
    name: Optional[str] = None
    authors: Optional[str] = None
    additional_data: Optional[bool] = None
    number_forward_passes: Optional[int] = None
    dataset: Optional[str] = None
    venue: Optional[str] = None
    architecture: Optional[str] = None
    eps: Optional[float] = None
    clean_acc: Optional[float] = None
    reported: Optional[float] = None
    corruptions_acc: Optional[str] = None
    autoattack_acc: Optional[str] = None
    footnote: Optional[str] = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        type=str,
                        default='Carmon2019Unlabeled')
    parser.add_argument('--eps', type=float, default=8 / 255)
    parser.add_argument('--n_ex',
                        type=int,
                        default=100,
                        help='number of examples to evaluate on')
    parser.add_argument('--batch_size',
                        type=int,
                        default=500,
                        help='batch size for evaluation')
    parser.add_argument('--data_dir',
                        type=str,
                        default='./data',
                        help='where to store downloaded datasets')
    parser.add_argument('--model_dir',
                        type=str,
                        default='./models',
                        help='where to store downloaded models')
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='random seed')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='device to use for computations')
    parser.add_argument('--to_disk', type=bool, default=True)
    args = parser.parse_args()
    return args

