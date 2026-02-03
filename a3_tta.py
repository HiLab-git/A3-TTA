from copy import deepcopy
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.jit
import PIL
import torchvision.transforms as transforms
import my_transforms as my_transforms
from time import time
import logging
import math
from xml.etree.ElementInclude import FatalIncludeError
import torchvision.transforms.functional as FF
import os
import SimpleITK as sitk
from monai.losses import DiceLoss, DiceCELoss
import random
from utils.utils import rotate_single_random,derotate_single_random,add_gaussian_noise_3d
from robustbench.losses import DiceCeLoss
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_tta_transforms(gaussian_std: float=0.005, soft=False, clip_inputs=False):
    img_shape = (32, 32, 3)
    n_pixels = img_shape[0]

    clip_min, clip_max = 0.0, 1.0

    p_hflip = 0.5

    tta_transforms = transforms.Compose([
        my_transforms.Clip(0.0, 1.0), 
        # my_transforms.ColorJitterPro(
        #     brightness=[0.8, 1.2] if soft else [0.6, 1.4],
        #     contrast=[0.85, 1.15] if soft else [0.7, 1.3],
        #     saturation=[0.75, 1.25] if soft else [0.5, 1.5],
        #     hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
        #     gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        # ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),  
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1/16, 1/16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            resample=PIL.Image.BILINEAR,
            fillcolor=None
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=p_hflip),
        my_transforms.GaussianNoise(0, gaussian_std),
        my_transforms.Clip(clip_min, clip_max)
    ])
    return tta_transforms


def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


class CoTTA(nn.Module):
    """CoTTA adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False, mt_alpha=0.99, rst_m=0.1, ap=0.9):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.transform = get_tta_transforms()    
        self.mt = mt_alpha
        self.rst = rst_m
        self.ap = ap
        
    @torch.no_grad()    
    def get_SND(self, x, model_anchor):
        with torch.no_grad():
            b,c,w,h = x.shape
            entropy_list = []
            for i in range(b):
                anchor_prd = model_anchor(x[i:i+1]).softmax(1).detach()
                pred1 = anchor_prd.permute(0,2,3,1)
                pred1 = pred1.reshape(-1, pred1.size(3))
                pred1_rand = torch.randperm(pred1.size(0))
                # select_point = pred1_rand.shape[0]
                select_point = 100
                pred1 = F.normalize(pred1[pred1_rand[:select_point]])
                # print(select_point,pred1.shape)
                # print(torch.matmul(pred1, pred1.t()).shape,'113')
                pred1_en =  self.entropy(torch.matmul(pred1, pred1.t()) * 20)
                entropy_list.append(pred1_en)
        return entropy_list
    def find_closest_and_min(self, c_list):
        best_score = float('inf')
        best_index = None
        for i, c_value in enumerate(zip(c_list)):
            total_score = c_value
            if total_score[0] < best_score:
                best_score = total_score[0]
                best_index = i

        return best_index
    def forward(self, x, names):
        entropy_list = self.get_SND(x, self.model_anchor)
        index = self.find_closest_and_min(entropy_list)
        print(index,names[index],'86868686')
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        # Use this line to also restore the teacher model                         
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)


    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        outputs = self.model(x)
        # Teacher Prediction
        anchor_prob = torch.nn.functional.softmax(self.model_anchor(x), dim=1).max(1)[0]
        standard_ema = self.model_ema(x)
        # Augmentation-averaged Prediction
        N = 32 
        outputs_emas = []
        for i in range(N):
            outputs_  = self.model_ema(self.transform(x)).detach()
            outputs_emas.append(outputs_)
        # Threshold choice discussed in supplementary
        if anchor_prob.mean()<self.ap:
            outputs_ema = torch.stack(outputs_emas).mean(0)
        else:
            outputs_ema = standard_ema
        # Student update
        loss = (softmax_entropy(outputs, outputs_ema)).mean(0) 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Teacher update
        self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.model, alpha_teacher=self.mt)
        # Stochastic restore
        if True:
            for nm, m  in self.model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape)<self.rst).float().cuda() 
                        with torch.no_grad():
                            p.data = self.model_state[f"{nm}.{npp}"] * mask + p * (1.-mask)
        return outputs_ema
    @torch.no_grad()  # ensure grads in possible no grad context for testing
    def infer(self, x):
        # outputs = self.model(x)
        # Teacher Prediction
        # anchor_prob = torch.nn.functional.softmax(self.model_anchor(x), dim=1).max(1)[0]
        anchor_prob = torch.nn.functional.softmax(self.model_anchor(x), dim=1).max()
       
        standard_ema = self.model_ema(x)
        
        outputs_ema = standard_ema
        
        return outputs_ema
    def entropy(self, p, prob=True, mean=True):
        if prob:
            p = F.softmax(p, dim=1)
        en = -torch.sum(p * torch.log(p + 1e-5), 1)
        if mean:
            return torch.mean(en)
        else:
            return en


@torch.jit.script
def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    n, c, h, w =  x.shape
    entropy1 = -(x_ema.softmax(1) * x.log_softmax(1)).sum() / \
        (n * h * w * torch.log2(torch.tensor(c, dtype=torch.float)))
    return entropy1

def collect_params(model):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if True:#isinstance(m, nn.BatchNorm2d): collect all 
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
                    print(nm, np)
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what we update
    model.requires_grad_(False)
    # enable all trainable
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        else:
            m.requires_grad_(True)
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"

dicece_loss = DiceCeLoss(4)

class TTA(nn.Module):
    """TTA adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, anchor_model,optimizer, steps=2, episodic=False, mt_alpha=0.99, rst_m=0.1,):
        super().__init__()
        self.model = model
        self.steps = steps
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.optimizer = optimizer
        self.model_anchor = anchor_model
        self.model_state, self.model_ema,  = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.prototypes_src = torch.tensor([]).cuda()
        self.num_classes = 4 
        self.mt = mt_alpha
        self.criterion = nn.L1Loss()
        self.rst = rst_m
        self.max_pool = 40
        self.pool = Prototype_Pool(0.1,class_num=self.num_classes,max=self.max_pool).cuda()
        self.entropy_list = []
        self.cr_loss = ClassRatioLoss(temperature=1.)
    def forward(self, x, label_batch, names):
        self.label = label_batch
        
        if self.episodic:
            self.reset()
        for _ in range(1):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)
        return outputs
    def get_index(self):
        return self.index

    torch.autograd.set_detect_anomaly(True)
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        layer_fea = 'med' 
        bad_num = x.shape[0]
        with torch.no_grad():
            # entropy_list = self.get_SND_hh320(x, self.model_anchor.eval())
            # entropy_list = self.get_SND_ent(x, self.model_anchor.eval())
            # entropy_list = self.get_SND_drop(x, model)
            # entropy_list = self.get_SND_cc500(x, model)
            # entropy_list = self.get_SND_cc(x, model)
            entropy_list = self.get_SND_cc(x, model)
            # self.index = self.find_sorted_indices(entropy_list)
            # print(entropy_list)

        # # # reshape the feature
        latent_model_ = model.get_feature(x, loc = layer_fea) # ([10, 256, 20, 20])
        outputs = model.get_output(latent_model_,loc = layer_fea)
        
        b,c,w,h = latent_model_.shape
        sup_pixel = w
        latent_model = latent_model_.reshape(b,c,int(w/sup_pixel),sup_pixel,int(h/sup_pixel),sup_pixel)

        latent_model = latent_model.permute(0,2,4,1,3,5)


        bad_latent = latent_model
        bad_latent = bad_latent.reshape(bad_num*int(w/sup_pixel)*int(h/sup_pixel),c*sup_pixel*sup_pixel)

        fine_index = self.pool.update_pool(bad_latent, entropy_list)
        self.index = fine_index
        # print(fine_index,'866')

        bad_latent, out_latent, weight = self.pool(bad_latent,top_k = 1)

        bad_latent = bad_latent.view(bad_num,int(w/sup_pixel),int(h/sup_pixel),c,sup_pixel,sup_pixel)
        bad_latent = bad_latent.permute(0,3,1,4,2,5)
        bad_latent = bad_latent.reshape(bad_num,c,w,h)

        out_latent = out_latent.view(bad_num,int(w/sup_pixel),int(h/sup_pixel),c,sup_pixel,sup_pixel)
        out_latent = out_latent.permute(0,3,1,4,2,5)
        out_latent = out_latent.reshape(bad_num,c,w,h)

        mean_val = torch.mean(bad_latent)
        std_val = torch.std(bad_latent)


        # standardized_feature2 = (out_latent - mean_val) / std_val
        standardized_feature2 = out_latent
        standardized_feature2 = standardized_feature2*(1-weight[0]) + bad_latent*(weight[0])
        standardized_feature2 = (standardized_feature2 - mean_val) / std_val
        # print(weight[0], 'weight 0')
        latent_model_ = standardized_feature2
        # latent_model_[self.index[:bad_num]] = standardized_feature2
        


        outputs2 = model.get_output(latent_model_,loc = layer_fea)
        standard_ema = self.model_ema(x)
        loss_testu2 = (softmax_entropy(outputs2, outputs)).mean(0) 
        # loss_testu2 = (symmetric_cross_entropy(outputs2, outputs)).mean(0)
        loss_testu2_str = self.criterion(self.entropy(outputs2,mean=False) , self.entropy(outputs,mean=False)) 
        loss_testu = (softmax_entropy(outputs, standard_ema)).mean(0) 
        # loss = loss_testu2*1.0+loss_testu2_str*5.+loss_testu*0.2 ## prostate
        loss = loss_testu2*1.+loss_testu2_str*5.+loss_testu*1. ## mms
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        self.mt = 1. - loss_testu.item()
        self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.model, alpha_teacher=self.mt)

        return model(x)

    @torch.no_grad()  # ensure grads in possible no grad context for testing
    def infer(self, x):
        # outputs = self.model(x)
        # Teacher Prediction
        # anchor_prob = torch.nn.functional.softmax(self.model_anchor(x), dim=1).max(1)[0]
        self.model.eval()
        outputs = self.model(x)
        
        return outputs
    def entropy_T(self, p, prob=True, mean=True):
        if prob:
            p = F.softmax(p, dim=1)
        print(p.shape,p.max(),p.min(),'125')
        en = torch.matmul(p, p.transpose(2, 3))
        # p = -torch.sum(p * torch.log(p + 1e-5), 1)
        ###
        # p = p.cpu().detach().numpy()
        # predict_dir  = 'ppppppp.nii.gz'
        # out_lab_obj = sitk.GetImageFromArray(p[0]/1.0)
        # sitk.WriteImage(out_lab_obj, predict_dir)
        # ###
        
        if mean:
            return torch.mean(en)
        else:
            return en

    def entropy_f(self, p, prob=True, mean=True):
        en = -p * torch.log(p + 1e-5)
        # en = -torch.diag(p) * torch.log(torch.diag(p) + 1e-5)

        # print(en.shape, en.max())
        # print(p.shape,p.max(),p.min(),'155')
        if mean:
            return torch.mean(en)
        else:
            return en
    def entropy(self, p, prob=True, mean=True):
        if prob:
            p = F.softmax(p, dim=1)
        en = -torch.sum(p * torch.log(p + 1e-5), 1)
        if mean:
            return torch.mean(en)
        else:
            return en
        
    def filtering_entropy(self, p, num_class = 4, prob=True, mean=True):
        num_class = p.shape[1]
        e_margin = math.log(num_class)*0.30
        if prob:
            p = F.softmax(p, dim=1)
        en = -torch.sum(p * torch.log(p + 1e-5), 1)
        filter_ids_1 = torch.where(en < e_margin)
        en = en[filter_ids_1]
        if mean:
            return torch.mean(en)
        else:
            return en
        
    def filtering_mask(self, p, num_class = 4, prob=True, mean=True):
        num_class = p.shape[1]
        e_margin = math.log(num_class)*0.40
        if prob:
            p = F.softmax(p, dim=1)
        en = -torch.sum(p * torch.log(p + 1e-5), 1)
        mask = torch.zeros_like(en)
        filter_ids_1 = torch.where(en < e_margin)
        mask[filter_ids_1] = 1
        a_reshaped = mask.unsqueeze(1).repeat(1, num_class, 1, 1)
        return a_reshaped
       
    @torch.no_grad()  
    def get_SND_ent(self, x, model_anchor):
        entropy_list = []
        with torch.no_grad():
            b,c,w,h = x.shape
            for i in range(b):
                with torch.no_grad():
                    anchor_prd = model_anchor(x[i:i+1]).softmax(1).detach()
                pred1_en =  self.entropy(anchor_prd, prob=False)
                entropy_list.append(pred1_en)
        sorted_list = sorted(entropy_list)
        return sorted_list

    def get_SND_cc(self, x, model_anchor):
        entropy_list = []
        with torch.no_grad():
            b,c,w,h = x.shape
            for i in range(b):
                with torch.no_grad():
                    anchor_prd = model_anchor(x[i:i+1]).softmax(1).detach()
                pred1 = anchor_prd.permute(0,2,3,1)
                pred1 = pred1.reshape(-1, pred1.size(3))
                # print(pred1.shape, pred1.max(), pred1.min(),'210')
                # pred1 = F.normalize(pred1)
                # print(pred1.shape, pred1.max(), pred1.min(),pred1.t().shape,pred1.t().max(),pred1.t().min(),'220')
                pred1_en =  self.entropy_f(torch.matmul(pred1.t(), pred1)/102400.0)
                # pred1_en =  self.entropy_f(F.normalize(torch.matmul(pred1.t(), pred1)) )
                entropy_list.append(pred1_en)
        return entropy_list
    
    def get_SND_drop(self, x, model_anchor, num_inferences=10):
        entropy_list = []
        for module in model_anchor.modules(): 
            if isinstance(module, nn.Dropout): 
                module.train()
            elif isinstance(module, nn.BatchNorm2d): 
                module.eval()
        with torch.no_grad():
            b, c, w, h = x.shape
            for i in range(b):
                preds = []
                for _ in range(num_inferences):
                    anchor_prd = model_anchor(x[i:i+1]).softmax(1)
                    preds.append(anchor_prd)
                preds_tensor = torch.stack(preds)  # Shape: (num_inferences, c, w, h)
                pred_var = preds_tensor.var(dim=0).mean().item()
                entropy_list.append(pred_var)
        return entropy_list

    def get_SND_hh320(self, x, model_anchor):
        entropy_list = []
        with torch.no_grad():
            b,c,w,h = x.shape
            for i in range(b):
                with torch.no_grad():
                    pred1 = model_anchor(x[i:i+1]).softmax(1).detach()
                pred1 = pred1.permute(0,2,3,1)
                pred1 = pred1.reshape(-1, pred1.size(3))
                pred1_rand = torch.randperm(pred1.size(0))
                select_point = 200
                pred1 = F.normalize(pred1[pred1_rand[:select_point]])
                pred1_en =  self.entropy(torch.matmul(pred1, pred1.t()) * 20)
                entropy_list.append(pred1_en)
        sorted_list = sorted(entropy_list)
        return sorted_list


    def get_SND_cc500(self, x, model_anchor):
        with torch.no_grad():
            entropy_list = []
            b,c,w,h = x.shape
            for i in range(b):
                with torch.no_grad():
                    pred1 = model_anchor(x[i:i+1]).softmax(1).detach()
                pred1 = pred1.permute(0,2,3,1)
                pred1 = pred1.reshape(-1, pred1.size(3))
                pred1_rand = torch.randperm(pred1.size(0))
                select_point = 500
                pred1 = F.normalize(pred1[pred1_rand[:select_point]])
                pred1_en =  self.entropy(torch.matmul(pred1.t(), pred1))
                entropy_list.append(pred1_en)
        sorted_list = sorted(entropy_list)
        return sorted_list
        
    
    def get_dense_sup(self, x, model_anchor, norm_model):
        b,c,w,h = x.shape
        entropy_list = []
        mean_list = []
        var_list = []
        for i in range(b):
            anchor_prd = model_anchor(x[i:i+1]).softmax(1)
            if anchor_prd[0][1:].sum() < 1000:
                entropy = 1
            else:
                entropy = -(anchor_prd * torch.log2(anchor_prd + 1e-10)).sum() / (w*h)
            entropy_list.append(entropy)
            norm_model(x[i:i+1])
            layer_last = norm_model.enc.down_path[4]   
            target_bn_layer = layer_last.conv_conv[1]
            if target_bn_layer is not None:
                batch_mean = target_bn_layer.running_mean.clone().detach()
                batch_var = target_bn_layer.running_var.clone().detach()
                mean_list.append(batch_mean.mean())
                var_list.append(batch_var.mean())
            else:
                print("Target BN layer not found.")
        return entropy_list, mean_list, var_list
    
    def find_closest_and_min(self, c_list):
        best_score = float('inf')
        best_index = None
        for i, c_value in enumerate(zip(c_list)):
            total_score = c_value
            if total_score[0] < best_score:
                best_score = total_score[0]
                best_index = i
        return best_index
    
    def find_closest_and_max(self, c_list):
        best_score = float('-inf')
        best_index = None
        for i, c_value in enumerate(zip(c_list)):
            total_score = c_value
            if total_score[0] > best_score:
                best_score = total_score[0]
                best_index = i
        return best_index
    def find_sorted_indices(self, c_list):
        sorted_indices = sorted(enumerate(c_list), key=lambda x: x[1], reverse=True)
        sorted_indices = [index for index, _ in sorted_indices]
        
        return sorted_indices

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        # Use this line to also restore the teacher model                         
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
            
    def get_model_layer(self, model, restore_rate = 0.1):
        model_weights = model.model.state_dict()
        model_anchor_weights = self.model_anchor.state_dict()

        layer_diffs = {}
        total_layers = len(model_weights.keys())
        num_layers_to_show = int(total_layers * restore_rate)
        layers_to_compare = []
        for layer_name in model_anchor_weights.keys():
            print(layer_name,'265')
            model_weights[layer_name].float()
            model_anchor_weights[layer_name].float()
            print(model_weights[layer_name].float().shape)
            weight_diff = torch.norm(model_weights[layer_name].float() - model_anchor_weights[layer_name].float())
            layer_diffs[layer_name] = weight_diff
        # sorted_layers = sorted(layer_diffs.items(), key=lambda x: x[1])
        sorted_layers = sorted(layer_diffs.items(), key=lambda x: x[1], reverse=True)
        for i in range(num_layers_to_show):
            model_weights[sorted_layers[i][0]] = model_anchor_weights[sorted_layers[i][0]]
            # print(f"Layer: {sorted_layers[i][0]}, Weight Difference: {sorted_layers[i][1]}")
        model.model.load_state_dict(model_weights)
        return model
    # def get_model_channel(self, model, restore_rate = 0.1):
    #     model_weights = model.model.state_dict()
    #     self.model_state

    #     total_layers = len(model_weights.keys())
    #     layers_to_compare = []
    #     for layer_name in model_anchor_weights.keys():
    #         print(layer_name,'265')
    #         model_weights[layer_name].float()
    #         model_anchor_weights[layer_name].float()
    #         print(model_weights[layer_name].float().shape)
    #         weight_diff = torch.norm(model_weights[layer_name].float() - model_anchor_weights[layer_name].float())
    #         layer_diffs[layer_name] = weight_diff
    #     # sorted_layers = sorted(layer_diffs.items(), key=lambda x: x[1])
    #     sorted_layers = sorted(layer_diffs.items(), key=lambda x: x[1], reverse=True)
    #     for i in range(num_layers_to_show):
    #         model_weights[sorted_layers[i][0]] = model_anchor_weights[sorted_layers[i][0]]
    #         # print(f"Layer: {sorted_layers[i][0]}, Weight Difference: {sorted_layers[i][1]}")
    #     model.model.load_state_dict(model_weights)
    #     return model
    def contrastive_loss(self, features, labels=None, mask=None):
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).cuda()
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().cuda()
        else:
            mask = mask.float().cuda()

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_feature = self.projector(contrast_feature)
        contrast_feature = F.normalize(contrast_feature, p=2, dim=1)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss
    def loss_SND(self, anchor_prd):
        anchor_prd = anchor_prd.softmax(1)
        pred1 = anchor_prd.permute(0,2,3,1)
        pred1 = pred1.reshape(-1, pred1.size(3))
        pred1_rand = torch.randperm(pred1.size(0))
        select_point = 100
        pred1 = F.normalize(pred1[pred1_rand[:select_point]])
        pred1_en =  self.entropy(torch.matmul(pred1, pred1.t()) * 20)
        return pred1_en
    def randomHorizontalFlip(self, x):
        affine = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3).repeat(x.size(0), 1, 1)
        horizontal_flip = torch.tensor([-1, 0, 0, 0, 1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)
        # randomly flip some of the images in the batch
        mask = (torch.rand(x.size(0), device=x.device) > 0.5)
        affine[mask] = affine[mask] * horizontal_flip
        x = apply_affine(x, affine)
        return x, affine.detach()

    def randomRotate(self, x):
        affine = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3).repeat(x.size(0), 1, 1)
        rotation = torch.tensor([0, -1, 0, 1, 0, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)
        # randomly flip some of the images in the batch
        mask = (torch.rand(x.size(0), device=x.device) > 0.5)
        affine[mask] = rotation.repeat(mask.sum(), 1, 1)

        x = apply_affine(x, affine)
        
        return x, affine.detach()
    def randomVerticalFlip(self, x):
        affine = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3).repeat(x.size(0), 1, 1)
        vertical_flip = torch.tensor([1, 0, 0, 0, -1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)
        # randomly flip some of the images in the batch
        
        mask = (torch.rand(x.size(0), device=x.device) > 0.5)
        affine[mask] = affine[mask] * vertical_flip
        x = apply_affine(x, affine)
        return x, affine.detach()
    def randomResizeCrop(self, x):
        # TODO: Investigate different scale for x and y
        delta_scale_x = 0.2
        delta_scale_y = 0.2

        scale_matrix_x = torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)
        scale_matrix_y = torch.tensor([0, 0, 0, 0, 1, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)

        translation_matrix_x = torch.tensor([0, 0, 1, 0, 0, 0], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)
        translation_matrix_y = torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.float32, device=x.device).reshape(-1, 2, 3)

        delta_x = 0.5 * delta_scale_x * (2*torch.rand(x.size(0), 1, 1, device=x.device) - 1.0)
        delta_y = 0.5 * delta_scale_y * (2*torch.rand(x.size(0), 1, 1, device=x.device) -1.0)

        random_affine = (1 - delta_scale_x) * scale_matrix_x + (1 - delta_scale_y) * scale_matrix_y +\
                    delta_x * translation_matrix_x + \
                    delta_y * translation_matrix_y

        x = apply_affine(x, random_affine)
        return x, random_affine.detach()
    def get_pseudo_label(self, net, x, mult=3):
        preditions_augs = []
        # if is_training:
        #     net.eval()
        outnet = net(x)
        # preditions_augs.append(F.softmax(outnet, dim=1))
        preditions_augs.append(outnet)

        for i in range(mult-1):
            x_aug, rotate_affine = self.randomRotate(x)
            x_aug, vflip_affine = self.randomVerticalFlip(x_aug)
            x_aug, hflip_affine = self.randomHorizontalFlip(x_aug)

            # x_aug, hflip_affine = self.randomHorizontalFlip(x)
            # x_aug, crop_affine  = self.randomResizeCrop(x_aug)

            # get label on x_aug
            outnet = net(x_aug)
            pred_aug = outnet
            
            pred_aug = F.softmax(pred_aug, dim=1)
            pred_aug = apply_invert_affine(pred_aug, rotate_affine)
            pred_aug = apply_invert_affine(pred_aug, vflip_affine)
            pred_aug = apply_invert_affine(pred_aug, hflip_affine)

            preditions_augs.append(pred_aug)


        preditions = torch.stack(preditions_augs, dim=0).mean(dim=0) # batch x n_classes x h x w
        # renormalize the probability (due to interpolation of zeros, mean does not imply probability distribution over the classes)
        preditions = preditions / torch.sum(preditions, dim=1, keepdim=True)
        # if is_training:
        #     net.train()
        return preditions


@torch.jit.script
def symmetric_cross_entropy(x, x_ema):# -> torch.Tensor:
    n, c, h, w =  x.shape
    en = (-0.5*(x_ema.softmax(1) * x.log_softmax(1)).sum()-0.5*(x.softmax(1) * x_ema.log_softmax(1)).sum())
    en = en / \
        (n * h * w * torch.log2(torch.tensor(c, dtype=torch.float)))
    return en
def weight_symmetric_cross_entropy(x, x_ema, weights):# -> torch.Tensor:
    n, c, h, w =  x.shape
    assert len(weights) == n
    en = 0.
    for i in range(n):
        en += weights[i]*(-0.5*(x_ema[i:i+1].softmax(1) * x[i:i+1].log_softmax(1)).sum()-0.5*(x[i:i+1].softmax(1) * x_ema[i:i+1].log_softmax(1)).sum())
    en = en / \
        (n * h * w * torch.log2(torch.tensor(c, dtype=torch.float)))
    return en
@torch.jit.script
def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    n, c, h, w =  x.shape
    entropy1 = -(x_ema.softmax(1) * x.log_softmax(1)).sum() / \
        (n * h * w * torch.log2(torch.tensor(c, dtype=torch.float)))
    return entropy1

def collect_params(model):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        # if True:#isinstance(m, nn.BatchNorm2d): collect all 
        if 'dec1.last' not in nm:
            print(nm, '55',m, '496')
            for np, p in m.named_parameters():
                
                if np in ['weight', 'bias'] and p.requires_grad:
                # if p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    # optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, ema_model


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_tent_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model

def configure_norm_model(model, eps, momentum, reset_stats, no_stats):
    """Configure model for adaptation by test-time normalization."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            # use batch-wise statistics in forward
            m.train()
            # configure epsilon for stability, and momentum for updates
            m.eps = eps
            m.momentum = momentum
            if reset_stats:
                # reset state to estimate test stats without train stats
                m.reset_running_stats()
            if no_stats:
                # disable state entirely and use only batch stats
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
    return model

def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
    
def update_ema_variables(ema_model, model, alpha_teacher):
    # for ema_param, param in zip(ema_model.parameters(), model.parameters()):

    for ema_param, param in zip(ema_model.enc.parameters(), model.enc.parameters()):
    # for ema_param, param in zip(ema_model.dec1.parameters(), model.dec1.parameters()):
        # ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

def data_augmentation(image):
    random_numbers = random.sample(range(4), 2)
    if 0 in random_numbers:
        image = image.flip(-1)
    if 1 in random_numbers:
        image = image.flip(-2)
    if 2 in random_numbers:
        image = FF.rotate(image, 90)
    if 3 in random_numbers:
        image = FF.adjust_brightness(image, brightness_factor=0.2)
    return image
def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    # n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) 
def configure_debn_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what we update
    model.requires_grad_(True)
    # enable all trainable
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(False)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        else:
            m.requires_grad_(True)
    return model

def configure_cotta_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what we update
    model.requires_grad_(False)
    # enable all trainable
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        else:
            m.requires_grad_(True)
    return model

def apply_affine(x, affine):
    grid = torch.nn.functional.affine_grid(affine, x.size(), align_corners=False)
    out = torch.nn.functional.grid_sample(x, grid, padding_mode="reflection", align_corners=False)
    return out



class ClassRatioLoss__(torch.nn.Module):
    def __init__(self, temperature=1.0):
        super(ClassRatioLoss, self).__init__()
        self.temperature = temperature

    def forward(self, predicted_probs1, predicted_probs2):
        # Ensure input shapes are as expected
        assert predicted_probs1.shape == predicted_probs2.shape
        b, c, w, h = predicted_probs1.shape

        # Apply softmax along the channel dimension to get class probabilities
        probs1 = F.softmax(predicted_probs1, dim=1)
        probs2 = F.softmax(predicted_probs2, dim=1)

        # Calculate the class ratios
        class_ratios1 = torch.mean(probs1, dim=(2, 3), keepdim=True)
        class_ratios2 = torch.mean(probs2, dim=(2, 3), keepdim=True)

        # Calculate the KL Divergence between class ratios
        kl_divergence = F.kl_div(
            F.log_softmax(class_ratios1 / self.temperature, dim=1),
            class_ratios2 / self.temperature, reduction='batchmean'
        )
        # print(kl_divergence,'668')
        return kl_divergence

class ClassRatioLoss(torch.nn.Module):
    def __init__(self, temperature=1.0):
        super(ClassRatioLoss, self).__init__()
        self.temperature = temperature

    def forward(self, predicted_probs1, predicted_probs2):
        # Ensure input shapes are as expected
        assert predicted_probs1.shape == predicted_probs2.shape
        b, c, w, h = predicted_probs1.shape

        # Apply softmax along the channel dimension to get class probabilities
        probs1 = F.softmax(predicted_probs1, dim=1)
        probs2 = F.softmax(predicted_probs2, dim=1)

        # Calculate the class ratios using both methods
        class_ratios1_mean = torch.mean(probs1, dim=(2, 3), keepdim=True)
        class_ratios2_mean = torch.mean(probs2, dim=(2, 3), keepdim=True)

        # Calculate the class ratios by counting pixel numbers
        class_ratios1_pixel = torch.sum(probs1, dim=(2, 3), keepdim=True) / (w * h)
        class_ratios2_pixel = torch.sum(probs2, dim=(2, 3), keepdim=True) / (w * h)

        # Calculate the KL Divergence between class ratios
        kl_divergence1 = F.kl_div(F.log_softmax(class_ratios1_mean / self.temperature, dim=1),
                                  class_ratios2_mean / self.temperature, reduction='batchmean')

        kl_divergence2 = F.kl_div(F.log_softmax(class_ratios1_pixel / self.temperature, dim=1),
                                  class_ratios2_pixel / self.temperature, reduction='batchmean')

        # Combine the losses
        class_ratio_loss = (kl_divergence1 + kl_divergence2) / 2.0

        return class_ratio_loss




def apply_invert_affine(x, affine):
    # affine shape should be batch x 2 x 3
    # x shape should be batch x ch x h x w

    # get homomorphic transform
    H = torch.nn.functional.pad(affine, [0, 0, 0, 1], "constant", value=0.0)
    H[..., -1, -1] += 1.0

    inv_H = torch.inverse(H)
    inv_affine = inv_H[:, :2, :3]

    grid = torch.nn.functional.affine_grid(inv_affine, x.size(), align_corners=False)
    x = torch.nn.functional.grid_sample(x, grid, padding_mode="reflection", align_corners=False)
    return x


class Prototype_Pool(nn.Module):
    def __init__(self, delta=0.1, class_num=10, max=50):
        super(Prototype_Pool, self).__init__()
        self.class_num = class_num
        self.max_length = max
        self.delta = delta
        self.memory = []  # Use a list to store (feature, score) tuples

    def forward(self, x, top_k=5):
        mini_batch = 250
        outall = []
        
        if len(self.memory) == 0:
            return None, x, None
        
        memory_features = torch.stack([m[0] for m in self.memory])
        
        if x.shape[0] < mini_batch:
            cosine_similarities = torch.nn.functional.cosine_similarity(x.unsqueeze(1), memory_features.unsqueeze(0), dim=2)
            if memory_features.shape[0] >= top_k:
                outall = cosine_similarities.argsort(dim=1, descending=True)[:, :top_k]
            else:
                outall = cosine_similarities.argsort(dim=1, descending=True)[:, :1]
                outall = outall.repeat(1, top_k)
        else:
            for i in range(0, len(x), mini_batch):
                batch_a = x[i:i+mini_batch]
                cosine_similarities = torch.nn.functional.cosine_similarity(batch_a.unsqueeze(1), memory_features.unsqueeze(0), dim=2)
                top_indices = cosine_similarities.argsort(dim=1, descending=True)[:, :top_k]
                outall.append(top_indices)
            outall = torch.cat(outall, dim=0)
        
        rates = cosine_similarities[0][outall[0]].mean(0)
        weight = rates * torch.exp(cosine_similarities[0][outall[0]]) / torch.sum(torch.exp(cosine_similarities[0][outall[0]]))
        
        return memory_features[outall[:, 0]], x, weight


class Prototype_Pool(nn.Module):
    def __init__(self, delta=0.1, class_num=10, max=50):
        super(Prototype_Pool, self).__init__()
        self.class_num = class_num
        self.max_length = max
        self.delta = delta
        self.memory = []  # Use a list to store (feature, score) tuples

    def forward(self, x, top_k=5):
        mini_batch = 250
        outall = []
        
        if len(self.memory) == 0:
            return None, x, None
        
        memory_features = torch.stack([m[0] for m in self.memory])
        
        if x.shape[0] < mini_batch:
            cosine_similarities = torch.nn.functional.cosine_similarity(x.unsqueeze(1), memory_features.unsqueeze(0), dim=2)
            if memory_features.shape[0] >= top_k:
                outall = cosine_similarities.argsort(dim=1, descending=True)[:, :top_k]
            else:
                outall = cosine_similarities.argsort(dim=1, descending=True)[:, :1]
                outall = outall.repeat(1, top_k)
        else:
            for i in range(0, len(x), mini_batch):
                batch_a = x[i:i+mini_batch]
                cosine_similarities = torch.nn.functional.cosine_similarity(batch_a.unsqueeze(1), memory_features.unsqueeze(0), dim=2)
                top_indices = cosine_similarities.argsort(dim=1, descending=True)[:, :top_k]
                outall.append(top_indices)
            outall = torch.cat(outall, dim=0)
        
        rates = cosine_similarities[0][outall[0]].mean(0)
        weight = rates * torch.exp(cosine_similarities[0][outall[0]]) / torch.sum(torch.exp(cosine_similarities[0][outall[0]]))
        
        return memory_features[outall[:, 0]], x, weight

    def update_pool(self, features, scores):
        batch_size = len(features)
        top_k = max(1, int(batch_size // 2))
        fine_idx = []
        
        # Sort the scores list and get the indices of the top half
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i])
        top_indices = sorted_indices#[:top_k]  # Select the top half indices

        # Add the top half of the batch to the memory
        for idx in top_indices:
            if len(self.memory) < self.max_length:
                self.memory.append((features[idx].detach(), scores[idx]))
            else:
                # Find the feature with the worst score
                worst_index = max(range(len(self.memory)), key=lambda i: self.memory[i][1])
                if scores[idx] < self.memory[worst_index][1]:
                    # Replace the worst feature with the new one if the new one has a better score
                    self.memory[worst_index] = (features[idx].detach(), scores[idx])
                elif scores[idx] > self.memory[worst_index][1] and scores[idx] < self.memory[worst_index][1]+0.004:
                    fine_idx.append(idx)
        return fine_idx
                # Replace the earliest entry in the memory (FIFO)
                # self.memory.pop(0)  # Remove the first element (earliest added)
                # self.memory.append((features[idx].detach(), scores[idx]))  # Add the new feature




