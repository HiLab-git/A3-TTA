import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import pandas as pd
import itertools
from torch.utils.data.sampler import Sampler
import SimpleITK as sitk
from PIL import Image
import albumentations as A
from torchvision import transforms
from scipy import ndimage
import torch.nn.functional as F

def load_image_as_nd_array(image_name):
    spacing = (1.0, 1.0, 1.0)
    if (image_name.endswith(".nii.gz") or image_name.endswith(".nii") or
        image_name.endswith(".mha")):
        img_obj    = sitk.ReadImage(image_name)
        spacing = img_obj.GetSpacing()
        data_array = sitk.GetArrayFromImage(img_obj)
 
    elif(image_name.endswith(".jpg") or image_name.endswith(".jpeg") or
         image_name.endswith(".tif") or image_name.endswith(".png")):
        data_array = np.asarray(Image.open(image_name))
    else:
        raise ValueError("unsupported image format")
    return data_array,spacing

class NiftyDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.csv_items  = pd.read_csv(csv_file)
        self.transform = transform
    def __len__(self):
        return len(self.csv_items)

    def __getlabel__(self, idx):
        csv_keys = list(self.csv_items.keys())
        label_idx = csv_keys.index('label')
        label_name = self.csv_items.iloc[idx, label_idx]
        label,_ = load_image_as_nd_array(label_name)
        label = label.copy()
        if len(label.shape) == 2:
            label = np.expand_dims(label, axis=0)
        label = np.asarray(label, np.int32)
        return label
   

    def __getitem__(self, idx):
        names_list, image_list = [], []

        image_name = self.csv_items.iloc[idx, 0]
        image_data,spacing = load_image_as_nd_array(image_name)
        names_list.append(image_name)
        image_list.append(image_data)
        image = np.concatenate(image_list, axis = 0)
        image = np.asarray(image, np.float32)    
        sample = {'image': image, 'names' : names_list[0]}
        sample['label'] = self.__getlabel__(idx) 
        if image.shape[-1] == 3:
            image = image.transpose(2,0,1)
            sample['image'] = image
        if self.transform:
            sample = self.transform(sample)
        sample['names'] = names_list[0] 
        sample['spacing'] = (spacing[2], spacing[1], spacing[0])
        
        return sample
def get_dataset(dataset, domain, online = False):
    if '2d' or 'Fundus' or 'fb' in dataset:
        transform_train = transforms.Compose([
                            ToTensor(),
                            ])
        transform_valid = transforms.Compose([
                            ToTensor(),
                            ])
        transform_test = transforms.Compose([
                            ToTensor(),
                            ])
    elif '3d' in dataset:
        transform_train = transforms.Compose([
                            ToTensor(),
                            ])
        transform_valid = transforms.Compose([
                            ToTensor(),
                            ]),
        transform_test = transforms.Compose([
                            ToTensor(),
                            ])
    db_train,db_valid,db_test = dataset_all(
        base_dir='./data',
        dataset=dataset,
        target=domain,
        transform_train = transform_train,
        transform_valid = transform_valid,
        transform_test = transform_test,
        online = online)
    return db_train,db_valid,db_test

def dataset_all(base_dir=None, dataset='fb',target='A',transform_train=None,transform_valid=None,transform_test=None,online=False):
    _base_dir = base_dir
    if online:
        all_file = os.path.join(_base_dir,dataset,target,'all.csv')
        if dataset == 'Fundus':
            all_dataset  = NiftyDataset(csv_file = all_file,transform=transform_train)
        else:
            all_dataset  = NiftyDataset(csv_file = all_file,transform=transform_train)
        return all_dataset,None,None
    else:
        train_file = os.path.join(_base_dir,dataset,target,'train.csv')
        valid_file = os.path.join(_base_dir,dataset,target,'valid.csv')
        test_file = os.path.join(_base_dir,dataset,target,'test.csv')
        train_dataset  = NiftyDataset(csv_file  = train_file,transform=transform_train)
        valid_dataset  = NiftyDataset(csv_file  = valid_file,transform=transform_valid)
        test_dataset  = NiftyDataset(csv_file  = test_file,transform=transform_test)
        return train_dataset,valid_dataset,test_dataset

class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, depth = False):
        self.output_size = output_size
        self.depth = depth

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])
        if self.depth:
            label = label[w1:w1 + self.output_size[0], :, :]
            image = image[w1:w1 + self.output_size[0], :, :]
        else:
            label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image': image, 'label': label}

class Scale_img(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        c,h,w = image.shape
        cc,hh,ww = self.output_size
        zoom = [1,hh/h,ww/w]
        image = ndimage.zoom(image,zoom,order=2)

        return {'image': image, 'label': label}
class Scale_imglab(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, depth = True):
        self.output_size = output_size
        self.depth = depth
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        c,h,w = image.shape
        cc,hh,ww = self.output_size
        zoom = [1,hh/h,ww/w]
        image = ndimage.zoom(image,zoom,order=2)
        label = ndimage.zoom(label,zoom,order=0)
        return {'image': image, 'label': label}
class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k, axes=(1,2))
        label = np.rot90(label, k, axes=(1,2))
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label,'onehot_label':onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        if len(sample['image'].shape) == 2:
            image = (torch.from_numpy(sample['image'])).unsqueeze(0)
            label = (torch.from_numpy(sample['label'])).unsqueeze(0).long()
        elif len(sample['image'].shape) == 3:
            image = (torch.from_numpy(sample['image']))
            label = (torch.from_numpy(sample['label'])).long()
        if 'onehot_label' in sample:
            return {'image': image, 'label': label,
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': image, 'label': label}

class AddGaussianNoise(object):
    def __init__(self, sigma=0.03):
        self.sigma = sigma

    def __call__(self, sample):
        image = sample['image']
        if isinstance(image, torch.Tensor):
            noise = torch.randn_like(image) * self.sigma
            noisy = torch.clamp(image + noise, -1.0, 1.0)
        else:
            noise = np.random.normal(0, self.sigma, size=image.shape)
            noisy = np.clip(image + noise, -1.0, 1.0)
            noisy = torch.from_numpy(noisy).float()
        sample['image'] = noisy
        return sample


# -------- Rician --------
class AddRicianNoise(object):
    def __init__(self, sigma=0.03):
        self.sigma = sigma

    def __call__(self, sample):
        image = sample['image']
        if isinstance(image, torch.Tensor):
            img01 = (image + 1.0) / 2.0  # [-1,1] → [0,1]
            n1 = torch.randn_like(img01) * self.sigma
            n2 = torch.randn_like(img01) * self.sigma
            noisy01 = torch.sqrt((img01 + n1) ** 2 + n2 ** 2)
            noisy01 = torch.clamp(noisy01, 0.0, 1.0)
            noisy = (noisy01 * 2.0) - 1.0  # [0,1] → [-1,1]

        else:
            n1 = np.random.normal(0, self.sigma, size=image.shape)
            n2 = np.random.normal(0, self.sigma, size=image.shape)
            noisy = np.sqrt((image + n1) ** 2 + n2 ** 2)
            noisy = np.clip(noisy, -1.0, 1.0)
            noisy = torch.from_numpy(noisy).float()
        sample['image'] = noisy
        return sample



class AddMotionBlur(object):
    def __init__(self, k=11, axis=-1):
        self.k = k
        self.axis = axis

    def __call__(self, sample):
        image = sample['image']
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()

        if image.dim() == 2:   # H,W -> C,H,W
            image = image.unsqueeze(0)
        elif image.dim() != 3:
            raise ValueError("Unsupported image shape for MotionBlur")

        img = image.unsqueeze(0)  # (1,C,H,W)
        C = img.size(1)

        if self.axis == -1:  # 水平
            kernel = torch.ones((1, 1, 1, self.k), device=img.device) / self.k
            padding = (0, self.k // 2)   # (pad_h, pad_w) ← 修正这里
        else:  # 垂直
            kernel = torch.ones((1, 1, self.k, 1), device=img.device) / self.k
            padding = (self.k // 2, 0)   # (pad_h, pad_w)

        kernel = kernel.expand(C, 1, *kernel.shape[2:])
        blurred = F.conv2d(img, kernel, padding=padding, groups=C)
        sample['image'] = blurred.squeeze(0)  # (C,H,W)
        return sample



class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

def scale(img,target_size=[1,1,320,416]):
    if len(img.shape) == 3:
        b,h,w = img.shape
    elif len(img.shape) == 4:
        b,c,h,w = img.shape
    bb,cc,hh,ww = target_size
    zoom = [1,hh/h,ww/w]
    img = ndimage.zoom(img,zoom,order=0)
    return img
def scal_spacing(img,lab, origin_spacing = [1,1,1]):
    # zoom = (origin_spacing[0].numpy()[0],origin_spacing[1].numpy()[0],origin_spacing[2].numpy()[0])
    # print(type(img),type(lab),type(zoom[0]),zoom)
    img = ndimage.zoom(img,zoom=origin_spacing,order=0)
    lab = ndimage.zoom(lab,zoom=origin_spacing,order=0)
    return img,lab
# def scal_spacing(img,lab,origin_spacing, target_spacing=[1, 1, 1], order=0):
#     # scale = np.array(origin_spacing) / np.array(target_spacing)
#     scale = []
#     # scale[0],scale[1],scale[2] = origin_spacing[0].numpy(),origin_spacing[1].numpy(),origin_spacing[2].numpy()
#     scale.extend([origin_spacing[0].numpy().astype(np.float),origin_spacing[1].numpy().astype(np.float),origin_spacing[2].numpy().astype(np.float)])
#     # print(origin_spacing[0].numpy(),'***',np.array(origin_spacing),'314')
#     # print(type(img),type(lab),type(scale),scale,'***',type(origin_spacing),type(target_spacing))
#     img = ndimage.zoom(img.astype(np.float), zoom=list(scale), order=order)
#     lab = ndimage.zoom(lab.astype(np.float), zoom=list(scale), order=order)
#     if order == 0:
#         img = img.astype(np.uint8)
#         lab = lab.astype(np.uint8)
#     else:
#         img = img.astype(np.float)
#         lab = lab.astype(np.float)
#         # jianghao hi nicaiwoshishei? ?  nicai ?ljs zaigeinidasaoweisheng,:) 好好打扫 niyeshiwomenzude 你，走了 我们少了人 请我们吃饭 ：不是SI）组多一个？喊他来 书伟也走了，辛苦了很辛苦
#         # 没关系，请我们吃饭就好（你猜猜我是谁？【笑   
#         # 请吃饭次数+1+1，等我回来X_X
        
#     return img,lab

def convert_2d(img = None,lab = None):
    x_shape = list(img.shape)
    if(len(x_shape) == 5):
        [N, C, D, H, W] = x_shape
        new_shape = [N*D, C, H, W]
        img = torch.transpose(img, 1, 2)
        img = torch.reshape(img, new_shape)
        if lab.shape == img.shape:
            lab = torch.transpose(lab, 1, 2)
            lab = torch.reshape(lab, new_shape)
        else:
            [N, C, D, H, W] = list(lab.shape)
            new_shape = [N*D, C, H, W]
            lab = torch.transpose(lab, 1, 2)
            lab = torch.reshape(lab, new_shape)
            lab = torch.transpose(lab, 1, 2)
            lab = torch.reshape(lab, new_shape)
            
    return img,lab