import os
from PIL import Image
import numpy as np
import torch
from torchvision.io import read_video, write_jpeg
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

__all__ = ('MVTecDataset', )

MVTEC_CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

class MVTecDataset(Dataset):
    def __init__(self, c, is_train=True):
        assert c.class_name in MVTEC_CLASS_NAMES, 'class_name: {}, should be in {}'.format(c.class_name, MVTEC_CLASS_NAMES)
        self.dataset_path = c.data_path
        #print("加载数据地址：",c.data_path)
        self.class_name = c.class_name
        self.is_train = is_train
        self.input_size = c.input_size
        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()

        # set transforms
        if is_train:
            self.transform_x = T.Compose([
                T.Resize(c.input_size, InterpolationMode.LANCZOS),
                T.ToTensor()])
        # test:
        else:
            self.transform_x = T.Compose([
                T.Resize(c.input_size, InterpolationMode.LANCZOS),
                T.ToTensor()])
        # mask
        self.transform_mask = T.Compose([
            T.Resize(c.input_size, InterpolationMode.NEAREST),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize(c.img_mean, c.img_std)])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        #x = Image.open(x).convert('RGB')
        x = Image.open(x)
        if self.class_name in ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                 'transistor', 'metal_nut', 'screw','toothbrush', 'zipper', 'tile', 'wood']:  # handle greyscale classes
            x = np.array(x)  # 转换为 numpy 数组
            if x.ndim == 2:  # 如果是二维（灰度图）
                x = np.expand_dims(np.array(x), axis=2)
                x = np.concatenate([x, x, x], axis=2)

            # 检查图像通道，确保为 RGB
            if x.ndim == 3 and x.shape[2] == 4:  # 判断是否为四通道图像
                x = x[:, :, :3]  # 覆盖前三个通道

            x = Image.fromarray(x.astype('uint8')).convert('RGB')
        #
        x = self.normalize(self.transform_x(x))
        #
        if y == 0:
            mask = torch.zeros([1, *self.input_size])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        #将lable和mask设为0
        y = torch.tensor(y)  # 将label转换为张量
        y = torch.zeros_like(y)
        mask=torch.zeros_like(mask)
        #print(f"图像的形状: {x.shape}")  # 输出每个图像的形状
        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')
        # print(f'当前加载数据阶段：{phase}')
        # print(f'图像目录: {img_dir}')
        # print(f'标注目录: {gt_dir}')
        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:
            #print(f'找到的图像类型: {img_types}')  # 打印发现的图像类型
            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            #print(f'当前正在检查的目录: {img_type_dir}')  # 打印当前检查的目录
            #if not os.path.isdir(img_type_dir):
                #continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)])
            #print(f'加载的测试样本数量: {len(img_fpath_list)}')  # 确认加载的图像数量
            # 提取图像名称并打印
            self.image_names = [os.path.basename(f) for f in img_fpath_list]
            x.extend(img_fpath_list)
            #print(f'图片列表: {len(img_fpath_list)}')
            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'
        # print(f'加载的样本数量: {len(x)}')  # 打印加载的样本数量
        # print(f'标签数量 (y): {len(y)}')
        # print(f'掩码数量 (mask): {len(mask)}')
        return list(x), list(y), list(mask)

def get_image_names(self):
    return self.image_names  # 添加获取 image_names 的方法

VISA_CLASS_NAMES = ['candle', 'capsules', 'cashew', 'chewinggum', 
                    'fryum', 'macaroni1', 'macaroni2', 
                    'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']

class VisADataset(Dataset):
    def __init__(self, c, is_train=True):
        assert c.class_name in VISA_CLASS_NAMES, 'class_name: {}, should be in {}'.format(c.class_name, MVTEC_CLASS_NAMES)
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.input_size = c.input_size
        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()
        # set transforms
        if is_train:
            self.transform_x = T.Compose([
                T.Resize(c.input_size, InterpolationMode.LANCZOS),
                T.ToTensor()])
        # test:
        else:
            self.transform_x = T.Compose([
                T.Resize(c.input_size, InterpolationMode.LANCZOS),
                T.ToTensor()])
        # mask
        self.transform_mask = T.Compose([
            T.Resize(c.input_size, InterpolationMode.NEAREST),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize(c.img_mean, c.img_std)])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        x = Image.open(x)
        x = self.normalize(self.transform_x(x))
        if y == 0:
            mask = torch.zeros([1, *self.input_size])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.png')
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)