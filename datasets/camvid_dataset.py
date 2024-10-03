# -*- coding: utf-8 -*-

import os

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as ff
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd


class LabelProcessor:
    """对标签图像的编码"""
    def __init__(self, file_path):
        self.colormap = self.read_color_map(file_path)  # 读取csv，获得label对应的rgb值, csv --> rgb
        self.cm2lbl = self.encode_label_pix(self.colormap)  # 对label的rgb值制作映射矩阵，rgb --> cls index
        self.names = pd.read_csv(file_path, sep=',').name.tolist()
    # 静态方法装饰器， 可以理解为定义在类中的普通函数，可以用self.<name>方式调用
    # 在静态方法内部不可以示例属性和实列对象，即不可以调用self.相关的内容
    # 使用静态方法的原因之一是程序设计的需要（简洁代码，封装功能等）
    @staticmethod
    def read_color_map(file_path):  # data process and load.ipynb: 处理标签文件中colormap的数据
        """
        读取csv中信息，获得各类别标签的rgb像素，以list形式返回
        :param file_path:
        :return: list， list[0] == [128, 128, 128] ...
        """
        pd_label_color = pd.read_csv(file_path, sep=',')
        colormap = []
        for i in range(len(pd_label_color.index)):
            tmp = pd_label_color.iloc[i]
            color = [tmp['r'], tmp['g'], tmp['b']]
            colormap.append(color)
        return colormap

    @staticmethod
    def encode_label_pix(colormap):
        """
        生成标签编码，返回哈希表
        key是像素值的编码，编码公式：(cm[0] * 256 + cm[1]) * 256 + cm[2]
        value是类别，0,1,2,3,...,11
        :param colormap:
        :return: ndarray, shape =  (16777216,)
        """
        cm2lbl = np.zeros(256 ** 3)
        for i, cm in enumerate(colormap):
            cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
        return cm2lbl

    def encode_label_img(self, img):
        """
        将rgb像素转换为 0-11 的标签形式
        :param img:
        :return:
        """
        data = np.array(img, dtype='int32')
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return np.array(self.cm2lbl[idx], dtype='int64')


class CamvidDataset(Dataset):
    def __init__(self, dir_img, dir_l, path_csv, crop_size=None):
        """para:
            file_path(list): 数据和标签路径,列表元素第一个为图片路径，第二个为标签路径
        """
        self.img_path = dir_img
        self.label_path = dir_l
        # 2 从路径中取出图片和标签数据的文件名保持到两个列表当中（程序中的数据来源）
        self.imgs = self.read_file(self.img_path)
        self.labels = self.read_file(self.label_path)
        # 3 初始化数据处理函数设置
        self.crop_size = crop_size
        # 初始化标签处理器
        self.label_processor = LabelProcessor(path_csv)
        self.names = pd.read_csv(path_csv, sep=',').name.tolist()
        self.cls_num = len(self.names)
        # 统计类别数量
        self.nums_per_cls = self.cal_cls_nums()

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]

        img = Image.open(img)
        label = Image.open(label).convert('RGB')

        img, label = self.center_crop(img, label, self.crop_size)
        img, label = self.img_transform(img, label)
        return img, label

    def __len__(self):
        if len(self.imgs) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(
                self.dir_img))   # 代码具有友好的提示功能，便于debug
        return len(self.imgs)

    def read_file(self, path):
        """从文件夹中读取数据"""
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list

    def center_crop(self, data, label, crop_size):
        """裁剪输入的图片和标签大小"""
        data = ff.center_crop(data, crop_size)
        label = ff.center_crop(label, crop_size)
        return data, label

    def img_transform(self, img, label):
        """对图片和标签做一些数值处理"""
        label = np.array(label)  # 以防不是np格式的数据
        label = Image.fromarray(label.astype('uint8'))
        transform_img = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        img = transform_img(img)
        label = self.label_processor.encode_label_img(label)  # RGB np.array -->   label index
        label = torch.from_numpy(label)

        return img, label

    def cal_cls_nums(self):
        counter = np.zeros((12,))
        for path in self.labels:
            label_img = Image.open(path).convert('RGB')
            label = self.label_processor.encode_label_img(label_img).flatten()
            count = np.bincount(label, minlength=12)
            counter += count
        return counter.tolist()


if __name__ == "__main__":

    camvid_dir = r"G:\deep_learning_data\camvid_from_paper"
    path_to_dict = os.path.join(camvid_dir, 'class_dict.csv')
    TRAIN_ROOT = os.path.join(camvid_dir, 'train')
    TRAIN_LABEL = os.path.join(camvid_dir, 'train_labels')

    crop_size = (360, 480)
    train_data = CamvidDataset(TRAIN_ROOT, TRAIN_LABEL, path_to_dict, crop_size)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0)

    for i, data in enumerate(train_loader):
        img_data, img_label = data
        print(img_data.shape, img_data.dtype, type(img_data))       # torch.float32
        print(img_label.shape, img_label.dtype, type(img_label))     # torch.longint(int64)


