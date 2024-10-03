# -*- coding: utf-8 -*-

import matplotlib
# matplotlib.use('agg')
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
import argparse
import torch
import numpy as np
from models.segnet import SegResNet
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--path_checkpoint', default=r"G:\project_class_bak\results\seg_baseline\07-29_11-08-camvid-baseline-basd-0.57\checkpoint_best.pkl",
                    help="path to your dataset")
parser.add_argument('--path_img', default=r"G:\deep_learning_data\camvid_from_paper\test\Seq05VD_f04050.png",
                    help="path to your dataset")
parser.add_argument('--path_csv', default=r"G:\deep_learning_data\camvid_from_paper\class_dict.csv",
                    help="path to your csv")
args = parser.parse_args()

if __name__ == '__main__':

    # 图片预处理定义
    transform_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = Image.open(args.path_img)
    img_tensor = transform_img(img)
    img_tensor.unsqueeze_(0)
    img_tensor = img_tensor.to(device)

    # 模型加载
    model = SegResNet(num_classes=12)
    checkpoint = torch.load(args.path_checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # 推理
    with torch.no_grad():
        outputs = model(img_tensor)
    outputs = torch.max(outputs, dim=1)
    pre_label = outputs[1].squeeze().cpu().data.numpy()

    # 结果展示
    pd_label_color = pd.read_csv(args.path_csv, sep=',')
    name_value = pd_label_color['name'].values
    num_class = len(name_value)
    colormap = []
    for i in range(num_class):
        tmp = pd_label_color.iloc[i]
        color = [tmp['r'], tmp['g'], tmp['b']]
        colormap.append(color)

    cm = np.array(colormap).astype('uint8')
    pre = cm[pre_label]
    pre1 = Image.fromarray(pre)

    plt.subplot(121)
    plt.imshow(pre1)
    plt.subplot(122)
    plt.imshow(img)
    plt.show()

    # 展示图例
    height = 20
    fake_img = np.zeros((height*12, 100, 3), dtype=np.uint8)
    for i in range(num_class):
        tmp = pd_label_color.iloc[i]
        color = [tmp['r'], tmp['g'], tmp['b']]
        fake_img[height*i:height*(i+1), :, :] = color

    plt.imshow(fake_img)

    for i in range(num_class):
        name = pd_label_color["name"][i]
        plt.text(10, 15 + height*i, name)
    plt.show()



