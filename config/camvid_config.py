# -*- coding: utf-8 -*-

import torch
from easydict import EasyDict

cfg = EasyDict()  # 访问属性的方式去使用key-value 即通过 .key获得value

cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.max_epoch = 150  # 150

cfg.crop_size = (360, 480)

# 梯度裁剪
cfg.hist_grad = True
cfg.is_clip = False
cfg.clip_value = 0.2

# batch size
cfg.train_bs = 8  # 8
cfg.valid_bs = 4    # 4
cfg.workers = 16  # 16

# 学习率
cfg.lr_init = 0.1  # pretraied_model::0.1
cfg.factor = 0.1
cfg.milestones = [75, 130]
cfg.weight_decay = 1e-4
cfg.momentum = 0.9

cfg.log_interval = 10






