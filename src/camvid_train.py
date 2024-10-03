# -*- coding: utf-8 -*-

import matplotlib
# matplotlib.use('agg')
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
import argparse
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.camvid_dataset import CamvidDataset
from tools.model_trainer import ModelTrainer
from tools.common_tools import *
from config.camvid_config import cfg
from models.unet import UNet, UNetResnet
from models.deeplabv3_plus import DeepLabV3Plus
from models.segnet import SegNet, SegResNet
from tools.evalution_segmentaion import calc_semantic_segmentation_iou
from datetime import datetime

setup_seed(12345)  # 先固定随机种子

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--lr', default=None, help='learning rate', type=float)
parser.add_argument('--max_epoch', default=None, type=int)
parser.add_argument('--train_bs', default=0, type=int)
parser.add_argument('--data_root_dir', default=r"G:\deep_learning_data\camvid_from_paper",
                    help="path to your dataset")
args = parser.parse_args()
cfg.lr_init = args.lr if args.lr else cfg.lr_init
cfg.train_bs = args.train_bs if args.train_bs else cfg.train_bs
cfg.max_epoch = args.max_epoch if args.max_epoch else cfg.max_epoch

if __name__ == '__main__':
    # step0: setting path
    path_model_101 = os.path.join(BASE_DIR, "..", "..", "data", "pretrained_model", "resnet101s-03a0f310.pth")  # deeplab
    path_model_50s = os.path.join(BASE_DIR, "..", "..", "data", "pretrained_model", "resnet50s-a75c83cf.pth")  # unet
    path_model_50 = os.path.join(BASE_DIR, "..", "..", "data", "pretrained_model", "resnet50-19c8e357.pth")  # segnet
    path_model_vgg = os.path.join(BASE_DIR, "..", "..", "data", "pretrained_model", "vgg16_bn-6c64b313.pth")
    res_dir = os.path.join(BASE_DIR, "..", "..", "results")
    logger, log_dir = make_logger(res_dir)

    # ------------------------------------ step 1/4 : 加载数据------------------------------------
    # 构建Dataset实例
    root_dir = args.data_root_dir
    train_img_dir = os.path.join(root_dir, 'train')
    train_lable_dir = os.path.join(root_dir, 'train_labels')
    valid_img_dir = os.path.join(root_dir, 'val')
    valid_label_dir = os.path.join(root_dir, 'val_labels')
    path_to_dict = os.path.join(root_dir, 'class_dict.csv')
    check_data_dir(train_img_dir)
    check_data_dir(train_lable_dir)
    check_data_dir(valid_img_dir)
    check_data_dir(valid_label_dir)

    train_data = CamvidDataset(train_img_dir, train_lable_dir, path_to_dict, cfg.crop_size)
    valid_data = CamvidDataset(valid_img_dir, valid_label_dir, path_to_dict, cfg.crop_size)
    train_loader = DataLoader(train_data, batch_size=cfg.train_bs, shuffle=True, num_workers=cfg.workers)
    valid_loader = DataLoader(valid_data, batch_size=cfg.valid_bs, num_workers=cfg.workers)

    # ------------------------------------ step 2/4 : 定义网络------------------------------------
    # model = DeepLabV3Plus(num_classes=train_data.cls_num, path_model=path_model_101)
    # model = UNet(num_classes=train_data.cls_num)
    # model = UNetResnet(num_classes=train_data.cls_num, backbone='resnet50', path_model=path_model_50s)
    # model = SegNet(num_classes=train_data.cls_num, path_model=path_model_vgg)
    model = SegResNet(num_classes=train_data.cls_num, path_model=path_model_50)
    model.to(cfg.device)

    # ------------------------------------ step 3/4 : 定义损失函数和优化器 ------------------------------------
    loss_f = nn.CrossEntropyLoss().to(cfg.device)
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr_init, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=cfg.factor, milestones=cfg.milestones)

    # ------------------------------------ step 4/4 : 训练 --------------------------------------------------
    # 记录训练配置信息
    logger.info("cfg:\n{}\n loss_f:\n{}\n scheduler:\n{}\n optimizer:\n{}\n model name:\n{}\nmodel:\n{}".format(
        cfg, loss_f, scheduler, optimizer, model._get_name(), model))

    loss_rec = {"train": [], "valid": []}
    acc_rec = {"train": [], "valid": []}
    miou_rec = {"train": [], "valid": []}
    best_miou, best_epoch = 0, 0
    grad_lst_epoch = []
    for epoch in range(cfg.max_epoch):

        loss_train, acc_train, mat_train, miou_train, grad_lst = ModelTrainer.train(
            train_loader, model, loss_f, cfg, optimizer, epoch, logger)
        loss_valid, acc_valid, mat_valid, miou_valid = ModelTrainer.valid(
            valid_loader, model, loss_f, cfg)

        grad_lst_epoch.extend(grad_lst)
        scheduler.step()

        logger.info("Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%}\n"
                    "Train loss:{:.4f} Train miou:{:.4f}\n"
                    "Valid loss:{:.4f} Valid miou:{:.4f}\n"
                    "LR:{}". format(epoch, cfg.max_epoch, acc_train, acc_valid, loss_train, miou_train,
                                    loss_valid, miou_valid, optimizer.param_groups[0]["lr"]))

        # 记录训练信息
        loss_rec["train"].append(loss_train), loss_rec["valid"].append(loss_valid)
        acc_rec["train"].append(acc_train), acc_rec["valid"].append(acc_valid)
        miou_rec["train"].append(miou_train), miou_rec["valid"].append(miou_valid)
        # 保存混淆矩阵图
        show_confMat(mat_train, train_data.names, "train", log_dir, epoch=epoch,
                     verbose=epoch == cfg.max_epoch - 1, perc=True)
        show_confMat(mat_valid, valid_data.names, "valid", log_dir, epoch=epoch,
                     verbose=epoch == cfg.max_epoch - 1, perc=True)
        # 保存loss曲线， acc曲线， miou曲线
        plt_x = np.arange(1, epoch + 2)
        plot_line(plt_x, loss_rec["train"], plt_x, loss_rec["valid"], mode="loss", out_dir=log_dir)
        plot_line(plt_x, acc_rec["train"], plt_x, acc_rec["valid"], mode="acc", out_dir=log_dir)
        plot_line(plt_x, miou_rec["train"], plt_x, miou_rec["valid"], mode="miou", out_dir=log_dir)
        # 保存模型
        if best_miou < miou_valid or epoch == cfg.max_epoch-1:

            best_epoch = epoch if best_miou < miou_valid else best_epoch
            best_miou = miou_valid if best_miou < miou_valid else best_miou
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch,
                          "best_miou": best_miou}
            pkl_name = "checkpoint_{}.pkl".format(epoch) if epoch == cfg.max_epoch-1 else "checkpoint_best.pkl"
            path_checkpoint = os.path.join(log_dir, pkl_name)
            torch.save(checkpoint, path_checkpoint)
            # 观察各类别的iou：
            iou_array = calc_semantic_segmentation_iou(mat_valid)
            info = ["{}_iou:{:.2f}".format(n, iou) for n, iou in zip(train_data.names, iou_array)]
            logger.info("Best mIoU in {}. {}".format(epoch, "\n".join(info)))

        if cfg.hist_grad:
            path_grad_png = os.path.join(log_dir, "grad_hist.png")
            logger.info("max grad in {}, is {}".format(grad_lst_epoch.index(max(grad_lst_epoch)), max(grad_lst_epoch)))
            import matplotlib.pyplot as plt

            plt.hist(grad_lst_epoch)
            plt.savefig(path_grad_png)
            logger.info(grad_lst_epoch)

    logger.info("{} done, best_miou: {:.4f} in :{}".format(
        datetime.strftime(datetime.now(), '%m-%d_%H-%M'), best_miou, best_epoch))



