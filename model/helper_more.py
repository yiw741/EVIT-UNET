import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn as nn

class Poly(_LRScheduler):
    def __init__(self, optimizer, num_epochs, iters_per_epoch=0, warmup_epochs=0, last_epoch=-1):
        self.iters_per_epoch = iters_per_epoch
        self.cur_iter = 0
        self.N = num_epochs * iters_per_epoch
        self.warmup_iters = warmup_epochs * iters_per_epoch
        super(Poly, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        T = self.last_epoch * self.iters_per_epoch + self.cur_iter
        factor = pow((1 - 1.0 * T / self.N), 0.9)
        if self.warmup_iters > 0 and T < self.warmup_iters:
            factor = 1.0 * T / self.warmup_iters

        self.cur_iter %= self.iters_per_epoch
        self.cur_iter += 1
        return [base_lr * factor for base_lr in self.base_lrs]

def eval_metrics(output, target, num_classes=21, ignore_index=255):
    """
    多分类语义分割评估指标

    Args:
        output: 模型输出 (B, C, H, W)
        target: 真实标签 (B, H, W)
        num_classes: 类别数（VOC默认为21）
        ignore_index: 忽略的类别索引

    Returns:
        [正确像素数, 总有效像素数, 每个类别交集, 每个类别并集]
    """
    # 预测结果
    predict = torch.argmax(output, dim=1)

    # 创建有效像素掩码（排除忽略的类别）
    valid_mask = (target != ignore_index)

    # 计算像素准确率
    correct, num_labeled = batch_pix_accuracy(predict, target, valid_mask)

    # 计算交集和并集
    inter, union = batch_intersection_union(predict, target, num_classes, valid_mask)

    return [
        np.round(correct, 5),
        np.round(num_labeled, 5),
        np.round(inter, 5),
        np.round(union, 5)
    ]

def batch_pix_accuracy(predict, target, valid_mask):
    """
    计算像素准确率

    Args:
        predict: 预测结果
        target: 真实标签
        valid_mask: 有效像素掩码

    Returns:
        正确像素数, 总有效像素数
    """
    pixel_labeled = valid_mask.sum()
    pixel_correct = ((predict == target) * valid_mask).sum()

    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"

    return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()

def batch_intersection_union(predict, target, num_classes, valid_mask):
    """
    计算每个类别的交集和并集

    Args:
        predict: 预测结果
        target: 真实标签
        num_classes: 类别数
        valid_mask: 有效像素掩码

    Returns:
        每个类别的交集, 每个类别的并集
    """
    # 对预测结果和目标应用有效掩码
    predict = predict * valid_mask.long()
    target = target * valid_mask.long()

    # 初始化交集、预测区域和标签区域
    area_inter = torch.zeros(num_classes, dtype=torch.float)
    area_pred = torch.zeros(num_classes, dtype=torch.float)
    area_lab = torch.zeros(num_classes, dtype=torch.float)

    # 计算每个类别的交集、预测区域和标签区域
    for cls in range(num_classes):
        area_inter[cls] = ((predict == cls) & (target == cls)).sum()
        area_pred[cls] = (predict == cls).sum()
        area_lab[cls] = (target == cls).sum()

    # 计算并集
    area_union = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"

    return area_inter.cpu().numpy(), area_union.cpu().numpy()

def calculate_metrics(inter, union, num_classes):
    """
    计算常用的语义分割评价指标

    Args:
        inter: 每个类别的交集
        union: 每个类别的并集
        num_classes: 类别数

    Returns:
        字典，包含mIoU、Pixel Accuracy等指标
    """
    # 防止除零
    eps = 1e-10

    # 每个类别的IoU
    iou = inter / (union + eps)

    # 平均IoU (mIoU)
    miou = np.nanmean(iou)

    # 总体像素准确率
    pixel_acc = np.sum(inter) / (np.sum(union) + eps)

    # 类别平均准确率
    class_acc = []
    for i in range(num_classes):
        if union[i] > 0:
            class_acc.append(inter[i] / union[i])
        else:
            class_acc.append(np.nan)
    mean_class_acc = np.nanmean(class_acc)

    return {
        'mIoU': miou,
        'Pixel Accuracy': pixel_acc,
        'Mean Class Accuracy': mean_class_acc,
        'IoU': iou
    }

def initialize_weights(*models):
    """
    权重初始化函数
    """
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
