import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import *
from medpy import metric

# def iou_score(output, target):
#     num = target.size(0)
#     if torch.is_tensor(output):
#         output = torch.sigmoid(output).view(num, -1).data.cpu().numpy()
#     if torch.is_tensor(target):
#         target = target.view(num, -1).data.cpu().numpy()
#     smooth = 1e-5
#     tp = (target * output)
#     fp = ((1 - target) * output)
#     fn = (target * (1 - output))
#     iou = (tp.sum(1) + smooth) / (fp.sum(1) + tp.sum(1) + fn.sum(1) + smooth)
#     iou = iou.sum() / num
#     return iou

def iou_score(output, target):
    smooth = 1e-5
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    tp = (target * output)
    fp = ((1 - target) * output)
    fn = (target * (1 - output))
    iou = (tp.sum() + smooth) / (fp.sum() + tp.sum() + fn.sum() + smooth)

    return iou
# def dice_coef(output, target):
#     num = target.size(0)
#     if torch.is_tensor(output):
#         output = torch.sigmoid(output).view(num, -1).data.cpu().numpy()
#     a = output.shape
#     if torch.is_tensor(target):
#         target = target.view(num, -1).data.cpu().numpy()
#     b = target.shape
#     smooth = 1e-5
#     tp = (target * output)
#     fp = ((1 - target) * output)
#     fn = (target * (1 - output))
#     dice = (2 * tp.sum(1) + smooth) / (target.sum(1) + output.sum(1) + smooth)
#     dice = dice.sum() / num
#     return dice
def dice_coef(output, target):
    smooth = 1e-5
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)
# def dice_coef(output, target):
#     smooth = 1e-5
#     output = torch.sigmoid(output).view(-1).data.cpu().numpy()
#     target = target.view(-1).data.cpu().numpy()
#     #intersection = (output * target).sum()
#     output[output > 0] = 1
#     target[target > 0] = 1
#     dice = metric.binary.dc(output, target)
#     return dice
def sensitivity(output, target):
    num = target.size(0)
    if torch.is_tensor(output):
        output = torch.sigmoid(output).view(num, -1).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.view(num, -1).data.cpu().numpy()
    smooth = 1e-5
    tp = (target * output)
    fp = ((1 - target) * output)
    fn = (target * (1 - output))
    Recall = (tp.sum(1) + smooth) / (tp.sum(1) + fn.sum(1) + smooth)
    Recall = Recall.sum() / num
    return Recall
def ppv(output, target):
    num = target.size(0)
    if torch.is_tensor(output):
        output = torch.sigmoid(output).view(num, -1).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.view(num, -1).data.cpu().numpy()
    smooth = 1e-5
    tp = (target * output)
    fp = ((1 - target) * output)
    fn = (target * (1 - output))
    Precision = (tp.sum(1) + smooth) / (tp.sum(1) + fp.sum(1) + smooth)
    Precision = Precision.sum() / num
    return Precision
#intersection TP
# def dice_coef(output, target):
#     smooth = 1.
#     N = target.size(0)
#     pred_flat = torch.sigmoid(output).view(N, -1).data.cpu().numpy()
#     gt_flat = target.view(N, -1).data.cpu().numpy()
#
#     intersection = (pred_flat * gt_flat).sum(1)
#     union = pred_flat.sum(1) + gt_flat.sum(1)
#     c = ((2. * intersection + smooth) / (union + smooth)).sum()
#     return c / N
# def iou_score(SR, GT):
#     # DC : Dice Coefficient
#     # threshold = 0.5
#     # SR = SR > threshold
#     # GT = GT == torch.max(GT)
#     # Inter = torch.sum((SR + GT) == 2)
#     # # Inter = torch.sum(SR * GT)
#     # DC = float(2 * Inter + 1e-5) / (float(torch.sum(SR) + torch.sum(GT)) + 1e-5)
#     iou = dice_coef(SR, GT) / 2 - dice_coef(SR, GT)
#     return iou
# def dice_coef(SR, GT):
#     # DC : Dice Coefficient
#     threshold = 0.5
#     SR = SR > threshold
#     GT = GT == torch.max(GT)
#     Inter = torch.sum((SR + GT) == 2)
#     # Inter = torch.sum(SR * GT)
#     DC = float(2 * Inter + 1e-5) / (float(torch.sum(SR) + torch.sum(GT)) + 1e-5)
#     return DC
# def sensitivity(SR, GT):
#     # DC : Dice Coefficient
#     threshold = 0.5
#     SR = SR > threshold
#     GT = GT == torch.max(GT)
#     # TP : True Positive
#     # FN : False Negative
#     TP = ((SR == 1) + (GT == 1)) == 2
#     FN = ((SR == 0) + (GT == 1)) == 2
#     SE = float(torch.sum(TP) + 1e-6) / (float(torch.sum(TP + FN)) + 1e-6)
#
#     return SE
# def ppv(SR, GT):
#     threshold = 0.5
#     SR = SR > threshold
#     GT = GT == torch.max(GT)
#     # TP : True Positive
#     # FP : False Positive
#     TP = ((SR == 1) + (GT == 1)) == 2
#     FP = ((SR == 1) + (GT == 0)) == 2
#     PC = float(torch.sum(TP) + 1e-6) / (float(torch.sum(TP + FP)) + 1e-6)
#     return PC
def tp(output, target):
    num = target.size(0)
    if torch.is_tensor(output):
        output = torch.sigmoid(output).view(num, -1).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.view(num, -1).data.cpu().numpy()
    smooth = 1e-5
    tp = (target * output)
    return tp.sum()/ num
def tn(output, target):
    num = target.size(0)
    if torch.is_tensor(output):
        output = torch.sigmoid(output).view(num, -1).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.view(num, -1).data.cpu().numpy()
    smooth = 1e-5
    tn = ((1 - target) * (1 - output))
    return tn.sum()/ num
def fp(output, target):
    num = target.size(0)
    if torch.is_tensor(output):
        output = torch.sigmoid(output).view(num, -1).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.view(num, -1).data.cpu().numpy()
    smooth = 1e-5
    fp = ((1 - target) * output)
    return fp.sum()/ num
def fn(output, target):
    num = target.size(0)
    if torch.is_tensor(output):
        output = torch.sigmoid(output).view(num, -1).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.view(num, -1).data.cpu().numpy()
    smooth = 1e-5
    fn = (target * (1 - output))
    return fn.sum() / num


def accuracy(output, target):
    num = target.size(0)
    if torch.is_tensor(output):
        output = torch.sigmoid(output).view(num, -1).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.view(num, -1).data.cpu().numpy()
    smooth = 1e-5
    tp = (target * output)
    tn = ((1 - target) * (1 - output))
    fp = ((1 - target) * output)
    fn = (target * (1 - output))
    acc = (tp.sum(1) + tn.sum(1) + smooth) / (tn.sum(1) + tp.sum(1) + fn.sum(1) + fp.sum(1) + smooth)
    acc = acc.sum() / num
    return acc