import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lovasz_losses import lovasz_hinge

__all__ = ['DiceLoss', 'BCEDiceLoss', 'LovaszHingeLoss', 'tversky_loss']

from torch.autograd import Variable


# I refered https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py

# class DiceLoss(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, pred, target):
#
# # pred = pred.squeeze(dim=1)
#
#         smooth = 1
#
#         dice = 0.
#         # dice系数的定义
#         for i in range(pred.size(1)):
#             dice += 2 * (pred[:,i] * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1) / (pred[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
#             target[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)
#         # 返回的是dice距离
#         dice = dice / pred.size(1)
#         return torch.clamp((1 - dice).mean(), 0, 1)
#
# class BCEDiceLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, input, target):
#         bce = F.binary_cross_entropy_with_logits(input, target)
#         smooth = 1e-5
#         input = torch.sigmoid(input)
#         num = target.size(0)
#         input = input.view(num, -1)
#         target = target.view(num, -1)
#         intersection = (input * target)
#         dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
#         dice = 1 - dice.sum() / num
#
#         return 0.8 * bce + 0.2 * dice


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return dice
#需要更改
class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, predict, target):
        pt_ = torch.sigmoid(predict) # sigmoide获取概率
        num = target.size(0)
        pt = pt_.view(num, -1)
        target = target.view(num, -1)
        #在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt)\
               - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        loss = loss.sum()/num
        return loss


def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1
    bin_label_weights = label_weights.view(-1, 1).expand(label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights

class GHMC(nn.Module):
    """

    Args:
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        use_sigmoid (bool): Can only be true for BCE based loss now.
        loss_weight (float): The weight of the total GHM-C loss.
    """

    def __init__(self, bins=10, momentum=0, loss_weight=1.0):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        edges = torch.arange(bins + 1).float() / bins
        self.register_buffer('edges', edges)
        self.edges[-1] += 1e-6
        if momentum > 0:
            acc_sum = torch.zeros(bins)
            self.register_buffer('acc_sum', acc_sum)
        self.loss_weight = loss_weight



    def forward(self, pred, target, label_weight=None, *args, **kwargs):
        """Calculate the GHM loss.

        Args:
            pred (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            target (float tensor of size [batch_num, class_num]):
                Binary class target for each sample.
            label_weight (float tensor of size [batch_num, class_num]):
                the value is 1 if the sample is valid and 0 if ignored.
        Returns:
            The gradient harmonized loss.
        """
        # the target should be binary class label
        pred = torch.sigmoid(pred)
        num = target.size(0)
        pred = pred.view(num, -1)
        target = target.view(num, -1)
        if pred.dim() != target.dim():
            target, label_weight = _expand_binary_labels(
            target, label_weight, pred.size(-1))

        label_weight = torch.ones((target.size(0),target.size(1))).cuda()

        target, label_weight = target.float(), label_weight.float()
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred).cuda()

        # gradient length
        g = torch.abs(pred.detach() - target).cuda()
        # valid label center
        valid = label_weight > 0
        # valid label number
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            # Divide the corresponding gradient value into the corresponding bin, 0-1
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            # samples exist in the bin
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    # moment calculate num bin
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                                      + (1 - mmt) * num_in_bin
                    # weights/num bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            # scale
            weights = weights / n

        loss = F.binary_cross_entropy_with_logits(
            pred, target, weights, reduction='sum') / tot

        return loss

class ssloss(nn.Module):
    def __init__(self):
        super(ssloss, self).__init__()

    def forward(self, input, target):
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        tp = (input * target).sum()
        fp = ((1 - target) * input).sum()
        fn = (target * (1 - input)).sum()
        tn = ((1 - input) * (1 - target)).sum()
        seloss = fn / (tp + fn)
        sploss = fp / (tn + fp)
        loss = seloss + sploss

        return loss

class mix(nn.Module):
    def __init__(self):
        super(mix, self).__init__()
        self.loss1 = GHMC()
        self.loss2 = ssloss()

    def forward(self,input, target, alpha=0.7):
        loss = alpha * self.loss1(input,target) + (1-alpha) * self.loss2(input, target)
        return loss



class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num

        return 0.5 * bce + dice


class BCEDiceLoss_half(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num

        return 0.5 * bce + 0.5 * dice
class tversky_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, alpha = 0.3, weights = None):
        smooth = 1e-5
        batch_size = targets.size(0)
        prob = torch.sigmoid(inputs).view(batch_size, -1)
        ref = targets.view(batch_size, -1)
        beta = 1.0 - alpha
        intersection = (prob * ref)
        # tp = (ref*prob).sum(1)
        fp = ((1 - ref) * prob)
        fn = (ref * (1 - prob))
        tversky = (intersection.sum(1) + smooth) /\
                  (intersection.sum(1) + alpha * fp.sum(1) + beta * fn.sum(1) + smooth)
        loss = 1 - tversky.sum() / batch_size
        return loss
class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss

