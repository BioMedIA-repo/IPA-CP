import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


def make_one_hot(labels, classes):
    one_hot = torch.cuda.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3],
                                     labels.size()[4]).zero_()
    target = one_hot.scatter_(1, labels.data, 1)
    return target



class DiceLoss(nn.Module):
    def __init__(self, smooth=1., ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))
        return loss


class CE_DiceLoss(nn.Module):
    def __init__(self):
        super(CE_DiceLoss, self).__init__()
        self.dice = DiceLoss(smooth=1e-5)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        return CE_loss + dice_loss


class EntropyLoss(nn.Module):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w x d
        output: batch_size x 1 x h x w x d
    """

    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, v):
        assert v.dim() == 5
        n, c, h, w, d = v.size()
        return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * d * np.log2(c))
