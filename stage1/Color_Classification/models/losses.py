import torch 
from torchvision import transforms

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from lib.net_util import *
from utils.parser import * 
from models.resnet import resnet50

opts = parse_opts()

def pixel_loss(conv_feat, mask, label, one_hot, model):

    batch_total = 0
    correct = 0
    batch_loss = 0

    # Iterating through batches
    for b in range(conv_feat.shape[0]):
        # Iterating through pixels embeddings
        for i in range(conv_feat.shape[-2]):
            for j in range(conv_feat.shape[-1]):
                # Penalize the pixels of interests
                if mask[b, 0, i, j] != 0:
                    pixel_feat = conv_feat[b, :, i, j]
                    pixel_feat = pixel_feat.contiguous().view(1, -1)
                    prediction = model.fc(pixel_feat)
                    batch_loss += opts.criterion[1](torch.squeeze(prediction), one_hot[b])
                    # Computing the precision
                    _, predicted = torch.max(prediction.data, 1)
                    correct += predicted.eq(label[b].data).cpu().sum()
                    batch_total += 1
    batch_loss /= batch_total

    return batch_loss, batch_total, correct