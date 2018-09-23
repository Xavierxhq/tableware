from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import torch

from utils import loss

class Evaluator(object):
    def __init__(self, model):
        self.model = model

    def evaluate(self, data_loader, margin, save_file_name):
        f = open(save_file_name, 'w+')
        count = 0.0
        summ = 0.0
        for i, inputs in enumerate(data_loader):
            imgs, pids = inputs
            data = imgs.cuda()
            with torch.no_grad():
                feature = self.model(data)
            feature.cpu()

            distmat = loss.euclidean_dist(feature, feature).cpu()
            for i, label_i in enumerate(pids):
                for j, label_j in enumerate(pids):

                    f.write('{}, {} :{}\r\n'.format(label_i, label_j, distmat[i, j]))

                    if distmat[i,j] <= margin and label_i == label_j:
                        summ = summ + 1.0
                    elif distmat[i,j] > margin and label_i != label_j:
                        summ = summ + 1.0
                    count = count + 1.0
        f.close()
        return summ / count
