from __future__ import print_function, division, absolute_import, unicode_literals

#import cv2
import glob
import numpy as np
from PIL import Image
import ipdb
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import scipy.misc
import torch


class BaseDataProvider(object):

    def __init__(self, a_min=None, a_max=None):
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf

    def _load_data_and_label(self):
        data, label, path = self._next_data()
        # data, label = self._augment_data(data, label) QUE HACE?
        # data, label = self._process_data_label(data, label) QUE HACE?

        nx = data.shape[1]
        ny = data.shape[0]

        data = np.expand_dims(data, axis=0)
        data = (np.transpose(data, (0, 3, 1, 2)))
        #print('Data Shape: ',data.shape)
        # print('Label Shape: ',label.shape)

        # return data.reshape(1, self.channels, ny, nx)
        return path, data, label.reshape(1, self.channels, ny, nx)

    def _process_data_label(self, data, label):
        for ich in range(self.channels):
            if np.amax(data[..., ich]) == 0:
                continue
            data[..., ich] -= float(np.amin(data[..., ich]))
            data[..., ich] /= float(np.amax(data[..., ich]))
        return data, label

    def _toTorchFloatTensor(self, img):
        img = torch.from_numpy(img.copy())
        return img

    def __call__(self, n):
        path, data, labels = self._load_data_and_label()
        nx = data.shape[2]
        ny = data.shape[3]
        X = torch.FloatTensor(n, self.channels, nx, ny).zero_()
        Y = torch.FloatTensor(n, 1, nx, ny).zero_()
        P = []

        #test1 = data[0, 0]
        #rarrg = data.shape[1]
        
        X[0] = self._toTorchFloatTensor(data[0])[0]
        # for ich in range(data.shape[1]):
        #    X[0, ich] = self._toTorchFloatTensor(data[0, ich])[0]
        Y[0, 0] = self._toTorchFloatTensor(labels[0, 0])[0]
        P.append(path)

        for i in range(1, n):
            if self.data_idx+1 >= self.n_data:
                break
            path, data, labels = self._load_data_and_label()

            X[i] = self._toTorchFloatTensor(data[0])[0]  # check if this is correct
            # for ich in range(data.shape[1]):
            #    X[i, ich] = self._toTorchFloatTensor(data[0, ich])[0]
            Y[i, 0] = self._toTorchFloatTensor(labels[0, 0])[0]
            P.append(path)

        return X, Y, P
