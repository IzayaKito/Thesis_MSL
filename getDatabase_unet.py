from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os

import numpy as np
import skimage.io as io

from image_util_unet import BaseDataProvider


class DataProvider(BaseDataProvider):

    def __init__(self, inputSize, fineSize, segtype, semi_rate, input_nc, path, a_min=0, a_max=100, mode=None):
        super(DataProvider, self).__init__(a_min, a_max)
        self.nx = inputSize
        self.ny = inputSize
        self.nx_f = fineSize
        self.ny_f = fineSize
        self.semi_rate = semi_rate
        self.segtype = segtype
        self.channels = input_nc
        self.path = path
        self.mode = mode
        self.data_idx = -1
        self.n_data = self._load_data()

    def _load_data(self):
        path_ = os.path.join(self.path, self.mode)
        filefolds = os.listdir(path_)
        self.imageNum = []
        self.filePath = []

        for isub, filefold in enumerate(filefolds):
            foldpath = os.path.join(path_, filefold)
            dataFold = sorted(os.listdir(foldpath))
            for inum, idata in enumerate(dataFold):
                dataNum = int(idata.split('.')[0])
                dataFold[inum] = dataNum
            dataFile = sorted(dataFold)
            for islice in range(1, len(dataFile)-1):
                self.imageNum.append((foldpath, dataFile[islice], isub))

        if self.mode == "train":
            np.random.shuffle(self.imageNum)

        return len(self.imageNum)

    def _shuffle_data_index(self):
        self.data_idx += 1
        if self.data_idx >= self.n_data:
            self.data_idx = 0
            if self.mode == "train":
                np.random.shuffle(self.imageNum)

    def _next_data(self):
        self._shuffle_data_index()
        filePath = self.imageNum[self.data_idx]
        data = np.zeros((self.nx, self.ny, self.channels))
        labels = np.zeros((self.nx, self.ny, self.channels))

        fileName = os.path.join(filePath[0], str(filePath[1]) + '.jpg')
        data = io.imread(fileName)
        data = data/255

        path = filePath[0] + str(filePath[1])
        return data, labels, path
