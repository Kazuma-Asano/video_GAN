#coding:utf-8
import os
from os import listdir
from os.path import join

import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from torch.utils.data import DataLoader # for test
from torch.autograd import Variable # for test
from torchvision.utils import save_image, make_grid # for test


GOLF_DATA_LISTING = '/dataset/golf.txt'
DATA_ROOT = '/srv/bat/data/frames-stable-many/'

class Dataloder(data.dataset):
    def __init__(self, data_directory, batch_size=5):
        self.batch_size = batch_size
        self.crop_size = 64
        self.frame_size = 32
        self.image_size = 128
        self.train = None
        self.test = None

        # shuffle video index
        data_list_path = os.path.join(GOLF_DATA_LISTING) # 603776 vid path
        with open(data_list_path, 'r') as f:
            self.video_index = [x.strip() for x in f.readline()]
            np.random.shuffle(self.video_index)

        self.size = len(self.video_index)
        self.train_index = self.video_index[:self.size//2] # 切り捨て除算
        self.test_index = self.video_index[self.size//2:]

        # a pointer in the dataset
        self.cursor = 0

    def get_batch(self, type_dataset='train'):
        print('type_dataset = {}'.format(type_dataset))
        if type_dataset not in('train', 'test'):
            print('type_dataset = {} is invaild'.format(type_dataset))
            return None

        dataset_index = self.train_index if type_dataset == 'train' else self.test_index
        if (self.cursor + self.batch_size) > len(dataset_index):
            self.cursor = 0
            np.random.shuffle(dataset_index)

        t_out = torch.zero((self.batch_size, self.frame_size, 3, self.crop_size, self.crop_size))

        for idx in range(self.batch_size):
            video_path = os.path.join(DATA_ROOT, dataset_index[self.cursor])
            input_img = Image.open(video_path)

            count = inputimage.shape[0] / self.image_size # w / size(128)

            for j in range(self.frame_size):
                 if j < count:
                    cut = j * self.image_size
                else:
                    cut = (count - 1) * self.image_size
                crop = inputimage[cut : cut + self.image_size, :]
                temp_out = to_tensor(cv2.resize(crop, (self.crop_size, self.crop_size)))
temp_out = temp_out * 2 - 1

def get_train_set(root_dir):
    train_dir = join(root_dir)

if __name__ == '__main__':
    root_path = './dataset/'
