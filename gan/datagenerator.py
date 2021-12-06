from tensorflow import keras
import os
import numpy as np
from PIL import Image
from utils import *
from config import CONFIG


class DataGenerator(keras.utils.Sequence):
    def __init__(self, hr_dir, lr_dir,hr_shape, down_sample_scale=4, batch_size=32):
        self.hr_shape = hr_shape
        self.lr_shape = (hr_shape[0] // down_sample_scale, hr_shape[1] // down_sample_scale, hr_shape[2])
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.down_sample_scale = down_sample_scale
        self.batch_size = batch_size
        self.hr_files = os.listdir(self.hr_dir)
        self.lr_files = os.listdir(self.lr_dir)
        self.on_epoch_end()

    def on_epoch_end(self):
        self.hr_files = os.listdir(self.hr_dir)[:CONFIG.DATA_SIZE]
        self.lr_files = os.listdir(self.lr_dir)[:CONFIG.DATA_SIZE]

        combined = np.array(self.hr_files)
        combined = np.vstack((combined,self.lr_files))
        combined = np.transpose(combined)
        np.random.shuffle(combined)
        
        self.hr_files = combined[:,0]
        self.lr_files = combined[:,1]

    def __len__(self):
        return len(self.hr_files) // self.batch_size

    def __getitem__(self, index):
        return self.__data_generation(index)

    def __data_generation(self, index):
        hr_batch = np.zeros((self.batch_size, *self.hr_shape))
        lr_batch = np.zeros((self.batch_size, *self.lr_shape))
        end_idx = (index + 1) * self.batch_size if len(self.hr_files) - index * self.batch_size >= 32 \
            else len(self.hr_files)
        for idx in range(index * self.batch_size, end_idx):
            img = Image.open(f'{self.hr_dir}/{self.hr_files[idx]}')
            lr = Image.open(f'{self.lr_dir}/{self.lr_files[idx]}')
            width, height = img.size
            hr_img = normalize_image(np.array(img))
            lr_img = normalize_image(np.array(lr))

            hr_batch[idx % self.batch_size] = hr_img
            lr_batch[idx % self.batch_size] = lr_img
        return lr_batch, hr_batch
