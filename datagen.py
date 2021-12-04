from tensorflow import keras
import os
import numpy as np
from PIL import Image
from utils import *
from config import CONFIG


class DataGenerator(keras.utils.Sequence):
    def __init__(self, hr_dir, hr_shape, down_sample_scale=4, batch_size=32):
        self.hr_shape = hr_shape
        self.lr_shape = (hr_shape[0] // down_sample_scale, hr_shape[1] // down_sample_scale, hr_shape[2])
        self.hr_dir = hr_dir
        self.down_sample_scale = down_sample_scale
        self.batch_size = batch_size
        self.hr_files = os.listdir(self.hr_dir)
        self.on_epoch_end()

    def on_epoch_end(self):
        self.hr_files = os.listdir(self.hr_dir)[:CONFIG.DATA_SIZE] * self.batch_size
        np.random.shuffle(self.hr_files)

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
            width, height = img.size
            # hr_img = normalize_image(preprocess_image(np.array(img)))
            # lr_img = normalize_image(preprocess_image(
            #     np.array(img.resize((width // self.down_sample_scale, height // self.down_sample_scale))),
            # ), min_value=0)
            hr_img = normalize_image(np.array(img))
            lr_img = normalize_image(np.array(img.resize((width // self.down_sample_scale, height // self.down_sample_scale))), min_value=0)

            hr_batch[idx % self.batch_size] = hr_img
            lr_batch[idx % self.batch_size] = lr_img
        return lr_batch, hr_batch


def test_data_generator():
    gen = DataGenerator('high_res', (256, 256, 3))
    assert len(gen[0]) == 2
    assert gen[0][0].shape == (CONFIG.BATCH_SIZE, 64, 64, 3)
    assert gen[0][1].shape == (CONFIG.BATCH_SIZE, 256, 256, 3)
    print('data generator test passed')


if __name__ == '__main__':
    test_data_generator()