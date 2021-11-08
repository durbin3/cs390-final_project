from tensorflow import keras
import os
import numpy as np
from PIL import Image


class DataGenerator(keras.utils.Sequence):
    def __init__(self, hr_dir, down_sample_scale=4, batch_size=32):
        self.hr_dir = hr_dir
        self.down_sample_scale = down_sample_scale
        self.batch_size = batch_size
        self.hr_files = os.listdir(self.hr_dir)

    def on_epoch_end(self):
        np.random.shuffle(self.hr_files)

    def __len__(self):
        return int(np.floor(len(self.hr_files) / self.batch_size))

    def __getitem__(self, index):
        return self.__data_generation(index)

    def __data_generation(self, index):
        hr_batch = []
        lr_batch = []
        for idx in range(index * self.batch_size, (index + 1) * self.batch_size):
            img = Image.open(f'{self.hr_dir}/{self.hr_files[idx]}')
            width, height = img.size
            hr_img = np.array(img)
            lr_img = np.array(img.resize((width // self.down_sample_scale, height // self.down_sample_scale)))
            hr_batch.append(hr_img)
            lr_batch.append(lr_img)
        return lr_batch, hr_batch


def test_data_generator():
    gen = DataGenerator('high_res')
    assert len(gen[0]) == 2  # lr, hr
    assert len(gen[0][0]) == 32  # batch size
    assert len(gen[0][1]) == 32
    assert gen[0][0][0].shape == (64, 64, 3)  # lr
    assert gen[0][1][0].shape == (256, 256, 3)  # hr
    print('data generator test passed')


if __name__ == '__main__':
    test_data_generator()