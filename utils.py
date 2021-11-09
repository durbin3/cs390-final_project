import os
from config import CONFIG
from PIL import Image
import numpy as np


def create_dir_if_not_exist(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def preprocess_image(img):
    return (img - 128) / 128


def deprocess_image(img):
    return np.clip((img + 1) * 128, 0, 255).astype(np.uint8)


def predict_random_image(generator):
    images = os.listdir(CONFIG.HR_DIR)
    img = Image.open(f'{CONFIG.HR_DIR}/{np.random.choice(images, 1)[0]}')
    img.show()
    lr_shape = tuple([CONFIG.INPUT_SHAPE[i] // CONFIG.DOWN_SAMPLE_SCALE for i in range(2)])
    lr = img.resize(lr_shape)
    lr.show()
    sr_img = generator.predict(np.array(lr)[np.newaxis, ...])
    Image.fromarray(deprocess_image(sr_img[0])).show()
