import os
from config import CONFIG
from PIL import Image
import numpy as np
import tensorflow as tf


def create_dir_if_not_exist(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def create_noisy_labels(true_label, size):
    if true_label == 1:
        labels = np.ones(size) - np.random.random(size) * CONFIG.D_INPUT_RANDOM
    else:
        labels = np.random.random(size) * CONFIG.D_INPUT_RANDOM
    flip_idx = np.random.choice(size, int(size * CONFIG.RAND_FLIP), replace=False)
    labels[flip_idx] = 1 - labels[flip_idx]
    return labels


def preprocess_image(img):
    return (img - 128) / 128


def deprocess_image(img):
    if isinstance(img, np.ndarray):
        return np.clip((img + 1) * 128, 0, 255).astype(np.uint8)
    else:
        return tf.multiply(tf.add(img, 1), 127.5)


def log_random_image(generator, writer):
    images = os.listdir(CONFIG.HR_DIR)
    img = Image.open(f'{CONFIG.HR_DIR}/{np.random.choice(images, 1)[0]}')
    lr_shape = tuple([CONFIG.INPUT_SHAPE[i] // CONFIG.DOWN_SAMPLE_SCALE for i in range(2)])
    lr = img.resize(lr_shape)
    sr_img = generator.predict(np.array(lr)[np.newaxis, ...])
    sr_img = deprocess_image(sr_img[0])
    with writer.as_default():
        tf.summary.image('hr_image', img)
        tf.summary.image('sr_image', sr_img)


def predict_random_image(generator, show=True):
    images = os.listdir(CONFIG.HR_DIR)
    img = Image.open(f'{CONFIG.HR_DIR}/{np.random.choice(images, 1)[0]}')
    if show:
        img.show()
    lr_shape = tuple([CONFIG.INPUT_SHAPE[i] // CONFIG.DOWN_SAMPLE_SCALE for i in range(2)])
    lr = img.resize(lr_shape)
    if show:
        lr.show()
    sr_img = generator.predict(np.array(lr)[np.newaxis, ...])
    if show:
        Image.fromarray(deprocess_image(sr_img[0])).show()
    return sr_img[0]