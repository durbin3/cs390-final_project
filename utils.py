import os
from config import CONFIG
from PIL import Image
import numpy as np
import tensorflow as tf
import datetime
from tensorboard import program
import tensorflow.keras as keras


class Logger:
    def __init__(self, generator: keras.Model, discriminator, port=6006):
        self.generator = generator
        self.discriminator = discriminator
        self.progress = read_progress()
        self.log_dir = 'logs/' + get_time()
        self.writer = tf.summary.create_file_writer(self.log_dir, flush_millis=1000)
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.log_dir, '--reload_interval', '1', '--port', f'{port}'])
        self.url = tb.launch()
        print(f"Tensorflow listening on {self.url}")

        self.steps = 0
        self.save_images = np.random.choice(os.listdir(CONFIG.HR_DIR), 5)

    def log_loss(self, name, value):
        with self.writer.as_default():
            tf.summary.scalar(name, value, step=self.steps)

    def log_image(self):
        images = os.listdir(CONFIG.HR_DIR)
        # img = Image.open(f'{CONFIG.HR_DIR}/{np.random.choice(images, 1)[0]}')
        img = Image.open(f'{CONFIG.HR_DIR}/{images[1]}')
        lr_shape = tuple([CONFIG.INPUT_SHAPE[i] // CONFIG.DOWN_SAMPLE_SCALE for i in range(2)])
        lr = img.resize(lr_shape)
        sr_img = self.generator.predict(np.array(lr)[np.newaxis, ...])[0].astype(np.uint8)
        with self.writer.as_default():
            tf.summary.image('lr_image', np.array(lr)[np.newaxis, ...], step=self.steps)
            tf.summary.image('hr_image', get_hr_image(img)[np.newaxis, ...], step=self.steps)
            tf.summary.image('sr_image', sr_img[np.newaxis, ...], step=self.steps)

    def log_img_distribution(self):
        dist = np.empty((0, 3))
        images = np.random.choice(os.listdir(CONFIG.HR_DIR), 100)
        for image in images:
            image = Image.open(f'{CONFIG.HR_DIR}/{image}')
            sr_img = self.generator.predict(downsample_image(image)[np.newaxis, ...])[0]
            dist = np.concatenate((dist, sr_img.reshape(-1, 3)))

        with self.writer.as_default():
            for i, c in enumerate(['R', 'G', 'B']):
                tf.summary.histogram(f'{c} distribution', dist[:, i], step=self.steps)

    def save_image_batch(self):
        for image in self.save_images:
            image = Image.open(f'{CONFIG.HR_DIR}/{image}')
            sr_img = self.generator.predict(downsample_image(image)[np.newaxis, ...])[0]
            Image.fromarray(sr_img).save(f'saved_images/{image.split(".")[0]}/{self.steps}.png')

    def log_weights(self):
        with self.writer.as_default():
            for layer in self.generator.layers:
                if 'conv' in layer.name.lower():
                    if len(layer.weights) > 0:
                        tf.summary.histogram(f'generator_{layer.name}', layer.weights[0], step=self.steps)
            for layer in self.discriminator.layers:
                if 'conv' in layer.name.lower():
                    if len(layer.weights) > 0:
                        tf.summary.histogram(f'discriminator_{layer.name}', layer.weights[0], step=self.steps)

    def log_gradients(self, name, gradient):
        with self.writer.as_default():
            tf.summary.histogram(name, gradient, step=self.steps)

    def save_progress(self, k, v):
        self.progress[k] = v
        write_progress(self.progress)

    def get_bool(self, k):
        return k in self.progress and bool(self.progress[k])

    def get_int(self, k):
        if k not in self.progress:
            return 0
        return int(self.progress[k])

    def step(self):
        self.steps += 1

    def reset(self):
        self.steps = 0


def read_progress():
    progress = {}
    if not os.path.exists('saved_weights/progress.txt'):
        return progress
    with open('saved_weights/progress.txt', 'r') as f:
        for line in f.readlines():
            line = line.split(':')
            progress[line[0]] = line[1]
    return progress


def write_progress(progress):
    with open('saved_weights/progress.txt', 'w') as f:
        for k, v in progress.items():
            f.write(f'{k}:{v}\n')


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
    return keras.applications.vgg19.preprocess_input(img)


def normalize_input(img):
    return img / 255


def normalize_output(img):
    return img / 127.5 - 1


def denormalize_image(img):
    return (img + 1) * 127.5


def get_hr_image(img):
    return np.array(img.resize(CONFIG.INPUT_SHAPE[:2]))


def downsample_image(img):
    return np.array(img.resize(tuple([CONFIG.INPUT_SHAPE[i] // CONFIG.DOWN_SAMPLE_SCALE for i in range(2)])))


def get_time():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")