import os
from config import CONFIG
from PIL import Image
import numpy as np
import tensorflow as tf
import datetime
from tensorboard import program
import tensorflow.keras as keras


class Logger:
    def __init__(self, generator: keras.Model, discriminator):
        self.generator = generator
        self.discriminator = discriminator
        self.log_dir = 'logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = tf.summary.create_file_writer(self.log_dir, flush_millis=1000)
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.log_dir, '--reload_interval', '1'])
        self.url = tb.launch()
        print(f"Tensorflow listening on {self.url}")

        self.steps = 0

    def log_loss(self, name, value):
        with self.writer.as_default():
            tf.summary.scalar(name, value, step=self.steps)

    def log_image(self):
        images = os.listdir(CONFIG.HR_DIR)
        # img = Image.open(f'{CONFIG.HR_DIR}/{np.random.choice(images, 1)[0]}')
        img = Image.open(f'{CONFIG.HR_DIR}/{images[0]}')
        lr_shape = tuple([CONFIG.INPUT_SHAPE[i] // CONFIG.DOWN_SAMPLE_SCALE for i in range(2)])
        lr = img.resize(lr_shape)
        sr_img = self.generator.predict(np.array(lr)[np.newaxis, ...])
        # sr_img = deprocess_image(denormalize_image(sr_img[0]))
        sr_img = denormalize_image(sr_img[0])
        with self.writer.as_default():
            tf.summary.image('hr_image', np.array(img)[np.newaxis, ...], step=self.steps)
            tf.summary.image('sr_image', sr_img[np.newaxis, ...], step=self.steps)

    def log_img_distribution(self):
        dist = np.empty((0, 3))
        for _ in range(100):
            sr_img = predict_random_image(self.generator, show=False)
            dist = np.concatenate((dist, sr_img.reshape(-1, 3)))

        with self.writer.as_default():
            for i, c in enumerate(['R', 'G', 'B']):
                tf.summary.histogram(f'{c} distribution', dist[:, i], step=self.steps)

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

    def step(self):
        self.steps += 1


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
    # return (img - 127.5) / 127.5
    return img/255
    return keras.applications.vgg19.preprocess_input(img)


def normalize_image(img, min_value=-1):
    if min_value == 0:
        return img / 255
    return (img - 127.5) / 127.5


def deprocess_image(x):
    img = x.copy().astype(np.float64)
    if x.ndim > 3:
        img = img.squeeze(0)
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]
    return np.clip(img, 0, 255).astype('uint8')


def denormalize_image(img):
    if isinstance(img, np.ndarray):
<<<<<<< HEAD
        # return np.clip((img + 1) * 127.5, 0, 255).astype(np.uint8)
        return np.clip(img*255,0,255).astype(np.uint8)
=======
        return np.clip((img + 1) * 127.5, 0, 255).astype(np.uint8)
>>>>>>> d19cdb9dd4019ac1e594d359482c1e2c6e0d3c3d
    else:
        # return tf.multiply(tf.add(img, 1), 127.5)
        return tf.multiply(img,255)


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


def predict_random_image(generator, show=False):
    images = os.listdir(CONFIG.HR_DIR)
    img = Image.open(f'{CONFIG.HR_DIR}/{np.random.choice(images, 1)[0]}')
    if show:
        img.show()
    else:
        img.save("images/hr.jpg")
    lr_shape = tuple([CONFIG.INPUT_SHAPE[i] // CONFIG.DOWN_SAMPLE_SCALE for i in range(2)])
    lr = img.resize(lr_shape)
    if show:
        lr.show()
    else: 
        lr.save("images/lr.jpg")
    sr_img = generator.predict(np.array(lr)[np.newaxis, ...])
    output_image = Image.fromarray(deprocess_image(sr_img[0]))
    if show:
        output_image.show()
    else:
        output_image.save("images/generated.jpg")
    # return sr_img[0]
    #     sr_img = denormalize_image(sr_img)
    #     Image.fromarray(deprocess_image(sr_img[0])).show()
    # return sr_img[0]
