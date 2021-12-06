import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # uncomment if using cpu
from tensorflow import keras
from tensorflow.keras.losses import MeanSquaredError, binary_crossentropy, BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf
from config import CONFIG
from model import *
from datagen import DataGenerator
import numpy as np
from utils import *
import tensorflow.keras.backend as K
from tensorboard import program
import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg19 import preprocess_input
from tqdm import tqdm


physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def vgg_loss(vgg):
    def loss(y_true, y_pred):
        y_true = denormalize_image(y_true)
        y_pred = denormalize_image(y_pred)
        y_true = preprocess_input(y_true)
        y_pred = preprocess_input(y_pred)
        ret = MeanSquaredError()(vgg(y_true) / 12.75, vgg(y_pred) / 12.75)
        return ret
    return loss


def discriminator_loss(hr, sr):
    return BinaryCrossentropy()(tf.ones_like(hr), hr) + BinaryCrossentropy()(tf.zeros_like(sr), sr)


def train():
    datagen = DataGenerator(CONFIG.HR_DIR,
                            CONFIG.INPUT_SHAPE,
                            down_sample_scale=CONFIG.DOWN_SAMPLE_SCALE,
                            batch_size=CONFIG.BATCH_SIZE)

    generator = get_generator((
        CONFIG.INPUT_SHAPE[0] // CONFIG.DOWN_SAMPLE_SCALE,
        CONFIG.INPUT_SHAPE[1] // CONFIG.DOWN_SAMPLE_SCALE,
        3))
    discriminator = get_discriminator(CONFIG.INPUT_SHAPE)

    g_opt = Adam(learning_rate=CONFIG.LR_START)
    d_opt = Adam(learning_rate=CONFIG.LR_START)

    vgg = get_vgg(CONFIG.INPUT_SHAPE)

    epochs = CONFIG.N_INIT_EPOCH
    if not CONFIG.RESTART:
        if os.path.exists('saved_weights/progress.txt'):
            with open('saved_weights/progress.txt', 'r') as f:
                epochs -= int(f.readline())
        generator_path = f'{CONFIG.SAVE_DIR}/generator.h5'
        if os.path.exists(generator_path):
            generator.load_weights(generator_path)

    logger = Logger(generator, discriminator)
    logger.log_image()
    logger.log_img_distribution()
    breakpoint()
    for epoch in range(epochs):
        print(f'Initial Training epoch {epoch}')
        for step, (lr, hr) in enumerate(tqdm(datagen)):
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)
            with tf.GradientTape() as g_tape:
                sr = generator(lr)
                loss = MeanSquaredError()(hr, sr)

            g_grad = g_tape.gradient(loss, generator.trainable_variables)
            g_opt.apply_gradients(zip(g_grad, generator.trainable_variables))

            logger.log_loss('Generator initial training', loss)
            if step % 100 == 0:
                logger.log_image()
                logger.log_img_distribution()
                logger.log_gradients('generator gradient', tf.concat([tf.reshape(grad, -1) for grad in g_grad], 0))
            logger.step()

        if epoch % CONFIG.SAVE_INTERVAL == 0:
            create_dir_if_not_exist(CONFIG.SAVE_DIR)
            generator.save_weights(f'{CONFIG.SAVE_DIR}/generator.h5')
            logger.save_progress(epoch)

        datagen.on_epoch_end()

    logger.reset()
    for epoch in range(CONFIG.N_EPOCH):
        print(f'Training epoch {epoch}')
        # train discriminator
        for step, (lr, hr) in enumerate(tqdm(datagen)):
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)
            with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
                sr = generator(lr)

                d_hr = discriminator(hr)
                d_sr = discriminator(sr)

                loss_x = vgg_loss(vgg)(hr, sr)
                loss_gen = BinaryCrossentropy()(tf.ones_like(d_sr), d_sr)
                loss_g = loss_x + CONFIG.D_WEIGHT * loss_gen

                loss_d = discriminator_loss(d_hr, d_sr)

            g_grad = g_tape.gradient(loss_g, generator.trainable_variables)
            g_opt.apply_gradients(zip(g_grad, generator.trainable_variables))

            if step % 3 == 0:
                d_grad = d_tape.gradient(loss_d, discriminator.trainable_variables)
                d_opt.apply_gradients(zip(d_grad, discriminator.trainable_variables))

            logger.log_loss('discriminator loss', tf.reduce_mean(loss_d))
            logger.log_loss('total loss', tf.reduce_mean(loss_g))

            if step % 50 == 0:
                logger.log_image()
                logger.log_img_distribution()
                logger.log_weights()

            if epoch % CONFIG.SAVE_INTERVAL == 0:
                create_dir_if_not_exist(CONFIG.SAVE_DIR)
                generator.save_weights(f'{CONFIG.SAVE_DIR}/generator.h5')
                discriminator.save_weights(f'{CONFIG.SAVE_DIR}/discriminator.h5')

            logger.step()
        datagen.on_epoch_end()

def main():
    train()


if __name__ == '__main__':
    main()
