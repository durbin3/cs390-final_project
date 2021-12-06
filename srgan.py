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


def build_gan(generator, discriminator, vgg) -> Model:
    discriminator.trainable = False
    input_shape = (*[CONFIG.INPUT_SHAPE[i] // CONFIG.DOWN_SAMPLE_SCALE for i in range(2)], 3)
    generator_input = Input(input_shape)
    generator_output = generator(generator_input)
    vgg_sr_input = preprocess_input(generator_output)
    generator_features = vgg(vgg_sr_input) / 12.75
    gan = Model(inputs=generator_input, outputs=[generator_features, discriminator(generator_output)])
    gan.compile(loss=['mse', 'binary_crossentropy'],
                loss_weights=[CONFIG.VGG_WEIGHT, CONFIG.D_WEIGHT],
                optimizer=Adam(learning_rate=CONFIG.LR_START))
    return gan


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

    lr = tf.Variable(CONFIG.LR_START)

    generator.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    for epoch in range(epochs):
        print(f'Initial Training epoch {epoch}')
        for step, (lr, hr) in enumerate(tqdm(datagen)):
            loss = generator.train_on_batch(lr, hr)

            logger.log_loss('Generator initial training', loss)
            if step % 100 == 0:
                logger.log_image()
                logger.log_img_distribution()
            logger.step()

        if logger.steps > 100000:
            lr.assign(CONFIG.LR_START / 10)

        if epoch % CONFIG.SAVE_INTERVAL == 0:
            create_dir_if_not_exist(CONFIG.SAVE_DIR)
            generator.save_weights(f'{CONFIG.SAVE_DIR}/generator_init_{epoch}.h5')
            logger.save_progress(epoch)

        datagen.on_epoch_end()

    logger.reset()
    discriminator.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy')
    gan = build_gan(generator, discriminator, vgg)
    for epoch in range(CONFIG.N_EPOCH):
        print(f'Training epoch {epoch}')
        # train discriminator
        for step, (lr, hr) in enumerate(tqdm(datagen)):
            sr = generator.predict(lr)

            if step % 50 == 0:
                discriminator.trainable = True
                d_loss_gen = discriminator.train_on_batch(sr, create_noisy_labels(0, CONFIG.BATCH_SIZE))
                d_loss_real = discriminator.train_on_batch(hr, create_noisy_labels(1, CONFIG.BATCH_SIZE))
                d_loss = 0.5 * np.add(d_loss_gen, d_loss_real)
                discriminator.trainable = False
                logger.log_loss('discriminator loss', tf.reduce_mean(d_loss))

            vgg_hr_input = preprocess_input(hr)
            vgg_features = vgg.predict(vgg_hr_input) / 12.75
            g_loss, _, _ = gan.train_on_batch(lr, [vgg_features, create_noisy_labels(1, CONFIG.BATCH_SIZE)])

            logger.log_loss('total loss', tf.reduce_mean(g_loss))
            if step % 50 == 0:
                logger.log_image()
                logger.log_img_distribution()

            if epoch % CONFIG.SAVE_INTERVAL == 0:
                create_dir_if_not_exist(CONFIG.SAVE_DIR)
                generator.save_weights(f'{CONFIG.SAVE_DIR}/generator_{epoch}.h5')
                discriminator.save_weights(f'{CONFIG.SAVE_DIR}/discriminator_{epoch}.h5')

            logger.step()
        datagen.on_epoch_end()


def main():
    train()


if __name__ == '__main__':
    main()
