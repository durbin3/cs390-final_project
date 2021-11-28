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

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def mse_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))


def vgg_loss(vgg):
    def loss(y_true, y_pred):
        y_true = deprocess_image(y_true)
        y_pred = deprocess_image(y_pred)
        ret = mse_loss(vgg(y_true), vgg(y_pred))
        return ret

    return loss


def content_loss(vgg):
    def loss(y_true, y_pred):
        return CONFIG.MSE_WEIGHT * mse_loss(y_true, y_pred) + vgg_loss(vgg)(y_true, y_pred)

    return loss


def build_gan(generator, discriminator, vgg) -> Model:
    discriminator.trainable = False
    input_shape = (*[CONFIG.INPUT_SHAPE[i] // CONFIG.DOWN_SAMPLE_SCALE for i in range(2)], 3)
    generator_input = Input(input_shape)
    generator_output = generator(generator_input)
    gan = Model(inputs=generator_input, outputs=[generator_output, discriminator(generator_output)])
    gan.compile(loss=['mse', 'binary_crossentropy'],
                loss_weights=[CONFIG.VGG_WEIGHT, CONFIG.D_WEIGHT],
                optimizer=Adam(learning_rate=CONFIG.LR_START))
    return gan


def train():
    log_dir = 'logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = tf.summary.create_file_writer(log_dir, flush_millis=1000)
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', log_dir, '--reload_interval', '1'])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")

    datagen = DataGenerator(CONFIG.HR_DIR,
                            CONFIG.INPUT_SHAPE,
                            down_sample_scale=CONFIG.DOWN_SAMPLE_SCALE,
                            batch_size=CONFIG.BATCH_SIZE)
    datagen_d = DataGenerator(CONFIG.HR_DIR,
                              CONFIG.INPUT_SHAPE,
                              down_sample_scale=CONFIG.DOWN_SAMPLE_SCALE,
                              batch_size=CONFIG.BATCH_SIZE_D)

    generator = get_generator((
        CONFIG.INPUT_SHAPE[0] // CONFIG.DOWN_SAMPLE_SCALE,
        CONFIG.INPUT_SHAPE[1] // CONFIG.DOWN_SAMPLE_SCALE,
        3))
    discriminator = get_discriminator(CONFIG.INPUT_SHAPE)
    vgg = get_vgg(CONFIG.INPUT_SHAPE)
    if CONFIG.LOAD_WEIGHTS:
        generator_path = f'{CONFIG.SAVE_DIR}/generator.h5'
        if os.path.exists(generator_path):
            generator.load_weights(generator_path)
        discriminator_path = f'{CONFIG.SAVE_DIR}/discriminator.h5'
        if os.path.exists(discriminator_path):
            discriminator.load_weights(f'{CONFIG.SAVE_DIR}/discriminator.h5')

    gan = build_gan(generator, discriminator, vgg)
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.1))

    iter = 0
    for epoch in range(CONFIG.N_EPOCH):
        print(f'Training epoch {epoch}')
        # train discriminator
        if epoch % 2 == 0:
            discriminator.trainable = True
            for step, (lr_batch, hr_batch) in enumerate(datagen_d):
                sr_batch = generator.predict(lr_batch)
                d_loss_real = discriminator.train_on_batch(hr_batch, create_noisy_labels(0, CONFIG.BATCH_SIZE_D))
                d_loss_gen = discriminator.train_on_batch(sr_batch, create_noisy_labels(1, CONFIG.BATCH_SIZE_D))
                d_loss = np.add(d_loss_real, d_loss_gen) / 2
                iter += 1
                with writer.as_default():
                    tf.summary.scalar('discriminator_loss', d_loss, step=iter)
            datagen_d.on_epoch_end()
            discriminator.trainable = False
        else:
            # train gan
            for step, (lr_batch, hr_batch) in enumerate(datagen):
                if (step > 500):
                    break
                
                print(f"\t Step: {step}",end='\r')
                generator.trainable = True
                gan_loss = gan.train_on_batch(lr_batch, [hr_batch, create_noisy_labels(1, CONFIG.BATCH_SIZE)])
                generator.trainable = False
                # print(f'\tgan loss: {gan_loss}')
                iter += 1
                with writer.as_default():
                    tf.summary.scalar('gan_loss', gan_loss[0], step=iter)
            datagen.on_epoch_end()
            print(f'\tgan loss: {gan_loss}')

        if epoch % CONFIG.PREVIEW_INTERVAL == 0:
            # predict_random_image(generator)
            log_random_image(generator, writer)

        if epoch % CONFIG.SAVE_INTERVAL == 0:
            create_dir_if_not_exist(CONFIG.SAVE_DIR)
            generator.save_weights(f'{CONFIG.SAVE_DIR}/generator.h5')
            discriminator.save_weights(f'{CONFIG.SAVE_DIR}/discriminator.h5')


def main():
    train()


if __name__ == '__main__':
    main()
