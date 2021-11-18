import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from tensorflow import keras
from tensorflow.keras.losses import mean_squared_error, binary_crossentropy
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from config import CONFIG
from model import *
from datagen import DataGenerator
import numpy as np
from utils import *
from tensorflow.python.framework.ops import numpy_text
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def vgg_loss(vgg):
    def loss(y_true, y_pred):
        return mean_squared_error(vgg(y_true), vgg(y_pred))
    return loss


def build_gan(generator, discriminator, vgg) -> Model:
    print("Building Gan")
    discriminator.trainable = False
    input_shape = (*[CONFIG.INPUT_SHAPE[i] // CONFIG.DOWN_SAMPLE_SCALE for i in range(2)], 3)
    generator_input = Input(input_shape)
    generator_output = generator(generator_input)
    gan = Model(inputs=generator_input, outputs=[generator_output, discriminator(generator_output)])
    gan.compile(loss=[vgg_loss(vgg), 'binary_crossentropy'],
                loss_weights=[1, 1e-3],
                optimizer=Adam(learning_rate=CONFIG.LR_START))
    return gan


def train():
    print(f'Constructing Models')
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
    vgg.trainable = False
    generator.compile(loss=vgg_loss(vgg), optimizer=Adam(learning_rate=CONFIG.LR_START))
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=CONFIG.LR_START))
    gan = build_gan(generator, discriminator, vgg)

    predict_random_image(generator)

    print("Training")
    for epoch in range(CONFIG.N_EPOCH):
        gan_loss = 0
        d_loss = 0
        print("Epoch: ", epoch)
        for step, (lr_batch, hr_batch) in enumerate(datagen):
            # print("\tStep: ", step, "/",datagen.__len__(),end='\r')
            print("\tStep: ",step,"/", min(500,datagen.__len__()),end='\r')
            if (step >= min(500,datagen.__len__())):
                print()
                step = 0
                break
            sr_batch = generator.predict(lr_batch)

            # train discriminator
            if step % 25 == 0:
                discriminator.trainable = True
                d_loss_real = discriminator.train_on_batch(hr_batch, tf.ones(CONFIG.BATCH_SIZE))
                d_loss_gen = discriminator.train_on_batch(sr_batch, tf.zeros(CONFIG.BATCH_SIZE))
                d_loss = np.add(d_loss_real, d_loss_gen) / 2
                discriminator.trainable = False

            # train gan
            gan_loss = gan.train_on_batch(lr_batch, [hr_batch, tf.ones(CONFIG.BATCH_SIZE)])
            # print(f'Training epoch {epoch} step {step}')
            # print(f'\tdiscriminator loss: {d_loss}')
            # print(f'\tdiscriminator accuracy: {100 * d_loss[1]}%')
            # print(f'\tgan loss: {gan_loss}')

        if epoch % CONFIG.SAVE_INTERVAL == 0:
            create_dir_if_not_exist(CONFIG.SAVE_DIR)
            generator.save(f'{CONFIG.SAVE_DIR}/generator')
            discriminator.save(f'{CONFIG.SAVE_DIR}/discriminator')
            
        print("\tLosses: \tDiscriminator: ",d_loss,"\tGenerator: ",gan_loss)
        predict_random_image(generator)
        datagen.on_epoch_end()

def main():
    print("Main")
    train()


if __name__ == '__main__':
    main()
