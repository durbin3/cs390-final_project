from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Input, PReLU, BatchNormalization, Add, Dense, LeakyReLU, Flatten, UpSampling2D
from tensorflow.keras.activations import tanh,sigmoid
import tensorflow as tf
from config import CONFIG


def get_generator(input_shape):
    inp = Input(input_shape)
    x = Conv2D(64, 3, padding='same', activation=PReLU())(inp)
    conv1 = x

    for _ in range(CONFIG.B):
        rx = Conv2D(64, 3, padding='same')(x)
        rx = BatchNormalization(axis=1)(rx)
        rx = PReLU()(rx)
        rx = Conv2D(64, 3, padding='same')(rx)
        rx = BatchNormalization(axis=1)(rx)
        rx = Add()([rx, x])
        x = rx

    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization(axis=1)(x)
    x = Add()([x, conv1])

    x = Conv2D(256, 3, padding='same')(x)
    x = UpSampling2D(CONFIG.DOWN_SAMPLE_SCALE // 2)(x)
    x = PReLU()(x)
    x = Conv2D(256, 3, padding='same')(x)
    x = UpSampling2D(CONFIG.DOWN_SAMPLE_SCALE // 2)(x)
    x = PReLU()(x)

    x = Conv2D(3, 1, padding='same', activation=tanh)(x)
    return Model(inputs=inp, outputs=x, name='srgan_generator')


def get_discriminator(input_shape):
    inp = Input(input_shape)
    x = Conv2D(64, 3, 1, padding='same')(inp)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(64, 3, 2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, 3, 1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, 3, 2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, 3, 1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, 3, 2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(512, 3, 1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(512, 3, 2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Flatten()(x)
    x = Dense(1024, activation=LeakyReLU(0.2))(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(1, activation=sigmoid)(x)
    return Model(inputs=inp, outputs=x, name='srgan_discriminator')


def get_vgg(input_shape):
    vgg19 = VGG19(
        include_top=False,
        input_shape=input_shape,
        weights='imagenet'
    )
    return Model(inputs=vgg19.inputs, outputs=vgg19.layers[9].output, name='srgan_vgg')
