from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Input, PReLU, BatchNormalization, Add, Dense, LeakyReLU, Flatten, \
    UpSampling2D, Lambda, Conv2DTranspose
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.initializers import RandomNormal
import tensorflow as tf
from config import CONFIG
from utils import *


def PixelShuffler():
    return Lambda(lambda x: tf.nn.depth_to_space(x, CONFIG.DOWN_SAMPLE_SCALE // 2))


def get_generator(input_shape):
    inp = Input(input_shape)
    x = Lambda(normalize_input)(inp)
    x = Conv2D(64, 9, padding='same')(x)
    x = PReLU(shared_axes=[1, 2])(x)
    conv1 = x

    for _ in range(CONFIG.B):
        rx = Conv2D(64, 3, padding='same')(x)
        rx = BatchNormalization(momentum=CONFIG.MOMENTUM)(rx)
        rx = PReLU(shared_axes=[1, 2])(rx)
        rx = Conv2D(64, 3, padding='same')(rx)
        rx = BatchNormalization(momentum=CONFIG.MOMENTUM)(rx)
        rx = Add()([rx, x])
        x = rx

    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization(momentum=CONFIG.MOMENTUM)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = Add()([x, conv1])

    x = Conv2D(256, 3, padding='same')(x)
    x = UpSampling2D(size=2)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = Conv2D(256, 3, padding='same')(x)
    x = UpSampling2D(size=2)(x)
    x = PReLU(shared_axes=[1, 2])(x)

    x = Conv2D(3, 9, padding='same', activation='tanh')(x)
    x = Lambda(denormalize_image)(x)
    return Model(inputs=inp, outputs=x, name='srgan_generator')


def get_discriminator(input_shape):
    inp = Input(input_shape)
    x = Lambda(normalize_output)(inp)
    x = Conv2D(64, 3, 1, padding='same')(x)
    x = BatchNormalization(momentum=CONFIG.MOMENTUM)(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(64, 3, 2, padding='same')(x)
    x = BatchNormalization(momentum=CONFIG.MOMENTUM)(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, 3, 1, padding='same')(x)
    x = BatchNormalization(momentum=CONFIG.MOMENTUM)(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, 3, 2, padding='same')(x)
    x = BatchNormalization(momentum=CONFIG.MOMENTUM)(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, 3, 1, padding='same')(x)
    x = BatchNormalization(momentum=CONFIG.MOMENTUM)(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, 3, 2, padding='same')(x)
    x = BatchNormalization(momentum=CONFIG.MOMENTUM)(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(512, 3, 1, padding='same')(x)
    x = BatchNormalization(momentum=CONFIG.MOMENTUM)(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(512, 3, 2, padding='same')(x)
    x = BatchNormalization(momentum=CONFIG.MOMENTUM)(x)
    x = LeakyReLU(0.2)(x)

    x = Flatten()(x)
    x = Dense(1024)(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=inp, outputs=x, name='srgan_discriminator')


def get_vgg(input_shape):
    vgg19 = VGG19(
        include_top=False,
        input_shape=input_shape,
        weights='imagenet'
    )
    vgg19.trainable = False
    for layer in vgg19.layers:
        layer.trainable = False
    model = Model(inputs=vgg19.inputs, outputs=vgg19.layers[20].output, name='srgan_vgg')
    model.trainable = False
    return model
