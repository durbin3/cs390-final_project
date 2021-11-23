from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Input, PReLU, BatchNormalization, Add, Dense, LeakyReLU, Flatten, \
    UpSampling2D
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.initializers import RandomNormal
import tensorflow as tf
from config import CONFIG


def get_generator(input_shape):
    init = RandomNormal(mean=0.0, stddev=0.02)

    inp = Input(input_shape)
    x = Conv2D(64, 9, padding='same', activation=PReLU(), kernel_initializer=init)(inp)
    conv1 = x

    for _ in range(CONFIG.B):
        rx = Conv2D(64, 3, padding='same', kernel_initializer=init)(x)
        rx = BatchNormalization(momentum=CONFIG.MOMENTUM)(rx)
        rx = PReLU(shared_axes=[1, 2])(rx)
        rx = Conv2D(64, 3, padding='same', kernel_initializer=init)(rx)
        rx = BatchNormalization(momentum=CONFIG.MOMENTUM)(rx)
        rx = Add()([rx, x])
        x = rx

    x = Conv2D(64, 3, padding='same', kernel_initializer=init)(x)
    x = BatchNormalization(momentum=CONFIG.MOMENTUM)(x)
    x = Add()([x, conv1])

    x = Conv2D(256, 3, padding='same', kernel_initializer=init)(x)
    x = UpSampling2D(CONFIG.DOWN_SAMPLE_SCALE // 2)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = Conv2D(256, 3, padding='same', kernel_initializer=init)(x)
    x = UpSampling2D(CONFIG.DOWN_SAMPLE_SCALE // 2)(x)
    x = PReLU(shared_axes=[1, 2])(x)

    x = Conv2D(3, 9, padding='same', activation='tanh', kernel_initializer=init)(x)
    return Model(inputs=inp, outputs=x, name='srgan_generator')


def get_discriminator(input_shape):
    inp = Input(input_shape)
    x = Conv2D(64, 3, 1, padding='same')(inp)
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
    x = Dense(1, activation=sigmoid)(x)
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
    model = Model(inputs=vgg19.inputs, outputs=vgg19.get_layer('block5_conv4').output, name='srgan_vgg')
    model.trainable = False
    return model
