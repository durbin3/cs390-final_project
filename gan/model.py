from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Input, PReLU, BatchNormalization, add, Dense, LeakyReLU, Flatten, \
    UpSampling2D, Dropout
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.initializers import RandomNormal
import tensorflow as tf
from config import CONFIG



############################################################
################# Model Generation #########################
############################################################

def build_generator(input):
    print("Building Generator")
    layers = Conv2D(64,(3,3),padding='same')(input)
    layers = PReLU(shared_axes=[1,2])(layers)
    x = layers
    
    for _ in range(16):
        layers = residual_block(layers)
    
    layers = Conv2D(64, (3,3), padding="same")(layers)
    layers = BatchNormalization(momentum=0.5)(layers)
    layers = add([layers,x])
    layers = upscale_block(layers)
    layers = upscale_block(layers)
    op = Conv2D(3, (9,9), padding="same")(layers)
    return Model(inputs=input, outputs=op)

def build_discriminator(input):
    print("Building Discriminator")
    filters = 64
    
    layer1 = discriminator_block(input,  filters,normalization=False)
    layer2 = discriminator_block(layer1, filters,2)
    layer3 = discriminator_block(layer2, filters*2)
    layer4 = discriminator_block(layer3, filters*2, stride=2)
    layer5 = discriminator_block(layer4, filters*4)
    layer6 = discriminator_block(layer5, filters*4, stride=2)
    layer7 = discriminator_block(layer6, filters*8)
    layer8 = discriminator_block(layer7, filters*8, stride=2)

    d8_5 = Flatten()(layer8)
    d9 = Dense(filters*16)(d8_5)
    d10 = LeakyReLU(alpha=0.2)(d9)
    final = Dense(1, activation='sigmoid')(d10)
    return Model(input, final)

def build_vgg():
    print("Building VGG")
    vgg = VGG19(weights="imagenet",
                include_top=False,
                input_shape=CONFIG.INPUT_SHAPE)
    vgg.outputs = [vgg.layers[9].output]
    
    img = Input(shape=CONFIG.INPUT_SHAPE)
    x = img
    for layer in range(10):
        vgg.layers[layer].trainable = False
        x = vgg.layers[layer](x)
    
    
    return Model(inputs=img, outputs=x )


def build_gan(generator,discriminator,vgg,lr,hr):
    print("Building GAN")
    generated_img = generator(lr)
    generated_features = vgg(generated_img)
    discriminator.trainable = False
    real = discriminator(generated_img)
    return Model([lr,hr],[real,generated_features])


def discriminator_block(input, filters, stride=1,normalization=True):
    layer = Conv2D(filters, (3,3),stride,padding='same')(input)
    layer = LeakyReLU(alpha=.2)(layer)
    if normalization:
        layer = BatchNormalization(momentum=.8)(layer)
        
    return layer

def residual_block(x):
    residual = Conv2D(64, (3,3), padding='same')(x)
    residual = BatchNormalization(momentum = .5)(residual)
    residual = PReLU(shared_axes = [1,2])(residual)
    residual = Conv2D(64, (3,3), padding= "same")(residual)
    residual = BatchNormalization(momentum= .5)(residual)

    return add([x,residual])

def upscale_block(x):
    upscaled = Conv2D(256, (3,3), padding="same")(x)
    upscaled = UpSampling2D( size = 2 )(upscaled)
    upscaled = PReLU(shared_axes=[1,2])(upscaled)

    return upscaled
