from keras import Model
from keras.layers import Conv2D, PReLU, BatchNormalization, Flatten
from keras.layers import UpSampling2D, LeakyReLU, Dense, Input, add
from keras.applications.vgg19 import VGG19
import pandas as pd
from PIL import Image
import numpy as np 
import os
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt
from config import *
from datagen import DataGenerator
from utils import *

def main():
    data = get_data()
    lr_input = Input(shape=(CONFIG.INPUT_SHAPE[0]//4,CONFIG.INPUT_SHAPE[1]//4,3))
    hr_input = Input(shape=CONFIG.INPUT_SHAPE)
    generator,discriminator,vgg,gan = buildModel(lr_input,hr_input)
    gan = trainModel(gan,generator,discriminator,vgg,data)

############################################################
##################### Model Pipeline #######################
############################################################
def get_data():
    datagen = DataGenerator(CONFIG.HR_DIR,
                        CONFIG.INPUT_SHAPE,
                        down_sample_scale=CONFIG.DOWN_SAMPLE_SCALE,
                        batch_size=CONFIG.BATCH_SIZE)
    return datagen


def buildModel(lr_shape,hr_shape):
    generator = build_generator(lr_shape)
    discriminator = build_discriminator(hr_shape)
    discriminator.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    vgg = build_vgg()
    vgg.trainable = False
    
    gan = build_gan(generator,discriminator,vgg,lr_shape,hr_shape)
    gan.compile(loss=['binary_crossentropy',wasserstein_loss ],loss_weights=[1e-3,1],optimizer='adam')
    return generator,discriminator,vgg,gan

def trainModel(gan,generator,discriminator,vgg,data):
    print("Training")
    epochs = 1000
    for e in range(epochs):
        print("Epoch: ",e)
        # gen_label = np.zeros((CONFIG.BATCH_SIZE, 1))
        gen_label = create_noisy_labels(0,CONFIG.BATCH_SIZE)
        real_label = create_noisy_labels(1,CONFIG.BATCH_SIZE)
        # real_label = np.ones((CONFIG.BATCH_SIZE,1))
        g_losses = []
        d_losses = []
        for step, (lr_batch,hr_batch) in enumerate(data):
            print("\tStep: ",step, "/",min(500,data.__len__()), end='\r')
            if (step >= min(500,data.__len__())):
                print()
                step = 0
                break

            gen_imgs = generator.predict_on_batch(lr_batch)


            #Train the discriminator
            if step % 50 == 0:
                discriminator.trainable = True
                d_loss_gen = discriminator.train_on_batch(gen_imgs,gen_label)
                d_loss_real = discriminator.train_on_batch(hr_batch,real_label)
                d_loss = 0.5 * np.add(d_loss_gen, d_loss_real)
                d_losses.append(d_loss)
                discriminator.trainable = False
            
            
            #Train the generator
            image_features = vgg.predict(hr_batch)
            g_loss, _, _ = gan.train_on_batch([lr_batch, hr_batch], [real_label, image_features])

            g_losses.append(g_loss)
        g_losses = np.array(g_losses)
        d_losses = np.array(d_losses)

        g_loss = np.sum(g_losses, axis=0) / len(g_losses)
        d_loss = np.sum(d_losses, axis=0) / len(d_losses)
        print("\tg_loss:", g_loss, "d_loss:", d_loss)
        predict_random_image(generator)
    return gan


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

def wasserstein_loss(y_true, y_pred):
	return K.mean(y_true * y_pred)
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

############################################################
############################################################

if __name__ == '__main__':
    main()
    
    