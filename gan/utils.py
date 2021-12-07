import os
from config import CONFIG
from PIL import Image
import numpy as np
import tensorflow as tf 
import tensorflow.keras as keras
import tensorflow.keras.backend as K


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
    # return keras.applications.vgg19.preprocess_input(img)
    return img/255


def normalize_image(img, min_value=-1):
    return img / 255.0
    # return (img - 127.5) / 127.5


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
        return np.clip(img,0,255).astype(np.uint8)
    else:
        return tf.multiply(img,255)

def predict_random_image(generator, epoch=None, show=False):
    create_dir_if_not_exist('images')
    images = os.listdir(CONFIG.HR_DIR)
    img = Image.open(f'{CONFIG.HR_DIR}/{np.random.choice(images, 1)[0]}')
    lr_shape = tuple([CONFIG.INPUT_SHAPE[i] // CONFIG.DOWN_SAMPLE_SCALE for i in range(2)])
    lr = img.resize(lr_shape)
    sr_img = generator.predict(np.array(lr)[np.newaxis, ...])
    output_image = Image.fromarray(denormalize_image(sr_img[0]))
    if show:
        img.show()
        lr.show()
        output_image.show()
    else:
        if epoch is None:
            img.save("images/hr.jpg")
            lr.save("images/lr.jpg")
            output_image.save("images/generated.jpg")
            Image.fromarray(np.clip(sr_img[0]*255,0,255).astype(np.uint8)).save("images/generated_2.jpg")
            Image.fromarray(deprocess_image(sr_img[0])).save("images/generated_keras.jpg")
            Image.fromarray(np.clip((sr_img[0]+1)*127.5,0,255).astype(np.uint8)).save("images/generated_3.jpg")
        else:
            path = f"images/epoch_{epoch}/"
            img.save(f"{path}hr.jpg")
            lr.save(f"{path}lr.jpg")
            output_image.save(f"{path}generated.jpg")
            Image.fromarray(np.clip(sr_img[0]*255,0,255).astype(np.uint8)).save(f"{path}generated_2.jpg")
            Image.fromarray(deprocess_image(sr_img[0])).save(f"{path}generated_keras.jpg")
            Image.fromarray(np.clip((sr_img[0]+1)*127.5,0,255).astype(np.uint8)).save(f"{path}generated_3.jpg")



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