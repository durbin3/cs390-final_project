import numpy as np 
from model import *
from config import *
from utils import *
from datagenerator import DataGenerator
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
                            CONFIG.LR_DIR,
                        CONFIG.INPUT_SHAPE,
                        down_sample_scale=CONFIG.DOWN_SAMPLE_SCALE,
                        batch_size=CONFIG.BATCH_SIZE)
    
    print("DataGen Length: ", datagen.__len__())
    return datagen


def buildModel(lr_shape,hr_shape):
    generator = build_generator(lr_shape)
    discriminator = build_discriminator(hr_shape)
    discriminator.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    vgg = build_vgg()
    vgg.trainable = False
    
    gan = build_gan(generator,discriminator,vgg,lr_shape,hr_shape)
    gan.compile(loss=['binary_crossentropy','mse' ],loss_weights=[1e-3,1],optimizer='adam')
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
            print("\tStep: ",step, "/",min(10000,data.__len__()), end='\r')
            if (step >= min(10000,data.__len__())):
                print()
                step = 0
                break

            gen_imgs = generator.predict_on_batch(lr_batch)


            #Train the discriminator
            if step %50 == 0:
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
        data.on_epoch_end()
    return gan


############################################################
############################################################

if __name__ == '__main__':
    main()
    
    