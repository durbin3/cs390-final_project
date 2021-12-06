import os
from PIL import Image
from  skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt

width,height = 16
def down_sample_image(folder,name):
    path = folder + name
    raw_image = Image.open(path)
    image = np.array(raw_image)
    new_image = resize(image, (width,height), anti_aliasing=False)
    plt.imsave("low_res/"+name,new_image)
    

if not os.path.exists("low_res"):
    os.mkdir("low_res")

i = 0
for image in os.listdir("high_res"):
    if i % 21 == 0:
        print("Downscaling image: ", i,"\t",image)
    down_sample_image("high_res/",image)
    i += 1
    
print("Done Downscaling Images")

