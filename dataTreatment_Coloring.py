"""
In this coding file, we apply the coloring filter to the images.
"""



import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import sys
import cv2
from string import digits
from fimage import FImage
from fimage.filters import Sepia
from fimage.presets import Preset
from fimage.filters import Contrast, Brightness, Saturation
from fimage.filters import Contrast

class MyOwnPreset(Preset):
    transformations = [
        Contrast(30),
        Saturation(50),
        Brightness(10),
    ]


def load_images_from_folder(folder):

    # folder = input("Enter folder name:")

    toolbar_width = 50

    # setup toolbar
    print("Loading...")
    sys.stdout.write("|%s|" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

    counter = 0
    percentage = 0
    folderSize = 0
    barsCounted = 0
    images = []
    labels = []
    final_labels = []
    label_dict = {}

    for _, _, files in os.walk(folder):
        
        for f in files:
            
            folderSize += 1

    for filename in os.listdir(folder):

        temp = str.maketrans('', '', digits)
        label = filename.translate(temp)

        labels.append(label.split("_")[0])
    
    labels = np.unique(labels)

    for i,l in enumerate(labels):
        label_dict[l] = i

    for filename in (np.array(os.listdir(folder))):

        img = FImage(os.path.join(folder,filename))

        if img is not None:

            #img.apply(MyOwnPreset())
            img.apply(Sepia(40))

            images.append(img)

            temp = str.maketrans('', '', digits)
            label = filename.translate(temp)

            final_labels.append(label_dict[label.split("_")[0]])

            if not os.path.exists(folder + '_colored'):        
                os.mkdir(folder + '_colored')

            img.save(fp = folder + '_colored' + os.sep + filename.split(".")[0] + '_colored.jpg')

            counter += 1

            percentage = int(counter/folderSize * 100)

            if(percentage > barsCounted*2):
                sys.stdout.write("█")
                sys.stdout.flush()
                barsCounted += 1

    sys.stdout.write("|\n")

    print("All images have been loaded!")
    print("Labels counter:" + str(len(labels)))
    print("Images counter:" + str(len(images)))
    return (np.asarray(images), np.array(final_labels))


path = "C:\\Users\\afons\\Desktop\\New folder\\Cadeiras\\Curso Samsung\\Projeto\\Datasets\\Model\\Model again\\without clothes\\Testing clothless100x100"
load_images_from_folder(path)