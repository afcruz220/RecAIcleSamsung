"""
In this file, we apply the image resizing plus the image rotation.
"""


from wand.image import Image as Imagewand
import os
import numpy as np
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt
import time
import sys
import cv2
from string import digits


def get_rotate(img, counter):
    if counter%4 == 1:
        image_scaled = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif counter%4 == 2:
        image_scaled = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        image_scaled = cv2.rotate(img, cv2.ROTATE_180)
    return image_scaled


def load_images_from_folder_resize(folder):

    toolbar_width = 50

    # setup toolbar
    print("Loading...")
    sys.stdout.write("|%s|" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

    defined_width = 100
    defined_height = 100

    barsCounted = 0 
    folderSize = 0
    counter = 0
    images = []

    for _, _, files in os.walk(folder):
        

        for f in files:
            
            folderSize += 1

    for filename in os.listdir(folder):

        img = cv.imread(os.path.join(folder,filename))

        if img is not None:

            height, width, _ = img.shape

            widthScale = defined_width / width
            heightScale = defined_height / height

            image_scaled = cv.resize(img, None, fx = widthScale , fy= heightScale)

            if (counter%4) != 0:
                image_scaled = get_rotate(image_scaled, counter)         

            height, width, _ = image_scaled.shape

            images.append(image_scaled)

            counter += 1

            percentage = int(counter/(folderSize) * 100)
            
            if(percentage > barsCounted*2):
                sys.stdout.write("█")
                sys.stdout.flush()
                barsCounted += 1

            if not os.path.exists(folder + '100x100'):        
                os.mkdir(folder + '100x100')

            cv2.imwrite(folder + '100x100' + os.sep + filename, image_scaled)

    sys.stdout.write("|\n")

    print(str(counter) + " images loaded and resized to " + str(defined_width) + "x" + str(defined_height))

    return np.array(images)


path1 = "C:\\Users\\afons\\Desktop\\New folder\\Cadeiras\\Curso Samsung\\Projeto\\Datasets\\Model\\Model again\\without clothes\\Train"
path2 = "C:\\Users\\afons\\Desktop\\New folder\\Cadeiras\\Curso Samsung\\Projeto\\Datasets\\Model\\Model again\\without clothes\\Test"

path3 = "C:\\Users\\afons\\Desktop\\New folder\\Cadeiras\\Curso Samsung\\Projeto\\Datasets\\Model\\Model again\\without clothes\\Everything clothless100x100_colored"
path4 = "C:\\Users\\afons\\Desktop\\New folder\\Cadeiras\\Curso Samsung\\Projeto\\Datasets\\Model\\Model again\\without clothes\\Testing clothless100x100_colored"



load_images_from_folder_resize(path3)


