"""
In this file, we apply the polluting technique for data augmentation.
We also have a function for image renaming, in case we need it.
"""

import wand
from wand.image import Image
import os
import cv2 as cv

folder = "C:\\Users\\afons\\Desktop\\New folder\\Cadeiras\\Curso Samsung\\Projeto\\Datasets\\Model\\Model again\\without clothes\\Testing - Copy (2) clothless"


for filename in os.listdir(folder):
        

    img = os.path.join(folder,filename)
    if img is not None:

        with Image(filename=img) as img_pol:

            if not os.path.exists(folder + '_polluted'):        
                os.mkdir(folder + '_polluted')

            img_pol.noise("laplacian", attenuate = 1.0)
            img_pol.save(filename = folder + '_polluted' + os.sep + filename.split(".")[0] + '_polluted.jpg')



def rename_images():

    folder = "C:\\Users\\afons\\Desktop\\New folder\\Cadeiras\\Curso Samsung\\Projeto\\Datasets\\Model\\Model again\\without clothes\\Testing clothless - Copy"
    value = input("Enter extra value to add to the image name:")

    for filename in (os.listdir(folder)):

        filename_temp = filename.split(".")[0]

        dst = f"{filename_temp}{value}.jpg"
        src =f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
        dst =f"{folder}/{dst}"

        # rename() function will
        # rename all the files
        os.rename(src, dst)


rename_images()

    