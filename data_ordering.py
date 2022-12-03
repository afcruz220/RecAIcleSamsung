"""This code was used to reorder the image labels. 
Since we wanted to join many datasets, we had to make sure that, for example, the image 'battery1' wasn't repeated.
Therefore, this is the code we used.
"""

import os, os.path



trash = 137 
glass = 1002
paper = 593
cardboard = 806
metal = 738
plastic = 893

imgs = {}
path = "/Users/afons/Desktop/New folder/Cadeiras/Curso Samsung/Projeto/Datasets/Original Garbage Classification/test/Vidrio" #path for folder with images
valid_images = [".jpg",".gif",".png",".tga"] #valid image types
n = 914

for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    name = 'glass' + str(n) + '.jpg' #here we put the label we want to use + number of the image
    current_name = path + '/' + str(f)
    new_name = path + '/' + str(name)
    os.rename(current_name, new_name)
    n  += 1

print(n-1)


