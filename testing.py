#Our main file

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import train
from tensorflow import keras
from keras import layers
import cv2
import os
import re
import sys
from string import digits
import math


from keras.models import Sequential # We will use the Sequential API.
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

from keras.optimizers import Adam


def runModel():


    folder = "C:\\Users\\afons\\Desktop\\New folder\\Cadeiras\\Curso Samsung\\Projeto\\Datasets\\Model\\Model again\\without clothes\\Train_colored"


    label_dict = get_labels(folder) # dictionary with all 5 labels and the corresponding value

    number_of_batches = 7 # nr divisions dataset
    list_of_train_images_path = shuffle_list(folder) # shuffling dataset
    batches = cut_list(list_of_train_images_path, number_of_batches) # dividing dataset

    myModel = None

    learn_rate = 0.00005

    loss_history_callback = save_losses("trash_iD")



    path2 = "C:\\Users\\guilh\\OneDrive\\Ambiente de Trabalho\\uni\\3o ano\\Rec.ai.cle\\Testing"
    path0 = "C:\\Users\\afons\\Desktop\\New folder\\Cadeiras\\Curso Samsung\\Projeto\\Datasets\\Model\\Model again\\without clothes\\Test_colored"

    (testX, testY) = load_images_from_folder(path0) #loading in test images
    
    n_epochs = 20 #20
    n_batch_size = 4 #4

    my_optimizer=Adam(learning_rate=learn_rate)

    accuracy_list = []

    counter = 0

    for batch in batches:
        
        counter +=1

        images_batch_X, images_batch_Y = data_load(folder,batch, label_dict) #loading in batch of training images

        if myModel == None: myModel = createModel(images_batch_X, images_batch_Y) #singleton to reassure only one model is used (batch of images only used for sizing matters)

        myModel.compile(loss = "sparse_categorical_crossentropy", optimizer = my_optimizer, metrics=["accuracy"])

        history = myModel.fit(images_batch_X, images_batch_Y, epochs=n_epochs, batch_size = n_batch_size, callbacks=[loss_history_callback], validation_split=0.3)

        ACC = (myModel.evaluate(testX, testY, verbose=0))[1]
        accuracy_list.append(ACC)
        print("Test Accuracy ", counter , " : {}".format(np.round(ACC,3)))



        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()


        plt.plot(loss_history_callback.losses)

        plt.show()

def createModel(trainX, trainY):

    #creating the model and adding all the filters and layers

    shape_image = trainX[0].shape
    n_labels = len(np.unique(trainY))

    
    my_model = Sequential()
    my_model.add(Conv2D(input_shape=shape_image,filters=128,kernel_size=(3,3), activation="relu"))
    my_model.add(MaxPooling2D(pool_size=(2,2),padding='same', strides= 2))
    my_model.add(Dropout(rate = 0.1))
    my_model.add(Conv2D(filters=128,kernel_size=(3,3), activation="relu"))
    my_model.add(MaxPooling2D(pool_size=(2,2),padding='same', strides= 2))
    my_model.add(Dropout(rate = 0.2))
    my_model.add(Conv2D(filters=128,kernel_size=(3,3), activation="relu"))
    my_model.add(MaxPooling2D(pool_size=(2,2),padding='same', strides= 2))
    my_model.add(Dropout(rate = 0.3))
    my_model.add(Conv2D(filters=128,kernel_size=(3,3), activation="relu"))
    my_model.add(MaxPooling2D(pool_size=(2,2),padding='same', strides= 2))
    my_model.add(Dropout(rate = 0.5))
    my_model.add(Conv2D(filters=128,kernel_size=(3,3), activation="relu"))
    my_model.add(MaxPooling2D(pool_size=(2,2),padding='same', strides= 2))
    my_model.add(Flatten())
    my_model.add(Dense(units = 2048, activation="relu"))
    my_model.add(Dropout(rate=0.7))
    my_model.add(Dense(units = 2048, activation = 'relu'))
    my_model.add(Dense(units = n_labels, activation="softmax"))

    return my_model


class save_losses(tf.keras.callbacks.Callback): 
    
    #Save the losses, mse, and accuracy of a model to a file in the intermediates folder
    
    def __init__(self, model_name):
        # Make sure the test model save path exists
        if not os.path.exists("intermediates/"):
            os.mkdir("intermediates/")
        self.model_name = model_name

        self.metric_file = "intermediates/metrics.txt"

    def on_train_begin(self, logs={}):
        self.losses = []
        self.mses = []
        self.acc = []

    # Save to a file when the training ends
    def on_train_end(self, logs={}):
        np.savetxt(self.metric_file, self.losses)

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))



def shuffle_list(folder):

    print("Shuffling list...")

    images = []

    for filename in os.listdir(folder):

        images.append(filename)
        
    list_of_train_images_path = np.random.permutation(images)

    print("List shuffled!")
    
    return list_of_train_images_path 


def data_load(folder, batch, label_dict):

    #loading images from training folder

    toolbar_width = 50
    counter = 0
    barsCounted = 0

    # setup toolbar
    print("Loading batch...")
    sys.stdout.write("|%s|" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

    images = []

    final_labels= []

    for filename in batch:

        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)

            temp = str.maketrans('', '', digits)
            label = filename.translate(temp)

            final_labels.append(label_dict[label.split("_")[0]])

            counter += 1

            percentage = int(counter/len(batch) * 100)

            if(percentage > barsCounted*2):
                sys.stdout.write("█")
                sys.stdout.flush()
                barsCounted += 1

    sys.stdout.write("|\n")
    print("Batch loaded!")

    return (np.asarray(images), np.array(final_labels))


def get_labels(folder):

    print("Gathering labels...")

    label_dict = {}
    labels = []

    for filename in os.listdir(folder):

        temp = str.maketrans('', '', digits)
        label = filename.translate(temp)

        labels.append(label.split("_")[0])

    final_labels, count = np.unique(labels, return_counts= True)

    createGraphic(final_labels, count)

    for i,l in enumerate(final_labels):
        label_dict[l] = i

    print("Labelling done!")

    return label_dict

def createGraphic(x_axis, y_axis):

    plt.bar(x_axis, y_axis)
    plt.title('Garbage distribution')
    plt.xlabel('Types of garbage')
    plt.ylabel('Count')
    plt.show()





def cut_list(shuffled_images, n_cycles):

    print("Cutting list")

    img_list = []
    size = len(shuffled_images)
    images_per_batch = math.floor(size/n_cycles)
    beginning = 0 
    count = 1

    for n in range(n_cycles):
        remainder = size - images_per_batch*(count-1)
        if remainder < images_per_batch:
            break
        else:
            img_list.append(shuffled_images[beginning:(images_per_batch*count)])
        beginning = math.floor((size/n_cycles))*count
        count += 1

    print("Cutting done!")

    return np.asarray(img_list)




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

    indexes = (np.random.permutation(len(os.listdir(folder))))

    for filename in (np.array(os.listdir(folder)))[indexes]:#[indexes]:

        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)

            temp = str.maketrans('', '', digits)
            label = filename.translate(temp)

            final_labels.append(label_dict[label.split("_")[0]])

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


runModel()