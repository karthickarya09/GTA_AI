import numpy as np
import os
import pandas as pd
from collections import deque
from keras.layers import Input
from random import shuffle
print("Before Keras")
import keras.callbacks as kcb
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, AveragePooling2D, Flatten
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import SGD
from keras.models import load_model
print("Keras Imported")

FILE_I_END = 20

WIDTH = 288
HEIGHT = 162
LR = 1e-3
EPOCHS = 11



def main():
    cb = kcb.TensorBoard(log_dir='~/logs', histogram_freq=1, write_graph=True, write_grads=True, batch_size=100, write_images=False)
    base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(WIDTH, HEIGHT,3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(9, activation='softmax')(x)
    MODEL_NAME=''
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    if(MODEL_NAME!=''):
        model = model.load('{}.h5'.format(MODEL_NAME))
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
    # Learning rate is changed to 0.001
    
    print("Started Compiling")
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    print("Model Compiled")
    print("Starting Training and Epochs.")
    for e in range(1,EPOCHS):
        data_order = [i for i in range(1, FILE_I_END+1)]
        print(data_order)
        shuffle(data_order)
        print(data_order)
        for i in data_order:
            if(MODEL_NAME != ''):
                model = load_model('{}.h5'.format(MODEL_NAME))
            print("Epoch-{}, Dataset-{}".format(e, i))
            try:
                training_data = []
                temp = np.load('w-training_data-{}.npy'.format(i))
                for data in temp:
                    training_data.append([data[0], data[1]])
                temp = np.load('a-training_data-{}.npy'.format(i))
                for data in temp:
                    training_data.append([data[0], data[1]])
                temp = np.load('s-training_data-{}.npy'.format(i))
                for data in temp:
                    training_data.append([data[0], data[1]])
                temp = np.load('d-training_data-{}.npy'.format(i))
                for data in temp:
                    training_data.append([data[0], data[1]])
                temp = np.load('wa-training_data-{}.npy'.format(i))
                for data in temp:
                    training_data.append([data[0], data[1]])
                temp = np.load('wd-training_data-{}.npy'.format(i))
                for data in temp:
                    training_data.append([data[0], data[1]])
                temp = np.load('sa-training_data-{}.npy'.format(i))
                for data in temp:
                    training_data.append([data[0], data[1]])
                temp = np.load('sd-training_data-{}.npy'.format(i))
                for data in temp:
                    training_data.append([data[0], data[1]])
                temp = np.load('nk-training_data-{}.npy'.format(i))
                for data in temp:
                    training_data.append([data[0], data[1]])
                shuffle(training_data)
                shuffle(training_data)
                shuffle(training_data)
                t = int(0.9*len(training_data))
                train = training_data[:t]
                print(len(train))
                test = training_data[t:]
                print(len(test))
                training_data = []
                X_train = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,3)
                Y_train = [i[1] for i in train]
                X_valid = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,3)
                Y_valid = [i[1] for i in test]
                model.fit(X_train, Y_train, batch_size=64, epochs=1, shuffle=True, verbose=1, validation_data=(X_valid, Y_valid))
                MODEL_NAME="inception_model_64"
                model.save("{}.h5".format(MODEL_NAME))
                del model
            except Exception as ex:
                print(str(ex))
    
    print("Training Complete! Saving Model")
    model.save("inception_model_64_Final_model")

main()