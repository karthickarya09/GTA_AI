import numpy as np
from keras.layers import Input
from random import shuffle
print("Before Keras")
# from keras.models import Model
# from keras.layers import Dense, GlobalAveragePooling2D, AveragePooling2D, Flatten
# from keras.applications.inception_v3 import InceptionV3
# from keras.optimizers import SGD
# from keras.models import load_model
import matplotlib.pyplot as plt
print("Keras Imported")







def train():

    WIDTH = 288
    HEIGHT = 162
    
    graph_data=[]
    # this is the model we will train
    # model = load_model('inception_model_32_1_1.h5')
    print("Model Exists and Loaded")
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    print("Starting Training and Epochs.")

    for i in [1,2]:
        # model = load_model('inception_model_32_1_1.h5')
        # print("Model Loaded")
        # print("Dataset-{}".format(i))
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
            print("TRAIN SHAPE", np.shape(training_data))
            t = int(0.9*len(training_data))
            train = training_data[:t]
            print(len(train))
            test = training_data[t:]
            print(len(test))
            training_data = []
            X_train = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,3)
            print("X_Train shape", np.shape(X_train))
            Y_train = [i[1] for i in train]
            X_valid = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,3)
            Y_valid = [i[1] for i in test]
            #history = model.fit(X_train, Y_train, batch_size=32, epochs=1, shuffle=True, verbose=1, validation_data=(X_valid, Y_valid))
            X_train=Y_train=X_valid=Y_valid=[]
            # model.save("inception_model_32_1_1.h5")
            # if(i!=order[-1]):
            #     del model
            # acc = history.history['acc'][0]
            # val_acc = history.history['val_acc'][0]
            # loss = history.history['loss'][0]
            # val_loss = history.history['val_loss'][0]
            # graph_data.append([acc, val_acc, loss, val_loss]) 
            # del history
            
        except Exception as ex:
            print(str(ex))

    # print("Training Complete! Saving Model")
    # model.save("inception_model_32_1_1.h5")
    # model.save("inception_model_32_{}.h5".format(epoch))
    # del model

    # x = []

    # temp = np.load("graph_info_1.npy")
    # print("Graph file LOADED")
    # for data in temp:
    #     x.append(data)
    # temp=[]
    # for data in graph_data:
    #     x.append(data)
    # graph_data=[]

    # np.save("graph_info_1.npy", x)
    # x = []
    print("Graph Saved")
train()