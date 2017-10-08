import numpy as np
from keras.layers import Input
from random import shuffle
print("Before Keras")
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, AveragePooling2D, Flatten
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import SGD
from keras.models import load_model
import matplotlib.pyplot as plt
print("Keras Imported")

FILE_I_END = 20


WIDTH = 288
HEIGHT = 162
LR = 1e-3
EPOCHS = 11

ord_1 = [8, 17, 4, 15, 6, 19]
ord_2 = [9, 15, 18, 10, 3, 16]
ord_3 = [7, 13, 11]
order = ord_3


def main():
    base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(WIDTH, HEIGHT,3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(9, activation='softmax')(x)
    
    graph_data=[]
    MODEL_NAME='inception_model_32_1'

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    if(MODEL_NAME!=''):
        model = load_model('{}.h5'.format(MODEL_NAME))
        print("Model Exists and Loaded")
    else:
        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in base_model.layers:
            layer.trainable = False
        print("Started Compiling")
        sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        print("Model Compiled")

    print("Starting Training and Epochs.")

    for i in order:
        if(MODEL_NAME !=''):
            model = load_model('{}.h5'.format(MODEL_NAME))
            print("Model Loaded")
        print("Dataset-{}".format(i))
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
            history = model.fit(X_train, Y_train, batch_size=32, epochs=1, shuffle=True, verbose=1, validation_data=(X_valid, Y_valid))
            X_train=Y_train=X_valid=Y_valid=[]
            model.save("{}.h5".format(MODEL_NAME))
            if(i!=order[-1]):
                del model
            acc = history.history['acc'][0]
            val_acc = history.history['val_acc'][0]
            loss = history.history['loss'][0]
            val_loss = history.history['val_loss'][0]
            graph_data.append([acc, val_acc, loss, val_loss]) 
            del history
            
        except Exception as ex:
            print(str(ex))

    print("Training Complete! Saving Model")
    model.save("inception_model_32_1.h5")
    model.save("inception_model_32_19.h5")
    del model

    x = []

    temp = np.load("graph_info.npy")
    print("Graph file LOADED")
    for data in temp:
        x.append(data)
    temp=[]
    for data in graph_data:
        x.append(data)
    graph_data=[]

    print("Generating Graph")
    acc = [data[0] for data in x]
    val_acc = [data[1] for data in x]
    loss = [data[2] for data in x]
    val_loss = [data[3] for data in x]
    # print(acc)
    # print(val_acc)
    # print(loss)
    # print(val_loss)
    plt.plot(acc)
    plt.plot(val_acc)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    np.save("graph_info.npy", x)
    x = []
    print("Graph Saved")
main()