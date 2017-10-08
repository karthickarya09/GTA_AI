import numpy as np


def main():
    for i in range(1,21):
        training_data = []
        print(i)
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
        print(len(training_data))

main()