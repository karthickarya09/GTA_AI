import numpy as np
from random import shuffle

def main():
    count = 0
    for i in range(1,21):
        print(i)
        training_data = np.load('w-training_data-{}.npy'.format(i))
        shuffle(training_data)
        np.save('w-training_data-{}'.format(i), training_data)

        training_data = np.load('a-training_data-{}.npy'.format(i))
        shuffle(training_data)
        np.save('a-training_data-{}'.format(i), training_data)

        training_data = np.load('s-training_data-{}.npy'.format(i))
        shuffle(training_data)
        np.save('s-training_data-{}'.format(i), training_data)

        training_data = np.load('d-training_data-{}.npy'.format(i))
        shuffle(training_data)
        np.save('d-training_data-{}'.format(i), training_data)

        training_data = np.load('wa-training_data-{}.npy'.format(i))
        shuffle(training_data)
        np.save('wa-training_data-{}'.format(i), training_data)

        training_data = np.load('wd-training_data-{}.npy'.format(i))
        shuffle(training_data)
        np.save('wd-training_data-{}'.format(i), training_data)

        training_data = np.load('sa-training_data-{}.npy'.format(i))
        shuffle(training_data)
        np.save('sa-training_data-{}'.format(i), training_data)

        training_data = np.load('sd-training_data-{}.npy'.format(i))
        shuffle(training_data)
        np.save('sd-training_data-{}'.format(i), training_data)

        training_data = np.load('nk-training_data-{}.npy'.format(i))
        shuffle(training_data)
        np.save('nk-training_data-{}'.format(i), training_data)
        
main()