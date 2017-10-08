from random import shuffle
import numpy as np

def main():
    
    w_count=0
    nk_count=0
    for i in range(1,21):
        print(i)
        training_data = np.load('w-training_data-{}.npy'.format(i))
        w_count = w_count + len(training_data)
        # shuffle(training_data)
        # training_data = training_data[:int(len(training_data)/3)]
        # np.save('w-training_data-{}.npy'.format(i), training_data)
        training_data = np.load('nk-training_data-{}.npy'.format(i))
        nk_count = nk_count + len(training_data)
        # shuffle(training_data)
        # training_data = training_data[:int(len(training_data)/2)]
        # np.save('nk-training_data-{}.npy'.format(i), training_data)
    print(w_count, nk_count)
    # while True:
    #     for data in x:
    #         cv2.imshow("Test-{}".format(data[1]), data[0])
    #         if cv2.waitKey(25) & 0xFF == ord('q'):
    #             cv2.destroyAllWindows()
    #             break
main()