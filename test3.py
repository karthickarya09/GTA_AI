from random import shuffle
import numpy as np
import cv2
def main():
    x = np.load('w-training_data-1.npy')

    print(len(x))
    shuffle(x)
    print(len(x))

    np.save('TEST_save3.npy', x)
    # while True:
    #     for data in x:
    #         cv2.imshow("Test-{}".format(data[1]), data[0])
    #         if cv2.waitKey(25) & 0xFF == ord('q'):
    #             cv2.destroyAllWindows()
    #             break
main()