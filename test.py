import numpy as np
import cv2

def main():
    train_data = np.load('training_data-1.npy') 
    # for i in range(2,50):
    #     print(i)
    #     train_data += np.load(('training_data-{}.npy').format(i))
    for data in train_data:
        img = data[0]
        choice = data[1]
        cv2.imshow('Screen', img)
        print('Output', choice)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
main()