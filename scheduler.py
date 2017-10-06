from random import shuffle
from train2 import train
import time

data = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19]
def main():
    for epoch in range(15,21):
        print("Epoch:", epoch)
        shuffle(data)
        shuffle(data)
        shuffle(data)
        ord_1 = data[:6]
        ord_2 = data[6:12]
        ord_3 = data[12:]
        for sequence in range(1,4):
            print("Sending ord_{}".format(sequence))
            if(sequence==1):
                train(epoch, ord_1)
            elif(sequence==2):
                train(epoch, ord_2)
            else:
                train(epoch, ord_3)
            print("Sleeping")
            time.sleep(5) 
main()
