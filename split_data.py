import numpy as np
import pandas as pd
from random import shuffle
from collections import Counter

def main():
    w = []
    a = []
    s = [] 
    d = [] 
    wa = []
    wd = []
    sa = []
    sd = []
    nk = []
    wlen = 0
    alen = 0
    slen = 0
    dlen = 0
    walen = 0
    wdlen = 0
    salen = 0
    sdlen = 0
    nklen = 0
    for i in range(1,201):
        print(i)
        
        training_data = np.load('training_data-{}.npy'.format(i))
        for data in training_data:
            img = data[0]
            choice = data[1]
            if choice == [1,0,0,0,0,0,0,0,0]:
                w.append([img, choice])
            elif choice == [0,1,0,0,0,0,0,0,0]:
                a.append([img, choice])
            elif choice == [0,0,1,0,0,0,0,0,0]:
                s.append([img, choice])
            elif choice == [0,0,0,1,0,0,0,0,0]:
                d.append([img, choice])
            elif choice == [0,0,0,0,1,0,0,0,0]:
                wa.append([img, choice])
            elif choice == [0,0,0,0,0,1,0,0,0]:
                wd.append([img, choice])
            elif choice == [0,0,0,0,0,0,1,0,0]:
                sa.append([img, choice])
            elif choice == [0,0,0,0,0,0,0,1,0]:
                sd.append([img, choice])
            elif choice == [0,0,0,0,0,0,0,0,1]:
                nk.append([img, choice])
            else:
                print("No Matches")
        wlen += len(w)
        alen += len(a)
        slen += len(s)
        dlen += len(d)
        walen += len(wa)
        wdlen += len(wd)
        salen += len(sa)
        sdlen += len(sd)
        nklen += len(nk)
        if(i%10==0):
            np.save('w-training_data-{}.npy'.format(int(i/10)), w)
            np.save('a-training_data-{}.npy'.format(int(i/10)), a)
            np.save('s-training_data-{}.npy'.format(int(i/10)), s)
            np.save('d-training_data-{}.npy'.format(int(i/10)), d)
            np.save('wa-training_data-{}.npy'.format(int(i/10)), wa)
            np.save('wd-training_data-{}.npy'.format(int(i/10)), wd)
            np.save('sa-training_data-{}.npy'.format(int(i/10)), sa)
            np.save('sd-training_data-{}.npy'.format(int(i/10)), sd)
            np.save('nk-training_data-{}.npy'.format(int(i/10)), nk)
            w = []
            a = []
            s = [] 
            d = [] 
            wa = []
            wd = []
            sa = []
            sd = []
            nk = []    
            
    print(wlen, alen, slen, dlen, walen, wdlen, salen, sdlen, nklen)
main()
