from random import shuffle

data = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19]
print(data)
shuffle(data)
print(data)
ord_1 = data[:6]
ord_2 = data[6:12]
ord_3 = data[12:]


print(ord_1)
print(ord_2)
print(ord_3)

