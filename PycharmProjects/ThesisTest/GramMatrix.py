import numpy as np
from keras import backend as K

def createGram(w,h):
    x = np.zeros((4,3))
    # y = np.zeros((6,6))

    y = np.copy(x)
    b = np.copy(x)
    if True:
        for i in range(0,3):
            # y = np.copy(x)
            for j in range(i,3):
                y[i][j]=64
                # print('X')
                # print(y)
                # z=np.dot(y, y.T)
                # print('Gram')
                # print(z)

        for i in range(0,3):
            # y = np.copy(x)
            for j in range(0,i):
                b[i][j]=64
                # print('X')
                # print(y)
                # c=np.dot(b, b.T)
                # print('Gram')
                # print(c)
    else:
        y[0][1]=1
        b[1][1]=1
        b[1][2]=1

    z=np.dot(y, y.T)
    c=np.dot(b, b.T)
    print('Y')
    print(y)
    print('First Gram')
    print(z)

    print('B')
    print(b)
    print('Second Gram')
    print(c)

    print('First difference')
    print((z-c)**2)

    print('Second difference')
    print((c-z)**2)
    return c


