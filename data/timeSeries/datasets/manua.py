import numpy as np

class mauna():
    @staticmethod
    def get_Xy():
        f = open('data/timeSeries/data/mauna.txt')
        contents = f.readlines()
        X = []
        y = []
        for line in contents:
            line = line.strip().split(' ')
            xx, yy = line[0], line[1]
            X.append(float(xx))
            y.append(float(yy))

        return np.array(X, dtype=np.double), np.array(y, dtype=np.double)
