import math
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt

Xin = np.linspace(-1.0, 1.0, 101)


def yTraining(x):
    return float(-1.0 + 2.0 * (math.sin(math.pi / 2.0 * (x + 1.0))**4 + 0.25 * math.sin(math.pi / 2.0 * (1 - abs(math.cos(math.pi / 4.0 * (x + 1.0))))**4)**4))

def activationFunction(x):
    return math.tanh(x)

def yOut(xin, W):
    yout = 0.0;
    for k in range(5):
        yout += W[k+10] * activationFunction(W[k]*xin + W[k+5]);

    yout += W[15]
    return yout

def optimizationFunction(W):
    f = 0.0;
    for w in W:
        if (abs(w) > 5.0):
            return 10000.0
    for x in Xin:
        f += (yOut(x, W) - yTraining(x))**2
    return float(f / 101.0)


if __name__ == '__main__':
    X0 = np.random.uniform(low=-5.0, high=5.0, size=16)
    resW = []
    res = 0
    i = 0
    while(True):
        res = opt.minimize(optimizationFunction, X0, args=(), method='nelder-mead', options={'ftol' : 1e-10, 'xtol': 1e-8})
        if res.fun <= 1e-4:
            resW = res.x
            break
        X0 = res.x
        resW = res.x
        i += 1

    print("Result of function is: {}".format(res.fun))
    print("Weights are:")
    i = 0
    for w in resW:
        i += 1
        print("{}. {}".format(i, w))
    yTrainingData = []
    yOutData = []
    Xin = np.linspace(-1, 1, 100)
    for x in Xin:
        yTrainingData.append(yTraining(x))
        yOutData.append(yOut(x, resW))

    plt.plot(Xin, yTrainingData, label='y_training')
    plt.plot(Xin, yOutData, 'o', label='y_out')
    plt.legend()
    plt.show()





