import time
from random import Random

from Net import Net


def rectU(x):
    if x > 0:
        return x
    return -x


"""
We define a model relating the wind speed as a function of the turbine speed: the power coefficient model.
the model: Power = Cp * (1/2) * air density * swept area * wind speed^3
we train the neural net (RNN) to vary the parameters of the model according to labeled data.
This data should be retrieved from a real wind speeed sensor, with the data from the turbine.
"""


def main():
    # seed random
    r = Random()
    r.seed(time.time())

    mapping = [[[i, r.random()] for i in range(4)] for _ in range(8)] + \
              [[[_, r.random()] for _ in range(4, 12)] for __ in range(0, 4)]
    net = Net(mapping, [r.random() for i in range(len(mapping))], [0 for _ in range(len(mapping))],
              [rectU], len(mapping) + 4)

    net.propagate([1 for _ in range(4)])


if __name__ == '__main__':
    main()
