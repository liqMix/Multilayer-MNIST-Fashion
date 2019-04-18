import math

# Sigmoid activation definition

EPSILON = 0.00001
def sigmoid(x):
    try:
        denom = 1 + math.exp(-x)
    except OverflowError:
        return EPSILON
    return 1 / denom


def arctan(x):
    return math.atan(x)