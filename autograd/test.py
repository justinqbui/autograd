import numpy as np
from scalar import Scalar

def scalarAddTest():
    x = Scalar(3.0)
    y = Scalar(4.0)
    z = x + y
    assert z.value == 7.0

def scalarMultiplyTest():
    x = Scalar(3.0)
    y = Scalar(4.0)
    z = x * y
    assert z.value == 12.0

def scalarBackpropAdditionTest():
    x = Scalar(3.0)
    y = Scalar(4.0)
    z = x + y
    z.backward()
    assert x.grad == 1.0
    assert y.grad == 1.0

def scalarBackpropMultiplicationTest():
    x = Scalar(3.0)
    y = Scalar(4.0)
    z = x * y
    z.backward()
    assert x.grad == 4.0
    assert y.grad == 3.0

if __name__ == "__main__":
    scalarAddTest()
    scalarMultiplyTest()
    scalarBackpropAdditionTest()
    scalarBackpropMultiplicationTest()