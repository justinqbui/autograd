import torch
from scalar import Scalar
from layers import Tensor, Linear

epsilon = 1e-6

def scalarAddTest():
    x = Scalar(3.0)
    y = Scalar(4.0)
    z = x + y
    z.backward()

    a = torch.Tensor([3.0])
    a.requires_grad = True
    b = torch.Tensor([4.0])
    b.requires_grad = True
    c = a + b
    c.backward()

    assert abs(c.item() - z.value) < epsilon
    assert abs(a.grad.item() - x.grad) < epsilon
    assert abs(b.grad.item() - y.grad) < epsilon

def scalarMultiplyTest():
    x = Scalar(3.0)
    y = Scalar(4.0)
    z = x * y
    z.backward()

    a = torch.Tensor([3.0])
    a.requires_grad = True
    b = torch.Tensor([4.0])
    b.requires_grad = True
    c = a * b
    c.backward()

    assert abs(c.item() - z.value) < epsilon
    assert abs(a.grad.item() - x.grad) < epsilon
    assert abs(b.grad.item() - y.grad) < epsilon


def tensorTest():
    x = Tensor(2)
    y = Tensor(2)
    print(x, y)
    z = x(y)
    z.backward()

    a = torch.tensor([1.0, 2.0], requires_grad=True)
    b = torch.tensor([3.0, 4.0], requires_grad=True)
    c = (a * b).sum()
    c.backward()

if __name__ == "__main__":
    scalarAddTest()
    scalarMultiplyTest()
    tensorTest()
