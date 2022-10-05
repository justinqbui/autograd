from scalar import Scalar
import random

class Tensor:
    def __init__(self, in_features: int):
        self.w = [Scalar(random.uniform(-1, 1)) for _ in range(in_features)]
    
    def __repr__(self):
        return f"Tensor({self.w})"
    
    def __call__(self, x: 'Tensor') -> 'Tensor':
        for w_i, x_i in zip(self.w, x.w):
            print(w_i, x_i)
        return sum((w_i * x_i for w_i, x_i in zip(self.w, x)))
    

class Linear:
    def __init__(self, 
    in_features: int, 
    out_features: int, 
    bias: bool=True):
        self.w = [Tensor(in_features) for _ in range(out_features)]
        if bias:
            self.b = Scalar(0.0)
    
    def __repr__(self):
        return f"Linear({self.w})"
    
    def __call__(self, x: 'Tensor') -> 'Tensor':
        return [w(x) for w in self.w]

class Conv2d:
    def __init__(self, 
    in_channels: int, 
    out_channels: int, 
    kernel_size: int, 
    stride: int, 
    padding: int, 
    bias: bool=True):
        
        self.w = [Tensor(kernel_size) for _ in range(out_channels)]
        if bias:
            self.b = Scalar(0.0)
        self.stride = stride
        self.padding = padding
    
    def __repr__(self):
        return f"Conv2d({self.w})"
    
    def __call__(self, x: 'Tensor') -> 'Tensor':
        pass
        
    

        

    

        