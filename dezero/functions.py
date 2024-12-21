
import numpy as np
import dezero
from dezero.core import Function, Variable, as_variable, as_array

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs   #inputs: [y.grad(var instance)] -> unpacking
        gx = gy * cos(x)
        return gx

def sin(x):
    return Sin()(x)

class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx

def cos(x):
    return Cos()(x)

class Exp(Function):
    def forward(self ,x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # outputs: [weakref(var)]
        gx = gy * y
        return gx

def exp(x):
    return Exp()(x)

class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]() # weakref
        gx = (1- y ** 2)*gy
        return gx

def tanh(x):
    return Tanh()(x)
