import numpy as np
from layers.conv2d import Conv2D
from layers.maxpooling import MaxPool2D
from layers.fc import FullyConnected
from layers.activation import ReLU

class Model:
    def __init__(self):
        self.layers = []
        
    def add(self, layer):
        """添加层到模型中"""
        self.layers.append(layer)
        
    def forward(self, inputs):
        """前向传播"""
        x = inputs
        for layer in self.layers:
            x = layer.forward(x)
        return x
        
    def backward(self, grad_output):
        """反向传播"""
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
        
    def get_params_and_grads(self):
        """获取所有参数和梯度"""
        params_and_grads = []
        for layer in self.layers:
            for param_name, param in layer.params.items():
                grad = layer.grads[param_name]
                params_and_grads.append((param, grad))
        return params_and_grads



