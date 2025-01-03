import numpy as np


class Layer:
    def __init__(self):
        self.params = {}
        self.grads = {}
        self.input_shape = None
        self.output_shape = None
        
    def forward(self, inputs):
        """前向传播"""
        raise NotImplementedError
        
    def backward(self, grad_output):
        """反向传播"""
        raise NotImplementedError
        
    def _init_output_shape(self):
        """初始化输出形状"""
        raise NotImplementedError

