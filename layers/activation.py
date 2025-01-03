import numpy as np
from .common_layer import Layer

class ReLU(Layer):
    def __init__(self):
        """ReLU激活函数"""
        super().__init__()
        
    def _init_output_shape(self):
        """初始化输出形状"""
        self.output_shape = self.input_shape
        
    def forward(self, inputs):
        """
        前向传播
        :param inputs: 输入数据
        :return: ReLU激活后的结果
        """
        if self.input_shape is None:
            self.input_shape = inputs.shape
            self._init_output_shape()
            
        self.inputs = inputs
        return np.maximum(0, inputs)
        
    def backward(self, grad_output):
        """
        反向传播
        :param grad_output: 输出梯度
        :return: 输入梯度
        """
        grad_input = grad_output.copy()
        grad_input[self.inputs <= 0] = 0
        return grad_input
