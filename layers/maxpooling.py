import numpy as np
from .common_layer import Layer

class MaxPool2D(Layer):
    def __init__(self, pool_size, stride=None):
        super().__init__()
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.stride = stride if stride is not None else self.pool_size
        self.stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        
    def _init_output_shape(self):
        h = (self.input_shape[1] - self.pool_size[0])//self.stride[0] + 1
        w = (self.input_shape[2] - self.pool_size[1])//self.stride[1] + 1
        self.output_shape = (self.input_shape[0], h, w, self.input_shape[3])
        
    def forward(self, inputs):
        if self.input_shape is None:
            self.input_shape = inputs.shape
            self._init_output_shape()
            
        self.inputs = inputs
        output = np.zeros(self.output_shape)
        self.max_indices = np.zeros_like(output, dtype=np.int32)
        
        for i in range(self.output_shape[1]):
            for j in range(self.output_shape[2]):

                # 计算池化窗口的起始和结束位置
                h_start = i * self.stride[0]
                h_end = h_start + self.pool_size[0]
                w_start = j * self.stride[1]
                w_end = w_start + self.pool_size[1]
                
                window = inputs[:, h_start:h_end, w_start:w_end, :]  # 获取池化窗口
                output[:, i, j, :] = np.max(window, axis=(1,2))  # 计算池化窗口的最大值
                
                # 保存最大值的索引，用于反向传播
                window_reshaped = window.reshape(window.shape[0], -1, window.shape[-1])
                self.max_indices[:, i, j, :] = np.argmax(window_reshaped, axis=1)
                
        return output
        
    def backward(self, grad_output):
        grad_input = np.zeros_like(self.inputs)
        
        for i in range(self.output_shape[1]):
            for j in range(self.output_shape[2]):
                h_start = i * self.stride[0]
                h_end = h_start + self.pool_size[0]
                w_start = j * self.stride[1]
                w_end = w_start + self.pool_size[1]
                
                for b in range(self.input_shape[0]):  # batch size
                    for c in range(self.input_shape[3]):  # channels
                        idx = self.max_indices[b, i, j, c]
                        h_idx = idx // self.pool_size[1]
                        w_idx = idx % self.pool_size[1]
                        grad_input[b, h_start+h_idx, w_start+w_idx, c] += grad_output[b, i, j, c]
                        
        return grad_input
