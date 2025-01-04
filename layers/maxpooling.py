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
        """
        前向传播
        :param inputs: 输入数据，形状为(batch_size, height, width, channels)
        """
        if self.input_shape is None:
            self.input_shape = inputs.shape
            self._init_output_shape()
            
        batch_size = inputs.shape[0]
        output = np.zeros((batch_size,) + self.output_shape[1:])
        
        # 执行池化操作
        for i in range(self.output_shape[1]):
            for j in range(self.output_shape[2]):
                h_start = i * self.stride[0]
                h_end = h_start + self.pool_size[0]
                w_start = j * self.stride[1]
                w_end = w_start + self.pool_size[1]
                
                window = inputs[:, h_start:h_end, w_start:w_end, :]
                output[:, i, j, :] = np.max(window, axis=(1,2))
        
        return output
