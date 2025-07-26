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
        # 保存输入用于反向传播
        self.inputs = inputs
        
        # 执行池化操作并记录最大值位置
        self.max_indices = np.zeros_like(output, dtype=int)
        
        # 向量化操作替代部分循环
        for i in range(self.output_shape[1]):
            for j in range(self.output_shape[2]):
                h_start = i * self.stride[0]
                h_end = h_start + self.pool_size[0]
                w_start = j * self.stride[1]
                w_end = w_start + self.pool_size[1]
                
                window = inputs[:, h_start:h_end, w_start:w_end, :]
                # 获取最大值
                output[:, i, j, :] = np.max(window, axis=(1,2))
                # 记录最大值位置（简化处理，只记录一个位置）
                self.max_indices[:, i, j, :] = np.argmax(window.reshape(window.shape[0], -1, window.shape[3]), axis=1)
        
        return output
        
    def backward(self, grad_output):
        """
        反向传播
        :param grad_output: 输出梯度
        :return: 输入梯度
        """
        grad_input = np.zeros_like(self.inputs)
        
        # 使用向量化操作优化反向传播
        for i in range(self.output_shape[1]):
            for j in range(self.output_shape[2]):
                h_start = i * self.stride[0]
                h_end = h_start + self.pool_size[0]
                w_start = j * self.stride[1]
                w_end = w_start + self.pool_size[1]
                
                # 向量化操作替代部分循环
                idx = self.max_indices[:, i, j, :]  # (batch, channels)
                h_idx = idx // self.pool_size[1] + h_start  # (batch, channels)
                w_idx = idx % self.pool_size[1] + w_start   # (batch, channels)
                
                # 使用高级索引进行向量化赋值
                batch_idx = np.arange(grad_output.shape[0])[:, None]  # (batch, 1)
                channel_idx = np.arange(grad_output.shape[3])[None, :]  # (1, channels)
                
                grad_input[batch_idx, h_idx, w_idx, channel_idx] += grad_output[:, i, j, :]
        
        return grad_input