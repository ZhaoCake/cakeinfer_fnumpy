import numpy as np
from .common_layer import Layer

class FullyConnected(Layer):
    def __init__(self, output_dim):
        """
        全连接层
        :param output_dim: 输出维度
        """
        super().__init__()
        self.output_dim = output_dim
        
    def _init_params(self, input_dim):
        """
        初始化权重和偏置
        :param input_dim: 输入维度
        """
        # 使用He初始化
        scale = np.sqrt(2.0 / input_dim)
        self.params['W'] = np.random.normal(
            0.0,
            scale,
            (input_dim, self.output_dim)  # 初始化权重, 形状为(input_dim, output_dim)
        )
        self.params['b'] = np.zeros(self.output_dim)  # 初始化偏置, 形状为(output_dim,)
        
    def _init_output_shape(self):
        """初始化输出形状"""
        self.output_shape = (self.input_shape[0], self.output_dim)
        
    def forward(self, inputs):
        """
        前向传播
        :param inputs: 输入数据，形状为(batch_size, input_dim)
        :return: 输出数据，形状为(batch_size, output_dim)
        """
        if self.input_shape is None:
            self.input_shape = inputs.shape
            # 如果输入是多维的（比如来自卷积层），需要将其展平
            if len(self.input_shape) > 2:
                input_dim = np.prod(self.input_shape[1:])
            else:
                input_dim = self.input_shape[1]
            self._init_params(input_dim)
            self._init_output_shape()
            
        self.inputs = inputs
        # 如果输入是多维的，需要将其展平
        if len(inputs.shape) > 2:
            inputs_reshaped = inputs.reshape(inputs.shape[0], -1)
        else:
            inputs_reshaped = inputs
            
        # 线性变换：y = Wx + b
        output = np.dot(inputs_reshaped, self.params['W']) + self.params['b']
        return output
        
    def backward(self, grad_output):
        """
        反向传播
        :param grad_output: 输出梯度，形状为(batch_size, output_dim)
        :return: 输入梯度，形状与输入数据相同
        """
        # 如果输入是多维的，需要将其展平
        if len(self.inputs.shape) > 2:
            inputs_reshaped = self.inputs.reshape(self.inputs.shape[0], -1)
        else:
            inputs_reshaped = self.inputs
            
        # 计算权重和偏置的梯度
        self.grads['W'] = np.dot(inputs_reshaped.T, grad_output)
        self.grads['b'] = np.sum(grad_output, axis=0)
        
        # 计算输入梯度
        grad_input = np.dot(grad_output, self.params['W'].T)
        
        # 如果原始输入是多维的，需要恢复其形状
        if len(self.inputs.shape) > 2:
            grad_input = grad_input.reshape(self.inputs.shape)
            
        return grad_input
