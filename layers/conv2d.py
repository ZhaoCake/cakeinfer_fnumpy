import numpy as np
from .common_layer import Layer
from utils.logger import logger

class Conv2D(Layer):
    def __init__(self, filters, kernel_size, stride=1, padding=0):
        """
        Conv2D 卷积层
        :param filters: 卷积核数量
        :param kernel_size: 卷积核大小
        :param stride: 步幅
        :param padding: 填充
        """
        super().__init__()
        self.filters = filters  # 卷积核数量
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
    def _init_params(self, input_channels):
        """
        初始化权重和偏置
        :param input_channels: 输入通道数
        """
        # 权重形状为(height, width, in_channels, filters)以适应NHWC格式
        scale = np.sqrt(2.0 / (input_channels * self.kernel_size[0] * self.kernel_size[1]))
        self.params['W'] = np.random.normal(
            0.0,
            scale,
            (self.kernel_size[0], self.kernel_size[1], input_channels, self.filters)
        )
        self.params['b'] = np.zeros(self.filters)
        
        logger.debug(f"Initialized Conv2D layer with:")
        logger.shape_info("Weights", self.params['W'])
        logger.shape_info("Bias", self.params['b'])
        
    def _init_output_shape(self):
        """初始化输出形状"""
        h = (self.input_shape[1] + 2*self.padding[0] - self.kernel_size[0])//self.stride[0] + 1
        w = (self.input_shape[2] + 2*self.padding[1] - self.kernel_size[1])//self.stride[1] + 1
        self.output_shape = (self.input_shape[0], h, w, self.filters)
        
    def _pad(self, x):
        """填充输入"""
        return np.pad(
            x,
            ((0,0), (self.padding[0],self.padding[0]), (self.padding[1],self.padding[1]), (0,0)),
            'constant'
        )
        
    def forward(self, inputs):
        """
        前向传播
        :param inputs: 输入数据，形状为(batch_size, height, width, channels)
        """
        if self.input_shape is None:
            self.input_shape = inputs.shape
            self._init_params(inputs.shape[-1])
            self._init_output_shape()
            logger.debug(f"Input shape: {self.input_shape}")
            logger.debug(f"Output shape: {self.output_shape}")
            
        self.inputs = inputs
        
        # 填充
        if self.padding[0] > 0 or self.padding[1] > 0:
            padded = self._pad(inputs)
        else:
            padded = inputs
            
        # 初始化输出
        batch_size = inputs.shape[0]  # 获取实际的批次大小
        output = np.zeros((batch_size,) + self.output_shape[1:])  # 使用实际批次大小
        
        try:
            # 执行卷积操作（向量化版本）
            for i in range(self.output_shape[1]):
                for j in range(self.output_shape[2]):
                    h_start = i * self.stride[0]
                    h_end = h_start + self.kernel_size[0]
                    w_start = j * self.stride[1]
                    w_end = w_start + self.kernel_size[1]
                    
                    receptive_field = padded[:, h_start:h_end, w_start:w_end, :]
                    
                    # 修改卷积计算以避免广播问题
                    for k in range(self.filters):
                        # 确保维度匹配
                        kernel = self.params['W'][:, :, :, k]  # (kh, kw, c)
                        # 扩展维度以匹配批次
                        kernel = kernel[np.newaxis, :, :, :]  # (1, kh, kw, c)
                        
                        # 执行卷积计算
                        output[:, i, j, k] = np.sum(
                            receptive_field * kernel,  # 现在广播是安全的
                            axis=(1,2,3)
                        ) + self.params['b'][k]
                        
        except Exception as e:
            logger.error(f"Error in Conv2D forward pass: {str(e)}")
            logger.error(f"Shapes:")
            logger.error(f"inputs: {inputs.shape}")
            logger.error(f"padded: {padded.shape}")
            logger.error(f"receptive_field: {receptive_field.shape}")
            logger.error(f"weights: {self.params['W'].shape}")
            logger.error(f"output: {output.shape}")
            raise
            
        return output
        
    def backward(self, grad_output):
        """反向传播"""
        try:
            # 初始化梯度
            self.grads['W'] = np.zeros_like(self.params['W'])
            self.grads['b'] = np.zeros_like(self.params['b'])
            grad_input = np.zeros_like(self.inputs)
            
            # 填充处理
            if self.padding[0] > 0 or self.padding[1] > 0:
                padded = self._pad(self.inputs)
                grad_padded = np.zeros_like(padded)
            else:
                padded = self.inputs
                grad_padded = grad_input
                
            # 计算梯度（向量化版本）
            for i in range(self.output_shape[1]):
                for j in range(self.output_shape[2]):
                    h_start = i * self.stride[0]
                    h_end = h_start + self.kernel_size[0]
                    w_start = j * self.stride[1]
                    w_end = w_start + self.kernel_size[1]
                    
                    receptive_field = padded[:, h_start:h_end, w_start:w_end, :]
                    
                    for k in range(self.filters):
                        self.grads['W'][:, :, :, k] += np.sum(
                            receptive_field * grad_output[:, i, j, k][:, None, None, None],
                            axis=0
                        )
                        self.grads['b'][k] += np.sum(grad_output[:, i, j, k])
                        
                        grad_padded[:, h_start:h_end, w_start:w_end, :] += \
                            self.params['W'][:, :, :, k] * grad_output[:, i, j, k][:, None, None, None]
                            
            # 如果有填充，去除填充部分
            if self.padding[0] > 0 or self.padding[1] > 0:
                grad_input = grad_padded[
                    :,
                    self.padding[0]:-self.padding[0],
                    self.padding[1]:-self.padding[1],
                    :
                ]
                
        except Exception as e:
            logger.error(f"Error in Conv2D backward pass: {str(e)}")
            logger.debug(f"Current shapes:")
            logger.shape_info("grad_output", grad_output)
            logger.shape_info("receptive_field", receptive_field)
            logger.shape_info("weights", self.params['W'])
            logger.shape_info("grad_input", grad_input)
            raise
            
        return grad_input
