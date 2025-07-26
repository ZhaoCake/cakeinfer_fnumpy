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
        
    def _im2col(self, inputs):
        """
        将输入图像转换为列矩阵形式 (im2col)
        :param inputs: 输入数据，形状为(batch_size, height, width, channels)
        :return: 列矩阵，形状为(kernel_height * kernel_width * input_channels, output_height * output_width * batch_size)
        """
        batch_size, input_height, input_width, input_channels = inputs.shape
        kernel_height, kernel_width = self.kernel_size
        stride_h, stride_w = self.stride
        output_height, output_width = self.output_shape[1], self.output_shape[2]
        
        # 创建输出矩阵
        col_matrix = np.zeros((
            batch_size,
            kernel_height * kernel_width * input_channels,
            output_height * output_width
        ))
        
        # 填充输入
        if self.padding[0] > 0 or self.padding[1] > 0:
            padded_inputs = self._pad(inputs)
        else:
            padded_inputs = inputs
            
        # im2col操作
        col_idx = 0
        for i in range(output_height):
            for j in range(output_width):
                h_start = i * stride_h
                h_end = h_start + kernel_height
                w_start = j * stride_w
                w_end = w_start + kernel_width
                
                # 提取当前感受野并重塑
                receptive_field = padded_inputs[:, h_start:h_end, w_start:w_end, :]  # (batch, kh, kw, c)
                col_matrix[:, :, col_idx] = receptive_field.reshape(batch_size, -1)  # (batch, kh*kw*c)
                col_idx += 1
                
        return col_matrix
    
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
        
        # 使用im2col + GEMM实现卷积
        batch_size = inputs.shape[0]
        output_height, output_width = self.output_shape[1], self.output_shape[2]
        
        # 将输入转换为列矩阵形式
        col_matrix = self._im2col(inputs)  # (batch, kh*kw*c, oh*ow)
        
        # 重塑权重矩阵 (kh, kw, c, f) -> (kh*kw*c, f)
        weights_reshaped = self.params['W'].reshape(-1, self.filters)
        
        # 执行批量矩阵乘法: (batch, kh*kw*c, oh*ow) x (kh*kw*c, f) -> (batch, oh*ow, f)
        output = np.matmul(col_matrix.transpose(0, 2, 1), weights_reshaped)  # (batch, oh*ow, f)
        
        # 添加偏置并重塑输出形状
        output = output + self.params['b']  # 广播加法
        output = output.reshape(batch_size, output_height, output_width, self.filters)
        
        return output
        
        
    def backward(self, grad_output):
        """反向传播"""
        batch_size, output_height, output_width, _ = grad_output.shape
        
        # 重塑梯度输出 (batch, oh, ow, f) -> (batch, oh*ow, f)
        grad_output_reshaped = grad_output.reshape(batch_size, -1, self.filters)
        
        # 计算权重梯度
        # 获取输入的列矩阵形式
        col_matrix = self._im2col(self.inputs)  # (batch, kh*kw*c, oh*ow)
        
        # 计算权重梯度: (kh*kw*c, batch, oh*ow) x (batch, oh*ow, f) -> (kh*kw*c, f)
        weights_grad = np.matmul(col_matrix.transpose(1, 0, 2).reshape(-1, batch_size * output_height * output_width),
                                grad_output_reshaped.reshape(batch_size * output_height * output_width, -1))
        
        # 重塑权重梯度到原始形状
        self.grads['W'] = weights_grad.reshape(self.kernel_size[0], self.kernel_size[1], 
                                              self.input_shape[3], self.filters)
        
        # 计算偏置梯度
        self.grads['b'] = np.sum(grad_output, axis=(0, 1, 2))
        
        # 计算输入梯度
        # 重塑权重 (kh, kw, c, f) -> (f, kh*kw*c)
        weights_reshaped = self.params['W'].reshape(-1, self.filters).T
        
        # 执行矩阵乘法计算输入梯度: (batch, oh*ow, f) x (f, kh*kw*c) -> (batch, oh*ow, kh*kw*c)
        grad_input_col = np.matmul(grad_output_reshaped, weights_reshaped)  # (batch, oh*ow, kh*kw*c)
        
        # 将列矩阵形式转换回图像形式 (col2im)
        grad_input = self._col2im(grad_input_col.transpose(0, 2, 1), batch_size, 
                                 self.input_shape[1], self.input_shape[2], self.input_shape[3])
        
        return grad_input
    
    def _col2im(self, col_matrix, batch_size, input_height, input_width, input_channels):
        """
        将列矩阵转换回图像形式 (col2im)
        :param col_matrix: 列矩阵，形状为(batch_size, kh*kw*c, oh*ow)
        :param batch_size: 批次大小
        :param input_height: 输入高度
        :param input_width: 输入宽度
        :param input_channels: 输入通道数
        :return: 图像张量，形状为(batch_size, height, width, channels)
        """
        kernel_height, kernel_width = self.kernel_size
        stride_h, stride_w = self.stride
        output_height, output_width = self.output_shape[1], self.output_shape[2]
        
        # 初始化输出梯度
        grad_input = np.zeros((batch_size, input_height, input_width, input_channels))
        
        if self.padding[0] > 0 or self.padding[1] > 0:
            padded_grad_input = np.zeros((
                batch_size,
                input_height + 2*self.padding[0],
                input_width + 2*self.padding[1],
                input_channels
            ))
        else:
            padded_grad_input = grad_input
            
        # col2im操作
        col_idx = 0
        for i in range(output_height):
            for j in range(output_width):
                h_start = i * stride_h
                h_end = h_start + kernel_height
                w_start = j * stride_w
                w_end = w_start + kernel_width
                
                # 从列矩阵中提取数据并加到对应位置
                col_data = col_matrix[:, :, col_idx].reshape(batch_size, kernel_height, kernel_width, input_channels)
                padded_grad_input[:, h_start:h_end, w_start:w_end, :] += col_data
                col_idx += 1
                
        # 如果有填充，去除填充部分
        if self.padding[0] > 0 or self.padding[1] > 0:
            grad_input = padded_grad_input[
                :,
                self.padding[0]:input_height+self.padding[0],
                self.padding[1]:input_width+self.padding[1],
                :
            ]
            
        return grad_input
