import numpy as np
from .common_model import Model
from layers import Conv2D, MaxPool2D, FullyConnected, ReLU
from utils.logger import logger

class LeNet(Model):
    def __init__(self, num_classes=10):
        """
        LeNet模型实现
        :param num_classes: 分类数量，默认为10（MNIST数据集）
        """
        super().__init__()
        
        # 第一个卷积块：Conv(6) -> ReLU -> MaxPool
        self.conv1 = Conv2D(filters=6, kernel_size=5, stride=1, padding=2)  # 输入: 32x32 -> 32x32
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(pool_size=2, stride=2)  # 32x32 -> 16x16
        
        # 第二个卷积块：Conv(16) -> ReLU -> MaxPool
        self.conv2 = Conv2D(filters=16, kernel_size=5, stride=1, padding=0)  # 16x16 -> 12x12
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(pool_size=2, stride=2)  # 12x12 -> 6x6
        
        # 全连接层
        self.fc1 = FullyConnected(output_dim=120)  # 6x6x16 = 576 -> 120
        self.relu3 = ReLU()
        
        self.fc2 = FullyConnected(output_dim=84)  # 120 -> 84
        self.relu4 = ReLU()
        
        self.fc3 = FullyConnected(output_dim=num_classes)  # 84 -> num_classes
        
        # 将所有层添加到列表中
        self.layers = [
            self.conv1, self.relu1, self.pool1,
            self.conv2, self.relu2, self.pool2,
            self.fc1, self.relu3,
            self.fc2, self.relu4,
            self.fc3
        ]
        
        logger.debug("LeNet model initialized with layers:")
        for i, layer in enumerate(self.layers):
            logger.debug(f"layer_{i}_{layer.__class__.__name__}")
        
    def forward(self, x):
        """前向传播"""
        # 保存原始形状
        original_batch_size = x.shape[0]
        
        # 确保输入是批次形式
        if len(x.shape) == 3:
            x = x[np.newaxis, ...]
            
        # 前向传播
        for layer in self.layers:
            x = layer.forward(x)
            
        return x
    
    def predict(self, x):
        """预测函数"""
        # 前向传播获取logits
        logits = self.forward(x)
        # 直接返回预测类别
        return np.argmax(logits, axis=1)
    
    def predict_proba(self, x):
        """获取概率分布"""
        logits = self.forward(x)
        # 计算softmax
        logits_max = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
    def compute_loss(self, x, y):
        """
        计算交叉熵损失
        :param x: 输入数据
        :param y: 真实标签（one-hot编码）
        :return: 损失值和梯度
        """
        logits = self.forward(x)
        
        # 计算softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # 计算交叉熵损失
        batch_size = x.shape[0]
        loss = -np.sum(y * np.log(probs + 1e-7)) / batch_size
        
        # 计算梯度
        grad = (probs - y) / batch_size
        
        return loss, grad