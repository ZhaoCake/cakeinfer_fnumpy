import numpy as np
from .common_model import Model
from layers import Conv2D, MaxPool2D, FullyConnected, ReLU

class LeNet(Model):
    def __init__(self, num_classes=10):
        """
        LeNet模型实现
        :param num_classes: 分类数量，默认为10（MNIST数据集）
        """
        super().__init__()
        
        # 第一个卷积块：Conv(6) -> ReLU -> MaxPool
        self.add(Conv2D(filters=6, kernel_size=5, stride=1, padding=2))  # 输入: 32x32 -> 32x32
        self.add(ReLU())
        self.add(MaxPool2D(pool_size=2, stride=2))  # 32x32 -> 16x16
        
        # 第二个卷积块：Conv(16) -> ReLU -> MaxPool
        self.add(Conv2D(filters=16, kernel_size=5, stride=1, padding=0))  # 16x16 -> 12x12
        self.add(ReLU())
        self.add(MaxPool2D(pool_size=2, stride=2))  # 12x12 -> 6x6
        
        # 全连接层
        self.add(FullyConnected(output_dim=120))  # 6x6x16 = 576 -> 120
        self.add(ReLU())
        
        self.add(FullyConnected(output_dim=84))  # 120 -> 84
        self.add(ReLU())
        
        self.add(FullyConnected(output_dim=num_classes))  # 84 -> num_classes
        
    def predict(self, x):
        """
        预测函数
        :param x: 输入数据，形状为(batch_size, 32, 32, 1)
        :return: 预测结果，形状为(batch_size, num_classes)
        """
        logits = self.forward(x)
        return np.argmax(logits, axis=1)
        
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