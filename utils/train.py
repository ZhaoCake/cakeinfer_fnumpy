import numpy as np
from tqdm import tqdm
import time
from .dataloader import download_mnist, create_batches
from .weight_io import save_weights
from utils.logger import logger

class SGD:
    """简单的随机梯度下降优化器"""
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        
    def update(self, params_and_grads):
        """更新参数"""
        for param, grad in params_and_grads:
            param -= self.learning_rate * grad


def evaluate(model, images, labels, batch_size=32):
    """
    评估模型性能
    :return: 准确率
    """
    correct = 0
    total = 0
    
    for batch_images, batch_labels in create_batches(images, labels, batch_size, shuffle=False):
        predictions = model.predict(batch_images)
        correct += np.sum(predictions == np.argmax(batch_labels, axis=1))
        total += len(predictions)
        
    return correct / total


def train(model, epochs=10, batch_size=32, learning_rate=0.01, data_dir='dataset', weights_dir='weights'):
    """
    训练模型
    :param model: 模型实例
    :param epochs: 训练轮数
    :param batch_size: 批次大小
    :param learning_rate: 学习率
    :param data_dir: 数据集存储目录
    :param weights_dir: 权重保存目录
    """
    logger.info("Starting training...")
    logger.info(f"Parameters: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
    
    # 加载数据集
    print("正在加载数据集...")
    train_images, train_labels, test_images, test_labels = download_mnist(data_dir)
    
    # 创建优化器
    optimizer = SGD(learning_rate)
    
    # 计算总批次数（向上取整）
    n_batches = (len(train_images) + batch_size - 1) // batch_size
    
    # 训练循环
    best_accuracy = 0
    best_epoch = 0
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    print("\n开始训练...")
    try:
        for epoch in range(epochs):
            epoch_start_time = time.time()
            epoch_losses = []
            
            # 训练一个epoch
            progress_bar = tqdm(
                create_batches(train_images, train_labels, batch_size),
                desc=f"Epoch {epoch + 1}/{epochs}",
                total=n_batches
            )
            
            for batch_images, batch_labels in progress_bar:
                # 前向传播
                loss, grad = model.compute_loss(batch_images, batch_labels)
                epoch_losses.append(loss)
                
                # 反向传播
                model.backward(grad)
                
                # 更新参数
                optimizer.update(model.get_params_and_grads())
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f"{loss:.4f}"
                })
                
            # 计算epoch的平均损失
            epoch_loss = np.mean(epoch_losses)
            train_losses.append(epoch_loss)
            
            # 评估训练集和测试集性能
            train_accuracy = evaluate(model, train_images, train_labels, batch_size)
            test_accuracy = evaluate(model, test_images, test_labels, batch_size)
            
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
            
            # 保存最佳模型
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_epoch = epoch + 1
                # 保存权重
                save_weights(
                    model,
                    f"lenet_epoch_{best_epoch}_acc_{best_accuracy:.4f}",
                    weights_dir
                )
            
            # 打印训练信息
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.4f}")
            logger.info(f"Train accuracy: {train_accuracy:.4f}")
            logger.info(f"Test accuracy: {test_accuracy:.4f}")
            logger.info(f"Best test accuracy: {best_accuracy:.4f}")
            logger.info("-" * 50)
            
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'best_accuracy': best_accuracy
    }


if __name__ == '__main__':
    from models import LeNet
    
    # 创建模型
    model = LeNet()
    
    # 训练模型
    history = train(
        model,
        epochs=10,
        batch_size=32,
        learning_rate=0.01
    )
    
    print("\n训练完成！")
    print(f"最佳测试集准确率: {history['best_accuracy']:.4f}")
