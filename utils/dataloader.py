import numpy as np
import gzip
import os
import urllib.request
import struct

def check_mnist_files(data_dir):
    """
    检查MNIST数据文件是否存在
    :param data_dir: 数据目录
    :return: bool, 所有文件是否存在
    """
    required_files = [
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz',
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz'
    ]
    
    return all(
        os.path.exists(os.path.join(data_dir, f)) 
        for f in required_files
    )

def download_mnist_files(data_dir):
    """
    下载MNIST数据集文件
    :param data_dir: 数据存储目录
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    # 正确的文件名和URL对应关系
    files = {
        'train-images-idx3-ubyte.gz': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    }
    
    # 下载数据文件
    for filename, url in files.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"正在下载 {filename}...")
            try:
                urllib.request.urlretrieve(url, filepath)
            except Exception as e:
                print(f"下载 {filename} 失败: {e}")
                raise

def load_mnist_data(data_dir):
    """
    加载MNIST数据集
    :param data_dir: 数据目录
    :return: (训练图像, 训练标签, 测试图像, 测试标签)
    """
    try:
        # 读取训练图像
        with gzip.open(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'), 'rb') as f:
            magic, size, rows, cols = struct.unpack('>IIII', f.read(16))
            train_images = np.frombuffer(f.read(), dtype=np.uint8).reshape(size, rows, cols)
            
        # 读取训练标签
        with gzip.open(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'), 'rb') as f:
            magic, size = struct.unpack('>II', f.read(8))
            train_labels = np.frombuffer(f.read(), dtype=np.uint8)
            
        # 读取测试图像
        with gzip.open(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'), 'rb') as f:
            magic, size, rows, cols = struct.unpack('>IIII', f.read(16))
            test_images = np.frombuffer(f.read(), dtype=np.uint8).reshape(size, rows, cols)
            
        # 读取测试标签
        with gzip.open(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'), 'rb') as f:
            magic, size = struct.unpack('>II', f.read(8))
            test_labels = np.frombuffer(f.read(), dtype=np.uint8)
            
        return train_images, train_labels, test_images, test_labels
        
    except Exception as e:
        print(f"加载数据集失败: {e}")
        raise

def preprocess_mnist_data(images, labels):
    """
    预处理MNIST数据
    :param images: 图像数据
    :param labels: 标签数据
    :return: 预处理后的图像和标签
    """
    # 将图像转换为float32类型，并归一化到[0,1]范围
    images = images.astype(np.float32) / 255.0
    
    # 将28x28图像填充到32x32（LeNet的标准输入大小）
    padded_images = np.zeros((images.shape[0], 32, 32), dtype=np.float32)
    padded_images[:, 2:30, 2:30] = images
    
    # 添加通道维度，转换为NHWC格式
    padded_images = padded_images.reshape(-1, 32, 32, 1)
    
    # 将标签转换为one-hot编码
    one_hot_labels = np.zeros((labels.shape[0], 10))
    one_hot_labels[np.arange(labels.shape[0]), labels] = 1
    
    return padded_images, one_hot_labels

def download_mnist(data_dir='dataset'):
    """
    下载并处理MNIST数据集
    :param data_dir: 数据存储目录，默认为'dataset'
    :return: (训练图像, 训练标签, 测试图像, 测试标签)
    """
    # 检查数据文件是否存在
    if not check_mnist_files(data_dir):
        print("数据文件不完整，开始下载MNIST数据集...")
        download_mnist_files(data_dir)
    else:
        print("检测到现有数据集，直接加载...")
    
    # 加载原始数据
    print("正在加载数据...")
    train_images, train_labels, test_images, test_labels = load_mnist_data(data_dir)
    
    # 预处理数据
    print("正在处理数据...")
    train_images, train_labels = preprocess_mnist_data(train_images, train_labels)
    test_images, test_labels = preprocess_mnist_data(test_images, test_labels)
    
    print("数据集准备完成！")
    print(f"训练集形状: {train_images.shape}")
    print(f"训练集标签形状: {train_labels.shape}")
    print(f"测试集形状: {test_images.shape}")
    print(f"测试集标签形状: {test_labels.shape}")
    
    return train_images, train_labels, test_images, test_labels


def create_batches(images, labels, batch_size, shuffle=True):
    """
    创建数据批次，填充最后一个不完整的批次
    :param images: 图像数据
    :param labels: 标签数据
    :param batch_size: 批次大小
    :param shuffle: 是否打乱数据
    :return: 生成器，每次返回一个批次的数据
    """
    assert len(images) == len(labels)
    n_samples = len(images)
    
    if shuffle:
        indices = np.random.permutation(n_samples)
    else:
        indices = np.arange(n_samples)
        
    for start_idx in range(0, n_samples, batch_size):
        batch_indices = indices[start_idx:min(start_idx + batch_size, n_samples)]
        
        # 如果是最后一个不完整的批次，进行填充
        if len(batch_indices) < batch_size:
            # 从已有数据中随机选择样本进行填充
            pad_indices = np.random.choice(indices[:start_idx], batch_size - len(batch_indices))
            batch_indices = np.concatenate([batch_indices, pad_indices])
            
        yield images[batch_indices], labels[batch_indices]


if __name__ == '__main__':
    # 测试数据集下载和加载
    train_images, train_labels, test_images, test_labels = download_mnist()
    
    # 测试批次生成
    batch_size = 32
    for batch_images, batch_labels in create_batches(train_images, train_labels, batch_size):
        print(f"批次形状: {batch_images.shape}, {batch_labels.shape}")
        break  # 只测试一个批次
