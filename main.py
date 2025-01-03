import os
import argparse
from models import LeNet
from utils import train

def parse_args():
    parser = argparse.ArgumentParser(description='训练LeNet模型识别MNIST数字')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--data-dir', type=str, default='data', help='数据集存储目录')
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 确保数据目录存在
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    
    # 创建并训练模型
    print("初始化LeNet模型...")
    model = LeNet()
    
    # 开始训练
    history = train(
        model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        data_dir=args.data_dir
    )
    
    print("\n训练完成！")
    print(f"最佳测试集准确率: {history['best_accuracy']:.4f}")

if __name__ == '__main__':
    main() 