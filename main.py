import os
import argparse
from models import LeNet
from utils import train
from utils.weight_io import load_weights
from utils.logger import logger
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='LeNet-MNIST 训练与推理')
    
    # 通用参数
    parser.add_argument('--mode', type=str, choices=['train', 'infer', 'test'], required=True,
                       help='运行模式：train（训练）、infer（推理）或 test（测试）')
    parser.add_argument('--data-dir', type=str, default='dataset',
                       help='数据集存储目录')
    parser.add_argument('--weights-dir', type=str, default='weights',
                       help='权重存储目录')
    
    # 训练相关参数
    parser.add_argument('--epochs', type=int, default=10,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='学习率')
    
    # 推理相关参数
    parser.add_argument('--model-name', type=str,
                       help='用于推理的模型名称（例如：lenet_epoch_1_acc_0.9312）')
    parser.add_argument('--save-intermediate', action='store_true',
                       help='是否保存中间层输出')
    parser.add_argument('--test-sample', type=int, default=0,
                       help='测试样本的索引（默认使用第一个测试样本）')
    
    # 测试相关参数
    parser.add_argument('--save-predictions', action='store_true',
                       help='是否保存测试预测结果')
    parser.add_argument('--batch-size-test', type=int, default=32,
                       help='测试时的批次大小')
    
    return parser.parse_args()

def infer(model, test_images, test_labels, sample_idx=0, save_intermediate=False):
    """
    模型推理
    :param model: 模型实例
    :param test_images: 测试图像
    :param test_labels: 测试标签
    :param sample_idx: 测试样本索引
    :param save_intermediate: 是否保存中间层输出
    """
    logger.info("开始推理...")
    
    # 选择测试样本
    test_image = test_images[sample_idx:sample_idx+1]
    true_label = np.argmax(test_labels[sample_idx])
    
    logger.info(f"测试样本索引: {sample_idx}")
    logger.info(f"输入形状: {test_image.shape}")
    logger.info(f"真实标签: {true_label}")
    
    # 保存输入数据
    if save_intermediate:
        os.makedirs('intermediate', exist_ok=True)
        # 保存原始输入数据
        test_image.astype(np.float32).tofile('intermediate/test_input.bin')
        # 保存可视化图像
        visual_image = (test_image[0, :, :, 0] * 255).astype(np.uint8)
        try:
            from PIL import Image
            Image.fromarray(visual_image).save('intermediate/test_input.png')
            logger.info("已保存输入图像到 intermediate/test_input.png")
        except ImportError:
            logger.warning("未安装PIL库，跳过图像保存")
        logger.info("已保存输入数据到 intermediate/test_input.bin")
    
    # 前向传播并记录中间结果
    intermediate = {}
    x = test_image
    
    for i, layer in enumerate(model.layers):
        x = layer.forward(x)
        layer_name = f"layer_{i}_{layer.__class__.__name__}"
        intermediate[layer_name] = x.copy()
        
        # 记录层的输出信息
        logger.info(f"\n{layer_name}:")
        logger.info(f"输出形状: {x.shape}")
        logger.info(f"数值范围: [{x.min():.6f}, {x.max():.6f}]")
        logger.info(f"均值: {x.mean():.6f}")
        logger.info(f"标准差: {x.std():.6f}")
        
        # 保存中间输出
        if save_intermediate:
            output_file = f'intermediate/{layer_name}.bin'
            x.astype(np.float32).tofile(output_file)
            logger.info(f"已保存层输出到 {output_file}")
    
    # 使用模型的predict函数进行预测
    predicted_class = model.predict(test_image)[0]  # [0]因为predict返回batch的结果
    
    # 计算置信度
    final_output = x[0]  # 移除batch维度
    logits_max = np.max(final_output, keepdims=True)
    exp_logits = np.exp(final_output - logits_max)
    probabilities = exp_logits / np.sum(exp_logits)
    confidence = probabilities[predicted_class]
    
    # 输出详细的预测信息
    logger.info(f"\n预测结果:")
    logger.info(f"预测类别: {predicted_class}")
    logger.info(f"预测置信度: {confidence:.4f}")
    logger.info(f"真实类别: {true_label}")
    logger.info(f"预测正确: {predicted_class == true_label}")
    
    # 输出所有类别的概率分布
    logger.info("\n所有类别的概率分布:")
    for i, prob in enumerate(probabilities):
        logger.info(f"类别 {i}: {prob:.4f}")
    
    if save_intermediate:
        # 保存概率分布
        np.savetxt('intermediate/probabilities.txt', probabilities, fmt='%.6f')
        logger.info("已保存概率分布到 intermediate/probabilities.txt")
    
    return predicted_class, true_label, intermediate, probabilities

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 确保数据目录存在
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    
    # 创建模型
    logger.info("初始化LeNet模型...")
    model = LeNet()
    
    if args.mode == 'train':
        # 训练模式
        history = train(
            model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            data_dir=args.data_dir,
            weights_dir=args.weights_dir
        )
        logger.info("\n训练完成！")
        logger.info(f"最佳测试集准确率: {history['best_accuracy']:.4f}")
        
    elif args.mode == 'infer':
        # 推理模式
        if not args.model_name:
            raise ValueError("推理模式需要指定 --model-name 参数")
            
        # 加载权重
        logger.info(f"加载模型权重: {args.model_name}")
        load_weights(model, args.model_name, args.weights_dir)
        
        # 加载测试数据
        from utils.dataloader import download_mnist
        _, _, test_images, test_labels = download_mnist(args.data_dir)
        
        # 执行推理
        predicted_class, true_label, _, probabilities = infer(
            model,
            test_images,
            test_labels,
            sample_idx=args.test_sample,
            save_intermediate=args.save_intermediate
        )

    elif args.mode == 'test':
        # 测试模式
        if not args.model_name:
            raise ValueError("测试模式需要指定 --model-name 参数")
            
        # 加载权重
        logger.info(f"加载模型权重: {args.model_name}")
        load_weights(model, args.model_name, args.weights_dir)
        
        # 加载测试数据
        from utils.dataloader import download_mnist
        _, _, test_images, test_labels = download_mnist(args.data_dir)
        
        # 执行测试
        from utils.test import evaluate_model
        accuracy, predictions, confidences = evaluate_model(
            model,
            test_images,
            test_labels,
            batch_size=args.batch_size_test
        )
        
        # 保存预测结果
        if args.save_predictions:
            os.makedirs('results', exist_ok=True)
            true_labels = np.argmax(test_labels, axis=1)
            results = np.column_stack((
                true_labels,
                predictions,
                confidences
            ))
            np.savetxt(
                'results/test_predictions.txt',
                results,
                fmt=['%d', '%d', '%.6f'],
                header='true_label predicted_label confidence',
                comments=''
            )
            logger.info("已保存预测结果到 results/test_predictions.txt")

if __name__ == '__main__':
    main() 