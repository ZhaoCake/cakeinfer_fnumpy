import numpy as np
from tqdm import tqdm
from utils.logger import logger

def evaluate_model(model, test_images, test_labels, batch_size=32):
    """评估模型性能"""
    n_samples = len(test_images)
    n_correct = 0
    predictions = []
    confidences = []
    
    logger.info("开始评估模型...")
    logger.info(f"测试样本数量: {n_samples}")
    
    # 计算完整批次的数量
    n_complete_batches = n_samples // batch_size
    
    # 只处理完整的批次
    for i in tqdm(range(n_complete_batches)):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_images = test_images[start_idx:end_idx]
        batch_labels = test_labels[start_idx:end_idx]
        
        # 预测和计算置信度
        batch_predictions = model.predict(batch_images)
        probs = model.predict_proba(batch_images)
        batch_confidences = np.max(probs, axis=1)
        
        # 获取真实标签
        true_labels = np.argmax(batch_labels, axis=1)
        
        # 统计正确预测
        n_correct += np.sum(batch_predictions == true_labels)
        predictions.extend(batch_predictions)
        confidences.extend(batch_confidences)
    
    # 处理最后一个不完整批次（如果有的话）
    remaining = n_samples - n_complete_batches * batch_size
    if remaining > 0:
        start_idx = n_complete_batches * batch_size
        end_idx = n_samples
        batch_images = test_images[start_idx:end_idx]
        batch_labels = test_labels[start_idx:end_idx]
        
        # 预测和计算置信度
        batch_predictions = model.predict(batch_images)
        probs = model.predict_proba(batch_images)
        batch_confidences = np.max(probs, axis=1)
        
        # 获取真实标签
        true_labels = np.argmax(batch_labels, axis=1)
        
        # 统计正确预测
        n_correct += np.sum(batch_predictions == true_labels)
        predictions.extend(batch_predictions)
        confidences.extend(batch_confidences)
        
        logger.info(f"处理最后 {remaining} 个样本（不完整批次）")
    
    # 更新实际处理的样本数
    n_processed = n_samples
    accuracy = n_correct / n_processed
    
    # 生成详细报告
    logger.info("\n评估结果:")
    logger.info(f"处理样本数: {n_processed}")
    logger.info(f"正确预测数: {n_correct}")
    logger.info(f"准确率: {accuracy:.4f}")
    
    # 计算每个类别的准确率
    class_correct = np.zeros(10)
    class_total = np.zeros(10)
    true_labels = np.argmax(test_labels[:n_processed], axis=1)
    
    for i in range(len(predictions)):
        label = true_labels[i]
        class_total[label] += 1
        if predictions[i] == label:
            class_correct[label] += 1
    
    logger.info("\n各类别准确率:")
    for i in range(10):
        if class_total[i] > 0:
            class_accuracy = class_correct[i] / class_total[i]
            logger.info(f"类别 {i}: {class_accuracy:.4f} ({int(class_correct[i])}/{int(class_total[i])})")
    
    # 计算置信度统计
    confidences = np.array(confidences)
    logger.info("\n置信度统计:")
    logger.info(f"平均置信度: {np.mean(confidences):.4f}")
    logger.info(f"最小置信度: {np.min(confidences):.4f}")
    logger.info(f"最大置信度: {np.max(confidences):.4f}")
    
    return accuracy, predictions, confidences