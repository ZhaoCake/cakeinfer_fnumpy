import os
import numpy as np
import json
from typing import Dict, Any
from utils.logger import logger  # 添加logger

def save_weights(model, model_name: str, weights_dir: str = 'weights'):
    """
    保存模型权重
    :param model: 模型实例
    :param model_name: 模型名称
    :param weights_dir: 权重保存目录
    """
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    weights_dict = {}
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'params') and layer.params:
            layer_name = f"layer_{i}_{layer.__class__.__name__}"
            for param_name, param in layer.params.items():
                key = f"{layer_name}_{param_name}"
                weights_dict[key] = param
                logger.debug(f"Prepared {key}: shape={param.shape}, dtype={param.dtype}")

    npz_path = os.path.join(weights_dir, f'{model_name}.npz')
    np.savez(npz_path, **weights_dict)
    logger.info(f"模型权重已保存到: {npz_path}")

def load_weights(model, model_name: str, weights_dir: str = 'weights'):
    """加载模型权重"""
    npz_path = os.path.join(weights_dir, f'{model_name}.npz')
    logger.info(f"Loading weights from {npz_path}")
    try:
        weights = np.load(npz_path)
    except Exception as e:
        logger.error(f"Failed to load weights file: {e}")
        raise

    total_params = 0
    # 先清空所有layer的params
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'params'):
            layer.params.clear()

    # 遍历npz文件的所有key，自动分配到对应layer
    for key in weights:
        # key格式: layer_{i}_{LayerName}_{param_name}
        parts = key.split('_')
        if len(parts) < 4:
            logger.warning(f"Unrecognized weight key: {key}")
            continue
        layer_idx = int(parts[1])
        layer_name = parts[2]
        param_name = '_'.join(parts[3:])
        if 0 <= layer_idx < len(model.layers):
            layer = model.layers[layer_idx]
            if not hasattr(layer, 'params'):
                layer.params = {}
            layer.params[param_name] = weights[key].copy()
            logger.debug(f"Loaded {key}: shape={weights[key].shape}, dtype={weights[key].dtype}")
            if np.isnan(weights[key]).any():
                logger.error(f"Found NaN in {key}")
            if np.isinf(weights[key]).any():
                logger.error(f"Found Inf in {key}")
            total_params += weights[key].size
        else:
            logger.warning(f"Layer index {layer_idx} out of range for key {key}")
    logger.info(f"Successfully loaded {total_params} parameters")
    logger.info("Weight validation completed")