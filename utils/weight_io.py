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
        
    # 准备权重结构信息
    weight_struct = {
        'model_name': model_name,
        'layer_weights': {}
    }
    
    # 二进制文件的写入位置指针
    current_pos = 0
    
    # 打开二进制文件
    bin_path = os.path.join(weights_dir, f'{model_name}.bin')
    with open(bin_path, 'wb') as f:
        # 遍历所有层
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'params') and layer.params:
                layer_name = f"layer_{i}_{layer.__class__.__name__}"
                weight_struct['layer_weights'][layer_name] = {}
                
                # 保存每个参数的信息
                for param_name, param in layer.params.items():
                    # 记录参数信息
                    weight_struct['layer_weights'][layer_name][param_name] = {
                        'shape': list(param.shape),  # 转换为list以便JSON序列化
                        'dtype': str(param.dtype),
                        'offset': current_pos,
                        'size': param.size
                    }
                    
                    # 写入参数数据
                    param.tofile(f)
                    current_pos += param.nbytes
                    
                    logger.debug(f"Saved {layer_name}.{param_name}: shape={param.shape}, "
                               f"dtype={param.dtype}, offset={current_pos-param.nbytes}")
    
    # 保存结构文件
    struct_path = os.path.join(weights_dir, f'{model_name}.modelstruct')
    with open(struct_path, 'w') as f:
        json.dump(weight_struct, f, indent=4)
        
    logger.info(f"模型权重已保存到: {weights_dir}")
    logger.info(f"- 权重文件: {model_name}.bin")
    logger.info(f"- 结构文件: {model_name}.modelstruct")

def load_weights(model, model_name: str, weights_dir: str = 'weights'):
    """加载模型权重"""
    # 读取结构文件
    struct_path = os.path.join(weights_dir, f'{model_name}.modelstruct')
    logger.info(f"Loading weights from {struct_path}")
    
    try:
        with open(struct_path, 'r') as f:
            weight_struct = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load modelstruct file: {e}")
        raise
    
    # 打开二进制文件
    bin_path = os.path.join(weights_dir, f'{model_name}.bin')
    logger.info(f"Loading weights from {bin_path}")
    
    try:
        with open(bin_path, 'rb') as f:
            # 遍历所有层
            for i, layer in enumerate(model.layers):
                layer_name = f"layer_{i}_{layer.__class__.__name__}"
                logger.debug(f"Processing layer: {layer_name}")
                
                if hasattr(layer, 'params'):
                    if layer_name in weight_struct['layer_weights']:
                        # 加载每个参数
                        for param_name, param_info in weight_struct['layer_weights'][layer_name].items():
                            # 定位到参数位置
                            f.seek(param_info['offset'])
                            
                            # 读取参数数据
                            shape = tuple(param_info['shape'])
                            dtype = np.dtype(param_info['dtype'])
                            size = param_info['size']
                            
                            logger.debug(f"Loading {param_name}: shape={shape}, dtype={dtype}, size={size}")
                            
                            try:
                                param_data = np.fromfile(f, dtype=dtype, count=size)
                                param_data = param_data.reshape(shape)
                                layer.params[param_name] = param_data.astype(dtype)
                            except Exception as e:
                                logger.error(f"Failed to load parameter {param_name}: {e}")
                                raise
                    else:
                        logger.warning(f"Layer {layer_name} not found in weight file")
    except Exception as e:
        logger.error(f"Failed to load weights: {e}")
        raise
    
    # 验证权重加载
    total_params = 0
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'params'):
            layer_name = f"layer_{i}_{layer.__class__.__name__}"
            for param_name, param in layer.params.items():
                if np.isnan(param).any():
                    logger.error(f"Found NaN in {layer_name}.{param_name}")
                if np.isinf(param).any():
                    logger.error(f"Found Inf in {layer_name}.{param_name}")
                total_params += param.size
                logger.debug(f"Verified {layer_name}.{param_name}: shape={param.shape}")
    
    logger.info(f"Successfully loaded {total_params} parameters")
    logger.info("Weight validation completed") 