# LeNet-MNIST Pure NumPy Implementation

> After writing it, I really feel like I finally understand the network and parameters. The next step is to write the inference framework for Cpp, optimize it with HLS, and deploy it on FPGA

> 2025-07-26
> 发现数据集网站找不到该数据了，请自行查找下载该数据集到dataset文件夹或者你指定的数据集文件夹下。

```
  ❯❯ /home/zhaocake/WorkSpace/Vision/cakeinfer_fnumpy : python main.py --epochs 10 --batch-size 64 --lr 0.005 --data-dir dataset
初始化LeNet模型...
2025-01-03 18:11:17,952 - fnumpy - INFO - Starting training...
2025-01-03 18:11:17,953 - fnumpy - INFO - Parameters: epochs=10, batch_size=64, lr=0.005
正在加载数据集...
检测到现有数据集，直接加载...
正在加载数据...
正在处理数据...
数据集准备完成！
训练集形状: (60000, 32, 32, 1)
训练集标签形状: (60000, 10)
测试集形状: (10000, 32, 32, 1)
测试集标签形状: (10000, 10)

开始训练...
Epoch 1/10: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [06:34<00:00,  2.38it/s, loss=0.1655]
模型权重已保存到: weights
- 权重文件: lenet_epoch_1_acc_0.9312.bin
- 结构文件: lenet_epoch_1_acc_0.9312.modelstruct
2025-01-03 18:19:51,873 - fnumpy - INFO - Epoch 1/10 - loss: 0.6265
2025-01-03 18:19:51,873 - fnumpy - INFO - Train accuracy: 0.9280
2025-01-03 18:19:51,873 - fnumpy - INFO - Test accuracy: 0.9312
2025-01-03 18:19:51,873 - fnumpy - INFO - Best test accuracy: 0.9312
2025-01-03 18:19:51,874 - fnumpy - INFO - --------------------------------------------------
Epoch 2/10:  99%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎| 933/938 [06:41<00:02,  2.39it/s, loss=0.1544]
```

---


一个使用纯NumPy实现的LeNet-5卷积神经网络训练推理框架，专注于单线程串行计算。

## 特点

- 纯NumPy实现，无深度学习框架依赖
- 单线程串行计算，适合学习CNN底层原理
- 完整的训练和推理功能
- 支持模型权重的保存和加载
- 详细的日志记录系统

## 项目结构

```
.
├── layers/                 # 网络层实现
│   ├── __init__.py
│   ├── common_layer.py    # 基础层类
│   ├── conv2d.py         # 卷积层
│   ├── fc.py             # 全连接层
│   ├── activation.py     # 激活函数
│   └── maxpooling.py     # 最大池化层
├── models/                # 模型实现
│   ├── __init__.py
│   ├── common_model.py   # 基础模型类
│   └── lenet.py          # LeNet模型
├── utils/                 # 工具函数
│   ├── __init__.py
│   ├── dataloader.py     # 数据加载
│   ├── logger.py         # 日志系统
│   ├── train.py          # 训练函数
│   └── weight_io.py      # 权重读写
├── main.py               # 主程序
└── requirements.txt      # 依赖包列表
```

## 环境要求

- NumPy
- tqdm

## 安装

安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 训练模型
```bash
python main.py --mode train \
    --epochs 10 \
    --batch-size 64 \
    --lr 0.005 \
    --data-dir dataset \
    --weights-dir weights
```

参数说明：
- `--mode`: 运行模式，'train'用于训练（必需）
- `--epochs`: 训练轮数（默认：10）
- `--batch-size`: 批次大小（默认：32）
- `--lr`: 学习率（默认：0.01）
- `--data-dir`: 数据集存储目录（默认：'dataset'）
- `--weights-dir`: 权重保存目录（默认：'weights'）

### 2. 模型推理
```bash
python main.py --mode infer \
    --model-name lenet_epoch_10_acc_0.9806 \
    --test-sample 0 \
    --save-intermediate \
    --data-dir dataset \
    --weights-dir weights
```

推理参数说明：
- `--mode`: 运行模式，'infer'用于推理（必需）
- `--model-name`: 要加载的模型名称（必需）
- `--test-sample`: 测试样本索引（默认：0）
- `--save-intermediate`: 是否保存中间层输出（可选）
- `--data-dir`: 数据集目录（默认：'dataset'）
- `--weights-dir`: 权重目录（默认：'weights'）

### 3. 模型测试
```bash
python main.py --mode test \
    --model-name lenet_epoch_10_acc_0.9806 \
    --data-dir dataset \
    --weights-dir weights \
    --save-predictions \
    --batch-size-test 64
```

测试参数说明：
- `--mode`: 运行模式，'test'用于测试（必需）
- `--model-name`: 要加载的模型名称（必需）
- `--save-predictions`: 是否保存预测结果（可选）
- `--batch-size-test`: 测试时的批次大小（默认：32）

测试输出示例：
```
评估结果:
总样本数: 10000
正确预测数: 9806
准确率: 0.9806

各类别准确率:
类别 0: 0.9921 (992/1000)
类别 1: 0.9897 (990/1000)
类别 2: 0.9845 (985/1000)
...

置信度统计:
平均置信度: 0.9823
最小置信度: 0.5123
最大置信度: 0.9999
```

如果使用 `--save-predictions`，将在 `results/` 目录下生成：
- `test_predictions.txt`: 包含每个样本的真实标签、预测标签和置信度

### 4. 中间层输出
推理时使用 `--save-intermediate` 参数会在 `intermediate/` 目录下生成：
- `test_input.bin`: 输入数据
- `layer_X_YYY.bin`: 每层的输出数据
- 所有数据以32位浮点数（float32）格式存储
- 数据按行优先（C-style）顺序存储

示例日志输出：
```
2025-01-03 18:11:17,952 - fnumpy - INFO - Starting inference...
2025-01-03 18:11:17,953 - fnumpy - INFO - Test sample index: 0
2025-01-03 18:11:17,953 - fnumpy - INFO - Input shape: (1, 32, 32, 1)

layer_0_Conv2D:
输出形状: (1, 32, 32, 6)
数值范围: [-0.123456, 0.123456]
均值: 0.000123
标准差: 0.012345

...（其他层的输出信息）

预测结果:
预测类别: 7
真实类别: 7
预测正确: True
```

### 5. 文件说明
- 权重文件保存在 `weights/` 目录下
- 包含二进制权重文件（.bin）和结构描述文件（.modelstruct）
- 中间层输出保存在 `intermediate/` 目录下
- 测试结果保存在 `results/` 目录下

### 6. 日志记录
- 日志文件保存在 `logs/` 目录下
- 包含详细的训练过程和调试信息
- 记录每层的输出形状和统计信息
- 便于与其他实现进行对比验证
- 包含完整的测试评估报告

## 实现细节

- 使用NHWC数据格式
- 支持动态批次大小
- 自动填充最后一个不完整批次
- 使用He初始化
- 支持权重的二进制存储

## 权重格式说明

> 权重格式参考了[ncnn](https://github.com/Tencent/ncnn)把网络结构和参数分开存储的方式，这样没有什么其他依赖，方便cpp做推理框架


### 权重文件格式（.npz）

当前权重保存和加载采用NumPy官方的`.npz`格式，具有更高的可靠性和兼容性。

- 每个参数以`layer_X_LayerName_param`为key存储在npz文件中。
- 保存和加载均通过`np.savez`和`np.load`实现，无需手动管理偏移、shape、dtype等。
- 推荐仅在Python/NumPy环境下直接加载使用。

示例：
```python
# 保存权重（自动完成，无需手动操作）
np.savez('weights/lenet_epoch_10_acc_0.9806.npz', **weights_dict)

# 加载权重
weights = np.load('weights/lenet_epoch_10_acc_0.9806.npz')
for key in weights:
    print(key, weights[key].shape)
```

> 旧的自定义二进制权重（.bin/.modelstruct）已废弃，现仅保留npz格式。

## 注意事项

1. 此实现专注于教育目的，不适用于生产环境
2. 单线程串行计算，训练速度较慢
3. 首次运行时会自动下载MNIST数据集
4. 建议在CPU上运行，未针对GPU优化

## TODO

- [ ] 模型权重的保存和加载，应该是这个问题导致了测试结果不对头。

## 许可证

[LICENSE](./LICENSE)

## 贡献

欢迎提交Issue和Pull Request。

