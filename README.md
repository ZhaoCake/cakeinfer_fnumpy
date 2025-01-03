# LeNet-MNIST Pure NumPy Implementation

> After writing it, I really feel like I finally understand the network and parameters. The next step is to write the inference framework for Cpp, optimize it with HLS, and deploy it on FPGA

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

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/lenet-numpy.git
cd lenet-numpy
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 训练模型：
```bash
python main.py --epochs 10 --batch-size 64 --lr 0.005 --data-dir dataset
```

参数说明：
- `--epochs`: 训练轮数（默认：10）
- `--batch-size`: 批次大小（默认：32）
- `--lr`: 学习率（默认：0.01）
- `--data-dir`: 数据集存储目录（默认：'dataset'）

2. 模型权重：
- 权重文件保存在 `weights/` 目录下
- 包含二进制权重文件（.bin）和结构描述文件（.modelstruct）

3. 训练日志：
- 日志文件保存在 `logs/` 目录下
- 包含详细的训练过程和调试信息

## 实现细节

- 使用NHWC数据格式
- 支持动态批次大小
- 自动填充最后一个不完整批次
- 使用He初始化
- 支持权重的二进制存储

## 权重格式说明

> 权重格式参考了[ncnn](https://github.com/Tencent/ncnn)把网络结构和参数分开存储的方式，这样没有什么其他依赖，方便cpp做推理框架

### 模型结构文件 (.modelstruct)

JSON格式的文本文件，记录了权重的元数据：
```json
{
    "model_name": "lenet_epoch_1_acc_0.9312",
    "layer_weights": {
        "layer_0_Conv2D": {
            "W": {
                "shape": [5, 5, 1, 6],    // [height, width, in_channels, filters]
                "dtype": "float32",
                "offset": 0,              // 在二进制文件中的偏移量
                "size": 150               // 参数总数
            },
            "b": {
                "shape": [6],
                "dtype": "float32",
                "offset": 600,
                "size": 6
            }
        },
        "layer_3_Conv2D": {
            "W": {
                "shape": [5, 5, 6, 16],
                "dtype": "float32",
                "offset": 624,
                "size": 2400
            },
            "b": {
                "shape": [16],
                "dtype": "float32",
                "offset": 10224,
                "size": 16
            }
        }
        // ... 其他层的权重信息
    }
}
```

### 二进制权重文件 (.bin)

按照modelstruct中指定的偏移量和数据类型存储的原始权重数据：
- 所有数据使用小端序（Little-Endian）存储
- 默认使用32位浮点数（float32）
- 数据按层顺序连续存储
- 每层内部按 weights(W) -> bias(b) 顺序排列
- 多维数组按行优先（C-style）展平

示例：
```cpp
// 读取卷积层权重的C++代码示例
struct ConvWeight {
    float W[5][5][1][6];  // NHWC格式
    float b[6];
};

// 从文件指定偏移量读取
ConvWeight conv1;
std::ifstream file("lenet_epoch_1_acc_0.9312.bin", std::ios::binary);
file.seekg(0);  // 使用modelstruct中的offset
file.read(reinterpret_cast<char*>(&conv1), sizeof(ConvWeight));
```

## 注意事项

1. 此实现专注于教育目的，不适用于生产环境
2. 单线程串行计算，训练速度较慢
3. 首次运行时会自动下载MNIST数据集
4. 建议在CPU上运行，未针对GPU优化

## 许可证

[LICENSE](./LICENSE)

## 贡献

欢迎提交Issue和Pull Request。

