# 垃圾分类智能识别系统

这是一个适合 Python 初学者阅读的生活垃圾四分类 Web 应用。项目使用 PyTorch、torchvision、ResNet18 和 Streamlit。运行一个命令即可自动获取公开图片、划分训练集、验证集与测试集、训练模型、保存权重并输出评估结果。

## 支持类别

| 英文类别 | 中文名称 |
| --- | --- |
| `recyclable` | 可回收物 |
| `kitchen_waste` | 厨余垃圾 |
| `hazardous_waste` | 有害垃圾 |
| `other_waste` | 其他垃圾 |

## 数据来源

项目会自动组合两个公开数据源：

| 数据来源 | 用途 |
| --- | --- |
| [TrashNet 官方仓库](https://github.com/garythung/trashnet) | 纸板、玻璃、金属、纸张和塑料，统一映射为可回收物 |
| [waste-garbage-management-dataset](https://huggingface.co/datasets/omasteam/waste-garbage-management-dataset) | 下载 `plastic`、`biological`、`battery` 和 `trash`，用于补充塑料可回收物、厨余垃圾、有害垃圾和其他垃圾 |

补充数据集使用 MIT 许可。详细映射见 [dataset/README.md](dataset/README.md)。

## 项目结构

```text
.
├── app.py                         # Streamlit 网页应用
├── prepare_trashnet.py            # 自动下载和解压 TrashNet
├── prepare_dataset.py             # 自动下载补充图片并生成四分类目录
├── train.py                       # 一键准备数据并训练 ResNet18
├── predict.py                     # 单张图片预测脚本
├── requirements.txt               # Python 依赖列表
├── dataset/
│   ├── README.md                  # 数据集目录和映射说明
│   └── four_class_split/          # 自动生成的四分类 train/val/test
└── models/
    ├── garbage_resnet18.pth       # 训练后生成的最佳模型
    ├── confusion_matrix.png       # 独立测试集混淆矩阵
    ├── training_curves.png        # 损失和准确率曲线
    ├── training_history.csv       # 每轮训练记录
    └── metrics.json               # 测试集准确率和 F1 指标
```

## 安装步骤

### 1. 进入项目目录

```powershell
cd "C:\Users\hsttv\Documents\Codex\2026-06-01\python-python-web-1-python-2"
```

### 2. 创建并激活虚拟环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. 安装依赖

```powershell
python -m pip install -r requirements.txt
```

如果需要使用 NVIDIA 显卡加速训练，请根据 [PyTorch 官网安装页](https://pytorch.org/get-started/locally/) 选择与你电脑环境匹配的安装命令。

## 一键获取数据并训练

```powershell
python train.py
```

该命令会自动执行：

1. 下载并解压 TrashNet。
2. 从 MIT 许可补充数据集中下载塑料可回收物、厨余垃圾、有害垃圾和其他垃圾图片。
3. 合并图片并生成四分类目录。
4. 按照固定随机种子划分训练集、验证集和测试集，默认为 80%、10% 和 10%。
5. 使用 `ImageFolder` 读取数据。
6. 加载 ImageNet 预训练 ResNet18，将最后一层修改为四分类层。
7. 使用温和的类别权重减轻图片数量不均衡带来的影响。
8. 根据验证损失自动降低学习率，并在长时间没有改善时提前停止训练。
9. 将验证集表现最佳的模型保存为 `models/garbage_resnet18.pth`。
10. 在独立测试集上输出准确率、精确率、召回率、F1 分数和混淆矩阵。
11. 保存训练曲线、每轮训练记录和测试集指标。

第一次运行还会自动下载 ResNet18 预训练权重，因此需要联网。

自动生成的测试集不会参与训练或模型选择，可以更客观地评价模型。但这些图片仍然来自公开数据集。准备课程展示时，建议再用手机拍摄一批新图片，单独测试真实场景效果。

## 常用训练参数

```powershell
python train.py --epochs 20 --batch-size 16
python train.py --fine-tune
python train.py --force-prepare
python train.py --val-ratio 0.1 --test-ratio 0.1
```

- 默认只训练最后一层，速度较快，适合入门。
- `--fine-tune` 会训练整个 ResNet18，通常更慢，但可能提升准确率。
- `--force-prepare` 会重新生成四分类训练集、验证集和测试集目录。
- `--no-class-weights` 可以关闭默认启用的类别权重。
- `--class-weight-power` 可以调整类别权重强度，默认值为 `0.5`。
- `--early-stopping-patience 0` 可以关闭提前停止。

## 命令行预测

训练完成后，可以先测试一张图片：

```powershell
python predict.py "你的图片路径.jpg"
```

## 启动网页

训练完成后运行：

```powershell
python -m streamlit run app.py
```

然后打开：

```text
http://localhost:8501
```

网页会显示原始图片、预测类别、置信度、投放建议和处理后的图片。
