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

项目会自动组合三个公开数据源：

| 数据来源 | 用途 |
| --- | --- |
| [TrashNet 官方仓库](https://github.com/garythung/trashnet) | 纸板、玻璃、金属、纸张和塑料，统一映射为可回收物 |
| [waste-garbage-management-dataset](https://huggingface.co/datasets/omasteam/waste-garbage-management-dataset) | 下载 `plastic`、`biological`、`battery` 和 `trash`，用于补充塑料可回收物、厨余垃圾、有害垃圾和其他垃圾 |
| [RealWaste](https://archive.ics.uci.edu/dataset/908/realwaste) | 下载真实垃圾处理环境图片，用于增强复杂背景下的可回收物、厨余垃圾和其他垃圾识别 |

补充数据集使用 MIT 许可，RealWaste 使用 CC BY 4.0 许可。详细映射见 [dataset/README.md](dataset/README.md)。

## 项目结构

```text
.
├── app.py                         # Streamlit 网页应用
├── prepare_trashnet.py            # 自动下载和解压 TrashNet
├── prepare_realwaste.py           # 自动下载和解压 UCI RealWaste
├── prepare_dataset.py             # 自动下载补充图片并生成四分类目录
├── feedback.py                    # 保存和移动用户纠错反馈
├── review_feedback.py             # 命令行审核用户纠错反馈
├── run_app.py                     # 桌面版启动器，打包后自动打开浏览器
├── build_exe.bat                  # 生成可双击运行的程序文件夹
├── build_installer.bat            # 生成 Windows 安装包
├── installer/
│   └── setup.iss                  # Inno Setup 安装脚本
├── train.py                       # 一键准备数据并训练 ResNet18
├── predict.py                     # 单张图片预测脚本
├── requirements.txt               # Python 依赖列表
├── requirements-desktop.txt       # 桌面安装版依赖列表
├── dataset/
│   ├── README.md                  # 数据集目录和映射说明
│   ├── user_feedback/             # 本地用户纠错反馈，不上传 GitHub
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
3. 从 UCI 下载并解压 CC BY 4.0 许可的 RealWaste 真实环境图片。
4. 读取已经人工审核通过的用户纠错反馈。
5. 合并图片并生成四分类目录。
6. 按照固定随机种子划分训练集、验证集和测试集，默认为 80%、10% 和 10%。
7. 使用 `ImageFolder` 读取数据。
8. 加载 ImageNet 预训练 ResNet18，将最后一层修改为四分类层。
9. 使用温和的类别权重减轻图片数量不均衡带来的影响。
10. 根据验证损失自动降低学习率，并在长时间没有改善时提前停止训练。
11. 将验证集表现最佳的模型保存为 `models/garbage_resnet18.pth`。
12. 在独立测试集上输出准确率、精确率、召回率、F1 分数和混淆矩阵。
13. 保存训练曲线、每轮训练记录和测试集指标。

第一次运行还会自动下载 ResNet18 预训练权重。RealWaste 压缩包约为 657 MB，因此首次运行需要联网并等待下载。

自动生成的测试集不会参与训练或模型选择，可以更客观地评价模型。但这些图片仍然来自公开数据集。准备课程展示时，建议再用手机拍摄一批新图片，单独测试真实场景效果。

## 直接下载训练好的模型

如果只想运行网页，不想重新训练，可以从 GitHub Release 下载已经训练好的模型：

[garbage_resnet18.pth](https://github.com/morio-22/garbage-classification-system/releases/tag/garbage-resnet18-realwaste-v1)

下载后放到项目的 `models/` 目录中，最终路径应为：

```text
models/garbage_resnet18.pth
```

然后直接启动网页：

```powershell
python -m streamlit run app.py
```

## 制作桌面安装包

项目支持打包成 Windows 安装包，并且安装包可以自带训练好的模型。普通用户安装后不需要安装 Python，也不需要单独下载模型。

如果只想使用软件，可以直接在 GitHub Release 中下载安装包：

```text
GarbageClassificationSystem_Setup.exe
```

下载后双击安装，即可从开始菜单打开 `Garbage Classification System`。

桌面安装版只保留运行功能：

- 图片识别
- 用户反馈
- 管理员反馈审核页面
- 本地保存待审核、已通过、已拒绝的反馈图片

桌面安装版不包含训练和数据集下载功能。重新下载数据集、重新训练模型、输出混淆矩阵等操作仍然在 GitHub 开发版中完成。这样可以减少安装包体积，也避免普通用户误操作训练模型。

打包前请确认本机已经存在模型文件：

```text
models/garbage_resnet18.pth
```

第一步，生成可双击运行的程序文件夹：

```powershell
.\build_exe.ps1
```

这个脚本会使用 `requirements-desktop.txt`，只安装和打包桌面推理所需依赖。

也可以直接双击 `build_exe.bat`，它会调用同一个 PowerShell 脚本。

成功后会生成：

```text
dist/GarbageClassificationSystem/GarbageClassificationSystem.exe
```

请先双击这个 exe 测试，确认能自动打开网页。

第二步，安装 [Inno Setup 6](https://jrsoftware.org/isinfo.php)，然后生成安装包：

```powershell
.\build_installer.ps1
```

也可以直接双击 `build_installer.bat`。

成功后会生成：

```text
installer/Output/GarbageClassificationSystem_Setup.exe
```

这个安装包会包含：

- 桌面版启动程序
- Streamlit、PyTorch、torchvision 等运行依赖
- `models/garbage_resnet18.pth` 模型文件

因此普通用户只需要下载安装包、双击安装、再从桌面快捷方式打开即可。

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

网页支持一次上传一张或多张图片。每张图片都会显示原始图片、预测类别、置信度、四类概率、投放建议和处理后的图片。如果模型不确定，页面会显示“无法确定”，而不是强行输出投放类别。

网页左侧边栏可以切换两个页面：

- `图片识别`：普通用户上传图片并查看预测结果。
- `反馈审核`：管理员直观看到待审核图片，并点击通过、拒绝或改类别通过。

为了避免把“根本不是垃圾”的图片强行分进四类，网页还加入了拒识机制：

- 最高置信度低于 `60%` 时，显示“无法确定”。
- 第一名和第二名概率差距小于 `15%` 时，显示“无法确定”。

这时系统不会给出明确投放类别，而是提示用户确认图片是否真的是待投放垃圾，或重新拍摄更清楚的图片。

## 预测错误反馈

网页中每张图片下方都有“识别错了或不是垃圾？提交反馈”。用户选择正确情况并确认后，图片会先保存到：

```text
dataset/user_feedback/pending/
```

这些图片不会直接参与训练，避免用户误判导致模型学习错误标签。反馈除了四类垃圾，还支持：

- `not_garbage`：不是垃圾 / 不适合分类
- `unclear_image`：图片太模糊 / 主体不清楚

这两类特殊反馈只用于记录问题样本，默认不会进入四分类训练集。

审核流程如下：

```powershell
python review_feedback.py --list
python review_feedback.py --approve "dataset/user_feedback/pending/recyclable/某张图片.jpg"
python review_feedback.py --reject "dataset/user_feedback/pending/recyclable/某张图片.jpg"
```

如果用户选错了类别，但图片本身有价值，可以在审核通过时重新指定类别：

```powershell
python review_feedback.py --approve "dataset/user_feedback/pending/other_waste/某张图片.jpg" --category recyclable
```

审核通过的图片会移动到 `dataset/user_feedback/approved/`，下一次重新准备数据并训练时才会被使用：

```powershell
python train.py --force-prepare
```

也可以直接在网页中审核：启动 Streamlit 后，在左侧边栏选择 `反馈审核`。页面会显示待审核图片、模型预测、用户选择、置信度、备注和四类概率。

本地开发时可以不设置管理员密码。如果项目公开部署，建议设置环境变量 `ADMIN_PASSWORD` 或 Streamlit secrets 中的 `ADMIN_PASSWORD`，这样只有输入正确密码的人才能进入审核页面。
