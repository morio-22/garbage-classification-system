# 垃圾四分类数据集说明

不需要手工下载或整理图片。运行以下命令即可自动获取公开图片、生成目录并训练模型：

```powershell
python train.py
```

## 自动生成的目录

```text
dataset/
├── trashnet_raw/                 # TrashNet：提供可回收物图片
├── waste_management_raw/         # MIT 许可补充数据集
│   ├── biological/               # 映射为厨余垃圾
│   ├── battery/                  # 映射为有害垃圾
│   ├── plastic/                  # 补充塑料可回收物图片
│   └── trash/                    # 映射为其他垃圾
├── realwaste_raw/                # CC BY 4.0 许可的真实垃圾处理环境图片
├── user_feedback/                # 用户纠错反馈，默认不会上传 GitHub
│   ├── pending/                  # 用户刚提交、尚未审核的图片
│   ├── approved/                 # 审核通过后才会参与下一次训练
│   └── rejected/                 # 审核拒绝，不参与训练
└── four_class_split/
    ├── train/                    # 自动划分的训练集，默认为 80%
    │   ├── recyclable/
    │   ├── kitchen_waste/
    │   ├── hazardous_waste/
    │   └── other_waste/
    ├── val/                      # 自动划分的验证集，默认为 10%
    │   ├── recyclable/
    │   ├── kitchen_waste/
    │   ├── hazardous_waste/
    │   └── other_waste/
    └── test/                     # 独立测试集，默认为 10%
        ├── recyclable/
        ├── kitchen_waste/
        ├── hazardous_waste/
        └── other_waste/
```

## 类别映射

| 最终类别 | 中文名称 | 图片来源类别 |
| --- | --- | --- |
| `recyclable` | 可回收物 | TrashNet：`cardboard`、`glass`、`metal`、`paper`、`plastic`；补充数据集：`plastic`；RealWaste：`Cardboard`、`Glass`、`Metal`、`Paper`、`Plastic` |
| `kitchen_waste` | 厨余垃圾 | 补充数据集：`biological`；RealWaste：`Food Organics`、`Vegetation` |
| `hazardous_waste` | 有害垃圾 | 补充数据集：`battery` |
| `other_waste` | 其他垃圾 | 补充数据集：`trash`；RealWaste：`Miscellaneous Trash`、`Textile Trash` |

如果存在 `dataset/user_feedback/approved/类别名/`，其中已经人工审核通过的图片也会加入对应类别。`pending/` 和 `rejected/` 中的图片不会参与训练。

## 数据来源

- [TrashNet 官方 GitHub 仓库](https://github.com/garythung/trashnet)
- [waste-garbage-management-dataset](https://huggingface.co/datasets/omasteam/waste-garbage-management-dataset)：MIT 许可
- [RealWaste - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/908/realwaste)：CC BY 4.0 许可；作者为 Sam Single、Saeid Iranmanesh、Raad Raad

`test/` 中的图片不会参与训练和模型选择，但仍然来自上述公开数据源。课程展示前，建议再用手机拍摄一批新图片，额外检查真实场景中的识别效果。

说明：RealWaste 的 `Vegetation` 在本项目的四分类体系中映射为厨余垃圾，用于表示容易腐烂的有机物。不同城市的具体投放规则可能不同，实际使用时应以当地垃圾分类规定为准。

## 用户反馈审核

网页提交的纠错图片会先进入 `dataset/user_feedback/pending/`。这样做是为了避免用户误判把错误标签直接加入训练集。

更直观的审核方式是启动网页后，在左侧边栏选择 `反馈审核`。页面会直接显示待审核图片、模型预测、用户选择、备注和概率，并提供通过、拒绝和改类别通过操作。

如果想使用命令行审核，可以先列出待审核图片：

```powershell
python review_feedback.py --list
```

确认图片和类别都正确后，通过审核：

```powershell
python review_feedback.py --approve "dataset/user_feedback/pending/recyclable/某张图片.jpg"
```

如果图片类别不对但仍然有价值，可以重新指定类别：

```powershell
python review_feedback.py --approve "dataset/user_feedback/pending/other_waste/某张图片.jpg" --category recyclable
```

如果图片模糊、类别不确定或用户判断明显错误，则拒绝：

```powershell
python review_feedback.py --reject "dataset/user_feedback/pending/other_waste/某张图片.jpg"
```

审核完成后，重新生成数据划分并训练：

```powershell
python train.py --force-prepare
```

## 重新划分数据集

默认情况下，脚本会复用已经准备好的目录。如果需要重新划分，请运行：

```powershell
python train.py --force-prepare
```

可以修改验证集、测试集比例和随机种子。脚本会自动识别参数变化并重新划分：

```powershell
python train.py --val-ratio 0.1 --test-ratio 0.1 --seed 42
```
