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

## 数据来源

- [TrashNet 官方 GitHub 仓库](https://github.com/garythung/trashnet)
- [waste-garbage-management-dataset](https://huggingface.co/datasets/omasteam/waste-garbage-management-dataset)：MIT 许可
- [RealWaste - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/908/realwaste)：CC BY 4.0 许可；作者为 Sam Single、Saeid Iranmanesh、Raad Raad

`test/` 中的图片不会参与训练和模型选择，但仍然来自上述公开数据源。课程展示前，建议再用手机拍摄一批新图片，额外检查真实场景中的识别效果。

说明：RealWaste 的 `Vegetation` 在本项目的四分类体系中映射为厨余垃圾，用于表示容易腐烂的有机物。不同城市的具体投放规则可能不同，实际使用时应以当地垃圾分类规定为准。

## 重新划分数据集

默认情况下，脚本会复用已经准备好的目录。如果需要重新划分，请运行：

```powershell
python train.py --force-prepare
```

可以修改验证集、测试集比例和随机种子。脚本会自动识别参数变化并重新划分：

```powershell
python train.py --val-ratio 0.1 --test-ratio 0.1 --seed 42
```
