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
│   └── trash/                    # 映射为其他垃圾
└── four_class_split/
    ├── train/                    # 自动划分的训练集，默认为 80%
    │   ├── recyclable/
    │   ├── kitchen_waste/
    │   ├── hazardous_waste/
    │   └── other_waste/
    └── val/                      # 自动划分的验证集，默认为 20%
        ├── recyclable/
        ├── kitchen_waste/
        ├── hazardous_waste/
        └── other_waste/
```

## 类别映射

| 最终类别 | 中文名称 | 图片来源类别 |
| --- | --- | --- |
| `recyclable` | 可回收物 | TrashNet：`cardboard`、`glass`、`metal`、`paper`、`plastic` |
| `kitchen_waste` | 厨余垃圾 | 补充数据集：`biological` |
| `hazardous_waste` | 有害垃圾 | 补充数据集：`battery` |
| `other_waste` | 其他垃圾 | 补充数据集：`trash` |

## 数据来源

- [TrashNet 官方 GitHub 仓库](https://github.com/garythung/trashnet)
- [waste-garbage-management-dataset](https://huggingface.co/datasets/omasteam/waste-garbage-management-dataset)：MIT 许可

## 重新划分数据集

默认情况下，脚本会复用已经准备好的目录。如果需要重新划分，请运行：

```powershell
python train.py --force-prepare
```

可以修改验证集比例和随机种子：

```powershell
python train.py --val-ratio 0.2 --seed 42 --force-prepare
```
