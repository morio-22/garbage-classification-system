"""自动准备 TrashNet，并使用预训练 ResNet18 完成训练。"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18

from predict import DEFAULT_MODEL_PATH
from prepare_dataset import (
    FOUR_CLASS_NAMES,
    prepare_four_class_dataset,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFUSION_MATRIX_PATH = PROJECT_DIR / "models" / "confusion_matrix.png"


def validate_dataset(dataset_dir: Path) -> None:
    """检查自动生成的 train 和 val 目录是否包含垃圾四分类图片。"""

    for split_name in ("train", "val"):
        split_dir = dataset_dir / split_name
        if not split_dir.exists():
            raise FileNotFoundError(f"缺少数据目录：{split_dir}")

        actual_class_names = {
            folder.name for folder in split_dir.iterdir() if folder.is_dir()
        }
        if actual_class_names != set(FOUR_CLASS_NAMES):
            raise ValueError(
                f"{split_dir} 的类别目录不正确。\n"
                f"应为：{sorted(FOUR_CLASS_NAMES)}\n"
                f"实际为：{sorted(actual_class_names)}"
            )


def create_data_loaders(
    dataset_dir: Path,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """使用 ImageFolder 创建训练集和验证集加载器。"""

    # ImageNet 预训练模型使用以下均值和标准差完成图片标准化。
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # 训练时加入随机裁剪和翻转，帮助模型适应不同拍摄角度。
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # 验证时不需要随机变化，保证每次评估结果可比较。
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = datasets.ImageFolder(
        dataset_dir / "train",
        transform=train_transform,
    )
    val_dataset = datasets.ImageFolder(
        dataset_dir / "val",
        transform=val_transform,
    )

    # ImageFolder 会按照目录名称的字母顺序分配数字标签。
    # 保存这个顺序后，预测脚本才能正确解释模型输出。
    if train_dataset.classes != val_dataset.classes:
        raise ValueError("训练集和验证集的类别顺序不一致。")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, train_dataset.classes


def create_model(num_classes: int, fine_tune: bool) -> nn.Module:
    """加载预训练 ResNet18，并替换最后一层。"""

    # 第一次运行时，torchvision 会自动下载 ImageNet 预训练权重。
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # 初学阶段默认只训练最后一层，速度更快，也更适合普通电脑。
    # 添加 --fine-tune 参数后，可以训练整个网络以尝试提高准确率。
    if not fine_tune:
        for parameter in model.parameters():
            parameter.requires_grad = False

    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, num_classes)
    return model


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: Adam,
    device: torch.device,
) -> tuple[float, float]:
    """完成一轮训练，返回平均损失和训练准确率。"""

    model.train()

    # 默认只训练最后一层时，让冻结的批归一化层保持评估模式。
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            parameters = list(module.parameters())
            if parameters and all(not parameter.requires_grad for parameter in parameters):
                module.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_images = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct_predictions += (logits.argmax(dim=1) == labels).sum().item()
        total_images += images.size(0)

    return total_loss / total_images, correct_predictions / total_images


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    num_classes: int,
) -> tuple[float, float, torch.Tensor]:
    """在验证集上评估模型，并生成混淆矩阵。"""

    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_images = 0
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_function(logits, labels)
            predictions = logits.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            correct_predictions += (predictions == labels).sum().item()
            total_images += images.size(0)

            # 行表示真实类别，列表示模型预测类别。
            indexes = labels.cpu() * num_classes + predictions.cpu()
            confusion_matrix += torch.bincount(
                indexes,
                minlength=num_classes * num_classes,
            ).reshape(num_classes, num_classes)

    return (
        total_loss / total_images,
        correct_predictions / total_images,
        confusion_matrix,
    )


def save_checkpoint(
    model: nn.Module,
    class_names: list[str],
    model_path: Path,
    val_accuracy: float,
) -> None:
    """保存模型权重、类别顺序和最佳验证准确率。"""

    model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "architecture": "resnet18",
        "dataset": "TrashNet + waste-garbage-management-dataset",
        "class_names": class_names,
        "model_state_dict": model.state_dict(),
        "val_accuracy": val_accuracy,
    }
    torch.save(checkpoint, model_path)


def print_confusion_matrix(matrix: torch.Tensor, class_names: list[str]) -> None:
    """在终端中输出便于阅读的混淆矩阵。"""

    print("\n混淆矩阵：行表示真实类别，列表示预测类别。")
    print(" " * 14 + "".join(f"{name[:8]:>10}" for name in class_names))
    for class_name, row in zip(class_names, matrix.tolist()):
        values = "".join(f"{value:>10}" for value in row)
        print(f"{class_name[:12]:>12}  {values}")


def save_confusion_matrix_plot(
    matrix: torch.Tensor,
    class_names: list[str],
    output_path: Path,
) -> None:
    """将混淆矩阵保存为 PNG 图片。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix.numpy(), cmap="Blues")
    figure.colorbar(image, ax=axis)

    axis.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Four-class garbage validation confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")

    for row_index in range(len(class_names)):
        for column_index in range(len(class_names)):
            axis.text(
                column_index,
                row_index,
                str(matrix[row_index, column_index].item()),
                ha="center",
                va="center",
                color="black",
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    print(f"混淆矩阵图片已保存：{output_path}")


def main() -> None:
    """自动准备 TrashNet，训练模型并输出评估结果。"""

    parser = argparse.ArgumentParser(description="自动准备四分类垃圾图片并训练 ResNet18")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="每批图片数量")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="学习率")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="划分数据集的随机种子")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="读取图片的子进程数量。Windows 初学者建议保留默认值 0。",
    )
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="训练整个 ResNet18。默认只训练最后一层，速度更快。",
    )
    parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="重新生成垃圾四分类的 train 和 val 划分目录。",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="模型保存路径",
    )
    parser.add_argument(
        "--confusion-matrix-path",
        type=Path,
        default=DEFAULT_CONFUSION_MATRIX_PATH,
        help="混淆矩阵图片保存路径",
    )
    args = parser.parse_args()

    # 固定 PyTorch 随机种子，使重复训练更容易比较。
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_dir = prepare_four_class_dataset(
        val_ratio=args.val_ratio,
        seed=args.seed,
        force=args.force_prepare,
    )
    validate_dataset(dataset_dir)
    train_loader, val_loader, class_names = create_data_loaders(
        dataset_dir,
        args.batch_size,
        args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")
    print(f"类别顺序：{class_names}")

    model = create_model(len(class_names), args.fine_tune).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )

    best_val_accuracy = 0.0
    best_confusion_matrix = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            device,
        )
        val_loss, val_accuracy, confusion_matrix = evaluate(
            model,
            val_loader,
            loss_function,
            device,
            len(class_names),
        )

        # 每轮都会输出训练准确率，便于观察模型学习过程。
        print(
            f"第 {epoch:02d}/{args.epochs:02d} 轮 | "
            f"训练损失：{train_loss:.4f} | "
            f"训练准确率：{train_accuracy:.2%} | "
            f"验证损失：{val_loss:.4f} | "
            f"验证准确率：{val_accuracy:.2%}"
        )

        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_confusion_matrix = confusion_matrix.clone()
            save_checkpoint(model, class_names, args.model_path, best_val_accuracy)
            print(f"已保存当前最佳模型：{args.model_path}")

    if best_confusion_matrix is None:
        raise RuntimeError("没有生成混淆矩阵，请检查训练轮数。")

    print(f"\n训练完成。最佳验证准确率：{best_val_accuracy:.2%}")
    print_confusion_matrix(best_confusion_matrix, class_names)
    save_confusion_matrix_plot(
        best_confusion_matrix,
        class_names,
        args.confusion_matrix_path,
    )


if __name__ == "__main__":
    main()
