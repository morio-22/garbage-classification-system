"""自动准备 TrashNet，并使用预训练 ResNet18 完成训练。"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18

from predict import DEFAULT_MODEL_PATH
from prepare_dataset import (
    FOUR_CLASS_NAMES,
    prepare_four_class_dataset,
)

# 使用无需桌面窗口的后端，保证脚本在普通终端和服务器上都能保存图片。
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFUSION_MATRIX_PATH = PROJECT_DIR / "models" / "confusion_matrix.png"
DEFAULT_TRAINING_CURVES_PATH = PROJECT_DIR / "models" / "training_curves.png"
DEFAULT_HISTORY_PATH = PROJECT_DIR / "models" / "training_history.csv"
DEFAULT_METRICS_PATH = PROJECT_DIR / "models" / "metrics.json"


def validate_dataset(dataset_dir: Path) -> None:
    """检查自动生成的 train、val 和 test 目录是否包含垃圾四分类图片。"""

    for split_name in ("train", "val", "test"):
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
    class_weight_power: float,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str], torch.Tensor]:
    """使用 ImageFolder 创建训练、验证和测试集加载器。"""

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
    test_dataset = datasets.ImageFolder(
        dataset_dir / "test",
        transform=val_transform,
    )

    # ImageFolder 会按照目录名称的字母顺序分配数字标签。
    # 保存这个顺序后，预测脚本才能正确解释模型输出。
    if not (
        train_dataset.classes == val_dataset.classes == test_dataset.classes
    ):
        raise ValueError("训练集、验证集和测试集的类别顺序不一致。")

    # 图片数量较少的类别会获得更高权重，减轻类别数量不均衡带来的影响。
    # 默认使用平方根反比例，避免对少数类别补偿过强，导致模糊图片被过度纠正。
    class_counts = torch.bincount(
        torch.tensor(train_dataset.targets),
        minlength=len(train_dataset.classes),
    )
    class_weights = (
        len(train_dataset) / (len(train_dataset.classes) * class_counts)
    ).pow(class_weight_power)

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
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset.classes,
        class_weights,
    )


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
    """在验证集或测试集上评估模型，并生成混淆矩阵。"""

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


def load_checkpoint(model: nn.Module, model_path: Path, device: torch.device) -> dict:
    """载入验证集表现最佳的模型，用于最终测试。"""

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def calculate_classification_metrics(
    matrix: torch.Tensor,
    class_names: list[str],
) -> dict[str, dict[str, float | int]]:
    """根据混淆矩阵计算每个类别的精确率、召回率和 F1 分数。"""

    metrics: dict[str, dict[str, float | int]] = {}
    for class_index, class_name in enumerate(class_names):
        true_positive = matrix[class_index, class_index].item()
        predicted_count = matrix[:, class_index].sum().item()
        actual_count = matrix[class_index, :].sum().item()

        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / actual_count if actual_count else 0.0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        metrics[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "support": actual_count,
        }

    # 宏平均会给四个类别相同权重，避免只关注图片最多的可回收物。
    metrics["macro_average"] = {
        metric_name: sum(
            float(metrics[class_name][metric_name]) for class_name in class_names
        )
        / len(class_names)
        for metric_name in ("precision", "recall", "f1_score")
    }
    metrics["macro_average"]["support"] = sum(
        int(metrics[class_name]["support"]) for class_name in class_names
    )
    return metrics


def print_classification_report(
    metrics: dict[str, dict[str, float | int]],
    class_names: list[str],
) -> None:
    """在终端中输出测试集分类指标。"""

    print("\n测试集分类指标：")
    print(f"{'类别':<18}{'精确率':>10}{'召回率':>10}{'F1':>10}{'图片数':>10}")
    for class_name in [*class_names, "macro_average"]:
        row = metrics[class_name]
        print(
            f"{class_name:<18}"
            f"{float(row['precision']):>10.2%}"
            f"{float(row['recall']):>10.2%}"
            f"{float(row['f1_score']):>10.2%}"
            f"{int(row['support']):>10}"
        )


def save_training_history(history: list[dict[str, float]], output_path: Path) -> None:
    """将每轮训练结果保存为 CSV，便于后续整理实验报告。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(f"训练记录已保存：{output_path}")


def save_training_curves(history: list[dict[str, float]], output_path: Path) -> None:
    """保存损失和准确率曲线，方便判断模型是否过拟合。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], marker="o", label="val")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()

    axes[1].plot(
        epochs,
        [row["train_accuracy"] for row in history],
        marker="o",
        label="train",
    )
    axes[1].plot(
        epochs,
        [row["val_accuracy"] for row in history],
        marker="o",
        label="val",
    )
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    print(f"训练曲线已保存：{output_path}")


def save_metrics(
    best_val_accuracy: float,
    test_accuracy: float,
    metrics: dict[str, dict[str, float | int]],
    output_path: Path,
) -> None:
    """将最终评估指标保存为 JSON，便于网页或实验报告继续使用。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "best_val_accuracy": best_val_accuracy,
        "test_accuracy": test_accuracy,
        "test_macro_f1": metrics["macro_average"]["f1_score"],
        "test_classification_report": metrics,
    }
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"测试指标已保存：{output_path}")


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
        title="Four-class garbage test confusion matrix",
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
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="测试集比例")
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
        help="重新生成垃圾四分类的 train、val 和 test 划分目录。",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="关闭类别权重。默认会提高图片较少类别的重要程度。",
    )
    parser.add_argument(
        "--class-weight-power",
        type=float,
        default=0.5,
        help="类别权重强度。默认为 0.5，即使用平方根反比例权重。",
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=1,
        help="验证损失连续多少轮未改善后降低学习率。",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="验证准确率连续多少轮未改善后提前停止。设置为 0 可关闭。",
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
    parser.add_argument(
        "--training-curves-path",
        type=Path,
        default=DEFAULT_TRAINING_CURVES_PATH,
        help="训练曲线图片保存路径",
    )
    parser.add_argument(
        "--history-path",
        type=Path,
        default=DEFAULT_HISTORY_PATH,
        help="每轮训练记录保存路径",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="最终测试指标保存路径",
    )
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("训练轮数必须大于等于 1。")
    if args.lr_patience < 0 or args.early_stopping_patience < 0:
        raise ValueError("等待轮数不能为负数。")
    if args.class_weight_power < 0:
        raise ValueError("类别权重强度不能为负数。")

    # 固定 PyTorch 随机种子，使重复训练更容易比较。
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_dir = prepare_four_class_dataset(
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        force=args.force_prepare,
    )
    validate_dataset(dataset_dir)
    train_loader, val_loader, test_loader, class_names, class_weights = create_data_loaders(
        dataset_dir,
        args.batch_size,
        args.num_workers,
        args.class_weight_power,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")
    print(f"类别顺序：{class_names}")

    model = create_model(len(class_names), args.fine_tune).to(device)
    if args.no_class_weights:
        loss_function = nn.CrossEntropyLoss()
        print("类别权重：已关闭")
    else:
        loss_function = nn.CrossEntropyLoss(weight=class_weights.to(device))
        weight_text = ", ".join(
            f"{class_name}={weight:.3f}"
            for class_name, weight in zip(class_names, class_weights.tolist())
        )
        print(f"类别权重：{weight_text}")

    optimizer = Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.lr_patience,
    )

    best_val_accuracy = -1.0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            device,
        )
        val_loss, val_accuracy, _ = evaluate(
            model,
            val_loader,
            loss_function,
            device,
            len(class_names),
        )
        scheduler.step(val_loss)
        current_learning_rate = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "learning_rate": current_learning_rate,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )

        # 每轮都会输出训练准确率，便于观察模型学习过程。
        print(
            f"第 {epoch:02d}/{args.epochs:02d} 轮 | "
            f"训练损失：{train_loss:.4f} | "
            f"训练准确率：{train_accuracy:.2%} | "
            f"验证损失：{val_loss:.4f} | "
            f"验证准确率：{val_accuracy:.2%} | "
            f"学习率：{current_learning_rate:.6f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            save_checkpoint(model, class_names, args.model_path, best_val_accuracy)
            print(f"已保存当前最佳模型：{args.model_path}")
        else:
            epochs_without_improvement += 1

        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print("验证准确率长时间没有提高，提前结束训练。")
            break

    save_training_history(history, args.history_path)
    save_training_curves(history, args.training_curves_path)

    # 最终测试必须使用验证集表现最佳的权重，不能直接使用最后一轮权重。
    checkpoint = load_checkpoint(model, args.model_path, device)
    test_loss, test_accuracy, test_confusion_matrix = evaluate(
        model,
        test_loader,
        loss_function,
        device,
        len(class_names),
    )
    metrics = calculate_classification_metrics(test_confusion_matrix, class_names)

    print(f"\n训练完成。最佳验证准确率：{best_val_accuracy:.2%}")
    print(f"独立测试集损失：{test_loss:.4f}，准确率：{test_accuracy:.2%}")
    print_classification_report(metrics, class_names)
    print_confusion_matrix(test_confusion_matrix, class_names)
    save_confusion_matrix_plot(
        test_confusion_matrix,
        class_names,
        args.confusion_matrix_path,
    )
    save_metrics(best_val_accuracy, test_accuracy, metrics, args.metrics_path)

    # 在权重文件中记录测试结果，预测脚本仍然可以照常载入该文件。
    checkpoint["test_accuracy"] = test_accuracy
    checkpoint["test_macro_f1"] = metrics["macro_average"]["f1_score"]
    torch.save(checkpoint, args.model_path)


if __name__ == "__main__":
    main()
