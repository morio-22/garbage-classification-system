"""使用训练好的 ResNet18 模型识别单张垃圾图片。"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

# 项目根目录。使用绝对路径后，从其他目录运行脚本也不会找错模型文件。
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_DIR / "models" / "garbage_resnet18.pth"

# 桌面版只需要推理，不需要导入 prepare_dataset.py 中的下载和划分数据逻辑。
FOUR_CLASS_NAMES = (
    "recyclable",
    "kitchen_waste",
    "hazardous_waste",
    "other_waste",
)


def get_device() -> torch.device:
    """优先使用显卡；没有可用显卡时自动使用 CPU。"""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model(num_classes: int = len(FOUR_CLASS_NAMES)) -> nn.Module:
    """创建 ResNet18，并将最后一层替换为垃圾四分类层。"""

    # 推理时不需要再次下载 ImageNet 权重，因为稍后会加载训练好的权重。
    model = resnet18(weights=None)

    # ResNet18 原本可以识别 1000 个 ImageNet 类别。
    # 这里把最后的全连接层改为输出 4 个垃圾类别。
    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, num_classes)
    return model


def load_model(
    model_path: Path | str = DEFAULT_MODEL_PATH,
    device: torch.device | None = None,
) -> tuple[nn.Module, list[str], torch.device]:
    """加载训练脚本保存的模型权重和类别顺序。"""

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"找不到模型文件：{model_path}。请先运行 python train.py 完成训练。"
        )

    selected_device = device or get_device()

    # weights_only=True 表示只读取张量和基础数据，适合加载本项目保存的权重文件。
    checkpoint = torch.load(
        model_path,
        map_location=selected_device,
        weights_only=True,
    )
    class_names = checkpoint["class_names"]

    # 防止模型文件与当前垃圾四分类类别不匹配。
    if set(class_names) != set(FOUR_CLASS_NAMES):
        raise ValueError(
            "模型中的类别名称不正确。"
            f"应为 {sorted(FOUR_CLASS_NAMES)}，实际为 {class_names}。"
        )

    model = create_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(selected_device)

    # 推理时必须切换到评估模式，确保模型行为稳定。
    model.eval()
    return model, class_names, selected_device


def predict_image(
    image: Image.Image,
    model: nn.Module,
    class_names: list[str],
    device: torch.device,
) -> tuple[str, float]:
    """预测 PIL 图片，返回垃圾四分类类别和置信度。"""

    probabilities = predict_probabilities(image, model, class_names, device)
    category = max(probabilities, key=probabilities.get)
    return category, probabilities[category]


def predict_probabilities(
    image: Image.Image,
    model: nn.Module,
    class_names: list[str],
    device: torch.device,
) -> dict[str, float]:
    """预测 PIL 图片，返回四个垃圾类别各自的概率。"""

    # 使用官方 ResNet18 预训练权重配套的图片预处理步骤。
    # 其中包含缩放、裁剪、转张量和标准化。
    preprocess = ResNet18_Weights.DEFAULT.transforms()
    image_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)

    # 推理阶段不需要计算梯度，可以减少内存占用并提升速度。
    with torch.no_grad():
        logits = model(image_tensor)
        probability_tensor = torch.softmax(logits, dim=1).squeeze(0)

    return {
        class_name: probability_tensor[index].item()
        for index, class_name in enumerate(class_names)
    }


def predict_file(
    image_path: Path | str,
    model_path: Path | str = DEFAULT_MODEL_PATH,
) -> tuple[str, float]:
    """加载模型并预测一个本地图片文件。"""

    model, class_names, device = load_model(model_path)
    with Image.open(image_path) as image:
        return predict_image(image, model, class_names, device)


def main() -> None:
    """提供命令行预测入口。"""

    parser = argparse.ArgumentParser(description="使用 ResNet18 识别单张垃圾图片")
    parser.add_argument("image", help="待识别图片路径，例如 test.jpg")
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="训练好的模型路径",
    )
    args = parser.parse_args()

    category, confidence = predict_file(args.image, args.model_path)
    print(f"垃圾类别：{category}")
    print(f"置信度：{confidence:.2%}")


if __name__ == "__main__":
    main()
