"""自动下载公开数据，并生成中国生活垃圾四分类目录。"""

import argparse
import json
import random
import shutil
from pathlib import Path

from prepare_trashnet import IMAGE_SUFFIXES, prepare_trashnet_source


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"
SUPPLEMENTAL_RAW_DIR = DATASET_DIR / "waste_management_raw"
FOUR_CLASS_SPLIT_DIR = DATASET_DIR / "four_class_split"

# 该补充数据集使用 MIT 许可，并提供厨余、有害和其他垃圾所需类别。
SUPPLEMENTAL_REPO_ID = "omasteam/waste-garbage-management-dataset"
SUPPLEMENTAL_LICENSE = "MIT"
SUPPLEMENTAL_CLASS_NAMES = ("battery", "biological", "trash")

# 最终模型使用以下四个中国生活垃圾类别。
FOUR_CLASS_NAMES = (
    "recyclable",
    "kitchen_waste",
    "hazardous_waste",
    "other_waste",
)

# TrashNet 的五种材质都属于可回收物。
TRASHNET_RECYCLABLE_CLASS_NAMES = (
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
)


def list_images(folder: Path) -> list[Path]:
    """返回目录下所有支持的图片文件。"""

    return sorted(
        file_path
        for file_path in folder.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_SUFFIXES
    )


def count_images(folder: Path) -> int:
    """统计目录中的图片数量。"""

    return len(list_images(folder))


def supplemental_data_is_ready() -> bool:
    """检查补充数据集需要的三个类别是否已经下载完成。"""

    return all(
        count_images(SUPPLEMENTAL_RAW_DIR / class_name) > 0
        for class_name in SUPPLEMENTAL_CLASS_NAMES
    )


def download_supplemental_dataset() -> Path:
    """只下载补充数据集中需要的厨余、有害和其他垃圾图片。"""

    if supplemental_data_is_ready():
        print(f"已存在补充数据集，跳过下载：{SUPPLEMENTAL_RAW_DIR}")
        return SUPPLEMENTAL_RAW_DIR

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "缺少 huggingface_hub。请先运行：python -m pip install -r requirements.txt"
        ) from error

    SUPPLEMENTAL_RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("正在下载厨余垃圾、有害垃圾和其他垃圾补充图片，请稍候……")

    # 只下载需要的三个目录，避免下载与当前四分类无关的图片。
    snapshot_download(
        repo_id=SUPPLEMENTAL_REPO_ID,
        repo_type="dataset",
        local_dir=SUPPLEMENTAL_RAW_DIR,
        allow_patterns=[
            "battery/*",
            "biological/*",
            "trash/*",
            "README.md",
        ],
    )

    if not supplemental_data_is_ready():
        raise RuntimeError("补充数据集下载不完整，请检查网络后重新运行。")

    print(f"补充图片下载完成：{SUPPLEMENTAL_RAW_DIR}")
    return SUPPLEMENTAL_RAW_DIR


def collect_source_images(
    trashnet_source_dir: Path,
    supplemental_source_dir: Path,
) -> dict[str, list[tuple[str, Path]]]:
    """收集四个最终类别的图片来源，并给每张图片添加来源前缀。"""

    source_images: dict[str, list[tuple[str, Path]]] = {
        class_name: [] for class_name in FOUR_CLASS_NAMES
    }

    # TrashNet 中纸板、玻璃、金属、纸张和塑料都归入可回收物。
    for material_name in TRASHNET_RECYCLABLE_CLASS_NAMES:
        for image_path in list_images(trashnet_source_dir / material_name):
            source_images["recyclable"].append((f"trashnet_{material_name}", image_path))

    # 补充数据集中的类别名称与最终四分类类别一一对应。
    supplemental_mapping = {
        "kitchen_waste": "biological",
        "hazardous_waste": "battery",
        "other_waste": "trash",
    }
    for target_name, source_name in supplemental_mapping.items():
        for image_path in list_images(supplemental_source_dir / source_name):
            source_images[target_name].append((f"hf_{source_name}", image_path))

    for class_name, images in source_images.items():
        if len(images) < 2:
            raise ValueError(f"类别 {class_name} 的图片不足，至少需要 2 张。")

    return source_images


def get_split_counts(split_dir: Path) -> dict[str, dict[str, int]] | None:
    """检查四分类目录结构，并返回每个类别的图片数量。"""

    counts: dict[str, dict[str, int]] = {}
    for split_name in ("train", "val"):
        counts[split_name] = {}

        for class_name in FOUR_CLASS_NAMES:
            class_dir = split_dir / split_name / class_name
            image_count = count_images(class_dir) if class_dir.exists() else 0
            if image_count == 0:
                return None

            counts[split_name][class_name] = image_count

    return counts


def safe_remove_generated_directory(path: Path) -> None:
    """只删除 dataset 目录下由本脚本生成的四分类目录。"""

    resolved_path = path.resolve()
    allowed_parent = DATASET_DIR.resolve()
    allowed_names = {"four_class_split", "four_class_split.tmp"}

    if resolved_path.parent != allowed_parent or resolved_path.name not in allowed_names:
        raise ValueError(f"拒绝删除非自动生成目录：{resolved_path}")

    if resolved_path.exists():
        shutil.rmtree(resolved_path)


def copy_split_images(
    source_images: dict[str, list[tuple[str, Path]]],
    val_ratio: float,
    seed: int,
) -> dict[str, dict[str, int]]:
    """按照固定随机种子，将四分类图片复制到 train 和 val 目录。"""

    if not 0 < val_ratio < 1:
        raise ValueError("验证集比例必须在 0 和 1 之间。")

    temporary_dir = DATASET_DIR / "four_class_split.tmp"
    safe_remove_generated_directory(temporary_dir)
    temporary_dir.mkdir(parents=True)

    random_generator = random.Random(seed)

    for class_name, images in source_images.items():
        shuffled_images = images.copy()
        random_generator.shuffle(shuffled_images)

        # 每个类别至少保留一张训练图片和一张验证图片。
        val_count = min(
            len(shuffled_images) - 1,
            max(1, round(len(shuffled_images) * val_ratio)),
        )
        split_images = {
            "val": shuffled_images[:val_count],
            "train": shuffled_images[val_count:],
        }

        for split_name, image_items in split_images.items():
            target_dir = temporary_dir / split_name / class_name
            target_dir.mkdir(parents=True, exist_ok=True)

            for index, (source_prefix, source_path) in enumerate(image_items):
                # 添加来源和序号，避免不同目录中存在同名图片时互相覆盖。
                target_name = f"{source_prefix}_{index:05d}{source_path.suffix.lower()}"
                shutil.copy2(source_path, target_dir / target_name)

    counts = get_split_counts(temporary_dir)
    if counts is None:
        raise RuntimeError("自动生成四分类目录失败，请检查图片来源。")

    marker = {
        "sources": {
            "trashnet": "https://github.com/garythung/trashnet",
            "supplemental": f"https://huggingface.co/datasets/{SUPPLEMENTAL_REPO_ID}",
        },
        "supplemental_license": SUPPLEMENTAL_LICENSE,
        "mapping": {
            "recyclable": list(TRASHNET_RECYCLABLE_CLASS_NAMES),
            "kitchen_waste": ["biological"],
            "hazardous_waste": ["battery"],
            "other_waste": ["trash"],
        },
        "seed": seed,
        "val_ratio": val_ratio,
        "counts": counts,
    }
    (temporary_dir / ".prepared.json").write_text(
        json.dumps(marker, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    safe_remove_generated_directory(FOUR_CLASS_SPLIT_DIR)
    temporary_dir.replace(FOUR_CLASS_SPLIT_DIR)
    return counts


def prepare_four_class_dataset(
    val_ratio: float = 0.2,
    seed: int = 42,
    force: bool = False,
) -> Path:
    """下载公开图片并生成可供 ImageFolder 使用的四分类目录。"""

    if not force:
        existing_counts = get_split_counts(FOUR_CLASS_SPLIT_DIR)
        if existing_counts is not None:
            print(f"已存在四分类数据集，跳过准备步骤：{FOUR_CLASS_SPLIT_DIR}")
            print_split_summary(existing_counts)
            return FOUR_CLASS_SPLIT_DIR

    trashnet_source_dir = prepare_trashnet_source()
    supplemental_source_dir = download_supplemental_dataset()
    source_images = collect_source_images(trashnet_source_dir, supplemental_source_dir)

    print(f"正在生成垃圾四分类目录，验证集比例：{val_ratio:.0%}")
    counts = copy_split_images(source_images, val_ratio, seed)
    print(f"四分类数据集准备完成：{FOUR_CLASS_SPLIT_DIR}")
    print_split_summary(counts)
    return FOUR_CLASS_SPLIT_DIR


def print_split_summary(counts: dict[str, dict[str, int]]) -> None:
    """在终端中输出每个类别的划分数量。"""

    print("四分类数据集统计：")
    for class_name in FOUR_CLASS_NAMES:
        train_count = counts["train"][class_name]
        val_count = counts["val"][class_name]
        print(f"  {class_name:<16} 训练集：{train_count:>4} 张，验证集：{val_count:>4} 张")


def main() -> None:
    """允许用户单独运行四分类数据准备脚本。"""

    parser = argparse.ArgumentParser(description="自动准备垃圾四分类图片数据")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--force",
        action="store_true",
        help="重新生成四分类 train 和 val 目录",
    )
    args = parser.parse_args()

    prepare_four_class_dataset(args.val_ratio, args.seed, args.force)


if __name__ == "__main__":
    main()
