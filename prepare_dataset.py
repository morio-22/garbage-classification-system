"""自动下载公开数据，并生成中国生活垃圾四分类目录。"""

import argparse
import json
import random
import shutil
from pathlib import Path

from prepare_realwaste import (
    REALWASTE_DATASET_PAGE,
    REALWASTE_LICENSE,
    prepare_realwaste_source,
)
from prepare_trashnet import IMAGE_SUFFIXES, prepare_trashnet_source


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"
SUPPLEMENTAL_RAW_DIR = DATASET_DIR / "waste_management_raw"
FOUR_CLASS_SPLIT_DIR = DATASET_DIR / "four_class_split"
# 修改数据来源或映射规则时递增版本号，使旧目录自动重新生成。
DATASET_PREPARATION_VERSION = 3

# 该补充数据集使用 MIT 许可，并提供塑料、厨余、有害和其他垃圾所需类别。
SUPPLEMENTAL_REPO_ID = "omasteam/waste-garbage-management-dataset"
SUPPLEMENTAL_LICENSE = "MIT"
SUPPLEMENTAL_CLASS_NAMES = ("battery", "biological", "plastic", "trash")

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

# RealWaste 来自真实垃圾处理环境，可以补充复杂背景图片。
REALWASTE_MAPPING = {
    "recyclable": ("Cardboard", "Glass", "Metal", "Paper", "Plastic"),
    "kitchen_waste": ("Food Organics", "Vegetation"),
    "other_waste": ("Miscellaneous Trash", "Textile Trash"),
}


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
    """检查补充数据集需要的四个类别是否已经下载完成。"""

    return all(
        count_images(SUPPLEMENTAL_RAW_DIR / class_name) > 0
        for class_name in SUPPLEMENTAL_CLASS_NAMES
    )


def download_supplemental_dataset() -> Path:
    """只下载补充数据集中需要的塑料、厨余、有害和其他垃圾图片。"""

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
    print("正在下载塑料、厨余垃圾、有害垃圾和其他垃圾补充图片，请稍候……")

    # 只下载需要的四个目录，避免下载与当前四分类无关的图片。
    snapshot_download(
        repo_id=SUPPLEMENTAL_REPO_ID,
        repo_type="dataset",
        local_dir=SUPPLEMENTAL_RAW_DIR,
        allow_patterns=[
            "battery/*",
            "biological/*",
            "plastic/*",
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
    realwaste_source_dir: Path,
) -> dict[str, list[tuple[str, Path]]]:
    """收集四个最终类别的图片来源，并给每张图片添加来源前缀。"""

    source_images: dict[str, list[tuple[str, Path]]] = {
        class_name: [] for class_name in FOUR_CLASS_NAMES
    }

    # TrashNet 中纸板、玻璃、金属、纸张和塑料都归入可回收物。
    for material_name in TRASHNET_RECYCLABLE_CLASS_NAMES:
        for image_path in list_images(trashnet_source_dir / material_name):
            source_images["recyclable"].append((f"trashnet_{material_name}", image_path))

    # 增加补充数据集中的塑料图片，提升对真实塑料瓶等常见物品的适应能力。
    for image_path in list_images(supplemental_source_dir / "plastic"):
        source_images["recyclable"].append(("hf_plastic", image_path))

    # 补充数据集中的类别名称与最终四分类类别一一对应。
    supplemental_mapping = {
        "kitchen_waste": "biological",
        "hazardous_waste": "battery",
        "other_waste": "trash",
    }
    for target_name, source_name in supplemental_mapping.items():
        for image_path in list_images(supplemental_source_dir / source_name):
            source_images[target_name].append((f"hf_{source_name}", image_path))

    # RealWaste 图片来自真实垃圾处理环境，用于增强复杂背景下的识别能力。
    for target_name, source_names in REALWASTE_MAPPING.items():
        for source_name in source_names:
            source_prefix = source_name.lower().replace(" ", "_")
            for image_path in list_images(realwaste_source_dir / source_name):
                source_images[target_name].append((f"realwaste_{source_prefix}", image_path))

    for class_name, images in source_images.items():
        if len(images) < 3:
            raise ValueError(f"类别 {class_name} 的图片不足，至少需要 3 张。")

    return source_images


def get_split_counts(split_dir: Path) -> dict[str, dict[str, int]] | None:
    """检查四分类目录结构，并返回每个类别的图片数量。"""

    counts: dict[str, dict[str, int]] = {}
    for split_name in ("train", "val", "test"):
        counts[split_name] = {}

        for class_name in FOUR_CLASS_NAMES:
            class_dir = split_dir / split_name / class_name
            image_count = count_images(class_dir) if class_dir.exists() else 0
            if image_count == 0:
                return None

            counts[split_name][class_name] = image_count

    return counts


def split_is_current(
    split_dir: Path,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> bool:
    """检查已经生成的数据是否使用当前划分参数。"""

    marker_path = split_dir / ".prepared.json"
    if get_split_counts(split_dir) is None or not marker_path.exists():
        return False

    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False

    return (
        marker.get("version") == DATASET_PREPARATION_VERSION
        and marker.get("seed") == seed
        and marker.get("val_ratio") == val_ratio
        and marker.get("test_ratio") == test_ratio
    )


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
    test_ratio: float,
    seed: int,
) -> dict[str, dict[str, int]]:
    """按照固定随机种子，将图片复制到 train、val 和 test 目录。"""

    if not 0 < val_ratio < 1 or not 0 < test_ratio < 1:
        raise ValueError("验证集和测试集比例必须在 0 和 1 之间。")
    if val_ratio + test_ratio >= 1:
        raise ValueError("验证集比例与测试集比例之和必须小于 1。")

    temporary_dir = DATASET_DIR / "four_class_split.tmp"
    safe_remove_generated_directory(temporary_dir)
    temporary_dir.mkdir(parents=True)

    random_generator = random.Random(seed)

    for class_name, images in source_images.items():
        shuffled_images = images.copy()
        random_generator.shuffle(shuffled_images)

        # 三份数据互不重叠，并且每个类别至少保留一张图片。
        val_count = max(1, round(len(shuffled_images) * val_ratio))
        test_count = max(1, round(len(shuffled_images) * test_ratio))
        while val_count + test_count >= len(shuffled_images):
            if val_count >= test_count and val_count > 1:
                val_count -= 1
            elif test_count > 1:
                test_count -= 1
            else:
                raise ValueError(f"类别 {class_name} 的图片不足，至少需要 3 张。")

        split_images = {
            "val": shuffled_images[:val_count],
            "test": shuffled_images[val_count : val_count + test_count],
            "train": shuffled_images[val_count + test_count :],
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
        "version": DATASET_PREPARATION_VERSION,
        "sources": {
            "trashnet": "https://github.com/garythung/trashnet",
            "supplemental": f"https://huggingface.co/datasets/{SUPPLEMENTAL_REPO_ID}",
            "realwaste": REALWASTE_DATASET_PAGE,
        },
        "supplemental_license": SUPPLEMENTAL_LICENSE,
        "realwaste_license": REALWASTE_LICENSE,
        "mapping": {
            "recyclable": [
                *TRASHNET_RECYCLABLE_CLASS_NAMES,
                "supplemental:plastic",
                *[f"realwaste:{name}" for name in REALWASTE_MAPPING["recyclable"]],
            ],
            "kitchen_waste": [
                "biological",
                *[f"realwaste:{name}" for name in REALWASTE_MAPPING["kitchen_waste"]],
            ],
            "hazardous_waste": ["battery"],
            "other_waste": [
                "trash",
                *[f"realwaste:{name}" for name in REALWASTE_MAPPING["other_waste"]],
            ],
        },
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
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
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    force: bool = False,
) -> Path:
    """下载公开图片并生成可供 ImageFolder 使用的四分类目录。"""

    if not force:
        if split_is_current(FOUR_CLASS_SPLIT_DIR, val_ratio, test_ratio, seed):
            existing_counts = get_split_counts(FOUR_CLASS_SPLIT_DIR)
            assert existing_counts is not None
            print(f"已存在四分类数据集，跳过准备步骤：{FOUR_CLASS_SPLIT_DIR}")
            print_split_summary(existing_counts)
            return FOUR_CLASS_SPLIT_DIR

    trashnet_source_dir = prepare_trashnet_source()
    supplemental_source_dir = download_supplemental_dataset()
    realwaste_source_dir = prepare_realwaste_source()
    source_images = collect_source_images(
        trashnet_source_dir,
        supplemental_source_dir,
        realwaste_source_dir,
    )

    print(
        f"正在生成垃圾四分类目录，验证集比例：{val_ratio:.0%}，"
        f"测试集比例：{test_ratio:.0%}"
    )
    counts = copy_split_images(source_images, val_ratio, test_ratio, seed)
    print(f"四分类数据集准备完成：{FOUR_CLASS_SPLIT_DIR}")
    print_split_summary(counts)
    return FOUR_CLASS_SPLIT_DIR


def print_split_summary(counts: dict[str, dict[str, int]]) -> None:
    """在终端中输出每个类别的划分数量。"""

    print("四分类数据集统计：")
    for class_name in FOUR_CLASS_NAMES:
        train_count = counts["train"][class_name]
        val_count = counts["val"][class_name]
        test_count = counts["test"][class_name]
        print(
            f"  {class_name:<16} 训练集：{train_count:>4} 张，"
            f"验证集：{val_count:>4} 张，测试集：{test_count:>4} 张"
        )


def main() -> None:
    """允许用户单独运行四分类数据准备脚本。"""

    parser = argparse.ArgumentParser(description="自动准备垃圾四分类图片数据")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--force",
        action="store_true",
        help="重新生成四分类 train、val 和 test 目录",
    )
    args = parser.parse_args()

    prepare_four_class_dataset(args.val_ratio, args.test_ratio, args.seed, args.force)


if __name__ == "__main__":
    main()
