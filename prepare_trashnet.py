"""自动下载、解压并划分 TrashNet 数据集。"""

import argparse
import json
import random
import shutil
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile


# TrashNet 官方 GitHub 仓库目前指向 Hugging Face 下载数据集。
TRASHNET_ARCHIVE_URL = (
    "https://huggingface.co/datasets/garythung/trashnet/"
    "resolve/main/dataset-resized.zip?download=true"
)

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"
RAW_DIR = DATASET_DIR / "trashnet_raw"
ARCHIVE_PATH = RAW_DIR / "dataset-resized.zip"
EXTRACT_DIR = RAW_DIR / "extracted"
SPLIT_DIR = DATASET_DIR / "trashnet_split"

# TrashNet 原始数据集包含以下六种材质类别。
TRASHNET_CLASS_NAMES = (
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(folder: Path) -> int:
    """统计目录中的图片数量。"""

    return sum(
        1
        for file_path in folder.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_SUFFIXES
    )


def download_archive() -> None:
    """从 TrashNet 官方下载地址获取压缩包。"""

    if ARCHIVE_PATH.exists() and ARCHIVE_PATH.stat().st_size > 0:
        print(f"已存在数据集压缩包，跳过下载：{ARCHIVE_PATH}")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = ARCHIVE_PATH.with_suffix(".zip.part")
    request = Request(TRASHNET_ARCHIVE_URL, headers={"User-Agent": "Python TrashNet Demo"})

    print("正在下载 TrashNet 数据集，请稍候……")
    with urlopen(request) as response, temporary_path.open("wb") as output_file:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded_size = 0

        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break

            output_file.write(chunk)
            downloaded_size += len(chunk)

            # 下载地址未提供总大小时，仍然显示已下载的数据量。
            if total_size:
                progress = downloaded_size / total_size
                print(f"\r下载进度：{progress:.1%}", end="")
            else:
                print(f"\r已下载：{downloaded_size / 1024 / 1024:.1f} MB", end="")

    temporary_path.replace(ARCHIVE_PATH)
    print(f"\n下载完成：{ARCHIVE_PATH}")


def ensure_safe_archive(zip_file: ZipFile, destination: Path) -> None:
    """检查压缩包路径，防止文件被解压到目标目录之外。"""

    destination = destination.resolve()
    for member in zip_file.infolist():
        target_path = (destination / member.filename).resolve()
        if destination != target_path and destination not in target_path.parents:
            raise ValueError(f"压缩包包含不安全路径：{member.filename}")


def extract_archive() -> None:
    """解压下载后的 TrashNet 压缩包。"""

    if find_source_directory() is not None:
        print(f"已存在解压后的数据集，跳过解压：{EXTRACT_DIR}")
        return

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    print("正在解压 TrashNet 数据集……")

    with ZipFile(ARCHIVE_PATH) as zip_file:
        ensure_safe_archive(zip_file, EXTRACT_DIR)
        zip_file.extractall(EXTRACT_DIR)

    print(f"解压完成：{EXTRACT_DIR}")


def find_source_directory() -> Path | None:
    """查找包含 TrashNet 六个类别子目录的原始图片目录。"""

    if not EXTRACT_DIR.exists():
        return None

    candidates = [EXTRACT_DIR, *EXTRACT_DIR.rglob("*")]
    for folder in candidates:
        if not folder.is_dir():
            continue

        child_directories = {child.name for child in folder.iterdir() if child.is_dir()}
        if set(TRASHNET_CLASS_NAMES).issubset(child_directories):
            return folder

    return None


def get_split_counts(split_dir: Path) -> dict[str, dict[str, int]] | None:
    """检查自动划分后的目录结构，并返回每个类别的图片数量。"""

    counts: dict[str, dict[str, int]] = {}
    for split_name in ("train", "val"):
        counts[split_name] = {}

        for class_name in TRASHNET_CLASS_NAMES:
            class_dir = split_dir / split_name / class_name
            image_count = count_images(class_dir) if class_dir.exists() else 0
            if image_count == 0:
                return None

            counts[split_name][class_name] = image_count

    return counts


def safe_remove_generated_directory(path: Path) -> None:
    """只删除 dataset 目录下由本脚本生成的划分目录。"""

    resolved_path = path.resolve()
    allowed_parent = DATASET_DIR.resolve()
    allowed_names = {"trashnet_split", "trashnet_split.tmp"}

    if resolved_path.parent != allowed_parent or resolved_path.name not in allowed_names:
        raise ValueError(f"拒绝删除非自动生成目录：{resolved_path}")

    if resolved_path.exists():
        shutil.rmtree(resolved_path)


def split_dataset(
    source_dir: Path,
    val_ratio: float,
    seed: int,
) -> dict[str, dict[str, int]]:
    """按照固定随机种子，将 TrashNet 自动划分为训练集和验证集。"""

    if not 0 < val_ratio < 1:
        raise ValueError("验证集比例必须在 0 和 1 之间。")

    temporary_dir = DATASET_DIR / "trashnet_split.tmp"
    safe_remove_generated_directory(temporary_dir)
    temporary_dir.mkdir(parents=True)

    random_generator = random.Random(seed)

    for class_name in TRASHNET_CLASS_NAMES:
        image_paths = sorted(
            file_path
            for file_path in (source_dir / class_name).iterdir()
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_SUFFIXES
        )
        if len(image_paths) < 2:
            raise ValueError(f"类别 {class_name} 的图片数量不足，至少需要 2 张图片。")

        random_generator.shuffle(image_paths)
        # 即使用户把验证集比例设得很高，也至少保留一张训练图片。
        val_count = min(
            len(image_paths) - 1,
            max(1, round(len(image_paths) * val_ratio)),
        )
        split_images = {
            "val": image_paths[:val_count],
            "train": image_paths[val_count:],
        }

        for split_name, paths in split_images.items():
            target_dir = temporary_dir / split_name / class_name
            target_dir.mkdir(parents=True, exist_ok=True)

            for source_path in paths:
                shutil.copy2(source_path, target_dir / source_path.name)

    counts = get_split_counts(temporary_dir)
    if counts is None:
        raise RuntimeError("自动划分数据集失败，请检查下载后的图片目录。")

    marker = {
        "source": TRASHNET_ARCHIVE_URL,
        "seed": seed,
        "val_ratio": val_ratio,
        "counts": counts,
    }
    (temporary_dir / ".prepared.json").write_text(
        json.dumps(marker, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    safe_remove_generated_directory(SPLIT_DIR)
    temporary_dir.replace(SPLIT_DIR)
    return counts


def prepare_trashnet_dataset(
    val_ratio: float = 0.2,
    seed: int = 42,
    force: bool = False,
) -> Path:
    """执行完整准备流程，并返回可供 ImageFolder 使用的数据目录。"""

    if not force:
        existing_counts = get_split_counts(SPLIT_DIR)
        if existing_counts is not None:
            print(f"已存在划分后的 TrashNet 数据集，跳过准备步骤：{SPLIT_DIR}")
            print_split_summary(existing_counts)
            return SPLIT_DIR

    source_dir = prepare_trashnet_source()

    print(f"正在自动划分训练集和验证集，验证集比例：{val_ratio:.0%}")
    counts = split_dataset(source_dir, val_ratio, seed)
    print(f"数据集准备完成：{SPLIT_DIR}")
    print_split_summary(counts)
    return SPLIT_DIR


def prepare_trashnet_source() -> Path:
    """下载并解压 TrashNet，返回包含六个原始类别的图片目录。"""

    download_archive()
    extract_archive()

    source_dir = find_source_directory()
    if source_dir is None:
        raise FileNotFoundError("解压后未找到 TrashNet 六个类别目录。")

    return source_dir


def print_split_summary(counts: dict[str, dict[str, int]]) -> None:
    """在终端中输出每个类别的划分数量。"""

    print("数据集划分统计：")
    for class_name in TRASHNET_CLASS_NAMES:
        train_count = counts["train"][class_name]
        val_count = counts["val"][class_name]
        print(f"  {class_name:<10} 训练集：{train_count:>3} 张，验证集：{val_count:>3} 张")


def main() -> None:
    """允许用户单独运行数据集准备脚本。"""

    parser = argparse.ArgumentParser(description="自动下载并划分 TrashNet 数据集")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--force",
        action="store_true",
        help="重新生成 train 和 val 划分目录",
    )
    args = parser.parse_args()

    prepare_trashnet_dataset(args.val_ratio, args.seed, args.force)


if __name__ == "__main__":
    main()
