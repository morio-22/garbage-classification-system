"""自动下载并解压 UCI RealWaste 数据集。"""

from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

from prepare_trashnet import IMAGE_SUFFIXES, ensure_safe_archive


# UCI Machine Learning Repository 提供的 RealWaste 官方下载地址。
REALWASTE_ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/908/realwaste.zip"
REALWASTE_DATASET_PAGE = "https://archive.ics.uci.edu/dataset/908/realwaste"
REALWASTE_LICENSE = "CC BY 4.0"

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"
RAW_DIR = DATASET_DIR / "realwaste_raw"
ARCHIVE_PATH = RAW_DIR / "realwaste.zip"
EXTRACT_DIR = RAW_DIR / "extracted"

# RealWaste 包含以下九种真实垃圾处理环境类别。
REALWASTE_CLASS_NAMES = (
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Miscellaneous Trash",
    "Paper",
    "Plastic",
    "Textile Trash",
    "Vegetation",
)


def count_images(folder: Path) -> int:
    """统计目录中的图片数量。"""

    return sum(
        1
        for file_path in folder.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_SUFFIXES
    )


def download_archive() -> None:
    """从 UCI 官方地址下载 RealWaste 压缩包。"""

    if ARCHIVE_PATH.exists() and ARCHIVE_PATH.stat().st_size > 0:
        print(f"已存在 RealWaste 压缩包，跳过下载：{ARCHIVE_PATH}")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = ARCHIVE_PATH.with_suffix(".zip.part")
    request = Request(
        REALWASTE_ARCHIVE_URL,
        headers={"User-Agent": "Python Garbage Classification Demo"},
    )

    print("正在下载 RealWaste 数据集，文件约 657 MB，请稍候……")
    with urlopen(request) as response, temporary_path.open("wb") as output_file:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded_size = 0

        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break

            output_file.write(chunk)
            downloaded_size += len(chunk)

            if total_size:
                progress = downloaded_size / total_size
                print(f"\r下载进度：{progress:.1%}", end="")
            else:
                print(f"\r已下载：{downloaded_size / 1024 / 1024:.1f} MB", end="")

    temporary_path.replace(ARCHIVE_PATH)
    print(f"\nRealWaste 下载完成：{ARCHIVE_PATH}")


def find_source_directory() -> Path | None:
    """查找包含 RealWaste 九个类别子目录的原始图片目录。"""

    if not EXTRACT_DIR.exists():
        return None

    expected_names = set(REALWASTE_CLASS_NAMES)
    candidates = [EXTRACT_DIR, *EXTRACT_DIR.rglob("*")]
    for folder in candidates:
        if not folder.is_dir():
            continue

        child_directories = {child.name for child in folder.iterdir() if child.is_dir()}
        if expected_names.issubset(child_directories):
            return folder

    return None


def extract_archive() -> None:
    """安全解压 RealWaste 压缩包。"""

    if find_source_directory() is not None:
        print(f"已存在解压后的 RealWaste 数据集，跳过解压：{EXTRACT_DIR}")
        return

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    print("正在解压 RealWaste 数据集……")

    with ZipFile(ARCHIVE_PATH) as zip_file:
        ensure_safe_archive(zip_file, EXTRACT_DIR)
        zip_file.extractall(EXTRACT_DIR)

    print(f"RealWaste 解压完成：{EXTRACT_DIR}")


def prepare_realwaste_source() -> Path:
    """下载并解压 RealWaste，返回包含九个原始类别的图片目录。"""

    download_archive()
    extract_archive()

    source_dir = find_source_directory()
    if source_dir is None:
        raise FileNotFoundError("解压后未找到 RealWaste 九个类别目录。")

    for class_name in REALWASTE_CLASS_NAMES:
        if count_images(source_dir / class_name) == 0:
            raise FileNotFoundError(f"RealWaste 类别目录中没有图片：{class_name}")

    return source_dir


def main() -> None:
    """允许用户单独下载和解压 RealWaste。"""

    source_dir = prepare_realwaste_source()
    print(f"RealWaste 图片目录：{source_dir}")
    for class_name in REALWASTE_CLASS_NAMES:
        print(f"  {class_name:<20} {count_images(source_dir / class_name):>4} 张")


if __name__ == "__main__":
    main()
