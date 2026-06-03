"""用户纠错反馈的保存和审核工具。"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent
FEEDBACK_DIR = PROJECT_DIR / "dataset" / "user_feedback"
PENDING_FEEDBACK_DIR = FEEDBACK_DIR / "pending"
APPROVED_FEEDBACK_DIR = FEEDBACK_DIR / "approved"
REJECTED_FEEDBACK_DIR = FEEDBACK_DIR / "rejected"

# 用户反馈仍然只允许进入这四个最终类别。
CATEGORY_NAMES = (
    "recyclable",
    "kitchen_waste",
    "hazardous_waste",
    "other_waste",
)


def ensure_feedback_directories() -> None:
    """创建待审核、已通过和已拒绝三个反馈目录。"""

    for root_dir in (
        PENDING_FEEDBACK_DIR,
        APPROVED_FEEDBACK_DIR,
        REJECTED_FEEDBACK_DIR,
    ):
        for category_name in CATEGORY_NAMES:
            (root_dir / category_name).mkdir(parents=True, exist_ok=True)


def sanitize_filename(file_name: str) -> str:
    """把上传文件名转换成安全的文件名片段。"""

    stem = Path(file_name).stem.strip().lower()
    safe_stem = re.sub(r"[^a-z0-9_-]+", "_", stem)
    safe_stem = safe_stem.strip("_")
    return safe_stem or "uploaded_image"


def get_utc_timestamp() -> str:
    """生成适合放进文件名和元数据的 UTC 时间字符串。"""

    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def make_unique_path(target_path: Path) -> Path:
    """如果目标文件已存在，就在文件名后追加序号，避免覆盖旧反馈。"""

    if not target_path.exists():
        return target_path

    for index in range(1, 1000):
        candidate = target_path.with_name(
            f"{target_path.stem}_{index}{target_path.suffix}"
        )
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"无法生成不重复的反馈文件名：{target_path}")


def save_json(path: Path, data: dict) -> None:
    """用 UTF-8 保存 JSON 元数据，方便之后人工审核。"""

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_feedback_metadata(image_path: Path) -> dict:
    """读取某张反馈图片旁边的 JSON 元数据。"""

    metadata_path = image_path.with_suffix(".json")
    if not metadata_path.exists():
        return {}

    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_pending_feedback(
    image: Image.Image,
    original_file_name: str,
    predicted_category: str,
    confidence: float,
    probabilities: dict[str, float],
    corrected_category: str,
    note: str = "",
) -> Path:
    """把用户纠错保存到待审核区，不直接加入训练集。"""

    if corrected_category not in CATEGORY_NAMES:
        raise ValueError(f"未知反馈类别：{corrected_category}")

    ensure_feedback_directories()
    timestamp = get_utc_timestamp()
    short_id = uuid4().hex[:8]
    safe_name = sanitize_filename(original_file_name)
    target_dir = PENDING_FEEDBACK_DIR / corrected_category
    image_path = make_unique_path(target_dir / f"{timestamp}_{short_id}_{safe_name}.jpg")

    # 统一保存为 RGB JPEG，避免 PNG 透明通道等格式差异影响后续训练。
    image.convert("RGB").save(image_path, format="JPEG", quality=95)

    metadata = {
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "original_file_name": original_file_name,
        "image_file": image_path.name,
        "predicted_category": predicted_category,
        "confidence": float(confidence),
        "probabilities": {
            category_name: float(probability)
            for category_name, probability in probabilities.items()
        },
        "corrected_category": corrected_category,
        "note": note.strip(),
        "review_tip": "只有人工审核通过后，这张图片才会进入下一次训练。",
    }
    save_json(image_path.with_suffix(".json"), metadata)
    return image_path


def list_pending_feedback() -> list[Path]:
    """列出所有待审核反馈图片。"""

    ensure_feedback_directories()
    return sorted(PENDING_FEEDBACK_DIR.rglob("*.jpg"))


def ensure_pending_image_path(image_path: Path) -> Path:
    """确认传入路径确实位于 pending 目录下，防止误移动其他文件。"""

    resolved_image_path = image_path.resolve()
    resolved_pending_dir = PENDING_FEEDBACK_DIR.resolve()

    try:
        resolved_image_path.relative_to(resolved_pending_dir)
    except ValueError as error:
        raise ValueError(f"只能审核 pending 目录下的反馈图片：{image_path}") from error

    if not resolved_image_path.exists():
        raise FileNotFoundError(f"反馈图片不存在：{image_path}")

    return resolved_image_path


def move_feedback_image(
    image_path: Path,
    target_root_dir: Path,
    status: str,
    target_category: str | None = None,
) -> Path:
    """把反馈图片连同元数据一起移动到审核结果目录。"""

    source_image_path = ensure_pending_image_path(image_path)
    source_category = source_image_path.parent.name
    final_category = target_category or source_category
    if final_category not in CATEGORY_NAMES:
        raise ValueError(f"未知目标类别：{final_category}")

    ensure_feedback_directories()
    metadata = load_feedback_metadata(source_image_path)
    target_dir = target_root_dir / final_category
    target_dir.mkdir(parents=True, exist_ok=True)
    target_image_path = make_unique_path(target_dir / source_image_path.name)

    shutil.move(str(source_image_path), str(target_image_path))

    source_metadata_path = source_image_path.with_suffix(".json")
    if source_metadata_path.exists():
        source_metadata_path.unlink()

    metadata.update(
        {
            "status": status,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "approved_category": final_category if status == "approved" else None,
        }
    )
    save_json(target_image_path.with_suffix(".json"), metadata)
    return target_image_path


def approve_feedback(image_path: Path, category: str | None = None) -> Path:
    """审核通过反馈；通过后的图片会参与下一次数据准备和训练。"""

    return move_feedback_image(image_path, APPROVED_FEEDBACK_DIR, "approved", category)


def reject_feedback(image_path: Path) -> Path:
    """拒绝错误或不确定的反馈；拒绝后的图片不会参与训练。"""

    return move_feedback_image(image_path, REJECTED_FEEDBACK_DIR, "rejected")
