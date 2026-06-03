"""命令行审核用户纠错反馈。"""

from __future__ import annotations

import argparse
from pathlib import Path

from feedback import (
    CATEGORY_NAMES,
    approve_feedback,
    list_pending_feedback,
    load_feedback_metadata,
    reject_feedback,
)


CATEGORY_LABELS = {
    "recyclable": "可回收物",
    "kitchen_waste": "厨余垃圾",
    "hazardous_waste": "有害垃圾",
    "other_waste": "其他垃圾",
}


def format_category(category_name: str | None) -> str:
    """把英文类别转换成更容易阅读的中文类别。"""

    if not category_name:
        return "未知"
    return f"{CATEGORY_LABELS.get(category_name, category_name)}（{category_name}）"


def print_pending_feedback() -> None:
    """在终端列出所有待审核反馈，方便人工检查。"""

    pending_images = list_pending_feedback()
    if not pending_images:
        print("当前没有待审核反馈。")
        return

    print("待审核反馈图片：")
    for index, image_path in enumerate(pending_images, start=1):
        metadata = load_feedback_metadata(image_path)
        predicted_category = metadata.get("predicted_category")
        corrected_category = metadata.get("corrected_category", image_path.parent.name)
        confidence = metadata.get("confidence")
        note = metadata.get("note") or "无"
        confidence_text = "未知" if confidence is None else f"{confidence:.2%}"
        print(f"[{index}] {image_path}")
        print(f"    模型预测：{format_category(predicted_category)}")
        print(f"    用户选择：{format_category(corrected_category)}")
        print(f"    置信度：{confidence_text}")
        print(f"    备注：{note}")


def main() -> None:
    """解析命令行参数，并执行列出、通过或拒绝操作。"""

    parser = argparse.ArgumentParser(description="审核网页提交的垃圾分类纠错反馈")
    parser.add_argument("--list", action="store_true", help="列出所有待审核反馈")
    parser.add_argument("--approve", type=Path, help="审核通过某张 pending 图片")
    parser.add_argument("--reject", type=Path, help="拒绝某张 pending 图片")
    parser.add_argument(
        "--category",
        choices=CATEGORY_NAMES,
        help="审核通过时可重新指定正确类别",
    )
    args = parser.parse_args()

    if args.approve and args.reject:
        raise ValueError("--approve 和 --reject 不能同时使用。")

    if args.approve:
        target_path = approve_feedback(args.approve, args.category)
        print(f"已审核通过：{target_path}")
        print("重新训练前请运行：python train.py --force-prepare")
        return

    if args.reject:
        target_path = reject_feedback(args.reject)
        print(f"已拒绝反馈：{target_path}")
        return

    print_pending_feedback()


if __name__ == "__main__":
    main()
