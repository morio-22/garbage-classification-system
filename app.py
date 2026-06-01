"""垃圾分类智能识别系统：Streamlit 网页应用。"""

from io import BytesIO

import streamlit as st
from PIL import Image, ImageOps

from predict import DEFAULT_MODEL_PATH, load_model, predict_image


# 四类垃圾的中文名称、投放建议和处理后图片边框颜色。
GARBAGE_INFO = {
    "recyclable": {
        "label": "可回收物",
        "advice": "请尽量保持物品清洁干燥，投入可回收物收集容器。",
        "color": "#2E86DE",
    },
    "kitchen_waste": {
        "label": "厨余垃圾",
        "advice": "请沥干水分，去除塑料袋等包装后，投入厨余垃圾收集容器。",
        "color": "#27AE60",
    },
    "hazardous_waste": {
        "label": "有害垃圾",
        "advice": "请轻放并保持完整，投入有害垃圾收集容器，避免造成污染。",
        "color": "#E74C3C",
    },
    "other_waste": {
        "label": "其他垃圾",
        "advice": "请确认没有可回收部分后，投入其他垃圾收集容器。",
        "color": "#7F8C8D",
    },
}


@st.cache_resource
def load_trained_model():
    """加载并缓存模型，避免网页每次刷新时重复读取权重。"""

    return load_model(DEFAULT_MODEL_PATH)


def create_processed_image(image: Image.Image, category: str) -> Image.Image:
    """根据预测类别给图片添加彩色边框。"""

    border_color = GARBAGE_INFO[category]["color"]
    return ImageOps.expand(image, border=12, fill=border_color)


def main() -> None:
    """创建网页界面并处理用户上传的图片。"""

    st.set_page_config(page_title="垃圾分类智能识别系统", page_icon="♻️", layout="wide")

    st.title("垃圾分类智能识别系统")
    st.caption("使用公开垃圾图片数据和预训练 ResNet18 构建的生活垃圾四分类识别系统。")

    # 网页预测依赖训练后生成的模型文件。
    # 如果模型不存在，先告诉用户一键训练命令。
    if not DEFAULT_MODEL_PATH.exists():
        st.warning("尚未找到训练好的模型，请先运行一键训练命令。")
        st.code("python train.py", language="powershell")
        st.info("脚本会自动准备四分类图片、训练模型并输出混淆矩阵。")
        return

    uploaded_file = st.file_uploader(
        "请上传一张垃圾图片",
        type=["jpg", "jpeg", "png"],
        help="支持 JPG、JPEG 和 PNG 格式。",
    )

    if uploaded_file is None:
        st.info("请先上传图片，系统会在这里显示真实模型的识别结果。")
        return

    image_bytes = uploaded_file.getvalue()

    # 尝试读取图片。如果文件损坏，则提示用户重新上传。
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        st.error("图片读取失败，请重新上传一张有效的 JPG 或 PNG 图片。")
        return

    st.subheader("1. 原始图片")
    st.image(image, caption="用户上传的图片", width="stretch")

    # 加载训练好的模型并完成真实推理。
    try:
        model, class_names, device = load_trained_model()
        category, confidence = predict_image(image, model, class_names, device)
    except Exception as error:
        st.error(f"模型加载或预测失败：{error}")
        return

    category_info = GARBAGE_INFO[category]
    processed_image = create_processed_image(image, category)

    st.subheader("2. 识别结果")
    category_column, confidence_column = st.columns(2)
    with category_column:
        st.metric("预测类别", category_info["label"])
        st.caption(f"模型内部类别名称：{category}")
    with confidence_column:
        st.metric("模型置信度", f"{confidence:.2%}")

    st.success(f"投放建议：{category_info['advice']}")

    st.subheader("3. 处理后的图片")
    st.image(
        processed_image,
        caption=f"识别结果：{category_info['label']}（彩色边框表示系统已经完成处理）",
        width="stretch",
    )

    with st.expander("关于当前模型"):
        st.write(
            "当前模型使用公开图片训练四个类别："
            "可回收物、厨余垃圾、有害垃圾和其他垃圾。"
        )
        st.write(
            "可回收物图片来自 TrashNet；厨余垃圾、有害垃圾和其他垃圾图片"
            "来自 MIT 许可的 waste-garbage-management-dataset。"
        )
        st.write(f"本次预测使用设备：{device}")


if __name__ == "__main__":
    main()
