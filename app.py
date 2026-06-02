"""垃圾分类智能识别系统：Streamlit 网页应用。"""

from io import BytesIO

import streamlit as st
from PIL import Image, ImageOps

from predict import DEFAULT_MODEL_PATH, load_model, predict_probabilities


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
LOW_CONFIDENCE_THRESHOLD = 0.6


@st.cache_resource
def load_trained_model(model_modified_time: float):
    """加载并缓存模型；模型文件更新后会自动重新读取。"""

    # 修改时间只用于刷新缓存，模型仍然从固定路径加载。
    del model_modified_time
    return load_model(DEFAULT_MODEL_PATH)


def create_processed_image(image: Image.Image, category: str) -> Image.Image:
    """根据预测类别给图片添加彩色边框。"""

    border_color = GARBAGE_INFO[category]["color"]
    return ImageOps.expand(image, border=12, fill=border_color)


def show_probability_bars(probabilities: dict[str, float]) -> None:
    """按照概率从高到低显示四类垃圾的预测结果。"""

    st.markdown("**四类概率**")
    for category, probability in sorted(
        probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        label = GARBAGE_INFO[category]["label"]
        st.write(f"{label}：{probability:.2%}")
        st.progress(probability)


def show_prediction_result(
    image: Image.Image,
    file_name: str,
    model,
    class_names: list[str],
    device,
) -> None:
    """显示一张图片的原图、预测结果、概率和处理后图片。"""

    probabilities = predict_probabilities(image, model, class_names, device)
    category = max(probabilities, key=probabilities.get)
    confidence = probabilities[category]
    category_info = GARBAGE_INFO[category]
    processed_image = create_processed_image(image, category)

    st.subheader(f"图片：{file_name}")
    image_column, result_column = st.columns([1, 1])

    with image_column:
        st.image(image, caption="用户上传的原始图片", width="stretch")
        st.image(
            processed_image,
            caption=f"处理后图片：{category_info['label']}",
            width="stretch",
        )

    with result_column:
        category_column, confidence_column = st.columns(2)
        with category_column:
            st.metric("预测类别", category_info["label"])
        with confidence_column:
            st.metric("模型置信度", f"{confidence:.2%}")

        if confidence < LOW_CONFIDENCE_THRESHOLD:
            st.warning("模型置信度较低，建议人工确认后再投放。")

        st.success(f"投放建议：{category_info['advice']}")
        show_probability_bars(probabilities)


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

    uploaded_files = st.file_uploader(
        "请上传一张或多张垃圾图片",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="支持 JPG、JPEG 和 PNG 格式，可以一次选择多张图片。",
    )

    if not uploaded_files:
        st.info("请先上传图片，系统会逐张显示真实模型的识别结果。")
        return

    # 模型只需要加载一次，然后复用于本次上传的所有图片。
    try:
        model, class_names, device = load_trained_model(
            DEFAULT_MODEL_PATH.stat().st_mtime
        )
    except Exception as error:
        st.error(f"模型加载失败：{error}")
        return

    st.write(f"本次共上传 {len(uploaded_files)} 张图片。")
    for file_index, uploaded_file in enumerate(uploaded_files):
        if file_index > 0:
            st.divider()

        # 尝试读取图片。如果文件损坏，则提示用户并继续处理其他图片。
        try:
            image = Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")
            show_prediction_result(
                image,
                uploaded_file.name,
                model,
                class_names,
                device,
            )
        except Exception as error:
            st.error(f"{uploaded_file.name} 处理失败：{error}")

    with st.expander("关于当前模型"):
        st.write(
            "当前模型使用公开图片训练四个类别："
            "可回收物、厨余垃圾、有害垃圾和其他垃圾。"
        )
        st.write(
            "可回收物图片来自 TrashNet；厨余垃圾、有害垃圾和其他垃圾图片"
            "来自 MIT 许可的 waste-garbage-management-dataset。"
        )
        st.write(
            "模型还会使用 CC BY 4.0 许可的 RealWaste 图片，"
            "增强真实垃圾处理环境和复杂背景下的识别能力。"
        )
        st.write(f"本次预测使用设备：{device}")


if __name__ == "__main__":
    main()
