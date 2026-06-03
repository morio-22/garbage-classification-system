"""垃圾分类智能识别系统：Streamlit 网页应用。"""

import os
from io import BytesIO

import streamlit as st
from PIL import Image, ImageOps

import feedback as feedback_module
from predict import DEFAULT_MODEL_PATH, load_model, predict_probabilities


FALLBACK_FEEDBACK_CATEGORY_LABELS = {
    "recyclable": "可回收物",
    "kitchen_waste": "厨余垃圾",
    "hazardous_waste": "有害垃圾",
    "other_waste": "其他垃圾",
    "not_garbage": "不是垃圾 / 不适合分类",
    "unclear_image": "图片太模糊 / 主体不清楚",
}
FEEDBACK_CATEGORY_LABELS = getattr(
    feedback_module,
    "FEEDBACK_CATEGORY_LABELS",
    FALLBACK_FEEDBACK_CATEGORY_LABELS,
)

# 兼容旧版 feedback.py：如果 Streamlit 进程仍然加载到旧模块，
# 这里补齐特殊反馈类别，避免页面保存“不是垃圾”时报未知类别。
feedback_module.CATEGORY_NAMES = tuple(FEEDBACK_CATEGORY_LABELS.keys())
PENDING_FEEDBACK_DIR = feedback_module.PENDING_FEEDBACK_DIR
approve_feedback = feedback_module.approve_feedback
list_pending_feedback = feedback_module.list_pending_feedback
load_feedback_metadata = feedback_module.load_feedback_metadata
reject_feedback = feedback_module.reject_feedback
save_pending_feedback = feedback_module.save_pending_feedback


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
AMBIGUOUS_MARGIN_THRESHOLD = 0.15
UNCERTAIN_BORDER_COLOR = "#F1C40F"
CATEGORY_LABEL_OPTIONS = {
    category_info["label"]: category_name
    for category_name, category_info in GARBAGE_INFO.items()
}
FEEDBACK_LABEL_OPTIONS = {
    category_label: category_name
    for category_name, category_label in FEEDBACK_CATEGORY_LABELS.items()
}
CATEGORY_NAME_TO_LABEL = {
    category_name: category_info["label"]
    for category_name, category_info in GARBAGE_INFO.items()
}
CATEGORY_NAME_TO_LABEL.update(FEEDBACK_CATEGORY_LABELS)


@st.cache_resource
def load_trained_model(model_modified_time: float):
    """加载并缓存模型；模型文件更新后会自动重新读取。"""

    # 修改时间只用于刷新缓存，模型仍然从固定路径加载。
    del model_modified_time
    return load_model(DEFAULT_MODEL_PATH)


def create_processed_image(image: Image.Image, category: str) -> Image.Image:
    """根据预测类别给图片添加彩色边框。"""

    border_color = GARBAGE_INFO.get(category, {}).get("color", UNCERTAIN_BORDER_COLOR)
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


def get_prediction_decision(probabilities: dict[str, float]) -> dict:
    """根据概率判断是否应该拒识，而不是强行输出垃圾类别。"""

    sorted_probabilities = sorted(
        probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    top_category, top_confidence = sorted_probabilities[0]
    second_confidence = (
        sorted_probabilities[1][1] if len(sorted_probabilities) > 1 else 0.0
    )
    margin = top_confidence - second_confidence
    reject_reasons: list[str] = []

    if top_confidence < LOW_CONFIDENCE_THRESHOLD:
        reject_reasons.append(
            f"最高置信度低于 {LOW_CONFIDENCE_THRESHOLD:.0%}"
        )

    if margin < AMBIGUOUS_MARGIN_THRESHOLD:
        reject_reasons.append(
            f"第一名和第二名概率差距小于 {AMBIGUOUS_MARGIN_THRESHOLD:.0%}"
        )

    return {
        "category": top_category,
        "confidence": top_confidence,
        "margin": margin,
        "should_reject": bool(reject_reasons),
        "reject_reasons": reject_reasons,
    }


def show_feedback_form(
    image: Image.Image,
    file_name: str,
    predicted_category: str,
    confidence: float,
    probabilities: dict[str, float],
    form_key: str,
    default_feedback_category: str | None = None,
) -> None:
    """显示纠错反馈表单，并把反馈保存到待审核目录。"""

    with st.expander("识别错了或不是垃圾？提交反馈"):
        st.write(
            "反馈会先保存到待审核目录，不会直接进入训练集。"
            "四类垃圾图片审核通过后才会用于训练；"
            "不是垃圾和图片模糊反馈只用于记录问题样本。"
        )
        st.caption(
            "当前允许反馈类别："
            + "、".join(FEEDBACK_CATEGORY_LABELS.values())
        )

        option_labels = list(FEEDBACK_LABEL_OPTIONS.keys())
        fallback_category = default_feedback_category or predicted_category
        fallback_label = CATEGORY_NAME_TO_LABEL.get(
            fallback_category,
            CATEGORY_NAME_TO_LABEL[predicted_category],
        )
        default_index = (
            option_labels.index(fallback_label)
            if fallback_label in option_labels
            else 0
        )

        with st.form(f"{form_key}_feedback_form"):
            selected_label = st.selectbox(
                "请选择你认为正确的情况",
                option_labels,
                index=default_index,
                key=f"{form_key}_corrected_category",
            )
            note = st.text_input(
                "备注（可选）",
                placeholder="例如：这不是垃圾，或者这张图太模糊",
                key=f"{form_key}_feedback_note",
            )
            confirmed = st.checkbox(
                "我确认这条反馈是真实的",
                key=f"{form_key}_feedback_confirmed",
            )
            submitted = st.form_submit_button("提交反馈")

        if not submitted:
            return

        corrected_category = FEEDBACK_LABEL_OPTIONS[selected_label]
        if not confirmed:
            st.warning("请先勾选确认框，避免误操作把错误标签提交给系统。")
            return

        if corrected_category == predicted_category and default_feedback_category is None:
            st.info("你选择的类别与模型预测一致，因此没有保存为纠错样本。")
            return

        try:
            feedback_path = save_pending_feedback(
                image=image,
                original_file_name=file_name,
                predicted_category=predicted_category,
                confidence=confidence,
                probabilities=probabilities,
                corrected_category=corrected_category,
                note=note,
            )
        except Exception as error:
            st.error(f"反馈保存失败：{error}")
            return

        st.success(
            "反馈已保存到待审核区，审核通过后才会用于重新训练。"
            f"保存位置：{feedback_path.relative_to(PENDING_FEEDBACK_DIR.parent)}"
        )


def format_category(category_name: str | None) -> str:
    """把英文类别转换成中文类别，页面上更容易看懂。"""

    if not category_name:
        return "未知"
    label = CATEGORY_NAME_TO_LABEL.get(category_name, category_name)
    return f"{label}（{category_name}）"


def show_review_probability_bars(probabilities: dict[str, float]) -> None:
    """在审核页面中显示模型保存下来的四类概率。"""

    if not probabilities:
        st.info("这条反馈没有保存概率信息。")
        return

    st.markdown("**提交时模型概率**")
    for category, probability in sorted(
        probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        label = CATEGORY_NAME_TO_LABEL.get(category, category)
        st.write(f"{label}：{probability:.2%}")
        st.progress(float(probability))


def get_admin_password() -> str:
    """读取管理员密码；优先使用环境变量，其次使用 Streamlit secrets。"""

    password_from_environment = os.environ.get("ADMIN_PASSWORD", "")
    if password_from_environment:
        return password_from_environment

    try:
        return st.secrets.get("ADMIN_PASSWORD", "")
    except Exception:
        return ""


def check_admin_access() -> bool:
    """检查是否允许访问审核页面。"""

    admin_password = get_admin_password()
    if not admin_password:
        st.warning(
            "当前没有配置管理员密码，适合本地开发使用。"
            "如果公开部署，建议设置 ADMIN_PASSWORD。"
        )
        return True

    entered_password = st.text_input("请输入管理员密码", type="password")
    if entered_password != admin_password:
        st.info("输入正确的管理员密码后，才能查看和处理待审核图片。")
        return False

    return True


def show_feedback_review_page() -> None:
    """显示管理员反馈审核页面。"""

    st.title("待审核反馈图片")
    st.caption(
        "这里显示用户提交的纠错反馈。只有审核通过的图片，才会进入下一次训练。"
    )
    if not check_admin_access():
        return

    pending_images = list_pending_feedback()
    if not pending_images:
        st.success("当前没有待审核反馈。")
        return

    st.info(f"当前共有 {len(pending_images)} 张待审核图片。")
    st.warning(
        "审核建议：图片清楚、类别确定才点通过；看不清或类别不确定就拒绝。"
    )

    option_labels = list(FEEDBACK_LABEL_OPTIONS.keys())
    for image_index, image_path in enumerate(pending_images, start=1):
        metadata = load_feedback_metadata(image_path)
        predicted_category = metadata.get("predicted_category")
        corrected_category = metadata.get("corrected_category", image_path.parent.name)
        confidence = metadata.get("confidence")
        probabilities = metadata.get("probabilities", {})
        note = metadata.get("note") or "无"
        original_file_name = metadata.get("original_file_name", image_path.name)
        default_label = CATEGORY_NAME_TO_LABEL.get(
            corrected_category,
            GARBAGE_INFO["other_waste"]["label"],
        )
        default_index = (
            option_labels.index(default_label)
            if default_label in option_labels
            else 0
        )

        with st.container(border=True):
            st.subheader(f"{image_index}. {original_file_name}")
            image_column, detail_column = st.columns([1, 1])

            with image_column:
                try:
                    st.image(
                        Image.open(image_path),
                        caption=f"待审核图片：{image_path.name}",
                        width="stretch",
                    )
                except Exception as error:
                    st.error(f"图片打开失败：{error}")
                    continue

            with detail_column:
                st.write(f"**模型预测：** {format_category(predicted_category)}")
                st.write(f"**用户选择：** {format_category(corrected_category)}")
                if confidence is None:
                    st.write("**模型置信度：** 未知")
                else:
                    st.write(f"**模型置信度：** {float(confidence):.2%}")
                st.write(f"**用户备注：** {note}")
                st.caption(f"文件位置：{image_path}")
                show_review_probability_bars(probabilities)

                final_label = st.selectbox(
                    "审核后的正确类别",
                    option_labels,
                    index=default_index,
                    key=f"review_category_{image_path.stem}",
                )
                final_category = FEEDBACK_LABEL_OPTIONS[final_label]

                approve_column, reject_column = st.columns(2)
                with approve_column:
                    if st.button(
                        "通过",
                        key=f"approve_{image_path.stem}",
                        type="primary",
                    ):
                        try:
                            target_path = approve_feedback(image_path, final_category)
                        except Exception as error:
                            st.error(f"审核通过失败：{error}")
                        else:
                            st.success(f"已通过并移动到：{target_path}")
                            st.rerun()

                with reject_column:
                    if st.button("拒绝", key=f"reject_{image_path.stem}"):
                        try:
                            target_path = reject_feedback(image_path)
                        except Exception as error:
                            st.error(f"拒绝失败：{error}")
                        else:
                            st.success(f"已拒绝并移动到：{target_path}")
                            st.rerun()

    st.divider()
    st.info("审核完成后，重新训练请运行：python train.py --force-prepare")


def show_prediction_result(
    image: Image.Image,
    file_name: str,
    file_index: int,
    model,
    class_names: list[str],
    device,
) -> None:
    """显示一张图片的原图、预测结果、概率和处理后图片。"""

    probabilities = predict_probabilities(image, model, class_names, device)
    decision = get_prediction_decision(probabilities)
    category = decision["category"]
    confidence = decision["confidence"]
    should_reject = decision["should_reject"]
    category_info = GARBAGE_INFO[category]
    border_category = "uncertain" if should_reject else category
    processed_image = create_processed_image(image, border_category)

    st.subheader(f"图片：{file_name}")
    image_column, result_column = st.columns([1, 1])

    with image_column:
        st.image(image, caption="用户上传的原始图片", width="stretch")
        st.image(
            processed_image,
            caption=(
                "处理后图片：模型无法确定"
                if should_reject
                else f"处理后图片：{category_info['label']}"
            ),
            width="stretch",
        )

    with result_column:
        category_column, confidence_column = st.columns(2)
        with category_column:
            if should_reject:
                st.metric("识别结果", "无法确定")
            else:
                st.metric("预测类别", category_info["label"])
        with confidence_column:
            st.metric("最高置信度", f"{confidence:.2%}")

        if should_reject:
            st.warning(
                "系统无法确认这张图片是否适合垃圾分类，"
                "因此不会强行给出投放类别。"
            )
            st.info(
                "建议：请确认拍摄对象确实是待投放垃圾，"
                "并尽量让单个物品占据画面中央、背景简单、光线充足。"
            )
            st.caption("拒识原因：" + "；".join(decision["reject_reasons"]))
            default_feedback_category = "not_garbage"
        else:
            st.success(f"投放建议：{category_info['advice']}")
            default_feedback_category = None

        show_probability_bars(probabilities)
        show_feedback_form(
            image=image,
            file_name=file_name,
            predicted_category=category,
            confidence=confidence,
            probabilities=probabilities,
            form_key=f"image_{file_index}",
            default_feedback_category=default_feedback_category,
        )


def show_prediction_page() -> None:
    """显示普通用户使用的图片识别页面。"""

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
                file_index,
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
        st.write(
            "网页中的纠错反馈会先进入待审核目录；只有运行 review_feedback.py "
            "审核通过的图片，才会加入下一次训练。"
        )
        st.write(
            "当最高置信度低于 60%，或前两类概率差距小于 15% 时，"
            "系统会显示无法确定，避免把明显不合适的图片强行分成某类垃圾。"
        )
        st.write(f"本次预测使用设备：{device}")


def main() -> None:
    """创建网页界面，并根据侧边栏选择显示不同页面。"""

    st.set_page_config(page_title="垃圾分类智能识别系统", page_icon="♻️", layout="wide")

    page_name = st.sidebar.radio(
        "页面",
        ["图片识别", "反馈审核"],
        help="普通用户使用图片识别；管理员使用反馈审核。",
    )

    if page_name == "反馈审核":
        show_feedback_review_page()
    else:
        show_prediction_page()


if __name__ == "__main__":
    main()
