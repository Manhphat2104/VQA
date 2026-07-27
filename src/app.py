import os
import sys
import gc
import torch
import streamlit as st
from PIL import Image

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Chỉ import hàm dự đoán cho model B2 - PaliGemma Fine-tuned
from Inference.predict_B2 import run_predict_B2

B2_CHECKPOINT_DIR = r"D:\Code\Deep Learning\Endterm\B2_finetuned_model"

# Cấu hình giao diện Streamlit
st.set_page_config(page_title="Hệ Thống VQA Tiếng Việt", layout="wide")

def clear_vram():
    """Hàm dọn dẹp bộ nhớ RAM và GPU sau khi dự đoán xong"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# GIAO DIỆN CHÍNH
st.title(" Hệ Thống Hỏi Đáp Trên Ảnh (VQA) - PaliGemma Fine-tuned")

col1, col2 = st.columns([1, 1])
with col1:
    st.header("1. Cung cấp đầu vào")
    uploaded_file = st.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png"])
    question = st.text_input("Nhập câu hỏi tiếng Việt:")

    submit_btn = st.button("Chạy dự đoán", use_container_width=True, type="primary")

    if submit_btn:
        if uploaded_file is None or not question.strip():
            st.warning(" Vui lòng cung cấp đầy đủ Ảnh và Câu hỏi!")
        else:
            with st.spinner("Đang tính toán..."):
                try:
                    ans = run_predict_B2(uploaded_file, question, B2_CHECKPOINT_DIR)
                    clear_vram()

                    # HIỂN THỊ KẾT QUẢ NGAY DƯỚI NÚT BẤM
                    st.markdown("---")
                    st.success(" Kết quả dự đoán:")
                    st.info(f"**{ans.upper()}**")

                except Exception as e:
                    st.error(f" Lỗi: {e}")
                    clear_vram()

with col2:
    st.header("2. Ảnh đầu vào")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)