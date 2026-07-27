# Hệ Thống Hỏi Đáp Trên Ảnh (VQA) Tiếng Việt - PaliGemma Fine-tuned

## 📖 Giới thiệu
Dự án này xây dựng một hệ thống Visual Question Answering (VQA) hỗ trợ giải đáp các câu hỏi tiếng Việt dựa trên nội dung hình ảnh. Hệ thống sử dụng mô hình nền tảng **PaliGemma** (`google/paligemma-3b-pt-224`). 

Để tối ưu hóa quá trình huấn luyện trên các thiết bị có tài nguyên giới hạn, dự án áp dụng kỹ thuật **LoRA** (Low-Rank Adaptation) kết hợp lượng tử hóa 4-bit (**BitsAndBytes**).

## 🚀 Tính năng nổi bật
- **Fine-tuning hiệu năng cao:** Ứng dụng PEFT/LoRA và QLoRA (4-bit) để huấn luyện mô hình 3 tỷ tham số mà không yêu cầu cấu hình phần cứng quá "khủng". Chỉnh sửa trọng số tập trung vào `q_proj` và `v_proj`.
- **Đánh giá đa chiều:** Script đánh giá tự động tính toán các chỉ số quan trọng: VQA Accuracy, BLEU-4, ROUGE-L, và BERTScore.
- **Giao diện thân thiện:** Cung cấp một Web App được xây dựng bằng Streamlit cho phép tải ảnh, nhập câu hỏi và theo dõi kết quả trực quan. Quản lý VRAM tự động để tránh tràn bộ nhớ GPU.

## 📁 Cấu trúc các file chính
- `app.py`: Giao diện ứng dụng người dùng chạy bằng Streamlit.
- `B2_model.py`: Script dùng để Load, Flatten Dataset, định nghĩa Data Collator và tiến hành Fine-tune mô hình.
- `Evalute_B2.py`: Tập lệnh chạy mô hình trên tập Test để xuất các chỉ số độ đo (Accuracy, BLEU, ROUGE, BERTScore).
- `Inference/predict_B2.py`: Cung cấp hàm `run_predict_B2` chịu trách nhiệm Load LoRA adapter để suy luận (inference) câu trả lời từ mô hình.

## 🛠️ Cài đặt & Yêu cầu hệ thống
Bạn cần có môi trường Python và GPU hỗ trợ CUDA. Cài đặt các thư viện phụ thuộc bằng lệnh:
```bash
pip install torch torchvision
pip install transformers peft datasets bitsandbytes accelerate
pip install streamlit pillow
pip install nltk rouge_score bert_score
```

## 🏃 Hướng dẫn sử dụng

### 1. Huấn luyện mô hình (Training)
Trước khi chạy, hãy đảm bảo bạn đã cấu hình đúng đường dẫn thư mục lưu ảnh và file `.json` chứa dataset (Train/Val) trong file huấn luyện.
```bash
python B2_model.py
```
*Mô hình và Processor sau khi fine-tune sẽ được lưu vào thư mục `B2_finetuned_model/`.*

### 2. Đánh giá mô hình (Evaluation)
Để kiểm tra độ chính xác của mô hình trên tập Test:
```bash
python Evalute_B2.py
```

### 3. Khởi chạy Ứng dụng (Web App)
Để mở giao diện người dùng, sử dụng lệnh:
```bash
streamlit run app.py
```
Sau đó truy cập vào địa chỉ Local URL (thường là `http://localhost:8501`) hiện trên Terminal. Bạn có thể tải một tấm ảnh bất kỳ, gõ câu hỏi bằng tiếng Việt và nhấn **Chạy dự đoán**.
