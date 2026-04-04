# Agent Documentation & State Tracking (Visual Genome Captioning)

*Tài liệu này được dùng để AI Agent (Antigravity) và người dùng có thể cùng theo dõi tiến độ, cấu trúc dự án và lên kế hoạch sửa đổi dần.+*

---

## 1. Trạng Thái Dự Án (Project Status)

**Mục tiêu cốt lõi**: Trích xuất Đối tượng, Thuộc tính (Task 1) + Mối quan hệ (Task 2) → Sinh câu mô tả (Captioning).
**Môi trường ưu tiên**: Kaggle / Google Colab (yêu cầu chạy mượt mà trên 1 notebook duy nhất `complete_pipeline.ipynb`).

✅ **Đã hoàn thành**:
- Thiết kế kiến trúc tổng quan.
- Tạo cấu trúc thư mục (`configs/`, `data/`, `src/`, `notebooks/`).
- Tạo các file config YAML (`config.yaml`, `task1_config.yaml`, `task2_config.yaml`).
- Viết module tải dữ liệu gốc từ VG/Stanford dạng ZIP (`src/data/download.py`).
- Khởi tạo Base Dataset, Transforms (`src/data/dataset.py`, `src/data/transforms.py`).
- Khởi tạo Dataset logic cho Task 1 và Task 2 (`src/data/task1_dataset.py`, `src/data/task2_dataset.py`).
- Viết Base visual encoder và extractors (`src/features/visual_encoder.py`, `src/features/roi_extractor.py`, `src/features/feature_extractor.py`).
- Nhúng & test thành công tất cả thư viện (không có lỗi file import).
- Cấu trúc Models cơ bản (`models/base_model.py`, `models/task1/*`, `models/task2/*`, `models/caption/*`).
- Base logic Trainer (`src/training/trainer.py`, `task1_trainer.py`, `task2_trainer.py`).
- Đã tạo Notebook lõi `notebooks/complete_pipeline.ipynb` đúng yêu cầu.
- (Người dùng đã) Cập nhật lại README chuẩn hóa cho Colab Workflow.

❌ **Cần tinh chỉnh / Đang thực hiện**:
- Khâu **Preprocessing Data**: Rất quan trọng, hiện tại trống. Cần chuyển đổi dữ liệu file JSON gốc của VG sang file JSON cấu trúc ngắn gọn (`data/processed`) + Generate Dictionary (Vocab).
- Rà soát mô hình `CaptionGenerator` (khi cần).

---

## 2. Các Hàm & Module Đã Có (Inventory)

### 2.1. Module `src/data`
- `download.py`:
  - `download_file(url, dest)`: Tải file dung lượng lớn.
  - `unzip_file(...)`: Giải nén.
  - `download_and_extract_metadata(...)`: Auto tải link Stanford Zip -> JSON nguyên bản.
  - `download_vg_images(...)`: Tải ảnh từ Standford URL trực tiếp.
- `dataset.py`:
  - `BaseVGDataset`: Class trừu tượng chứa logic base xử lý ROI cropping chung.
- `transforms.py`:
  - `get_train_transforms`, `get_val_transforms`: Transform chuẩn.
- `task1_dataset.py`:
  - `ObjectAttributeDataset`: Dataset nhận input bbox -> ảnh crop ROI -> nhãn object và attributes.
- `task2_dataset.py`:
  - `RelationshipDataset`: Dataset nhận cặp bbox -> union_bbox ảnh crop -> nhãn relation.

### 2.2. Module `src/features`
- `roi_extractor.py`: 
  - `extract_roi`, `extract_union_roi`, `compute_spatial_features`. 
- `visual_encoder.py`:
  - `VisualEncoder`: Build từ backbone Resnet/ViT.
- `feature_extractor.py`:
  - Extract feature batching (hữu ích để tạo cache `.pt` file).

### 2.3. Module `src/models`
- `task1/object_classifier.py` & `attribute_classifier.py`: Linear head cho classification.
- `task2/relation_classifier.py`: MLPs dự đoán quan hệ.
- `caption/caption_generator.py`: Framework sinh caption (gắn LLM/T5 - *cần rà soát logic nối text ngầm*).

---

## 3. Kế hoạch sửa đổi (Next Steps)

1. **Viết Module Data Preprocessing (`src/data/preprocessing.py`)**:
   - Nhận Input là file thô: `objects_v1_2.json`, `attributes.json`, `relationships_v1_2.json`.
   - Tính toán tần suất class: Lọc bỏ các thuộc tính, đối tượng, hay relationships quá hiếm (thresh-hold frequency).
   - Xây dựng file Vocabulary: `object_vocab.json`, `attribute_vocab.json`, `relation_vocab.json`.
   - Tạo file xuất data `data/processed/task1/train/annotations.json` và tương tự cho task2 với index thay cho text string.

2. **Check và Fix Modules Logic**:
   - Chạy mô phỏng (dry-run) `task1_trainer` và `task2_trainer` trên một few-shot data.
   - Hoàn thiện luồng cuối cho Captioning Module.

---
*Cập nhật lần cuối: Xác minh kiến trúc module & chuẩn bị bắt đầu viết Preprocessing*
