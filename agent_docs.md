# Agent Documentation & State Tracking (Visual Genome Captioning)

*Tài liệu này dùng để theo dõi tiến độ, cấu trúc dự án và các quyết định kỹ thuật hiện hành.*

---

## 1. Trạng Thái Dự Án (Project Status)

**Mục tiêu cốt lõi**: Trích xuất Đối tượng, Thuộc tính (Task 1) + Mối quan hệ (Task 2) → Sinh caption demo.
**Môi trường ưu tiên**: Kaggle / Google Colab, với [notebooks/complete_pipeline.ipynb](notebooks/complete_pipeline.ipynb) là entry point chính.

✅ **Đã hoàn thành**:
- Thiết kế kiến trúc tổng quan.
- Tạo cấu trúc thư mục (`configs/`, `data/`, `src/`, `notebooks/`).
- Tạo các file config YAML (`config.yaml`, `task1_config.yaml`, `task2_config.yaml`).
- Chuẩn hóa comment cho các tham số enum và tham số chức năng trong YAML để dễ chỉnh sửa.
- Viết module tải dữ liệu gốc từ VG/Stanford dạng ZIP (`src/data/download.py`).
- Khởi tạo Base Dataset, Transforms (`src/data/dataset.py`, `src/data/transforms.py`).
- Khởi tạo Dataset logic cho Task 1 và Task 2 (`src/data/task1_dataset.py`, `src/data/task2_dataset.py`).
- Hoàn thiện preprocessing raw → processed cho Task 1/2, gồm alias file gốc, vocab top-K, split train/val/test, strict sample mode và `image_info` cho Task 2 (`src/data/preprocessing.py`).
- Viết Base visual encoder và extractors (`src/features/visual_encoder.py`, `src/features/roi_extractor.py`, `src/features/feature_extractor.py`).
- Hoàn thiện module metric theo hướng accuracy + F1 và gắn vào pipeline train/eval (`src/evaluation/metrics.py`).
- Nhúng & test thành công các thư viện chính, không còn lỗi import cốt lõi.
- Cấu trúc models cơ bản (`src/models/base_model.py`, `src/models/task1/*`, `src/models/task2/*`, `src/models/caption/*`).
- Base logic trainer (`src/training/trainer.py`, `src/training/task1_trainer.py`, `src/training/task2_trainer.py`) đã được sửa để train thật, checkpoint đúng và load best checkpoint đúng path.
- Sửa feature cache để khớp key, xử lý cache rỗng an toàn và đồng bộ cache path với config.
- Cập nhật `CaptionGenerator` để nhận đúng output thực tế từ trainer.
- Biến [notebooks/complete_pipeline.ipynb](notebooks/complete_pipeline.ipynb) thành full pipeline config-driven: load YAML, fail-fast khi thiếu dữ liệu, strict sample mode, preprocess, extract feature, train, evaluate test, caption demo.
- Thêm [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) làm notebook khám phá dữ liệu riêng.
- Cập nhật [README.md](README.md) để phản ánh workflow hiện tại và bỏ các hướng dẫn hardcode cũ.

❌ **Cần tinh chỉnh / Đang thực hiện**:
- Chạy lại pipeline end-to-end trên full dataset để xác minh ổn định ngoài chế độ debug (`MAX_SAMPLES`).
- Đồng bộ scheduler config vào notebook/trainer nếu muốn `configs/config.yaml.scheduler` được dùng thật.
- Nếu cần cho báo cáo cuối, bổ sung thêm bảng kết quả / metric phụ sau khi full-run xong.
- Tối ưu tốc độ pre-extract feature và checkpoint cleanup nếu chạy trên Colab/Kaggle với bộ dữ liệu lớn.

---

## 2. Các Hàm & Module Đã Có (Inventory)

### 2.1. Module `src/data`
- `download.py`:
  - `download_file(url, dest)`: Tải file dung lượng lớn.
  - `unzip_file(...)`: Giải nén.
  - `download_and_extract_metadata(...)`: Tải metadata Stanford ZIP → JSON nguyên bản.
  - `download_vg_images(...)`: Tải ảnh từ Stanford VG image hosts.
- `dataset.py`:
  - `BaseVGDataset`: Class trừu tượng chứa logic ROI cropping chung.
- `transforms.py`:
  - `get_train_transforms`, `get_val_transforms`: Transform chuẩn.
- `task1_dataset.py`:
  - `ObjectAttributeDataset`: Dataset nhận input bbox → ảnh crop ROI → nhãn object và attributes.
- `task2_dataset.py`:
  - `RelationshipDataset`: Dataset nhận cặp bbox → union_bbox ảnh crop → nhãn relation.

### 2.2. Module `src/features`
- `roi_extractor.py`:
  - `extract_roi`, `extract_union_roi`, `compute_spatial_features`.
- `visual_encoder.py`:
  - `VisualEncoder`: Build từ backbone ResNet/ViT.
- `feature_extractor.py`:
  - Extract feature batching để tạo cache `.pt`.

### 2.3. Module `src/models`
- `task1/object_classifier.py` & `attribute_classifier.py`:
  - `ObjectClassifier`: single-label classification head, dùng cho 1 nhãn object chính mỗi sample.
  - `AttributeClassifier`: multi-label classification head, dùng sigmoid/BCE và số nhãn active có thể thay đổi theo từng object.
- `task2/relation_classifier.py`: MLP dự đoán quan hệ.
- `caption/caption_generator.py`: Template-based caption generator, đã được nối lại để đọc output thực tế từ Task 1/2.
- `notebooks/01_eda.ipynb`: Notebook khám phá dữ liệu, thống kê phân phối nhãn và xem mẫu ảnh.

### 2.4. Module `src/evaluation`
- `metrics.py`:
  - Classification metrics cho Task 1/2: accuracy + F1.
  - Multi-label Task 1 attributes có thêm exact-match accuracy và sample-wise accuracy.
  - Caption metrics vẫn còn trong module nhưng không phải trọng tâm hiện tại.

---

## 3. Kế hoạch sửa đổi (Next Steps)

1. **Xác minh full run**
  - Chạy lại notebook end-to-end để refresh outputs và kiểm tra ổn định trên data thật.

2. **Hoàn thiện wiring còn treo**
  - Nếu cần, nối `scheduler` config vào flow train thực tế.
  - Tinh chỉnh logging / metrics phụ cho báo cáo cuối.

---
*Cập nhật lần cuối: README và agent docs đã được đồng bộ với pipeline config-driven hiện tại*
