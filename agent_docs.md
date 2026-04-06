# Agent Documentation & State Tracking (Visual Genome Captioning)

*Tài liệu này dùng để theo dõi tiến độ, cấu trúc dự án và các quyết định kỹ thuật hiện hành.*

---

## 1. Trạng Thái Dự Án (Project Status)

**Mục tiêu cốt lõi**: tách Object, Attribute, Relation thành ba pipeline riêng, sau đó ghép checkpoint qua E2E wrapper để demo caption.
**Môi trường ưu tiên**: notebook-driven workflow trong VS Code / Colab, nhưng các entry point chính là các notebook pipeline riêng.

✅ **Đã xác nhận**:
- `get_errors` trên workspace hiện trả về no errors.
- [notebooks/object_pipeline.ipynb](notebooks/object_pipeline.ipynb), [notebooks/attribute_pipeline.ipynb](notebooks/attribute_pipeline.ipynb), [notebooks/relation_pipeline.ipynb](notebooks/relation_pipeline.ipynb), và [notebooks/e2e_wrapper.ipynb](notebooks/e2e_wrapper.ipynb) là các entry point public hiện tại.
- [notebooks/complete_pipeline.ipynb](notebooks/complete_pipeline.ipynb) vẫn tồn tại như workflow legacy và đã được làm sạch output.
- [README.md](README.md) đã bỏ tham chiếu tới notebook không tồn tại như `01_eda.ipynb`.
- Object và attribute dùng config/model/trainer riêng; shared preprocessing vẫn giữ `task1`/`task2` chỉ ở tầng compatibility.
- Config public hiện tại là:
  - `configs/config.yaml`
  - `configs/object_config.yaml`
  - `configs/attribute_config.yaml`
  - `configs/relation_config.yaml`

❗ **Cần giữ nhất quán**:
- Object và Attribute là hai bài toán độc lập, không trộn chung config/model/trainer.
- `task1`/`task2` chỉ dùng trong shared compatibility layer và legacy notebook.
- Không thêm lại `01_eda.ipynb`, `task1_train.ipynb`, `task2_train.ipynb` vào docs vì các file đó không còn trong workspace.

❌ **Còn lại / theo dõi**:
- Chạy full dataset end-to-end để xác minh ổn định ngoài chế độ debug.
- Nếu muốn dọn sâu hơn, đánh giá việc giữ hay bỏ các wrapper compatibility; hiện tại nên giữ để không phá import cũ.

---

## 2. Các Hàm & Module Đã Có (Inventory)

### 2.1. Notebook entry points
- `notebooks/object_pipeline.ipynb`
- `notebooks/attribute_pipeline.ipynb`
- `notebooks/relation_pipeline.ipynb`
- `notebooks/e2e_wrapper.ipynb`
- `notebooks/complete_pipeline.ipynb`

### 2.2. Config
- `configs/config.yaml`: global settings, runtime controls, sampling, logging.
- `configs/object_config.yaml`: object-specific backbone, model, augmentation, eval.
- `configs/attribute_config.yaml`: attribute-specific backbone, model, augmentation, eval.
- `configs/relation_config.yaml`: relation-specific backbone, spatial, fusion, eval.

### 2.3. Primary APIs
- Trainers:
  - `src/training/object_trainer.py` -> `ObjectTrainer`
  - `src/training/attribute_trainer.py` -> `AttributeTrainer`
  - `src/training/relation_trainer.py` -> `RelationTrainer`
- Model namespaces:
  - `src/models/object/`
  - `src/models/attribute/`
  - `src/models/relation/`
  - `src/models/e2e/`
- Dataset wrappers:
  - `src/data/object_dataset.py`
  - `src/data/attribute_dataset.py`
  - `src/data/relation_dataset.py`
- Feature wrappers:
  - `src/features/object_feature_extractor.py`
  - `src/features/attribute_feature_extractor.py`
  - `src/features/relation_feature_extractor.py`

### 2.4. Compatibility layer
- `src/data/task1_dataset.py`
- `src/data/task2_dataset.py`
- `src/features/feature_extractor.py`
- `src/models/task1/`
- `src/models/task2/`

---

## 3. Ghi chú kỹ thuật
- `VisualGenomeE2EModel` chỉ load ba checkpoint object/attribute/relation, không phải joint multi-task model.
- Feature cache nằm dưới từng processed dir trong thư mục `features/`.
- `pipeline.training_mode` chấp nhận `object`, `attribute`, `relation`, `all`, `e2e`; alias cũ `task1`, `task2`, `both` vẫn còn cho tương thích.
- Nếu thêm import hoặc notebook mới, đừng dùng tên legacy làm public API mới.
- Tài liệu user-facing phải khớp với naming `object`/`attribute`/`relation`, còn `task1`/`task2` chỉ xuất hiện khi cần backwards compatibility.

---
*Cập nhật lần cuối: README và agent docs đã được đồng bộ với pipeline split object/attribute/relation hiện tại*
