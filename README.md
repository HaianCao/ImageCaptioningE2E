## 🚀 Quick Start (Colab/Kaggle)

### 1. Clone Repository
```bash
!git clone https://github.com/your-username/visual-genome-caption.git
%cd visual-genome-caption
```

### 2. Run Complete Pipeline Notebook
```python
# Open and run: notebooks/complete_pipeline.ipynb
# This notebook contains EVERYTHING:
# ✅ Install all packages
# ✅ Download data from Stanford (~1.5GB)
# ✅ Preprocess data
# ✅ Extract features
# ✅ Train Task 1, Task 2, or both
# ✅ Evaluate models
# ✅ Run inference demo
```

### 3. Configure Training Mode
At the top of `complete_pipeline.ipynb`, set your training preferences:

```python
# Choose what to train
TRAINING_MODE = 'both'  # 'task1', 'task2', or 'both'

# Data settings
DOWNLOAD_DATA = True    # Download data automatically
IMAGE_DOWNLOAD_MODE = 'sample'  # 'none', 'sample', or 'all'

# Training parameters
MAX_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
```

### 4. Run the Notebook
- **First run**: Set `DOWNLOAD_DATA = True` to download and setup everything
- **Subsequent runs**: Set `DOWNLOAD_DATA = False` to skip download
- **Training only**: Change `TRAINING_MODE` to train specific tasks

## 📋 Manual Setup (Alternative)

If you prefer manual setup:

```bash
# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other packages
pip install transformers timm datasets tqdm omegaconf scikit-learn wandb
pip install opencv-python pillow matplotlib seaborn plotly
```

## � Data Sources

**Visual Genome Dataset** is downloaded directly from Stanford University servers:

- **Objects**: https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects_v1_2.json.zip (413MB)
- **Attributes**: https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/attributes.json.zip (462MB)  
- **Relationships**: https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationships_v1_2.json.zip (709MB)
- **Images**: https://cs.stanford.edu/people/rak248/VG_100K/ (on-demand, ~20GB total)

*Note: HuggingFace dataset is no longer supported due to deprecated loading scripts.*

## 🗂️ Project Structure

```
prj/
├── configs/                    # YAML configurations
│   ├── config.yaml             # Main config
│   ├── task1_config.yaml       # Task 1 config
│   └── task2_config.yaml       # Task 2 config
│
├── data/                       # Data (auto-downloaded)
│   ├── raw/                    # Raw data from Stanford VG
│   └── processed/              # Preprocessed data
│
├── src/                        # Source code modules
│   ├── data/                   # Data processing
│   ├── features/               # Feature extraction
│   ├── models/                 # Model definitions
│   ├── training/               # Training infrastructure
│   ├── evaluation/             # Metrics & evaluation
│   └── utils/                  # Utilities
│
└── notebooks/                  # Jupyter notebooks (entry points)
    ├── complete_pipeline.ipynb         # ⭐ MAIN NOTEBOOK - Everything in one place
    ├── 00_setup_and_download.ipynb    # Individual setup (legacy)
    ├── 01_data_exploration.ipynb      # Data exploration
    ├── 02_feature_extraction.ipynb    # Feature caching
    ├── task1_train.ipynb              # Task 1 training only
    └── task2_train.ipynb              # Task 2 training only
```
    ├── 02_data_processing.ipynb
    ├── 03_feature_extraction.ipynb
    ├── task1_train.ipynb
    ├── task1_eval.ipynb
    ├── task2_train.ipynb
    ├── task2_eval.ipynb
    └── caption_generation.ipynb
```

## 💡 Usage Examples

### Complete Pipeline (Recommended)
Use `notebooks/complete_pipeline.ipynb` for everything in one notebook:

#### First Time Setup + Training
```python
TRAINING_MODE = 'both'          # Train both tasks
DOWNLOAD_DATA = True            # Download data
IMAGE_DOWNLOAD_MODE = 'sample'  # Download sample images
MAX_EPOCHS = 10
BATCH_SIZE = 32
```

#### Quick Training (Data Already Downloaded)
```python
TRAINING_MODE = 'task1'         # Train only Task 1
DOWNLOAD_DATA = False           # Skip download
IMAGE_DOWNLOAD_MODE = 'none'    # No images needed
MAX_EPOCHS = 5
BATCH_SIZE = 64
```

#### Task 2 Only
```python
TRAINING_MODE = 'task2'         # Train only Task 2
DOWNLOAD_DATA = False
MAX_EPOCHS = 15
LEARNING_RATE = 2e-4
```

### Individual Notebooks (Advanced)
For specific tasks, use individual notebooks:
- `task1_train.ipynb` - Object & attribute classification only
- `task2_train.ipynb` - Relationship classification only
- `00_setup_and_download.ipynb` - Setup only

## Pipeline Thực nghiệm

### Local (Development)

```bash
git clone https://github.com/<your-username>/visual-genome-caption.git
cd visual-genome-caption
pip install -r requirements.txt
```

### Kaggle / Colab (Training)

```bash
# Clone repo
!git clone https://github.com/<your-username>/visual-genome-caption.git
%cd visual-genome-caption
!pip install -r requirements.txt

# Sau đó mở và chạy notebooks tương ứng
```

## Backbone Models Được Hỗ Trợ

| Backbone | Config key | Ghi chú |
|---|---|---|
| ResNet-50 | `resnet50` | Nhanh, nhẹ |
| ResNet-101 | `resnet101` | Cân bằng |
| EfficientNet-B3 | `efficientnet_b3` | Hiệu quả tham số |
| EfficientNet-B5 | `efficientnet_b5` | Tốt hơn, nặng hơn |
| ViT-Base/16 | `vit_base_16` | Attention-based |
| ViT-Large/16 | `vit_large_16` | Mạnh hơn, cần nhiều RAM |

## Caption Models Được Hỗ Trợ

| Model | Config key |
|---|---|
| GPT-2 | `gpt2` |
| GPT-2 Medium | `gpt2-medium` |
| T5-Small | `t5-small` |
| T5-Base | `t5-base` |

## Dataset

- **Visual Genome**: [HuggingFace](https://huggingface.co/datasets/ranjaykrishna/visual_genome)
- Objects & Attributes: ~108k ảnh, ~3.8M objects, ~2.8M attributes
- Relationships: ~2.3M relationships

## Requirements

Xem `requirements.txt` để biết đầy đủ dependencies.
