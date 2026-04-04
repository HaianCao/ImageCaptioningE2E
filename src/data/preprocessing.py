"""
Data Preprocessing Utilities.

Công dụng:
- Xử lý các file json gốc từ Visual Genome.
- Tính toán top-K classes phổ biến cho objects, attributes, relationships.
- Chuyển đổi nhãn (string) sang index (int) bằng Vocabulary.
- Lưu trữ file data đã qua xử lý (processed annotations) thành các split Train/Val/Test.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm


def build_vocab(items: List[str], max_size: int = 1000, unk_token: str = "<UNK>") -> Dict[str, int]:
    """
    Xây dựng vocabulary (Từ điển class -> index) từ danh sách các mục.
    Lọc bỏ các mục hiếm xuất hiện (chỉ giữ lại max_size mục phổ biến nhất).
    """
    counter = Counter(items)
    # Lấy top-k mục phổ biến nhất
    most_common = counter.most_common(max_size)
    
    vocab = {unk_token: 0}
    for idx, (word, freq) in enumerate(most_common, start=1):
        vocab[word] = idx
        
    return vocab


def preprocess_task1(
    raw_objects_path: str,
    raw_attributes_path: str,
    output_dir: str,
    max_objects: int = 150,
    max_attributes: int = 100
):
    """
    Tiền xử lý file cho Task 1 (Objects & Attributes).
    """
    print("--- [Task 1] Bắt đầu Preprocessing ---")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    print("Đang load objects file...")
    try:
        with open(raw_objects_path, "r", encoding="utf-8") as f:
            objects_data = json.load(f)
    except FileNotFoundError:
        print(f"[Error] Không tìm thấy {raw_objects_path}. Hãy tải dữ liệu trước.")
        return

    print("Đang load attributes file...")
    try:
        with open(raw_attributes_path, "r", encoding="utf-8") as f:
            attributes_data = json.load(f)
    except FileNotFoundError:
        print(f"[Error] Không tìm thấy {raw_attributes_path}. Hãy tải dữ liệu trước.")
        return
        
    # Map attributes by image_id
    attr_dict = {img['image_id']: img['attributes'] for img in attributes_data}
    
    # 2. Build Frequencies
    all_obj_names = []
    all_attr_names = []
    
    print("Đang đếm tần suất object & attributes...")
    valid_samples = []
    for img_obj in tqdm(objects_data, desc="Parsing"):
        img_id = img_obj["image_id"]
        attrs_in_img = attr_dict.get(img_id, [])
        attr_lookup = {a["object_id"]: a for a in attrs_in_img}
        
        for obj in img_obj["objects"]:
            name = str(obj.get("names", [""])[0]).lower().strip()
            if not name: continue
            
            all_obj_names.append(name)
            
            # Khớp attributes nếu có
            obj_attrs_raw = []
            if obj["object_id"] in attr_lookup:
                obj_attrs_raw = attr_lookup[obj["object_id"]].get("attributes", [])
                
            obj_attrs = [str(a).lower().strip() for a in obj_attrs_raw if a]
            all_attr_names.extend(obj_attrs)
            
            valid_samples.append({
                "image_id": img_id,
                "object_id": obj["object_id"],
                "bbox": [obj["x"], obj["y"], obj["w"], obj["h"]],
                "name": name,
                "attributes": obj_attrs
            })
            
    # 3. Build Vocabs
    obj_vocab = build_vocab(all_obj_names, max_size=max_objects)
    attr_vocab = build_vocab(all_attr_names, max_size=max_attributes)
    
    with open(out_path / "object_vocab.json", "w") as f:
        json.dump(obj_vocab, f, indent=2)
    with open(out_path / "attribute_vocab.json", "w") as f:
        json.dump(attr_vocab, f, indent=2)
    
    # 4. Map to Index
    print("Đang mapping dữ liệu sang integer indices...")
    processed_samples = []
    for s in valid_samples:
        obj_idx = obj_vocab.get(s["name"], obj_vocab["<UNK>"])
        if obj_idx == 0: continue # Bỏ qua Unknown Object để train tập trung
        
        attr_indices = []
        for x in s["attributes"]:
            if x in attr_vocab:
                attr_indices.append(attr_vocab[x])
                
        processed_samples.append({
            "image_id": s["image_id"],
            "object_id": s["object_id"],
            "bbox": s["bbox"],
            "object_label": obj_idx,
            "attribute_labels": attr_indices
        })
        
    print(f"Tổng hợp được {len(processed_samples)} samples hợp lệ.")
    
    # 5. Split & Save
    train_split = int(0.7 * len(processed_samples))
    val_split = int(0.85 * len(processed_samples))
    
    splits = {
        "train": processed_samples[:train_split],
        "val": processed_samples[train_split:val_split],
        "test": processed_samples[val_split:]
    }
    
    for split_name, data in splits.items():
        split_dir = out_path / split_name
        split_dir.mkdir(exist_ok=True)
        with open(split_dir / "annotations.json", "w") as f:
            json.dump({"samples": data}, f)
            
    print(f"--- Hoàn tất Task 1 Preprocessing ---")


def preprocess_task2(
    raw_relationships_path: str,
    output_dir: str,
    max_relations: int = 150
):
    """
    Tiền xử lý file cho Task 2 (Relationships).
    """
    print("--- [Task 2] Bắt đầu Preprocessing ---")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(raw_relationships_path, "r", encoding="utf-8") as f:
            rel_data = json.load(f)
    except FileNotFoundError:
        print(f"[Error] Không tìm thấy {raw_relationships_path}. Hãy tải dữ liệu trước.")
        return
        
    all_predicates = []
    valid_samples = []
    
    for img_rel in tqdm(rel_data, desc="Parsing Relationships"):
        img_id = img_rel["image_id"]
        
        for rel in img_rel["relationships"]:
            pred = str(rel.get("predicate", "")).lower().strip()
            if not pred: continue
            
            all_predicates.append(pred)
            
            subj = rel["subject"]
            obj = rel["object"]
            
            valid_samples.append({
                "image_id": img_id,
                "relationship_id": rel["relationship_id"],
                "subject_bbox": [subj["x"], subj["y"], subj["w"], subj["h"]],
                "object_bbox": [obj["x"], obj["y"], obj["w"], obj["h"]],
                "subject_name": subj.get("name", ""),
                "object_name": obj.get("name", ""),
                "predicate": pred
            })
            
    rel_vocab = build_vocab(all_predicates, max_size=max_relations)
    
    with open(out_path / "relation_vocab.json", "w") as f:
        json.dump(rel_vocab, f, indent=2)
        
    print("Đang mapping dữ liệu sang integer indices...")
    processed_samples = []
    for s in valid_samples:
        rel_idx = rel_vocab.get(s["predicate"], rel_vocab["<UNK>"])
        if rel_idx == 0: continue # Bỏ qua relation lạ
        
        processed_samples.append({
            "image_id": s["image_id"],
            "relationship_id": s["relationship_id"],
            "subject_bbox": s["subject_bbox"],
            "object_bbox": s["object_bbox"],
            "subject_name": s["subject_name"],
            "object_name": s["object_name"],
            "relation_label": rel_idx
        })
        
    train_split = int(0.7 * len(processed_samples))
    val_split = int(0.85 * len(processed_samples))
    
    splits = {
        "train": processed_samples[:train_split],
        "val": processed_samples[train_split:val_split],
        "test": processed_samples[val_split:]
    }
    
    for split_name, data in splits.items():
        split_dir = out_path / split_name
        split_dir.mkdir(exist_ok=True)
        with open(split_dir / "annotations.json", "w") as f:
            json.dump({"samples": data}, f)
            
    print(f"--- Hoàn tất Task 2 Preprocessing ---")


def build_vocab_and_splits(
    task: str = "task1", 
    raw_dir: str = "data/raw", 
    processed_dir: str = "data/processed"
):
    """
    Entrypoint để gọi từ notebook.
    """
    r_dir = Path(raw_dir)
    p_dir = Path(processed_dir) / task
    
    if task == "task1":
        preprocess_task1(
            str(r_dir / "objects.json"),
            str(r_dir / "attributes.json"),
            str(p_dir)
        )
    elif task == "task2":
        preprocess_task2(
            str(r_dir / "relationships.json"),
            str(p_dir)
        )
    else:
        print(f"Unknown task: {task}")