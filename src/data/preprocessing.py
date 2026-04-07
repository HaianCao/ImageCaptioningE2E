"""
Data Preprocessing Utilities.

Công dụng:
- Xử lý các file json gốc từ Visual Genome.
- Tính toán top-K classes phổ biến cho objects, attributes, relationships.
- Chuyển đổi nhãn (string) sang index (int) bằng Vocabulary.
- Lưu trữ file data đã qua xử lý (processed annotations) thành các split Train/Val/Test.
"""

import json
import math
import random
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional

from sklearn.model_selection import train_test_split
from tqdm import tqdm


RAW_FILE_ALIASES = {
    "objects.json": ["objects_v1_2.json"],
    "relationships.json": ["relationships_v1_2.json"],
    "attributes.json": [],
    "image_data.json": [],
}

ALIAS_FILE_PATH = Path(__file__).resolve().parents[2] / "data" / "aliases" / "visual_genome_aliases.txt"


def _normalize_alias_phrase(text: str) -> str:
    """Normalize alias keys without guessing morphology."""
    cleaned = str(text).lower().strip().replace("-", " ")
    if not cleaned:
        return cleaned
    return " ".join(cleaned.split())


@lru_cache(maxsize=1)
def _load_alias_maps() -> Dict[str, Dict[str, str]]:
    """Load manually reviewed alias maps for objects and relations."""
    if not ALIAS_FILE_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file alias: {ALIAS_FILE_PATH}")

    alias_maps: Dict[str, Dict[str, str]] = {}
    current_section = None

    with open(ALIAS_FILE_PATH, "r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            section_key = line.lower()
            if section_key in {"[objects]", "objects"}:
                current_section = "objects"
                alias_maps.setdefault(current_section, {})
                continue
            if section_key in {"[relations]", "relations"}:
                current_section = "relations"
                alias_maps.setdefault(current_section, {})
                continue

            if current_section is None:
                raise ValueError(
                    f"Alias file phải khai báo [objects] hoặc [relations] trước dữ liệu, lỗi ở dòng {line_number}."
                )

            parts = [_normalize_alias_phrase(part) for part in line.split(",")]
            parts = [part for part in parts if part]
            if len(parts) < 2:
                raise ValueError(
                    f"Alias line phải có ít nhất 1 canonical và 1 alias, lỗi ở dòng {line_number}: {raw_line.rstrip()}"
                )

            canonical_value = parts[0]
            section_map = alias_maps.setdefault(current_section, {})
            for alias_value in parts[1:]:
                existing = section_map.get(alias_value)
                if existing is not None and existing != canonical_value:
                    raise ValueError(
                        f"Alias '{alias_value}' bị map hai lần: '{existing}' và '{canonical_value}' (dòng {line_number})."
                    )
                section_map[alias_value] = canonical_value

    if "objects" not in alias_maps or "relations" not in alias_maps:
        raise ValueError("Alias file phải chứa cả [objects] và [relations].")

    return alias_maps


def _apply_alias(text: str, section: str) -> str:
    """Apply a manual alias map until it reaches a stable canonical form."""
    alias_maps = _load_alias_maps()
    if section not in alias_maps:
        raise KeyError(f"Unknown alias section: {section}")

    current = _normalize_alias_phrase(text)
    seen = set()
    while current in alias_maps[section] and current not in seen:
        seen.add(current)
        current = alias_maps[section][current]
    return current


def _normalize_object_name(text: str) -> str:
    return _apply_alias(text, "objects")


def _normalize_relation_name(text: str) -> str:
    return _apply_alias(text, "relations")


def _select_balanced_image_ids(
    image_to_labels: Dict[int, List[str]],
    sample_size: int,
    seed: int,
    max_labels_per_image: int = 5,
) -> List[int]:
    """Select image ids with a bias toward images containing rare labels."""
    unique_image_ids = list(image_to_labels.keys())
    if sample_size >= len(unique_image_ids):
        return unique_image_ids

    label_frequency = Counter()
    for labels in image_to_labels.values():
        label_frequency.update(set(labels))

    rng = random.Random(seed)
    scored_images = []
    for image_id, labels in image_to_labels.items():
        unique_labels = list(dict.fromkeys(labels))
        rarity_scores = sorted(
            (
                1.0 / label_frequency[label]
                for label in unique_labels
                if label in label_frequency and label_frequency[label] > 0
            ),
            reverse=True,
        )
        if not rarity_scores:
            continue

        top_scores = rarity_scores[:max_labels_per_image]
        image_weight = sum(top_scores) / len(top_scores)
        key = math.log(rng.random()) / max(image_weight, 1e-12)
        scored_images.append((key, image_id))

    scored_images.sort(reverse=True)
    return [image_id for _, image_id in scored_images[:sample_size]]


def _select_sample_image_ids(
    image_to_labels: Dict[int, List[str]],
    sample_size: Optional[int],
    sample_strategy: str,
    seed: int,
) -> Optional[List[int]]:
    if sample_size is None:
        return None

    unique_image_ids = list(image_to_labels.keys())
    if sample_size <= 0:
        return []
    if sample_size >= len(unique_image_ids):
        return unique_image_ids

    strategy = str(sample_strategy).lower().strip()
    if strategy == "balanced":
        return _select_balanced_image_ids(image_to_labels, sample_size=sample_size, seed=seed)

    return random.Random(seed).sample(unique_image_ids, sample_size)


def _resolve_raw_file(raw_file_path: str) -> Path:
    """Resolve canonical VG raw filenames and known aliases."""
    raw_path = Path(raw_file_path)
    if raw_path.exists():
        return raw_path

    candidates = [raw_path.name, *RAW_FILE_ALIASES.get(raw_path.name, [])]
    for candidate_name in candidates:
        candidate_path = raw_path.parent / candidate_name
        if candidate_path.exists():
            return candidate_path

    raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {raw_file_path}")


def _split_image_ids(
    image_ids: List[int],
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Split image ids into train/val/test buckets using the requested ratios."""
    unique_ids = list(dict.fromkeys(image_ids))
    if not unique_ids:
        return [], [], []

    total_ratio = sum(split_ratios)
    if total_ratio <= 0:
        raise ValueError("split_ratios phải có tổng dương.")

    val_ratio = split_ratios[1] / total_ratio
    test_ratio = split_ratios[2] / total_ratio

    if len(unique_ids) == 1:
        return unique_ids, [], []
    if len(unique_ids) == 2:
        return unique_ids[:1], unique_ids[1:], []
    if val_ratio + test_ratio <= 0:
        return unique_ids, [], []

    train_ids, temp_ids = train_test_split(
        unique_ids,
        test_size=val_ratio + test_ratio,
        random_state=seed,
        shuffle=True,
    )

    temp_ids = list(temp_ids)
    if len(temp_ids) < 2:
        if val_ratio >= test_ratio:
            return list(train_ids), temp_ids, []
        return list(train_ids), [], temp_ids

    val_share = val_ratio / (val_ratio + test_ratio)
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=1.0 - val_share,
        random_state=seed,
        shuffle=True,
    )

    return list(train_ids), list(val_ids), list(test_ids)


def build_vocab(
    items: List[str],
    max_size: int = 1000,
    unk_token: str = "<UNK>",
    normalizer: Optional[Callable[[str], str]] = None,
) -> Dict[str, int]:
    """
    Xây dựng vocabulary (Từ điển class -> index) từ danh sách các mục.
    Lọc bỏ các mục hiếm xuất hiện (chỉ giữ lại max_size mục phổ biến nhất).
    """
    normalized_items = [normalizer(item) if normalizer is not None else item for item in items]
    counter = Counter(normalized_items)
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
    max_attributes: int = 100,
    sample_image_ids: Optional[List[int]] = None,
    sample_size: Optional[int] = None,
    sample_strategy: str = "random",
    sample_focus: str = "combined",
    split_by_image_id: bool = False,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
):
    """
    Tiền xử lý file cho Task 1 (Objects & Attributes).
    """
    print("--- [Task 1] Bắt đầu Preprocessing ---")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    print("Đang load objects file...")
    raw_objects_path = _resolve_raw_file(raw_objects_path)
    try:
        with open(raw_objects_path, "r", encoding="utf-8") as f:
            objects_data = json.load(f)
    except FileNotFoundError:
        print(f"[Error] Không tìm thấy {raw_objects_path}. Hãy tải dữ liệu trước.")
        return

    print("Đang load attributes file...")
    raw_attributes_path = _resolve_raw_file(raw_attributes_path)
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
    object_image_to_labels = defaultdict(set)
    attribute_image_to_labels = defaultdict(set)
    
    print("Đang đếm tần suất object & attributes...")
    valid_samples = []
    for img_obj in tqdm(objects_data, desc="Parsing"):
        img_id = img_obj["image_id"]
        attrs_in_img = attr_dict.get(img_id, [])
        attr_lookup = {a["object_id"]: a for a in attrs_in_img}
        
        for obj in img_obj["objects"]:
            name = _normalize_object_name(str(obj.get("names", [""])[0]).lower().strip())
            if not name: continue
            
            all_obj_names.append(name)
            
            # Khớp attributes nếu có
            obj_attrs_raw = []
            if obj["object_id"] in attr_lookup:
                obj_attrs_raw = attr_lookup[obj["object_id"]].get("attributes", [])
                
            obj_attrs = [str(a).lower().strip() for a in obj_attrs_raw if a]
            all_attr_names.extend(obj_attrs)

            object_image_to_labels[img_id].add(f"object::{name}")
            for attr_name in obj_attrs:
                attribute_image_to_labels[img_id].add(f"attribute::{attr_name}")
            
            valid_samples.append({
                "image_id": img_id,
                "object_id": obj["object_id"],
                "bbox": [obj["x"], obj["y"], obj["w"], obj["h"]],
                "name": name,
                "attributes": obj_attrs
            })

    if sample_image_ids is None and split_by_image_id and sample_size is not None:
        focus_key = str(sample_focus).lower().strip()
        if focus_key == "object":
            focus_labels = object_image_to_labels
        elif focus_key == "attribute":
            focus_labels = attribute_image_to_labels
        elif focus_key == "combined":
            focus_labels = defaultdict(set)
            for image_id in set(object_image_to_labels) | set(attribute_image_to_labels):
                focus_labels[image_id].update(object_image_to_labels.get(image_id, set()))
                focus_labels[image_id].update(attribute_image_to_labels.get(image_id, set()))
        else:
            raise ValueError("sample_focus phải là 'combined', 'object', hoặc 'attribute'.")

        sample_image_ids = _select_sample_image_ids(
            image_to_labels=focus_labels,
            sample_size=sample_size,
            sample_strategy=sample_strategy,
            seed=seed,
        )

    sample_image_set = set(sample_image_ids) if sample_image_ids is not None else None
    if sample_image_set is not None:
        print(f"[Task 1] Giới hạn preprocessing trên {len(sample_image_set)} image_id (focus={sample_focus}, strategy={sample_strategy}).")

    selected_samples = valid_samples
    if sample_image_set is not None:
        selected_samples = [sample for sample in valid_samples if sample["image_id"] in sample_image_set]
    else:
        selected_samples = valid_samples

    all_obj_names = [sample["name"] for sample in selected_samples]
    all_attr_names = [attr for sample in selected_samples for attr in sample["attributes"]]
            
    # 3. Build Vocabs
    obj_vocab = build_vocab(all_obj_names, max_size=max_objects, normalizer=_normalize_object_name)
    attr_vocab = build_vocab(all_attr_names, max_size=max_attributes)
    
    with open(out_path / "object_vocab.json", "w") as f:
        json.dump(obj_vocab, f, indent=2)
    with open(out_path / "attribute_vocab.json", "w") as f:
        json.dump(attr_vocab, f, indent=2)
    
    # 4. Map to Index
    print("Đang mapping dữ liệu sang integer indices...")
    processed_samples = []
    for s in selected_samples:
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
    if split_by_image_id:
        image_ids = [sample["image_id"] for sample in processed_samples]
        train_ids, val_ids, test_ids = _split_image_ids(
            image_ids,
            split_ratios=split_ratios,
            seed=seed,
        )

        split_lookup = {image_id: "train" for image_id in train_ids}
        split_lookup.update({image_id: "val" for image_id in val_ids})
        split_lookup.update({image_id: "test" for image_id in test_ids})

        splits = {"train": [], "val": [], "test": []}
        for sample in processed_samples:
            split_name = split_lookup.get(sample["image_id"], "train")
            splits[split_name].append(sample)
    else:
        train_split = int(split_ratios[0] * len(processed_samples))
        val_split = int((split_ratios[0] + split_ratios[1]) * len(processed_samples))

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
    max_relations: int = 150,
    raw_image_data_path: Optional[str] = None,
    sample_image_ids: Optional[List[int]] = None,
    sample_size: Optional[int] = None,
    sample_strategy: str = "random",
    split_by_image_id: bool = False,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
):
    """
    Tiền xử lý file cho Task 2 (Relationships).
    """
    print("--- [Task 2] Bắt đầu Preprocessing ---")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    raw_relationships_path = _resolve_raw_file(raw_relationships_path)
    try:
        with open(raw_relationships_path, "r", encoding="utf-8") as f:
            rel_data = json.load(f)
    except FileNotFoundError:
        print(f"[Error] Không tìm thấy {raw_relationships_path}. Hãy tải dữ liệu trước.")
        return

    if raw_image_data_path is None:
        raw_image_data_path = str(Path(raw_relationships_path).with_name("image_data.json"))

    raw_image_data_path = _resolve_raw_file(raw_image_data_path)
    try:
        with open(raw_image_data_path, "r", encoding="utf-8") as f:
            image_data = json.load(f)
    except FileNotFoundError:
        print(f"[Error] Không tìm thấy {raw_image_data_path}. Hãy tải dữ liệu trước.")
        return

    image_records = image_data if isinstance(image_data, list) else image_data.get("images", image_data.get("samples", []))
    image_info_map = {}
    for img in image_records:
        image_id = img.get("image_id")
        if image_id is None:
            continue
        image_info_map[image_id] = {
            "width": img.get("width"),
            "height": img.get("height"),
        }
        
    all_predicates = []
    image_to_labels = defaultdict(set)
    valid_samples = []
    
    for img_rel in tqdm(rel_data, desc="Parsing Relationships"):
        img_id = img_rel["image_id"]
        
        for rel in img_rel["relationships"]:
            pred = str(rel.get("predicate", "")).lower().strip()
            if not pred: continue
            
            all_predicates.append(pred)
            image_to_labels[img_id].add(f"relation::{_normalize_relation_name(pred)}")
            
            subj = rel["subject"]
            obj = rel["object"]
            
            valid_samples.append({
                "image_id": img_id,
                "relationship_id": rel["relationship_id"],
                "subject_bbox": [subj["x"], subj["y"], subj["w"], subj["h"]],
                "object_bbox": [obj["x"], obj["y"], obj["w"], obj["h"]],
                "subject_name": subj.get("name", ""),
                "object_name": obj.get("name", ""),
                "predicate": pred,
                "image_info": image_info_map.get(img_id, {}),
            })

    if sample_image_ids is None and split_by_image_id and sample_size is not None:
        sample_image_ids = _select_sample_image_ids(
            image_to_labels=image_to_labels,
            sample_size=sample_size,
            sample_strategy=sample_strategy,
            seed=seed,
        )

    sample_image_set = set(sample_image_ids) if sample_image_ids is not None else None
    if sample_image_set is not None:
        print(f"[Task 2] Giới hạn preprocessing trên {len(sample_image_set)} image_id (strategy={sample_strategy}).")

    selected_samples = valid_samples
    if sample_image_set is not None:
        selected_samples = [sample for sample in valid_samples if sample["image_id"] in sample_image_set]

    all_predicates = [sample["predicate"] for sample in selected_samples]
    
    rel_vocab = build_vocab(all_predicates, max_size=max_relations, normalizer=_normalize_relation_name)
    
    with open(out_path / "relation_vocab.json", "w") as f:
        json.dump(rel_vocab, f, indent=2)
        
    print("Đang mapping dữ liệu sang integer indices...")
    processed_samples = []
    for s in selected_samples:
        rel_idx = rel_vocab.get(_normalize_relation_name(s["predicate"]), rel_vocab["<UNK>"])
        if rel_idx == 0: continue # Bỏ qua relation lạ
        
        processed_samples.append({
            "image_id": s["image_id"],
            "relationship_id": s["relationship_id"],
            "subject_bbox": s["subject_bbox"],
            "object_bbox": s["object_bbox"],
            "subject_name": _normalize_object_name(s["subject_name"]),
            "object_name": _normalize_object_name(s["object_name"]),
            "relation_label": rel_idx,
            "image_info": s.get("image_info", {}),
        })
        
    if split_by_image_id:
        image_ids = [sample["image_id"] for sample in processed_samples]
        train_ids, val_ids, test_ids = _split_image_ids(
            image_ids,
            split_ratios=split_ratios,
            seed=seed,
        )

        split_lookup = {image_id: "train" for image_id in train_ids}
        split_lookup.update({image_id: "val" for image_id in val_ids})
        split_lookup.update({image_id: "test" for image_id in test_ids})

        splits = {"train": [], "val": [], "test": []}
        for sample in processed_samples:
            split_name = split_lookup.get(sample["image_id"], "train")
            splits[split_name].append(sample)
    else:
        train_split = int(split_ratios[0] * len(processed_samples))
        val_split = int((split_ratios[0] + split_ratios[1]) * len(processed_samples))

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
    processed_dir: str = "data/processed",
    max_objects: int = 150,
    max_attributes: int = 100,
    max_relations: int = 150,
    sample_image_ids: Optional[List[int]] = None,
    sample_size: Optional[int] = None,
    sample_strategy: str = "random",
    sample_focus: str = "combined",
    split_by_image_id: bool = False,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
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
            str(p_dir),
            max_objects=max_objects,
            max_attributes=max_attributes,
            sample_image_ids=sample_image_ids,
            sample_size=sample_size,
            sample_strategy=sample_strategy,
            sample_focus=sample_focus,
            split_by_image_id=split_by_image_id,
            split_ratios=split_ratios,
            seed=seed,
        )
    elif task == "task2":
        preprocess_task2(
            str(r_dir / "relationships.json"),
            str(p_dir),
            max_relations=max_relations,
            raw_image_data_path=str(r_dir / "image_data.json"),
            sample_image_ids=sample_image_ids,
            sample_size=sample_size,
            sample_strategy=sample_strategy,
            split_by_image_id=split_by_image_id,
            split_ratios=split_ratios,
            seed=seed,
        )
    else:
        print(f"Unknown task: {task}")


def preprocess_object_attribute(
    raw_objects_path: str,
    raw_attributes_path: str,
    output_dir: str,
    max_objects: int = 150,
    max_attributes: int = 100,
    sample_image_ids: Optional[List[int]] = None,
    sample_size: Optional[int] = None,
    sample_strategy: str = "random",
    sample_focus: str = "combined",
    split_by_image_id: bool = False,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
):
    """Explicit preprocessing entrypoint for the object/attribute pipeline."""
    return preprocess_task1(
        raw_objects_path=raw_objects_path,
        raw_attributes_path=raw_attributes_path,
        output_dir=output_dir,
        max_objects=max_objects,
        max_attributes=max_attributes,
        sample_image_ids=sample_image_ids,
        sample_size=sample_size,
        sample_strategy=sample_strategy,
        sample_focus=sample_focus,
        split_by_image_id=split_by_image_id,
        split_ratios=split_ratios,
        seed=seed,
    )


def preprocess_relation(
    raw_relationships_path: str,
    output_dir: str,
    max_relations: int = 150,
    raw_image_data_path: Optional[str] = None,
    sample_image_ids: Optional[List[int]] = None,
    sample_size: Optional[int] = None,
    sample_strategy: str = "random",
    split_by_image_id: bool = False,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
):
    """Explicit preprocessing entrypoint for the relation pipeline."""
    return preprocess_task2(
        raw_relationships_path=raw_relationships_path,
        output_dir=output_dir,
        max_relations=max_relations,
        raw_image_data_path=raw_image_data_path,
        sample_image_ids=sample_image_ids,
        sample_size=sample_size,
        sample_strategy=sample_strategy,
        split_by_image_id=split_by_image_id,
        split_ratios=split_ratios,
        seed=seed,
    )


def build_object_attribute_vocab_and_splits(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed/object_attribute",
    max_objects: int = 150,
    max_attributes: int = 100,
    sample_image_ids: Optional[List[int]] = None,
    sample_size: Optional[int] = None,
    sample_strategy: str = "random",
    sample_focus: str = "combined",
    split_by_image_id: bool = False,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
):
    """Object/attribute-specific vocabulary and split builder."""
    r_dir = Path(raw_dir)
    return preprocess_object_attribute(
        raw_objects_path=str(r_dir / "objects.json"),
        raw_attributes_path=str(r_dir / "attributes.json"),
        output_dir=processed_dir,
        max_objects=max_objects,
        max_attributes=max_attributes,
        sample_image_ids=sample_image_ids,
        sample_size=sample_size,
        sample_strategy=sample_strategy,
        sample_focus=sample_focus,
        split_by_image_id=split_by_image_id,
        split_ratios=split_ratios,
        seed=seed,
    )


def build_relation_vocab_and_splits(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed/relation",
    max_relations: int = 150,
    sample_image_ids: Optional[List[int]] = None,
    sample_size: Optional[int] = None,
    sample_strategy: str = "random",
    split_by_image_id: bool = False,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
):
    """Relation-specific vocabulary and split builder."""
    r_dir = Path(raw_dir)
    return preprocess_relation(
        raw_relationships_path=str(r_dir / "relationships.json"),
        output_dir=processed_dir,
        max_relations=max_relations,
        raw_image_data_path=str(r_dir / "image_data.json"),
        sample_image_ids=sample_image_ids,
        sample_size=sample_size,
        sample_strategy=sample_strategy,
        split_by_image_id=split_by_image_id,
        split_ratios=split_ratios,
        seed=seed,
    )