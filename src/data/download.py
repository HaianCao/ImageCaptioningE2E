"""
Module tải dữ liệu từ server của lab Stanford - Visual Genome Dataset.

Cung cấp các hàm để:
- Tải các file annotation dạng zip từ Stanford
- Giải nén tự động và convert sang file JSON chuẩn
- Tải ảnh theo danh sách image_id từ Stanford URL
- Kiểm tra tính toàn vẹn của dữ liệu tại thư mục tải về
"""

import os
import json
import zipfile
import requests
from pathlib import Path
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# URL tải file ZIP gốc
VG_URLS = {
    "objects": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects_v1_2.json.zip",
    "attributes": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/attributes.json.zip",
    "relationships": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationships_v1_2.json.zip",
    "image_data": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip"
}


def download_file(url: str, dest_path: str, chunk_size: int = 1048576) -> str:
    """
    Tải file lớn có thanh tiến trình.
    
    Args:
        url: Link tải trực tiếp
        dest_path: Đường dẫn lưu file
        chunk_size: Kích thước chunk (1MB)
    
    Returns:
        Đường dẫn file đã tải
    """
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=f"Tải {Path(dest_path).name}",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = f.write(data)
            bar.update(size)
            
    return dest_path


def unzip_file(zip_path: str, extract_to: str) -> List[str]:
    """
    Giải nén file zip và trả về danh sách file được giải nén.
    """
    extracted_files = []
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        extracted_files = [os.path.join(extract_to, f) for f in zip_ref.namelist()]
        
    print(f"[Unzip] Đã giải nén {len(extracted_files)} files từ {Path(zip_path).name}")
    return extracted_files


def download_and_extract_metadata(
    setup_types: Optional[List[str]] = None,
    raw_dir: str = "data/raw",
    keep_zip: bool = False
) -> Dict[str, str]:
    """
    Tải tự động và giải nén các file metadata cần thiết từ Stanford.
    
    Args:
        setup_types: Danh sách ['objects', 'attributes', 'relationships', 'image_data']
        raw_dir: Thư mục chứa dữ liệu thô tải về
        keep_zip: Có giữ lại file zip hay không (mặc định False: xóa để nhẹ disk)
        
    Returns:
        Dict mapping loại data -> đường dẫn file json đã giải nén
    """
    if setup_types is None:
        setup_types = ["objects", "attributes", "relationships", "image_data"]
        
    raw_path = Path(raw_dir)
    results = {}
    
    for dtype in setup_types:
        if dtype not in VG_URLS:
            print(f"[Warning] Bỏ qua '{dtype}', không có link Download.")
            continue
            
        url = VG_URLS[dtype]
        zip_name = url.split('/')[-1]
        zip_path = raw_path / zip_name
        
        # Dự đoán tên file JSON sẽ được giải nén
        json_name = zip_name.replace('.zip', '')
        json_path = raw_path / json_name
        
        if json_path.exists():
            print(f"[Skip] {json_name} đã tồn tại.")
            results[dtype] = str(json_path)
            continue
            
        print(f"\n--- Đang xử lý {dtype} ---")
        try:
            # Tải file
            download_file(url, str(zip_path))
            
            # Giải nén
            extracted = unzip_file(str(zip_path), str(raw_path))
            if extracted:
                results[dtype] = extracted[0] # Giả sử 1 zip chứa 1 json
            
            # Dọn dẹp
            if not keep_zip and zip_path.exists():
                os.remove(zip_path)
                print(f"[Cleanup] Đã xóa {zip_name}")
                
        except Exception as e:
            print(f"[Error] Không tải được {dtype}: {str(e)}")
            
    return results


def save_dataset_as_json(
    data: Any,
    save_path: str,
) -> None:
    """
    Lưu object Dict/List thành file JSON.
    Hàm này dùng để hỗ trợ tương thích ngược với code cũ nếu cần lưu mảng dict nội bộ.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Save] Đang lưu cấu trúc dữ liệu → {save_path}")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def download_vg_images(
    image_ids: List[int],
    image_dir: str = "data/raw/images",
    url_template: str = "https://cs.stanford.edu/people/rak248/VG_100K/{image_id}.jpg",
    max_workers: int = 8,
) -> Dict[int, str]:
    """
    Tải ảnh gốc từ Visual Genome theo danh sách image_id tĩnh.

    Args:
        image_ids: Danh sách image ID cần tải (VD: [1, 2, 3])
        image_dir: Thư mục lưu ảnh (.jpg)
        url_template: URL template với {image_id} placeholder
        max_workers: Số threads tải song song đa luồng

    Returns:
        Dict mapping image_id -> local file path đã tải
    """
    image_dir_path = Path(image_dir)
    image_dir_path.mkdir(parents=True, exist_ok=True)

    results = {}
    to_download = []

    # Kiểm tra file đã tồn tại trên disk chưa
    for img_id in image_ids:
        # Ở VG_100K, một số ảnh thuộc part 2: VG_100K_2. Để đơn giản ta giả sử dùng URL cơ bản.
        # Ở kịch bản thực tế, Image URL có sẵn trong dữ liệu image_data.json
        local_path = image_dir_path / f"{img_id}.jpg"
        if local_path.exists():
            results[img_id] = str(local_path)
        else:
            to_download.append(img_id)

    print(f"[Images] {len(results)} ảnh đã có | {len(to_download)} ảnh cần tải")
    if not to_download:
        return results

    def _download_one(img_id: int) -> tuple:
        url = url_template.format(image_id=img_id)
        local_path = image_dir_path / f"{img_id}.jpg"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            local_path.write_bytes(resp.content)
            return img_id, str(local_path)
        except Exception:
            # Fallback URL thứ 2 của VG
            url_2 = f"https://cs.stanford.edu/people/rak248/VG_100K_2/{img_id}.jpg"
            try:
                resp = requests.get(url_2, timeout=30)
                resp.raise_for_status()
                local_path.write_bytes(resp.content)
                return img_id, str(local_path)
            except Exception:
                return img_id, None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download_one, img_id): img_id for img_id in to_download}
        pbar = tqdm(as_completed(futures), total=len(to_download), desc="Downloading images")
        for future in pbar:
            img_id, path = future.result()
            if path:
                results[img_id] = path

    success = sum(1 for id in to_download if id in results)
    print(f"[Images] Tải thành công: {success}/{len(to_download)} ảnh mới")
    return results


def verify_download(
    raw_dir: str = "data/raw",
    expected_files: Optional[List[str]] = None,
) -> Dict[str, bool]:
    """
    Kiểm tra các file dữ liệu đã được giải nén đầy đủ chưa.

    Args:
        raw_dir: Thư mục chứa dữ liệu thô json
        expected_files: Danh sách file json mong muốn

    Returns:
        Dict trạng thái từng file -> exists (bool)
    """
    if expected_files is None:
        expected_files = [
            "objects.json",
            "attributes.json",
            "relationships.json",
            "image_data.json",
        ]

    raw_path = Path(raw_dir)
    status = {}
    print("\n--- Kiểm tra dữ liệu RAW ---")
    for fname in expected_files:
        fpath = raw_path / fname
        status[fname] = fpath.exists()
        icon = "✅" if fpath.exists() else "❌"
        size = f"({fpath.stat().st_size / 1e6:.1f} MB)" if fpath.exists() else "(Thiếu file)"
        print(f"  {icon}  {fname} {size}")

    return status
