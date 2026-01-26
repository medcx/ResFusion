import os
import SimpleITK as sitk
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# -------------------- 配置 --------------------
data_dir = "/data/XJH/data/Brain/BraTS/MICCAI_BraTS2021_TrainingData/"
output_dir = "/data1/XJH/Imgtrans/data/Havard/MyDatasets/BraTs"
selected_modalities = ["t1n", "t2w"]
mask_name = "mask"

mid_slice_range = 5      # 中间切片范围
final_size = 192         # 填充或裁剪到固定尺寸
n_workers = 8            # 并行线程数
# --------------------------------------------

os.makedirs(output_dir, exist_ok=True)

# -------------------- 辅助函数 --------------------
def load_sitk_image(path):
    return sitk.ReadImage(path)

def pad_or_crop(img_np, target_size=192):
    h, w = img_np.shape
    pad_h = max(target_size - h, 0)
    pad_w = max(target_size - w, 0)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    img_padded = np.pad(img_np, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
    start_y = (img_padded.shape[0] - target_size) // 2
    start_x = (img_padded.shape[1] - target_size) // 2
    return img_padded[start_y:start_y+target_size, start_x:start_x+target_size]

def process_patient(patient_folder):
    patient_path = os.path.join(data_dir, patient_folder)
    if not os.path.isdir(patient_path):
        return []

    files = [i for i in os.listdir(patient_path) if i.endswith('.nii.gz')]
    patient_files = {f.split('-')[-1].split('.')[0]: os.path.join(patient_path, f) for f in files}

    seg_path = patient_files.get("seg")
    t2_path = patient_files.get("t2f")
    if seg_path is None or t2_path is None:
        return []

    # 读取 T2 并裁剪
    t2_data = sitk.GetArrayFromImage(load_sitk_image(t2_path))
    coords = np.argwhere(t2_data > 0)
    if coords.size == 0:
        return []
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1

    # 裁剪 seg
    seg_data = sitk.GetArrayFromImage(load_sitk_image(seg_path))[z0:z1, y0:y1, x0:x1]

    # 裁剪其他模态并归一化
    modality_data_dict = {}
    for modality in selected_modalities:
        modality_path = patient_files.get(modality)
        if modality_path is None:
            continue
        img_data = sitk.GetArrayFromImage(load_sitk_image(modality_path))[z0:z1, y0:y1, x0:x1]
        img_data_norm = ((img_data - img_data.min()) / (img_data.ptp() + 1e-5) * 255).astype(np.uint8)
        modality_data_dict[modality] = img_data_norm

    # 提取中间切片
    z_mid = seg_data.shape[0] // 2
    start_idx = max(0, z_mid - mid_slice_range)
    end_idx = min(seg_data.shape[0], z_mid + mid_slice_range + 1)

    patient_slices = []
    for slice_idx in range(start_idx, end_idx):
        seg_slice = seg_data[slice_idx]
        if np.any(seg_slice > 0):
            patient_slices.append({
                "patient": patient_folder,
                "slice_idx": slice_idx,
                "seg": seg_slice.copy(),
                "modalities": {k: v[slice_idx].copy() for k, v in modality_data_dict.items()}
            })
    return patient_slices

# -------------------- 并行处理患者 --------------------
all_slices = []
with ThreadPoolExecutor(max_workers=n_workers) as executor:
    futures = [executor.submit(process_patient, pf) for pf in os.listdir(data_dir)]
    for f in tqdm(futures):
        result = f.result()
        all_slices.extend(result)

# -------------------- 打乱划分 --------------------
np.random.seed(42)
np.random.shuffle(all_slices)
train_slices, test_slices = train_test_split(all_slices, test_size=0.2, random_state=42)

# -------------------- 批量写入 PNG（单线程） --------------------
def save_slice_png(item, subset):
    patient_folder = item["patient"]
    slice_idx = item["slice_idx"]
    seg_slice = pad_or_crop(item["seg"], final_size)
    # 保存 mask
    mask_dir = os.path.join(output_dir, subset, mask_name)
    os.makedirs(mask_dir, exist_ok=True)
    mask_path = os.path.join(mask_dir, f"{patient_folder}_{slice_idx:03d}.png")
    Image.fromarray(((seg_slice / 4) * 255).astype(np.uint8)).save(mask_path)
    # 保存模态
    # for modality, img_slice in item["modalities"].items():
    #     img_slice = pad_or_crop(img_slice, final_size)
    #     save_dir = os.path.join(output_dir, subset, modality)
    #     os.makedirs(save_dir, exist_ok=True)
    #     save_path = os.path.join(save_dir, f"{patient_folder}_{slice_idx:03d}.png")
    #     Image.fromarray(img_slice).save(save_path)

for subset, slices in [("train", train_slices), ("test", test_slices)]:
    for item in tqdm(slices):
        save_slice_png(item, subset)
