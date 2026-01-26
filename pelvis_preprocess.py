import os
import numpy as np
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split


# ------------------------- 工具函数 -------------------------
def resample_to_new_spacing(image, new_spacing=(1.0, 1.0, 1.0)):
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    if original_spacing != new_spacing:
        new_size = [int(round(original_size[i] * (original_spacing[i] / new_spacing[i]))) for i in range(3)]
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetDefaultPixelValue(0)
        return resampler.Execute(image)
    return image

def normalize_image(image):
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    mn, mx = arr.min(), arr.max()
    arr = (arr - mn) / (mx - mn) if mx > mn else np.zeros_like(arr)
    norm_img = sitk.GetImageFromArray(arr)
    norm_img.CopyInformation(image)
    return norm_img

def crop_black_border_3d(arr):
    coords = np.argwhere(arr > 0)
    if coords.size == 0:
        return arr, (0, arr.shape[0], 0, arr.shape[1], 0, arr.shape[2])
    z_min, x_min, y_min = coords.min(axis=0)
    z_max, x_max, y_max = coords.max(axis=0) + 1
    return arr[z_min:z_max, x_min:x_max, y_min:y_max], (z_min, z_max, x_min, x_max, y_min, y_max)

def pad_and_center_crop(ct_img, mr_img, target_size):
    h, w = ct_img.shape
    pad_h = max(target_size - h, 0)
    pad_w = max(target_size - w, 0)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    ct_padded = np.pad(ct_img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")
    mr_padded = np.pad(mr_img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")

    start_y = (ct_padded.shape[0] - target_size) // 2
    start_x = (ct_padded.shape[1] - target_size) // 2

    ct_crop = ct_padded[start_y:start_y + target_size, start_x:start_x + target_size]
    mr_crop = mr_padded[start_y:start_y + target_size, start_x:start_x + target_size]

    return ct_crop, mr_crop

# ------------------------- 生成单个患者切片 -------------------------
def generate_slices(ct_path, mr_path, patient_id, target_size=256, slice_size=None, new_spacing=(1.0,1.0,2.5), discard_slices=5):
    img_ct = normalize_image(sitk.ReadImage(ct_path))
    img_ct = resample_to_new_spacing(img_ct, new_spacing)
    arr_ct = sitk.GetArrayFromImage(img_ct)

    img_mr = normalize_image(sitk.ReadImage(mr_path))
    img_mr = resample_to_new_spacing(img_mr, new_spacing)
    arr_mr = sitk.GetArrayFromImage(img_mr)

    assert arr_ct.shape == arr_mr.shape

    # 去除黑边
    arr_mr_crop, crop_box = crop_black_border_3d(arr_mr)
    z_min, z_max, x_min, x_max, y_min, y_max = crop_box
    arr_ct_crop = arr_ct[z_min:z_max, x_min:x_max, y_min:y_max]

    arr_ct = arr_ct_crop
    arr_mr = arr_mr_crop

    Z = arr_mr.shape[0]

    # ---------------------- 去掉前后 discard_slices 张 ----------------------
    start_slice = discard_slices
    end_slice = Z - discard_slices

    # 如果 slice_size 指定了，只取中间 slice_size 张
    if slice_size is not None:
        mid = (start_slice + end_slice) // 2
        half = slice_size // 2
        start_slice = max(start_slice, mid - half)
        end_slice = min(end_slice, mid + half)

    slices = []
    for z in range(start_slice, end_slice):
        slice_ct, slice_mr = pad_and_center_crop(arr_ct[z], arr_mr[z], target_size)
        slice_ct = (slice_ct * 255).clip(0, 255).astype(np.uint8)
        slice_mr = (slice_mr * 255).clip(0, 255).astype(np.uint8)
        slices.append((slice_ct, slice_mr, f"{patient_id}_{z}.png"))

    return slices
# ------------------------- 保存切片 -------------------------
def save_slices(slices, output_dir_ct, output_dir_mr):
    os.makedirs(output_dir_ct, exist_ok=True)
    os.makedirs(output_dir_mr, exist_ok=True)
    for ct_slice, mr_slice, fname in slices:
        Image.fromarray(ct_slice).save(os.path.join(output_dir_ct, fname))
        Image.fromarray(mr_slice).save(os.path.join(output_dir_mr, fname))

# ------------------------- 主流程 -------------------------
if __name__ == "__main__":
    import random

    target_size = 256
    new_spacing = (1.5, 1.5, 5.625)
    slice_size = None
    num_workers = 4
    test_size = 0.2

    input_path = "/data1/XJH/Dataset/SynthRAD2023/pelvis"
    output_path = "/data1/XJH/Imgtrans/data/Havard/MyDatasets/pelvis"

    patients = sorted([p for p in os.listdir(input_path) if p.startswith("1P")])
    print(f"Total patients: {len(patients)}")

    all_slices = []

    # ------------------------- 多线程生成所有切片 -------------------------
    def task(patient_id):
        root = os.path.join(input_path, patient_id)
        ct_path = os.path.join(root, "ct.nii.gz")
        mr_path = os.path.join(root, "mr.nii.gz")
        return generate_slices(ct_path, mr_path, patient_id, target_size, slice_size, new_spacing)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(task, patients), total=len(patients), desc="Generating slices"))

    # flatten list
    for r in results:
        all_slices.extend(r)

    print(f"Total slices generated: {len(all_slices)}")

    # ------------------------- 切片级别划分 train/test -------------------------
    train_slices, test_slices = train_test_split(all_slices, test_size=test_size, random_state=42)
    print(f"Train slices: {len(train_slices)}, Test slices: {len(test_slices)}")

    # ------------------------- 保存 -------------------------
    train_ct_dir = os.path.join(output_path, "train", "CT")
    train_mr_dir = os.path.join(output_path, "train", "MRI")
    test_ct_dir = os.path.join(output_path, "test", "CT")
    test_mr_dir = os.path.join(output_path, "test", "MRI")

    save_slices(train_slices, train_ct_dir, train_mr_dir)
    save_slices(test_slices, test_ct_dir, test_mr_dir)

    print("All slices saved successfully!")
