import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import glob
import requests
import zipfile
import shutil
import random

def download_and_unzip(url, save_path, extract_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        with zipfile.ZipFile(save_path, 'r') as z:
            z.testzip()
        
        temp_extract = os.path.join(os.path.dirname(save_path), "temp_extract")
        os.makedirs(temp_extract, exist_ok=True)
        with zipfile.ZipFile(save_path, 'r') as z:
            z.extractall(temp_extract)
        
        os.makedirs(extract_path, exist_ok=True)
        extracted_folders = os.listdir(temp_extract)
        if 'icpr_prepared' in extracted_folders:
            src = os.path.join(temp_extract, 'icpr_prepared')
            for item in os.listdir(src):
                src_item = os.path.join(src, item)
                dst_item = os.path.join(extract_path, item)
                if os.path.exists(dst_item):
                    if os.path.isdir(dst_item):
                        shutil.rmtree(dst_item)
                    else:
                        os.remove(dst_item)
                shutil.move(src_item, extract_path)
        
        shutil.rmtree(temp_extract)
        os.remove(save_path)
        
    except Exception as e:
        raise Exception(f"Failed to download or unzip: {str(e)}")

def setup_dataset():
    URL = r"https://www.dropbox.com/scl/fi/tscgh3pxwzfvesnu6l6uv/icpr_prepared.zip?rlkey=8oay8sod3jc1hvwhgqvylaefr&st=udj92wmp&dl=1"
    dataset_name = "retinal_blood_vessel_icpr_seg"
    dataset_zip_path = os.path.join(os.getcwd(), f"{dataset_name}.zip")
    dataset_path = os.path.join(os.getcwd(), dataset_name)

    if not os.path.exists(dataset_path):
        download_and_unzip(URL, dataset_zip_path, dataset_path)
    
    return dataset_path

def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class RetinalDataset(Dataset):
    def __init__(self, image_dir, mask_dir, size=(512, 512), include_filename=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.size = size
        self.include_filename = include_filename
        
        extensions = ["*.[tT][iI][fF]", "*.png", "*.jpg", "*.jpeg", "*.tiff", "*.bmp", "*.gif"]
        self.image_files = []
        self.mask_files = []
        
        for ext in extensions:
            self.image_files.extend(sorted(glob.glob(os.path.join(image_dir, ext))))
            self.mask_files.extend(sorted(glob.glob(os.path.join(mask_dir, ext))))
        
        self.image_files = sorted(list(set(self.image_files)))
        self.mask_files = sorted(list(set(self.mask_files)))
        
        assert len(self.image_files) == len(self.mask_files), "Mismatch between images and masks"
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        mask_path = self.mask_files[idx]
        
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, self.size)
        x = np.transpose(image, (2, 0, 1))
        x = x / 255.0
        x = x.astype(np.float32)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.size)
        y = mask / 255.0
        y = y.astype(np.float32)
        
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        y = y.unsqueeze(0)
        
        if self.include_filename:
            return x, y, os.path.basename(image_path)
        return x, y