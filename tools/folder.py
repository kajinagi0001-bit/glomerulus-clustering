import os
import sys
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class NestedImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        for class_idx, (root, dirs, files) in enumerate(os.walk(root_dir)):
            if files:
                class_name = os.path.basename(root)
                if class_name not in self.class_to_idx:
                    self.class_to_idx[class_name] = class_idx
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(root, file))
                        self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path


class SplitNestedImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train', train_ratio=0.8, seed=42):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        
        # 元のNestedImageDatasetからデータを収集
        base_dataset = NestedImageDataset(root_dir, transform=None)
        
        # データの分割
        np.random.seed(seed)
        indices = np.random.permutation(len(base_dataset))
        split_idx = int(len(indices) * train_ratio)
        
        if split == 'train':
            selected_indices = indices[:split_idx]
        else:  # val
            selected_indices = indices[split_idx:]
            
        # 選択されたインデックスのデータのみを保持
        self.image_paths = [base_dataset.image_paths[i] for i in selected_indices]
        self.labels = [base_dataset.labels[i] for i in selected_indices]
        self.class_to_idx = base_dataset.class_to_idx

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path