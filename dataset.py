import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import load_sar_image

class SARDatasetOptimized(Dataset):
    def __init__(self, file_list, patch_size=64, patches_per_file=1000):
        self.patches = [] 
        print(f"Initializing Dataset with {len(file_list)} scenes...")
        
        for i, path in enumerate(file_list):
            print(f"   [{i+1}/{len(file_list)}] Processing: {os.path.basename(path)}...")
            full_img = load_sar_image(path)
            
            if full_img is None: continue
                
            h, w = full_img.shape
            for _ in range(patches_per_file):
                x = np.random.randint(0, h - patch_size)
                y = np.random.randint(0, w - patch_size)
                patch = full_img[x:x+patch_size, y:y+patch_size].copy()
                self.patches.append(patch)
            
            del full_img 
            
        print(f"Dataset Ready: {len(self.patches)} patches loaded.")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        clean_patch = self.patches[idx]
        
        inp = clean_patch.copy()
        target = clean_patch.copy()
        
        mask = np.random.rand(*inp.shape) < 0.1
        noise = np.random.normal(0, 0.2, inp.shape)
        inp[mask] = noise[mask]
        
        return (torch.tensor(inp).float().unsqueeze(0), 
                torch.tensor(target).float().unsqueeze(0))