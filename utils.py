import os
import cv2
import zipfile
import numpy as np
import pywt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def load_sar_image(path):
    if not os.path.exists(path):
        return None
    
    try:
        with zipfile.ZipFile(path, 'r') as z:
            tiff_files = [f for f in z.namelist() if f.endswith(".tiff") and "vv" in f.lower()]
            if not tiff_files:
                return None
            
            target_file = tiff_files[0]
            with z.open(target_file) as f:
                file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
                if img is None:
                    return None

        img = img.astype(np.float32)
        img = np.log1p(img)
        img = img / (np.max(img) + 1e-7)
        return img
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def wavelet_denoise(img, wavelet='db1', level=2):
    coeffs = pywt.wavedec2(img, wavelet, level=level)
    cA, details = coeffs[0], coeffs[1:]
    
    new_details = []
    for cH, cV, cD in details:
        sigma = np.median(np.abs(cD)) / 0.6745
        threshold = sigma * 1.5
        new_details.append((
            pywt.threshold(cH, threshold, 'soft'),
            pywt.threshold(cV, threshold, 'soft'),
            pywt.threshold(cD, threshold, 'soft')
        ))
    
    return pywt.waverec2([cA] + new_details, wavelet)

def calculate_metrics(clean, denoised):
    p = psnr(clean, denoised, data_range=1.0)
    s = ssim(clean, denoised, data_range=1.0)
    return p, s