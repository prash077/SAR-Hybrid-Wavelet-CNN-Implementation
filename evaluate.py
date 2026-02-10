import torch
import numpy as np
import glob
import os
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

from config import TEST_DIR, MODEL_SAVE_PATH
from utils import load_sar_image, calculate_metrics, wavelet_denoise
from model import SimpleSAR_CNN

def evaluate():
    test_files = glob.glob(os.path.join(TEST_DIR, "*.zip"))
    if len(test_files) < 2:
        print("Need at least 2 files in GroupA for Temporal Averaging.")
        return

    print("Generating Ground Truth (Temporal Average)...")
    
    first_img = load_sar_image(test_files[0])
    h, w = first_img.shape
    running_sum = first_img.copy().astype(np.float64)
    count = 1
    
    noisy_input = first_img.copy()
    del first_img

    for i in range(1, len(test_files)):
        img = load_sar_image(test_files[i])
        if img is not None:
            if img.shape != (h, w):
                img = cv2.resize(img, (w, h))
            running_sum += img
            count += 1
            del img
            
    clean_ref = (running_sum / count).astype(np.float32)
    print("Ground Truth Created.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleSAR_CNN().to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    cx, cy = h//2, w//2
    crop = 512
    noisy_crop = noisy_input[cx:cx+crop, cy:cy+crop]
    clean_crop = clean_ref[cx:cx+crop, cy:cy+crop]

    input_tensor = torch.tensor(noisy_crop).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        cnn_out = model(input_tensor).squeeze().cpu().numpy()

    med_out = median_filter(noisy_crop, size=3)
    
    wav_out = wavelet_denoise(noisy_crop)
    if wav_out.shape != clean_crop.shape:
        wav_out = cv2.resize(wav_out, (clean_crop.shape[1], clean_crop.shape[0]))

    p_cnn, s_cnn = calculate_metrics(clean_crop, cnn_out)
    p_med, s_med = calculate_metrics(clean_crop, med_out)
    p_wav, s_wav = calculate_metrics(clean_crop, wav_out)

    print("\nFINAL RESULTS TABLE")
    print("-" * 40)
    print(f"{'METHOD':<20} | {'PSNR (dB)':<10} | {'SSIM':<10}")
    print("-" * 40)
    print(f"{'Median Filter':<20} | {p_med:.2f}       | {s_med:.4f}")
    print(f"{'Wavelet Only':<20} | {p_wav:.2f}       | {s_wav:.4f}")
    print(f"{'Proposed Hybrid':<20} | {p_cnn:.2f}       | {s_cnn:.4f}")
    print("-" * 40)

    plt.figure(figsize=(15, 5))
    plt.subplot(1,3,1); plt.title("Noisy Input"); plt.imshow(noisy_crop, cmap='gray'); plt.axis('off')
    plt.subplot(1,3,2); plt.title(f"Our Model\nPSNR: {p_cnn:.2f}"); plt.imshow(cnn_out, cmap='gray'); plt.axis('off')
    plt.subplot(1,3,3); plt.title("Ground Truth"); plt.imshow(clean_crop, cmap='gray'); plt.axis('off')
    plt.savefig("final_results.png")
    print("Result image saved as final_results.png")

if __name__ == "__main__":
    evaluate()