import argparse
import csv
import os
from datetime import datetime

import kornia
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def rgb_to_ycbcr(img):
    if img.ndim == 3:
        img = img.unsqueeze(0)
    if img.dtype == torch.uint8:
        img = img.float()
    if img.max() > 1.0:
        img = img / 255.0  

    T = torch.tensor([
        [65.481, 128.553, 24.966],
        [-37.797, -74.203, 112.000],
        [112.000, -93.786, -18.214]
    ], dtype=img.dtype, device=img.device) / 255
    
    offset = torch.tensor([16, 128, 128], dtype=img.dtype, device=img.device)
    ycbcr_img = torch.zeros_like(img)
    
    for p in range(3):
        ycbcr_img[:, p, :, :] = (T[p, 0] * img[:, 0, :, :] + 
                                 T[p, 1] * img[:, 1, :, :] + 
                                 T[p, 2] * img[:, 2, :, :] + 
                                 offset[p] / 255)
                                 
    return ycbcr_img


def load_image_as_tensor(img_path, device, target_size=None):
    img = Image.open(img_path).convert("RGB")
    if target_size is not None:
        img = img.resize(target_size, Image.LANCZOS)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1))
    
    return img_tensor.unsqueeze(0).to(device)


def get_gt_filename(result_filename):
    stem, ext = os.path.splitext(result_filename)
    if stem.endswith('_00'):
        stem = stem[:-3]
    return stem + ext


def main():
    parser = argparse.ArgumentParser(description='Evaluate GTAV-NightRain results')
    parser.add_argument('--test_set', default='set1', type=str, choices=['set1', 'set2', 'set3'])
    
    # TODO: Set path to results directory
    parser.add_argument('--result_dir', default='', type=str)
    
    # TODO: Set path to GTAV-NightRain ground truth directory
    parser.add_argument('--gt_dir', default='', type=str)
    
    parser.add_argument('--model_name', default='UNet_RLP_RPIM', type=str)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*70)
    print(f"Evaluating: GTAV-NightRain {args.test_set}")
    print("="*70)
    
    result_path = os.path.join(args.result_dir, args.model_name, f'GTAV-NightRain_{args.test_set}')
    if not os.path.exists(result_path):
        print(f"Results not found: {result_path}")
        return
        
    gt_path = os.path.join(args.gt_dir, args.test_set, 'target')
    if not os.path.exists(gt_path):
        print(f"GT directory not found: {gt_path}")
        return

    result_files = sorted([f for f in os.listdir(result_path) if f.endswith('.png')])
    if len(result_files) == 0:
        print(f"No result images found")
        return
    
    print(f"Results: {result_path}")
    print(f"GT: {gt_path}")
    print(f"Processing {len(result_files)} images...")
    print(f"✓ GT will be resized to match result dimensions (960×540)")
    
    total_psnr, total_ssim = 0.0, 0.0
    failed = 0
    failed_files = []
    per_image_results = []
    
    for img_file in tqdm(result_files, desc=f'Evaluating GTAV {args.test_set}'):
        try:
            result_img_path = os.path.join(result_path, img_file)
            result_img = load_image_as_tensor(result_img_path, device)
            _, _, result_h, result_w = result_img.shape
            
            gt_filename = get_gt_filename(img_file)
            gt_img_path = os.path.join(gt_path, gt_filename)
            
            if not os.path.exists(gt_img_path):
                print(f"\nGT not found: {gt_img_path}")
                failed += 1
                failed_files.append(img_file)
                continue
            
            gt_img = load_image_as_tensor(gt_img_path, device, target_size=(result_w, result_h))
            
            if result_img.shape != gt_img.shape:
                print(f"\nShape mismatch: {img_file}")
                failed += 1
                failed_files.append(img_file)
                continue
            
            input_y = rgb_to_ycbcr(result_img)[:, 0, :, :]
            gt_y = rgb_to_ycbcr(gt_img)[:, 0, :, :]
            
            psnr_val = kornia.metrics.psnr(input_y.unsqueeze(1), gt_y.unsqueeze(1), max_val=1.0).item()
            ssim_val = kornia.metrics.ssim(input_y.unsqueeze(1), gt_y.unsqueeze(1), window_size=11, max_val=1.0).mean().item()
            
            total_psnr += psnr_val
            total_ssim += ssim_val
            
            per_image_results.append({
                'image': img_file,
                'psnr': psnr_val,
                'ssim': ssim_val
            })
            
        except Exception as e:
            print(f"\nError: {img_file}: {e}")
            failed += 1
            failed_files.append(img_file)

    img_num = len(result_files)
    if img_num - failed > 0:
        avg_psnr = total_psnr / (img_num - failed)
        avg_ssim = total_ssim / (img_num - failed)
        
        print(f"\n{'='*70}")
        print(f"Results for GTAV-NightRain {args.test_set}:")
        print(f"  Processed: {img_num - failed}/{img_num}")
        if failed > 0:
            print(f"  Failed: {failed}")
        print(f"  PSNR: {avg_psnr:.4f} dB")
        print(f"  SSIM: {avg_ssim:.4f}")
        print(f"{'='*70}")
    
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(result_path, f'evaluation_results_{timestamp}.csv')
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f'# RLP Evaluation Results - GTAV-NightRain {args.test_set}'])
            writer.writerow(['# Model:', args.model_name])
            writer.writerow(['# Date:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
            writer.writerow(['# GT resized to match result dimensions (960×540)'])
            writer.writerow([])
            writer.writerow(['# Summary'])
            writer.writerow(['Total Images', img_num - failed])
            writer.writerow(['Failed', failed])
            writer.writerow(['Average PSNR (dB)', f'{avg_psnr:.4f}'])
            writer.writerow(['Average SSIM', f'{avg_ssim:.4f}'])
            writer.writerow([])
            writer.writerow(['# Per-Image Results'])
            writer.writerow(['Image Name', 'PSNR (dB)', 'SSIM'])
            
            for result in per_image_results:
                writer.writerow([result['image'], f"{result['psnr']:.4f}", f"{result['ssim']:.4f}"])
        
        print(f"\nCSV saved: {csv_path}")

        txt_path = os.path.join(result_path, 'evaluation_results.txt')
        with open(txt_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"GTAV-NightRain {args.test_set} Evaluation Results\n")
            f.write("="*70 + "\n\n")
            f.write(f"Model: {args.model_name}\n")
            f.write(f"GT resized to match result dimensions (960×540)\n\n")
            f.write(f"Processed: {img_num - failed}/{img_num}\n")
            if failed > 0:
                f.write(f"Failed: {failed}\n")
            f.write(f"\nAverage PSNR: {avg_psnr:.4f} dB\n")
            f.write(f"Average SSIM: {avg_ssim:.4f}\n")
        
        print(f"✓ TXT saved: {txt_path}")
        
        if failed_files and len(failed_files) <= 10:
            print(f"\n Failed files: {', '.join(failed_files)}")
    else:
        print("\n All images failed")


if __name__ == "__main__":
    main()