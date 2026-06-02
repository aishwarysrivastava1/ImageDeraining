import os
import argparse
import csv
from datetime import datetime
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from dataset_gtrain import DatasetTest

def main():
    parser = argparse.ArgumentParser(description='Evaluate MFDNet on GTRain')
    parser.add_argument('--result_dir', default='./results/MFDNet/GTRain', type=str, help='Directory of generated images')
    parser.add_argument('--gt_dir', default='d:/Deraining/RLP/rlp/data/GT-RAIN_test', type=str, help='Directory of GT images')
    parser.add_argument('--csv_dir', default='./results/MFDNet', type=str, help='Directory to save CSV')
    args = parser.parse_args()

    os.makedirs(args.csv_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(args.csv_dir, f'MFDNet_GTRain_results_{timestamp}.csv')

    dataset = DatasetTest(args.gt_dir)
    
    per_image_results = []
    avg_psnr = 0.0
    avg_ssim = 0.0
    valid_count = 0
    missing_gt = 0
    missing_result = 0

    for idx in range(len(dataset)):
        # GTRain DatasetTest returns rainy_tensor, name
        # We need the GT image for evaluation
        path, name = dataset.samples[idx]
        
        # Ground truth is named with -C- instead of -R-
        scene_dir = os.path.dirname(path)
        base_name = os.path.basename(path)
        
        gt_name = base_name.replace('-R-', '-C-')
        gt_path = os.path.join(scene_dir, gt_name)
        if not os.path.exists(gt_path):
            # Try single clean file
            gt_name = base_name.split('-R-')[0] + '-C-000.png'
            gt_path = os.path.join(scene_dir, gt_name)
            if not os.path.exists(gt_path):
                missing_gt += 1
                print(f"GT not found for rainy input: {path}")
                continue
                
        res_path = os.path.join(args.result_dir, name + '.png')
        if not os.path.exists(res_path):
            missing_result += 1
            print(f"Result not found for {name}")
            continue
            
        gt_img = cv2.imread(gt_path)
        res_img = cv2.imread(res_path)
        
        # Ensure sizes match
        h, w = res_img.shape[:2]
        gt_img = cv2.resize(gt_img, (w, h))

        p = psnr(gt_img, res_img)
        s = ssim(gt_img, res_img, multichannel=True, channel_axis=2)

        avg_psnr += p
        avg_ssim += s
        valid_count += 1
        
        per_image_results.append({
            'name': name,
            'psnr': p,
            'ssim': s
        })

    if valid_count > 0:
        avg_psnr /= valid_count
        avg_ssim /= valid_count

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Image Name', 'PSNR (dB)', 'SSIM'])
        for res in per_image_results:
            writer.writerow([res['name'], f"{res['psnr']:.4f}", f"{res['ssim']:.4f}"])
        writer.writerow([])
        writer.writerow(['Average', f"{avg_psnr:.4f}", f"{avg_ssim:.4f}"])

    print(f"GTRain Evaluation Complete. Average PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
    print(f"Matched pairs: {valid_count}, Missing GT: {missing_gt}, Missing Results: {missing_result}")
    print(f"Results saved to {csv_path}")

if __name__ == '__main__':
    main()
