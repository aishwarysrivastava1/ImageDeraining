import os
import argparse
import csv
from datetime import datetime
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from dataset_gtav import DatasetTest

def get_gt_filename(input_name):
    stem, ext = os.path.splitext(input_name)
    if stem.endswith('_00'):
        stem = stem[:-3]
    return stem + ext

def main():
    parser = argparse.ArgumentParser(description='Evaluate MFDNet on GTAV')
    parser.add_argument('--result_dir', default='./results/MFDNet/GTAV', type=str, help='Directory of generated images')
    parser.add_argument('--gt_dir', default='d:/Deraining/RLP/rlp/data/GTAV-NightRain', type=str, help='Directory of GT images')
    parser.add_argument('--csv_dir', default='./results/MFDNet', type=str, help='Directory to save CSV')
    args = parser.parse_args()

    os.makedirs(args.csv_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for test_set in ['set1', 'set2', 'set3']:
        csv_path = os.path.join(args.csv_dir, f'MFDNet_GTAV_{test_set}_results_{timestamp}.csv')
        per_image_results = []
        avg_psnr = 0.0
        avg_ssim = 0.0
        valid_count = 0
        dataset = DatasetTest(args.gt_dir, test_set=test_set)
        set_result_dir = os.path.join(args.result_dir, test_set)
        
        for idx in range(len(dataset)):
            img_path, name = dataset.samples[idx]
            
            img_dir, img_file = os.path.split(img_path)
            gt_dir = os.path.join(os.path.dirname(img_dir), 'target')
            gt_path = os.path.join(gt_dir, get_gt_filename(img_file))
            res_path = os.path.join(set_result_dir, name + '.png')
            
            if not os.path.exists(res_path) or not os.path.exists(gt_path):
                continue
                
            gt_img = cv2.imread(gt_path)
            res_img = cv2.imread(res_path)
            
            h, w = res_img.shape[:2]
            gt_img = cv2.resize(gt_img, (w, h))

            p = psnr(gt_img, res_img)
            s = ssim(gt_img, res_img, multichannel=True, channel_axis=2)

            avg_psnr += p
            avg_ssim += s
            valid_count += 1
            
            per_image_results.append({'name': name, 'psnr': p, 'ssim': s})

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

        print(f"GTAV {test_set} Evaluation Complete. Average PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
        print(f"Results saved to {csv_path}")

if __name__ == '__main__':
    main()
