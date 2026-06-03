import argparse
import csv
import os
from datetime import datetime
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from dataset_realrain import DatasetTest
def main():
    parser = argparse.ArgumentParser(description='Evaluate NeRD on RealRain')
    parser.add_argument('--result_dir', default='./results/NeRD/RealRain', type=str)
    parser.add_argument('--gt_dir', default='./data', type=str)
    parser.add_argument('--csv_dir', default='./results/NeRD', type=str)
    args = parser.parse_args()
    os.makedirs(args.csv_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    variant_info = [
        ('realrain_h', 'RealRain-1k-H'),
        ('realrain_l', 'RealRain-1k-L'),
        ('synrain', 'SynRain'),
    ]
    for variant, variant_dir in variant_info:
        csv_path = os.path.join(args.csv_dir, f'NeRD_{variant_dir}_results_{timestamp}.csv')
        per_image_results = []
        avg_psnr = 0.0
        avg_ssim = 0.0
        valid_count = 0
        dataset = DatasetTest(args.gt_dir, variant=variant)
        variant_result_dir = os.path.join(args.result_dir, variant_dir)
        for idx in range(len(dataset)):
            img_path, name = dataset.samples[idx]
            gt_path = img_path.replace('input', 'target')
            res_path = os.path.join(variant_result_dir, name + '.png')
            if not os.path.exists(res_path) or not os.path.exists(gt_path):
                continue
            gt_img = cv2.imread(gt_path)
            res_img = cv2.imread(res_path)
            height, width = res_img.shape[:2]
            gt_img = cv2.resize(gt_img, (width, height))
            p = psnr(gt_img, res_img)
            s = ssim(gt_img, res_img, channel_axis=2)
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
            writer.writerow(['Average', f'{avg_psnr:.4f}', f'{avg_ssim:.4f}'])
        print(f'{variant_dir} Evaluation Complete. Average PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}')
        print(f'Results saved to {csv_path}')
if __name__ == '__main__':
    main()
