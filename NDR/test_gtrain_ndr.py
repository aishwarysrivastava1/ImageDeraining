import os
import argparse
import logging
import torch
import cv2
import numpy as np
import csv
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import gc

import options.options as option
from utils import util
from models import create_model

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data'))
from data.Dataset_gtrain import Dataset_GTRain

def rgb_to_ycbcr(img):
    if len(img.shape) == 3:
        img = np.transpose(img, (1, 2, 0))
    Y = 16 + (65.481 * img[:, :, 0] + 128.553 * img[:, :, 1] + 24.966 * img[:, :, 2]) / 255.0
    
    return Y


def compute_psnr(img1, img2, peak=1.0):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(peak * peak / mse)


def compute_ssim(img1, img2):
    ssims = []
    for i in range(3):
        ssims.append(_ssim(img1[i], img2[i]))
    return np.array(ssims).mean()


def _ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--csv_dir', type=str, default='./results/csv', 
                       help='Directory to save CSV results')
    args = parser.parse_args()
    
    opt = option.parse(args.opt, is_train=False)
    opt['dist'] = False
    os.makedirs(args.csv_dir, exist_ok=True)
    
    if 'log' in opt['path'] and opt['path']['log']:
        os.makedirs(opt['path']['log'], exist_ok=True)

    util.setup_logger('base', opt['path']['log'], 'test_gtrain', 
                     level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))
    opt = option.dict_to_nonedict(opt)

    torch.backends.cudnn.benchmark = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    logger.info('Creating dataset...')
    test_dataset = Dataset_GTRain(opt['datasets']['test_gtrain'])
    
    logger.info(f'Number of test images: {len(test_dataset)}')

    logger.info('Creating model...')
    model = create_model(opt)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f'NDR_GTRain_results_{timestamp}.csv'
    csv_path = os.path.join(args.csv_dir, csv_filename)
    per_image_results = []
    
    logger.info('Starting testing...')
    avg_psnr = 0.0
    avg_ssim = 0.0
    idx = 0
    
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset)), desc='Testing GT-Rain'):
            if i % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            data = test_dataset[i]
            
            val_data = {
                'lq': data['lq'].unsqueeze(0),
                'gt': data['gt'].unsqueeze(0)
            }

            model.feed_data_test(val_data)
            model.test()
            
            visuals = model.get_current_visuals()
            out_img = visuals['out_img'].cpu().numpy()
            gt_img = visuals['gt_img'].cpu().numpy()

            c, h, w = gt_img.shape
            out_img = out_img[:c, :h, :w]

            gt_img = np.clip(gt_img, 0, 1)
            out_img = np.clip(out_img, 0, 1)

            psnr_val = compute_psnr(out_img, gt_img, 1.0)
            ssim_val = compute_ssim(out_img * 255, gt_img * 255)

            avg_psnr += psnr_val
            avg_ssim += ssim_val
            idx += 1

            lq_filename = Path(data['lq_path']).name
            gt_filename = Path(data['gt_path']).name
            
            per_image_results.append({
                'image_name': lq_filename,
                'gt_name': gt_filename,
                'psnr_db': psnr_val,
                'ssim': ssim_val
            })

            del out_img, gt_img, visuals, val_data
            
            if i % 10 == 0:
                gc.collect()

    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx
    logger.info(f'Saving results to {csv_path}...')
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header
        writer.writerow(['# NDR-Restore GT-Rain Evaluation Results'])
        writer.writerow([])

        # Model info
        writer.writerow(['# Model Information'])
        writer.writerow(['Model', 'NDR-Restore'])
        writer.writerow(['Test Dataset', 'GT-Rain'])
        writer.writerow([])
        
        # Overall metrics
        writer.writerow(['# Overall Metrics'])
        writer.writerow(['Total Images', idx])
        writer.writerow(['Average PSNR (dB)', f'{avg_psnr:.4f}'])
        writer.writerow(['Average SSIM', f'{avg_ssim:.4f}'])
        writer.writerow([])
        
        # Per-image results
        writer.writerow(['# Per-Image Results'])
        writer.writerow(['Image Name', 'GT Name', 'PSNR (dB)', 'SSIM'])
        
        for result in per_image_results:
            writer.writerow([
                result['image_name'],
                result['gt_name'],
                f"{result['psnr_db']:.4f}",
                f"{result['ssim']:.4f}"
            ])
            
    logger.info('='*70)
    logger.info('GT-Rain Test Results:')
    logger.info('='*70)
    logger.info(f'Total Images: {idx}')
    logger.info(f'Average PSNR: {avg_psnr:.4f} dB')
    logger.info(f'Average SSIM: {avg_ssim:.4f}')
    logger.info('='*70)
    logger.info(f'Results saved to: {csv_path}')


if __name__ == '__main__':
    main()