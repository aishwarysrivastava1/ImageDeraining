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
from collections import defaultdict
import gc

import options.options as option
from utils import util
from models import create_model

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data'))
from data.Dataset_rainds import Dataset_RainDS


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


def brutal_memory_cleanup():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--csv_dir', type=str, default='./results/csv', 
                       help='Directory to save CSV results')
    parser.add_argument('--model_path', type=str, default='',
                       help='Path to pretrained model')
    args = parser.parse_args()
    
    opt = option.parse(args.opt, is_train=False)
    opt['dist'] = False
    opt['path']['pretrain_model_G'] = args.model_path

    os.makedirs(args.csv_dir, exist_ok=True)

    if 'log' in opt['path'] and opt['path']['log']:
        os.makedirs(opt['path']['log'], exist_ok=True)

    util.setup_logger('base', opt['path']['log'], 'test_rainds_brutal', 
                     level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))

    opt = option.dict_to_nonedict(opt)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,expandable_segments:True,roundup_power2_divisions:2"

    torch.set_grad_enabled(False)
    torch.autograd.set_grad_enabled(False)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        gc.collect()

        logger.info(f'GPU Device: {torch.cuda.get_device_name(0)}')
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f'Total GPU Memory: {total_mem:.2f} GB')

        free_mem = torch.cuda.mem_get_info()[0] / 1024**3
        logger.info(f'Free GPU Memory: {free_mem:.2f} GB')

    logger.info('Creating dataset...')
    test_dataset = Dataset_RainDS(opt['datasets']['test_rainds'])
    
    logger.info(f'Number of test images: {len(test_dataset)}')
    logger.info('Creating model...')
    model = create_model(opt)
   
    if hasattr(model, 'netG'):
        model.netG.eval()
        for param in model.netG.parameters():
            param.requires_grad = False
        for module in model.netG.modules():
            if hasattr(module, 'training'):
                module.training = False

    dataset_variant = test_dataset.dataset_name
    logger.info(f'Testing on: {dataset_variant}')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f'NDR_{dataset_variant}_results_{timestamp}.csv'
    csv_path = os.path.join(args.csv_dir, csv_filename)
  
    per_image_results = []
    rain_type_metrics = defaultdict(lambda: {'psnr': [], 'ssim': []})
    
    logger.info('Starting BRUTAL memory-optimized testing...')
    logger.info(f'Model path: {args.model_path}')
    logger.info('Memory cleanup after EVERY image')

    failed_count = 0
    success_count = 0

    for i in tqdm(range(len(test_dataset)), desc=f'Testing {dataset_variant}'):
        try:
            if i % 10 == 0 and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                logger.info(f'Image {i}: GPU Allocated: {allocated:.3f}GB, Reserved: {reserved:.3f}GB')

            data = test_dataset[i]
            rain_type = data['rain_type']

            lq_tensor = data['lq'].unsqueeze(0)
            gt_tensor = data['gt'].unsqueeze(0)
            
            val_data = {
                'lq': lq_tensor,
                'gt': gt_tensor
            }

            model.feed_data_test(val_data)
            model.test()
            
            visuals = model.get_current_visuals()
         
            out_img_cpu = visuals['out_img'].detach().cpu().numpy().copy()
            gt_img_cpu = visuals['gt_img'].detach().cpu().numpy().copy()

            del visuals
            del val_data
            del lq_tensor
            del gt_tensor

            brutal_memory_cleanup()

            c, h, w = gt_img_cpu.shape
            out_img_cpu = out_img_cpu[:c, :h, :w]

            gt_img_cpu = np.clip(gt_img_cpu, 0, 1)
            out_img_cpu = np.clip(out_img_cpu, 0, 1)
        
            psnr_val = compute_psnr(out_img_cpu, gt_img_cpu, 1.0)
            ssim_val = compute_ssim(out_img_cpu * 255, gt_img_cpu * 255)

            rain_type_metrics[rain_type]['psnr'].append(psnr_val)
            rain_type_metrics[rain_type]['ssim'].append(ssim_val)
            rain_type_metrics['overall']['psnr'].append(psnr_val)
            rain_type_metrics['overall']['ssim'].append(ssim_val)

            lq_filename = Path(data['lq_path']).name
            gt_filename = Path(data['gt_path']).name
            
            per_image_results.append({
                'image_name': lq_filename,
                'gt_name': gt_filename,
                'rain_type': rain_type,
                'psnr_db': psnr_val,
                'ssim': ssim_val
            })
            
            success_count += 1

            del out_img_cpu, gt_img_cpu
            del data

            brutal_memory_cleanup()
            
        except torch.cuda.OutOfMemoryError as e:
            failed_count += 1
            logger.error(f'CUDA OOM at image {i}: {e}')
            logger.error(f'Image: {test_dataset.lq_paths[i]}')
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
            gc.collect()
            continue
            
        except Exception as e:
            failed_count += 1
            logger.error(f'Error processing image {i}: {e}')
            brutal_memory_cleanup()
            continue

    brutal_memory_cleanup()
    logger.info(f'\n{"="*70}')
    logger.info(f'Processing Summary:')
    logger.info(f'  Successfully processed: {success_count}')
    logger.info(f'  Failed: {failed_count}')
    logger.info(f'  Total: {len(test_dataset)}')
    logger.info(f'{"="*70}')

    results_summary = {}
    for rain_type in ['raindrop', 'rainstreak', 'rainstreak_raindrop', 'overall']:
        if rain_type in rain_type_metrics and len(rain_type_metrics[rain_type]['psnr']) > 0:
            avg_psnr = sum(rain_type_metrics[rain_type]['psnr']) / len(rain_type_metrics[rain_type]['psnr'])
            avg_ssim = sum(rain_type_metrics[rain_type]['ssim']) / len(rain_type_metrics[rain_type]['ssim'])
            count = len(rain_type_metrics[rain_type]['psnr'])
            
            results_summary[rain_type] = {
                'psnr': avg_psnr,
                'ssim': avg_ssim,
                'count': count
            }

    logger.info(f'Saving results to {csv_path}...')
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header
        writer.writerow([f'# NDR-Restore {dataset_variant} Evaluation Results'])
        writer.writerow([])
        
        # Model info
        writer.writerow(['# Model Information'])
        writer.writerow(['Model', 'NDR-Restore'])
        writer.writerow(['Pretrained Model', args.model_path])
        writer.writerow(['Test Dataset', dataset_variant])
        writer.writerow([])
        
        # Overall metrics
        writer.writerow(['# Overall Metrics Summary'])
        writer.writerow(['Rain Type', 'Image Count', 'Average PSNR (dB)', 'Average SSIM'])
        
        for rain_type in ['raindrop', 'rainstreak', 'rainstreak_raindrop', 'overall']:
            if rain_type in results_summary:
                r = results_summary[rain_type]
                writer.writerow([
                    rain_type.replace('_', ' ').title(),
                    r['count'],
                    f"{r['psnr']:.4f}",
                    f"{r['ssim']:.4f}"
                ])
        
        writer.writerow([])
        
        # Per-image results
        writer.writerow(['# Per-Image Results'])
        writer.writerow(['Image Name', 'GT Name', 'Rain Type', 'PSNR (dB)', 'SSIM'])
        
        for result in per_image_results:
            writer.writerow([
                result['image_name'],
                result['gt_name'],
                result['rain_type'],
                f"{result['psnr_db']:.4f}",
                f"{result['ssim']:.4f}"
            ])

    logger.info('='*70)
    logger.info(f'{dataset_variant} Test Results:')
    logger.info('='*70)
    
    for rain_type in ['raindrop', 'rainstreak', 'rainstreak_raindrop', 'overall']:
        if rain_type in results_summary:
            r = results_summary[rain_type]
            logger.info(f'{rain_type.replace("_", " ").title()}:')
            logger.info(f'  Images: {r["count"]}')
            logger.info(f'  PSNR: {r["psnr"]:.4f} dB')
            logger.info(f'  SSIM: {r["ssim"]:.4f}')
    
    logger.info('='*70)
    logger.info(f'Results saved to: {csv_path}')


if __name__ == '__main__':
    main()