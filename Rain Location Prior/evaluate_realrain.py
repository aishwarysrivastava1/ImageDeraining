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
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    return img_tensor


def main():
    parser = argparse.ArgumentParser(description='Evaluate RealRain results')
    parser.add_argument('--variant', default='realrain_h', type=str,
                        choices=['realrain_h', 'realrain_l', 'synrain'])
    
    # TODO: Set path to results directory
    parser.add_argument('--result_dir', default='', type=str)
    
    # TODO: Set path to RealRain ground truth base directory
    parser.add_argument('--gt_dir', default='', type=str)
    
    parser.add_argument('--model_name', default='UNet_RLP_RPIM', type=str)
    
    # TODO: Set path to CSV save directory
    parser.add_argument('--csv_dir', default='', type=str)

    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    variant_names = {
        'realrain_h': 'RealRain-1k-H',
        'realrain_l': 'RealRain-1k-L',
        'synrain': 'SynRain-13k'
    }
    variant_display = variant_names.get(args.variant, args.variant)
    
    print("="*70)
    print(f"Evaluating: {variant_display}")
    print("="*70)

    result_path = os.path.join(args.result_dir, args.model_name, variant_display)
    
    if not os.path.exists(result_path):
        print(f"❌ Results directory not found: {result_path}")
        print(f"   Make sure you ran test_realrain.py first!")
        return

    if args.variant == 'realrain_h':
        gt_path = os.path.join(args.gt_dir, 'RealRain-1k', 'RealRain-1k-H', 'test', 'target')
    elif args.variant == 'realrain_l':
        gt_path = os.path.join(args.gt_dir, 'RealRain-1k', 'RealRain-1k-L', 'test', 'target')
    elif args.variant == 'synrain':
        gt_path = os.path.join(args.gt_dir, 'SynRain-13k', 'test', 'target')
    else:
        print(f"Unknown variant: {args.variant}")
        return
    
    if not os.path.exists(gt_path):
        print(f"Ground truth directory not found: {gt_path}")
        return

    result_files = sorted([f for f in os.listdir(result_path) if f.endswith('.png')])
    
    if len(result_files) == 0:
        print(f"No result images found in {result_path}")
        return
    
    print(f"Results directory: {result_path}")
    print(f"GT directory: {gt_path}")
    print(f"Device: {device}")
    print(f"Processing {len(result_files)} images...")
    print(f"GT images will be resized to match result dimensions")
    
    total_psnr = 0.0
    total_ssim = 0.0
    img_num = len(result_files)
    failed = 0
    failed_files = []
    per_image_results = []
    
    for img_file in tqdm(result_files, desc=f'Evaluating {variant_display}'):
        try:
            result_img_path = os.path.join(result_path, img_file)
            result_img = load_image_as_tensor(result_img_path, device)
            
            _, _, result_h, result_w = result_img.shape

            gt_img_path = os.path.join(gt_path, img_file)
            if not os.path.exists(gt_img_path):
                print(f"\n GT not found: {img_file}")
                failed += 1
                failed_files.append(img_file)
                continue

            gt_img = load_image_as_tensor(gt_img_path, device, target_size=(result_w, result_h))

            if result_img.shape != gt_img.shape:
                print(f"\n Dimension mismatch after resize: {img_file}")
                print(f"    Result: {result_img.shape}, GT: {gt_img.shape}")
                failed += 1
                failed_files.append(img_file)
                continue

            input_y = rgb_to_ycbcr(result_img)[:, 0, :, :]
            gt_y = rgb_to_ycbcr(gt_img)[:, 0, :, :]
            
            psnr_val = kornia.metrics.psnr(
                input_y.unsqueeze(1),
                gt_y.unsqueeze(1),
                max_val=1.0
            )
            total_psnr += psnr_val.item()
            
            ssim_val = kornia.metrics.ssim(
                input_y.unsqueeze(1),
                gt_y.unsqueeze(1),
                window_size=11,
                max_val=1.0
            ).mean()
            total_ssim += ssim_val.item()

            per_image_results.append({
                'image_name': img_file,
                'psnr_db': psnr_val.item(),
                'ssim': ssim_val.item()
            })
        
        except Exception as e:
            print(f"\nError processing {img_file}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            failed_files.append(img_file)
            continue

    if img_num - failed > 0:
        avg_psnr = total_psnr / (img_num - failed)
        avg_ssim = total_ssim / (img_num - failed)
        
        print(f"\n{'='*70}")
        print(f"Results for {variant_display}:")
        print(f"{'='*70}")
        print(f"  Processed: {img_num - failed}/{img_num} images")
        if failed > 0:
            print(f"  Failed: {failed} images")
        print(f"  PSNR: {avg_psnr:.4f} dB")
        print(f"  SSIM: {avg_ssim:.4f}")
        print(f"{'='*70}")

        results_file = os.path.join(result_path, 'evaluation_results.txt')
        with open(results_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"{variant_display} Evaluation Results\n")
            f.write("="*70 + "\n\n")
            f.write(f"Model: {args.model_name}\n")
            f.write(f"Results directory: {result_path}\n")
            f.write(f"GT directory: {gt_path}\n")
            f.write(f"Note: GT images resized to match result dimensions\n\n")
            f.write(f"Processed: {img_num - failed}/{img_num} images\n")
            if failed > 0:
                f.write(f"Failed: {failed} images\n")
            f.write(f"\nAverage PSNR: {avg_psnr:.4f} dB\n")
            f.write(f"Average SSIM: {avg_ssim:.4f}\n\n")
            
            if failed_files:
                f.write("Failed files:\n")
                for ff in failed_files:
                    f.write(f"  - {ff}\n")
        
        print(f"\nResults saved to: {results_file}")
        
        if args.csv_dir:
            os.makedirs(args.csv_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_filename = f'{args.model_name}_{variant_display}_results_{timestamp}.csv'
            csv_path = os.path.join(args.csv_dir, csv_filename)

            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([f'# {variant_display} Evaluation Results'])
                writer.writerow(['# Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                writer.writerow(['# Note: GT images resized to match result dimensions'])
                writer.writerow([])
                writer.writerow(['# Model Information'])
                writer.writerow(['Model', args.model_name])
                writer.writerow(['Test Dataset', variant_display])
                writer.writerow(['Results Directory', result_path])
                writer.writerow(['GT Directory', gt_path])
                writer.writerow([])
                writer.writerow(['# Overall Metrics'])
                writer.writerow(['Total Images', img_num])
                writer.writerow(['Processed Images', img_num - failed])
                writer.writerow(['Failed Images', failed])
                writer.writerow(['Average PSNR (dB)', f'{avg_psnr:.4f}'])
                writer.writerow(['Average SSIM', f'{avg_ssim:.4f}'])
                writer.writerow([])
                writer.writerow(['# Per-Image Results'])
                writer.writerow(['Image Name', 'PSNR (dB)', 'SSIM'])

                for result in per_image_results:
                    writer.writerow([
                        result['image_name'],
                        f"{result['psnr_db']:.4f}",
                        f"{result['ssim']:.4f}"
                    ])

            print(f" CSV results saved to: {csv_path}")

        if failed_files and len(failed_files) <= 10:
            print(f"\n Failed files:")
            for ff in failed_files:
                print(f"    - {ff}")
        elif failed_files:
            print(f"\n{len(failed_files)} files failed (see evaluation_results.txt)")
    
    else:
        print(f"\nAll images failed")


if __name__ == "__main__":
    main()