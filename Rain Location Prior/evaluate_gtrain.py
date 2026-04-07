import os

import kornia
import torch
from torchvision.io import read_image
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


def find_gt_path(result_filename, gt_dir):
    result_name = result_filename.replace('.png', '').replace('.jpg', '')
    parts = result_name.split('_')
    
    scene_name = None
    scene_path = None
    
    for i in range(len(parts)):
        potential_scene = '_'.join(parts[:i+1])
        potential_path = os.path.join(gt_dir, potential_scene)
        if os.path.exists(potential_path):
            scene_name = potential_scene
            scene_path = potential_path
    
    if scene_name is None:
        raise ValueError(f"Could not find scene directory for {result_filename}")

    all_files = os.listdir(scene_path)
    clean_files = [f for f in all_files if '-C-' in f and f.endswith('.png')]
    
    if len(clean_files) == 0:
        raise ValueError(f"No clean frames found in {scene_path}")
        
    scene_prefix = scene_name + '_'
    if result_name.startswith(scene_prefix):
        pattern_with_frame = result_name[len(scene_prefix):]
    else:
        pattern_with_frame = result_name.split(scene_name + '_', 1)[-1] if scene_name + '_' in result_name else result_name

    if '-R-' in pattern_with_frame:
        gt_pattern = pattern_with_frame.replace('-R-', '-C-')
        gt_filename = f"{gt_pattern}.png"
        gt_path = os.path.join(scene_path, gt_filename)
        
        if os.path.exists(gt_path):
            return gt_path
            
        base_pattern = pattern_with_frame.split('-R-')[0]
        gt_filename = f"{base_pattern}-C-000.png"
        gt_path = os.path.join(scene_path, gt_filename)
        
        if os.path.exists(gt_path):
            return gt_path

    if len(clean_files) > 1:
        if '-R-' in result_name:
            frame_num = result_name.split('-R-')[-1]
            possible_patterns = [
                f"{scene_name}-C-{frame_num}.png",
                f"{scene_name}-Webcam-C-{frame_num}.png",
            ]
            for pattern in possible_patterns:
                gt_path = os.path.join(scene_path, pattern)
                if os.path.exists(gt_path):
                    return gt_path

    if clean_files:
        gt_path = os.path.join(scene_path, clean_files[0])
        if os.path.exists(gt_path):
            return gt_path
    
    raise ValueError(f"Could not find matching GT for {result_filename} in {scene_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # TODO: Set path to root evaluation directory
    root_dir = ''
    
    # TODO: Set path to GT-Rain ground truth directory
    gt_dir = ''
    
    datasets = ['']
    methods = ['UNet_RLP_RPIM']

    for dataset in datasets:
        print(f"\nDataset: {dataset if dataset else 'GT-Rain'}")
        
        for method in methods:
            print(f"\n{'='*70}")
            print(f"Evaluating: {method}")
            print(f"{'='*70}")
            
            if dataset:
                file_path = os.path.join(root_dir, dataset, method)
            else:
                file_path = os.path.join(root_dir, method)
                
            if not os.path.exists(file_path):
                print(f"Results directory not found: {file_path}")
                print(f"Make sure you ran test.py first!")
                continue
                
            image_files = [f for f in os.listdir(file_path) 
                          if f.endswith(('.jpg', '.png'))]
            
            if len(image_files) == 0:
                print(f"❌ No result images found in {file_path}")
                continue

            total_psnr = 0.0
            total_ssim = 0.0
            img_num = len(image_files)
            failed = 0
            failed_files = []

            print(f"Processing {img_num} images...")
            print(f"GT directory: {gt_dir}")
            print(f"Device: {device}")
            
            for img_file in tqdm(image_files, desc=f'Evaluating {method}'):
                try:
                    result_path = os.path.join(file_path, img_file)
                    input_img = read_image(result_path).float().unsqueeze(0).to(device)

                    gt_path = find_gt_path(img_file, gt_dir)
                    gt_img = read_image(gt_path).float().unsqueeze(0).to(device)

                    input_y = rgb_to_ycbcr(input_img)[:, 0, :, :]
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
                    
                except Exception as e:
                    print(f"\nError processing {img_file}: {e}")
                    failed += 1
                    failed_files.append(img_file)
                    continue

            if img_num - failed > 0:
                avg_psnr = total_psnr / (img_num - failed)
                avg_ssim = total_ssim / (img_num - failed)
                
                print(f"\n{'='*70}")
                print(f"Results for {method}:")
                print(f"{'='*70}")
                print(f"  Processed: {img_num - failed}/{img_num} images")
                if failed > 0:
                    print(f"  Failed: {failed} images")
                print(f"  PSNR: {avg_psnr:.4f} dB")
                print(f"  SSIM: {avg_ssim:.4f}")
                print(f"{'='*70}")
                
                results_file = os.path.join(file_path, 'evaluation_results.txt')
                with open(results_file, 'w') as f:
                    f.write("="*70 + "\n")
                    f.write(f"GT-Rain Evaluation Results\n")
                    f.write("="*70 + "\n\n")
                    f.write(f"Model: {method}\n")
                    f.write(f"Results directory: {file_path}\n")
                    f.write(f"GT directory: {gt_dir}\n\n")
                    f.write(f"Processed: {img_num - failed}/{img_num} images\n")
                    if failed > 0:
                        f.write(f"Failed: {failed} images\n")
                    f.write(f"\nAverage PSNR: {avg_psnr:.4f} dB\n")
                    f.write(f"Average SSIM: {avg_ssim:.4f}\n\n")
                    
                    if failed_files:
                        f.write("Failed files:\n")
                        for ff in failed_files:
                            f.write(f"  - {ff}\n")
                
                print(f"\n Results saved to: {results_file}")
                
                if failed_files and len(failed_files) <= 10:
                    print(f"\nFailed files:")
                    for ff in failed_files:
                        print(f"    - {ff}")
                elif failed_files:
                    print(f"\n{len(failed_files)} files failed (see evaluation_results.txt)")
                
            else:
                print(f"\nAll images failed for {method}")
                print(f"   Check that:")
                print(f"   1. Result filenames match GT-Rain format")
                print(f"   2. GT directory path is correct: {gt_dir}")
                print(f"   3. Scene folders exist in GT directory")


if __name__ == "__main__":
    main()