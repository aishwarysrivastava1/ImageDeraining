import argparse
import gc
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

from dataset_rainds import DatasetTest
from models import model_utils
from utils import expand2square


def brutal_memory_cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RainDS deraining - BRUTAL OPTIMIZATION')
    parser.add_argument('--gpus', default='0', type=str)
    parser.add_argument('--variant', default='rainds_real', type=str, choices=['rainds_real', 'rainds_syn'])
    parser.add_argument('--rain_type', default='all', type=str, choices=['all', 'raindrop', 'rainstreak', 'rainstreak_raindrop'])
    
    # TODO: Set path to input directory
    parser.add_argument('--input_dir', default='', type=str)
    
    # TODO: Set path to results directory
    parser.add_argument('--result_dir', default='', type=str)
    
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--model_name', default='UNet_RLP_RPIM', type=str)
    
    # TODO: Set path to model weights
    parser.add_argument('--weights', default='', type=str)
    
    parser.add_argument('--arch', default='UNet', type=str)
    parser.add_argument('--use_rlp', action='store_true', default=True)
    parser.add_argument('--use_rpim', action='store_true', default=True)
    parser.add_argument('--train_ps', type=int, default=256)
    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument('--win_size', type=int, default=8)
    parser.add_argument('--token_projection', type=str, default='linear')
    parser.add_argument('--token_mlp', type=str, default='leff')
    parser.add_argument('--query_embed', action='store_true', default=False)
    parser.add_argument('--dd_in', type=int, default=3)
    
    args = parser.parse_args()
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,expandable_segments:True,roundup_power2_divisions:2"
    
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)
    
    args.batch_size = 1
    
    if not args.use_rlp and 'RLP' in args.model_name:
        args.use_rlp = True
    if not args.use_rpim and 'RPIM' in args.model_name:
        args.use_rpim = True
        
    variant_display = 'RainDS_real' if 'real' in args.variant else 'RainDS_syn'
    
    print("="*70)
    print(f"BRUTAL OPTIMIZATION - {variant_display}")
    print("="*70)
    print(f"Architecture: {args.arch}, RLP: {args.use_rlp}, RPIM: {args.use_rpim}")
    print(f"Rain type: {args.rain_type}, Batch: {args.batch_size}")
    
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"✅ Brutal optimizations: Cleanup every image, Resize to 640×360")
    
    print("="*70)
    brutal_memory_cleanup()
    
    model_restoration = model_utils.get_arch(args)
    checkpoint = torch.load(args.weights, map_location='cpu')
    
    try:
        model_restoration.load_state_dict(checkpoint['state_dict'])
    except:
        try:
            model_restoration.load_state_dict(checkpoint)
        except:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            state_dict = checkpoint.get('state_dict', checkpoint)
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            model_restoration.load_state_dict(new_state_dict)
            
    model_restoration.cuda().eval()
    for p in model_restoration.parameters():
        p.requires_grad = False
        
    test_dataset = DatasetTest(args.input_dir, variant=args.variant, rain_type=args.rain_type)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    print(f"Loaded {len(test_dataset)} images")
    
    result_dir = os.path.join(args.result_dir, args.model_name, variant_display)
    os.makedirs(result_dir, exist_ok=True)
    print(f"✓ Results: {result_dir}\n")
    
    processed, failed = 0, 0
    
    with torch.no_grad():
        for i, (input_img, filename) in enumerate(tqdm(test_loader, desc=f'Testing {variant_display}')):
            try:
                if i % 50 == 0 and torch.cuda.is_available():
                    print(f"\nImage {i}/{len(test_dataset)}: GPU {torch.cuda.memory_allocated(0)/1024**3:.3f}GB")
                    
                input_img = input_img.cuda()

                if 'Uformer' in args.arch:
                    b, c, h, w = input_img.size()
                    input_padded, mask = expand2square(input_img, factor=128)
                    restored, _ = model_restoration(input_padded)
                    restored = torch.masked_select(restored, mask.bool()).reshape(b, c, h, w)
                    del input_padded, mask
                else:
                    restored, _ = model_restoration(input_img)
                
                restored = torch.clamp(restored, 0, 1)
                restored_cpu = restored.detach().cpu().numpy().copy()
                del input_img, restored
                brutal_memory_cleanup()
                
                restored_cpu = restored_cpu.transpose(0, 2, 3, 1)
                for batch in range(len(restored_cpu)):
                    restored_img = np.uint8(restored_cpu[batch] * 255)
                    restored_img = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)
                    save_path = os.path.join(result_dir, filename[batch] + '.png')
                    cv2.imwrite(save_path, restored_img)
                
                del restored_cpu, restored_img
                processed += 1
                
            except torch.cuda.OutOfMemoryError as e:
                failed += 1
                print(f"\n❌ OOM at {i}: {e}")
                brutal_memory_cleanup()
            except Exception as e:
                failed += 1
                print(f"\n❌ Error at {i}: {e}")
                brutal_memory_cleanup()
    
    brutal_memory_cleanup()
    
    print("\n" + "="*70)
    print(f"Testing completed!")
    print(f"Processed: {processed}, Failed: {failed}, Total: {len(test_dataset)}")
    print(f"Results: {result_dir}")
    print("="*70)
    print(f"\n✓ Next: python evaluate_rainds.py --variant {args.variant}")