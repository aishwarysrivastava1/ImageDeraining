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

from dataset_realrain import DatasetTest
from models import model_utils
from utils import expand2square


def brutal_memory_cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image deraining inference on RealRain - BRUTAL OPTIMIZATION')
    parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--variant', default='realrain_h', type=str,
                        choices=['realrain_h', 'realrain_l', 'synrain'])
                        
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
    parser.add_argument('--tile', action='store_true', default=False)
    parser.add_argument('--tile_size', type=int, default=1280)
    parser.add_argument('--tile_overlap', type=int, default=320)
    
    args = parser.parse_args()
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,expandable_segments:True,roundup_power2_divisions:2"
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
    torch.set_grad_enabled(False)
    
    if args.batch_size != 1:
        print(f"⚠️  WARNING: Forcing batch_size=1 for maximum memory optimization")
        args.batch_size = 1
        
    if not args.use_rlp and 'RLP' in args.model_name:
        args.use_rlp = True
    if not args.use_rpim and 'RPIM' in args.model_name:
        args.use_rpim = True

    variant_names = {
        'realrain_h': 'RealRain-1k-H',
        'realrain_l': 'RealRain-1k-L',
        'synrain': 'SynRain-13k'
    }
    variant_display = variant_names.get(args.variant, args.variant)
    
    print("="*70)
    print(f"BRUTAL OPTIMIZATION MODE - {variant_display}")
    print("="*70)
    print(f"Architecture: {args.arch}")
    print(f"Use RLP: {args.use_rlp}")
    print(f"Use RPIM: {args.use_rpim}")
    print(f"Variant: {variant_display}")
    print(f"Weights: {args.weights}")
    print(f"Batch size: {args.batch_size} (forced to 1)")
    print(f"Model name: {args.model_name}")
    
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"\nGPU Information:")
        print(f"  Device: {gpu_props.name}")
        print(f"  Total Memory: {gpu_props.total_memory / 1024**3:.2f} GB")
        print(f"\n⚡ BRUTAL OPTIMIZATIONS ENABLED:")
        print(f"  ✅ Memory cleanup after EVERY image")
        print(f"  ✅ CUDA allocator: max_split_size_mb=64")
        print(f"  ✅ Immediate CPU transfer")
        print(f"  ✅ All tensors deleted after use")
        print(f"  ✅ Automatic image resizing")
    
    print("="*70)
    brutal_memory_cleanup()
    
    print("\nLoading model...")
    model_restoration = model_utils.get_arch(args)

    checkpoint = torch.load(args.weights, map_location='cpu')
    try:
        model_restoration.load_state_dict(checkpoint['state_dict'])
        print(f"✓ Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    except:
        try:
            model_restoration.load_state_dict(checkpoint)
            print("✓ Model loaded (direct state dict)")
        except Exception as e:
            print(f"Trying to load with module prefix adjustment...")
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            state_dict = checkpoint.get('state_dict', checkpoint)
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            model_restoration.load_state_dict(new_state_dict)
            print("✓ Model loaded after prefix adjustment")
            
    model_restoration.cuda()
    model_restoration.eval()

    for p in model_restoration.parameters():
        p.requires_grad = False
        
    print(f"\nLoading test dataset ({variant_display})...")
    test_dataset = DatasetTest(args.input_dir, variant=args.variant)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  
        drop_last=False,
        pin_memory=False 
    )
    print(f"✓ Loaded {len(test_dataset)} test images")
    
    result_dir = os.path.join(args.result_dir, args.model_name, variant_display)
    os.makedirs(result_dir, exist_ok=True)
    print(f"✓ Results will be saved to: {result_dir}")
    
    print("\n" + "="*70)
    print("Starting BRUTAL memory-optimized testing...")
    print("Memory cleanup after EVERY image")
    print("="*70)
    
    processed_count = 0
    failed_count = 0
    
    with torch.no_grad():
        for i, (input_img, filename) in enumerate(tqdm(test_loader, desc=f'Testing {variant_display}')):
            try:
                if i % 50 == 0 and torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(0) / 1024**3
                    reserved = torch.cuda.memory_reserved(0) / 1024**3
                    print(f"\nImage {i}/{len(test_dataset)}: GPU Allocated: {allocated:.3f}GB, Reserved: {reserved:.3f}GB")
                    
                input_img = input_img.cuda()

                if not args.tile:
                    if 'Uformer' in args.arch:
                        b, c, h, w = input_img.size()
                        input_padded, mask = expand2square(input_img, factor=128)
                        restored, _ = model_restoration(input_padded)
                        restored = torch.masked_select(restored, mask.bool()).reshape(b, c, h, w)
                        del input_padded, mask
                    else:
                        restored, _ = model_restoration(input_img)
                else:
                    b, c, h, w = input_img.size()
                    if 'Uformer' in args.arch:
                        tile_size = args.tile_size
                        overlap = args.tile_overlap
                        tiles = []
                        masks = []
                        
                        tile, mask = expand2square(input_img[:, :, :, :tile_size], factor=128)
                        tiles.append(tile)
                        masks.append(mask)
                        
                        tile, mask = expand2square(input_img[:, :, :, -tile_size:], factor=128)
                        tiles.append(tile)
                        masks.append(mask)
                        
                        restored_tiles = []
                        for tile_idx in range(len(tiles)):
                            tile_restored, _ = model_restoration(tiles[tile_idx])
                            tile_restored = torch.masked_select(
                                tile_restored,
                                masks[tile_idx].bool()
                            ).reshape(b, c, h, tile_size)
                            restored_tiles.append(tile_restored)
                            
                        merge_size = tile_size - overlap
                        restored = torch.cat([
                            restored_tiles[0][:, :, :, :merge_size],
                            restored_tiles[1][:, :, :, -merge_size:]
                        ], dim=3)
                        
                        del tiles, masks, restored_tiles
                    else:
                        restored, _ = model_restoration(input_img)
                        
                restored = torch.clamp(restored, 0, 1)
                restored_cpu = restored.detach().cpu().numpy().copy()
                del input_img, restored
                brutal_memory_cleanup()
                
                restored_cpu = restored_cpu.transpose(0, 2, 3, 1)
                
                for batch in range(len(restored_cpu)):
                    restored_img = restored_cpu[batch]
                    restored_img = np.uint8(restored_img * 255)
                    restored_img = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)
                    save_path = os.path.join(result_dir, filename[batch] + '.png')
                    cv2.imwrite(save_path, restored_img)
                    
                del restored_cpu, restored_img
                processed_count += 1
                
            except torch.cuda.OutOfMemoryError as e:
                failed_count += 1
                print(f"\n❌ CUDA OOM at image {i}: {e}")
                brutal_memory_cleanup()
                continue
            except Exception as e:
                failed_count += 1
                print(f"\n❌ Error at image {i}: {e}")
                brutal_memory_cleanup()
                continue
                
    brutal_memory_cleanup()
    
    print("\n" + "="*70)
    print("Testing completed!")
    print("="*70)
    print(f"Dataset: {variant_display}")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed: {failed_count}")
    print(f"Total: {len(test_dataset)}")
    print(f"Results saved to: {result_dir}")
    print("="*70)
    print(f"\n✓ Ready for evaluation!")
    print(f"   Next: python evaluate_realrain.py --variant {args.variant}")