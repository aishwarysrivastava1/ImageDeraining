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

from models import model_utils
from utils import expand2square


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image deraining inference on GT-Rain')
    parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
    
    # TODO: Set path to input directory (GT-RAIN_test)
    parser.add_argument('--input_dir', default='', type=str)
    
    # TODO: Set path to results directory
    parser.add_argument('--result_dir', default='', type=str)
    
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--model_name', default='UNet_RLP_RPIM', type=str)
    
    # TODO: Set path to model weights
    parser.add_argument('--weights', default='', type=str)
    
    parser.add_argument('--arch', default='UNet', type=str)
    parser.add_argument('--use_rlp', action='store_true')
    parser.add_argument('--use_rpim', action='store_true')
    parser.add_argument('--dataset', default='GT-Rain', type=str)
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
    
    if args.arch in ['UNet', 'Uformer_T']:
        pass
    else:
        args.arch = ''.join([x if x in args.model_name else '' for x in ['UNet', 'Uformer_T']])
        
    if not args.use_rlp and 'RLP' in args.model_name:
        args.use_rlp = True
    if not args.use_rpim and 'RPIM' in args.model_name:
        args.use_rpim = True

    print("="*70)
    print("Testing Configuration - GT-Rain Dataset")
    print("="*70)
    print(f"Architecture: {args.arch}")
    print(f"Use RLP: {args.use_rlp}")
    print(f"Use RPIM: {args.use_rpim}")
    print(f"Dataset: {args.dataset}")
    print(f"Input directory: {args.input_dir}")
    print(f"Result directory: {args.result_dir}")
    print(f"Weights: {args.weights}")
    print(f"Batch size: {args.batch_size}")
    print(f"Model name: {args.model_name}")
    
    if args.arch == 'Uformer_T':
        print(f"Uformer settings:")
        print(f"  - train_ps: {args.train_ps}")
        print(f"  - embed_dim: {args.embed_dim}")
        print(f"  - win_size: {args.win_size}")
        print(f"  - token_projection: {args.token_projection}")
        print(f"  - token_mlp: {args.token_mlp}")
    print("="*70)

    print("\nLoading model...")
    model_restoration = model_utils.get_arch(args)

    if torch.cuda.device_count() > 1:
        model_restoration = torch.nn.DataParallel(model_restoration)
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")

    checkpoint = torch.load(args.weights, map_location='cpu')
    try:
        model_restoration.load_state_dict(checkpoint['state_dict'])
        print(f"✓ Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    except:
        try:
            model_restoration.load_state_dict(checkpoint)
            print("✓ Model loaded (direct state dict)")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Trying to load with module prefix removal...")
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            model_restoration.load_state_dict(new_state_dict)
            print("✓ Model loaded after prefix adjustment")
            
    print(f"===> Testing using weights: {args.weights}")
    
    model_restoration.cuda()
    model_restoration.eval()
    for p in model_restoration.parameters():
        p.requires_grad = False
        
    print("\nLoading test dataset...")
    test_dataset = model_utils.get_dataset(args, mode='test')
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0,
        drop_last=False, 
        pin_memory=True
    )
    print(f"Loaded {len(test_dataset)} test images")
    
    if hasattr(args, 'model_name') and args.model_name:
        result_dir = os.path.join(args.result_dir, args.model_name)
    else:
        model_suffix = ''
        if args.use_rlp:
            model_suffix += '_RLP'
        if args.use_rpim:
            model_suffix += '_RPIM'
        result_dir = os.path.join(args.result_dir, args.arch + model_suffix)
        
    os.makedirs(result_dir, exist_ok=True)
    print(f"✓ Results will be saved to: {result_dir}")
    
    print("\n" + "="*70)
    print("Starting testing...")
    print("="*70)

    with torch.no_grad():
        for i, (input_img, filename) in enumerate(tqdm(test_loader, desc='Testing')):
            if i % 50 == 0:
                torch.cuda.empty_cache()
            
            input_img = input_img.cuda()
            
            if not args.tile:
                if 'Uformer' in args.arch:
                    b, c, h, w = input_img.size()
                    input_padded, mask = expand2square(input_img, factor=128)
                    restored, _ = model_restoration(input_padded)
                    restored = torch.masked_select(restored, mask.bool()).reshape(b, c, h, w)
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
                else:
                    restored, _ = model_restoration(input_img)
                    
            restored = torch.clamp(restored, 0, 1)
            restored = restored.permute(0, 2, 3, 1).cpu().numpy()
            
            for batch in range(len(restored)):
                restored_img = restored[batch]
                restored_img = np.uint8(restored_img * 255)
                restored_img = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)
                save_path = os.path.join(result_dir, filename[batch] + '.png')
                cv2.imwrite(save_path, restored_img)
                
    print("\n" + "="*70)
    print("Testing completed!")
    print("="*70)
    print(f"Total images processed: {len(test_dataset)}")
    print(f"Results saved to: {result_dir}")
    print("="*70)