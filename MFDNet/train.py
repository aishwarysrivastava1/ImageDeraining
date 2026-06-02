import os
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_gtrain import DatasetTrain
from MFDNet import HPCNet as mfdnet
from SSIM import SSIM
import losses
import utils

def main():
    parser = argparse.ArgumentParser(description='Train MFDNet on GTRain')
    parser.add_argument('--train_dir', default='d:/Deraining/RLP/rlp/data/GT-RAIN_train', type=str, help='Directory of training images')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs to train')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size (fits in 8GB)')
    parser.add_argument('--lr', default=4e-4, type=float, help='Initial learning rate')
    parser.add_argument('--patch_size', default=128, type=int, help='Training patch size')
    parser.add_argument('--save_freq', default=50, type=int, help='Epoch frequency to save model')
    parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    args = parser.parse_args()

    gpus = args.gpus
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    torch.backends.cudnn.benchmark = True

    file_psnr = 'MFD_PSNR.txt'
    file_loss = 'MFD_LOSS.txt'
    
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    start_epoch = 1

    model_dir = os.path.join('.', 'logs')
    utils.mkdir(model_dir)

    model_restoration = mfdnet()
    model_restoration.cuda()

    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 0:
        print(f"\n\nLet's use {torch.cuda.device_count()} GPUs!\n\n")

    optimizer = optim.Adam(model_restoration.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.StepLR(step_size=60, gamma=0.8, optimizer=optimizer)
    scheduler.step()

    if args.resume:
        path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
        if os.path.exists(path_chk_rest):
            utils.load_checkpoint(model_restoration, path_chk_rest)
            start_epoch = utils.load_start_epoch(path_chk_rest) + 1
            utils.load_optim(optimizer, path_chk_rest)
            for i in range(1, start_epoch):
                scheduler.step()
            print("==> Resuming Training with learning rate:", scheduler.get_lr()[0])

    if len(device_ids) > 1:
        model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)

    criterion_char = losses.CharbonnierLoss().cuda()
    criterion_edge = losses.EdgeLoss().cuda()
    criterion_SSIM = SSIM().cuda()

    print('===> Loading training dataset')
    train_dataset = DatasetTrain(args.train_dir, {'patch_size': args.patch_size})
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, drop_last=False, pin_memory=True, persistent_workers=True)

    print(f'===> Start Epoch {start_epoch} End Epoch {args.epochs}')

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        epoch_loss = 0
        SSIM_all = 0
        train_sample = 0

        model_restoration.train()
        
        for i, data in enumerate(tqdm(train_loader), 0):
            # Fast zero_grad optimization
            model_restoration.zero_grad(set_to_none=True)

            # dataset_gtrain returns rainy, clean, scene
            input_ = data[0].cuda(non_blocking=True)
            target = data[1].cuda(non_blocking=True)

            restored = model_restoration(input_)
            
            loss_char0 = criterion_char(restored[0], target)
            loss_char1 = criterion_char(restored[1], input_)
            loss_edge0 = criterion_edge(restored[0], target)
            loss_edge1 = criterion_edge(restored[1], input_)
            loss_SSIM0 = criterion_SSIM(restored[0], target)
            loss_SSIM1 = criterion_SSIM(restored[1], input_)
            
            loss = 0.3 * (loss_char0 + 0.2 * loss_char1) + (0.2 * (loss_edge0)) - (0.15 * (loss_SSIM0 + 0.2 * loss_SSIM1))
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            SSIM_all += loss_SSIM0.item()
            train_sample += 1
            
        SSIM_val = SSIM_all / train_sample
        scheduler.step()

        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tSSIM: {:.4f}\tLearningRate {:.8f}".format(
            epoch, time.time() - epoch_start_time, epoch_loss, SSIM_val, scheduler.get_lr()[0]))
        print("------------------------------------------------------------------")
        
        with open(file_loss, 'a+') as loss_file:
            loss_file.write(f'Epoch: {epoch}, Time: {time.time() - epoch_start_time:.4f}, Loss: {epoch_loss:.4f}, SSIM: {SSIM_val:.4f}, LearningRate: {scheduler.get_lr()[0]:.8f}\n')

        # Save checkpoint periodically
        if epoch % args.save_freq == 0:
            torch.save({'epoch': epoch,
                        'state_dict': model_restoration.state_dict() if len(device_ids) == 1 else model_restoration.module.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))

        # Save latest model
        torch.save({'epoch': epoch,
                    'state_dict': model_restoration.state_dict() if len(device_ids) == 1 else model_restoration.module.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, "model_latest.pth"))

if __name__ == '__main__':
    main()
