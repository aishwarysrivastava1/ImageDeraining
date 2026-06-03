import os
import argparse
import time
import random
import kornia
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from nerd_setup import setup_paths
setup_paths()
import losses              
import utils              
from dataset_gtrain import DatasetTrain              
from model import MultiscaleNet              
from warmup_scheduler import GradualWarmupScheduler              
def main():
    parser = argparse.ArgumentParser(description='Train NeRD on GTRain')
    parser.add_argument('--train_dir', default='./data', type=str)
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--batch_size', default=2, type=int, help='Batch size (NeRD is memory-heavy)')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--patch_size', default=128, type=int)
    parser.add_argument('--save_freq', default=50, type=int)
    parser.add_argument('--gpus', default='0', type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--num_workers', default=0, type=int, help='DataLoader workers (0 to bypass multiprocessing freezes on Windows)')
    args = parser.parse_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    torch.backends.cudnn.benchmark = True
    file_loss = 'NeRD_LOSS.txt'
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    start_epoch = 1
    model_dir = os.path.join('.', 'logs')
    utils.mkdir(model_dir)
    model_restoration = MultiscaleNet()
    model_restoration.cuda()
    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 0:
        print(f'\n\nUsing {torch.cuda.device_count()} GPU(s)\n\n')
    optimizer = optim.Adam(model_restoration.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs - warmup_epochs, eta_min=1e-6
    )
    scheduler = GradualWarmupScheduler(
        optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine
    )
    if args.resume:
        path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
        if os.path.exists(path_chk_rest):
            utils.load_checkpoint(model_restoration, path_chk_rest)
            start_epoch = utils.load_start_epoch(path_chk_rest) + 1
            utils.load_optim(optimizer, path_chk_rest)
            for _ in range(1, start_epoch):
                scheduler.step()
            print('==> Resuming training with learning rate:', scheduler.get_lr()[0])
    if len(device_ids) > 1:
        model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)
    criterion_char = losses.CharbonnierLoss()
    criterion_edge = losses.EdgeLoss()
    criterion_fft = losses.fftLoss()
    criterion_l1 = nn.L1Loss()
    print('===> Loading training dataset')
    train_dataset = DatasetTrain(args.train_dir, {'patch_size': args.patch_size})
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    print(f'===> Start Epoch {start_epoch} End Epoch {args.epochs}')
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        train_sample = 0
        model_restoration.train()
        for data in tqdm(train_loader, desc=f'Epoch {epoch}'):
            model_restoration.zero_grad(set_to_none=True)
            input_ = data[0].cuda(non_blocking=True)
            target_ = data[1].cuda(non_blocking=True)
            target = kornia.geometry.transform.build_pyramid(target_, 3)
            restored = model_restoration(input_)
            loss_fft = (
                criterion_fft(restored[0], target[0])
                + criterion_fft(restored[1], target[1])
                + criterion_fft(restored[2], target[2])
            )
            loss_char = (
                criterion_char(restored[0], target[0])
                + criterion_char(restored[1], target[1])
                + criterion_char(restored[2], target[2])
            )
            loss_edge = (
                criterion_edge(restored[0], target[0])
                + criterion_edge(restored[1], target[1])
                + criterion_edge(restored[2], target[2])
            )
            loss_l1 = criterion_l1(restored[3], target[1]) + criterion_l1(restored[5], target[2])
            loss = loss_char + 0.01 * loss_fft + 0.05 * loss_edge + 0.1 * loss_l1
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            train_sample += 1
        scheduler.step()
        print('------------------------------------------------------------------')
        print(
            f'Epoch: {epoch}\tTime: {time.time() - epoch_start_time:.4f}\t'
            f'Loss: {epoch_loss:.4f}\tLearningRate {scheduler.get_lr()[0]:.8f}'
        )
        print('------------------------------------------------------------------')
        with open(file_loss, 'a+', encoding='utf-8') as loss_file:
            loss_file.write(
                f'Epoch: {epoch}, Time: {time.time() - epoch_start_time:.4f}, '
                f'Loss: {epoch_loss:.4f}, LearningRate: {scheduler.get_lr()[0]:.8f}\n'
            )
        state_dict = (
            model_restoration.state_dict()
            if len(device_ids) == 1
            else model_restoration.module.state_dict()
        )
        checkpoint = {
            'epoch': epoch,
            'state_dict': state_dict,
            'optimizer': optimizer.state_dict(),
        }
        if epoch % args.save_freq == 0:
            torch.save(checkpoint, os.path.join(model_dir, f'model_epoch_{epoch}.pth'))
        torch.save(checkpoint, os.path.join(model_dir, 'model_latest.pth'))
if __name__ == '__main__':
    main()
