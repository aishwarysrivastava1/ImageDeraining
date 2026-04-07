import os
import torch
import torch.optim as optim
import random
import time
import numpy as np
import datetime
from tqdm import tqdm 

from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from timm.utils import NativeScaler
import torch
torch.cuda.empty_cache()

from loss import CharbonnierLoss
from models import model_utils
from options import parse_options

if __name__ == "__main__":
    opt = parse_options().parse_args()
    print(opt)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    torch.cuda.set_device(int(opt.gpu))
    torch.backends.cudnn.benchmark = True

    rlp_suffix = "_RLP" if opt.use_rlp else ""
    rpim_suffix = "_RPIM" if opt.use_rpim else ""
    arch = opt.arch + rlp_suffix + rpim_suffix + opt.env

    log_dir = os.path.join(opt.save_dir, 'deraining', opt.dataset, arch)
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.datetime.now().isoformat().replace(':', '-')
    logname = os.path.join(log_dir, timestamp + '.txt')

    print("Now time is : ", datetime.datetime.now().isoformat())
    result_dir = os.path.join(log_dir, 'results')
    model_dir  = os.path.join(log_dir, 'models')
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    model_restoration = model_utils.get_arch(opt)

    with open(logname, 'a') as f:
        f.write(str(opt) + '\n')
        f.write(str(model_restoration) + '\n')

    start_epoch = 1
    if opt.optimizer.lower() == 'adam':
        optimizer = optim.Adam(
            model_restoration.parameters(),
            lr=opt.lr_initial,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=opt.weight_decay
        )
    elif opt.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(
            model_restoration.parameters(),
            lr=opt.lr_initial,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=opt.weight_decay
        )
    else:
        raise Exception("Error optimizer...")

    if torch.cuda.device_count() > 1:
        model_restoration = torch.nn.DataParallel(model_restoration)
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    model_restoration.cuda()

    if opt.warmup:
        print("Using warmup and cosine strategy!")
        warmup_epochs = opt.warmup_epochs
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, opt.nepoch - warmup_epochs, eta_min=1e-6
        )
        scheduler = GradualWarmupScheduler(
            optimizer, multiplier=1, total_epoch=warmup_epochs,
            after_scheduler=scheduler_cosine
        )
        scheduler.step()
    else:
        step = 50
        print("Using StepLR, step={}!".format(step))
        scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
        scheduler.step()

    if opt.resume:
        path_chk_rest = opt.pretrain_weights
        print("Resume from " + path_chk_rest)
        model_utils.load_checkpoint(model_restoration, path_chk_rest)
        start_epoch = model_utils.load_start_epoch(path_chk_rest) + 1
        model_utils.load_optim(optimizer, path_chk_rest)

        for _ in range(1, start_epoch):
            scheduler.step()

        print('------------------------------------------------------------------------------')
        print("==> Resuming Training with learning rate:", scheduler.get_last_lr()[0])
        print('------------------------------------------------------------------------------')

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, opt.nepoch - start_epoch + 1, eta_min=1e-6
        )

    criterion = CharbonnierLoss().cuda()
    
    print('===> Loading datasets')
    img_options_train = {'patch_size': opt.train_ps}
    train_dataset = model_utils.get_training_data(opt.dataset, opt.train_dir, img_options_train)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=0,          
        pin_memory=True,
        drop_last=False
    )

    print("Sizeof training set: ", len(train_dataset))

    print('===> Start Epoch {}, End Epoch {}'.format(start_epoch, opt.nepoch))

    loss_scaler = NativeScaler()
    torch.cuda.empty_cache()

    for epoch in range(start_epoch, opt.nepoch + 1):
        epoch_start_time = time.time()
        epoch_loss = 0

        for _, data in enumerate(tqdm(train_loader), 0):
            optimizer.zero_grad()

            input = data[0].cuda()
            gt    = data[1].cuda()
            input_name = data[2]

            with torch.cuda.amp.autocast():
                restored, _ = model_restoration(input)
                loss = criterion(restored, gt)

            loss_scaler(loss, optimizer, parameters=model_restoration.parameters())
            epoch_loss += loss.item()

        scheduler.step()

        print("------------------------------------------------------------------")
        print(
            "Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.8f}".format(
                epoch, time.time() - epoch_start_time, epoch_loss,
                scheduler.get_last_lr()[0]
            )
        )
        print("------------------------------------------------------------------")

        with open(logname, 'a') as f:
            f.write(
                "Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}\n".format(
                    epoch, time.time() - epoch_start_time,
                    epoch_loss, scheduler.get_last_lr()[0]
                )
            )

        if epoch % opt.checkpoint == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'state_dict': model_restoration.state_dict(),
                    'optimizer': optimizer.state_dict()
                },
                os.path.join(model_dir, "model_epoch_{}.pth".format(epoch))
            )

    print("Now time is : ", datetime.datetime.now().isoformat())