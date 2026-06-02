import os
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from skimage.util import img_as_ubyte
import utils
from MFDNet import HPCNet as mfdnet
from dataset_gtrain import DatasetTest

def main():
    parser = argparse.ArgumentParser(description='MFDNet Testing on GTRain')
    parser.add_argument('--input_dir', default='d:/Deraining/RLP/rlp/data/GT-RAIN_test', type=str, help='Directory of test images')
    parser.add_argument('--result_dir', default='./results/MFDNet/GTRain', type=str, help='Directory for results')
    parser.add_argument('--weights', default='./logs/model_latest.pth', type=str, help='Path to weights')
    parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    torch.backends.cudnn.benchmark = True
    
    utils.mkdir(args.result_dir)

    print("===> Testing using weights: ", args.weights)
    model = mfdnet()
    utils.load_checkpoint(model, args.weights)
    model.eval().cuda()

    test_dataset = DatasetTest(args.input_dir)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=True)

    with torch.no_grad():
        for i, data_test in enumerate(tqdm(test_loader, desc='Testing GTRain'), 0):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            input_ = data_test[0].cuda()
            filenames = data_test[1]

            with torch.amp.autocast('cuda'):
                restored = model(input_)
                
            restored = torch.clamp(restored[0], 0, 1)
            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

            for batch in range(len(restored)):
                restored_img = img_as_ubyte(restored[batch])
                utils.save_img(os.path.join(args.result_dir, filenames[batch] + '.png'), restored_img)

if __name__ == '__main__':
    main()
