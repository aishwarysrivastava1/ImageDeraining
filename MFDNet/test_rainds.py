import os
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from skimage.util import img_as_ubyte
import utils
from MFDNet import HPCNet as mfdnet
from dataset_rainds import DatasetTest

def main():
    parser = argparse.ArgumentParser(description='MFDNet Testing on RainDS')
    parser.add_argument('--input_dir', default='d:/Deraining/RLP/rlp/data/RainDS', type=str, help='Directory of test images')
    parser.add_argument('--result_dir', default='./results/MFDNet/RainDS', type=str, help='Directory for results')
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

    variant_dirs = {
        'rainds_real': 'Rainds_Real',
        'rainds_syn': 'Rainds_Syn',
    }
    for variant, variant_dir in variant_dirs.items():
        variant_result_dir = os.path.join(args.result_dir, variant_dir)
        utils.mkdir(variant_result_dir)
        test_dataset = DatasetTest(args.input_dir, variant=variant, rain_type='all')
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=True)

        with torch.no_grad():
            for i, data_test in enumerate(tqdm(test_loader, desc=f'Testing {variant}'), 0):
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()

                input_ = data_test[0].cuda()
                gt_ = data_test[1] # Not needed for testing generation
                filenames = data_test[2]

                with torch.amp.autocast('cuda'):
                    restored = model(input_)
                    
                restored = torch.clamp(restored[0], 0, 1)
                restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

                for batch in range(len(restored)):
                    restored_img = img_as_ubyte(restored[batch])
                    utils.save_img(os.path.join(variant_result_dir, filenames[batch] + '.png'), restored_img)

if __name__ == '__main__':
    main()
