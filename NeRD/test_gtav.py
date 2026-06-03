import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset_gtav import DatasetTest
from nerd_inference import configure_gpu, load_model, nerd_infer, save_outputs
def main():
    parser = argparse.ArgumentParser(description='NeRD Testing on GTAV-NightRain')
    parser.add_argument('--input_dir', default='./data', type=str)
    parser.add_argument('--result_dir', default='./results/NeRD/GTAV', type=str)
    parser.add_argument('--weights', default='./logs/model_latest.pth', type=str)
    parser.add_argument('--gpus', default='0', type=str)
    parser.add_argument('--win_size', default=256, type=int)
    args = parser.parse_args()
    configure_gpu(args.gpus)
    print('===> Testing using weights:', args.weights)
    model = load_model(args.weights)
    for test_set in ['set1', 'set2', 'set3']:
        set_result_dir = os.path.join(args.result_dir, test_set)
        test_dataset = DatasetTest(args.input_dir, test_set=test_set)
        test_loader = DataLoader(
            dataset=test_dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=True
        )
        with torch.no_grad():
            for data_test in tqdm(test_loader, desc=f'Testing GTAV {test_set}'):
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
                input_ = data_test[0].cuda()
                filenames = data_test[1]
                restored = nerd_infer(model, input_, win_size=args.win_size)
                save_outputs(restored, filenames, set_result_dir)
if __name__ == '__main__':
    main()
