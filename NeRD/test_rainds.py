import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset_rainds import DatasetTest
from nerd_inference import configure_gpu, load_model, nerd_infer, save_outputs
def main():
    parser = argparse.ArgumentParser(description='NeRD Testing on RainDS')
    parser.add_argument('--input_dir', default='./data', type=str)
    parser.add_argument('--result_dir', default='./results/NeRD/RainDS', type=str)
    parser.add_argument('--weights', default='./logs/model_latest.pth', type=str)
    parser.add_argument('--gpus', default='0', type=str)
    parser.add_argument('--win_size', default=256, type=int)
    args = parser.parse_args()
    configure_gpu(args.gpus)
    print('===> Testing using weights:', args.weights)
    model = load_model(args.weights)
    variant_dirs = {
        'rainds_real': 'Rainds_Real',
        'rainds_syn': 'Rainds_Syn',
    }
    for variant, variant_dir in variant_dirs.items():
        variant_result_dir = os.path.join(args.result_dir, variant_dir)
        test_dataset = DatasetTest(args.input_dir, variant=variant, rain_type='all')
        test_loader = DataLoader(
            dataset=test_dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=True
        )
        with torch.no_grad():
            for data_test in tqdm(test_loader, desc=f'Testing {variant_dir}'):
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
                input_ = data_test[0].cuda()
                filenames = data_test[2]
                restored = nerd_infer(model, input_, win_size=args.win_size)
                save_outputs(restored, filenames, variant_result_dir)
if __name__ == '__main__':
    main()
