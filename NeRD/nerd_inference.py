import os
import torch
from skimage.util import img_as_ubyte
from nerd_setup import setup_paths
setup_paths()
import utils              
from layers import window_partitionx, window_reversex              
from model import MultiscaleNet              
def configure_gpu(gpus='0'):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    torch.backends.cudnn.benchmark = True
def load_model(weights):
    model = MultiscaleNet()
    utils.load_checkpoint(model, weights)
    model.eval().cuda()
    return model
def nerd_infer(model, input_, win_size=256):
    _, _, height, width = input_.shape
    input_windows, batch_list = window_partitionx(input_, win_size)
    with torch.amp.autocast('cuda'):
        restored = model(input_windows)
    restored = window_reversex(restored[0], win_size, height, width, batch_list)
    return torch.clamp(restored, 0, 1)
def save_outputs(restored, filenames, result_dir):
    utils.mkdir(result_dir)
    restored_np = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
    for batch_idx in range(len(restored_np)):
        restored_img = img_as_ubyte(restored_np[batch_idx])
        utils.save_img(os.path.join(result_dir, filenames[batch_idx] + '.png'), restored_img)
