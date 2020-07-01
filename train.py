

import argparse
import yaml
import sys
import utils
import torch
import numpy as np


from tqdm import tqdm
from pathlib import Path
from thop import profile


# local modules
from loss import flow_loss
from model.unet import UNet
from utils.util import CropParameters
from data_loader.data_loader import EventDataLoader
from events_contrast_maximization.utils.event_utils import cvshow_voxel_grid, cvshow_all


def main(args):
    dataset_kwargs = {'transforms': {},
                      'max_length': None,
                      'sensor_resolution': None,
                      'num_bins': 9,
                      'voxel_method': {'method': args.voxel_method,
                                       'k': args.k,
                                       't': args.t,
                                       'sliding_window_w': args.sliding_window_w,
                                       'sliding_window_t': args.sliding_window_t}
                      }

    unet_kwargs = {
        'base_num_channels': 32, # written as '64' in EVFlowNet tf code
        'num_encoders': 4,
        'num_residual_blocks': 2,  # transition
        'num_output_channels': 2,  # (x, y) displacement
        'skip_type': 'concat',
        'norm': None,
        'use_upsample_conv': True,
        'kernel_size': 3,
        'channel_multiplier': 2,
        'num_bins': 9
    }

    


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ev_loader = EventDataLoader(args.h5_file_path, dataset_kwargs=dataset_kwargs)
    H, W = ev_loader.H, ev_loader.W

    model = UNet(unet_kwargs)
    model = model.to(device)
    model.train()
    crop = CropParameters(W, H, 4)
    # tmp_voxel = crop.pad(torch.randn(1, 9, H, W).to(device))
    # F, P = profile(model, inputs=(tmp_voxel, ))


    for i, item in enumerate(tqdm(ev_loader)):

        voxel = item['voxel'].to(device)
        voxel = crop.pad(voxel)

        xs, ys, ts, ps = item['events']
        xs = xs.squeeze().to(device)
        ys = ys.squeeze().to(device)
        ts = ts.squeeze().to(device)
        ps = ps.squeeze().to(device)

        print(xs.shape, xs.dtype, xs.device)
        print(ys.shape, ys.dtype, ys.device)
        print(ts.shape, ts.dtype, ts.device)
        print(ps.shape, ps.dtype, ps.device)

        flow = model(voxel)
        print(flow.shape, flow.shape, flow.device)
        raise
        loss = flow_loss(item['voxel'], flow)
        print(loss.requires_grad)

        # print(output)
        
    raise





if __name__ == "__main__":

    # global var 
    torch.set_default_tensor_type(torch.FloatTensor)


    # add Training parser
    parser = argparse.ArgumentParser(description='UEB-Learning')
    # TODO: implement reload logic
    parser.add_argument('--checkpoint_path', required=False, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('--h5_file_path', required=True, type=str,
                        help='path to hdf5 events')
    parser.add_argument('--output_folder', default="results/", type=str,
                        help='where to save outputs to')
    parser.add_argument('--is_flow', action='store_true',
                        help='If true, save output to flow npy file')
    parser.add_argument('--update', action='store_true',
                        help='Set this if using updated models')
    parser.add_argument('--voxel_method', default='k_events', type=str,
                        help='which method should be used to form the voxels',
                        choices=['between_frames', 'k_events', 't_seconds', 'random_k_events'])
    parser.add_argument('--k', type=int,
                        help='new voxels are formed every k events (required if voxel_method is k_events)')
    parser.add_argument('--sliding_window_w', type=int,
                        help='sliding_window size (required if voxel_method is k_events)')
    parser.add_argument('--t', type=float,
                        help='new voxels are formed every t seconds (required if voxel_method is t_seconds)')
    parser.add_argument('--sliding_window_t', type=float,
                        help='sliding_window size in seconds (required if voxel_method is t_seconds)')
    parser.add_argument('--loader_type', default='H5', type=str,
                        help='Which data format to load (HDF5 recommended)')

    args = parser.parse_args()
    main(args)