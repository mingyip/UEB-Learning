

import argparse
import yaml
import sys
import utils
import torch
import numpy as np

from tqdm import tqdm
from pathlib import Path
from thop import profile

import torch.optim as optim
import torch.nn as nn


# local modules
from loss import compute_loss
from model.unet import UNet
from utils.util import CropParameters
from data_loader.data_loader import EventDataLoader
from events_contrast_maximization.utils.event_utils import cvshow_voxel_grid, cvshow_all, events_to_image_torch


def warp_events_with_flow_torch(events, flow, sensor_size=(180, 240)):

    eps = torch.finfo(flow.dtype).eps
    xs, ys, ts, ps = events

    xs = xs.type(torch.long).to(flow.device)
    ys = ys.type(torch.long).to(flow.device)
    ts = ts.to(flow.device)
    ps = ps.type(torch.long).to(flow.device)


    # TODO: Check if ts is correct calibration here
    ts = (ts - ts[0]) / (ts[-1] - ts[0] + eps)
    
    xs_ = xs + ts * flow[0,ys,xs]
    ys_ = ys + ts * flow[0,ys,xs]

    img = events_to_image_torch(xs_, ys_, ps, sensor_size=sensor_size, interpolation='bilinear', padding=False)
    return img

def main(args):
    dataset_kwargs = {'transforms': {},
                      'max_length': None,
                      'sensor_resolution': None,
                      'preload_events': False,
                      'num_bins': 16,
                      'voxel_method': {'method': 'random_k_events',
                                       'k': 60000,
                                       't': 0.5,
                                       'sliding_window_w': 500,
                                       'sliding_window_t': 0.1}
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
        'num_bins': 16
    }


    torch.autograd.set_detect_anomaly(True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ev_loader = EventDataLoader(args.h5_file_path, 
                            batch_size=1,
                            num_workers=6,
                            shuffle=True,
                            pin_memory=True,
                            dataset_kwargs=dataset_kwargs
                            )

    H, W = ev_loader.H, ev_loader.W

    model = UNet(unet_kwargs)
    model = model.to(device)
    model.train()
    crop = CropParameters(W, H, 4)

    print("=== Let's use", torch.cuda.device_count(), "GPUs!")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999))
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01)
    # raise
    # tmp_voxel = crop.pad(torch.randn(1, 9, H, W).to(device))
    # F, P = profile(model, inputs=(tmp_voxel, ))



    for idx in range(10):
        # for i, item in enumerate(tqdm(ev_loader)):
        for i, item in enumerate(ev_loader):

            events = item['events']
            voxel = item['voxel'].to(device)
            voxel = crop.pad(voxel)

            model.zero_grad()
            optimizer.zero_grad()

            flow = model(voxel) * 10
           
            flow = torch.clamp(flow, min=-40, max=40)
            loss = compute_loss(events, flow)
            loss.backward()
            
            # cvshow_voxel_grid(voxel.squeeze()[0:2].cpu().numpy())
            # raise
            optimizer.step()

            if i % 10 == 0:
                print(idx, 
                        i, '\t',
                        "{0:.2f}".format(loss.data.item()),
                        "{0:.2f}".format(torch.max(flow[0,0]).item()),
                        "{0:.2f}".format(torch.min(flow[0,0]).item()),
                        "{0:.2f}".format(torch.max(flow[0,1]).item()),
                        "{0:.2f}".format(torch.min(flow[0,1]).item()),)

                xs, ys, ts, ps = events
                print_voxel = voxel[0].sum(axis=0).cpu().numpy()
                print_flow = flow[0].clone().detach().cpu().numpy()
                print_co = warp_events_with_flow_torch((xs[0][ps[0]==1], ys[0][ps[0]==1], ts[0][ps[0]==1], ps[0][ps[0]==1]), flow[0].clone().detach(), sensor_size=(H,W))
                print_co = crop.pad(print_co)
                print_co = print_co.cpu().numpy()

                cvshow_all(idx=idx*10000+i, voxel=print_voxel, flow=flow[0].clone().detach().cpu().numpy(), frame=None, compensated=print_co)

        





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