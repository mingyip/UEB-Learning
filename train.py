

import argparse
import yaml
import sys
import utils
import torch
import numpy as np


from tqdm import tqdm
from pathlib import Path




def main(args):
    
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
                        choices=['between_frames', 'k_events', 't_seconds'])
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