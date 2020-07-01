
import numpy as np
import torch
import cv2 as cv

from tqdm import tqdm
from torch.utils.data import DataLoader


# local modules
from .dataset import DynamicH5Dataset
from utils.util import CropParameters


class EventDataLoader(DataLoader):
    """
    Construct UEB-image
    """

    def __init__(self, data_path, num_workers=1, pin_memory=True, shuffle=True, dataset_kwargs=None):

        print("init EventDataloader")
        if dataset_kwargs is None:
            dataset_kwargs = {}
        dataset = DynamicH5Dataset(data_path, **dataset_kwargs)
        super().__init__(dataset, batch_size=1, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

        self.H, self.W = dataset.sensor_resolution
        




if __name__ == "__main__":
    from events_contrast_maximization.utils.event_utils import cvshow_voxel_grid, cvshow_all

    dataset_kwargs = {'transforms': {},
                      'max_length': None,
                      'sensor_resolution': None,
                      'num_bins': 9,
                      'voxel_method': {'method': 'random_k_events',
                                       'k': 30000,
                                       't': 0.5,
                                       'sliding_window_w': 500,
                                       'sliding_window_t': 0.1}
                      }


    ev_loader = EventDataLoader('data/office.h5', shuffle=False, dataset_kwargs=dataset_kwargs)
    H, W = ev_loader.H, ev_loader.W

    # crop = CropParameters(W, H, 32)
    for i, item in enumerate(tqdm(ev_loader)):

        data_source_idx = item['data_source_idx']
        t = item['timestamp']
        dt = item['dt']
        voxel = item['events']
        # frame = item['frame']
        # flow = item['flow']

        cvshow_voxel_grid(voxel.squeeze().cpu().numpy())

        # cvshow_all(voxel.squeeze().cpu().numpy(),
        #             flow.squeeze().cpu().numpy(),
        #             frame.squeeze().cpu().numpy())
