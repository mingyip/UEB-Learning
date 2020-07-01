

import time
import torch
import h5py
import numpy as np
import os
import cv2 
from skimage.color import hsv2rgb




def event_loss(events, flow, p=1):
    
    eps = torch.finfo(flow.dtype).eps
    H, W = flow.shape[1:]
    x, y, t = events

    # Estimate events position after flow
    x_ = torch.clamp(x + t * flow[0,y,x], min=0, max=W-1)
    y_ = torch.clamp(y + t * flow[1,y,x], min=0, max=H-1)

    x0 = torch.floor(x_)
    x1 = x0 + 1
    y0 = torch.floor(y_)
    y1 = y0 + 1

    # Interpolation ratio
    x0_ = x_-x0
    x1_ = x1-x_
    y0_ = y_-y0
    y1_ = y1-y_

    Ra = x0_ * y0_
    Rb = x1_ * y0_
    Rc = x0_ * y1_
    Rd = x1_ * y1_

    # weighted timestamp
    t *= p
    Ta = Ra * t
    Tb = Rb * t
    Tc = Rc * t
    Td = Rd * t

    # # Prevent R and T to be zero
    Ra += eps; Rb += eps; Rc += eps; Rd += eps
    Ta += eps; Tb += eps; Tc += eps; Td += eps

    # Calculate interpolation flatterned index of 4 corners for all events
    x1_ = torch.clamp(x1, max=W-1)
    y1_ = torch.clamp(y1, max=H-1)

    y1_W = y1_ * W
    y0_W = y0  * W
    Ia = (x1_ + y1_W).type(torch.long)
    Ib = (x0  + y1_W).type(torch.long)
    Ic = (x1_ + y0_W).type(torch.long)
    Id = (x0  + y0_W).type(torch.long)


    # Compute the nominator and denominator
    nominator = torch.zeros((W*H), dtype=flow.dtype, device=flow.device)
    denominator = torch.zeros((W*H), dtype=flow.dtype, device=flow.device)

    denominator.index_add_(0, Ia, Ra)
    denominator.index_add_(0, Ib, Rb)
    denominator.index_add_(0, Ic, Rc)
    denominator.index_add_(0, Id, Rd)

    nominator.index_add_(0, Ia, Ta)
    nominator.index_add_(0, Ib, Tb)
    nominator.index_add_(0, Ic, Tc)
    nominator.index_add_(0, Id, Td)

    loss = (nominator / (denominator + eps)) ** 2
    return loss.sum()




def flow_loss(events, flow):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x, y, p, t = events
    print(x.shape)
    print(y.shape)
    print(p.shape)
    print(t.shape)
    raise

    t = (t - t[0]) / (t[-1] - t[0])
    n = p == False

    x = torch.tensor(x, dtype=torch.int64, device=device)
    y = torch.tensor(y, dtype=torch.int64, device=device)
    p = torch.tensor(p, device=device)
    t = torch.tensor(t, dtype=torch.float, device=device)
    flow = torch.tensor(flow, device=device)

    # normalize timestamp for {forward, backward} x {positive, negative} events
    # TODO: check if tp[p][-1]
    t_fp = t[p][-1] - t[p]  # t[p][-1] should be 1
    t_bp = t[p][0] - t[p]   # t[p][0] should be 0
    t_fn = t[n][-1] - t[n]
    t_bn = t[n][0] - t[n]

    # Calculate loss of all 4 cases {forward, backward} x {positive, negative} events
    fp_loss = event_loss((x[p], y[p], t_fp), flow, 1)
    bp_loss = event_loss((x[p], y[p], t_bp), flow, 1)
    fn_loss = event_loss((x[n], y[n], t_fn), flow, -1)
    bn_loss = event_loss((x[n], y[n], t_bn), flow, -1)

    positive_loss = fp_loss + bp_loss
    negative_loss = fn_loss + bn_loss
    total_loss = positive_loss + negative_loss

    return total_loss



if __name__ == "__main__":


    visualize_frame_idx = 20


    f = h5py.File('data/office.h5', 'r')
    xs = f['events/xs']
    ys = f['events/ys']
    ts = f['events/ts']
    ps = f['events/ps']


    start_t = 0
    for idx, (i_, f_) in enumerate(zip(f['images'], f['flow'])):


        end_t = f['images/' + i_].attrs['timestamp']
        mask = (ts[()]>=start_t) & (ts[()]<end_t)
        x, y, p, t = xs[mask], ys[mask], ps[mask], ts[mask]


        if len(x) == 0:
            start_t = f['images/' + i_].attrs['timestamp']
            continue

        best_idx = -1
        best_loss = -1
        flow = f['flow/' + f_][()]

        # alpha is for visualization purpose
        alpha = 0.0001
        for i in range(400):

            flow_ = flow * (alpha * (i))
            loss  = flow_loss((x,y,p,t), flow_)

            if best_loss == -1 or best_loss > loss.item():
                best_loss = loss.item()
                best_idx = i

        print(idx, " Best loss: ", best_idx, best_loss)


        start_t = f['images/' + i_].attrs['timestamp']