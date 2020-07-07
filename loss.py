

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

    t = t * p
    Ta = Ra * t
    Tb = Rb * t
    Tc = Rc * t
    Td = Rd * t

    # # Prevent R and T to be zero
    Ra = Ra+eps; Rb = Rb+eps; Rc = Rc+eps; Rd = Rd+eps
    Ta = Ta+eps; Tb = Tb+eps; Tc = Tc+eps; Td = Td+eps

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


def compute_loss(events, flow):

    loss = compute_event_flow_loss(events, flow) + \
            compute_smoothness_loss(flow)

    return loss


def compute_event_flow_loss(events, flow):

    # TODO: move device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    xs, ys, ts, ps = events
    eps = torch.finfo(flow.dtype).eps
    assert ((ps == 1) | (ps == -1)).all(), "Convert Polarity to format -1 and 1"


    loss = 0
    for x, y, t, p, f in zip(xs, ys, ts, ps, flow):

        n = p == -1
        p = p == 1
        t = (t - t[0]) / (t[-1] - t[0] + eps)

        xp = x[p].to(device).type(torch.long)
        yp = y[p].to(device).type(torch.long)
        tp = t[p].to(device).type(torch.float)

        xn = x[n].to(device).type(torch.long)
        yn = y[n].to(device).type(torch.long)
        tn = t[n].to(device).type(torch.float)

        t_fp = tp[-1] - tp   # t[p][-1] should be 1
        t_bp = tp[0]  - tp   # t[p][0] should be 0
        t_fn = tn[-1] - tn
        t_bn = tn[0]  - tn

        fp_loss = event_loss((xp, yp, t_fp), f, 1)
        bp_loss = event_loss((xp, yp, t_bp), f, -1)
        fn_loss = event_loss((xn, yn, t_fn), f, 1)
        bn_loss = event_loss((xn, yn, t_bn), f, -1)

        loss = loss + fp_loss + bp_loss + fn_loss + bn_loss
    return loss

def compute_smoothness_loss(flow):
    """
    Local smoothness loss, as defined in equation (5) of the paper.
    The neighborhood here is defined as the 8-connected region around each pixel.
    """
    flow_ucrop = flow[..., 1:]
    flow_dcrop = flow[..., :-1]
    flow_lcrop = flow[..., 1:, :]
    flow_rcrop = flow[..., :-1, :]

    # print(flow_ucrop.shape, flow_dcrop.shape, flow_lcrop.shape, flow_rcrop.shape)

    flow_ulcrop = flow[..., 1:, 1:]
    flow_drcrop = flow[..., :-1, :-1]
    flow_dlcrop = flow[..., :-1, 1:]
    flow_urcrop = flow[..., 1:, :-1]

    # print(flow_ulcrop.shape, flow_drcrop.shape, flow_dlcrop.shape, flow_urcrop.shape)
    # raise

    smoothness_loss = charbonnier_loss(flow_lcrop - flow_rcrop) +\
                      charbonnier_loss(flow_ucrop - flow_dcrop) +\
                      charbonnier_loss(flow_ulcrop - flow_drcrop) +\
                      charbonnier_loss(flow_dlcrop - flow_urcrop)
    smoothness_loss /= 4.
    
    return smoothness_loss * 0.1

def charbonnier_loss(delta, alpha=0.45, epsilon=1e-3):
    """
    Robust Charbonnier loss, as defined in equation (4) of the paper.
    """
    loss = torch.mean(torch.pow((delta ** 2 + epsilon ** 2), alpha))
    return loss


if __name__ == "__main__":


    visualize_frame_idx = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

        x = torch.from_numpy(x).to(device).unsqueeze(0)
        y = torch.from_numpy(y).to(device).unsqueeze(0)
        p = torch.from_numpy(p).to(device).unsqueeze(0)
        p = p.type(torch.float32) * 2.0 - 1.0
        # print("11111")
        # print(torch.sum(p==1.0))
        # print(torch.sum(p==-1.0))
        # print(x.shape)
        # print("11111")

        t = torch.from_numpy(t).to(device).unsqueeze(0)
        flow = torch.from_numpy(flow).to(device)


        # alpha is for visualization purpose
        alpha = 0.0001
        for i in range(400):

            flow_ = flow * (alpha * (i))
            loss  = compute_event_flow_loss((x,y,p,t), flow_)

            if best_loss == -1 or best_loss > loss.item():
                best_loss = loss.item()
                best_idx = i

        print(idx, " Best loss: ", best_idx, best_loss)


        start_t = f['images/' + i_].attrs['timestamp']