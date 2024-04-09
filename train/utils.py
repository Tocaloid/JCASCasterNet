# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch import nn
from torch import pi
from torch.fft import fft, ifft

import numpy as np
from numpy.random import choice

import os
import random
import shutil
from scipy import io
from math import sqrt, log10

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import stem,plot
from math import log, log2, log10, sqrt

from my_log import logger, rmLogger



def SELoss(
        H,  # b, N_C, K, M
        F,  # (b,) N_C, M, K
        SNR, 
    ):
    SNR=10**(SNR/10);

    Y=H@F;  # b, N_C, K, K
    b, N_C, K, _=Y.shape;

    power_noise=calMeanPower(F)/SNR;

    Y=Y.reshape(b*N_C, K, K).abs().square();  # b*N_C, K, K
    idx=torch.arange(K);
    power_signal=Y[:, idx, idx];  # b*N_C, K
    power_inter=Y.sum(-1)-power_signal;  # b*N_C, K
    SINR=power_signal/(power_inter+power_noise);
    res=torch.log(1+SINR)/log(2);

    return -1*res.mean();  # 1



def RMSELoss(
        x, 
        y,   
    ):
    b=x.shape[0];

    x=x.reshape(b, -1);
    y=y.reshape(b, -1);

    res=(x-y).abs().square().mean(-1).sqrt().mean();
    return res;



def NMSELoss(
        x, 
        y,   
    ):
    b=x.shape[0];

    x=x.reshape(b, -1);
    y=y.reshape(b, -1);

    res=(x-y).abs().square().mean(-1);
    res=res/y.abs().square().mean(-1);
    
    return res.mean();



def CosLoss(
        x,  # b, N_C, 1, M
        y,  # b, N_C, 1, M
    ):
    tmp_xy=(x@y.mH).abs();  # b, N_C, 1, 1
    tmp_xx=(x@x.mH).abs().sqrt();  # b, N_C, 1, 1
    tmp_yy=(y@y.mH).abs().sqrt();  # b, N_C, 1, 1
    
    cos=tmp_xy/tmp_xx/tmp_yy;

    return -cos.mean();



def theta2label(
        theta, 
        grid_num, 
    ):
    b=theta.shape[0];

    idx=torch.round((theta/180)*grid_num).long()[:, :, 0];
    label=torch.zeros(b, grid_num);
    label.scatter_(
        dim=-1, 
        index=idx, 
        src=torch.ones_like(idx).float(), 
    );

    return label;


def calMeanPower(x: torch.Tensor):
    return x.abs().square().mean();



def awgn(
        x, 
        ofdm_dim_idx, 
        s, 
        SNR, 
    ):
    device=x.device;
    SNR=10**(SNR/10);

    npower=calMeanPower(s)/SNR;
    noise=torch.randn(*x.shape, dtype=torch.complex64).to(device)*torch.sqrt(npower);

    return x+fft(noise, dim=ofdm_dim_idx)/sqrt(noise.shape[ofdm_dim_idx]);



def env_init(seed: int):
    random.seed(seed);
    os.environ['PYHTONHASHSEED']=str(seed);
    np.random.seed(seed);
    torch.manual_seed(seed);
    torch.cuda.manual_seed(seed);
    torch.backends.cudnn.deterministic=True;



def main():
    pass;



if __name__=="__main__":
    logger("utils.py");
    main();
    rmLogger();