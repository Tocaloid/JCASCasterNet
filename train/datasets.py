# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch import nn
from torch.fft import fft, ifft
from torch.utils.data import Dataset

import numpy as np
from numpy import pi
from numpy.random import choice

import hdf5storage

import os
import time
import random
import shutil
import copy
from tqdm import tqdm
from scipy import io
from math import sqrt, log10

from utils import env_init

from my_log import logger, rmLogger



def genH(
        N_gen       , 
        b           , 
        N_C         , 
        K           , 
        M_com       , 
        L_share     , 
        L_com       , 
        L_rad       , 
        R           , 
        device      , 
    ):
    theta_share=torch.rand(b, L_share, 1).to(device)*pi;  # b, L_share, 1
    theta_com=torch.rand(b, L_com, 1).to(device)*pi;  # b, L_com, 1
    theta_com[:, 0:L_share, :]=theta_share.clone().detach();
    theta_rad=torch.rand(b, L_rad, 1).to(device)*pi;  # b, L_rad, 1
    theta_rad[:, 0:L_share, :]=theta_share.clone().detach();

    r_share=torch.rand(b, L_share, 1).to(device)*R;  # b, L_share, 1
    r_com=torch.rand(b, L_com, 1).to(device)*R;  # b, L_com, 1
    r_com[:, 0:L_share, :]=r_share.clone().detach();
    r_rad=torch.rand(b, L_rad, 1).to(device)*R;  # b, L_rad, 1
    r_rad[:, 0:L_share, :]=r_share.clone().detach();

    theta_UE=torch.rand(b, K, 1).to(device)*pi;  # b, K, 1
    r_UE=torch.rand(b, K, 1).to(device)*R;  # b, K, 1
    def genBatch(
            b, 
        ):
        # H_com
        theta_f=torch.rand(b*K, L_com+1, 1).to(device)*2*pi;  # b*K, L_com+1, 1
        rg_f=torch.arange(N_C).to(device).reshape(1, 1, N_C).float();  # 1, 1, N_C
        v_f=torch.exp(1j*(theta_f@rg_f)).unsqueeze(-1);  # b*K, L_com+1, N_C, 1

        theta_a=torch.cos(torch.cat(
            (theta_com.repeat(K, 1, 1),  # b*K, L_com, 1
             theta_UE.reshape(b*K, 1, 1), ),  # b*K, 1, 1
            1, 
        ))*pi;  # b*K, L_com+1, 1
        rg_a=torch.arange(M_com).to(device).reshape(1, 1, M_com).float();  # 1, 1, M_com
        v_a=torch.exp(1j*(theta_a@rg_a)).unsqueeze(-2);  # b*K, L_com+1, 1, M_com

        gain=torch.randn(b*K, L_com+1, 1, 1, dtype=torch.complex64).to(device);  # b*K, L_com+1, 1, 1

        H_com=(gain*(v_f@v_a)).sum(1)/sqrt(L_com+1);  # b*K, N_C, M_com
        H_com=H_com.reshape(b, K, N_C, M_com).transpose(1, 2);  # b, N_C, K, M

        # H_rad
        theta_f=torch.rand(b, L_rad, 1).to(device)*2*pi;  # b, L_rad, 1
        rg_f=torch.arange(N_C).to(device).reshape(1, 1, N_C).float();  # 1, 1, N_C
        v_f=torch.exp(1j*(theta_f@rg_f)).unsqueeze(-1);  # b, L_rad, N_C, 1

        theta_Tx=torch.cos(theta_rad)*pi;  # b, L_rad, 1
        rg_Tx=torch.arange(M_com).to(device).reshape(1, 1, M_com).float();  # 1, 1, M_com
        v_Tx=torch.exp(1j*(theta_Tx@rg_Tx)).unsqueeze(-2);  # b, L_rad, 1, M_com

        gain=torch.randn(b, L_rad, 1, 1, dtype=torch.complex64).to(device);  # b, L_rad, 1, 1

        H_rad=(gain*(v_f@v_Tx)).sum(1)/sqrt(L_rad);  # b, N_C, M_com
        H_rad=H_rad.unsqueeze(-2);  # b, N_C, 1, M_com

        return H_com,  \
               H_rad;  

    H_com=torch.zeros(N_gen, N_C, K, M_com, dtype=torch.complex64);
    H_rad=torch.zeros(N_gen, N_C, 1, M_com, dtype=torch.complex64);
    cnt=0;
    for _ in range(N_gen//b):
        vcnt=cnt+b;

        H_com[cnt:vcnt],  \
        H_rad[cnt:vcnt]=genBatch(b);
        
        cnt=vcnt;

    if cnt!=N_gen:
        H_com[cnt:vcnt],  \
        H_rad[cnt:vcnt]=genBatch(N_gen-cnt);

    return H_com            .cpu(),  \
           H_rad            .cpu(),  \
           theta_share      .cpu(),  \
           theta_com        .cpu(),  \
           theta_rad        .cpu(),  \
           r_share          .cpu(),  \
           r_com            .cpu(),  \
           r_rad            .cpu(),  \
           theta_UE         .cpu(),  \
           r_UE             .cpu();



def main():
    env_init(42);
    H_com            ,  \
    H_rad            ,  \
    theta_share      ,  \
    theta_com        ,  \
    theta_rad        ,  \
    r_share          ,  \
    r_com            ,  \
    r_rad            ,  \
    theta_UE         ,  \
    r_UE             =genH(
        N_gen       =2560       , 
        b           =256        , 
        N_C         =32         , 
        K           =4          , 
        M_com       =32         , 
        L_share     =6          , 
        L_com       =8          , 
        L_rad       =12         , 
        R           =100        , 
        device      =0          , 
    );



if __name__=="__main__":
    logger("datasets.py");
    main();
    rmLogger();



