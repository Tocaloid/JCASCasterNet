# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
from torch.linalg import svd, eig, inv, pinv
from torch.fft import fft, ifft

import numpy as np

import math
from math import sqrt, pi

import os
from copy import deepcopy

from model import JCASCasterNet, ComCasterNet, RadCasterNet

from datasets import genH

from utils import env_init          ,  \
                  calMeanPower      ,  \
                  SELoss            ,  \
                  RMSELoss          ,  \
                  NMSELoss          ,  \
                  CosLoss           ,  \
                  theta2label

from my_log import logger, rmLogger

from tqdm import tqdm



def main():
    env_init(42);
    N_C         =32          ;
    M_com       =32          ;
    N_RF_com    =2           ;
    Q_bsc       =2           ;
    Q_ehc       =2           ;
    K           =N_RF_com    ;
    B           =16          ;
    SNR_com     =10          ;
    SNR_rad     =10          ;
    d_model     =256         ;
    device      =0           ;
    net=JCASCasterNet(
        N_C         =N_C         , 
        M_com       =M_com       , 
        N_RF_com    =N_RF_com    , 
        Q_bsc       =Q_bsc       , 
        Q_ehc       =Q_ehc       , 
        K           =K           , 
        B           =B           , 
        SNR_com     =SNR_com     , 
        SNR_rad     =SNR_rad     , 
        d_model     =d_model     , 
    ).to(device);
    opt=torch.optim.AdamW(
        net.parameters(), 
        lr=1e-4, 
    );

    b           =256         ;
    L_share     =6           ;
    L_com       =3*K         ;
    L_rad       =6           ;
    R           =100         ;
    SNR         =10          ;

    acc_num=8;
    interval=100;
    for i in tqdm(range(1_000_000)):
        opt.zero_grad();
        net.train();

        for _ in range(acc_num):
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
                N_gen       =b           , 
                b           =b           , 
                N_C         =N_C         , 
                K           =K           , 
                M_com       =M_com       , 
                L_share     =L_share     , 
                L_com       =L_com       , 
                L_rad       =L_rad       , 
                R           =R           , 
                device      =device      , 
            );

            H_com=H_com.to(device);
            H_rad=H_rad.to(device);

            F, H_rad_est=net(
                H_com=H_com, 
                H_rad=H_rad, 
            );

            loss=0;

            loss_SE=SELoss(H_com, F, SNR);
            loss=loss+loss_SE;

            loss_Cos=CosLoss(H_rad_est, H_rad);
            loss=loss+loss_Cos;

            loss=loss/acc_num;
            loss.backward();
        
        opt.step();
            
        if (i+1)%interval==0:
            net.eval();
            N_eval=2;

            loss_SE_acc=0;
            loss_Cos_acc=0;
            for _ in range(N_eval):
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
                    N_gen       =10*b        , 
                    b           =10*b        , 
                    N_C         =N_C         , 
                    K           =K           , 
                    M_com       =M_com       , 
                    L_share     =L_share     , 
                    L_com       =L_com       , 
                    L_rad       =L_rad       , 
                    R           =R           , 
                    device      =device      , 
                );

                H_com=H_com.to(device);
                H_rad=H_rad.to(device);

                with torch.no_grad():
                    F, H_rad_est=net(
                        H_com=H_com, 
                        H_rad=H_rad, 
                    );

                    loss_SE_acc+=SELoss(H_com, F, SNR).item()/N_eval;
                    loss_Cos_acc+=CosLoss(H_rad_est, H_rad).item()/N_eval;
            

            logger("SE: {}".format(-loss_SE_acc));
            logger("Cos: {}".format(-loss_Cos_acc));
            logger("lr: {}".format(opt.param_groups[0]['lr']));

            folder_path="./BaseCKPT_Cos";
            if not os.path.exists(folder_path):
                os.makedirs(folder_path);

            save_path=folder_path+"/ckpt"+str(i+1)+".pth.tar";
            torch.save(
                {'state_dict': net.state_dict(), 
                 'opt_dict': opt.state_dict(), 
                 }, save_path);



if __name__=="__main__":
    logger("main.py");
    main();
    rmLogger();