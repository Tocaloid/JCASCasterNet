# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
from torch.linalg import svd, eig, inv, pinv
from torch.fft import fft, ifft

import numpy as np

import math
from math import sqrt, pi

from utils import calMeanPower, awgn, env_init, SELoss

from datasets import genH

from my_log import logger, rmLogger



#==================================================================================#
# utils
#==================================================================================#
class TransposeLayer(nn.Module):
    def __init__(
            self, 
            i, 
            j, 
        ):
        super().__init__();
        self.i=i;
        self.j=j;

    def forward(
            self, 
            x, 
        ):
        return x.transpose(self.i, self.j);



class MergeLayer(nn.Module):
    def __init__(
            self, 
            i, 
        ):
        super().__init__();
        self.i=i;

    def forward(
            self, 
            x, 
        ):
        assert self.i!=-1;
        assert self.i!=len(x.shape)-1;
        if self.i!=-2:
            return x.reshape(*x.shape[:self.i], -1, *x.shape[self.i+2:]);
        else:
            return x.reshape(*x.shape[:self.i], -1);



class SplitLayer(nn.Module):
    def __init__(
            self, 
            i, 
            n1=None, 
            n2=None, 
        ):
        super().__init__();
        assert (n1 is not None and n2 is None) or (n1 is None and n2 is not None);
        self.i=i;
        self.n1=n1;
        self.n2=n2;

    def forward(
            self, 
            x, 
        ):
        if self.i!=-1:
            if self.n2 is None:
                return x.reshape(*x.shape[:self.i], self.n1, -1, *x.shape[self.i+1:]);
            else:
                return x.reshape(*x.shape[:self.i], -1, self.n2, *x.shape[self.i+1:]);
        else:
            if self.n2 is None:
                return x.reshape(*x.shape[:self.i], self.n1, -1);
            else:
                return x.reshape(*x.shape[:self.i], -1, self.n2);



class PermuteLayer(nn.Module):
    def __init__(
            self, 
            permute_order, 
        ):
        super().__init__();
        self.permute_order=permute_order;
    
    def forward(
            self, 
            x, 
        ):
        return x.permute(self.permute_order);



class ReshapeLayer(nn.Module):
    def __init__(
            self, 
            shape, 
        ):
        super().__init__();
        self.shape=shape;
    
    def forward(
            self, 
            x, 
        ):
        return x.reshape(self.shape);



class RepeatLayer(nn.Module):
    def __init__(
            self, 
            times, 
        ):
        super().__init__();
        self.times=times;
    
    def forward(
            self, 
            x, 
        ):
        return x.repeat(self.times);



class Complex2realLayer(nn.Module):
    def __init__(
            self, 
            i, 
        ):
        super().__init__();
        self.i=i;
    
    def forward(
            self, 
            x, 
        ):
        return torch.cat((x.real, x.imag), dim=self.i);



class Real2complexLayer(nn.Module):
    def __init__(
            self, 
            i, 
        ):
        super().__init__();
        self.i=i;
    
    def forward(
            self, 
            x, 
        ):
        x=x.transpose(0, self.i);
        x=x[0]+1j*x[1];
        x=x.unsqueeze(0);
        x=x.transpose(0, self.i);
        if self.i==-1 or self.i==len(x.shape)-1:
            i_stable=self.i-1;
        else:
            i_stable=self.i;
        
        x=MergeLayer(i_stable)(x);
        return x;



class NormLayer(nn.Module):
    def __init__(
            self, 
            d, 
            c_index=-1, 
        ):
        super().__init__();
        self.norm=nn.Sequential(
            TransposeLayer(c_index, -1), 
            nn.LayerNorm(d), 
            TransposeLayer(c_index, -1), 
        );
    
    def forward(
            self, 
            x, 
        ):
        return self.norm(x);



#==================================================================================#
# model
#==================================================================================#
feedback_bits=2;

def calF_RF(
        Theta,  # *, 1, M, N_RF
    ):
    F_RF=torch.exp(1j*Theta);

    return F_RF;



def normF_BB(
        F_RF,  # *, 1, M, N_RF
        F_BB,  # *, N_C, N_RF, _
    ):
    N_C, N_RF=F_BB.shape[-3:-1];

    F=F_RF@F_BB;  # *, N_C, M, _
    power=(F.abs()**2).sum([-3, -2, -1], keepdim=True);  # *, 1, 1, 1
    F_BB=F_BB/power.sqrt()*sqrt(N_C*N_RF);

    return F_BB;



def calPilot(
        pilot_Theta,  # *, Q, 1  , M   , N_RF
        pilot_BB   ,  # *, Q, N_C, N_RF, 1
    ):
    _, Q, _  , M   , N_RF=pilot_Theta.shape;
    _, Q, N_C, N_RF, _   =pilot_BB   .shape;

    pilot_RF=calF_RF(pilot_Theta);  # *, Q, 1, M, N_RF
    pilot_BB=normF_BB(pilot_RF, pilot_BB);  # *, Q, N_C, N_RF, 1
    pilot=pilot_RF@pilot_BB;  # *, Q, N_C, M, 1
    pilot=pilot.transpose(1, 4).reshape(-1, N_C, M, Q);  # *, N_C, M, Q

    return pilot;



class PositionalEncoding(nn.Module):
    def __init__(
            self, 
            d_model, 
            dropout, 
            max_len, 
        ):
        super().__init__()
        position=torch.arange(max_len).unsqueeze(1);
        div_term=torch.exp(torch.arange(0, d_model, 2)*(-math.log(10000.0)/d_model));
        pe=torch.zeros(max_len, 1, d_model);
        pe[:, 0, 0::2]=torch.sin(position*div_term);
        pe[:, 0, 1::2]=torch.cos(position*div_term);
        self.register_buffer('pe', pe);
        self.dropout=nn.Dropout(p=dropout);
    
    def forward(
            self, 
            x, 
        ):
        x=x+self.pe[:x.size(0)];
        return self.dropout(x);



#=======================================================================================================================
#=======================================================================================================================
# Number to Bit Function Defining
def Num2Bit(Num, B):
    Num_ = Num.type(torch.uint8)
    def integer2bit(integer, num_bits=B):
        dtype = integer.type()
        device = integer.device;
        exponent_bits = -torch.arange(1-B, 1).type(dtype).to(device);
        exponent_bits = exponent_bits.repeat(integer.shape + (1,))
        out = integer.unsqueeze(-1) // 2 ** exponent_bits
        return out% 2
    bit = integer2bit(Num_)
    bit = bit.reshape(-1, Num_.shape[1] * B)
    return bit.type(torch.float32)
def Bit2Num(Bit, B):
    device=Bit.device;
    Bit_ = Bit.type(torch.float32)
    Bit_ = torch.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
    num = torch.zeros(Bit_[:, :, 0].shape).to(device)
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
    return num
#=======================================================================================================================
#=======================================================================================================================
# Quantization and Dequantization Layers Defining
class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2**B-1e-6;
        out = torch.round(x*step-0.5);
        out = Num2Bit(out, B)
        return out.type(x.type())
    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of constant arguments to forward must be None.
        # Gradient of a number is the sum of its B bits.
        b, _ = grad_output.shape
        grad_num = torch.sum(grad_output.reshape(b, -1, ctx.constant), dim=2) / ctx.constant
        return grad_num, None
class Dequantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2**B;
        out = Bit2Num(x, B)
        out = (out+0.5)/step;
        return out.type(x.type())
    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        # repeat the gradient of a Num for B time.
        b, c = grad_output.shape
        grad_output = grad_output.unsqueeze(2) / ctx.constant
        grad_bit = grad_output.expand(b, c, ctx.constant)
        return torch.reshape(grad_bit, (-1, c * ctx.constant)), None
class QuantizationLayer(nn.Module):
    def __init__(self, B):
        super(QuantizationLayer, self).__init__()
        self.B = B
    def forward(self, x):
        out = Quantization.apply(x, self.B)
        return out
class DequantizationLayer(nn.Module):
    def __init__(self, B):
        super(DequantizationLayer, self).__init__()
        self.B = B
    def forward(self, x):
        out = Dequantization.apply(x, self.B)
        return out



class SemanticEncoder(nn.Module):
    def __init__(
            self        , 
            N_C         , 
            M_com       , 
            N_RF_com    , 
            Q_bsc       , 
            Q_ehc       , 
            K           , 
            B           , 
            SNR_com     , 
            SNR_rad     , 
            d_model     , 
        ):
        super().__init__();

        Q=Q_bsc+Q_ehc;

        self.register_buffer("N_C"         , torch.tensor(N_C         ));
        self.register_buffer("M_com"       , torch.tensor(M_com       ));
        self.register_buffer("N_RF_com"    , torch.tensor(N_RF_com    ));
        self.register_buffer("Q_bsc"       , torch.tensor(Q_bsc       ));
        self.register_buffer("Q_ehc"       , torch.tensor(Q_ehc       ));
        self.register_buffer("K"           , torch.tensor(K           ));
        self.register_buffer("B"           , torch.tensor(B           ));
        self.register_buffer("SNR_com"     , torch.tensor(SNR_com     ));
        self.register_buffer("SNR_rad"     , torch.tensor(SNR_rad     ));
        self.register_buffer("d_model"     , torch.tensor(d_model     ));

        d_model=d_model//K;

        self.embed=nn.Sequential(  # b, N_C, K, Q
            TransposeLayer(1, 2),  # b, K, N_C, Q
            ReshapeLayer([-1, N_C, Q]),  # b*K, N_C, Q
            Complex2realLayer(-1),  # b*K, N_C, 2*Q
            nn.Linear(2*Q, d_model),  # b*K, N_C, d_model
        );  # b*K, N_C, d_model
        self.backbone=nn.Sequential(  # b*K, N_C, d_model
            TransposeLayer(0, 1),  # N_C, b*K, d_model
            PositionalEncoding(d_model, 0., N_C), 
            nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=d_model, 
                    nhead=8, 
                    dim_feedforward=4*d_model, 
                    dropout=0., 
                    activation="gelu", 
                    norm_first=False, 
                ), 
                num_layers=6, 
            ), 
            TransposeLayer(0, 1),  # b*K, N_C, d_model
        );  # b*K, N_C, d_model
        self.head=nn.Sequential(  # b*K, N_C, d_model
            ReshapeLayer([-1, N_C*d_model]),  # b*K, N_C*d_model
            nn.Linear(N_C*d_model, B//feedback_bits),  # b*K, B//feedback_bits
            nn.BatchNorm1d(B//feedback_bits), 
            nn.Sigmoid(), 
            QuantizationLayer(feedback_bits), 
            ReshapeLayer([-1, K, B]),  # b, K, B
        );  # b, K, B

    def forward(
            self        , 
            measurement ,  # b, N_C, K, Q
        ):
        x=self.embed(measurement);  # b*K, N_C, d_model
        x=self.backbone(x);
        feedback=self.head(x);  # b, K, B

        return feedback;



class SemanticDecoder(nn.Module):
    def __init__(
            self        , 
            N_C         , 
            M_com       , 
            N_RF_com    , 
            Q_bsc       , 
            Q_ehc       , 
            K           , 
            B           , 
            SNR_com     , 
            SNR_rad     , 
            d_model     , 
        ):
        super().__init__();

        Q=Q_bsc+Q_ehc;

        self.register_buffer("N_C"         , torch.tensor(N_C         ));
        self.register_buffer("M_com"       , torch.tensor(M_com       ));
        self.register_buffer("N_RF_com"    , torch.tensor(N_RF_com    ));
        self.register_buffer("Q_bsc"       , torch.tensor(Q_bsc       ));
        self.register_buffer("Q_ehc"       , torch.tensor(Q_ehc       ));
        self.register_buffer("K"           , torch.tensor(K           ));
        self.register_buffer("B"           , torch.tensor(B           ));
        self.register_buffer("SNR_com"     , torch.tensor(SNR_com     ));
        self.register_buffer("SNR_rad"     , torch.tensor(SNR_rad     ));
        self.register_buffer("d_model"     , torch.tensor(d_model     ));

        d_model=d_model//K;

        self.embed_pilot_Theta=nn.Sequential(  # b, N_C, M_com, Q
            Complex2realLayer(-1),  # b, N_C, M_com, 2*Q
            TransposeLayer(1, 2),  # b, M_com, N_C, 2*Q
            ReshapeLayer([-1, M_com, N_C*2*Q]),  # b, M_com, N_C*2*Q
            nn.Linear(N_C*2*Q, d_model),  # b, M_com, d_model
            ReshapeLayer([-1, 1, M_com, d_model]),  # b, 1, M_com, d_model
            RepeatLayer([1, K, 1, 1]),  # b, K, M_com, d_model
            ReshapeLayer([-1, M_com, d_model]),  # b*K, M_com, d_model
        );  # b*K, M_com, d_model
        self.embed_echo_Theta=nn.Sequential(  # b, N_C, 1, Q
            TransposeLayer(1, 3),  # b, Q, 1, N_C
            Complex2realLayer(-1),  # b, Q, 1, 2*N_C
            ReshapeLayer([-1, Q, 1*2*N_C]),  # b, Q, 1*2*N_C
            nn.Linear(1*2*N_C, d_model),  # b, Q, d_model
            nn.Conv1d(Q, M_com, 1),  # b, M_com, d_model
            ReshapeLayer([-1, 1, M_com, d_model]),  # b, 1, M_com, d_model
            RepeatLayer([1, K, 1, 1]),  # b, K, M_com, d_model
            ReshapeLayer([-1, M_com, d_model]),  # b*K, M_com, d_model
        );  # b*K, M_com, d_model
        self.embed_feedback_Theta=nn.Sequential(  # b, K, B
            ReshapeLayer([-1, B]),  # b*K, B
            DequantizationLayer(feedback_bits), 
            nn.Linear(B//feedback_bits, M_com*d_model),  # b*K, M_com*d_model
            ReshapeLayer([-1, M_com, d_model]),  # b*K, M_com, d_model
        );  # b*K, M_com, d_model
        self.backbone_Theta=nn.Sequential(  # b*K, M_com, d_model
            TransposeLayer(0, 1),  # M_com, b*K, d_model
            PositionalEncoding(d_model, 0., M_com), 
            nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=d_model, 
                    nhead=8, 
                    dim_feedforward=4*d_model, 
                    dropout=0., 
                    activation="gelu", 
                    norm_first=False, 
                ), 
                num_layers=6, 
            ), 
            TransposeLayer(0, 1),  # b*K, M_com, d_model
        );  # b*K, M_com, d_model
        self.F_Theta_head=nn.Sequential(  # b*K, M_com, d_model
            nn.Linear(d_model, 1),  # b*K, M_com, 1
            ReshapeLayer([-1, K, M_com, 1]),  # b, K, M_com, 1
            TransposeLayer(1, 3),  # b, 1, M_com, K
        );  # b, 1, M_com, N_RF_com (K==N_RF_com)

        d_model=d_model*K;

        self.embed_pilot_BB=nn.Sequential(  # b, N_C, M_com, Q
            Complex2realLayer(-1),  # b, N_C, M_com, 2*Q
            ReshapeLayer([-1, N_C, M_com*2*Q]),  # b, N_C, M_com*2*Q
            nn.Linear(M_com*2*Q, d_model),  # b, N_C, d_model
        );  # b, N_C, d_model
        self.embed_echo_BB=nn.Sequential(  # b, N_C, 1, Q
            Complex2realLayer(-1),  # b, N_C, 1, 2*Q
            ReshapeLayer([-1, N_C, 1*2*Q]),  # b, N_C, 1*2*Q
            nn.Linear(1*2*Q, d_model),  # b, N_C, d_model
        );  # b, N_C, d_model
        self.embed_feedback_BB=nn.Sequential(  # b, K, B
            ReshapeLayer([-1, B]),  # b*K, B
            DequantizationLayer(feedback_bits), 
            ReshapeLayer([-1, K*B//feedback_bits]),  # b, K*B//feedback_bits
            nn.Linear(K*B//feedback_bits, N_C*d_model),  # b, N_C*d_model
            ReshapeLayer([-1, N_C, d_model]),  # b, N_C, d_model
        );  # b, N_C, d_model
        self.embed_F_RF=nn.Sequential(  # b, 1, M_com, N_RF_com
            Complex2realLayer(-1),  # b, 1, M_com, 2*N_RF_com
            ReshapeLayer([-1, 1, M_com*2*N_RF_com]),  # b, 1, M_com*2*N_RF_com
            nn.Linear(M_com*2*N_RF_com, d_model),  # b, 1, d_model
        );  # b, 1, d_model
        self.backbone_BB=nn.Sequential(  # b, N_C+1, d_model
            TransposeLayer(0, 1),  # N_C+1, b, d_model
            PositionalEncoding(d_model, 0., N_C+1), 
            nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=d_model, 
                    nhead=8, 
                    dim_feedforward=4*d_model, 
                    dropout=0., 
                    activation="gelu", 
                    norm_first=False, 
                ), 
                num_layers=6, 
            ), 
            TransposeLayer(0, 1),  # b, N_C+1, d_model
        );  # b, N_C+1, d_model
        self.F_BB_head=nn.Sequential(  # b, N_C, d_model
            nn.Linear(d_model, N_RF_com*K*2),  # b, N_C, N_RF_com*K*2
            ReshapeLayer([-1, N_C, N_RF_com, K, 2]),  # b, N_C, N_RF_com, K, 2
            Real2complexLayer(-1),  # b, N_C, N_RF_com, K
        );  # b, N_C, N_RF_com, K

    def forward(
            self        , 
            pilot       ,  # b, N_C, M_com, Q
            echo        ,  # b, N_C, 1    , Q
            feedback    ,  # b, K, B
        ):
        b, N_C, M_com, Q=pilot.shape;

        z_pilot_Theta=self.embed_pilot_Theta(pilot.detach());  # b*K, M_com, d_model
        z_echo_Theta=self.embed_echo_Theta(echo.detach());  # b*K, M_com, d_model
        z_feedback_Theta=self.embed_feedback_Theta(feedback);  # b*K, M_com, d_model
        
        z_Theta=z_pilot_Theta+z_echo_Theta+z_feedback_Theta;  # b*K, M_com, d_model
        z_Theta=self.backbone_Theta(z_Theta);
        F_RF=calF_RF(self.F_Theta_head(z_Theta[:, 0:M_com]));  # b, 1, M_com, N_RF_com

        z_pilot_BB=self.embed_pilot_BB(pilot);  # b, N_C, d_model
        z_echo_BB=self.embed_echo_BB(echo);  # b, N_C, d_model
        z_feedback_BB=self.embed_feedback_BB(feedback);  # b, N_C, d_model
        z_F_RF=self.embed_F_RF(F_RF);  # b, 1, d_model

        z_BB=torch.cat((
            z_pilot_BB+z_echo_BB+z_feedback_BB, 
            z_F_RF, 
        ), dim=1);  # b, N_C+1, d_model
        z_BB=self.backbone_BB(z_BB);
        F_BB=normF_BB(F_RF, self.F_BB_head(z_BB[:, 0:N_C]));  # b, N_C, N_RF_com, K
        
        F=F_RF@F_BB;  # b, N_C, M_com, K

        return F;



class ComCasterNet(nn.Module):
    def __init__(
            self        , 
            N_C         , 
            M_com       , 
            N_RF_com    , 
            Q_bsc       , 
            Q_ehc       , 
            K           , 
            B           , 
            SNR_com     , 
            SNR_rad     , 
            d_model     , 
        ):
        super().__init__();

        Q=Q_bsc+Q_ehc;

        self.register_buffer("N_C"         , torch.tensor(N_C         ));
        self.register_buffer("M_com"       , torch.tensor(M_com       ));
        self.register_buffer("N_RF_com"    , torch.tensor(N_RF_com    ));
        self.register_buffer("Q_bsc"       , torch.tensor(Q_bsc       ));
        self.register_buffer("Q_ehc"       , torch.tensor(Q_ehc       ));
        self.register_buffer("K"           , torch.tensor(K           ));
        self.register_buffer("B"           , torch.tensor(B           ));
        self.register_buffer("SNR_com"     , torch.tensor(SNR_com     ));
        self.register_buffer("SNR_rad"     , torch.tensor(SNR_rad     ));
        self.register_buffer("d_model"     , torch.tensor(d_model     ));

        self.register_buffer("echo", torch.zeros(1, N_C, 1, Q).to(torch.complex64));

        self.pilot_Theta=nn.Parameter(torch.rand (1, Q, 1  , M_com   , N_RF_com)*2*pi);
        self.pilot_BB   =nn.Parameter(torch.randn(1, Q, N_C, N_RF_com, 1       , dtype=torch.complex64));

        self.SEnc=SemanticEncoder(
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
        );

        self.SDec=SemanticDecoder(
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
        );

    def forward(
            self        , 
            H_com       ,  # b, N_C, K, M_com
        ):
        b=H_com.shape[0];

        SNR_com=self.SNR_com;

        pilot=calPilot(
            pilot_Theta=self.pilot_Theta, 
            pilot_BB   =self.pilot_BB   ,
        );  # 1, N_C, M_com, Q
        
        measurement=H_com@pilot;  # b, N_C, K, Q
        measurement=awgn(measurement, 1, pilot, SNR_com);

        feedback=self.SEnc(measurement);  # b, K, B
        F=self.SDec(
            pilot.repeat(b, 1, 1, 1),  # b, N_C, M_com, Q
            self.echo.repeat(b, 1, 1, 1),  # b, N_C, 1, Q
            feedback, 
        );  # b, N_C, M_com, N_RF_com

        return F;



class EnhancedPilotDesignNet(nn.Module):
    def __init__(
            self        , 
            N_C         , 
            M_com       , 
            N_RF_com    , 
            Q_bsc       , 
            Q_ehc       , 
            K           , 
            B           , 
            SNR_com     , 
            SNR_rad     , 
            d_model     , 
        ):
        super().__init__();

        self.register_buffer("N_C"         , torch.tensor(N_C         ));
        self.register_buffer("M_com"       , torch.tensor(M_com       ));
        self.register_buffer("N_RF_com"    , torch.tensor(N_RF_com    ));
        self.register_buffer("Q_bsc"       , torch.tensor(Q_bsc       ));
        self.register_buffer("Q_ehc"       , torch.tensor(Q_ehc       ));
        self.register_buffer("K"           , torch.tensor(K           ));
        self.register_buffer("B"           , torch.tensor(B           ));
        self.register_buffer("SNR_com"     , torch.tensor(SNR_com     ));
        self.register_buffer("SNR_rad"     , torch.tensor(SNR_rad     ));
        self.register_buffer("d_model"     , torch.tensor(d_model     ));

        self.embed_pilot=nn.Sequential(  # b, N_C, M_com, Q_bsc
            Complex2realLayer(-1),  # b, N_C, M_com, 2*Q_bsc
            ReshapeLayer([-1, N_C, M_com*2*Q_bsc]),  # b, N_C, M_com*2*Q_bsc
            nn.Linear(M_com*2*Q_bsc, d_model),  # b, N_C, d_model
        );  # b, N_C, d_model
        self.embed_echo=nn.Sequential(  # b, N_C, 1, Q_bsc
            Complex2realLayer(-1),  # b, N_C, 1, 2*Q_bsc
            ReshapeLayer([-1, N_C, 1*2*Q_bsc]),  # b, N_C, 1*2*Q_bsc
            nn.Linear(1*2*Q_bsc, d_model),  # b, N_C, d_model
        );  # b, N_C, d_model
        self.backbone=nn.Sequential(  # b, N_C, d_model
            TransposeLayer(0, 1),  # N_C, b, d_model
            PositionalEncoding(d_model, 0., N_C), 
            nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=d_model, 
                    nhead=8, 
                    dim_feedforward=4*d_model, 
                    dropout=0., 
                    activation="gelu", 
                    norm_first=False, 
                ), 
                num_layers=6, 
            ), 
            TransposeLayer(0, 1),  # b, N_C, d_model
        );  # b, N_C, d_model
        self.p_Theta_head=nn.Sequential(  # b, N_C, d_model
            ReshapeLayer([-1, N_C*d_model]),  # b, N_C*d_model
            nn.Linear(N_C*d_model, Q_ehc*1*M_com*N_RF_com),  # b, Q_ehc*1*M_com*N_RF_com
            ReshapeLayer([-1, Q_ehc, 1, M_com, N_RF_com]),  # b, Q_ehc, 1, M_com, N_RF_com
        );  # b, Q_ehc, 1, M_com, N_RF_com
        self.p_BB_head=nn.Sequential(  # b, N_C, d_model
            ReshapeLayer([-1, N_C*d_model]),  # b, N_C*d_model
            nn.Linear(N_C*d_model, Q_ehc*N_C*N_RF_com*1*2),  # b, Q_ehc*N_C*N_RF_com*1*2
            ReshapeLayer([-1, Q_ehc, N_C, N_RF_com, 1, 2]),  # b, Q_ehc, N_C, N_RF_com, 1, 2
            Real2complexLayer(-1),  # b, Q_ehc, N_C, N_RF_com, 1
        );  # b, Q_ehc, N_C, N_RF_com, 1

    def forward(
            self        , 
            bsc_pilot   ,  # 1, N_C, M_com, Q_bsc
            bsc_echo    ,  # b, N_C, 1    , Q_bsc
        ):
        z_pilot=self.embed_pilot(bsc_pilot.detach());  # b, N_C, d_model
        z_echo=self.embed_echo(bsc_echo);  # b, N_C, d_model

        z=z_pilot+z_echo;
        z=self.backbone(z);
        ehc_pilot=calPilot(
            pilot_Theta=self.p_Theta_head(z),  # b, Q_ehc, 1  , M_com   , N_RF_com
            pilot_BB   =self.p_BB_head   (z),  # b, Q_ehc, N_C, N_RF_com, 1
        );  # b, N_C, M_com, Q_ehc

        return ehc_pilot;



class ChannelSemanticReconstructionNet(nn.Module):
    def __init__(
            self        , 
            N_C         , 
            M_com       , 
            N_RF_com    , 
            Q_bsc       , 
            Q_ehc       , 
            K           , 
            B           , 
            SNR_com     , 
            SNR_rad     , 
            d_model     , 
        ):
        super().__init__();

        Q=Q_bsc+Q_ehc;

        self.register_buffer("N_C"         , torch.tensor(N_C         ));
        self.register_buffer("M_com"       , torch.tensor(M_com       ));
        self.register_buffer("N_RF_com"    , torch.tensor(N_RF_com    ));
        self.register_buffer("Q_bsc"       , torch.tensor(Q_bsc       ));
        self.register_buffer("Q_ehc"       , torch.tensor(Q_ehc       ));
        self.register_buffer("K"           , torch.tensor(K           ));
        self.register_buffer("B"           , torch.tensor(B           ));
        self.register_buffer("SNR_com"     , torch.tensor(SNR_com     ));
        self.register_buffer("SNR_rad"     , torch.tensor(SNR_rad     ));
        self.register_buffer("d_model"     , torch.tensor(d_model     ));

        self.embed_pilot=nn.Sequential(  # b, N_C, M_com, Q
            Complex2realLayer(-1),  # b, N_C, M_com, 2*Q
            ReshapeLayer([-1, N_C, M_com*2*Q]),  # b, N_C, M_com*2*Q
            nn.Linear(M_com*2*Q, d_model),  # b, N_C, d_model
        );  # b, N_C, d_model
        self.embed_echo=nn.Sequential(  # b, N_C, 1, Q
            Complex2realLayer(-1),  # b, N_C, 1, 2*Q
            ReshapeLayer([-1, N_C, 1*2*Q]),  # b, N_C, 1*2*Q
            nn.Linear(1*2*Q, d_model),  # b, N_C, d_model
        );  # b, N_C, d_model
        self.embed_feedback=nn.Sequential(  # b, K, B
            ReshapeLayer([-1, B]),  # b*K, B
            DequantizationLayer(feedback_bits), 
            ReshapeLayer([-1, K*B//feedback_bits]),  # b, K*B//feedback_bits
            nn.Linear(K*B//feedback_bits, N_C*d_model),  # b, N_C*d_model
            ReshapeLayer([-1, N_C, d_model]),  # b, N_C, d_model
        );  # b, N_C, d_model
        self.backbone=nn.Sequential(  # b, N_C, d_model
            TransposeLayer(0, 1),  # N_C, b, d_model
            PositionalEncoding(d_model, 0., N_C), 
            nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=d_model, 
                    nhead=8, 
                    dim_feedforward=4*d_model, 
                    dropout=0., 
                    activation="gelu", 
                    norm_first=False, 
                ), 
                num_layers=6, 
            ), 
            TransposeLayer(0, 1),  # b, N_C, d_model
        );  # b, N_C, d_model
        self.head=nn.Sequential(  # b, N_C, d_model
            nn.Linear(d_model, 1*M_com*2),  # b, N_C, 1*M_com*2
            ReshapeLayer([-1, N_C, 1, M_com, 2]),  # b, N_C, 1, M_com, 2
            Real2complexLayer(-1),  # b, N_C, 1, M_com
        );  # b, N_C, 1, M_com

    def forward(
            self        , 
            pilot       ,  # b, N_C, M_com, Q
            echo        ,  # b, N_C, 1    , Q
            feedback    ,  # b, K, B
        ):
        b, N_C, M_com, Q=pilot.shape;
        
        z_pilot=self.embed_pilot(pilot.detach());  # b, N_C, d_model
        z_echo=self.embed_echo(echo);  # b, N_C, d_model
        z_feedback=self.embed_feedback(feedback.detach());  # b, N_C, d_model

        z=z_pilot+z_echo+z_feedback;  # b, N_C+1, d_model
        z=self.backbone(z[:, 0:N_C]);
        H_rad_est=self.head(z);  # b, N_C, 1, M_com

        return H_rad_est;



class RadCasterNet(nn.Module):
    def __init__(
            self        , 
            N_C         , 
            M_com       , 
            N_RF_com    , 
            Q_bsc       , 
            Q_ehc       , 
            K           , 
            B           , 
            SNR_com     , 
            SNR_rad     , 
            d_model     , 
        ):
        super().__init__();

        self.register_buffer("N_C"         , torch.tensor(N_C         ));
        self.register_buffer("M_com"       , torch.tensor(M_com       ));
        self.register_buffer("N_RF_com"    , torch.tensor(N_RF_com    ));
        self.register_buffer("Q_bsc"       , torch.tensor(Q_bsc       ));
        self.register_buffer("Q_ehc"       , torch.tensor(Q_ehc       ));
        self.register_buffer("K"           , torch.tensor(K           ));
        self.register_buffer("B"           , torch.tensor(B           ));
        self.register_buffer("SNR_com"     , torch.tensor(SNR_com     ));
        self.register_buffer("SNR_rad"     , torch.tensor(SNR_rad     ));
        self.register_buffer("d_model"     , torch.tensor(d_model     ));

        self.register_buffer("feedback", torch.zeros(1, K, B).float());

        self.bsc_pilot_Theta=nn.Parameter(torch.rand (1, Q_bsc, 1  , M_com   , N_RF_com)*2*pi);
        self.bsc_pilot_BB   =nn.Parameter(torch.randn(1, Q_bsc, N_C, N_RF_com, 1       , dtype=torch.complex64));

        self.EPDN=EnhancedPilotDesignNet(
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
        );
    
        self.CSRN=ChannelSemanticReconstructionNet(
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
        );
    
    def bsc_pilot(
            self        , 
        ):
        return calPilot(
            pilot_Theta=self.bsc_pilot_Theta, 
            pilot_BB   =self.bsc_pilot_BB   , 
        );  # 1, N_C, M_com, Q_bsc
    
    def forward(
            self        , 
            H_rad       ,  # b, N_C, 1, M_com
        ):
        b=H_rad.shape[0];

        SNR_rad=self.SNR_rad;

        # stage 1
        bsc_pilot=self.bsc_pilot();  # 1, N_C, M_com, Q_bsc

        bsc_echo=H_rad@bsc_pilot;  # b, N_C, 1, Q_bsc
        bsc_echo=awgn(bsc_echo, 1, bsc_pilot, SNR_rad);

        # stage 2
        ehc_pilot=self.EPDN(bsc_pilot, bsc_echo);  # b, N_C, M_com, Q_ehc

        ehc_echo=H_rad@ehc_pilot;  # b, N_C, 1, Q_ehc
        ehc_echo=awgn(ehc_echo, 1, ehc_pilot, SNR_rad);

        # Cat
        pilot=torch.cat((bsc_pilot.repeat(b, 1, 1, 1), ehc_pilot), dim=-1);  # b, N_C, M_com, Q
        echo=torch.cat((bsc_echo, ehc_echo), dim=-1);  # b, N_C, N_RF_rad, Q

        H_rad_est=self.CSRN(pilot, echo, self.feedback.repeat(b, 1, 1));  # b, N_C, 1, M_com

        return H_rad_est;



class JCASCasterNet(nn.Module):
    def __init__(
            self        , 
            N_C         , 
            M_com       , 
            N_RF_com    , 
            Q_bsc       , 
            Q_ehc       , 
            K           , 
            B           , 
            SNR_com     , 
            SNR_rad     , 
            d_model     , 
        ):
        super().__init__();

        self.register_buffer("N_C"         , torch.tensor(N_C         ));
        self.register_buffer("M_com"       , torch.tensor(M_com       ));
        self.register_buffer("N_RF_com"    , torch.tensor(N_RF_com    ));
        self.register_buffer("Q_bsc"       , torch.tensor(Q_bsc       ));
        self.register_buffer("Q_ehc"       , torch.tensor(Q_ehc       ));
        self.register_buffer("K"           , torch.tensor(K           ));
        self.register_buffer("B"           , torch.tensor(B           ));
        self.register_buffer("SNR_com"     , torch.tensor(SNR_com     ));
        self.register_buffer("SNR_rad"     , torch.tensor(SNR_rad     ));
        self.register_buffer("d_model"     , torch.tensor(d_model     ));
        
        self.CN=ComCasterNet(
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
        );

        self.SN=RadCasterNet(
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
        );

    def forward(
            self        , 
            H_com       ,  # b, N_C, K, M_com
            H_rad       ,  # b, N_C, 1, M_com
        ):
        b=H_rad.shape[0];

        SNR_com=self.CN.SNR_com;
        SNR_rad=self.SN.SNR_rad;

        # stage 1
        bsc_pilot=self.SN.bsc_pilot();  # 1, N_C, M_com, Q_bsc

        bsc_echo=H_rad@bsc_pilot;  # b, N_C, 1, Q_bsc
        bsc_echo=awgn(bsc_echo, 1, bsc_pilot, SNR_rad);
        
        bsc_measurement=H_com@bsc_pilot;  # b, N_C, K, Q_bsc
        bsc_measurement=awgn(bsc_measurement, 1, bsc_pilot, SNR_com);

        # stage 2
        ehc_pilot=self.SN.EPDN(bsc_pilot, bsc_echo);  # b, N_C, M_com, Q_ehc

        ehc_echo=H_rad@ehc_pilot;  # b, N_C, 1, Q_ehc
        ehc_echo=awgn(ehc_echo, 1, ehc_pilot, SNR_rad);

        ehc_measurement=H_com@ehc_pilot;  # b, N_C, K, Q_ehc
        ehc_measurement=awgn(ehc_measurement, 1, ehc_pilot, SNR_com);

        # Feedback
        measurement=torch.cat((bsc_measurement, ehc_measurement), dim=-1);  # b, N_C, K, Q
        feedback=self.CN.SEnc(measurement);  # b, K, B

        # Cat
        pilot=torch.cat((bsc_pilot.repeat(b, 1, 1, 1), ehc_pilot), dim=-1);  # b, N_C, M_com, Q
        echo=torch.cat((bsc_echo, ehc_echo), dim=-1);  # b, N_C, N_RF_rad, Q

        # Com
        F=self.CN.SDec(pilot, echo, feedback);  # b, N_C, M_com, N_RF_com

        # Rad
        H_rad_est=self.SN.CSRN(pilot, echo, feedback);  # b, N_C, 1, M_com

        return F, H_rad_est;



class MUSIC(nn.Module):
    def __init__(
            self, 
        ):
        super().__init__();

    def forward(
            self        , 
            H           ,  # b, N_C, 1, M
            eps         ,  # [deg]
        ):
        with torch.no_grad():
            b, N_C, _, M=H.shape;
            device=H.device;
            L=6;

            X=H.squeeze().transpose(-2, -1);  # b, M, N_C
            if b==1:
                X=X.unsqueeze(0);

            R=X@X.mH;  # b, M, M
            e, V=eig(R);
            idx=e.abs().argsort(dim=-1, descending=True);  # b, M

            idx_n=idx[:, L:].unsqueeze(-2).repeat(1, M, 1);  # b, M, M-L
            V_n=V.gather(  # b, M, M
                dim=-1, 
                index=idx_n, 
            );  # b, M, M-L

            R_n=V_n@V_n.mH;  # b, M, M
            R_n=R_n.unsqueeze(1);  # b, 1, M, M

            theta=torch.arange(0, 180, eps).to(device);  # [deg]
            theta=theta/180*pi;  # [rad]
            theta=torch.cos(theta)*pi;
            theta=theta.unsqueeze(0).unsqueeze(-1);  # 1, Num, 1
            rg=torch.arange(M).to(device).reshape(1, 1, M).float();  # 1, 1, M
            A=torch.exp(1j*(theta@rg)).unsqueeze(-1);  # 1, Num, M, 1

            P=1/(A.mH@R_n@A).squeeze();  # b, Num

            return P.real;

        


















