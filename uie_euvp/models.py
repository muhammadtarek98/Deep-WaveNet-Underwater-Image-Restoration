import argparse
import os, torchinfo
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

"""# Channel and Spatial Attention"""
class BasicConv(torch.nn.Module):
    def __init__(self, in_planes:int,
                 out_planes:int,
                 kernel_size:int,
                 stride:int=1,
                 padding:int=0,
                 dilation:int=1,
                 groups:int=1,
                 relu:bool=True,
                 bn:bool=False,
                 bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = torch.nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.bn = torch.nn.BatchNorm2d(num_features=out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = torch.nn.ReLU() if relu else None

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(torch.nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return x.view(x.size(0), -1)

class ChannelGate(torch.nn.Module):
    def __init__(self, gate_channels:int, reduction_ratio:int=16, pool_types:list=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = torch.nn.Sequential(
            Flatten(),
            torch.nn.Linear(in_features=gate_channels,out_features= gate_channels // reduction_ratio),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=gate_channels // reduction_ratio,out_features= gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x:torch.Tensor)->torch.Tensor:
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = torch.nn.functional.avg_pool2d( input=x, kernel_size=(x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = torch.nn.functional.max_pool2d( input=x, kernel_size=(x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = torch.nn.functional.lp_pool2d(input= x,norm_type= 2, kernel_size=(x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor:torch.Tensor)->torch.Tensor:
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(torch.nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return torch.cat( tensors=(torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(torch.nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(in_planes=2, out_planes=1,kernel_size= kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(torch.nn.Module):
    def __init__(self, gate_channels:int, reduction_ratio:int=16, pool_types:list=['avg', 'max'], no_spatial:bool=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class Conv2D_pxp(torch.nn.Module):

    def __init__(self, in_ch:int, out_ch:int, k:int,s:int,p:int):
        super(Conv2D_pxp, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k, stride=s, padding=p)
        self.bn = torch.nn.BatchNorm2d(num_features=out_ch)
        self.relu = torch.nn.PReLU(out_ch)

    def forward(self, input:torch.Tensor)->torch.Tensor:
        return self.relu(self.bn(self.conv(input)))


class CC_Module(torch.nn.Module):

    def __init__(self):
        super(CC_Module, self).__init__()   

        print("Color correction module for underwater images")

        self.layer1_1 = Conv2D_pxp(in_ch=1, out_ch=32, k=3,s=1,p=1)
        self.layer1_2 = Conv2D_pxp(in_ch=1, out_ch=32, k=5,s=1,p=2)
        self.layer1_3 = Conv2D_pxp(in_ch=1, out_ch=32, k=7,s=1,p=3)

        self.layer2_1 = Conv2D_pxp(in_ch=96, out_ch=32, k=3,s=1,p=1)
        self.layer2_2 = Conv2D_pxp(in_ch=96, out_ch=32, k=5,s=1,p=2)
        self.layer2_3 = Conv2D_pxp(in_ch=96, out_ch=32, k=7,s=1,p=3)
        
        self.local_attn_r = CBAM(gate_channels=64)
        self.local_attn_g = CBAM(gate_channels=64)
        self.local_attn_b = CBAM(gate_channels=64)

        self.layer3_1 = Conv2D_pxp(in_ch=192, out_ch=1, k=3,s=1,p=1)
        self.layer3_2 = Conv2D_pxp(in_ch=192, out_ch=1, k=5,s=1,p=2)
        self.layer3_3 = Conv2D_pxp(in_ch=192, out_ch=1, k=7,s=1,p=3)


        self.d_conv1 = torch.nn.ConvTranspose2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.d_bn1 = torch.nn.BatchNorm2d(num_features=32)
        self.d_relu1 = torch.nn.PReLU(32)

        self.global_attn_rgb = CBAM(35)

        self.d_conv2 = torch.nn.ConvTranspose2d(in_channels=35, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.d_bn2 = torch.nn.BatchNorm2d(num_features=3)
        self.d_relu2 = torch.nn.PReLU(3)


    def forward(self, input)->torch.Tensor:
        input_1 = torch.unsqueeze(input=input[:,0,:,:], dim=1)
        input_2 = torch.unsqueeze(input=input[:,1,:,:], dim=1)
        input_3 = torch.unsqueeze(input=input[:,2,:,:], dim=1)
        
        #layer 1
        l1_1=self.layer1_1(input_1) 
        l1_2=self.layer1_2(input_2) 
        l1_3=self.layer1_3(input_3) 

        #Input to layer 2
        input_l2=torch.cat(tensors=(l1_1,l1_2),dim=1)
        input_l2=torch.cat(tensors=(input_l2,l1_3),dim=1)
        
        #layer 2
        l2_1=self.layer2_1(input_l2) 
        l2_1 = self.local_attn_r(torch.cat(tensors=(l2_1, l1_1),dim=1))

        l2_2=self.layer2_2(input_l2) 
        l2_2 = self.local_attn_g(torch.cat(tensors=(l2_2, l1_2),dim=1))

        l2_3=self.layer2_3(input_l2) 
        l2_3 = self.local_attn_b(torch.cat(tensors=(l2_3, l1_3),dim=1))
        
        #Input to layer 3
        input_l3=torch.cat(tensors=(l2_1,l2_2),dim=1)
        input_l3=torch.cat(tensors=(input_l3,l2_3),dim=1)
        
        #layer 3
        l3_1=self.layer3_1(input_l3) 
        l3_2=self.layer3_2(input_l3) 
        l3_3=self.layer3_3(input_l3) 

        #input to decoder unit
        temp_d1=torch.add(input=input_1,other=l3_1)
        temp_d2=torch.add(input=input_2,other=l3_2)
        temp_d3=torch.add(input=input_3,other=l3_3)

        input_d1=torch.cat(tensors=(temp_d1,temp_d2),dim=1)
        input_d1=torch.cat(tensors=(input_d1,temp_d3),dim=1)
        
        #decoder
        output_d1=self.d_relu1(self.d_bn1(self.d_conv1(input_d1)))
        output_d1 = self.global_attn_rgb(torch.cat(tensors=(output_d1, input_d1),dim=1))
        final_output=self.d_relu2(self.d_bn2(self.d_conv2(output_d1)))
        
        return final_output 


inp=torch.randn(size=(1,3,720,720))
model=CC_Module()
torchinfo.summary(model=model,input_data=inp)
