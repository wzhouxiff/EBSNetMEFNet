import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

import pdb

class conv_block(nn.Module):
    def __init__(self, inc , outc, kernel_size=3, padding=1, stride=1, use_bias=True, activation=nn.ReLU(inplace=True), is_BN=False):
        super(conv_block, self).__init__()
        if is_BN:
            self.conv = nn.Sequential(OrderedDict([
                ("conv", nn.Conv2d(inc, outc, kernel_size, padding=padding, stride=stride, bias=use_bias)),
                ("bn", nn.BatchNorm2d(outc)),
                ("act", activation)
            ]))
        elif activation is not None:
            self.conv = nn.Sequential(OrderedDict([
                ("conv", nn.Conv2d(inc, outc, kernel_size, padding=padding, stride=stride, bias=use_bias)),
                ("act", activation)
            ]))
        else:
            self.conv = nn.Sequential(OrderedDict([
                ("conv", nn.Conv2d(inc, outc, kernel_size, padding=padding, stride=stride, bias=use_bias)),
            ]))

    def forward(self, input):
        return self.conv(input)

class fc(nn.Module):
    def __init__(self, inc, outc, activation=None, is_BN=False):
        super(fc, self).__init__()
        if is_BN:
            self.fc = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(inc, outc)),
                ("bn", nn.BatchNorm1d(outc)),
                ("act", activation),
            ]))
        elif activation is not None:
            self.fc = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(inc, outc)),
                ("act", activation),
            ]))
        else:
            self.fc = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(inc, outc)),
            ]))

    def forward(self, input):
        return self.fc(input)

class Guide(nn.Module):
    '''
    pointwise neural net
    '''
    def __init__(self, mode="PointwiseNN", inc=3, is_BN=True):
        super(Guide, self).__init__()
        self.mode = mode
        self.inc = inc

        if mode == "PointwiseNN":
            self.mode = "PointwiseNN"
            self.conv1 = conv_block(self.inc, 16, kernel_size=1, padding=0, is_BN=is_BN)
            self.conv2 = conv_block(16, 1, kernel_size=1, padding=0, activation=nn.Tanh())

        elif mode == "PointwiseCurve":
            '''
            # ccm: color correction matrix
            self.ccm = nn.Conv2d(3, 3, kernel_size=1)

            pixelwise_weight = torch.FloatTensor([1, 0, 0, 0, 1, 0, 0, 0, 1]) + torch.randn(1) * 1e-4
            pixelwise_bias = torch.FloatTensor([0, 0, 0])

            self.conv1x1.weight.data.copy_(pixelwise_weight.view(3, 3, 1, 1))
            self.conv1x1.bias.data.copy_(pixelwise_bias)

            # per channel curve
            pass

            # conv2d: num_output = 1
            self.conv1x1 = nn.Conv2d(3, 1, kernel_size=1)
            '''
            
            # Color space change
            self.ccm = nn.Conv2d(self.inc, self.inc, 1)
            idtity = np.identity(self.inc, dtype=np.float32) + np.random.randn(1).astype(np.float32)*1e-4
            self.ccm.weight.data = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(idtity), dim=2), dim=3)
            self.ccm.bias.data.fill_(0.)
            # self.ccm = nn.Parameter(torch.from_numpy(idtity)).cuda()
            # self.ccm_bias = nn.Parameter(torch.zeros([self.inc])).cuda()

            # Per-channel curve
            self.npts = 16
            shifts_ = np.linspace(0,1,self.npts, endpoint=False, dtype=np.float32)
            shifts_ = shifts_[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
            shifts_ = np.tile(shifts_, (1, self.inc, 1, 1, 1))
            self.shifts = nn.Parameter(torch.from_numpy(shifts_))
            # print('gggggg', self.shifts.require_grad)

            slopes_ = torch.zeros([1, self.inc, 1, 1, self.npts], dtype=torch.float32)
            slopes_[:,:,:,:,0] = 1.0
            self.slope = nn.Parameter(slopes_)

            self.conv = nn.Conv2d(self.inc, 1, 1)
            self.conv.weight.data.fill_(1.0/self.inc)
            self.conv.bias.data.fill_(0.)


    def forward(self, x):
        if self.mode == "PointwiseNN":
            guidemap = self.conv2(self.conv1(x))
            guidemap = torch.clamp(guidemap, 0., 1.)
        elif self.mode == 'PointwiseCurve':
            # guidemap = guidemap.view()
            guidemap = self.ccm(x)
            guidemap = torch.unsqueeze(guidemap, dim=4)
            guidemap = torch.sum(self.slope*F.relu(guidemap-self.shifts), dim=4)
            guidemap = self.conv(guidemap)
            
        return guidemap

class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, bilateral_grid, guidemap):
        N, _, H, W = guidemap.shape
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
        # hg = hg.type(torch.cuda.FloatTensor).repeat(N, 1, 1).unsqueeze(3) / (H-1) * 2 - 1
        # wg = wg.type(torch.cuda.FloatTensor).repeat(N, 1, 1).unsqueeze(3) / (W-1) * 2 - 1
        hg = hg.type(guidemap.dtype).repeat(N, 1, 1).unsqueeze(3) / (H-1) * 2 - 1
        wg = wg.type(guidemap.dtype).repeat(N, 1, 1).unsqueeze(3) / (W-1) * 2 - 1
        hg = hg.to(guidemap.device)
        wg = wg.to(guidemap.device)
        guidemap = guidemap.permute(0,2,3,1).contiguous()
        guidemap_guide = torch.cat([guidemap, hg, wg], dim=3).unsqueeze(1)

        coeff = F.grid_sample(bilateral_grid, guidemap_guide)

        return coeff.squeeze(2)
'''
class Transform(nn.Module):
    def __init__(self):
        super(Transform, self).__init__()

    def forward(self, coeff, full_res_input):
        # R = torch.sum(full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True) + coeff[:, 3:4, :, :]
        # G = torch.sum(full_res_input * coeff[:, 4:7, :, :], dim=1, keepdim=True) + coeff[:, 7:8, :, :]
        # B = torch.sum(full_res_input * coeff[:, 8:11, :, :], dim=1, keepdim=True) + coeff[:, 11:12, :, :]
        c = full_res_input.shape[1]

        R = torch.sum(full_res_input * coeff[:, 0:c, :, :], dim=1, keepdim=True) + coeff[:, c:c+1, :, :]
        G = torch.sum(full_res_input * coeff[:, c+1:2*c+1, :, :], dim=1, keepdim=True) + coeff[:, 2*c+1:2*c+2, :, :]
        B = torch.sum(full_res_input * coeff[:, 2*c+2:3*c+2, :, :], dim=1, keepdim=True) + coeff[:, 3*c+2:, :, :]

        return torch.cat([R, G, B], dim=1)
'''

class Transform(nn.Module):
    def __init__(self, outc=3):
        super(Transform, self).__init__()

        self.outc = outc

    def forward(self, coeff, full_res_input):
        b,c,h,w = full_res_input.shape
        full_res_input = torch.cat([full_res_input, torch.ones((b,1,h,w), dtype=full_res_input.dtype).to(full_res_input.device)], dim=1)

        if self.outc == 1:
            assert(full_res_input.shape[1] == coeff.shape[1]), 'the num of channels should be equal'
            return torch.sum(full_res_input * coeff, dim=1, keepdim=True)
        else:
            output = []
            c = full_res_input.shape[1]
            for i in range(self.outc):
                output.append(torch.sum(full_res_input * coeff[:,i*c:(i+1)*c, :,:], dim=1, keepdim=True))
                # output.append(torch.sum(full_res_input * coeff[:,i*(c+1):(i+1)*(c+1)-1, :,:], dim=1, keepdim=True))
            return torch.cat(output, dim=1)

class HDRNet(nn.Module):
    def __init__(self, inc=3, outc=3, guide_mode='PointwiseNN', is_BN=True):
        super(HDRNet, self).__init__()
        self.inc = inc
        self.outc = outc
        self.is_BN = is_BN

        self.downsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        self.activation = nn.ReLU(inplace=True)

        # -----------------------------------------------------------------------
        splat_layers = []
        for i in range(4):
            if i == 0:
                splat_layers.append(conv_block(self.inc, (2**i) * 8, kernel_size=3, padding=1, stride=2, activation=self.activation, is_BN=False))
            else:
                splat_layers.append(conv_block((2**(i-1) * 8), (2**(i)) * 8, kernel_size=3, padding=1, stride=2, activation=self.activation, is_BN=self.is_BN))

        self.splat_conv = nn.Sequential(*splat_layers)

        # -----------------------------------------------------------------------
        global_conv_layers = [
            conv_block(64, 64, stride=2, activation=self.activation, is_BN=self.is_BN),
            conv_block(64, 64, stride=2, activation=self.activation, is_BN=self.is_BN),
        ]
        self.global_conv = nn.Sequential(*global_conv_layers)

        global_fc_layers = [
            fc(1024, 256, activation=self.activation, is_BN=self.is_BN),
            fc(256, 128, activation=self.activation, is_BN=self.is_BN),
            fc(128, 64)
        ]
        self.global_fc = nn.Sequential(*global_fc_layers)

        # -----------------------------------------------------------------------
        local_layers = [
            conv_block(64, 64, activation=self.activation, is_BN=self.is_BN),
            conv_block(64, 64, use_bias=False, activation=None, is_BN=False),
        ]
        self.local_conv = nn.Sequential(*local_layers)

        # -----------------------------------------------------------------------
        # self.linear = nn.Conv2d(64, 96, kernel_size=1)
        self.linear = nn.Conv2d(64, 8*self.outc*(self.inc+1), kernel_size=1)

        self.guide_func = Guide(inc=self.inc, mode=guide_mode, is_BN=self.is_BN)
        self.slice_func = Slice()
        self.transform_func = Transform(self.outc)

    def forward(self, full_res_input):
        low_res_input = self.downsample(full_res_input)
        bs, _, _, _ = low_res_input.size()

        splat_fea = self.splat_conv(low_res_input)

        local_fea = self.local_conv(splat_fea)

        global_fea = self.global_conv(splat_fea)
        global_fea = self.global_fc(global_fea.view(bs, -1))

        fused = self.activation(global_fea.view(-1, 64, 1, 1) + local_fea)
        fused = self.linear(fused)

        # bilateral_grid = fused.view(-1, 12, 8, 16, 16)
        bilateral_grid = fused.view(-1, self.outc*(self.inc+1), 8, 16, 16)
        
        guidemap = self.guide_func(full_res_input)
        coeff = self.slice_func(bilateral_grid, guidemap)
        
        output = self.transform_func(coeff, full_res_input)

        return output

if __name__ == "__main__":
    # from torchsummary import summary
    input = torch.empty((2, 10, 512, 512)).normal_(0, 0.1)
    target = torch.empty((2, 1, 512, 512)).fill_(0.5)
    print (input.shape)
    # net = HDRNet(inc=10, outc=1).cuda()

    net = Guide(mode='PointwiseCurve', inc=10).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    critirion = nn.MSELoss().cuda()
    net.train()
    for i in range(5):
        for name, param in net.named_parameters():
            print ('----------------')
            print (name)
            print (torch.mean(param))
            print (param.shape)
            print ('****************')
            output = net(input.cuda())
        print (output.shape)
        loss = critirion(output, target.cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print (output.shape)

    # summary(net, (3,960,540))
    # print net
    print ('done')
    # slice_func = Slice()
    # bilateral_grid = torch.randn(4, 12, 8, 16, 16)
    # guide = torch.randn(4, 1, 256, 256)
    # slice_func(bilateral_grid, guide)


    pdb.set_trace()