import torch
import librosa
import functools
import numpy as np
from    torch import nn
from    torch.nn import functional as F 

class zcxNet(nn.Module):
    def __init__(self):
        super(zcxNet, self).__init__()

        self.kernel_size=[3, 1]
        self.stride_size=[2, 1]
        self.num_conv_filters=[29, 32, 32, 64, 64]
        self.num_linear_layer=[32,71]
        # 64, 29, 16, 1
        self.conv1 = nn.Sequential(nn.Conv2d(self.num_conv_filters[0], self.num_conv_filters[1], kernel_size=self.kernel_size, stride=self.stride_size,padding=(1,0)),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(self.num_conv_filters[1]))
        self.conv2 = nn.Sequential(nn.Conv2d(self.num_conv_filters[1], self.num_conv_filters[2], kernel_size=self.kernel_size, stride=self.stride_size,padding=(1,0)),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(self.num_conv_filters[2]))
        self.conv3 = nn.Sequential(nn.Conv2d(self.num_conv_filters[2], self.num_conv_filters[3], kernel_size=self.kernel_size, stride=self.stride_size,padding=(1,0)),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(self.num_conv_filters[3]))
        self.conv4 = nn.Sequential(nn.Conv2d(self.num_conv_filters[3], self.num_conv_filters[4], kernel_size=self.kernel_size, stride=self.stride_size,padding=(1,0)),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(self.num_conv_filters[4]))


        self.fc1 = nn.Linear(self.num_conv_filters[4], self.num_linear_layer[0])
        self.bn1 = nn.BatchNorm1d(self.num_linear_layer[0])
        self.fc2 = nn.Linear(self.num_linear_layer[0]+32, self.num_linear_layer[1])


    def forward(self, x, y):
        x = x.unsqueeze(1).float()
        x = x.permute(0,3,2,1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.shape[0], x.shape[1])
        x = self.bn1(torch.tanh(self.fc1(x)))
        x = torch.cat([x, y],dim=1) 
        x = self.fc2(x)
        return x

class TfaceGAN(nn.Module):
    def __init__(self, input_nc=256, output_nc=256, num_downs=6, ngf=64, norm_layer=nn.BatchNorm1d, use_dropout=False):
        super(TfaceGAN, self).__init__()
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  
        for i in range(1):          
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf*4, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf*4, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.conv11 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding = [2,0])
        self.bn11 = nn.BatchNorm2d(32)

        self.conv21 = nn.Conv2d(32, 128, kernel_size=4, stride=2, padding = [2,0])
        self.bn21 = nn.BatchNorm2d(128)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding = [2,0])
        self.bn31 = nn.BatchNorm2d(256)

        self.conv41 = nn.Conv2d(256, 256, kernel_size=[3,1], stride=1, padding = [1,0])
        self.bn41 = nn.BatchNorm2d(256)

        self.resize = nn.UpsamplingBilinear2d([128,1])
        self.conv42 = nn.Conv1d(327, 256, kernel_size=3, stride=1,padding=1)
        self.bn42 = nn.BatchNorm1d(256)

        self.conv52 = nn.Conv1d(256, 256, kernel_size=3, stride=1,padding=1)
        self.bn52 = nn.BatchNorm1d(256)

        self.conv62 = nn.Conv1d(256, 32, kernel_size=3, stride=1,padding=1)


        self.G1 = zcxNet()

    def forward(self, x,y):
        x1 = x[:,:,8,:].unsqueeze(1)
        out = F.leaky_relu(self.bn11(self.conv11(x1)))
        # 64,32,64,13
        out = F.leaky_relu(self.bn21(self.conv21(out)))
        # 64,128,32,5
        out = F.leaky_relu(self.bn31(self.conv31(out)))
        # 64, 256,32,2
        out = F.leaky_relu(self.bn41(self.conv41(out)))
        # 64, 256, 32, 2
        out = out.view(out.size(0),out.size(1),-1,1)
        # 64, 256, 64, 1
        out = self.resize(out)
        # 64, 256, 128,1
        out = out.squeeze(3)

        out = self.model(out)
        y = y.permute(0,2,1).repeat(1,1,128)   
        out = torch.cat([out, y],dim=1) 
        out = F.leaky_relu(self.bn42(self.conv42(out)))
        out = F.leaky_relu(self.bn52(self.conv52(out)))
        out = self.conv62(out).permute(0,2,1)


        fc_out   = []
        for step_t in range(x.size(1)):
            aa = out[:,step_t,:]
            bb = self.G1(x[: ,step_t , :, :],aa)
            fc_out.append(bb)
        return torch.stack(fc_out, dim = 1)

class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm1d
        else:
            use_bias = norm_layer == nn.InstanceNorm1d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv1d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose1d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose1d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose1d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        #print(x.shape)
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

class ResBlk(nn.Module):
    # resnet block
    def __init__(self, ch_in, ch_out, stride =1):
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        out = F.relu(out)

        return out

class ResNet18(nn.Module):

    def __init__(self, num_class):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(16)
        )
        # 4 blocks
        self.blk1 = ResBlk(16, 32, stride=3)
        self.blk2 = ResBlk(32, 64, stride=3)
        self.blk3 = ResBlk(64, 128, stride=2)
        self.blk4 = ResBlk(128, 256, stride=2)

        self.outlayer = nn.Linear(256*3*3, num_class)

    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        x = x.view(x.size(0), -1)
        x = self.outlayer(x)

        return x


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc=1, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 3
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        # (64,1,256,150)
        input = input.unsqueeze(1)
        return self.model(input)
