import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import math




def default_act():
    # return nn.ReLU()
    # return nn.LeakyReLU()
    # return nn.Sigmoid()
    # return nn.Tanh()
    return nn.SiLU()
    # return nn.ReLU6()



def default_bn(out_channels):
    return nn.GroupNorm(1, out_channels)
    # return nn.BatchNorm2d(out_channels)



def default_pool(pool_size):
    return nn.MaxPool2d(pool_size)
    # return nn.AvgPool2d(pool_size)


def get_padding_mode():
    # 'zeros', 'reflect', 'replicate' or 'circular'
    return 'zeros' 
# padding="same"

class DoubleConv(nn.Module):
    """(convolution => [LN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding="same", padding_mode=get_padding_mode(), bias=False),
            default_bn(mid_channels),
            default_act(),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding="same", padding_mode=get_padding_mode(), bias=False),
            default_bn(out_channels),
            default_act()
        )

    def forward(self, x):
        return self.double_conv(x)


# https://amaarora.github.io/2020/08/09/groupnorm.html
class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_features: int, kernel_size = 3, groups=1, act = None):
        # if act == None: nn.SiLU(inplace=True)
        super(ConvNormAct, self).__init__()
        self.convna = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_features,
                kernel_size=kernel_size,
                padding=kernel_size // 2, padding_mode=get_padding_mode(),
                groups=groups
            ),
            default_bn(out_features),
        )
        if act != "no_act": self.convna.add_module("act",default_act())
    
    def forward(self, x):
        return self.convna(x)



# https://amaarora.github.io/2020/07/24/SeNet.html#squeeze-and-excitation-block-in-pytorch
class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=12):
        super(SE_Block, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)




# https://towardsdatascience.com/residual-bottleneck-inverted-residual-linear-bottleneck-mbconv-explained-89d7b7e7c6bc
# https://pytorch.org/vision/main/_modules/torchvision/models/efficientnet.html
class MBConv(nn.Module):
    def __init__(self, in_channels: int, out_features: int, MBC_type = "depthwise", expansion: int = 4):

        expanded_features = in_channels * expansion
        super().__init__()

        if MBC_type == "depthwise":
            self.mbconv = nn.Sequential(
                # narrow -> wide
                ConvNormAct(in_channels, expanded_features, kernel_size=1),
                # wide -> wide
                ConvNormAct(expanded_features, expanded_features, kernel_size=3,  groups=expanded_features), # 
                # here you can apply SE
                SE_Block(expanded_features),
                # wide -> narrow
                ConvNormAct(expanded_features, out_features, kernel_size=1, act="no_act"),
            )
        elif MBC_type == "fused":
            self.mbconv = nn.Sequential(
                # narrow -> wide
                ConvNormAct(in_channels, expanded_features, kernel_size=3),
                # here you can apply SE
                SE_Block(expanded_features),
                # wide -> narrow
                ConvNormAct(expanded_features, out_features, kernel_size=1, act="no_act"),
            )
    
    def forward(self, x):
        # print("++++++++++++++")
        x1 = x
        x2 = self.mbconv(x)
        return x1 + x2 if x1.shape == x2.shape else x2

        


class DownMB(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, MBC_type, expansion, n_repeats = 2, pool_size=2):
        super().__init__()
        self.mbd = torch.nn.Sequential()
        self.mbd.add_module("maxpool", default_pool(pool_size))
        self.mbd.add_module("mbconv_0", MBConv(in_channels, out_channels, MBC_type, expansion))
        for i in range(n_repeats-1):
            self.mbd.add_module(f"mbconv_{i+1}",MBConv(out_channels, out_channels, MBC_type, expansion))


    def forward(self, x):
        return self.mbd(x)


class UpMB(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, MBC_type, expansion, n_repeats = 2, bilinear=True, scale_factor=2):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
            # self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=scale_factor, stride=scale_factor)
            # self.conv = DoubleConv(in_channels, out_channels)
        
        self.mbd = torch.nn.Sequential()
        
        for i in range(n_repeats-1):
            self.mbd.add_module(f"mbconv_{i}",MBConv(in_channels, in_channels, MBC_type, expansion))
        
        self.mbd.add_module(f"mbconv_{n_repeats-1}", MBConv(in_channels, out_channels, MBC_type, expansion))

        # self.mbd.add_module("mbconv_0", MBConv(in_channels, out_channels, MBC_type, expansion))
        # for i in range(n_repeats-1):
        #     self.mbd.add_module(f"mbconv_{i+1}",MBConv(out_channels, out_channels, MBC_type, expansion))

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # print(x1.shape)
        # print(x2.shape)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
  
        x = torch.cat([x2, x1], dim=1)
        # x = x2 + x1
        return self.mbd(x)






class OutIE(nn.Module):
    """(convolution => [LN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = in_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding="same", padding_mode=get_padding_mode(), bias=False),
            default_bn(mid_channels),
            default_act(),

            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding="same", padding_mode=get_padding_mode(), bias=False),
            default_bn(mid_channels),
            default_act(),

            nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding="same", padding_mode=get_padding_mode(), bias=False),
            # default_bn(out_channels),
            # default_act()
        )

    def forward(self, x):
        return self.double_conv(x)# + x_ie







#############################
################# UNet
##############################

class EffUNet(nn.Module):
    def __init__(self, n_channels, out_depth, bilinear=False, n_lyr=4, ch1=24, device="cuda:0"):
        super(EffUNet, self).__init__()
        self.n_channels = n_channels
        self.out_depth = out_depth
        self.bilinear = bilinear

        # ch1 = 24
        
        n_chs = [ch1* (2 ** power) for power in range(n_lyr+1)]
        n_rep_dn = [2, 2, 4, 4, 6]
        lyr_ts = ["fused", "fused", "depthwise", "depthwise", "depthwise"]
        n_rep_up = [6, 4, 4, 2, 2]
        expans = [1, 2, 4, 4, 6]
        pool_szs = [3, 3, 2, 2, 5]
        factor = 2 if bilinear else 1

        self.mparams = {"n_lyr": n_lyr, "bilinear": bilinear, "n_chs": n_chs, "n_rep_dn": n_rep_dn, "lyr_ts": lyr_ts, "n_rep_up": n_rep_up, "expans": expans, "pool_szs": pool_szs, "factor": factor}

        double_factor = 1

        self.inc = DoubleConv(n_channels, n_chs[0]*double_factor)
        # self.inc2 = DoubleConv(n_chs[0]*double_factor, n_chs[0])

        self.downs = nn.ModuleList()

        for i in range(n_lyr):
            out_chnl = n_chs[i+1] // factor if i == n_lyr-1 else n_chs[i+1]
            lyr = DownMB(n_chs[i], out_chnl, lyr_ts[i], expansion=expans[i], n_repeats=n_rep_dn[i], pool_size=pool_szs[i])
            self.downs.append(lyr)
        
        self.ups = self.ups_builder()
        # self.out_clean_ie = DoubleConv(n_chs[0], out_depth, n_chs[0]//2)
        self.out_clean_ie = OutIE(n_chs[0], out_depth, None)


    def ups_builder(self):
        ups = nn.ModuleList()
        for i in range(self.mparams["n_lyr"]):
            rev_i = self.mparams["n_lyr"]-i-1
            out_chnl = self.mparams["n_chs"][rev_i] if i == self.mparams["n_lyr"]-1 else self.mparams["n_chs"][rev_i] // self.mparams["factor"]

            lyr = UpMB(self.mparams["n_chs"][rev_i+1], out_chnl, self.mparams["lyr_ts"][rev_i], expansion=self.mparams["expans"][rev_i], n_repeats=self.mparams["n_rep_up"][i], bilinear=self.mparams["bilinear"], scale_factor=self.mparams["pool_szs"][rev_i])
            ups.append(lyr)
        
        return ups


    def forward(self, x):
        output_list = []
        # print(x.shape)
        

        for ic in range(self.n_channels):
            output_list.append(x[:,ic,:,:].unsqueeze(dim=1))
        # f01, f21 = x[:,0,:,:].unsqueeze(dim=1), x[:,1,:,:].unsqueeze(dim=1)
        # f0 = torch.unsqueeze(f0, dim=1)

        x0 = x
        
        x1 = self.inc(x0)
        # x1 = self.inc2(x1)
        xs = [x1]

        for dn in self.downs:
            tmp_x = dn(xs[-1])
            # print("dn")
            # print(tmp_x.shape)
            xs.append(tmp_x)

        # print()

        x_ie = xs[-1]
        rev_xs = xs[::-1]
        for up, xr in zip(self.ups, rev_xs[1:]):
            # print("x_ie", x_ie.shape)
            # print("xr", xr.shape)
            x_ie = up(x_ie, xr)
            # print(x_ie.shape)
        clean_ie = self.out_clean_ie(x_ie) # , x_ie_0

        # print(clean_ie.shape)

        # preds = torch.cat((f01, f21, clean_ie), 1)

        output_list.append(clean_ie)
        preds = torch.cat(output_list, 1)


        return preds





class TimeEmbedMini(nn.Module):

    def __init__(self, h, w, seq_len, embed_rf0=1):
        super().__init__()

        self.h, self.w = h, w
        self.seq_len = seq_len
        self.embeds = nn.Parameter(torch.rand(1, seq_len, h, w))
        # print(self.embeds.shape)

        self.embed_rf0 = embed_rf0

        # self.embeds_for_no_gt = nn.Parameter(torch.rand(2, 1, h, w))


        if embed_rf0:
            self.cnn = EffUNet(n_channels=3, out_depth=1, bilinear=True, n_lyr=4, ch1=3)


    def forward(self, x, last_gt, distance_from_gt, after_coarser, device="cuda"):


        embed_channel = self.embeds[after_coarser][distance_from_gt]

        
        bsz = x.shape[0]
        embed_channel = embed_channel.repeat(bsz, 1, 1, 1)

        spatial_shape = x.shape[-2:]

        embed_channel = F.interpolate(embed_channel, size=spatial_shape, scale_factor=None, mode='bilinear')

        if self.embed_rf0:
            cnn_input = torch.cat([x, last_gt, embed_channel], dim=1)
            embed_to_add = self.cnn(cnn_input)

            embed_to_add = embed_to_add[:,-1,:,:].unsqueeze(dim=1)

            post_embed_x = x + embed_to_add

            post_embed_x = torch.clamp(post_embed_x, 0, 1)
        
        else:
            embed_to_add = 0
            post_embed_x = x

        # print(post_embed_x.shape)

        return embed_channel, post_embed_x, embed_to_add






def count_parameters(module: nn.Module, trainable: bool = True) -> int:

    if trainable:
        num_parameters = sum(p.numel() for p in module.parameters()
                             if p.requires_grad)
    else:
        num_parameters = sum(p.numel() for p in module.parameters())

    return num_parameters



if __name__ == "__main__":

    pass
  
