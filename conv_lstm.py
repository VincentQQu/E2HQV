import torch
import torch.nn as nn
from models import MBConv, DoubleConv, UpMB
import torch.nn.functional as F



# in_channels: int, out_features: int, MBC_type = "depthwise", expansion: int = 4
class ConvLSTM(nn.Module):
    """Adapted from: https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py """

    def __init__(self, input_size, hidden_size, kernel_size, MBC_type="conv", expansion=2):
        super(ConvLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        pad = kernel_size // 2

        # cache a tensor filled with zeros to avoid reallocating memory at each inference step if --no-recurrent is enabled
        self.zero_tensors = {}

        # self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=pad)


        if MBC_type == "conv":
            self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=pad)
        
        else:
            self.Gates = MBConv(input_size + hidden_size, 4 * hidden_size, MBC_type, expansion)

    def forward(self, input_, prev_state=None):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:

            # create the zero tensor if it has not been created already
            state_size = tuple([batch_size, self.hidden_size] + list(spatial_size))
            if state_size not in self.zero_tensors:
                # allocate a tensor with size `spatial_size`, filled with zero (if it has not been allocated already)
                self.zero_tensors[state_size] = (
                    torch.zeros(state_size, dtype=input_.dtype).to(input_.device),
                    torch.zeros(state_size, dtype=input_.dtype).to(input_.device)
                )

            prev_state = self.zero_tensors[tuple(state_size)]

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)

        # print("stacked_inputs", input_.shape, prev_hidden.shape)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell


class DownConvLSTM(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, input_size, hidden_size, kernel_size, MBC_type="conv", expansion=2, pool_sz=2):
        super().__init__()

        self.mbd = ConvLSTM(input_size, hidden_size, kernel_size, MBC_type, expansion)

        self.out = DoubleConv(hidden_size, hidden_size)

        self.pool = nn.MaxPool2d(pool_sz)


    def forward(self, x, p_state):
        c_state = self.mbd(x, p_state)
        x = self.out(c_state[0]) + c_state[0]
        x = self.pool(x)

        return x, c_state
    



class UpConvLSTM(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, MBC_type, expansion, n_repeats = 2, bilinear=True, scale_factor=2):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
            # self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=scale_factor, stride=scale_factor)

        
        self.mbd = ConvLSTM(in_channels, out_channels, 3, MBC_type, expansion)

        self.out = DoubleConv(out_channels, out_channels)

        # print(in_channels, out_channels)

        # self.mbd = torch.nn.Sequential()
        
        # for i in range(n_repeats-1):
        #     self.mbd.add_module(f"convlstm_{i}",ConvLSTM(in_channels, in_channels, 3, MBC_type, expansion))
        
        # self.mbd.add_module(f"convlstm_{n_repeats-1}", ConvLSTM(in_channels, out_channels, 3, MBC_type, expansion))


    def forward(self, x1, x2, p_state):
        x1 = self.up(x1)

        # print(x1.shape)
        # print(x2.shape)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        # print(x2.shape)
        x = torch.cat([x2, x1], dim=1)
        # x = x2 + x1

        c_states = self.mbd(x, p_state)

        x = self.out(c_states[0]) + c_states[0]

        return x, c_states



class UNetConvLSTM(nn.Module):
    def __init__(self, input_size=5, output_size=1, n_lyr=3, decode_lstm=1):
        super(UNetConvLSTM, self).__init__()

        # Define the components of the UNet-like model here.
        # This will depend on the specifics of your ConvLSTM.
        self.input_size = input_size
        self.output_size = input_size
        self.n_lyr = n_lyr
        self.decode_lstm = decode_lstm



        kernel_sizes = [5, 3, 3, 3]
        n_chs = [5, 10, 20, 40]

        # 60, 20
        # 30, 10
        # 15, 5


        # n_rep_dn = [2, 2, 4, 4, 6]
        # lyr_ts = ["fused", "fused", "depthwise", "depthwise"]
        lyr_ts = ["conv", "conv", "conv", "conv"]

        # n_rep_up = [6, 4, 4, 2, 2]
        expans = [1, 2, 4, 4, 6]
        pool_szs = [3, 3, 2, 2, 5]


        self.inc = DoubleConv(input_size, n_chs[0])
        # self.down_convs = nn.ModuleList()
        # self.down_pools = nn.ModuleList()

        self.downs = nn.ModuleList()
        # DownConvLSTM
        


        i = 0

        while i < n_lyr:

            lyr = DownConvLSTM(n_chs[i], n_chs[i+1], kernel_sizes[i], lyr_ts[i], expans[i], pool_szs[i])
            self.downs.append(lyr)
            i += 1

        
        self.ups = nn.ModuleList()

        i = 0
        while i < n_lyr:
            rev_i = n_lyr-i

            if decode_lstm:
                in_ch = n_chs[rev_i] + n_chs[rev_i-1]
                out_ch = n_chs[rev_i-1]
                lyr = UpConvLSTM(in_ch, out_ch, lyr_ts[rev_i], expans[rev_i], 1, bilinear=True, scale_factor=pool_szs[rev_i])
            else:
                lyr = UpMB(n_chs[rev_i], n_chs[rev_i-1], lyr_ts[rev_i], expans[rev_i], 1, bilinear=True, scale_factor=pool_szs[rev_i])

            self.ups.append(lyr)

            i += 1

        self.outc = DoubleConv(n_chs[0], output_size)
        



    def forward(self, x, p_states):
        # Define the forward pass for each layer.
        # This will depend on the specifics of your ConvLSTM.

        x0 = self.inc(x)

        xs = [x0]
        
        cur_states = []

        i = 0

        while i < self.n_lyr:
            # print("xx")

            tmp_x, c_state = self.downs[i](xs[-1], p_states[i])

            # print(hidden.shape, cell.shape)

            cur_states.append( c_state )

            xs.append(tmp_x)

            i += 1

        x_m = xs[-1]
        rev_xs = xs[::-1]

        # print("after encoder")


        if self.decode_lstm:
            i = 0
            while i < self.n_lyr:
                up = self.ups[i]
                x_r = rev_xs[i+1]

                tmp_x, c_state = self.ups[i](x_m, x_r, p_states[i+self.n_lyr])

                # print(hidden.shape, cell.shape)


                cur_states.append( c_state )

                x_m = tmp_x

                i += 1

        else:
            for up, x_r in zip(self.ups, rev_xs[1:]):
                # print("x_ie", x_ie.shape)
                # print("xr", xr.shape)
                x_m = up(x_m, x_r)
                # print(x_ie.shape)
        
        output = self.outc(x_m)


        return output, cur_states




if __name__ == "__main__":

    model = ConvLSTM(input_size=5, hidden_size=5, kernel_size=3)
