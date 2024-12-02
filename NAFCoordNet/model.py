import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# LayerNorm2d remains the same
class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=(0, 2, 3)), grad_output.sum(dim=(0, 2, 3)), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

# PositionEncoding class
class PositionEncoding(nn.Module):
    def __init__(self, L):
        super(PositionEncoding, self).__init__()
        self.L = L
        self.channels = 8 * L  # 4 coordinates, each with 2L channels (sin and cos)
        freq_bands = 0.5 * np.pi * torch.arange(1, L + 1).float()
        self.register_buffer('freq_bands_buffer', freq_bands)

    def forward(self, index_list):
        # index_list: [Batch, H, W, 4]
        batch_size, height, width, _ = index_list.shape
        x_global = index_list[:, :, :, 0]
        y_global = index_list[:, :, :, 1]
        x_local = index_list[:, :, :, 2]
        y_local = index_list[:, :, :, 3]

        freq = self.freq_bands_buffer.view(1, 1, 1, self.L)

        sin_x_global = torch.sin(freq * x_global.unsqueeze(-1))
        cos_x_global = torch.cos(freq * x_global.unsqueeze(-1))
        sin_y_global = torch.sin(freq * y_global.unsqueeze(-1))
        cos_y_global = torch.cos(freq * y_global.unsqueeze(-1))

        sin_x_local = torch.sin(freq * x_local.unsqueeze(-1))
        cos_x_local = torch.cos(freq * x_local.unsqueeze(-1))
        sin_y_local = torch.sin(freq * y_local.unsqueeze(-1))
        cos_y_local = torch.cos(freq * y_local.unsqueeze(-1))

        # Concatenate all positional encodings
        pos_enc = torch.cat([
            sin_x_global, cos_x_global,
            sin_y_global, cos_y_global,
            sin_x_local, cos_x_local,
            sin_y_local, cos_y_local
        ], dim=-1)

        pos_enc = pos_enc.view(batch_size, height, width, -1)
        pos_enc = pos_enc.permute(0, 3, 1, 2)  # [Batch, C, H, W]

        return pos_enc

# Mask class used in CoordGate
class Mask(nn.Module):
    def __init__(self, num_in, num_hidden, num_out):
        super(Mask, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(num_in, num_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(num_hidden, num_out),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, height, width, num_in = x.shape
        x = x.reshape(-1, num_in)
        x = self.net(x)
        x = x.reshape(batch_size, height, width, -1)
        return x

# CoordGate class
class CoordGate(nn.Module):
    def __init__(self, num_in, num_hidden, CNN_num_out, CNN_num_in, CNN_num_out_conv, kernel_size, stride=1):
        super(CoordGate, self).__init__()
        self.mask = Mask(num_in, num_hidden, CNN_num_out)
        if CNN_num_in == CNN_num_out_conv:
            if kernel_size == 3:
                self.pre_conv = nn.Conv2d(
                    in_channels=CNN_num_in,
                    out_channels=CNN_num_out_conv,
                    kernel_size=3,
                    padding=1,
                    stride=stride,
                    groups=CNN_num_out_conv,
                    bias=True
                )
            else:
                self.pre_conv = nn.Conv2d(
                    in_channels=CNN_num_in,
                    out_channels=CNN_num_out_conv,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                    groups=CNN_num_out_conv,
                    bias=True
                )
        else:
            if kernel_size == 3:
                self.pre_conv = nn.Conv2d(
                    in_channels=CNN_num_in,
                    out_channels=CNN_num_out_conv,
                    kernel_size=3,
                    padding=1,
                    stride=stride,
                    groups=1,
                    bias=True
                )
            else:
                self.pre_conv = nn.Conv2d(
                    in_channels=CNN_num_in,
                    out_channels=CNN_num_out_conv,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                    groups=1,
                    bias=True
                )

    def forward(self, x, index_list):
        mask = self.mask(index_list)
        mask = mask.permute(0, 3, 1, 2)
        x = self.pre_conv(x)
        if mask.shape[1] != x.shape[1]:
            # Adjust mask channels to match x
            repeat_factor = x.shape[1] // mask.shape[1]
            mask = mask.repeat(1, repeat_factor, 1, 1)
        x = x * mask
        return x

# SimpleGate mechanism
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

# Simplified Channel Attention (SCA)
class SCA(nn.Module):
    def __init__(self, channels):
        super(SCA, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=True
        )

    def forward(self, x):
        y = self.global_pool(x)
        y = self.conv(y)
        return x * y

# NAFBlock with CoordGate
class NAFBlock(nn.Module):
    def __init__(self, c, num_in, num_hidden):
        super(NAFBlock, self).__init__()
        dw_channel = c * 2

        # Replace conv1 with CoordGate
        self.coord_gate1 = CoordGate(num_in, num_hidden, dw_channel, CNN_num_in=c, CNN_num_out_conv=dw_channel, kernel_size=1)
        # Replace conv2 with CoordGate (Depthwise)
        self.coord_gate2 = CoordGate(num_in, num_hidden, dw_channel, CNN_num_in=dw_channel, CNN_num_out_conv=dw_channel, kernel_size=3)
        # Replace conv3 with CoordGate
        self.coord_gate3 = CoordGate(num_in, num_hidden, c, CNN_num_in=dw_channel // 2, CNN_num_out_conv=c, kernel_size=1)

        # Simplified Channel Attention
        self.sca = SCA(dw_channel // 2)

        # SimpleGate
        self.sg = SimpleGate()

        # FFN Part
        ffn_channel = c * 2
        self.coord_gate4 = CoordGate(num_in, num_hidden, ffn_channel, CNN_num_in=c, CNN_num_out_conv=ffn_channel, kernel_size=1)
        self.coord_gate5 = CoordGate(num_in, num_hidden, c, CNN_num_in=ffn_channel // 2, CNN_num_out_conv=c, kernel_size=1)

        # Using LayerNorm2d
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x, index_list):
        inp = x

        x = self.norm1(x)

        x = self.coord_gate1(x, index_list)
        x = self.coord_gate2(x, index_list)
        x = self.sg(x)
        x = self.sca(x)
        x = self.coord_gate3(x, index_list)

        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.coord_gate4(x, index_list)
        x = self.sg(x)
        x = self.coord_gate5(x, index_list)

        return y + x * self.gamma

# Sequential module supporting multiple inputs
class SequentialMultiInput(nn.Sequential):
    def forward(self, x, index_list):
        for module in self._modules.values():
            x = module(x, index_list)
        return x

# NAFNet with CoordGate and PositionEncoding
class NAFNet(nn.Module):
    def __init__(self, img_channel=1, width=32, middle_blk_num=1,
                 enc_blk_nums=[1, 1, 1, 1], dec_blk_nums=[1, 1, 1, 1],
                 num_in=8 * 4, num_hidden=64, num_views=9, L=4, output_channels=1):
        super(NAFNet, self).__init__()

        self.num_views = num_views
        self.width = width
        self.L = L
        self.position_encoding = PositionEncoding(L=L)

        # Intro layers for each view
        self.intro = nn.ModuleList([
            CoordGate(num_in, num_hidden, width, CNN_num_in=img_channel, CNN_num_out_conv=width, kernel_size=3)
            for _ in range(num_views)
        ])

        # Shared encoders and decoders
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = SequentialMultiInput()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            encoder_layers = []
            for _ in range(num):
                encoder_layers.append(NAFBlock(chan, num_in, num_hidden))
            self.encoders.append(SequentialMultiInput(*encoder_layers))
            self.downs.append(nn.Conv2d(chan, chan * 2, kernel_size=2, stride=2))
            chan *= 2

        middle_layers = []
        for _ in range(middle_blk_num):
            middle_layers.append(NAFBlock(chan, num_in, num_hidden))
        self.middle_blks = SequentialMultiInput(*middle_layers)

        for num in dec_blk_nums:
            self.ups.append(nn.ConvTranspose2d(chan, chan // 2, kernel_size=2, stride=2))
            chan = chan // 2
            decoder_layers = []
            for _ in range(num):
                decoder_layers.append(NAFBlock(chan, num_in, num_hidden))
            self.decoders.append(SequentialMultiInput(*decoder_layers))

        # Ending layer
        self.ending = CoordGate(num_in, num_hidden, output_channels, CNN_num_in=chan, CNN_num_out_conv=output_channels, kernel_size=3)

    def forward(self, x, index_list):
        # x: [Batch, num_views, 1, H, W]
        # index_list: [Batch, num_views, H, W, 2]

        batch_size, num_views, _, H, W = x.size()
        feats = []
        for i in range(num_views):
            xi = x[:, i, :, :, :]  # [Batch, 1, H, W]
            index_i = index_list[:, i, :, :, :]  # [Batch, H, W, 2]
            # Generate positional encoding
            pos_enc = self.position_encoding(index_i)  # [Batch, 4L, H, W]
            pos_enc = pos_enc.permute(0, 2, 3, 1)  # [Batch, H, W, 4L]
            
            xi = self.intro[i](xi, pos_enc)
            feats.append(xi)  # [Batch, C, H, W]

        # Fuse features from all views
        x = torch.stack(feats, dim=1)  # [Batch, num_views, C, H, W]
        x, _ = x.max(dim=1)  # [Batch, C, H, W]

        # Build index pyramid
        index = index_list.mean(dim=1)  # [Batch, H, W, 2]
        pos_enc = self.position_encoding(index)
        pos_enc = pos_enc.permute(0, 2, 3, 1)  # [Batch, H, W, 4L]

        pos_enc_list = [pos_enc]
        enc_features = []

        # Encoder
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x, pos_enc)
            enc_features.append(x)
            x = down(x)
            # Downsample positional encoding
            index = F.interpolate(index.permute(0, 3, 1, 2), scale_factor=0.5, mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            pos_enc = self.position_encoding(index)
            pos_enc = pos_enc.permute(0, 2, 3, 1)
            pos_enc_list.append(pos_enc)

        # Middle blocks
        x = self.middle_blks(x, pos_enc)

        # Decoder
        for decoder, up, skip, pos_enc in zip(self.decoders, self.ups, reversed(enc_features), reversed(pos_enc_list[:-1])):
            x = up(x)
            x = x + skip
            x = decoder(x, pos_enc)

        # Ending
        x = self.ending(x, pos_enc_list[0])

        return x

# FPNet class
class FPNet(nn.Module):
    def __init__(self, L=4):
        super(FPNet, self).__init__()
        # Demix network, outputs [Batch, 9, H, W]
        self.demix_net = NAFNet(
            img_channel=1,   # Measurement is single-channel
            width=32,
            middle_blk_num=1,
            enc_blk_nums=[1, 1, 1, 1],
            dec_blk_nums=[1, 1, 1, 1],
            num_in=8 * L,
            num_hidden=64,
            num_views=9,    # Processing the measurement as a single input
            L=L,
            output_channels=9  # Outputting 9 channels for demixed images
        )
        # Reconstruction network, outputs [Batch, 1, H, W]
        self.recon_net = NAFNet(
            img_channel=1,   # Input is the 9-channel demixed output
            width=32,
            middle_blk_num=1,
            enc_blk_nums=[1, 1, 1],
            dec_blk_nums=[1, 1, 1],
            num_in=8 * L,
            num_hidden=64,
            num_views=9,    
            L=L,
            output_channels=1  
        )
        self.activation = nn.Sigmoid()

    def forward(self, x, index_list):
        # x: [Batch, num_views, H, W]
        # index_list: [Batch, num_views, H, W, 2]

        x_input = x.unsqueeze(2)  # [Batch, num_views, 1, H, W]

        demix_output = self.demix_net(x_input, index_list)  # [Batch, 1, H, W]
        demix_output = self.activation(demix_output)
        demix_output1 = demix_output.unsqueeze(2)
        recon_output = self.recon_net(demix_output1, index_list)  # [Batch, 1, H, W]
        recon_output = self.activation(recon_output)

        return demix_output, recon_output
