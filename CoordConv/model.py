import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

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
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=[0, 2, 3]), grad_output.sum(dim=[0, 2, 3]), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

# SimpleGate mechanism
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

# Simplified Channel Attention (SCA)
class SCA(nn.Module):
    def __init__(self, channels):
        super(SCA, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        y = self.global_pool(x)
        y = self.conv(y)
        return x * y

# NAFBlock with positional encoding
class NAFBlock(nn.Module):
    def __init__(self, c):
        super(NAFBlock, self).__init__()
        dw_channel = c * 2

        # Replace CoordGate with Conv2d layers
        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=3, padding=1, stride=1)

        # Simplified Channel Attention
        self.sca = SCA(dw_channel // 2)

        # SimpleGate
        self.sg = SimpleGate()

        # FFN Part
        ffn_channel = c * 2
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=3, padding=1, stride=1)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=3, padding=1, stride=1)

        # Using LayerNorm2d
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x, x_pe=None):
        # Add positional encoding if provided
        if x_pe is not None:
            x = x + x_pe

        inp = x

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.sca(x)
        x = self.conv3(x)

        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        return y + x * self.gamma

# NAFNet with positional encoding
class NAFNet(nn.Module):
    def __init__(self, img_channel=1, width=32, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1]):
        super(NAFNet, self).__init__()

        self.intro = nn.Conv2d(img_channel, width, kernel_size=3, padding=1, stride=1)
        self.ending = nn.Conv2d(width, 1, kernel_size=3, padding=1, stride=1)  # Output channel is 1

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            encoder_layers = []
            for _ in range(num):
                encoder_layers.append(NAFBlock(chan))
            self.encoders.append(nn.Sequential(*encoder_layers))
            # Downsampling layer
            self.downs.append(
                nn.Conv2d(chan, chan*2, kernel_size=3, padding=1, stride=2)
            )
            chan *= 2

        for _ in range(middle_blk_num):
            self.middle_blks.append(NAFBlock(chan))

        for num in dec_blk_nums:
            self.ups.append(
                nn.ConvTranspose2d(chan, chan // 2, kernel_size=2, stride=2)
            )
            chan = chan // 2
            decoder_layers = []
            for _ in range(num):
                decoder_layers.append(NAFBlock(chan))
            self.decoders.append(nn.Sequential(*decoder_layers))

    def forward(self, x, x_pe=None):
        x = self.intro(x)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            for block in encoder:
                x = block(x, x_pe)  # Pass positional encoding
            encs.append(x)
            x = down(x)
            if x_pe is not None:
                x_pe = F.avg_pool2d(x_pe, kernel_size=2, stride=2)  # Downsample positional encoding

        for blk in self.middle_blks:
            x = blk(x, x_pe)  # Pass positional encoding

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            if x_pe is not None:
                x_pe = F.interpolate(x_pe, scale_factor=2, mode='bilinear', align_corners=False)  # Upsample positional encoding
            for block in decoder:
                x = block(x, x_pe)  # Pass positional encoding

        x = self.ending(x)

        return x

def indexGenerate(x_start, y_start, p, size):
    xs = torch.linspace(x_start, x_start + p - 1, steps=p)
    ys = torch.linspace(y_start, y_start + p - 1, steps=p)
    y, x = torch.meshgrid(xs, ys, indexing='xy')
    x, y = x.unsqueeze(0)/size, y.unsqueeze(0)/size
    final = torch.cat((x, y), dim=0)
    final = torch.permute(final, (1, 2, 0))
    return final    

# Positional Encoding function
def positional_encoding(index, c):
    # index: [batch_size, height, width, 2]
    batch_size, height, width, _ = index.shape
    index = index.permute(0, 3, 1, 2)  # [batch_size, 2, height, width]

    # Create a linear projection to map 2D positions to c channels
    # Define a learnable linear layer
    pe_linear = nn.Conv2d(2, c, kernel_size=1, bias=False).to(index.device)
    # Initialize the weights to cover different frequencies
    nn.init.xavier_uniform_(pe_linear.weight)

    x_pe = pe_linear(index)
    return x_pe  # [batch_size, c, height, width]

# Main FPNet architecture with NAFNet
class FPNet(nn.Module):
    def __init__(self):
        super(FPNet, self).__init__()
        self.model = NAFNet(
            img_channel=1,
            width=32,
            middle_blk_num=1,
            enc_blk_nums=[1, 1, 1, 1],
            dec_blk_nums=[1, 1, 1, 1]
        )
        self.activation = nn.Sigmoid()

    def forward(self, x, index):
        # Generate positional encoding
        c = x.shape[1]  # Get the channel dimension
        x_pe = positional_encoding(index, c)
        x_pe = x_pe.to(x.device)  # Ensure x_pe is on the same device

        # Add positional encoding before 'inp = x'
        output = self.model(x)

        output = self.activation(output)
        return output  # Output shape: [batch_size, 1, height, width]

def indexDown(index_list):
    index_list = index_list.permute(0, 3, 1, 2)
    output = F.avg_pool2d(index_list, kernel_size=2, stride=2)
    output = output.permute(0, 2, 3, 1)
    return output  # Output shape: downsampled [batch_size, height//2, width//2, channels]
