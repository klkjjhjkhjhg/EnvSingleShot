import torch
import torch.nn as nn
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
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
    
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., padding_mode='zeros', norm_input=False):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True, padding_mode=padding_mode)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate, inplace=True) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate, inplace=True) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.norm_input = norm_input

    def forward(self, inp):
        if self.norm_input:
            inp = self.norm1(inp)
            x = inp
        else:
            x = inp
            x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class NAFNet(nn.Module):

    def __init__(self, in_channel=3, out_channel=10, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], res=False, tanh=True, **kwargs):
        super().__init__()

        self.res = res
        self.tanh = nn.Tanh() if tanh else tanh
        self.usePattern = kwargs.get('usePattern', False)
        self.pSameSize = kwargs.get('pSameSize', False)
        
        self.clamp = kwargs.get('clamp', True)
        self.upScale = kwargs.get('upScale', None)
        self.interpolate = kwargs.get('interpolate', False)
        self.pixshuffleUpsample = kwargs.get('pixshuffleUpsample', True)
        self.padding_mode=kwargs.get('padding_mode', 'zeros')
        self.norm_input=kwargs.get('norm_input', False)
        self.drop_out_rate=kwargs.get('drop_out_rate', 0.0)
        self.outFeat = kwargs.get('outFeat', False)
        self.lrelu = nn.LeakyReLU(0.2, True) if self.outFeat else self.outFeat
        if self.upScale is not None:
            self.ending = nn.Sequential(
                nn.Conv2d(in_channels=width, out_channels=out_channel * self.upScale**2, kernel_size=3, padding=1, stride=1, groups=1, bias=True, padding_mode=self.padding_mode),
                nn.PixelShuffle(self.upScale)
            )
        else:
            self.ending = nn.Conv2d(in_channels=width, out_channels=out_channel, kernel_size=3, padding=1, stride=1, groups=1,
                                bias=True, padding_mode=self.padding_mode)
        if self.usePattern:
            if not self.pSameSize:
                self.introPattern = self.build_intropattern(width)
                in_channel = in_channel + width
            else:
                in_channel = in_channel * 2
        self.intro = nn.Conv2d(in_channels=in_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                            bias=True, padding_mode=self.padding_mode)
        self.encoders, self.downs, width = self.build_encoders(width, enc_blk_nums)
        self.middle_blks, width = self.build_middle_blks(width, middle_blk_num)
        self.ups, self.decoders, self.padder_size, width = self.build_decoders(width, dec_blk_nums)

    def build_intropattern(self, width):
        introPattern = []
        introPattern.append(nn.Conv2d(in_channels=3, out_channels=width, kernel_size=2, stride=2))
        introPattern.append(NAFBlock(width, padding_mode=self.padding_mode, norm_input=self.norm_input, drop_out_rate=self.drop_out_rate))
        introPattern.append(nn.Conv2d(in_channels=width, out_channels=width, kernel_size=2, stride=2))
        introPattern.append(NAFBlock(width, padding_mode=self.padding_mode, norm_input=self.norm_input, drop_out_rate=self.drop_out_rate))
        return nn.Sequential(*introPattern)

    def build_encoders(self, width, enc_blk_nums):
        encoders = nn.ModuleList()
        downs = nn.ModuleList()
        for num in enc_blk_nums:
            encoders.append(
                nn.Sequential(
                    *[NAFBlock(width, padding_mode=self.padding_mode, norm_input=self.norm_input, drop_out_rate=self.drop_out_rate) for _ in range(num)]
                )
            )
            downs.append(
                nn.Conv2d(width, 2*width, 2, 2)
            )
            width = width * 2
        return encoders, downs, width

    def build_middle_blks(self, width, middle_blk_num):
        middle_blks = \
            nn.Sequential(
                *[NAFBlock(width, padding_mode=self.padding_mode, norm_input=self.norm_input, drop_out_rate=self.drop_out_rate) for _ in range(middle_blk_num)]
            )
        return middle_blks, width

    def build_decoders(self, width, dec_blk_nums):
        ups = nn.ModuleList()
        decoders = nn.ModuleList()
        for num in dec_blk_nums:
            ups.append(
                nn.Sequential(
                    nn.Conv2d(width, width * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            ) if self.pixshuffleUpsample else\
            ups.append(
                nn.Sequential(
                    nn.Conv2d(width, width // 2, 1, bias=False),
                    nn.Upsample(scale_factor=2, mode='bilinear')
                )
            )
            width = width // 2
            decoders.append(
                nn.Sequential(
                    *[NAFBlock(width, padding_mode=self.padding_mode, norm_input=self.norm_input, drop_out_rate=self.drop_out_rate) for _ in range(num)]
                )
            )

        padder_size = 2 ** len(self.encoders)
        return ups, decoders, padder_size, width

    def forward(self, inp, pattern=None, saveFeat=False):
        B, C, H, W = inp.shape
        midfeat=[]
        inp = self.check_image_size(inp)

        if self.usePattern:
            featPattern = self.introPattern(pattern) if not self.pSameSize else pattern
            featPattern = featPattern.broadcast_to(B,featPattern.shape[1], H, W) if featPattern.shape[0] == 1 else featPattern
            inp = torch.cat([inp, featPattern], dim = 1)
        x = self.intro(inp)
        midfeat.append(x)
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        midfeat.extend(encs)
        x = self.middle_blks(x)
        midfeat.append(x) #编码的结果
        for i, (decoder, up) in enumerate(zip(self.decoders, self.ups)):
            enc_skip = encs[::-1][i] if i < len(encs) else 0
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
            midfeat.append(x)

        x = self.ending(x)
        midfeat.append(x)
        
        if self.res:
            x = x + inp
        if self.tanh:
            x = self.tanh(x)
        elif self.clamp:
            x = x.clamp(-1+1e-6, 1-1e-6)
        elif self.outFeat:
            x = self.lrelu(x)
        if self.interpolate:
            x = F.interpolate(x, scale_factor=4, mode="bicubic", align_corners=False)
        if saveFeat:
            return x[:, :, :, :], midfeat
        else:
            return x

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
