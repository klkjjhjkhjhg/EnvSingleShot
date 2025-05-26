import torch
import torch.nn as nn
from archs.naf_arch import NAFBlock
from utils.utils import torch_norm

class LCNet(nn.Module):
    def __init__(self, in_channel, out_channel, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], use_attn_fuse=False, grey_lc=False, **kwargs):
        super(LCNet, self).__init__()
        self.use_attn_fuse = use_attn_fuse
        self.padding_mode = kwargs.get('padding_mode', 'zeros')
        self.norm_input = kwargs.get('norm_input', False)
        self.drop_out_rate = kwargs.get('drop_out_rate', 0.0)
        self.pixshuffleUpsample = kwargs.get('pixshuffleUpsample', True)
        self.grey_lc = grey_lc
        self.only_nrmlc = kwargs.get('only_nrmlc', False)
        self.predict_normal = kwargs.get('predict_normal', True)
        
        self.tanh = nn.Tanh()
        self.intro = nn.Conv2d(in_channels=in_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                    bias=True, padding_mode=self.padding_mode)
        self.mat_ending = nn.Conv2d(in_channels=width, out_channels=out_channel, kernel_size=3, padding=1, stride=1, groups=1,
                            bias=True, padding_mode=self.padding_mode)
        if self.predict_normal:
            self.nrm_ending = nn.Conv2d(in_channels=width, out_channels=2, kernel_size=3, padding=1, stride=1, groups=1,
                                bias=True, padding_mode=self.padding_mode)
        
        self.encoders, self.downs, img_width = self.build_encoders(width, enc_blk_nums)
        self.middle_blks, img_width = self.build_middle_blks(img_width, middle_blk_num)
        
        self.fuse_blks, _ = self.build_middle_blks(img_width * 2, 2)
        self.mat_intro = nn.Conv2d(in_channels=img_width * 2, out_channels=img_width, kernel_size=3, padding=1, stride=1, groups=1,
                    bias=True, padding_mode=self.padding_mode)
        self.mat_ups, self.mat_decoders, mat_width = self.build_decoders(img_width, dec_blk_nums)
        if self.predict_normal:
            self.nrm_ups, self.nrm_decoders, nrm_width = self.build_decoders(img_width, dec_blk_nums)

        inc = 3 * 6 if not self.grey_lc else 6
        inc += 3 * 6 if not self.only_nrmlc and self.predict_normal else 0
        self.lc_intro = nn.Conv2d(in_channels=inc, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                    bias=True, padding_mode=self.padding_mode)
        self.lc_encoders, self.lc_downs, lc_width = self.build_encoders(width, enc_blk_nums)

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

        return ups, decoders, width

    def forward(self, inp, env, renderer, v, plane_lcs=None):
        B, C, H, W = inp.shape
        x = self.intro(inp)
        encs = []
        #* enc imgs
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        
        #* mid blks
        mid_x = self.middle_blks(x)
        
        #* dec nrm
        x = mid_x
        if self.predict_normal:
            for i, (decoder, up) in enumerate(zip(self.nrm_decoders, self.nrm_ups)):
                enc_skip = encs[::-1][i] if i < len(encs) else 0
                x = up(x)
                x = x + enc_skip
                x = decoder(x)
            nrm = self.tanh(self.nrm_ending(x)) # (B, 2, H, W)
            n = self.cal_nrm(nrm)

            #* render lcs with base mats
            base_mat = [(0.0, 0.2), (0.2, 0.2), (0.6, 0.2), (0.0, 0.6), (0.2, 0.6), (0.6, 0.6)]
            lcs = []
            for mat in base_mat:
                d, rho = mat
                s = 0.6 - d
                d = torch.ones_like(n) * d
                s = torch.ones_like(n) * s
                r = torch.ones_like(n[:,:1]) * rho
                base_svbrdf = torch.cat([n/2+0.5,d,r,s], dim=1) * 2 - 1
                lc = renderer.render(base_svbrdf, env, v) * 2 - 1
                if self.grey_lc: lc = lc.mean(1, keepdim=True)
                lcs.append(lc)
            lcs_cat = torch.stack(lcs, dim=1) # linear color space (-1, 1)
            if plane_lcs is not None:
                plane_lcs = plane_lcs.view(B, -1, 3, H, W)*2-1 # linear color space (-1, 1)
                lcs_cat = torch.cat([lcs_cat, plane_lcs], dim=2)
            lcs_cat = lcs_cat.view(B, -1, H, W)
        else:
            n = torch.ones_like(inp)
            lcs_cat = plane_lcs*2-1
            lcs = list(torch.chunk(lcs_cat, 6, dim=1))

        #* enc lcs
        lc_feats = self.lc_intro(lcs_cat)
        lc_encs = [] #! How to use skip connections?
        x = lc_feats
        for encoder, down in zip(self.lc_encoders, self.lc_downs):
            x = encoder(x)
            lc_encs.append(x)
            x = down(x)
        lc_feats = x.view(B, -1, *x.shape[-2:])
        
        #* assemble all features
        if self.use_attn_fuse:
            # cross attention for feature fusion
            raise NotImplementedError("Not implemented yet")
        else:
            # simple concatenate
            x = torch.cat([lc_feats, mid_x], dim=1)
            encs_feat = []
            for lc, i in zip(lc_encs, encs):
                encs_feat.append(i)
        
        x = self.fuse_blks(x)
        x = self.mat_intro(x)
        #* decode final materials
        for i, (decoder, up) in enumerate(zip(self.mat_decoders, self.mat_ups)):
            enc_skip = encs_feat[::-1][i] if i < len(encs_feat) else 0
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.tanh(self.mat_ending(x))
        return x, n, torch.cat(lcs, dim=1)
    
    def cal_nrm(self, nrm):
        normal = torch_norm(torch.cat([nrm,torch.ones_like(nrm[:,:1])], dim=1),dim=1)
        return normal