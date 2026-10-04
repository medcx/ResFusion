from .Restormer import Restormer
from einops import rearrange
from .classifier import VitBlock, PatchEmbedding2D
from .warped import Warper2d, warp2D
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from .basic_ops import linear, normalization, zero_module, conv_nd


class TimeBlock(nn.Module):
    """
    Simplified 2D Spatial-Awareness Convolution Block.
    Args:
      in_ch: input channels
      out_ch: output channels
      K: number of spatial clusters (spatially adaptive kernels)
      kernel_size: kernel size for per-cluster convs
      reduction: channel reduction for internal MLP (if any)
    """
    def __init__(self, in_ch, out_ch, kernels=[3,3,3,3], strides=[1,1,1,1], emb_channels=None, use_scale_shift_norm=False):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.K = len(kernels)

        self.cluster_head = nn.Sequential(
            nn.Conv2d(in_ch, max(8, in_ch // 2), kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, in_ch // 2), self.K, kernel_size=1)
        )

        paddings = [(k // 2) for k in kernels]

        self.cluster_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, in_ch,
                          kernel_size=kernels[i],
                          stride=strides[i],
                          padding=paddings[i],
                          bias=False),
                nn.ReLU(inplace=True)
            )
            for i in range(self.K)
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=max(1, out_ch // 8), num_channels=out_ch),
            nn.ReLU(inplace=True)
        )

        self.use_scale_shift_norm = use_scale_shift_norm
        self.emb_channels = emb_channels
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_ch if use_scale_shift_norm else self.out_ch,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_ch),
            nn.SiLU(),
            zero_module(
                conv_nd(2, self.out_ch, self.out_ch, 3, padding=1)
            ),
        )
    def forward(self, x, emb):
        """
        x: (B, C, H, W)
        returns: (B, out_ch, H, W)
        """
        logits = self.cluster_head(x)  # (B, K, H, W)
        cluster_maps = F.softmax(logits, dim=1)  # soft assignment across K

        conv_outs = []
        for k in range(self.K):
            yk = self.cluster_convs[k](x)  # (B, C, H, W)
            conv_outs.append(yk)

        stacked = torch.stack(conv_outs, dim=1)
        weighted = stacked * cluster_maps.unsqueeze(2)  # (B, K, C, H, W)
        fused = weighted.sum(dim=1)
        h = self.fuse(fused)

        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        out = h + x
        out = F.relu(out, inplace=True)
        return out


class DownSample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # conv stride2 downsample with GN+ReLU
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=max(1, out_ch // 8), num_channels=out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class TRNet(nn.Module):
    """
    A simple encoder composed of multiple SACB2D blocks with downsampling.
    Returns list of multi-scale features: [feat1, feat2, ..., featN]
    """
    def __init__(self, in_channels=2, base_channels=32, num_stages=4, emb_dim=None):
        super().__init__()
        self.num_stages = num_stages
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=1, padding=3, bias=False),
            nn.GroupNorm(num_groups=max(1, base_channels // 8), num_channels=base_channels),
            nn.ReLU(inplace=True)
        )

        chs = [base_channels * (2 ** i) for i in range(num_stages)]
        self.blocks = nn.ModuleList([])
        self.downs = nn.ModuleList([])
        self.regists = nn.ModuleList([])
        prev_ch = base_channels
        self.up_block = nn.ModuleList([])
        for i, c in enumerate(chs):
            self.blocks.append(TimeBlock(prev_ch, c, emb_channels=emb_dim))
            self.regists.append(PRBlock(c, c))
            self.up_block.append(
                nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                # nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, padding=1, stride=1),
                # nn.Tanh(),
                )
            )
            if i < num_stages - 1:
                self.downs.append(DownSample(c, c * 2))
                prev_ch = c * 2
            else:
                prev_ch = c

    def forward(self, x, y, emb):
        x_feats: List[torch.Tensor] = []
        y_feats: List[torch.Tensor] = []
        x = self.stem(x)
        y = self.stem(y)
        for i in range(self.num_stages):
            x = self.blocks[i](x, emb)
            y = self.blocks[i](y, emb)
            x_feats.append(x)
            y_feats.append(y)
            if i < self.num_stages - 1:
                x = self.downs[i](x)
                y = self.downs[i](y)

        for i, (x_feat, y_feat, reg_block, up_block) in enumerate(zip(x_feats[::-1], y_feats[::-1],
                                                            self.regists[::-1], self.up_block[::-1])):
            if i == 0:
                flow = reg_block(x_feat, y_feat)
            else:
                flow = 2 * up_block(flow)
                x_feat_reg = reg(flow, x_feat)
                flow = flow + reg_block(x_feat_reg, y_feat)

        return flow


def reg(flow, feature):
    feature = Warper2d()(flow, feature)
    return feature


def project(x, image_size):
    """将 torch.Size([1, 512, 768]) 转换成图像形状[channel, W/p, H/p]"""
    W, H = image_size[0], image_size[1]
    x = rearrange(x, 'b (w h) hidden -> b w h hidden', w=W // 16, h=H // 16)
    x = x.permute(0, 3, 1, 2)
    return x

def r_loss(flow):
    dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
    dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
    dx = dx * dx
    dy = dy * dy
    d = torch.mean(dx) + torch.mean(dy)
    grad = d / 3.0
    return grad

image_warp = warp2D()
def project(x, image_size):
    """将 torch.Size([1, 512, 768]) 转换成图像形状[channel, W/p, H/p]"""
    W, H = image_size[0], image_size[1]
    x = rearrange(x, 'b (w h) hidden -> b w h hidden', w=W // 16, h=H // 16)
    x = x.permute(0, 3, 1, 2)
    return x
def img_warp(flow, I):

    return Warper2d()(flow, I)
def flow_integration_ir(flow1, flow2, flow3, flow4, flow5, flow6, flow7, flow8, flow9, flow10):
    up1 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
    up2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
    up3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
    up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    flow1, flow2 = up1(flow1)*16, up1(flow2)*16
    flow3, flow4 = up2(flow3)*8, up2(flow4)*8
    flow5, flow6 = up3(flow5)*4, up3(flow6)*4
    flow7, flow8 = up4(flow7)*2, up4(flow8)*2
    flow_neg = flow1 + flow3 + flow5 + flow7 + flow9
    flow_pos = flow2 + flow4 + flow6 + flow8 + flow10
    flow = flow_pos - flow_neg
    return flow, flow_neg, flow_pos

def reg(flow, feature):
    feature = Warper2d()(flow, feature)
    return feature

class model_classifer_lite(nn.Module):
    def __init__(self, in_c, num_heads, num_vit_blk, img_size, patch_size):
        super(model_classifer_lite, self).__init__()
        self.hidden_size = 256
        self.embedding = PatchEmbedding2D(in_c = in_c, embedding_dim=256, patch_size=patch_size)

        self.vit_blks = nn.Sequential()  #添加n个vit块
        for i in range(num_vit_blk):
            self.vit_blks.add_module(
                name=f'vit{i}',
                module=VitBlock(hidden_size=256,
                         num_heads=num_heads,
                         vit_drop=0.1, qkv_bias=False,
                         mlp_dim=256, mlp_drop=0.0)
            )
        self.norm = nn.LayerNorm(256)
        self.head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                  nn.GELU(),
                                  nn.Linear(self.hidden_size, 2))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

    def forward(self, x):
        class_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = self.vit_blks(x)
        class_token = x[:, 0]
        predict = self.head(class_token)
        return predict, class_token, x[:, 1:]

class Classifier_lite(nn.Module):
    def __init__(self, in_c, num_heads, num_vit_blk, img_size, patch_size):
        super(Classifier_lite, self).__init__()
        self.hidden_size = 256
        self.embedding = PatchEmbedding2D(in_c = in_c, embedding_dim=256, patch_size=patch_size)

        self.vit_blks = nn.Sequential()  #添加n个vit块
        for i in range(num_vit_blk):
            self.vit_blks.add_module(
                name=f'vit{i}',
                module=VitBlock(hidden_size=256,
                         num_heads=num_heads,
                         vit_drop=0.1, qkv_bias=False,
                         mlp_dim=256, mlp_drop=0.0)
            )
        self.norm = nn.LayerNorm(256)
        self.head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                  nn.GELU(),
                                  nn.Linear(self.hidden_size, 2))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

    def forward(self, x):
        x = self.embedding(x)  # image_embedding
        class_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = self.vit_blks(x)
        class_token = x[:, 0]
        predict = self.head(class_token)
        return predict, class_token, x[:, 1:]

class Transfer(nn.Module):
    def __init__(self, num_vit, num_heads):
        super(Transfer, self).__init__()
        self.num_vit = num_vit
        self.num_heads = num_heads
        self.hidden_dim = 256
        self.cls1 = nn.Parameter(torch.zeros(1, 1, 256))
        self.cls2 = nn.Parameter(torch.zeros(1, 1, 256))
        self.VitBLK1 = nn.Sequential()
        for i in range(self.num_vit):
            self.VitBLK1.add_module(name=f'vit{i}',
                                    module=VitBlock(hidden_size=self.hidden_dim,
                                                    num_heads=self.num_heads,
                                                    vit_drop=0.0,
                                                    qkv_bias=False,
                                                    mlp_dim=256,
                                                    mlp_drop=0.0))
        self.VitBLK2 = nn.Sequential()
        for i in range(self.num_vit):
            self.VitBLK2.add_module(name=f'vit{i}',
                                    module=VitBlock(hidden_size=self.hidden_dim,
                                                    num_heads=self.num_heads,
                                                    vit_drop=0.0,
                                                    qkv_bias=False,
                                                    mlp_dim=256,
                                                    mlp_drop=0.0))
    def forward(self, x1, x2, cls1, cls2):
        cls1, cls2 = cls1.unsqueeze(dim=1), cls2.unsqueeze(dim=1)
        cls1 = cls1.expand(-1, x1.shape[1], -1)
        cls2 = cls2.expand(-1, x1.shape[1], -1)
        x1, x2 = x1+cls2, x2 + cls1
        class_token1 = self.cls1.expand(x1.shape[0], -1, -1)
        class_token2 = self.cls2.expand(x1.shape[0], -1, -1)
        # x1, x2 = self.MLP1(x1), self.MLP2(x2)
        x1 = torch.cat((x1, class_token1), dim=1)
        x2 = torch.cat((x2, class_token2), dim=1)
        x1 = self.VitBLK1(x1)
        x2 = self.VitBLK2(x2)
        class_token1 = x1[:, 0, :]
        class_token2 = x2[:, 0, :]
        return  x1[:, 1:, :], x2[:, 1:, :], class_token1, class_token2

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.rb1 = Restormer(2, 128)
        self.rb2 = Restormer(128, 256)

    def forward(self, img):
        f = self.rb1(img)
        f_ = self.rb2(f)
        return f, f_

class ModelTransfer_lite(nn.Module):
    def __init__(self, num_vit, num_heads, img_size):
        super(ModelTransfer_lite, self).__init__()
        self.img_size = img_size
        self.transfer = Transfer(num_vit=num_vit, num_heads=num_heads)
        self.classifier = Classifier_lite(in_c=3, num_heads=4, num_vit_blk=2, img_size=self.img_size, patch_size=16)
        self.modal_dis = model_classifer_lite(in_c=3, num_heads=4, num_vit_blk=2, img_size=self.img_size, patch_size=16)

    def forward(self, img1, img2):

        pre1, cls1, x1_ = self.classifier(img1)
        pre2, cls2, x2_ = self.classifier(img2)
        x1, x2, new_cls1, new_cls2 = self.transfer(x1_, x2_, cls1, cls2)
        feature_pred1, _, _ = self.modal_dis(x1)
        feature_pred2, _, _ = self.modal_dis(x2)
        return  pre1, pre2, feature_pred1, feature_pred2, x1, x2, x1_, x2_  # 分类器预测结果，特征转换器分类结果


class CrossGuided(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CrossGuided, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, stride=1),
                                   # nn.InstanceNorm2d(in_channel),
                                   nn.ReLU(),
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, stride=1),
                                   # nn.InstanceNorm2d(in_channel),
                                   nn.ReLU(),
                                   )
    def forward(self, f1, f2):
        f1_hat = f1
        f1 = self.conv1(f1)
        f2 = self.conv2(f2)
        att_map = f1 * f2
        att_shape = att_map.shape
        att_map = torch.reshape(att_map, [att_shape[0], att_shape[1], -1])
        att_map = F.softmax(att_map, dim=2)
        att_map = torch.reshape(att_map, att_shape)
        f1 = f1 * att_map
        f1 = f1 + f1_hat
        return f1

class ResBlk(nn.Module):
    def __init__(self, in_channel):
        super(ResBlk, self).__init__()
        self.feature_output = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )
    def forward(self ,x):
        return x + self.feature_output(x)

class PRBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PRBlock, self).__init__()
        self.cross = CrossGuided(in_channel, out_channel)
        self.feature_output = nn.Sequential(
            ResBlk(in_channel),
        )

        self.flow_output = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=2, kernel_size=3, padding=1, stride=1),
            nn.Tanh(),
        )

    def forward(self, f1, f2): # f2是cat后的特征
        f1 = self.cross(f1, f2)
        f1 = self.feature_output(f1)
        f2 = self.flow_output(f1)
        return f2


class UpBlk(nn.Module):
    def __init__(self, in_c, out_c):
        super(UpBlk, self).__init__()
        self.up_sample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, stride=1),
        )
        self.conv1 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, padding=1, stride=1)
        self.in1 = nn.InstanceNorm2d(num_features=out_c)


    def forward(self, x):
        x = self.up_sample(x)
        x = self.conv1(x)
        x = self.in1(x)
        return F.leaky_relu(x)

class Decoder(nn.Module):
    def __init__(self, channels):
        super(Decoder, self).__init__()
        self.channels = channels

        self.up1 = UpBlk(self.channels[0], self.channels[1])
        self.up2 = UpBlk(self.channels[1], self.channels[2])
        self.up3 = UpBlk(self.channels[2], self.channels[3])
        self.up4 = UpBlk(self.channels[3], self.channels[4])

class RegNet_lite(nn.Module):
    def __init__(self):
        super(RegNet_lite, self).__init__()
        self.channels = [256, 64, 32, 16, 8, 1]
        self.f1_FR1 = PRBlock(in_channel=self.channels[0], out_channel=self.channels[1])
        self.f1_FR2 = PRBlock(in_channel=self.channels[1], out_channel=self.channels[2])
        self.f1_FR3 = PRBlock(in_channel=self.channels[2], out_channel=self.channels[3])
        self.f1_FR4 = PRBlock(in_channel=self.channels[3], out_channel=self.channels[4])
        self.f1_FR5 = PRBlock(in_channel=self.channels[4], out_channel=self.channels[5])

        self.f2_FR1 = PRBlock(in_channel=self.channels[0], out_channel=self.channels[1])
        self.f2_FR2 = PRBlock(in_channel=self.channels[1], out_channel=self.channels[2])
        self.f2_FR3 = PRBlock(in_channel=self.channels[2], out_channel=self.channels[3])
        self.f2_FR4 = PRBlock(in_channel=self.channels[3], out_channel=self.channels[4])
        self.f2_FR5 = PRBlock(in_channel=self.channels[4], out_channel=self.channels[5])
        self.D = Decoder(self.channels)

    def forward(self, f1, f2):

        f_cat = torch.cat((f1, f2), dim=1)
        f1_, flow1 = self.f1_FR1(f1, f_cat)
        f2_, flow2 = self.f2_FR1(f2, f_cat)
        f1 = reg(flow1, f1)
        f2 = reg(flow2, f2)
        f1 = self.D.up1(f1)
        f2 = self.D.up1(f2)

        f_cat = torch.cat((f1, f2), dim=1)
        f1_, flow3 = self.f1_FR2(f1_, f_cat)
        f2_, flow4 = self.f2_FR2(f2_, f_cat)
        f1 = reg(flow3, f1)
        f2 = reg(flow4, f2)
        f1 = self.D.up2(f1)
        f2 = self.D.up2(f2)

        f_cat = torch.cat((f1, f2), dim=1)
        f1_, flow5 = self.f1_FR3(f1_, f_cat)
        f2_, flow6 = self.f2_FR3(f2_, f_cat)
        f1 = reg(flow5, f1)
        f2 = reg(flow6, f2)
        f1 = self.D.up3(f1)
        f2 = self.D.up3(f2)

        f_cat = torch.cat((f1, f2), dim=1)
        f1_, flow7 = self.f1_FR4(f1_, f_cat)
        f2_, flow8 = self.f2_FR4(f2_, f_cat)
        f1 = reg(flow7, f1)
        f2 = reg(flow8, f2)
        f1 = self.D.up4(f1)
        f2 = self.D.up4(f2)

        f_cat = torch.cat((f1, f2), dim=1)
        f1_, flow9 = self.f1_FR5(f1_, f_cat)
        f2_, flow10 = self.f2_FR5(f2_, f_cat)

        f1 = reg(flow9, f1)
        f2 = reg(flow10, f2)

        flow, flow_neg, flow_pos = flow_integration_ir(flow1, flow2, flow3, flow4, flow5, flow6, flow7, flow8, flow9, flow10)
        flows = [flow1, flow2, flow3, flow4, flow5, flow6, flow7, flow8, flow9, flow10]

        return f1, f2, flows, flow, flow_neg, flow_pos



class UpSampler_V2(nn.Module):
    def __init__(self, in_c, out_c):
        super(UpSampler_V2, self).__init__()
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        self.up3 =  nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

    def forward(self, AU_F, BU_F, feature):
        AU_F = self.up1(AU_F)
        BU_F = self.up1(BU_F)
        feature = self.up3(feature)
        return AU_F, BU_F, feature


class FusionNet_lite(nn.Module):
    def __init__(self):
        super(FusionNet_lite, self).__init__()
        self.cn = [256, 64, 32, 16, 12, 8]

        self.F1 = Restormer(in_c=self.cn[0]*2, out_c=self.cn[1])
        self.up_sample1 = UpSampler_V2(in_c=self.cn[0], out_c=self.cn[1])

        self.F2 = Restormer(in_c=self.cn[1]*3, out_c=self.cn[2])
        self.up_sample2 = UpSampler_V2(in_c=self.cn[1], out_c=self.cn[2])

        self.F3 = Restormer(in_c=self.cn[2]*3, out_c=self.cn[3])
        self.up_sample3 = UpSampler_V2(in_c=self.cn[2], out_c=self.cn[3])

        self.F4 = Restormer(in_c=self.cn[3]*3, out_c=self.cn[4])
        self.up_sample4 =nn.Upsample(scale_factor=2, mode='bilinear')
        self.outLayer = nn.Sequential(Restormer(in_c=self.cn[4] + 16, out_c=self.cn[4]),
                                      Restormer(in_c=self.cn[4], out_c=1),
                                      nn.Sigmoid())

    def forward(self, AS_F, BS_F, AU_F, BU_F, flow):
        """
            AU_F, BU_F是原图同尺度
        """

        flow_d = F.upsample(flow, BU_F.size()[2:4], mode='bilinear') / 16
        BU_F_w = img_warp(flow_d, BU_F)
        feature = self.F1(torch.cat((AU_F, BU_F_w), dim=1))  # 通道数降低了

        AU_F, BU_F, feature = self.up_sample1(AU_F, BU_F, feature)
        flow_d = F.upsample(flow, BU_F.size()[2:4], mode='bilinear') / 8
        BU_F_w = img_warp(flow_d, BU_F)
        feature = self.F2(torch.cat([torch.cat([AU_F, BU_F_w], dim=1), feature], dim=1))

        AU_F, BU_F, feature = self.up_sample2(AU_F, BU_F, feature)
        flow_d = F.upsample(flow, BU_F.size()[2:4], mode='bilinear') / 4
        BU_F_w = img_warp(flow_d, BU_F)
        feature = self.F3(torch.cat([torch.cat([AU_F, BU_F_w], dim=1), feature], dim=1))

        AU_F, BU_F, feature = self.up_sample3(AU_F, BU_F, feature)
        flow_d = F.upsample(flow, BU_F.size()[2:4], mode='bilinear') / 2
        BU_F_w = img_warp(flow_d, BU_F)
        feature = self.F4(torch.cat([torch.cat([AU_F, BU_F_w], dim=1), feature], dim=1))

        feature = self.up_sample4(feature)
        BS_F_w = img_warp(flow, BS_F)
        S_F = torch.cat([AS_F, BS_F_w], dim=1)
        feature = self.outLayer(torch.cat([feature, S_F], dim=1))
        return feature

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    print(param_count)


# encoder = Encoder()
# count_param(encoder)
# transfer = ModelTransfer_lite(2,4, [256,256])
# count_param(transfer)
# reg_net = RegNet_lite()
# count_param(reg_net)
# fusion_net = FusionNet_lite()
# count_param(fusion_net)
#
# img1 = torch.rand(1, 1, 256, 256).cuda()
# img2 = torch.rand(1, 1, 256, 256).cuda()
# AS_F, feature1 = encoder(img1)
# BS_F, feature2 = encoder(img2)
#
# pre1, pre2, feature_pred1, feature_pred2, feature1, feature2, AU_F, BU_F = transfer(feature1, feature2) # 分类器， 模态鉴别器， 转换后特征， 转换前特征
#
# feature1 = project(feature1, [256, 256]).cuda()
# feature2 = project(feature2, [256, 256]).cuda()
# AU_F = project(AU_F, [256, 256])
# BU_F = project(BU_F, [256, 256])
#
# _, _, flows, flow, _, _ = reg_net(feature1, feature2)
# fusion_img = fusion_net(AS_F, BS_F, AU_F, BU_F, flow)
# print(fusion_img.shape)
#
# warped_img2 = image_warp(img2, flow)



