from collections import OrderedDict
from typing import Tuple, Union
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import clip
from einops import rearrange
import torch.nn as nn
from mmcv.cnn import normal_init
from torch import optim
import math


class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, scale=1., num_tadapter=1, num_frames=8, drop_path=0.):
        super().__init__()
        self.num_tadapter = num_tadapter
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.n_head = n_head

        self.MLP_Adapter = Adapter(d_model, skip_connect=False)
        self.S_Adapter = Adapter(d_model)
        self.scale = scale
        self.T_Adapter = Adapter(d_model, skip_connect=False)
        if num_tadapter == 2:
            self.T_Adapter_in = Adapter(d_model)
        self.num_frames = num_frames
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        ## x shape [HW+1, BT, D]
        n, bt, d = x.shape
        ## temporal adaptation
        xt = rearrange(x, 'n (b t) d -> t (b n) d', t=self.num_frames)
        if self.num_tadapter == 2:
            xt = self.T_Adapter(self.attention(self.T_Adapter_in(self.ln_1(xt))))
        else:
            xt = self.T_Adapter(self.attention(self.ln_1(xt)))
        xt = rearrange(xt, 't (b n) d -> n (b t) d', n=n)
        x = x + self.drop_path(xt)
        ## spatial adaptation
        x = x + self.S_Adapter(self.attention(self.ln_1(x)))
        ## joint adaptation
        xn = self.ln_2(x)
        x = x + self.mlp(xn) + self.drop_path(self.scale * self.MLP_Adapter(xn))
        return x


class Transformer(nn.Module):
    def __init__(self, num_frames, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, num_tadapter=1, scale=1., drop_path=0.1):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.layers)]
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, scale, num_tadapter, num_frames, dpr[i]) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class ViT_CLIP(nn.Module):
    ## ViT definition in CLIP image encoder
    def __init__(self, input_resolution: int, num_frames: int, patch_size: int, width: int, layers: int, heads: int, drop_path_rate, num_tadapter=1, adapter_scale=0.5, pretrained=None, use_local=True):
        super().__init__()
        self.use_local = use_local
        self.input_resolution = input_resolution
        self.pretrained = pretrained
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.layers = layers
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.num_frames = num_frames
        self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, width))

        self.transformer = Transformer(num_frames, width, layers, heads, num_tadapter=num_tadapter, scale=adapter_scale, drop_path=drop_path_rate)

        self.ln_post = LayerNorm(width)

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            ## Load OpenAI CLIP pretrained weights
            if self.layers == 12:
                clip_model, preprocess = clip.load("ViT-B/16", device="cpu")
            else:
                clip_model, preprocess = clip.load("ViT-L/14", device="cpu")
            pretrain_dict = clip_model.visual.state_dict()
            del clip_model
            del pretrain_dict['proj']
            msg = self.load_state_dict(pretrain_dict, strict=False)
            torch.cuda.empty_cache()
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

        ## initialize S_Adapter
        for n, m in self.transformer.named_modules():
            if 'S_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        ## initialize T_Adapter
        for n, m in self.transformer.named_modules():
            if 'T_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        ## initialize MLP_Adapter
        for n, m in self.transformer.named_modules():
            if 'MLP_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed', 'temporal_embedding'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'temporal_position_bias_table'}

    def forward(self, x: torch.Tensor, info_embedding):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        if self.use_local:
            x, x_local = x[:,:3,:,:], x[:,3:,:,:]
        x = self.conv1(x)  
        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        if self.use_local:
            x_local = self.conv1(x_local)  
            x_local = x_local.reshape(x_local.shape[0], x_local.shape[1], -1) 
            x_local = x_local.permute(0, 2, 1)
            x_local = torch.cat([self.class_embedding.to(x_local.dtype) + torch.zeros(x_local.shape[0], 1, x_local.shape[-1], dtype=x_local.dtype, device=x_local.device), x_local], dim=1)
            x_local = x_local + self.positional_embedding.to(x_local.dtype)

        info_embedding = rearrange(info_embedding, 'b t n d -> (b t) n d')
        if self.use_local:
            x = torch.cat([x, x_local, info_embedding], dim=1)
        else:
            x = torch.cat([x, info_embedding], dim=1)
        n = x.shape[1]
        x = rearrange(x, '(b t) n d -> (b n) t d', t=self.num_frames)
        x = x + self.temporal_embedding
        x = rearrange(x, '(b n) t d -> (b t) n d', n=n)
            
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        x = x[:, 0]
        x = rearrange(x, '(b t) d -> b d t',b=B,t=T)
        
        x = x.unsqueeze(-1).unsqueeze(-1)  # BDTHW for I3D head

        return x


class I3DHead_location(nn.Module):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 output_dim,
                 in_channels,
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01):
        super().__init__()
        self.output_dim = output_dim
        self.in_channels = in_channels
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.output_dim)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x, extra_info):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        b, c, t, h, w = x.size()
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = x.transpose(1, 2).contiguous()
        x = x.view(b*t, -1)
        extra_info = extra_info.view(b*t, -1)
        x = torch.cat([x, extra_info], dim=-1)
        # [N, in_channels]
        x = self.fc_cls(x)
        # [N, num_classes]
        x = x.view(b, t, -1)
        return x


class ViTCLIPClassifier_location(nn.Module):
    def __init__(self,
                 input_resolution=224,
                 num_frames=32, 
                 patch_size=16,
                 width=768,
                 layers=12,
                 heads=12,
                 num_classes=19,
                 hide_dim = 256,
                 drop_path_rate=0.2, 
                 adapter_scale=0.5,
                 dropout_ratio=0.5,
                 use_local=True,
                 head_type='3d',
                 pretrained="ViT-B/16") -> None:
        super().__init__()
        self.use_bias = True
        self.HEAD_LAYERS = 3
        self.head_size = hide_dim*2
        self.prior_prob = 0.01
        self.REG_HEAD_TIME_SIZE = 3
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        self.input_resolution = input_resolution
        self.bbox_embedding = nn.Embedding(input_resolution+1, width)
        self.head_type=head_type

        self.backbone = ViT_CLIP(
            input_resolution=input_resolution,
            num_frames=num_frames, 
            patch_size=patch_size,
            width=width,
            layers=layers,
            heads=heads,
            drop_path_rate=drop_path_rate, 
            adapter_scale=adapter_scale,
            pretrained=pretrained,
            use_local=use_local
            )
        self.backbone.init_weights()

        if self.head_type == '3d':
            self.action_cls_head = I3DHead_location(
                output_dim=hide_dim, 
                in_channels=width + 4, 
                spatial_type=None,
                dropout_ratio=dropout_ratio)
            self.location_cls_head = I3DHead_location(
                output_dim=hide_dim, 
                in_channels=width + 4, 
                spatial_type=None,
                dropout_ratio=dropout_ratio)
            self.cls_head = self.make_head(
                num_classes, self.REG_HEAD_TIME_SIZE, self.HEAD_LAYERS)
            nn.init.constant_(self.cls_head[-1].bias, bias_value)
        else:
            self.cls_head = I3DHead_location(
                output_dim=num_classes, 
                in_channels=width + 4, 
                spatial_type=None,
                dropout_ratio=dropout_ratio)

    def make_head(self, out_planes, time_kernel,  num_heads_layers):
        layers = []
        use_bias = self.use_bias
        head_size = self.head_size
        
        for kk in range(num_heads_layers):
            branch_kernel = 1
            bpad = 0
            layers.append(nn.Conv3d(head_size, head_size, kernel_size=(
                branch_kernel, 3, 3), stride=1, padding=(bpad, 1, 1), bias=use_bias))
            layers.append(nn.ReLU(True))

        tpad = time_kernel//2
        layers.append(nn.Conv3d(head_size, out_planes, kernel_size=(
            time_kernel, 3, 3), stride=1, padding=(tpad, 1, 1)))
        
        layers = nn.Sequential(*layers)

        for m in layers.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m.bias, 'data'):
                    nn.init.constant_(m.bias, 0)

        return layers

    def forward(self, x, boxes):
        # bs, c, s, w, h = x.size()
        boxes_long = (boxes*self.input_resolution).long()
        boxes_embedding = self.bbox_embedding(boxes_long) # [B, T, 4, 768]
        x = self.backbone(x, boxes_embedding)
        if self.head_type == '3d':
            x_act = self.action_cls_head(x, boxes) # [1, 1536, 8, 1, 1]
            x_loc = self.location_cls_head(x, boxes) # [1, 1536, 8, 1, 1]
            x = torch.cat([x_act, x_loc], dim=-1)
            b, t, d = x.size()
            x = x.transpose(1,2).contiguous()
            x = x.view(b, -1, t, 1, 1)
            x = self.cls_head(x).transpose(1,2).contiguous()
            x = x.view(b, t, -1)
        else:
            x = self.cls_head(x, boxes)
        return x


class ViTCLIPClassifier_location_base(nn.Module):
    def __init__(self,
                 input_resolution=224,
                 num_frames=32, 
                 patch_size=16,
                 width=768,
                 layers=12,
                 heads=12,
                 num_classes=19,
                 hide_dim = 256,
                 drop_path_rate=0.2, 
                 adapter_scale=0.5,
                 dropout_ratio=0.5,
                 use_local=True,
                 head_type='3d',
                 pretrained="ViT-B/16") -> None:
        super().__init__()
        self.use_bias = True
        self.HEAD_LAYERS = 3
        self.head_size = hide_dim*2
        self.prior_prob = 0.01
        self.REG_HEAD_TIME_SIZE = 3
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        self.input_resolution = input_resolution
        self.agent_class_num = 10
        self.location_class_num = 12
        self.action_class_num = 19

        # 增加宽、高信息
        self.bbox_embedding = nn.Embedding(input_resolution+1, width)
        # 增加一个agent类别embedding
        self.agent_embedding = nn.Embedding(self.agent_class_num, width)
        # 增加一个location类别embedding
        self.location_embedding = nn.Embedding(self.location_class_num, width)
        
        self.head_type = head_type
        self.use_local = use_local

        self.backbone = ViT_CLIP(
            input_resolution=input_resolution,
            num_frames=num_frames, 
            patch_size=patch_size,
            width=width,
            layers=layers,
            heads=heads,
            drop_path_rate=drop_path_rate, 
            adapter_scale=adapter_scale,
            pretrained=pretrained,
            use_local=use_local
            )
        self.backbone.init_weights()

        if self.head_type == '3d':
            self.action_cls_head = I3DHead_location(
                output_dim=hide_dim, 
                in_channels=width + 4, 
                spatial_type=None,
                dropout_ratio=dropout_ratio)
            self.location_cls_head = I3DHead_location(
                output_dim=hide_dim, 
                in_channels=width + 4, 
                spatial_type=None,
                dropout_ratio=dropout_ratio)
            self.cls_head = self.make_head(
                num_classes, self.REG_HEAD_TIME_SIZE, self.HEAD_LAYERS)
            nn.init.constant_(self.cls_head[-1].bias, bias_value)
        elif self.head_type == 'circle':
            self.action_cls_head = I3DHead_location(
                output_dim=self.action_class_num, 
                in_channels=width + hide_dim, 
                spatial_type=None,
                dropout_ratio=dropout_ratio)
            self.location_cls_head = I3DHead_location(
                output_dim=self.location_class_num, 
                in_channels=width + hide_dim, 
                spatial_type=None,
                dropout_ratio=dropout_ratio) # [B, T, 12]
            self.softmax = nn.Softmax(dim=-1) # [B, T, 12] -> max [B, T, 1] -> embedding# [B, T, 1, width=256]
            self.bbox_agent_to_hide = nn.Linear(7*width, hide_dim)
            self.all_to_hide = nn.Linear(8*width, hide_dim)
        else:
            self.cls_head = I3DHead_location(
                output_dim=num_classes, 
                in_channels=width + 4, 
                spatial_type=None,
                dropout_ratio=dropout_ratio)

    def make_head(self, out_planes, time_kernel,  num_heads_layers):
        layers = []
        use_bias = self.use_bias
        head_size = self.head_size
        
        for kk in range(num_heads_layers):
            branch_kernel = 1
            bpad = 0
            layers.append(nn.Conv3d(head_size, head_size, kernel_size=(
                branch_kernel, 3, 3), stride=1, padding=(bpad, 1, 1), bias=use_bias))
            layers.append(nn.ReLU(True))

        tpad = time_kernel//2
        layers.append(nn.Conv3d(head_size, out_planes, kernel_size=(
            time_kernel, 3, 3), stride=1, padding=(tpad, 1, 1)))
        
        layers = nn.Sequential(*layers)

        for m in layers.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m.bias, 'data'):
                    nn.init.constant_(m.bias, 0)

        return layers
    
    def bbox_thresh(self, x):
        x[x>1.0] = 1
        x[x<0.0] = 0
        return x

    def forward(self, x, boxes, agent_id):
        w = self.bbox_thresh(boxes[:,:,2] - boxes[:,:,0])
        h = self.bbox_thresh(boxes[:,:,3] - boxes[:,:,1])
        boxes = torch.cat([boxes, torch.stack([w, h], dim=-1)], dim=-1)
        bs, c, s, w, h = x.size()
        boxes_long = (boxes*self.input_resolution).long()
        boxes_embedding = self.bbox_embedding(boxes_long) # [B, T, 6, 768]
        agent_embedding = self.agent_embedding(agent_id) # [B, T, 1, 768]
        info_embedding = torch.cat([boxes_embedding, agent_embedding], dim=2) # [B, T, n, 768]
        x = self.backbone(x, info_embedding)
        if self.head_type == '3d':
            x_act = self.action_cls_head(x, boxes) # [1, 1536, 8, 1, 1]
            x_loc = self.location_cls_head(x, boxes) # [1, 1536, 8, 1, 1]
            x = torch.cat([x_act, x_loc], dim=-1)
            b, t, d = x.size()
            x = x.transpose(1,2).contiguous()
            x = x.view(b, -1, t, 1, 1)
            x = self.cls_head(x).transpose(1,2).contiguous()
            x = x.view(b, t, -1)
        elif self.head_type == 'circle':
            info_hide = self.bbox_agent_to_hide(info_embedding.view(bs*s, -1))
            x_loc = self.location_cls_head(x, info_hide) # [B, T, 19]
            x_loc_softmax = self.softmax(x_loc)
            loc_id = torch.argmax(x_loc_softmax, dim=-1, keepdim=True).long()
            loc_embedding = self.location_embedding(loc_id)
            new_embedding = torch.cat([info_embedding, loc_embedding], dim=2)
            new_hide = self.all_to_hide(new_embedding.view(bs*s, -1))
            x_act = self.action_cls_head(x, new_hide)
            x = torch.cat([x_act, x_loc], dim=-1)

        else:
            x = self.cls_head(x, boxes)
        return x


if __name__ == "__main__":
    '''
    # model settings
    model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ViT_CLIP',
        input_resolution=224,
        patch_size=16,
        num_frames=32,
        width=768,
        layers=12,
        heads=12,
        drop_path_rate=0.1),
    cls_head=dict(
        type='I3DHead',
        in_channels=768,
        num_classes=400,
        spatial_type='avg',
        dropout_ratio=0.5),
    test_cfg = dict(average_clips='prob'))
    '''
    device  = torch.device("cuda:3")
    model = ViTCLIPClassifier_location_base(
        input_resolution=224,
        num_frames=4, 
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        num_classes=22,
        drop_path_rate=0.2, 
        adapter_scale=0.5,
        head_type='circle',
        use_local=True
        ).to(device)
    for name, param in model.named_parameters():
        if 'lcoation_embedding' not in name and 'agent_embedding' not in name and 'bbox_embedding' not in name and 'temporal_embedding' not in name and 'ln_post' not in name and 'cls_head' not in name and 'Adapter' not in name:
            param.requires_grad = False
            # print('{}: {}'.format(name, param.requires_grad))
    for name, param in model.named_parameters():
        print('{}: {}'.format(name, param.requires_grad))
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_param = sum(p.numel() for p in model.parameters())
    print('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))
    inputs = torch.rand(size=(1, 6, 4, 224, 224)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    model.train()
    agent_id = (torch.rand(size=(1, 4, 1)) * 10).long()
    
    print(agent_id.max())
    output = model(inputs, torch.rand(size=(1, 4, 4)).to(device), agent_id.to(device))
    # output = model(inputs, torch.rand(size=(1, 4, 4)).to(device))
    print(output)
    print(output.shape)