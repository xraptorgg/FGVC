# ConvNeXt is a neural network architecture developed by researchers at Meta Platforms, Inc. and affiliates.

# This is based on the paper Zhuang Liu et al., "A ConvNet for the 2020s"
# at https://arxiv.org/abs/2201.03545

# Official repository: https://github.com/facebookresearch/ConvNeXt





# libraries and packages

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model







# Complete ConvNeXt architecture
# according to Zhuang Liu et al., "A ConvNet for the 2020s"

class ConvNeXt(nn.Module):
    """
    Complete ConvNeXt architecture.


    Args:
        in_channelss (int): number of input image channels. Default: 3
        num_classes (int): number of classes for classification head. Default: 1000
        block_config (list(int)): number of blocks at each stage. Default: [3, 3, 9, 3]
        stage_dims (int): feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_rate (float): stochastic depth rate. Default: 0.
        layer_init_scale (float): init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_channels = 3, num_classes = 1000,
                 block_config = [3, 3, 9, 3], stage_dims = [96, 192, 384, 768],
                 drop_rate = 0., layer_init_scale = 1e-6, head_init_scale = 0):
        super().__init__()
        
        
        self.downsample_layers = nn.ModuleList()
        
        stem = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = stage_dims[0], kernel_size = 4, stride = 4, padding = 0),
            LayerNorm(stage_dims[0], eps = 1e-6, data_format = "channels_first")
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(stage_dims[i], eps = 1e-6, data_format = "channels_first"),
                nn.Conv2d(in_channels = stage_dims[i], out_channels = stage_dims[i + 1], kernel_size = 2, stride = 2, padding = 0)
            )
            self.downsample_layers.append(downsample_layer)
            
            
        self.stages = nn.ModuleList()
        drop_rates=[x.item() for x in torch.linspace(0, drop_rate, sum(stage_dims))] 
        curr = 0
        for i, blocks in enumerate(block_config):
            stage = nn.Sequential(
                *[block(dim = stage_dims[i], init_scale_factor = layer_init_scale, drop_rate = drop_rates[curr + j]) for j in range(blocks)]
            )
            self.stages.append(stage)
            curr += blocks
        
        self.norm = nn.LayerNorm(stage_dims[-1], eps=1e-6)
        self.head = nn.Linear(in_features = stage_dims[-1], out_features = num_classes)
        
        self.apply(self.init_weights)
        
        if head_init_scale > 0:
            self.head.weight.data._mul(head_init_scale)
            self.head.bias.data._mul(head_init_scale)
        
    
    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std = 0.02)
            nn.init.constant_(m.bias, 0)
            
            
    def forward_arrange(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            
        
        return x
    
    def forward(self, x):
        x = self.forward_arrange(x)
        x = self.norm(x.mean([-2, -1]))
        x = self.head(x)
        
        return x







# Modified Residual block with inverted bottleneck,
# 7x7 convolution, GELU activation

class block(nn.Module):
    """
    Modified residual block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch.
    
    Args:
        dim (int): number of input channels.
        init_scale_factor (float): Init value for Layer Scale. Default: 1e-6.
        drop_rate (float): stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, init_scale_factor, drop_rate):
        super().__init__()
        
        self.dwconv = nn.Conv2d(in_channels = dim, out_channels = dim, kernel_size = 7, stride = 1, padding = 3, groups = dim)
        self.norm = LayerNorm(dim, eps = 1e-06)
        self.pwconv1 = nn.Linear(in_features = dim, out_features = 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(in_features = 4 * dim, out_features = dim)
        self.gamma = nn.Parameter(init_scale_factor * torch.ones((dim)), requires_grad = True) if init_scale_factor > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
        
    def forward(self, x):
        input = x
        
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        if self.gamma is not None:
            x = x * self.gamma
            
        x = x.permute(0, 3, 1, 2)
        
        x = input + self.drop_path(x)
        
        return x







# Layer Normalization

class LayerNorm(nn.Module):
    """ 
    Performs Layer Normalization, supports two data formats: channels_last (default) [N, H, W, C] or channels_first [N, C, H, W]. 
    

    Args:
        normalized_shape (int): shape of the input tensor over which normalization is applied
        eps (float): value added to the denominator for numerical stability. Default: 1e-6
        data_format (string): order in which dimensions are arranged. Default: "channels_last"
    """
    def __init__(self, normalized_shape, eps  =1e-6, data_format = "channels_last"):
        super().__init__()
        
        self.normalized_shape = (normalized_shape, )
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
            
        
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x





# URLs for pretrained ImageNet weights

model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}




# model instances

def convnext_tiny(pretrained = False, im_22k = False, device = "cpu", **kwargs):
    model = ConvNeXt(block_config = [3, 3, 9, 3], stage_dims = [96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if im_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url = url, map_location = device, check_hash = True)
        model.load_state_dict(checkpoint["model"])
    return model


def convnext_small(pretrained = False, im_22k = False, device = "cpu", **kwargs):
    model = ConvNeXt(block_config = [3, 3, 27, 3], stage_dims = [96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if im_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url = url, map_location = device, check_hash = True)
        model.load_state_dict(checkpoint["model"])
    return model


def convnext_base(pretrained = False, im_22k = False, device = "cpu", **kwargs):
    model = ConvNeXt(block_config = [3, 3, 27, 3], stage_dims = [128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if im_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url = url, map_location = device, check_hash = True)
        model.load_state_dict(checkpoint["model"])
    return model


def convnext_large(pretrained = False, im_22k = False, device = "cpu", **kwargs):
    model = ConvNeXt(block_config = [3, 3, 27, 3], stage_dims = [192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if im_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url = url, map_location = device, check_hash = True)
        model.load_state_dict(checkpoint["model"])
    return model



def convnext_xlarge(pretrained = False, im_22k = False, device = "cpu", **kwargs):
    model = ConvNeXt(block_config = [3, 3, 27, 3], stage_dims = [256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert im_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set im_22k = True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url = url, map_location = device, check_hash = True)
        model.load_state_dict(checkpoint["model"])
    return model





if __name__ == "__main__":
    x = torch.randn(size = (1, 3, 224, 224))
    model = convnext_small(pretrained = True)
    print(model(x))