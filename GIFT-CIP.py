from typing import Optional, Tuple, Union, Dict
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from transformer import TransformerEncoder
from GIFT_config import get_config



def make_divisible(
    v: Union[float, int],
    divisor: Optional[int] = 8,
    min_value: Optional[Union[float, int]] = None,
) -> Union[float, int]:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvLayer(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = 1,
        groups: Optional[int] = 1,
        bias: Optional[bool] = False,
        use_norm: Optional[bool] = True,
        use_act: Optional[bool] = True,
    ) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        assert isinstance(kernel_size, Tuple)
        assert isinstance(stride, Tuple)

        padding = (
            int((kernel_size[0] - 1) / 2),
            int((kernel_size[1] - 1) / 2),
        )

        block = nn.Sequential()

        conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=padding,
            bias=bias
        )

        block.add_module(name="conv", module=conv_layer)

        if use_norm:
            norm_layer = nn.BatchNorm2d(num_features=out_channels, momentum=0.1)
            block.add_module(name="norm", module=norm_layer)

        if use_act:
            act_layer = nn.SiLU()
            block.add_module(name="act", module=act_layer)

        self.block = block

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class InvertedResidual(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: Union[int, float],
        skip_connection: Optional[bool] = True,
    ) -> None:
        assert stride in [1, 2]
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)

        super().__init__()

        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_module(
                name="exp_1x1",
                module=ConvLayer(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1
                ),
            )

        block.add_module(
            name="conv_3x3",
            module=ConvLayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                stride=stride,
                kernel_size=3,
                groups=hidden_dim
            ),
        )

        block.add_module(
            name="red_1x1",
            module=ConvLayer(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                use_act=False,
                use_norm=True,
            ),
        )

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.stride = stride
        self.use_res_connect = (
            self.stride == 1 and in_channels == out_channels and skip_connection
        )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class MobileViTBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        transformer_dim: int,
        ffn_dim: int,
        n_transformer_blocks: int = 2,
        head_dim: int = 32,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        patch_h: int = 8,
        patch_w: int = 8,
        conv_ksize: Optional[int] = 3,
        *args,
        **kwargs
    ) -> None:
        super().__init__()

        conv_3x3_in = ConvLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1
        )
        conv_1x1_in = ConvLayer(
            in_channels=in_channels,
            out_channels=transformer_dim,
            kernel_size=1,
            stride=1,
            use_norm=False,
            use_act=False
        )

        conv_1x1_out = ConvLayer(
            in_channels=transformer_dim,
            out_channels=in_channels,
            kernel_size=1,
            stride=1
        )
        conv_3x3_out = ConvLayer(
            in_channels=2 * in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1
        )

        self.local_rep = nn.Sequential()
        self.local_rep.add_module(name="conv_3x3", module=conv_3x3_in)
        self.local_rep.add_module(name="conv_1x1", module=conv_1x1_in)

        assert transformer_dim % head_dim == 0
        num_heads = transformer_dim // head_dim

        global_rep = [
            TransformerEncoder(
                embed_dim=transformer_dim,
                ffn_latent_dim=ffn_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout
            )
            for _ in range(n_transformer_blocks)
        ]
        global_rep.append(nn.LayerNorm(transformer_dim))
        self.global_rep = nn.Sequential(*global_rep)

        self.conv_proj = conv_1x1_out
        self.fusion = conv_3x3_out

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = transformer_dim
        self.n_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.n_blocks = n_transformer_blocks
        self.conv_ksize = conv_ksize

    def unfolding(self, x: Tensor) -> Tuple[Tensor, Dict]:
        patch_w, patch_h = self.patch_w, self.patch_h
        patch_area = patch_w * patch_h
        batch_size, in_channels, orig_h, orig_w = x.shape

        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)

        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
            interpolate = True

        # number of patches along width and height
        num_patch_w = new_w // patch_w  # n_w
        num_patch_h = new_h // patch_h  # n_h
        num_patches = num_patch_h * num_patch_w  # N

        # [B, C, H, W] -> [B * C * n_h, p_h, n_w, p_w]
        x = x.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w)
        # [B * C * n_h, p_h, n_w, p_w] -> [B * C * n_h, n_w, p_h, p_w]
        x = x.transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] -> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        x = x.reshape(batch_size, in_channels, num_patches, patch_area)
        # [B, C, N, P] -> [B, P, N, C]
        x = x.transpose(1, 3)
        # [B, P, N, C] -> [BP, N, C]
        x = x.reshape(batch_size * patch_area, num_patches, -1)

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h,
        }

        return x, info_dict

    def folding(self, x: Tensor, info_dict: Dict) -> Tensor:
        n_dim = x.dim()
        assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(
            x.shape
        )
        # [BP, N, C] --> [B, P, N, C]
        x = x.contiguous().view(
            info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1
        )

        batch_size, pixels, num_patches, channels = x.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        # [B, P, N, C] -> [B, C, N, P]
        x = x.transpose(1, 3)
        # [B, C, N, P] -> [B*C*n_h, n_w, p_h, p_w]
        x = x.reshape(batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w)
        # [B*C*n_h, n_w, p_h, p_w] -> [B*C*n_h, p_h, n_w, p_w]
        x = x.transpose(1, 2)
        # [B*C*n_h, p_h, n_w, p_w] -> [B, C, H, W]
        x = x.reshape(batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w)
        if info_dict["interpolate"]:
            x = F.interpolate(
                x,
                size=info_dict["orig_size"],
                mode="bilinear",
                align_corners=False,
            )
        return x

    def forward(self, x: Tensor) -> Tensor:
        res = x

        fm = self.local_rep(x)

        # convert feature map to patches
        patches, info_dict = self.unfolding(fm)

        # learn global representations
        for transformer_layer in self.global_rep:
            patches = transformer_layer(patches)

        # [B x Patch x Patches x C] -> [B x C x Patches x Patch]
        fm = self.folding(x=patches, info_dict=info_dict)

        fm = self.conv_proj(fm)

        fm = self.fusion(torch.cat((res, fm), dim=1))
        return fm


class GIFT_CIP(nn.Module):

    def __init__(self, model_cfg: Dict,num_classes: int = 1000):
        super().__init__()

        image_channels = 1
        out_channels = 16

        #route 1
        self.conv_1_a = ConvLayer(in_channels=image_channels, out_channels=out_channels, kernel_size=3, stride=2)
        self.layer_1_a, out_channels_a = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer1"])
        self.layer_2_a, out_channels_a = self._make_layer(input_channel=out_channels_a, cfg=model_cfg["layer2"])
        self.layer_3_a, out_channels_a = self._make_layer(input_channel=out_channels_a, cfg=model_cfg["layer3"])
        #route 2
        self.conv_1_v = ConvLayer(in_channels=image_channels, out_channels=out_channels, kernel_size=3, stride=2)
        self.layer_1_v, out_channels_v = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer1"])
        self.layer_2_v, out_channels_v = self._make_layer(input_channel=out_channels_v, cfg=model_cfg["layer2"])
        self.layer_3_v, out_channels_v = self._make_layer(input_channel=out_channels_v, cfg=model_cfg["layer3"])
        #route 3
        self.conv_1_az = ConvLayer(in_channels=image_channels, out_channels=out_channels, kernel_size=3, stride=2)
        self.layer_1_az, out_channels_az = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer1"])
        self.layer_2_az, out_channels_az = self._make_layer(input_channel=out_channels_az, cfg=model_cfg["layer2"])
        self.layer_3_az, out_channels_az = self._make_layer(input_channel=out_channels_az, cfg=model_cfg["layer3"])
        #route 4
        self.conv_1_vz = ConvLayer(in_channels=image_channels, out_channels=out_channels, kernel_size=3, stride=2)
        self.layer_1_vz, out_channels_vz = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer1"])
        self.layer_2_vz, out_channels_vz = self._make_layer(input_channel=out_channels_vz, cfg=model_cfg["layer2"])
        self.layer_3_vz, out_channels_vz = self._make_layer(input_channel=out_channels_vz, cfg=model_cfg["layer3"])

        #route 5
        self.layer_4_av, out_channels_av = self._make_layer(input_channel=out_channels_a, cfg=model_cfg["layer4"])
        self.layer_5_av, out_channels_av = self._make_layer(input_channel=out_channels_av, cfg=model_cfg["layer5"])
        exp_channels = min(model_cfg["last_layer_exp_factor"] * out_channels_av, 960)
        self.conv_1x1_exp_av = ConvLayer(in_channels=out_channels_av,out_channels=exp_channels,kernel_size=1)
        self.layer_av = nn.Sequential()
        self.layer_av.add_module(name="global_pool", module=nn.AdaptiveAvgPool2d(1))
        self.layer_av.add_module(name="flatten", module=nn.Flatten())
        self.layer_av.add_module(name="dropout", module=nn.Dropout(p=model_cfg["cls_dropout"]))
        #route 6
        self.layer_4_avz, out_channels_avz = self._make_layer(input_channel=out_channels_az, cfg=model_cfg["layer4"])
        self.layer_5_avz, out_channels_avz = self._make_layer(input_channel=out_channels_avz, cfg=model_cfg["layer5"])
        exp_channels = min(model_cfg["last_layer_exp_factor"] * out_channels_avz, 960)
        self.conv_1x1_exp_avz = ConvLayer(in_channels=out_channels_avz,out_channels=exp_channels,kernel_size=1)
        self.layer_avz = nn.Sequential()
        self.layer_avz.add_module(name="global_pool", module=nn.AdaptiveAvgPool2d(1))
        self.layer_avz.add_module(name="flatten", module=nn.Flatten())
        self.layer_avz.add_module(name="dropout", module=nn.Dropout(p=model_cfg["cls_dropout"]))

        self.layer_cf_1 = nn.Linear(in_features=5, out_features=320)
        self.layer_cf_2 = nn.Linear(in_features=320, out_features=160)

        self.linear1 = nn.Linear(1440, 2560)
        self.linear2 = nn.Linear(2560, 1280)
        self.linear3 = nn.Linear(1280, 640)
        self.last_linear4 = nn.Linear(in_features=640, out_features=num_classes)

        # weight init
        self.apply(self.init_parameters)

    def _make_layer(self, input_channel, cfg: Dict) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(input_channel=input_channel, cfg=cfg)
        else:
            return self._make_mobilenet_layer(input_channel=input_channel, cfg=cfg)

    @staticmethod
    def _make_mobilenet_layer(input_channel: int, cfg: Dict) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1

            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio
            )
            block.append(layer)
            input_channel = output_channels

        return nn.Sequential(*block), input_channel

    @staticmethod
    def _make_mit_layer(input_channel: int, cfg: Dict) -> [nn.Sequential, int]:
        stride = cfg.get("stride", 1)
        block = []

        if stride == 2:
            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4)
            )

            block.append(layer)
            input_channel = cfg.get("out_channels")

        transformer_dim = cfg["transformer_channels"]
        ffn_dim = cfg.get("ffn_dim")
        num_heads = cfg.get("num_heads", 4)
        head_dim = transformer_dim // num_heads

        if transformer_dim % head_dim != 0:
            raise ValueError("Transformer input dimension should be divisible by head dimension. "
                             "Got {} and {}.".format(transformer_dim, head_dim))

        block.append(MobileViTBlock(
            in_channels=input_channel,
            transformer_dim=transformer_dim,
            ffn_dim=ffn_dim,
            n_transformer_blocks=cfg.get("transformer_blocks", 1),
            patch_h=cfg.get("patch_h", 2),
            patch_w=cfg.get("patch_w", 2),
            dropout=cfg.get("dropout", 0.1),
            ffn_dropout=cfg.get("ffn_dropout", 0.0),
            attn_dropout=cfg.get("attn_dropout", 0.1),
            head_dim=head_dim,
            conv_ksize=3
        ))

        return nn.Sequential(*block), input_channel

    @staticmethod
    def init_parameters(m):
        if isinstance(m, nn.Conv2d):
            if m.weight is not None:
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Linear,)):
            if m.weight is not None:
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            pass

    def forward(self, a: Tensor, v: Tensor, az: Tensor,vz: Tensor,cf: Tensor) -> Tensor: #a动脉期瘤内，v静脉期瘤内，az动脉期瘤周，vz静脉期瘤周，cf临床特征
        #route 1
        a = self.conv_1_a(a)
        a = self.layer_1_a(a)
        a = self.layer_2_a(a)
        a = self.layer_3_a(a)
        # route 2
        v = self.conv_1_v(v)
        v = self.layer_1_v(v)
        v = self.layer_2_v(v)
        v = self.layer_3_v(v)
        # route 3
        az = self.conv_1_az(az)
        az = self.layer_1_az(az)
        az = self.layer_2_az(az)
        az = self.layer_3_az(az)
        # route 4
        vz = self.conv_1_vz(vz)
        vz = self.layer_1_vz(vz)
        vz = self.layer_2_vz(vz)
        vz = self.layer_3_vz(vz)
        #融和
        av = 0.5 * a + 0.5 * v
        avz = 0.5 * az + 0.5 * vz
        # route 5
        av = self.layer_4_av(av)
        av = self.layer_5_av(av)
        av = self.conv_1x1_exp_av(av)
        av = self.layer_av(av)
        # route 6
        avz = self.layer_4_avz(avz)
        avz = self.layer_5_avz(avz)
        avz = self.conv_1x1_exp_avz(avz)
        avz = self.layer_avz(avz)
        #临床特征
        cf = self.layer_cf_1(cf)
        cf = self.layer_cf_2(cf)

        # av_avz = 0.5*av + 0.5*avz

        av_avz_cf = torch.cat((av, avz,cf), dim=1)

        output = self.linear1(av_avz_cf)
        output = self.linear2(output)
        # output = torch.relu(output)
        output = self.linear3(output)
        # output = torch.relu(output)
        last_output = self.last_linear4(output)
        return last_output


def GIFT_CIP_(num_classes: int = 1000):
    config = get_config("small")
    m = GIFT_CIP(config, num_classes=num_classes)
    return m

