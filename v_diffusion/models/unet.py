import math
import torch
import torch.nn as nn

try:
    from ..modules import Linear, Conv2d, Sequential, OneHot
    from ..functions import get_timestep_embedding
except ImportError:
    import sys
    from pathlib import Path
    PROJ_DIR = str(Path(__file__).resolve().parents[2])
    if PROJ_DIR not in sys.path:
        sys.path.append(PROJ_DIR)
    from v_diffusion.modules import Linear, Conv2d, Sequential, OneHot
    from v_diffusion.functions import get_timestep_embedding


DEFAULT_NONLINEARITY = nn.SiLU  # f(x)=x*sigmoid(x)


class DEFAULT_NORMALIZER(nn.GroupNorm):
    def __init__(self, num_channels, num_groups=32, eps=1e-6):
        super().__init__(num_groups=num_groups, num_channels=num_channels, eps=eps)


class AttentionBlock(nn.Module):
    normalize = DEFAULT_NORMALIZER

    def __init__(
            self,
            in_dim,
            head_dim=None,
            num_heads=None
    ):
        super(AttentionBlock, self).__init__()
        if head_dim is None:
            assert num_heads is not None and in_dim % num_heads == 0
            head_dim = in_dim // num_heads
        if num_heads is None:
            assert head_dim is not None and in_dim % head_dim == 0
            num_heads = in_dim // head_dim

        self.head_dim = head_dim
        self.num_heads = num_heads

        self.norm = self.normalize(in_dim)
        hid_dim = head_dim * num_heads
        self.proj_in = Conv2d(in_dim, 3 * hid_dim, 1)
        self.proj_out = Conv2d(hid_dim, in_dim, 1, init_scale=0.)

    @staticmethod
    def qkv(q, k, v):
        B, N, C, H, W = q.shape
        w = torch.einsum("bnchw, bncHW -> bnhwHW", q, k)
        w = torch.softmax(
            w.reshape(B, N, H, W, H * W) / math.sqrt(C), dim=-1)
        out = torch.einsum(
            "bnhwC, bncC -> bnchw",
            w, v.flatten(start_dim=3)).reshape(B, N * C, H, W)
        return out

    def forward(self, x, **kwargs):
        skip = x
        H, W = x.shape[2:]
        q, k, v = self.proj_in(self.norm(x)).reshape(
            -1, 3 * self.num_heads, self.head_dim, H, W
        ).chunk(3, dim=1)
        x = self.qkv(q, k, v)
        x = self.proj_out(x)
        return x + skip


class ResidualBlock(nn.Module):
    normalize = DEFAULT_NORMALIZER
    nonlinearity = DEFAULT_NONLINEARITY

    def __init__(
            self,
            in_channels,
            out_channels,
            embed_dim,
            drop_rate=0.,
            resampling="none"
    ):
        super(ResidualBlock, self).__init__()
        self.norm1 = self.normalize(in_channels)
        self.act1 = self.nonlinearity()
        self.conv1 = Conv2d(in_channels, out_channels, 3, 1, 1)
        self.fc = Linear(embed_dim, 2 * out_channels)
        self.norm2 = self.normalize(out_channels)
        self.act2 = self.nonlinearity()
        self.conv2 = Conv2d(out_channels, out_channels, 3, 1, 1, init_scale=0.)

        if resampling == "upsample":
            self.resample = nn.Upsample(scale_factor=2, mode="nearest")
        elif resampling == "downsample":
            self.resample = nn.AvgPool2d(2)
        else:
            self.resample = nn.Identity()

        self.skip = nn.Identity() if in_channels == out_channels else Conv2d(in_channels, out_channels, 1)
        self.dropout = nn.Dropout(p=drop_rate, inplace=True)

    def forward(self, x, t_emb):
        skip = self.skip(self.resample(x))
        # BigGAN-style up/down-sampling
        # norm -> act -> resample
        x = self.conv1(self.resample(self.act1(self.norm1(x))))
        t_emb = self.fc(self.act1(t_emb))[:, :, None, None]
        # conditioning via time/class-specific additive shift & multiplicative scale
        # different from DDPM implementation
        shift, scale = t_emb.chunk(2, dim=1)
        x = (1 + scale) * self.norm2(x) + shift
        x = self.conv2(self.dropout(self.act2(x)))
        return x + skip


class UNet(nn.Module):
    normalize = DEFAULT_NORMALIZER
    nonlinearity = DEFAULT_NONLINEARITY

    def __init__(
            self,
            in_channels,
            hid_channels,
            out_channels,
            ch_multipliers,
            num_res_blocks,
            apply_attn,
            embedding_dim=None,
            drop_rate=0.,
            head_dim=None,
            num_heads=None,
            num_classes=0,
            multitags=False,
            resample_with_res=True
    ):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        self.embedding_dim = embedding_dim or 4 * self.hid_channels
        self.levels = levels = len(ch_multipliers)
        self.ch_multipliers = ch_multipliers
        if isinstance(apply_attn, bool):
            apply_attn = [apply_attn for _ in range(levels)]
        self.apply_attn = apply_attn
        self.num_res_blocks = num_res_blocks
        self.drop_rate = drop_rate
        if head_dim is None and num_heads is None:
            num_heads = 1
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.multitags = multitags
        self.resample_with_res = resample_with_res

        self.time_embed = nn.Sequential(
            Linear(self.hid_channels, self.embedding_dim),
            self.nonlinearity(),
            Linear(self.embedding_dim, self.embedding_dim)
        )

        if self.num_classes > 0:
            if multitags:
                self.class_embed = nn.Linear(
                    self.num_classes, self.embedding_dim)
            else:
                self.class_embed = nn.Sequential(
                    OneHot(self.num_classes, exclude_zero=True),
                    Linear(self.num_classes, self.embedding_dim)
                )

        self.in_conv = Conv2d(in_channels, hid_channels, 3, 1, 1)
        self.downsamples = nn.ModuleDict(
            {f"level_{i}": self.downsample_level(i) for i in range(levels)})
        mid_channels = ch_multipliers[-1] * hid_channels
        mid_kwargs = dict(embed_dim=self.embedding_dim, drop_rate=drop_rate)
        self.middle = Sequential(
            ResidualBlock(mid_channels, mid_channels, **mid_kwargs),
            AttentionBlock(mid_channels, head_dim=head_dim, num_heads=num_heads),
            ResidualBlock(mid_channels, mid_channels, **mid_kwargs)
        )
        self.upsamples = nn.ModuleDict(
            {f"level_{i}": self.upsample_level(i) for i in range(levels)})
        self.out_conv = Sequential(
            self.normalize(hid_channels * ch_multipliers[0]),
            self.nonlinearity(),
            Conv2d(hid_channels * ch_multipliers[0], out_channels, 3, 1, 1, init_scale=0.))

    def get_level_block(self, level):
        block_cfgs = dict(
            embed_dim=self.embedding_dim,
            drop_rate=self.drop_rate)
        apply_attn = self.apply_attn[level]

        def block(in_chans, out_chans, resampling="none"):
            if apply_attn:
                return Sequential(ResidualBlock(
                    in_chans, out_chans, resampling=resampling, **block_cfgs),
                    AttentionBlock(out_chans, self.head_dim, self.num_heads))
            else:
                return ResidualBlock(
                    in_chans, out_chans, resampling=resampling, **block_cfgs)
        return block

    def downsample_level(self, level) -> nn.ModuleList:
        block = self.get_level_block(level)
        prev_chans = (self.ch_multipliers[level-1] if level else 1) * self.hid_channels
        curr_chans = self.ch_multipliers[level] * self.hid_channels
        modules = nn.ModuleList([block(prev_chans, curr_chans)])
        for _ in range(self.num_res_blocks - 1):
            modules.append(block(curr_chans, curr_chans))
        if level != self.levels - 1:
            if self.resample_with_res:
                _downsample = block(curr_chans, curr_chans, resampling="downsample")
            else:
                _downsample = Conv2d(curr_chans, curr_chans, 3, 2)
            modules.append(_downsample)
        return modules

    def upsample_level(self, level) -> nn.ModuleList:
        block = self.get_level_block(level)
        ch = self.hid_channels
        chs = list(map(lambda x: ch*x, self.ch_multipliers))
        next_chans = ch if level == 0 else chs[level - 1]
        prev_chans = chs[-1] if level == self.levels - 1 else chs[level + 1]
        curr_chans = chs[level]
        modules = nn.ModuleList([block(prev_chans + curr_chans, curr_chans)])
        for _ in range(self.num_res_blocks - 1):
            modules.append(block(2 * curr_chans, curr_chans))
        modules.append(block(next_chans + curr_chans, curr_chans))
        if level != 0:
            if self.resample_with_res:
                _upsample = block(curr_chans, curr_chans, resampling="upsample")
            else:
                _upsample = Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    Conv2d(curr_chans, curr_chans, 3, 1, 1))
            modules.append(_upsample)
        return modules

    def forward(self, x, t, y=None):
        t_emb = get_timestep_embedding(t, self.hid_channels)
        t_emb = self.time_embed(t_emb)
        if self.num_classes and y is not None:
            if self.multitags:
                assert y.ndim == 2
                y = y.div(torch.count_nonzero(
                    y, dim=1
                ).clamp(min=1.).sqrt().unsqueeze(1))
            t_emb += self.class_embed(y)
        # downsample
        hs = [self.in_conv(x)]
        for i in range(self.levels):
            downsample = self.downsamples[f"level_{i}"]
            for j, layer in enumerate(downsample):
                h = hs[-1]
                if j != self.num_res_blocks or self.resample_with_res:
                    hs.append(layer(h, t_emb=t_emb))
                else:
                    hs.append(layer(h))

        # middle
        h = self.middle(hs[-1], t_emb=t_emb)

        # upsample
        for i in range(self.levels-1, -1, -1):
            upsample = self.upsamples[f"level_{i}"]
            for j, layer in enumerate(upsample):
                if j != self.num_res_blocks + 1:
                    h = layer(torch.cat([h, hs.pop()], dim=1), t_emb=t_emb)
                elif self.resample_with_res:
                    h = layer(h, t_emb=t_emb)
                else:
                    h = layer(h)

        h = self.out_conv(h)
        return h


if __name__ == "__main__":
    model = UNet(3, 64, 3, (2, 2, 2), 3, (False, True, True))
    print(model)
    out = model(torch.randn(16, 3, 32, 32), t=torch.randint(1000, size=(16, )))
    print(out.shape)
