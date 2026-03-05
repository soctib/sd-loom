"""Original ldm/CompVis UNet, copied from Forge/ComfyUI with simplifications.

This is the same UNet architecture used by A1111, Forge, and ComfyUI — the original
CompVis/ldm code. Using this instead of diffusers' UNet2DConditionModel eliminates
the ~7% pixel difference caused by diffusers' reimplementation.

Simplifications vs Forge:
- No ControlNet support (apply_control removed)
- No extension hooks (transformer_options/block_modifiers/patches removed)
- No gradient checkpointing (inference only)
- No ConfigMixin (diffusers interop not needed)
- Attention inlined as PyTorch SDPA (matches Forge's attention_pytorch)
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat  # type: ignore[import-untyped]
from torch import nn

# ---------------------------------------------------------------------------
# Attention (matches Forge's attention_pytorch exactly)
# ---------------------------------------------------------------------------

def attention_function(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    heads: int, mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Multi-head attention via PyTorch SDPA (matches Forge's attention_pytorch)."""
    b, _, dim_head = q.shape
    dim_head //= heads
    q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
    return out.transpose(1, 2).reshape(b, -1, heads * dim_head)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def exists(val: object) -> bool:
    return val is not None


def default(val: object, d: object) -> object:  # noqa: ANN401
    return val if exists(val) else d


def conv_nd(dims: int, *args: object, **kwargs: object) -> nn.Module:
    if dims == 2:
        return nn.Conv2d(*args, **kwargs)  # type: ignore[arg-type]
    if dims == 3:
        return nn.Conv3d(*args, **kwargs)  # type: ignore[arg-type]
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims: int, *args: object, **kwargs: object) -> nn.Module:
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)  # type: ignore[arg-type]
    if dims == 2:
        return nn.AvgPool2d(*args, **kwargs)  # type: ignore[arg-type]
    if dims == 3:
        return nn.AvgPool3d(*args, **kwargs)  # type: ignore[arg-type]
    raise ValueError(f"unsupported dimensions: {dims}")


def timestep_embedding(
    timesteps: torch.Tensor, dim: int,
    max_period: int = 10000, repeat_only: bool = False,
) -> torch.Tensor:
    """Sinusoidal timestep embeddings. Matches Forge/Kohya exactly."""
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding


# ---------------------------------------------------------------------------
# Timestep blocks
# ---------------------------------------------------------------------------

class TimestepBlock(nn.Module):
    pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(  # type: ignore[override]
        self, x: torch.Tensor, emb: torch.Tensor,
        context: torch.Tensor | None = None,
        output_shape: list[int] | tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            elif isinstance(layer, Upsample):
                x = layer(x, output_shape=output_shape)
            else:
                x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------

class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(
        self, dim: int, dim_out: int | None = None,
        mult: int = 4, glu: bool = False, dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        project_in: nn.Module
        if not glu:
            project_in = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
        else:
            project_in = GEGLU(dim, inner_dim)
        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(
        self, query_dim: int, context_dim: int | None = None,
        heads: int = 8, dim_head: int = 64, dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None,
        value: torch.Tensor | None = None, mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(value) if value is not None else self.to_v(context)
        out = attention_function(q, k, v, self.heads, mask)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self, dim: int, n_heads: int, d_head: int, dropout: float = 0.0,
        context_dim: int | None = None, gated_ff: bool = True,
        checkpoint: bool = True, ff_in: bool = False,
        inner_dim: int | None = None, disable_self_attn: bool = False,
    ) -> None:
        super().__init__()
        self.ff_in: FeedForward | bool = ff_in or inner_dim is not None
        if inner_dim is None:
            inner_dim = dim
        self.is_res = inner_dim == dim
        if self.ff_in:
            self.norm_in = nn.LayerNorm(dim)
            self.ff_in = FeedForward(dim, dim_out=inner_dim, dropout=dropout, glu=gated_ff)
        self.disable_self_attn = disable_self_attn
        self.attn1 = CrossAttention(
            query_dim=inner_dim, heads=n_heads, dim_head=d_head, dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
        )
        self.norm1 = nn.LayerNorm(inner_dim)
        self.attn2 = CrossAttention(
            query_dim=inner_dim, context_dim=context_dim,
            heads=n_heads, dim_head=d_head, dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(inner_dim)
        self.ff = FeedForward(inner_dim, dim_out=dim, dropout=dropout, glu=gated_ff)
        self.norm3 = nn.LayerNorm(inner_dim)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.ff_in:
            x_skip = x
            x = self.ff_in(self.norm_in(x))  # type: ignore[operator]
            if self.is_res:
                x += x_skip

        n = self.norm1(x)
        context_attn1 = context if self.disable_self_attn else None
        n = self.attn1(n, context=context_attn1)
        x = x + n

        n = self.norm2(x)
        n = self.attn2(n, context=context)
        x = x + n

        x_skip_ff = x if self.is_res else None
        x = self.ff(self.norm3(x))
        if x_skip_ff is not None:
            x = x + x_skip_ff
        return x


class SpatialTransformer(nn.Module):
    def __init__(
        self, in_channels: int, n_heads: int, d_head: int,
        depth: int = 1, dropout: float = 0.0,
        context_dim: int | list[int] | None = None,
        disable_self_attn: bool = False, use_linear: bool = False,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim] * depth
        inner_dim = n_heads * d_head
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        if not use_linear:
            self.proj_in: nn.Conv2d | nn.Linear = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0,
            )
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim, n_heads, d_head, dropout=dropout,
                context_dim=context_dim[d] if context_dim else None,
                disable_self_attn=disable_self_attn, checkpoint=use_checkpoint,
            )
            for d in range(depth)
        ])
        if not use_linear:
            self.proj_out: nn.Conv2d | nn.Linear = nn.Conv2d(
                inner_dim, in_channels, kernel_size=1, stride=1, padding=0,
            )
        else:
            self.proj_out = nn.Linear(in_channels, inner_dim)
        self.use_linear = use_linear

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context_list: list[torch.Tensor | None]
        if not isinstance(context, list):
            context_list = [context] * len(self.transformer_blocks)
        else:
            context_list = context  # type: ignore[assignment]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context_list[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


# ---------------------------------------------------------------------------
# U-Net blocks
# ---------------------------------------------------------------------------

class Upsample(nn.Module):
    def __init__(
        self, channels: int, use_conv: bool, dims: int = 2,
        out_channels: int | None = None, padding: int = 1,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(
        self, x: torch.Tensor,
        output_shape: list[int] | tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        assert x.shape[1] == self.channels
        if self.dims == 3:
            shape = [x.shape[2], x.shape[3] * 2, x.shape[4] * 2]
            if output_shape is not None:
                shape[1] = output_shape[3]
                shape[2] = output_shape[4]
        else:
            shape = [x.shape[2] * 2, x.shape[3] * 2]
            if output_shape is not None:
                shape[0] = output_shape[2]
                shape[1] = output_shape[3]
        x = F.interpolate(x, size=shape, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(
        self, channels: int, use_conv: bool, dims: int = 2,
        out_channels: int | None = None, padding: int = 1,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding,
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    def __init__(
        self, channels: int, emb_channels: int, dropout: float,
        out_channels: int | None = None, use_conv: bool = False,
        use_scale_shift_norm: bool = False, dims: int = 2,
        use_checkpoint: bool = False, up: bool = False, down: bool = False,
        kernel_size: int | list[int] = 3, exchange_temb_dims: bool = False,
        skip_t_emb: bool = False,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm
        self.exchange_temb_dims = exchange_temb_dims

        if isinstance(kernel_size, list):
            padding = [k // 2 for k in kernel_size]
        else:
            padding = kernel_size // 2

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding),
        )

        self.updown = up or down
        if up:
            self.h_upd: nn.Module = Upsample(channels, False, dims)
            self.x_upd: nn.Module = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.skip_t_emb = skip_t_emb
        self.emb_layers: nn.Sequential | None
        if self.skip_t_emb:
            self.emb_layers = None
            self.exchange_temb_dims = False
        else:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    emb_channels,
                    2 * self.out_channels if use_scale_shift_norm else self.out_channels,
                ),
            )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            conv_nd(dims, self.out_channels, self.out_channels, kernel_size, padding=padding),
        )

        if self.out_channels == channels:
            self.skip_connection: nn.Module = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, kernel_size, padding=padding,
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(  # type: ignore[override]
        self, x: torch.Tensor, emb: torch.Tensor,
    ) -> torch.Tensor:
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = None
        if not self.skip_t_emb:
            assert self.emb_layers is not None
            emb_out = self.emb_layers(emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            h = out_norm(h)
            if emb_out is not None:
                scale, shift = torch.chunk(emb_out, 2, dim=1)
                h *= (1 + scale)
                h += shift
            h = out_rest(h)
        else:
            if emb_out is not None:
                if self.exchange_temb_dims:
                    emb_out = rearrange(emb_out, "b t c ... -> b c t ...")
                h = h + emb_out
            h = self.out_layers(h)

        return self.skip_connection(x) + h


# ---------------------------------------------------------------------------
# Main UNet
# ---------------------------------------------------------------------------

class IntegratedUNet2DConditionModel(nn.Module):
    """Original ldm/CompVis SDXL UNet. Matches Forge/ComfyUI/A1111 exactly."""

    def __init__(
        self, in_channels: int, model_channels: int, out_channels: int,
        num_res_blocks: int | list[int], dropout: float = 0,
        channel_mult: tuple[int, ...] | list[int] = (1, 2, 4, 8),
        conv_resample: bool = True, dims: int = 2,
        num_classes: int | str | None = None,
        use_checkpoint: bool = False,
        num_heads: int = -1, num_head_channels: int = -1,
        use_scale_shift_norm: bool = False, resblock_updown: bool = False,
        use_spatial_transformer: bool = False,
        transformer_depth: list[int] | int = 1,
        context_dim: int | None = None,
        disable_self_attentions: list[bool] | None = None,
        num_attention_blocks: list[int] | None = None,
        disable_middle_self_attn: bool = False,
        use_linear_in_transformer: bool = False,
        adm_in_channels: int | None = None,
        transformer_depth_middle: int | None = None,
        transformer_depth_output: list[int] | None = None,
    ) -> None:
        super().__init__()
        if context_dim is not None:
            assert use_spatial_transformer
        if num_heads == -1:
            assert num_head_channels != -1
        if num_head_channels == -1:
            assert num_heads != -1

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            self.num_res_blocks = num_res_blocks

        if isinstance(transformer_depth, int):
            transformer_depth_list = len(channel_mult) * [transformer_depth]
        else:
            transformer_depth_list = transformer_depth[:]

        if transformer_depth_output is None:
            transformer_depth_output_list = transformer_depth_list[:]
        else:
            transformer_depth_output_list = transformer_depth_output[:]

        self.num_classes = num_classes
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb: nn.Module = nn.Embedding(num_classes, time_embed_dim)  # type: ignore[arg-type]
            elif self.num_classes == "continuous":
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        nn.Linear(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        nn.Linear(time_embed_dim, time_embed_dim),
                    )
                )
            else:
                raise ValueError("Bad ADM")

        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))
        ])
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers: list[nn.Module] = [
                    ResBlock(
                        channels=ch, emb_channels=time_embed_dim, dropout=dropout,
                        out_channels=mult * model_channels, dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                num_transformers = transformer_depth_list.pop(0)
                if num_transformers > 0:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    disabled_sa = False
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(SpatialTransformer(
                            ch, num_heads, dim_head, depth=num_transformers,
                            context_dim=context_dim, disable_self_attn=disabled_sa,
                            use_checkpoint=use_checkpoint,
                            use_linear=use_linear_in_transformer,
                        ))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                if resblock_updown:
                    self.input_blocks.append(TimestepEmbedSequential(
                        ResBlock(
                            channels=ch, emb_channels=time_embed_dim, dropout=dropout,
                            out_channels=out_ch, dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm, down=True,
                        )
                    ))
                else:
                    self.input_blocks.append(TimestepEmbedSequential(
                        Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    ))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels

        mid_block: list[nn.Module] = [
            ResBlock(
                channels=ch, emb_channels=time_embed_dim, dropout=dropout,
                out_channels=None, dims=dims, use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            )
        ]
        if transformer_depth_middle is not None and transformer_depth_middle >= 0:
            mid_block += [
                SpatialTransformer(
                    ch, num_heads, dim_head, depth=transformer_depth_middle,
                    context_dim=context_dim, disable_self_attn=disable_middle_self_attn,
                    use_checkpoint=use_checkpoint, use_linear=use_linear_in_transformer,
                ),
                ResBlock(
                    channels=ch, emb_channels=time_embed_dim, dropout=dropout,
                    out_channels=None, dims=dims, use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                ),
            ]
        self.middle_block = TimestepEmbedSequential(*mid_block)
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        channels=ch + ich, emb_channels=time_embed_dim, dropout=dropout,
                        out_channels=model_channels * mult, dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                num_transformers = transformer_depth_output_list.pop()
                if num_transformers > 0:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    disabled_sa = False
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    if not exists(num_attention_blocks) or i < num_attention_blocks[level]:
                        layers.append(SpatialTransformer(
                            ch, num_heads, dim_head, depth=num_transformers,
                            context_dim=context_dim, disable_self_attn=disabled_sa,
                            use_checkpoint=use_checkpoint,
                            use_linear=use_linear_in_transformer,
                        ))
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    if resblock_updown:
                        layers.append(ResBlock(
                            channels=ch, emb_channels=time_embed_dim, dropout=dropout,
                            out_channels=out_ch, dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm, up=True,
                        ))
                    else:
                        layers.append(Upsample(ch, conv_resample, dims=dims, out_channels=out_ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            conv_nd(dims, model_channels, out_channels, 3, padding=1),
        )

    def forward(
        self, x: torch.Tensor, timesteps: torch.Tensor,
        context: torch.Tensor | None = None, y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert (y is not None) == (self.num_classes is not None)

        hs: list[torch.Tensor] = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(x.dtype)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y is not None and y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)

        h = self.middle_block(h, emb, context)

        for module in self.output_blocks:
            hsp = hs.pop()
            h = torch.cat([h, hsp], dim=1)
            del hsp
            output_shape = hs[-1].shape if len(hs) > 0 else None
            h = module(h, emb, context, output_shape=output_shape)

        h = self.out(h)
        return h.type(x.dtype)
