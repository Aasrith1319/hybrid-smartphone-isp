"""Networks. All operate on packed RGGB Bayer at half resolution.

Input  (B, 6, H/2, W/2): 4 compressed Bayer planes + 2 noise-parameter planes.
Output (B, 12, H/2, W/2) for the restorers: a residual that PixelShuffle turns into
full-resolution RGB in the compressed domain.

Compression is signed sqrt: it spreads dark values, roughly stabilises shot noise, and
keeps activations in a range that survives INT8 quantisation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.GroupNorm):
    def __init__(self, c):
        super().__init__(1, c)


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, 1)
        return a * b


class NAFBlock(nn.Module):
    """Nonlinear-activation-free block (Chen et al. 2022), used by the teacher."""

    def __init__(self, c):
        super().__init__()
        self.n1, self.n2 = LayerNorm2d(c), LayerNorm2d(c)
        self.c1 = nn.Conv2d(c, 2 * c, 1)
        self.dw = nn.Conv2d(2 * c, 2 * c, 3, padding=1, groups=2 * c)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1))
        self.c2 = nn.Conv2d(c, c, 1)
        self.c3 = nn.Conv2d(c, 2 * c, 1)
        self.c4 = nn.Conv2d(c, c, 1)
        self.gate = SimpleGate()
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = self.gate(self.dw(self.c1(self.n1(x))))
        x = x + self.beta * self.c2(y * self.sca(y))
        return x + self.gamma * self.c4(self.gate(self.c3(self.n2(x))))


class ConvBlock(nn.Module):
    """Two plain 3x3 convs with a skip: every op has an INT8 kernel."""

    def __init__(self, c):
        super().__init__()
        self.a = nn.Conv2d(c, c, 3, padding=1)
        self.b = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.b(F.relu(self.a(x)))


class UNet(nn.Module):
    def __init__(self, in_ch=6, out_ch=12, width=32, enc=(2, 2, 4), mid=6, dec=(2, 2, 2), block="naf"):
        super().__init__()
        B = NAFBlock if block == "naf" else ConvBlock
        self.intro = nn.Conv2d(in_ch, width, 3, padding=1)
        self.outro = nn.Conv2d(width, out_ch, 3, padding=1)
        self.encs, self.downs, self.ups, self.decs = (nn.ModuleList() for _ in range(4))
        c = width
        for n in enc:
            self.encs.append(nn.Sequential(*[B(c) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, 2 * c, 2, stride=2))
            c *= 2
        self.mid = nn.Sequential(*[B(c) for _ in range(mid)])
        for n in dec:
            self.ups.append(nn.Sequential(nn.Conv2d(c, 2 * c, 1, bias=False), nn.PixelShuffle(2)))
            c //= 2
            self.decs.append(nn.Sequential(*[B(c) for _ in range(n)]))
        self.pad_to = 2 ** len(enc)
        nn.init.zeros_(self.outro.weight)
        nn.init.zeros_(self.outro.bias)

    def forward(self, x):
        h, w = x.shape[-2:]
        p = self.pad_to
        if p > 1:
            x = F.pad(x, (0, (-w) % p, 0, (-h) % p), mode="replicate")
        y = self.intro(x)
        skips = []
        for e, d in zip(self.encs, self.downs):
            y = e(y)
            skips.append(y)
            y = d(y)
        y = self.mid(y)
        for u, d, s in zip(self.ups, self.decs, reversed(skips)):
            y = d(u(y) + s)
        y = self.outro(y)
        return y[..., :h, :w] if p > 1 else y


CONFIGS = {
    "teacher": dict(width=32, enc=(2, 2, 4), mid=6, dec=(2, 2, 2), block="naf"),
    "student": dict(width=16, enc=(1, 1, 2), mid=2, dec=(1, 1, 1), block="conv"),
    "student_s": dict(width=8, enc=(1, 1, 1), mid=1, dec=(1, 1, 1), block="conv"),
}


def build(name, **kw):
    return UNet(**{**CONFIGS[name], **kw})


def compress(x):
    return torch.sign(x) * torch.sqrt(x.abs() + 1e-8)


def expand(y):
    return torch.sign(y) * y * y


_K_G = torch.tensor([[0, 1, 0], [1, 4, 1], [0, 1, 0]], dtype=torch.float32) / 4
_K_RB = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32) / 4


def bilinear_demosaic(packed):
    """(B, 4, h, w) RGGB planes -> (B, 3, 2h, 2w); same kernels as classical.demosaic_bilinear."""
    b, _, h, w = packed.shape
    z = packed.new_zeros(b, 3, 2 * h, 2 * w)
    z[:, 0, 0::2, 0::2] = packed[:, 0]
    z[:, 1, 0::2, 1::2] = packed[:, 1]
    z[:, 1, 1::2, 0::2] = packed[:, 2]
    z[:, 2, 1::2, 1::2] = packed[:, 3]
    k = torch.stack([_K_RB, _K_G, _K_RB])[:, None].to(z)
    return F.conv2d(F.pad(z, (1, 1, 1, 1), mode="reflect"), k, groups=3)


def noise_planes(a, b, like):
    """Noise parameters as two constant planes, scaled to roughly [0, 1]."""
    a = (a.sqrt() * 16).view(-1, 1, 1, 1).expand(-1, 1, *like.shape[-2:])
    b = (b.sqrt() * 64).view(-1, 1, 1, 1).expand(-1, 1, *like.shape[-2:])
    return torch.cat([a, b], 1).to(like)


def restore(net, packed_wb, a, b):
    """White-balanced packed Bayer -> linear full-resolution camera RGB."""
    f = compress(packed_wb)
    res = F.pixel_shuffle(net(torch.cat([f, noise_planes(a, b, f)], 1)), 2)
    return expand(bilinear_demosaic(f) + res)


def srgb_oetf(x):
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp_min(1e-8) ** (1 / 2.4) - 0.055)


def render(lin, ccm):
    """Fixed classical colour stages, differentiable. `ccm` is (B, 3, 3)."""
    return srgb_oetf(torch.einsum("bij,bjhw->bihw", ccm, lin.clamp(0, 1)))
