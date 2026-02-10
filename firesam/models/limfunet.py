import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    # Squeeze-and-Excitation block.
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        if channels < reduction:
            reduction = max(1, channels // 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        se = self.avg_pool(x)
        se = F.relu(self.fc1(se), inplace=True)
        se = torch.sigmoid(self.fc2(se))
        return x * se


class GhostBlock(nn.Module):
    # Ghost block with SE as in LimFUNet.
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ratio: int = 2,
        leak: float = 0.1,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.ratio = ratio

        real_channels = out_channels // ratio
        ghost_channels = out_channels - real_channels
        if real_channels == 0:
            real_channels = out_channels
            ghost_channels = 0

        self.real_conv = nn.Sequential(
            nn.Conv2d(in_channels, real_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(real_channels),
            nn.LeakyReLU(leak, inplace=True),
        )

        if ghost_channels > 0:
            self.dw_conv = nn.Conv2d(
                real_channels,
                real_channels,
                kernel_size=3,
                padding=1,
                groups=real_channels,
                bias=False,
            )
            self.dw_bn = nn.BatchNorm2d(real_channels)
            self.dw_act = nn.LeakyReLU(leak, inplace=True)

            self.ghost_conv = nn.Conv2d(
                real_channels,
                ghost_channels,
                kernel_size=1,
                bias=False,
            )
            self.ghost_bn = nn.BatchNorm2d(ghost_channels)
            self.ghost_act = nn.LeakyReLU(leak, inplace=True)
        else:
            self.dw_conv = None
            self.dw_bn = None
            self.dw_act = None
            self.ghost_conv = None
            self.ghost_bn = None
            self.ghost_act = None

        self.se = SEBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real = self.real_conv(x)
        if self.ghost_conv is not None:
            ghost = self.dw_conv(real)
            ghost = self.dw_bn(ghost)
            ghost = self.dw_act(ghost)
            ghost = self.ghost_conv(ghost)
            ghost = self.ghost_bn(ghost)
            ghost = self.ghost_act(ghost)
            out = torch.cat([real, ghost], dim=1)
        else:
            out = real
        out = self.se(out)
        return out


class LimFUNetEncoder(nn.Module):
    # LimFUNet encoder with constant channel width.
    def __init__(
        self,
        in_channels: int = 6,
        base_channels: int = 32,
        ghost_ratio: int = 2,
        leak: float = 0.1,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(leak, inplace=True),
        )
        self.f1 = GhostBlock(base_channels, base_channels, ratio=ghost_ratio, leak=leak)
        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.f2 = GhostBlock(base_channels, base_channels, ratio=ghost_ratio, leak=leak)
        self.p2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.f3 = GhostBlock(base_channels, base_channels, ratio=ghost_ratio, leak=leak)
        self.p3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.f4 = GhostBlock(base_channels, base_channels, ratio=ghost_ratio, leak=leak)
        self.p4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.f5 = GhostBlock(base_channels, base_channels, ratio=ghost_ratio, leak=leak)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        f1 = self.f1(x)
        x = self.p1(f1)
        f2 = self.f2(x)
        x = self.p2(f2)
        f3 = self.f3(x)
        x = self.p3(f3)
        f4 = self.f4(x)
        x = self.p4(f4)
        f5 = self.f5(x)
        return f1, f2, f3, f4, f5


class UpBlock(nn.Module):
    # Decoder upsampling block.
    def __init__(self, base_channels: int = 32, leak: float = 0.1) -> None:
        super().__init__()
        in_channels = base_channels * 2 
        self.dw = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.dw_bn = nn.BatchNorm2d(in_channels)
        self.dw_act = nn.LeakyReLU(leak, inplace=True)

        self.pw = nn.Conv2d(in_channels, base_channels, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(base_channels)
        self.pw_act = nn.LeakyReLU(leak, inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.dw(x)
        x = self.dw_bn(x)
        x = self.dw_act(x)
        x = self.pw(x)
        x = self.pw_bn(x)
        x = self.pw_act(x)
        return x


class LimFUNetFire(nn.Module):
    # Promptable LimFUNet-Fire student.
    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 1,
        base_channels: int = 32,
        ghost_ratio: int = 2,
        leak: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = LimFUNetEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            ghost_ratio=ghost_ratio,
            leak=leak,
        )
        self.up4 = UpBlock(base_channels=base_channels, leak=leak)
        self.up3 = UpBlock(base_channels=base_channels, leak=leak)
        self.up2 = UpBlock(base_channels=base_channels, leak=leak)
        self.up1 = UpBlock(base_channels=base_channels, leak=leak)
        self.head = nn.Conv2d(
            base_channels,
            num_classes,
            kernel_size=3,
            padding=1,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1, f2, f3, f4, f5 = self.encoder(x)
        o = f5
        o = self.up4(o, f4)
        o = self.up3(o, f3)
        o = self.up2(o, f2)
        o = self.up1(o, f1)
        out = self.head(o)
        return out
