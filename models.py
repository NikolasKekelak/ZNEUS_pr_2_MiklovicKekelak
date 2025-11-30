import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


# ==========================================================
#  MODEL SELECTOR
# ==========================================================
def get_model(name: str, num_classes=10):
    if name == 'simple':
        return SimpleCNN(num_classes)
    if name == 'resnet':
        return RESNet(num_classes)
    if name == 'our-resnet':
        return Our_Res(num_classes)
    if name == 'small':
        return Smol(num_classes)
    if name == 'deep':
        return DeepDaddy(num_classes)
    if name == 'wide':
        return WideHips(num_classes)
    return None


# ==========================================================
#  BASIC BLOCK WITH OPTIONAL DOWNSAMPLING
# ==========================================================
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, downsample=False):
        super().__init__()

        stride = 2 if downsample else 1   # REPLACEMENT FOR MAXPOOL

        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),

            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


# ==========================================================
#  SMALL MODEL (ANY RESOLUTION)
# ==========================================================
class Smol(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.c1 = ConvBlock(3, 16, downsample=True)
        self.c2 = ConvBlock(16, 32, downsample=True)
        self.c3 = ConvBlock(32, 64, downsample=True)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(64, 256)
        self.drop = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


# ==========================================================
#  SIMPLE CNN (ANY RESOLUTION)
# ==========================================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.c1 = ConvBlock(3, 32, downsample=True)
        self.c2 = ConvBlock(32, 64, downsample=True)
        self.c3 = ConvBlock(64, 128, downsample=True)
        self.c4 = ConvBlock(128, 256, downsample=True)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, 512)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


# ==========================================================
#  DEEP MODEL (ANY RESOLUTION)
# ==========================================================
class DeepDaddy(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.c1 = ConvBlock(3, 32, downsample=True)
        self.c2 = ConvBlock(32, 64, downsample=True)
        self.c3 = ConvBlock(64, 128, downsample=True)
        self.c4 = ConvBlock(128, 256, downsample=True)
        self.c5 = ConvBlock(256, 256, downsample=True)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, 512)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


# ==========================================================
#  WIDE MODEL (ANY RESOLUTION)
# ==========================================================
class WideHips(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.c1 = ConvBlock(3, 64, downsample=True)
        self.c2 = ConvBlock(64, 128, downsample=True)
        self.c3 = ConvBlock(128, 256, downsample=True)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, 1024)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


# ==========================================================
#  PRETRAINED RESNET (ALREADY ANY RES)
# ==========================================================
class RESNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        # freeze base
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.model(x)


# ==========================================================
#  CUSTOM RESNET (ALREADY ANY RES)
# ==========================================================
class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.BatchNorm2d(c),
        )

    def forward(self, x):
        return F.relu(self.conv(x) + x)


class Our_Res(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.c1 = ConvBlock(3, 32, downsample=True)
        self.r1 = ResidualBlock(32)
        self.c2 = ConvBlock(32, 64, downsample=True)
        self.r2 = ResidualBlock(64)
        self.c3 = ConvBlock(64, 128, downsample=True)
        self.r3 = ResidualBlock(128)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.c1(x); x = self.r1(x)
        x = self.c2(x); x = self.r2(x)
        x = self.c3(x); x = self.r3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
