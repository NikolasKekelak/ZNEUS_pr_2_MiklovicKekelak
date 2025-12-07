import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights



def get_model(name: str, num_classes=10, rnn_type="gru"):
    name = name.lower()

    if name == "small":
        return SmolRNN(num_classes=num_classes, rnn_type=rnn_type)
    if name == "jack":
        return JackRNN(num_classes=num_classes, rnn_type=rnn_type)
    if name == "deep":
        return DeepRNN(num_classes=num_classes, rnn_type=rnn_type)
    if name == "wide":
        return WideRNN(num_classes=num_classes, rnn_type=rnn_type)
    if name == "our-resnet":
        return OurResRNN(num_classes=num_classes, rnn_type=rnn_type)

    return None



class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
        )

    def forward(self, x):
        return F.relu(self.conv(x) + x, inplace=True)


class SpatialRNNHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        bidirectional: bool = True,
        dropout: float = 0.2,
        rnn_type: str = "gru",
    ):
        super().__init__()
        rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        if rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=in_channels,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
        elif rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=in_channels,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError("rnn_type must be 'gru' or 'lstm'")

        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)

    def forward(self, x):
        b, c, h, w = x.shape
        seq = x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        out, h_n = self.rnn(seq)

        if isinstance(h_n, tuple):  # LSTM
            h_n = h_n[0]

        # take last layer hidden states
        last_layer = h_n[-self.num_directions:]  # [num_dir, B, hidden]
        last_layer = last_layer.transpose(0, 1).contiguous().view(b, -1)  # [B, hidden*num_dir]

        last_layer = self.drop(last_layer)
        return self.fc(last_layer)


class SmolRNN(nn.Module):
    def __init__(self, num_classes=10, rnn_type="gru"):
        super().__init__()
        self.c1 = ConvBlock(3, 16, downsample=True)
        self.c2 = ConvBlock(16, 32, downsample=True)
        self.c3 = ConvBlock(32, 64, downsample=True)

        self.head = SpatialRNNHead(
            in_channels=64,
            hidden_size=128,
            num_layers=1,
            num_classes=num_classes,
            bidirectional=True,
            dropout=0.2,
            rnn_type=rnn_type,
        )

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        return self.head(x)


class JackRNN(nn.Module):
    def __init__(self, num_classes=10, rnn_type="gru"):
        super().__init__()
        self.c1 = ConvBlock(3, 32, downsample=True)
        self.c2 = ConvBlock(32, 64, downsample=True)
        self.c3 = ConvBlock(64, 128, downsample=True)
        self.c4 = ConvBlock(128, 256, downsample=True)

        self.head = SpatialRNNHead(
            in_channels=256,
            hidden_size=256,
            num_layers=1,
            num_classes=num_classes,
            bidirectional=True,
            dropout=0.3,
            rnn_type=rnn_type,
        )

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        return self.head(x)


class DeepRNN(nn.Module):
    def __init__(self, num_classes=10, rnn_type="gru"):
        super().__init__()
        self.c1 = ConvBlock(3, 32, downsample=True)
        self.c2 = ConvBlock(32, 64, downsample=True)
        self.c3 = ConvBlock(64, 128, downsample=True)
        self.c4 = ConvBlock(128, 256, downsample=True)
        self.c5 = ConvBlock(256, 256, downsample=True)

        self.head = SpatialRNNHead(
            in_channels=256,
            hidden_size=256,
            num_layers=2,          # deeper recurrence
            num_classes=num_classes,
            bidirectional=True,
            dropout=0.35,
            rnn_type=rnn_type,
        )

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        return self.head(x)


class WideRNN(nn.Module):
    def __init__(self, num_classes=10, rnn_type="gru"):
        super().__init__()
        self.c1 = ConvBlock(3, 64, downsample=True)
        self.c2 = ConvBlock(64, 128, downsample=True)
        self.c3 = ConvBlock(128, 256, downsample=True)

        self.head = SpatialRNNHead(
            in_channels=256,
            hidden_size=512,       # wider recurrence
            num_layers=1,
            num_classes=num_classes,
            bidirectional=True,
            dropout=0.3,
            rnn_type=rnn_type,
        )

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        return self.head(x)


class OurResRNN(nn.Module):
    def __init__(self, num_classes=10, rnn_type="gru"):
        super().__init__()
        self.c1 = ConvBlock(3, 32, downsample=True)
        self.r1 = ResidualBlock(32)
        self.c2 = ConvBlock(32, 64, downsample=True)
        self.r2 = ResidualBlock(64)
        self.c3 = ConvBlock(64, 128, downsample=True)
        self.r3 = ResidualBlock(128)

        self.head = SpatialRNNHead(
            in_channels=128,
            hidden_size=256,
            num_layers=1,
            num_classes=num_classes,
            bidirectional=True,
            dropout=0.25,
            rnn_type=rnn_type,
        )

    def forward(self, x):
        x = self.c1(x); x = self.r1(x)
        x = self.c2(x); x = self.r2(x)
        x = self.c3(x); x = self.r3(x)
        return self.head(x)



if __name__ == "__main__":
    x = torch.randn(4, 3, 128, 128)
    for name in ["small", "jack", "deep", "wide", "our-resnet", "resnet"]:
        m = get_model(name, num_classes=10, rnn_type="gru")
        y = m(x)
        print(name, y.shape)
