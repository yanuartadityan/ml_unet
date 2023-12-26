import torch
import torch.nn as nn
import torchvision.transforms.functional as tf
from torchview import draw_graph


class DoubleConv(nn.Module):
    """Two layers of connected convolutional layers."""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """Implementation of all connected modules that resemble UNet."""

    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    in_channels=feature * 2,
                    out_channels=feature,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        copy_and_crop = []

        for down in self.downs:
            x = down(x)
            copy_and_crop.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        copy_and_crop = copy_and_crop[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            curr_crop = copy_and_crop[idx // 2]

            if x.shape != curr_crop.shape:
                tf.resize(x, size=curr_crop.shape[2:])

            cropped_node = torch.cat((curr_crop, x), dim=1)
            x = self.ups[idx + 1](cropped_node)

        x = self.final(x)

        return x


def test_shape():
    x = torch.randn((3, 1, 240, 160))
    model = UNet(in_channels=1, out_channels=1)
    pred = model(x)

    draw_model(model, x)

    print(x.shape)
    print(pred.shape)
    assert x.shape == pred.shape


def draw_model(model, input, device="meta"):
    import graphviz

    graphviz.set_jupyter_format("svg")
    model_graph = draw_graph(
        model=model,
        input_size=input.shape,
        device="cuda" if torch.cuda.is_available() else device,
        save_graph=True,
        filename="model",
        directory=".",
    )
    model_graph.visual_graph


if __name__ == "__main__":
    test_shape()
