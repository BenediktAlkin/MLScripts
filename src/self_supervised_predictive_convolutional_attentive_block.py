# from https://arxiv.org/abs/2111.09099
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image


class SSPCAB(nn.Module):
    def __init__(self, channels, kernel_dim=1, dilation=1):
        super(SSPCAB, self).__init__()
        self.pad = kernel_dim + dilation
        self.border_input = kernel_dim + 2 * dilation + 1

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_dim)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_dim)
        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_dim)
        self.conv4 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_dim)

    def forward(self, x):
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), "constant", 0)

        conv1in = x[:, :, :-self.border_input, :-self.border_input]
        conv2in = x[:, :, self.border_input:, :-self.border_input]
        conv3in = x[:, :, :-self.border_input, self.border_input:]
        conv4in = x[:, :, self.border_input:, self.border_input:]
        save_image(conv1in, "img_conv1.png")
        save_image(conv2in, "img_conv2.png")
        save_image(conv3in, "img_conv3.png")
        save_image(conv4in, "img_conv4.png")
        x1 = self.conv1(conv1in)
        x2 = self.conv2(conv2in)
        x3 = self.conv3(conv3in)
        x4 = self.conv4(conv4in)
        x = self.relu(x1 + x2 + x3 + x4)
        return x


def main():
    layer = SSPCAB(channels=1, kernel_dim=1, dilation=1)
    x = torch.linspace(0.3, 0.7, 4).unsqueeze(0)
    x = (x.T @ x).unsqueeze(0).unsqueeze(0)
    save_image(x, "img_og.png")
    y = layer(x)


if __name__ == "__main__":
    main()
