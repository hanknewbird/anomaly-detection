import torch.nn as nn
import torch.nn.functional as F
import torch


class NewNet(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv1 = nn.Conv2d(1, 64, (3, 3), stride=(1, 1), padding=1)
        # nn.BatchNorm2d(in_channels)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, (3, 3), stride=(2, 2), padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, (3, 3), stride=(2, 2), padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, (3, 3), stride=(2, 2), padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 64, (3, 3), stride=(2, 2), padding=1)
        self.bn5 = nn.BatchNorm2d(64)

        # nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.conv_tr1 = nn.ConvTranspose2d(
            64, 64, (3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.bn_tr1 = nn.BatchNorm2d(64)

        self.conv_tr2 = nn.ConvTranspose2d(
            128, 64, (3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.bn_tr2 = nn.BatchNorm2d(64)

        self.conv_tr3 = nn.ConvTranspose2d(
            128, 64, (3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.bn_tr3 = nn.BatchNorm2d(64)

        self.conv_tr4 = nn.ConvTranspose2d(
            128, 64, (3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.bn_tr4 = nn.BatchNorm2d(64)

        self.conv_output = nn.Conv2d(128, 1, (1, 1), (1, 1))
        self.bn_output = nn.BatchNorm2d(1)

    def forward(self, x):
        slope = 0.2

        # print(f"conv1(x)={self.conv1(x).shape}")

        x1 = F.leaky_relu((self.bn1(self.conv1(x))), slope)
        x2 = F.leaky_relu((self.bn2(self.conv2(x1))), slope)
        x3 = F.leaky_relu((self.bn3(self.conv3(x2))), slope)
        x4 = F.leaky_relu((self.bn4(self.conv4(x3))), slope)
        x5 = F.leaky_relu((self.bn5(self.conv5(x4))), slope)

        # print(f"x1={x1.shape}")
        # print(f"x2={x2.shape}")
        # print(f"x3={x3.shape}")
        # print(f"x4={x4.shape}")
        # print(f"x5={x5.shape}")

        x6 = F.leaky_relu(self.bn_tr1(self.conv_tr1(x5)), slope)
        x7 = F.leaky_relu(self.bn_tr2(
            self.conv_tr2(torch.cat([x6, x4], 1))), slope)
        x8 = F.leaky_relu(self.bn_tr3(
            self.conv_tr3(torch.cat([x7, x3], 1))), slope)
        x9 = F.leaky_relu(self.bn_tr4(
            self.conv_tr4(torch.cat([x8, x2], 1))), slope)

        output = F.leaky_relu(self.bn_output(
            self.conv_output(torch.cat([x9, x1], 1))), slope)

        # print(f"x6={x6.shape}")
        # print(f"x7={x7.shape}")
        # print(f"x8={x8.shape}")
        # print(f"x9={x9.shape}")
        # print(f"x10={output.shape}")

        return output


if __name__ == "__main__":
    x = torch.rand([2, 1, 512, 512])
    model = NewNet()
    y = model(x)
    print(x.shape, x.dtype)
    print(y.shape, y.dtype)
