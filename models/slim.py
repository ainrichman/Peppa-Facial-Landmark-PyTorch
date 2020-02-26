import torch
import torch.nn as nn
from thop import profile


def conv_bn(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def depth_conv2d(inp, oup, kernel=1, stride=1, pad=0):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel_size=kernel, stride=stride, padding=pad, groups=inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, kernel_size=1)
    )


def conv_dw(inp, oup, stride, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, padding, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


class Slim(nn.Module):
    def __init__(self):
        super(Slim, self).__init__()
        self.num_classes = 2

        self.conv1 = conv_bn(3, 16, 2)
        self.conv2 = conv_dw(16, 32, 1)
        self.conv3 = conv_dw(32, 32, 2)
        self.conv4 = conv_dw(32, 32, 1)
        self.conv5 = conv_dw(32, 64, 2)
        self.conv6 = conv_dw(64, 64, 1)
        self.conv7 = conv_dw(64, 64, 1)
        self.conv8 = conv_dw(64, 64, 1)

        self.conv9 = conv_dw(64, 128, 2)
        self.conv10 = conv_dw(128, 128, 1)
        self.conv11 = conv_dw(128, 128, 1)

        self.conv12 = conv_dw(128, 256, 2)
        self.conv13 = conv_dw(256, 256, 1)

        self.fc = nn.Linear(448, 143)

    def forward(self, inputs):
        x1 = self.conv1(inputs)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        output1 = x8
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        output2 = x11
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        output3 = x13
        output1 = output1.mean(3).mean(2)
        output2 = output2.mean(3).mean(2)
        output3 = output3.mean(3).mean(2)
        output = self.fc(torch.cat((output1, output2, output3), 1))
        return output


if __name__ == '__main__':
    model = Slim()
    model.eval()
    x = torch.randn(1, 3, 160, 160)
    flops, params = profile(model, inputs=(x,))
    print(flops)
