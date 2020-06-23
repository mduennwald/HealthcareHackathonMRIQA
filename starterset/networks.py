import torch
import torch.nn as nn


class ClassicCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=4, batch_size=8):
        super(ClassicCNN, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.block1 = self.classic_cnn_block(input_channels, 32, 3, stride=2)
        self.block2 = self.classic_cnn_block(32, 64, 3, stride=2)
        self.maxpool1 = nn.MaxPool2d(2)
        self.block3 = self.classic_cnn_block(64, 128, 3)
        self.block4 = self.classic_cnn_block(128, 256, 3, stride=2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.block5 = self.classic_cnn_block(256, 128, 3)
        self.block6 = self.classic_cnn_block(128, 64, 3, stride=2)
        self.block7 = nn.Conv2d(64, self.num_classes, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.maxpool1(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.maxpool2(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = torch.mean(x, dim=(2, 3))
        return x

    def classic_cnn_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


# add you own networks :)