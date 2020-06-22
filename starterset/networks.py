import torch
import torch.nn as nn
from torchvision.models import resnet18 as resnet


class ClassicCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=4, batch_size=8):
        super(ClassicCNN, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.block1 = ClassicCNN.classic_cnn_block(input_channels, 32, 3, stride=2)
        self.block2 = ClassicCNN.classic_cnn_block(32, 64, 3, stride=2)
        self.maxpool1 = nn.MaxPool2d(2)
        self.block3 = ClassicCNN.classic_cnn_block(64, 128, 3)
        self.block4 = ClassicCNN.classic_cnn_block(128, 256, 3, stride=2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.block5 = ClassicCNN.classic_cnn_block(256, 128, 3)
        self.block6 = ClassicCNN.classic_cnn_block(128, 64, 3, stride=2)
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

    @staticmethod
    def classic_cnn_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


# add you own networks :)
class CustomResNet(torch.nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.pre = ClassicCNN.classic_cnn_block(1, 3)
        self.resnet = resnet(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, *inputs):
        assert len(inputs) == 1
        x = inputs[0]
        x = self.pre(x)
        x = self.resnet(x)
        return x


def test_and_show_network():
    from torchsummary import summary
    net = CustomResNet(num_classes=5).cuda()
    summary(net, input_size=(1, 241, 257))

if __name__ == "__main__":
    test_and_show_network()
