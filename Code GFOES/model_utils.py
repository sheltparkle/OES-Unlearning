import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class AllCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(AllCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 96, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0)
        self.conv9 = nn.Conv2d(192, num_classes, kernel_size=1, stride=1, padding=0)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x


class ResNet50(nn.Module):
    def __init__(self, num_classes=100, pretrained=True):
        super().__init__()
        if pretrained:
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)  # Use pretrained model
        else:
            self.model = models.resnet50(weights=None)

        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def features(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        return x


class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()

        self.init_size = 8  # Initial image size
        self.nz = nz
        self.ngf = ngf
        self.nc = nc

        self.l1 = nn.Sequential(
            nn.Linear(self.nz, self.ngf * 8 * self.init_size ** 2),
            nn.Dropout(0.2)
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(self.ngf * 8),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 8, self.ngf * 4, 3, stride=1, padding=1),
            nn.Dropout2d(0.2),  # Add Dropout2d after conv layers with 0.2 probability
            nn.BatchNorm2d(self.ngf * 4, 0.8),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.ngf * 4, self.ngf * 2, 3, stride=1, padding=1),
            nn.Dropout2d(0.2),  # Add Dropout2d after conv layers with 0.2 probability
            nn.BatchNorm2d(self.ngf * 2, 0.8),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.ngf * 2, self.ngf, 3, stride=1, padding=1),
            nn.Dropout2d(0.2),  # Add Dropout2d after conv layers with 0.2 probability
            nn.BatchNorm2d(self.ngf, 0.8),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.ngf, self.nc, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], self.ngf * 8, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
