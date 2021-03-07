from torch import nn
from torch.nn import functional


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        layers = [
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3),
            nn.InstanceNorm2d(num_features=in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3),
            nn.InstanceNorm2d(num_features=in_features)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.model(x)


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=6):
        super(Generator, self).__init__()

        layers = [
            nn.ReflectionPad2d(padding=3),
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7),
            nn.InstanceNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        ]
        in_features = 64
        out_features = in_features * 2
        for i in range(2):
            layers += [
                nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(num_features=out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        for i in range(num_blocks):
            layers += [ResidualBlock(in_features)]

        out_features = in_features // 2
        for i in range(2):
            layers += [
                nn.ConvTranspose2d(in_channels=in_features, out_channels=out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(num_features=out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        layers += [
            nn.ReflectionPad2d(padding=3),
            nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()

        layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, padding=1),
            nn.InstanceNorm2d(num_features=512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, padding=1)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return functional.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
