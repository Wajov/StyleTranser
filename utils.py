import os
import random
from PIL import Image
import torch
from torch import nn
from torch.utils import data
from torchvision import transforms


class ImageDataset(data.Dataset):
    def __init__(self, path_X, path_Y):
        super(ImageDataset, self).__init__()
        files_X = os.listdir(path_X)
        files_Y = os.listdir(path_Y)
        self.images_X = sorted([os.path.join(path_X, image) for image in files_X])
        self.images_Y = sorted([os.path.join(path_Y, image) for image in files_Y])
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return min(len(self.images_X), len(self.images_Y))

    def __getitem__(self, index):
        item_X = self.transform(Image.open(self.images_X[index % len(self.images_X)]))
        item_Y = self.transform(Image.open(self.images_Y[index % len(self.images_Y)]))
        return {'X': item_X, 'Y': item_Y}


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant(m.bias.data, 0.0)


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_epoch):
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_epoch = decay_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_epoch) / (self.n_epochs - self.decay_epoch)


class ReplayBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        ans = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                ans.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    ans.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    ans.append(element)
        return torch.cat(ans)
