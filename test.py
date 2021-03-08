from PIL import Image
import torch
from torchvision import transforms

import models


if __name__ == '__main__':
    G = models.Generator(3, 3)
    G.load_state_dict(torch.load('output/G_X2Y.pth'))

    x = transforms.ToTensor()(Image.open('./picture/2.jpg')).unsqueeze(0)
    y = transforms.ToPILImage()(G(x).squeeze(0))
    y.show()
