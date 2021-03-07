from PIL import Image
import torch
from torchvision import transforms

import models


if __name__ == "__main__":
    G = models.Generator(3, 3)
    G.load_state_dict(torch.load("output/G_X2Y.pth"))

    transform = transforms.Compose([transforms.ToTensor()])
    x = transform(Image.open("./picture/1.jpg")).unsqueeze(0)
    y = G(x)
    y = y.squeeze(0)
    y = transforms.ToPILImage()(y)
    y.show()
