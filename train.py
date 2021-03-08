import time
import itertools
import torch
from torch.utils import data
from torch import nn
from torch import optim

import utils
import models

LR = 0.0002
BATCH_SIZE = 10
EPOCH = 0
N_EPOCH = 200
DECAY_EPOCH = 100


if __name__ == '__main__':
    dataset = utils.ImageDataset('picture/', 'painting')
    dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    G_X2Y = models.Generator(3, 3).cuda()
    G_Y2X = models.Generator(3, 3).cuda()
    D_X = models.Discriminator(3).cuda()
    D_Y = models.Discriminator(3).cuda()

    G_X2Y.apply(utils.weights_init_normal)
    G_Y2X.apply(utils.weights_init_normal)
    D_X.apply(utils.weights_init_normal)
    D_Y.apply(utils.weights_init_normal)

    optimizer_G = optim.Adam(itertools.chain(G_X2Y.parameters(), G_Y2X.parameters()), lr=LR, betas=(0.5, 0.999))
    optimizer_D_X = optim.Adam(D_X.parameters(), lr=LR, betas=(0.5, 0.999))
    optimizer_D_Y = optim.Adam(D_Y.parameters(), lr=LR, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=utils.LambdaLR(N_EPOCH, EPOCH, DECAY_EPOCH).step)
    lr_scheduler_D_X = torch.optim.lr_scheduler.LambdaLR(optimizer_D_X, lr_lambda=utils.LambdaLR(N_EPOCH, EPOCH, DECAY_EPOCH).step)
    lr_scheduler_D_Y = torch.optim.lr_scheduler.LambdaLR(optimizer_D_Y, lr_lambda=utils.LambdaLR(N_EPOCH, EPOCH, DECAY_EPOCH).step)

    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    target_real = torch.ones([BATCH_SIZE, 1]).cuda()
    target_fake = torch.zeros([BATCH_SIZE, 1]).cuda()

    fake_X_buffer = utils.ReplayBuffer()
    fake_Y_buffer = utils.ReplayBuffer()

    losses_G = []
    losses_D_X = []
    losses_D_Y = []

    for epoch in range(N_EPOCH):
        for batch in dataloader:
            t = time.time()

            real_X = batch['X'].cuda()
            real_Y = batch['Y'].cuda()

            # ======== Optimize G ========
            optimizer_G.zero_grad()

            # Identity loss
            same_Y = G_X2Y(real_Y)
            loss_identity_Y = criterion_identity(same_Y, real_Y) * 5.0
            same_X = G_Y2X(real_X)
            loss_identity_X = criterion_identity(same_X, real_X) * 5.0

            # GAN loss
            fake_Y = G_X2Y(real_X)
            pred_fake = D_Y(fake_Y)
            loss_GAN_Y2X = criterion_GAN(pred_fake, target_real)
            fake_X = G_Y2X(real_Y)
            pred_fake = D_X(fake_X)
            loss_GAN_X2Y = criterion_GAN(pred_fake, target_real)

            # Cycle loss
            recovered_X = G_Y2X(fake_Y)
            loss_cycle_XYX = criterion_cycle(recovered_X, real_X) * 10.0
            recovered_Y = G_X2Y(fake_X)
            loss_cycle_YXY = criterion_cycle(recovered_Y, real_Y) * 10.0

            # Total loss
            loss_G = loss_identity_X + loss_identity_Y + loss_GAN_X2Y + loss_GAN_Y2X + loss_cycle_XYX + loss_cycle_YXY
            loss_G.backward()

            optimizer_G.step()
            # ========================

            # ======== Optimize D_X ========
            optimizer_D_X.zero_grad()

            # Real loss
            pred_real = D_X(real_X)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_X = fake_X_buffer.push_and_pop(fake_X)
            pred_fake = D_X(fake_X)
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_X = (loss_D_real + loss_D_fake) * 0.5
            loss_D_X.backward()

            optimizer_D_X.step()
            # ========================

            # ======== Optimize D_Y ========
            optimizer_D_Y.zero_grad()

            # Real loss
            pred_real = D_Y(real_Y)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_Y = fake_Y_buffer.push_and_pop(fake_Y)
            pred_fake = D_Y(fake_Y)
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_Y = (loss_D_real + loss_D_fake) * 0.5
            loss_D_Y.backward()

            optimizer_D_Y.step()
            # ========================

            print(time.time() - t)
            print('{} {} {}'.format(loss_G.item(), loss_D_X.item(), loss_D_Y.item()))

            losses_G.append(loss_G.item())
            losses_D_X.append(loss_D_X.item())
            losses_D_Y.append(loss_D_Y.item())

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_X.step()
        lr_scheduler_D_Y.step()

        # Save models
        torch.save(G_X2Y.state_dict(), 'output/G_X2Y.pth')
        torch.save(G_Y2X.state_dict(), 'output/G_Y2X.pth')
        torch.save(D_X.state_dict(), 'output/D_X.pth')
        torch.save(D_Y.state_dict(), 'output/D_Y.pth')

        print('\n\nEpoch {} finished\n\n'.format(epoch))

    file = open('output/loss_G.txt', 'w')
    for loss in losses_G:
        file.write(str(loss) + '\n')

    file.close()
    file = open('output/loss_D_X.txt', 'w')
    for loss in losses_D_X:
        file.write(str(loss) + '\n')
    file.close()

    file = open('output/loss_D_Y.txt', 'w')
    for loss in losses_D_X:
        file.write(str(loss) + '\n')
    file.close()
