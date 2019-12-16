from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.optim as optim
import numpy as np

'''--------------------------------------------CONFIG--------------------------------------------'''


parser = argparse.ArgumentParser(description='cvProject')
parser.add_argument('--dataroot', type=str, default='data/celeba', help='path to dataset')
parser.add_argument('--inputSize', type=int, default=64, metavar='I',
                    help='number of workers (default: 4)')
parser.add_argument('--batchSize', type=int, default=128, metavar='B',
                    help='input batch size for training LSTM Cell (default: 1)')
parser.add_argument('--imageSize', type=int, default=64,
                    help='the height / width of the input image to network')
parser.add_argument('--epochs', type=int, default=20, metavar='E',
                    help='number of epochs to train (default: 30)')
parser.add_argument('--numWorkers', type=int, default=4, metavar='W',
                    help='number of workers (default: 4)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--nc', type=int, default=3, help='number of channels in training images')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64, help='number of features maps in Generator')
parser.add_argument('--ndf', type=int, default=64, help='number of features maps in Discriminator')
parser.add_argument('--nte', type=int, default=1024, help='the size of text embedding vector')
parser.add_argument('--nt', type=int, default=256, help='the reduced size of text embedding vector')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")

args = parser.parse_args()
if args.seed is None:
    args.seed = random.randint(1, 10000)
print("Random Seed: ", args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

global gpu, device

if torch.cuda.is_available():
    gpu = True
    device = 'cuda:0'
    print("Using GPU")
else:
    gpu = False
    device = 'cpu'
    print("Using CPU")

FloatTensor = torch.cuda.FloatTensor if gpu else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if gpu else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if gpu else torch.ByteTensor
Tensor = FloatTensor

'''----------------------------------------CUSTOM WEIGHTS----------------------------------------'''
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


'''---------------------------------------INSTANTIATE MODELS-------------------------------------'''
from model import Generator, Discriminator

netG = Generator(ngpu).to(device)
netG.apply(weights_init)
if args.netG != '':
    netG.load_state_dict(torch.load(args.netG))
# print(netG)

netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
if args.netD != '':
    netD.load_state_dict(torch.load(args.netD))
# print(netD)
if gpu:
    netD.to(device)
    netG.to(device)

'''----------------------------------------LOSS & OPTIMIZER--------------------------------------'''

criterion = nn.BCELoss()

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize).to(device)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1, device=device)
label = torch.FloatTensor(opt.batchSize).to(device)
real_label = 1
fake_label = 0

optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

'''---------------------------------------------DATA---------------------------------------------'''

# dataset = dataset.ImageFolder(root=args.dataroot, transform=transforms.Compose([
#                                transforms.Resize(args.image_size),
#                                transforms.CenterCrop(args.image_size),
#                                transforms.ToTensor(),
#                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                            ]))
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
#                                          shuffle=True, num_workers=int(args.numWorkers), pin_memory=True)
train_data
test_data
train_loader
test_loader

ngpu = int(args.ngpu)
nz = int(args.nz)
ngf = int(args.ngf)
ndf = int(args.ndf)
nte = int(args.nte)
nt = int(args.nt)
nc = 3


'''---------------------------------------------TRAIN--------------------------------------------'''

for epoch in range(args.epochs):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # TRAIN WITH REAL
        ###########################
        netD.zero_grad()
        # real_cpu is batch, data[0] returns all image tensors
        real_images, text_embedding, _ = data
        # real_cpu.size(0) returns number of images in batch
        batch_size = real_images.size(0)
        # creates tensor label of size batch_size and fills it with value of real_label=1)
        label = torch.full((batch_size,), real_label, device=device)
        if gpu:
            real_label.to(device)
            text_embedding.to(device)

        output_real = netD(real_images, text_embedding)
        errD_real = criterion(output_real, label)
        errD_real.backward()
        # Probability for each image of being real given by D averaged over all images in batch
        D_x = output_real.mean().item()

        ######################
        # TRAIN WITH MISMATCH
        ######################
        text_embedding_wrong = torch.randn(args.batchSize, nte, 1, 1).normal_(0, 1, device=device)
        label.fill_(fake_label)
        output_mismatch = netD(real_images, text_embedding_wrong)
        errD_mismatch = criterion(output_mismatch, label) * 0.5
        errD_mismatch.backward()

        ######################
        # TRAIN WITH FAKE
        ######################
        # noise = torch.randn(batch_size, nz, 1, 1, device=device)
        noise = torch.FloatTensor(args.batchSize, nz, 1, 1, device=device).normal_(0, 1, device=device)
        fake_images = netG(noise)
        # use fake labels, fake_label=0
        label.fill_(fake_label)
        # detach clears gradients because 'fake' has accumulated while being passed through the Generator
        output = netD(fake_images.detach(), text_embedding.detach())
        errD_fake = criterion(output, label) * 0.5
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake + errD_mismatch
        argsimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake_images)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        argsimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, args.niter, i, len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % args.outf,
                    normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.detach(),
                    '%s/fake_samples_epoch_%03d.png' % (args.outf, epoch),
                    normalize=True)

    # SAVE CHECKPOINTS
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (args.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (args.outf, epoch))
