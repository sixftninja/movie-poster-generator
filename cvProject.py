from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
import torch.optim as optim
import numpy as np
import pandas as pd
from skimage import io, transform
from PIL import Image
import pickle
from torch.utils.data.sampler import SubsetRandomSampler
import csv
#from visdom import visdom
# from collections import namedtuple
# from torch.utils.data.dataset import Dataset, DataLoader

'''--------------------------------------------CONFIG--------------------------------------------'''

# args = dict(
#     batchSize   =   64,
#     imageSize   =   64,
#     imgDataRoot =   '/scratch/ama1128/cvproject/data/poster',
#     txtDataRoot =   '/scratch/ama1128/cvproject/data/doc2vecEmbeddings.p',
#     num_workers =   4,
#     checkpoint  =   '',
#     lr          =   1e-4,
#     seed        =   99,
#     nc          =   3,
#     nz          =   100,
#     ngf         =   64,
#     ndf         =   64,
#     nte         =   1024,
#     nt          =   256,
#     beta1       =   0.5,
#     ngpu        =   1,
#     epochs      =   100,
#     momentum    =   .5,
#     netG        =   '',
#     netD        =   '',
# )
# args = namedtuple('Args', args.keys())(**args)

parser = argparse.ArgumentParser(description='cvProject')
parser.add_argument('--inputSize', type=int, default=64, metavar='I',
                    help='number of workers (default: 4)')
parser.add_argument('--batchSize', type=int, default=128, metavar='B',
                    help='input batch size')
parser.add_argument('--imageSize', type=int, default=64,
                    help='the height / width of the input image to network')
parser.add_argument('--epochs', type=int, default=20, metavar='E',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--num_workers', type=int, default=4, metavar='W',
                    help='number of workers (default: 4)')
parser.add_argument('--seed', type=int, default=99, metavar='S', help='random seed (default: 1)')
parser.add_argument('--nc', type=int, default=3, help='number of channels in training images')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64, help='number of features maps in Generator')
parser.add_argument('--ndf', type=int, default=64, help='number of features maps in Discriminator')
parser.add_argument('--nte', type=int, default=512, help='the size of text embedding vector')
parser.add_argument('--nt', type=int, default=256, help='the reduced size of text embedding vector')
parser.add_argument('--lrAdam', type=float, default=0.0002, help='learning rate for Adam(G) default=0.0002')
parser.add_argument('--lrSGD', type=float, default=0.0001, help='learning rate for SGD(D) default=0.0001')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum, default=0.9')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")

args = parser.parse_args()
torch.manual_seed(args.seed)

imgDataRoot             =   '/scratch/ama1128/cvproject/data/poster'
txtDataRoot             =   '/scratch/ama1128/cvproject/data/doc2vecEmbeddings.p'
modelPath               =   '/scratch/ama1128/cvproject/models'
generatedImagesPath     =   '/scratch/ama1128/cvproject/generated_images'

global gpu, device

if torch.cuda.is_available():
    gpu = True
    args.ngpu = torch.cuda.device_count()
    device = 'cuda:0'
    print("Using GPU")
    print("Number of GPUs: ", args.ngpu)
else:
    gpu = False
    device = 'cpu'
    print("Using CPU")

FloatTensor = torch.cuda.FloatTensor if gpu else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if gpu else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if gpu else torch.ByteTensor
Tensor = FloatTensor

'''------------------------------------------DATA------------------------------------------------'''


class textImageDataset(torch.utils.data.Dataset):
    def __init__(self, root=imgDataRoot,imageFiles=[], transform=None, text_path = txtDataRoot):
        # super(textImageDataset, self).__init__(root,transform)
        self.transform = transform
        self.root = root
        self.imageFiles = imageFiles
        self.embeddings = pickle.load(open(text_path,'rb'))

    def __getitem__(self, index):
        img_name = self.root+'/'+ self.imageFiles[index]
        image = Image.open(img_name)
        image = image.convert('RGB')
        embedding = torch.FloatTensor(self.embeddings[index])
        if self.transform is not None:
            image = self.transform(image)

        return (image,embedding)

    def __len__(self):
        return len(self.imageFiles)

data_transform = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor(),
            transforms.Normalize([0.4216, 0.3679, 0.3287], [0.2104, 0.1997, 0.1919])
        ])

imageFiles = [f for f in sorted(os.listdir(imgDataRoot))]

transformed_dataset = textImageDataset(root=imgDataRoot,imageFiles = imageFiles,
                                       text_path = txtDataRoot,
                                       transform = data_transform
                                      )

# dataset_size = len(imageFiles)
# validation_split = .2
# random_seed = 42
# shuffle_dataset = True

# indices = list(range(dataset_size))
# split = int(np.floor(validation_split * dataset_size))
# if shuffle_dataset :
#     np.random.seed(random_seed)
#     np.random.shuffle(indices)
# train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
# train_sampler = SubsetRandomSampler(train_indices)
# valid_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(transformed_dataset, batch_size=args.batchSize,
                          num_workers = args.num_workers, shuffle=True, drop_last=True
                         )
val_loader = DataLoader(transformed_dataset, batch_size=args.batchSize)

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
ngpu = int(args.ngpu)
nz   = int(args.nz)
ngf  = int(args.ngf)
ndf  = int(args.ndf)
nc   = int(args.nc)
nte  = int(args.nte)
nt   = int(args.nt)

from model import Generator, Discriminator

netG = Generator(ngpu, nz, ngf, nc, nte, nt).to(device)
netG.apply(weights_init)
if args.netG != '':
    netG.load_state_dict(torch.load(args.netG))
# print(netG)

netD = Discriminator(ngpu, nc, ndf, nte, nt).to(device)
netD.apply(weights_init)
if args.netD != '':
    netD.load_state_dict(torch.load(args.netD))
# print(netD)
# if ngpu>1:
#     netD = nn.DataParallel(netD)
#     netG = nn.DataParallel(netG)

'''----------------------------------------LOSS & OPTIMIZER--------------------------------------'''

criterion = nn.BCELoss()

input = torch.FloatTensor(args.batchSize, 3, args.imageSize, args.imageSize).to(device)
fixed_noise = torch.FloatTensor(args.batchSize, nz, 1, 1).normal_(0, 1).to(device)
label = torch.FloatTensor(args.batchSize).to(device)
real_label = random.uniform(0.8, 1.1)
fake_label = random.uniform(0.0, 0.3)

#optimizerD = optim.SGD(netD.parameters(), lr=args.lrSGD, momentum=args.momentum)
optimizerD = optim.Adam(netD.parameters(), lr=args.lrAdam, betas=(args.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=args.lrAdam, betas=(args.beta1, 0.999))

'''--------------------------------------------Visdom--------------------------------------------'''

# class VisdomLinePlotter(object):
#     """Plots to Visdom"""
#     def __init__(self, env_name='main'):
#         self.viz = Visdom()
#         self.env = env_name
#         self.plots = {}
#     def plot(self, var_name, split_name, title_name, x, y):
#         if var_name not in self.plots:
#             self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
#                 legend=[split_name],
#                 title=title_name,
#                 xlabel='Epochs',
#                 ylabel=var_name
#             ))
#         else:
#             self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

# plotter = VisdomLinePlotter()

'''-----------------------------------------AVERAGEMETER-----------------------------------------'''

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

'''-------------------------------------------GRADIENTS------------------------------------------'''
G_Losses            = []
D_Losses            = []
G_Grads             = []
D_Grads_real        = []
D_Grads_fake        = []
D_Grads_mismatch    = []
D_x_vals            = []
D_G_z1_vals         = []
D_G_z2_vals         = []

def store_gradients(model_name, epoch, batch_idx, named_parameters):
    if model_name == 'netG':
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                G_Grads.append([epoch, batch_idx, n, p.grad.abs().mean()])
    if model_name == 'netD_real':
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                D_Grads_real.append([epoch, batch_idx, n, p.grad.abs().mean()])
    if model_name == 'netD_fake':
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                D_Grads_fake.append([epoch, batch_idx, n, p.grad.abs().mean()])
    if model_name == 'netD_mismatch':
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                D_Grads_mismatch.append([epoch, batch_idx, n, p.grad.abs().mean()])

'''---------------------------------------------TRAIN--------------------------------------------'''

for epoch in range(args.epochs):
    for batch_idx, (data,txt) in enumerate(train_loader):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # TRAIN WITH REAL
        ###########################
        netD.zero_grad()
        # real_cpu is batch, data[0] returns all image tensors
        # real_images, text_embedding, _ = data
        real_images = data
        text_embedding = txt
        # real_cpu.size(0) returns number of images in batch
        batch_size = real_images.size(0)
        # creates tensor label of size batch_size and fills it with value of real_label=1)
        # label = torch.full((batch_size,), real_label, device=device)
        label = torch.full((batch_size,), random.uniform(0.8, 1.1), device=device)
        if gpu:
            real_images = real_images.to(device)
            text_embedding = text_embedding.to(device)

        # print('data cuda: ', real_images.is_cuda)
        # print('text cuda: ', text_embedding.is_cuda)
        # print('label cude: ', label.is_cuda)
        # print('model: ', next(netD.parameters()).is_cuda)

        output_real = netD(real_images, text_embedding)
        errD_real = criterion(output_real, label)
        errD_real.backward()
        store_gradients('netD_real', epoch, batch_idx, netD.named_parameters())
        # Probability for each image of being real given by D averaged over all images in batch
        D_x = output_real.mean().item()

        ######################
        # TRAIN WITH MISMATCH
        ######################
        text_embedding_wrong = torch.randn(args.batchSize, args.nte).normal_(0, 1).to(device)
        # label.fill_(fake_label)
        label.fill_(random.uniform(0.0, 0.3))
        output_mismatch = netD(real_images, text_embedding_wrong)
        errD_mismatch = criterion(output_mismatch, label) * 0.5
        errD_mismatch.backward()
        store_gradients('netD_mismatch', epoch, batch_idx, netD.named_parameters())

        ######################
        # TRAIN WITH FAKE
        ######################

        noise = torch.FloatTensor(args.batchSize, nz, 1, 1).normal_(0, 1).to(device)
        fake_images = netG(noise, text_embedding)
        # use fake labels
        # label.fill_(fake_label)
        label.fill_(random.uniform(0.0, 0.3))
        # detach clears gradients that fake images have accumulated while being passed through the Generator
        output = netD(fake_images.detach(), text_embedding.detach())
        errD_fake = criterion(output, label) * 0.5
        errD_fake.backward()
        store_gradients('netD_fake', epoch, batch_idx, netD.named_parameters())
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake + errD_mismatch

        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        # label.fill_(real_label)  # fake labels are real for generator cost
        label.fill_(random.uniform(0.8, 1.1))
        output = netD(fake_images, text_embedding)
        errG = criterion(output, label)
        errG.backward()
        store_gradients('netG', epoch, batch_idx, netG.named_parameters())
        D_G_z2 = output.mean().item()
        optimizerG.step()

        if batch_idx % 10 == 0:
            print('Epoch: [{}/{}] [{}/{}] Loss_D: {:.4f} Loss_G: {:.4f} D(x): {:.4f} D(G(z)): {:.4f}'.format(epoch, args.epochs, batch_idx*len(data), len(train_loader.dataset), errD.item(), errG.item(), D_x, (D_G_z1/D_G_z2)))

            D_x_vals.append([epoch, batch_idx, D_x])
            D_G_z1_vals.append([epoch, batch_idx, D_G_z1])
            D_G_z2_vals.append([epoch, batch_idx, D_G_z2])
            D_Losses.append([epoch, batch_idx, errD.item()])
            G_Losses.append([epoch, batch_idx, errG.item()])

        if batch_idx % 100 == 0:
            vutils.save_image(real_images, '%s/real_samples.png' % generatedImagesPath,
                              normalize=True)
            fake = netG(fixed_noise, text_embedding)
            vutils.save_image(fake.detach(),'%s/fake_samples_epoch_%03d.png' %
                              (generatedImagesPath, epoch), normalize=True)

    print('-------------------------------------------------------------------------------------')
    # VISUALIZE
    #plotter.plot('Loss_D', 'train', 'Discriminator Loss', epoch, )

    # SAVE CHECKPOINTS
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (modelPath, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (modelPath, epoch))

'''--------------------------------------------CSV_OUT-------------------------------------------'''

D_grads_real_df        = pd.DataFrame(D_Grads_real)
D_grads_real_df.to_csv('D_grads_real.csv')

D_grads_fake_df        = pd.DataFrame(D_Grads_fake)
D_grads_fake_df.to_csv('D_grads_fake.csv')

D_grads_mismatch_df    = pd.DataFrame(D_Grads_mismatch)
D_grads_mismatch_df.to_csv('D_grads_mismatch.csv')

D_x_df                 = pd.DataFrame(D_x_vals)
D_x_df.to_csv('D_x.csv')

D_G_z1_df              = pd.DataFrame(D_G_z1_vals)
D_G_z1_df.to_csv('D_G_z1.csv')

D_G_z2_df              = pd.DataFrame(D_G_z2_vals)
D_G_z2_df.to_csv('D_G_z2.csv')

D_Losses_df               = pd.DataFrame(D_Losses)
D_Losses_df.to_csv('D_Losses.csv')

G_Losses_df               = pd.DataFrame(G_Losses)
G_Losses_df.to_csv('G_Losses.csv')
