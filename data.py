from __future__ import print_function
import torchvision
from torchvision import transforms
from torchvision import datasets, models, transforms
# from torch.utils.data.dataset import Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader
import os
from skimage import io, transform
from PIL import Image
import pickle
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

class textImageDataset(torch.utils.data.Dataset):
    def __init__(self, root='/scratch/ans698/combined_dataset/poster',imageFiles=[], transform=None, text_path = 'scratch/sp5331/CV/doc2vecEmbeddings.p'):
        # super(textImageDataset, self).__init__(root,transform)
        self.transform = transform
        self.root = root
        self.imageFiles = imageFiles
        self.embeddings = pickle.load(open(text_path,'rb'))
        # print(len(self.embeddings))
        # print(self.embeddings[0])

    def __getitem__(self, index):
        img_name = self.root+'/'+ self.imageFiles[index]
        # image = io.imread(self.root+'/'+img_name)
        image = Image.open(img_name)
        image = image.convert('RGB')
        # print(img_name)



        embedding = torch.FloatTensor(self.embeddings[index])

        # print(len(embedding))

        if self.transform is not None:
            image = self.transform(image)

        # print(embedding.shape)
        # print(image.shape)

        return (image,embedding)

    def __len__(self):
        return len(self.imageFiles)

data_transform = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor()
        ])


imageFiles = [f for f in sorted(os.listdir('/scratch/ans698/combined_dataset/poster'))]

transformed_dataset = textImageDataset(root='/scratch/ans698/combined_dataset/poster',
    imageFiles = imageFiles,
    text_path = '/scratch/sp5331/CV/doc2vecEmbeddings.p',
    transform = data_transform
    )
# dataloader = DataLoader(transformed_dataset, batch_size=64,
                        # shuffle=True, num_workers=4)
dataset_size = len(imageFiles)
validation_split = .2
random_seed = 42
batch_size = 64
num_workers = 4
shuffle_dataset = True

print(dataset_size)

indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
print(train_indices)

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(transformed_dataset, batch_size=batch_size, sampler=train_sampler,num_workers = num_workers)
validation_loader = DataLoader(transformed_dataset, batch_size=batch_size, sampler=valid_sampler)

#print('dataloader')
for i, (img,text) in enumerate(train_loader):
    # print(img.shape)
    # print(len(text))
    print(img)
    print(text)
    print(text.shape())
    if i == 3:
        break
    # print(txt)

# def computeMeanSTD(train_loader):
#     mean = 0.
#     std = 0.
#     for images, _ in train_loader:
#         batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
#         images = images.view(batch_samples, images.size(1), -1)
#         mean += images.mean(2).sum(0)
#         std += images.std(2).sum(0)
#
#     mean /= len(train_loader.dataset)
#     std /= len(train_loader.dataset)
#     print(mean)
#     print(std)
#
# computeMeanSTD(train_loader)
