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

class textImageDataset(torch.utils.data.Dataset):
    def __init__(self, root='/scratch/ans698/combined_dataset/poster', transform=None,text_path = 'scratch/sp5331/CV/doc2vecEmbeddings.p'):
        # super(textImageDataset, self).__init__(root,transform)
        self.transform = transform
        self.root = root
        self.imageFiles = [f for f in sorted(os.listdir(self.root))]
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

transformed_dataset = textImageDataset(root='/scratch/ans698/combined_dataset/poster',
                                           text_path = '/scratch/sp5331/CV/doc2vecEmbeddings.p',
                                           transform = data_transform
                                           )

dataloader = DataLoader(transformed_dataset, batch_size=64,
                        shuffle=True, num_workers=4)
print('dataloader')
for i, (img,text) in enumerate(dataloader):
    # print(img.shape)
    # print(len(text))
    print(img)
    print(text)
    # print(txt)

