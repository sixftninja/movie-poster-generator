from __future__ import print_function
import torchvision
from torchvision import transforms
from torchvision import datasets, models, transforms
from torch.utils.data.dataset import Dataset

class textImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        super(textImageDataset, self).__init__(root,transform)
        self.transform = transform
        self.imgs = self.samples

    def getImg(self, img_path):
        # get images

    def getText(self, text_path):
        # get text embeddings


    def __getitem__(self, index):
        path, target = self.samples[index]
