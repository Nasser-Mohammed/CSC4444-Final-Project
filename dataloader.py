import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import read_image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image

# img = read_image("./data/archive(1)/train/German Sheperd/001.jpg")
# plt.imshow(img.permute(1, 2, 0))
# plt.show()

def setLabels(labels):
    prev = -1
    labelDict = {}
    for index, row in labels.iterrows():
        if row['labels'] not in labelDict:
            labelDict[row['labels']] = prev + 1
            prev += 1
    num2LabelDict = {v: k for k, v in labelDict.items()}
    return labelDict, num2LabelDict

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, setType, batchSize = 32, transform=None, target_transform=None): 
        if setType not in ['test', 'train', 'valid']:
            print("invalid dataset type, must be: test, train, valid")
            exit(1)
        self.ttlData = pd.read_csv(annotations_file)    
        self.setType = setType
        self.img_labels = self.ttlData[self.ttlData['data set'] == setType]
        self.label2NumDict, self.num2LabelDict = setLabels(self.img_labels)
        self.img_dir = img_dir
        self.batchSize = batchSize
        self.transform = transforms.Compose([transforms.ToTensor(),
                transforms.RandomHorizontalFlip()
                # Add other transformations if needed
            ])
        self.target_transform = target_transform
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        name_label = self.img_labels.iloc[idx, 1]
        label = self.label2NumDict[name_label]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
    def _getLabelDict(self):
        return self.label2NumDict, self.num2LabelDict



def create_dataloader():
    training_data = CustomImageDataset("data/archive(1)/dogs.csv", "data/archive(1)/", 'train')
    test_data = CustomImageDataset("data/archive(1)/dogs.csv", "data/archive(1)/", 'test')
    validation_data = CustomImageDataset("data/archive(1)/dogs.csv", "data/archive(1)/", 'valid')

    train_dataloader = DataLoader(training_data, batch_size=training_data.batchSize, shuffle=True, num_workers = 0)
    test_dataloader = DataLoader(test_data, batch_size=test_data.batchSize, shuffle=True, num_workers = 0)
    validation_dataloader = DataLoader(validation_data, batch_size= validation_data.batchSize, shuffle = False, num_workers = 0)

    return train_dataloader, test_dataloader, validation_dataloader