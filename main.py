import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
import torchvision
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataset_dir = "./dataset/"
    
    dataset = datasets.ImageFolder(dataset_dir, transform=transforms.Compose([
        transforms.Resize((150, 150)), transforms.ToTensor()
        ]))
    
    print("Type of masks to be identified: ", dataset.classes)
    
    # Setting sizes:
    batch_size = 128
    test_size = 400
    train_size = len(dataset) - test_size
    
    # Splitting Data into Training and Testing sets:
    train_data, test_data = td.random_split(dataset, [train_size, test_size])
    
    print("Total number of images in dataset: ", len(dataset))
    print("Size of Train Data : ", len(train_data))
    print("Size of Test Data : ", len(test_data))
    
    # load the train and test into batches.
    train_loader = td.dataloader.DataLoader(train_data, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = td.dataloader.DataLoader(test_data, batch_size * 2, num_workers=4, pin_memory=True)
    