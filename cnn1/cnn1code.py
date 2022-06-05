#Preparing the Dataset :

# To prepare a dataset from such a structure, PyTorch provides ImageFolder class which makes the task easy for us
# to prepare the dataset.
# We simply have to pass the directory of our data to it and it provides the dataset which we can use to train the model.

import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split


# # Visualizing the images:
# # To visualize images of a single batch, make_grid() can be used from torchvision utilities.
# # It gives us an overall view of images in batch in the form of an image grid.
#
#
#
#
# def show_batch(dl):
#     """Plot images grid of single batch"""
#     for images, labels in dl:
#         fig, ax = plt.subplots(figsize=(16, 12))
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
#         break
#
#
# show_batch(train_dl)

# Base Model For Image Classification:

# First, we prepare a base class that extends the functionality of torch.nn.Module (base class used to develop all neural networks).
# We add various functionalities to the base to train the model, validate the model, and get the result for each epoch.
# This is reusable and can be used for any image classification model, no need to rewrite this every time.

class ImageClassificationBase(nn.Module):

    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

# CNN Model For Classification:

class NaturalSceneClassification(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(82944, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 6)
        )

    def forward(self, xb):
        return self.network(xb)

# Hyperparameters, Model Training, And Evaluation:
# Now we have to train the natural scene classification model on the training dataset.
# So that first defines the fit, evaluation, and accuracy methods.

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):

        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)

    return history


# Plotting the graph for accuracies and losses to visualize how the model improves its accuracy after each epoch:
def plot_accuracies(history):
    """ Plot the history of accuracies"""
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');


def plot_losses(history):
    """ Plot the losses in each epoch"""
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs');




if __name__ == "__main__":
    # train and test data directory
    data_dir = "./seg_train/seg_train"
    test_data_dir = "./seg_test/seg_test"

    # load the train and test data
    dataset = ImageFolder(data_dir, transform=transforms.Compose([
        transforms.Resize((150, 150)), transforms.ToTensor()
    ]))
    test_dataset = ImageFolder(test_data_dir, transforms.Compose([
        transforms.Resize((150, 150)), transforms.ToTensor()
    ]))

    # The torchvision.transforms module provides various functionality to preprocess the images,
    # here first we resize the image for (150*150) shape and then transforms them into tensors.

    # The image label set according to the class index in data.classes.

    print("Follwing classes are there : \n", dataset.classes)

    # output:
    # Follwing classes are there :
    # ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

    # Splitting Data and Prepare Batches:
    batch_size = 128
    val_size = 2000
    train_size = len(dataset) - val_size

    train_data, val_data = random_split(dataset, [train_size, val_size])
    print(f"Length of Train Data : {len(train_data)}")
    print(f"Length of Validation Data : {len(val_data)}")

    # output
    # Length of Train Data : 12034
    # Length of Validation Data : 2000

    # load the train and validation into batches.
    train_dl = DataLoader(train_data, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_data, batch_size * 2, num_workers=4, pin_memory=True)

    model=NaturalSceneClassification()
    num_epochs = 30
    opt_func = torch.optim.Adam
    lr = 0.001
    # fitting the model on training data and record the result after each epoch
    history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)

    # plot_accuracies(history)
    # plot_losses(history)