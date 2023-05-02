from models import resnet18
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import os

"""
This is the module you will submit to the autograder.

There are several function and variable definitions, here, that raise RuntimeErrors.
You should replace each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

If you are not sure how to use PyTorch, you may want to take a look at the tutorial.
"""


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


"""
1.  Define and build a PyTorch Dataset
"""


class CIFAR10(Dataset):
    def __init__(self, data_files, transform=None, target_transform=None):
        """
        Initialize your dataset here. Note that transform and target_transform
        correspond to your data transformations for train and test respectively.
        """

        self.data = []
        self.labels = []
        self.transforms = transform

        for file in data_files:
            data_file = unpickle(file)
            self.data.extend(data_file[b'data'])
            self.labels.extend(data_file[b'labels'])

        self.data = np.array(self.data).reshape(-1, 3,
                                                32, 32).transpose(0, 2, 3, 1)

    def __len__(self):
        """
        Return the length of your dataset here.
        """

        return len(self.data)

    def __getitem__(self, idx):
        """
        Obtain a sample from your dataset.

        Parameters:
            x:      an integer, used to index into your data.

        Outputs:
            y:      a tuple (image, label), although this is arbitrary so you can use whatever you would like.
        """

        used_label = self.labels[idx]
        used_data = self.data[idx]

        if self.transforms is not None:
            used_data = self.transforms(used_data)
        return (used_data, used_label)


def get_preprocess_transform(mode):
    """
    Parameters:
        mode:           "train" or "test" mode to obtain the corresponding transform
    Outputs:
        transform:      a torchvision transforms object e.g. transforms.Compose([...]) etc.
    """

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    ])


def build_dataset(data_files, transform=None):
    """
    Parameters:
        data_files:      a list of strings e.g. "cifar10_batches/data_batch_1" corresponding to the CIFAR10 files to load data
        transform:       the preprocessing transform to be used when loading a dataset sample
    Outputs:
        dataset:      a PyTorch dataset object to be used in training/testing
    """

    return CIFAR10(data_files=data_files, transform=transform)


"""
2.  Build a PyTorch DataLoader
"""


def build_dataloader(dataset, loader_params):
    """
    Parameters:
        dataset:         a PyTorch dataset to load data
        loader_params:   a dict containing all the parameters for the loader.

    Please ensure that loader_params contains the keys "batch_size" and "shuffle" corresponding to those
    respective parameters in the PyTorch DataLoader class.

    Outputs:
        dataloader:      a PyTorch dataloader object to be used in training/testing
    """

    return DataLoader(dataset, batch_size=loader_params["batch_size"], shuffle=loader_params["shuffle"])


"""
3. (a) Build a neural network class.
"""


class FinetuneNet(torch.nn.Module):
    def __init__(self):
        """
        Initialize your neural network here. Remember that you will be performing finetuning
        in this network so follow these steps:

        1. Initialize convolutional backbone with pretrained model parameters.
        2. Freeze convolutional backbone.
        3. Initialize linear layer(s).
        """
        super().__init__()
        ################# Your Code Starts Here #################
        resnet18Test = resnet18()

        PATH = "resnet18.pt"
        checkpoint = torch.load(PATH)
        resnet18Test.load_state_dict(checkpoint)

        self.new_model = nn.Sequential(
            resnet18Test.conv1,
            resnet18Test.bn1,
            resnet18Test.relu,
            resnet18Test.maxpool,
            resnet18Test.layer1,
            resnet18Test.layer2,
            resnet18Test.layer3,
            resnet18Test.layer4,
            resnet18Test.avgpool,
        )

        for param in self.new_model.parameters():
            param.requires_grad_(False)

        self.fc = nn.Linear(512, 8)

        self.loss = nn.CrossEntropyLoss()
        ################## Your Code Ends here ##################

    def forward(self, x):
        """
        Perform a forward pass through your neural net.

        Parameters:
            x:      an (N, input_size) tensor, where N is arbitrary.

        Outputs:
            y:      an (N, output_size) tensor of output from the network
        """
        ################# Your Code Starts Here #################
        model_pred = self.new_model(x.float())
        inp = model_pred.view(model_pred.size(0), -1)
        y_pred = self.fc(inp)
        return y_pred

        ################## Your Code Ends here ##################


"""
3. (b)  Build a model
"""


def build_model(trained=False):
    """
    Parameters:
        trained:         a bool value specifying whether to use a model checkpoint

    Outputs:
        model:           the model to be used for training/testing
    """

    return FinetuneNet()


"""
4.  Build a PyTorch optimizer
"""


def build_optimizer(optim_type, model_params, hparams):
    """
    Parameters:
        optim_type:      the optimizer type e.g. "Adam" or "SGD"
        model_params:    the model parameters to be optimized
        hparams:         the hyperparameters (dict type) for usage with learning rate

    Outputs:
        optimizer:       a PyTorch optimizer object to be used in training
    """
    optimizer_class = getattr(torch.optim, optim_type)
    optimizer = optimizer_class(model_params, lr=hparams)

    return optimizer


"""
5. Training loop for model
"""


def train(train_dataloader, model, loss_fn, optimizer):
    """
    Train your neural network.

    Iterate over all the batches in dataloader:
        1.  The model makes a prediction.
        2.  Calculate the error in the prediction (loss).
        3.  Zero the gradients of the optimizer.
        4.  Perform backpropagation on the loss.
        5.  Step the optimizer.

    Parameters:
        train_dataloader:   a dataloader for the training set and labels
        model:              the model to be trained
        loss_fn:            loss function
        optimizer:          optimizer
    """

    ################# Your Code Starts Here #################
    for inputs, labels in train_dataloader:
        optimizer.zero_grad()
        prediction = model(inputs)
        loss = loss_fn(prediction, labels)
        loss.backward()
        optimizer.step()

    ################## Your Code Ends here ##################


"""
6. Testing loop for model
"""


def test(test_dataloader, model):
    """
    This part is optional.

    You can write this part to monitor your model training process.

    Test your neural network.
        1.  Make sure gradient tracking is off, since testing set should only
            reflect the accuracy of your model and should not update your model.
        2.  The model makes a prediction.
        3.  Calculate the error in the prediction (loss).
        4.  Print the loss.

    Parameters:
        test_dataloader:    a dataloader for the testing set and labels
        model:              the model that you will use to make predictions


    Outputs:
        test_acc:           the output test accuracy (0.0 <= acc <= 1.0)
    """

    acc = 0
    length = 0
    for features, labels in test_dataloader:
        length += len(labels)
        pred = model(features)
        _, predicted = torch.max(pred, 1)
        acc += (predicted == labels).sum().item()
    print(acc/length)


"""
7. Full model training and testing
"""


def run_model():
    """
    The autograder will call this function and measure the accuracy of the returned model.
    Make sure you understand what this function does.
    Do not modify the signature of this function (names and parameters).

    Please run your full model training and testing within this function.

    Outputs:
        model:              trained model
    """
    model = build_model()
    optimizer = build_optimizer("SGD", model.parameters(), 0.01)
    train_dataset = build_dataset(
        ['cifar10_batches/data_batch_1', 'cifar10_batches/data_batch_2', 'cifar10_batches/data_batch_3',
         'cifar10_batches/data_batch_4', "cifar10_batches/data_batch_5"], transform=get_preprocess_transform("train"))

    train_dataloader = build_dataloader(
        train_dataset, loader_params={"batch_size": 64, "shuffle": True})

    test_dataset = build_dataset(
        ['cifar10_batches/test_batch'], transform=get_preprocess_transform("test"))

    test_dataloader = build_dataloader(
        test_dataset, loader_params={"batch_size": 64, "shuffle": True})

    train(train_dataloader, model, model.loss, optimizer)

    model.eval()
    return model
