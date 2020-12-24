# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST
from tqdm.notebook import trange, tqdm


class MnistNet(nn.Module):
    def __init__(self, n_classes=10):
        super(MnistNet, self).__init__()
        # in_channels = 1 and img_size = 28x28
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )  # out_channels = 32 and img_size = 14x14 for mnist
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )  # out num_channels = 64 and imf_size = 7x7
        self.drop_out = nn.Dropout()  # for avoid oferfitting
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, n_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)  # flatten
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = F.softmax(out, dim=1)
        return out


# %%

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


def dataset_loader(dataset_class, root, train, transform, download, shuffle, batch_size):
    return DataLoader(dataset_class(root, train, transform, download=download), shuffle=shuffle, batch_size=batch_size)


def get_train_dataloader(ds_cls, transform, batch_size):
    return dataset_loader(
        ds_cls,
        root='./models',
        train=True,
        transform=transform,
        download=True,
        shuffle=True,
        batch_size=batch_size
    )


def get_test_dataloader(ds_cls, transform, batch_size):
    return dataset_loader(
        ds_cls,
        root='./models',
        train=False,
        transform=transform,
        download=True,
        shuffle=False,
        batch_size=batch_size
    )


metrics = {
    'train': {
        'Accuracy': [],
        'loss': [],
        'step': []
    }
}


def train_model(dataset_class,
                criterion,
                optimizer_cls,
                learning_rate=1e-3,
                batch_size=32,
                model=None,
                num_epochs=20,
                name_prefix=''
                ):
    if model is None:
        model = MnistNet()

    optimizer = optimizer_cls(model.parameters(), lr=learning_rate)

    train_loader = get_train_dataloader(dataset_class, transform, batch_size)

    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    model.train()
    train_metric = metrics['train']

    def plots(metrs):
        import matplotlib.pyplot as plt

        names = filter(lambda s: s != 'step', list(metrs.keys()))

        for name in names:
            plt.plot(metrs['step'], metrs[name], label=name)
            plt.xlabel('epoch')
            plt.show()

    for epoch in trange(num_epochs):
        i = -1
        for images, labels in tqdm(train_loader):
            i += 1
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            train_metric['Accuracy'].append(correct / total)
            train_metric['loss'].append(loss.item())
            train_metric['step'].append(epoch + 1 + (i / total_step))

        plots(train_metric)

    torch.save(model.state_dict(), f'{name_prefix}net.ckpt')
    return model


# %%

def show_img_matrix(images_matrix):
    nrow = len(images_matrix)
    ncol = len(images_matrix[0])
    f, axarr = plt.subplots(
        nrow, ncol,
        gridspec_kw=dict(wspace=0.1, hspace=0.1,
                         top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                         left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1)),
        figsize=(ncol + 1, nrow + 1),
        sharey='row', sharex='col',  # optionally
    )

    for ax in axarr.ravel():
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_axis_off()

    for i in range(len(images_matrix)):
        for j in range(len(images_matrix[i])):
            image = images_matrix[i][j]
            axarr[i, j].imshow(image)
    plt.show()


def test_model(dataset_class,
               model=None,
               batch_size=32):
    model.eval()
    test_loader = get_test_dataloader(dataset_class, transform, batch_size)
    img_arr = np.zeros((10, 10, 28, 28, 1))
    y_true_pred = [[], []]
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            y_true_pred[0].extend(labels)
            y_true_pred[1].extend(predicted)

            for idx, (true, pred) in enumerate(zip(labels, predicted)):
                img_arr[true, pred] = images[idx].reshape((28, 28, 1))

        acc = accuracy_score(y_true=y_true_pred[0], y_pred=y_true_pred[1])
        size = len(y_true_pred[0])
        print(
            'Test accuracy {:.2f}% - {}/{}'.format(acc * 100, acc * size, size)
        )
        conf_matrix = confusion_matrix(y_true=y_true_pred[0], y_pred=y_true_pred[1])
        index, columns = conf_matrix.shape
        df_cm = pd.DataFrame(conf_matrix, index=range(index),
                             columns=range(columns))
        plt.figure(figsize=(10, 7))
        vmax = np.unique(y_true_pred[1], return_counts=True)[1].max()
        sn.heatmap(df_cm, annot=True, cmap='coolwarm', vmin=0, vmax=vmax, fmt='.4g')
        show_img_matrix(img_arr)


# %%
def trained_model(file_path):
    model = MnistNet()
    model.load_state_dict(torch.load(file_path))
    return model


# %%
model = train_model(MNIST, criterion=nn.CrossEntropyLoss(), optimizer_cls=torch.optim.Adam, name_prefix='mnist')

# model = trained_model('/Users/ifkbhit/itmo/ml-git/data/conv_net_model.ckpt')

test_model(MNIST, model)

# %%
model = train_model(FashionMNIST, criterion=nn.CrossEntropyLoss(), optimizer_cls=torch.optim.Adam,
                    name_prefix='fashion_mnist')
test_model(FashionMNIST, model)
