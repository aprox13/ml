# %%
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm.notebook import trange, tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn

torch.manual_seed(42)
np.random.seed(42)


# %%

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


def conf_matrix(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
    index, columns = conf_matrix.shape
    df_cm = pd.DataFrame(conf_matrix, index=range(index),
                         columns=range(columns))
    plt.figure(figsize=(10, 7))
    vmax = np.unique(y_pred, return_counts=True)[1].max()
    sn.heatmap(df_cm, annot=True, cmap='coolwarm', vmin=0, vmax=vmax, fmt='.4g')


# %%

metrics = {
    'train': {
        'Accuracy': [],
        'loss': [],
        'step': []
    }
}


def train_test_model(dataset_class,
                     criterion,
                     optimizer_cls,
                     learning_rate=1e-3,
                     batch_size=32,
                     model=None,
                     num_epochs=20,
                     ):
    if model is None:
        model = MnistNet()

    optimizer = optimizer_cls(model.parameters(), lr=learning_rate)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

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
            # Прямой запуск
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            # Обратное распространение и оптимизатор
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Отслеживание точности
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            train_metric['Accuracy'].append(correct / total)
            train_metric['loss'].append(loss.item())
            train_metric['step'].append(epoch + 1 + (i / total_step))

            # if (i + 1) % 100 == 0:
            #     print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
            #           .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
            #                   (correct / total) * 100))

        plots(train_metric)

    y_true = []
    y_pred = []
    image_matrix = np.zeros((28, 28, 1))
    model.eval()
    test_loader = get_test_dataloader(dataset_class, transform, batch_size)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels)
            y_pred.extend(predicted)
            for idx, (true, pred) in enumerate(zip(labels, predicted)):
                image_matrix[true, pred] = images[idx]

        print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))

    # Сохраняем модель и строим график
    torch.save(model.state_dict(), 'conv_net_model.ckpt')
    conf_matrix(y_true, y_pred)
    show_img_matrix(image_matrix)


# %%
train_test_model(MNIST, criterion=nn.CrossEntropyLoss(), optimizer_cls=torch.optim.Adam)
