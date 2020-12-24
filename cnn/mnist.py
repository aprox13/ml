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
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import seaborn as sn

SEED = 1000
torch.manual_seed(SEED)
np.random.seed(SEED)


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

PLOT_BY_EPOCH_FREQ = 2
IN_EPOCH_VERBOSE = False


def tqdm_in_epoch(data, **kwargs):
    if IN_EPOCH_VERBOSE:
        return tqdm(data, **kwargs)
    return data


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
    # model = MnistNet()
    # model.load_state_dict(torch.load('conv_net_model.ckpt'))

    optimizer = optimizer_cls(model.parameters(), lr=learning_rate)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_loader = get_train_dataloader(dataset_class, transform, batch_size)
    test_loader = get_test_dataloader(dataset_class, transform, batch_size)

    model.train()

    def plots(metrs, title=''):
        import matplotlib.pyplot as plt

        names = filter(lambda s: s != 'step', list(metrs.keys()))
        xx = list(range(1, len(metrs[TRAIN_ACC]) + 1))
        for name in names:
            if len(metrs[name]) == len(xx):
                plt.plot(xx, metrs[name], label=name)
        plt.xlabel('Epoch')
        plt.title(title)
        plt.legend()
        plt.show()

    TEST_ACC = 'Test accuracy'
    TRAIN_ACC = 'Train accuracy'
    TRAIN_LOSS_MEAN = 'Train loss mean'
    metrics = {
        TRAIN_ACC: [],
        TEST_ACC: [],
        TRAIN_LOSS_MEAN: [],
    }

    def test_model(return_matrix=False, metered=True):
        y_true = []
        y_pred = []

        image_matrix = np.zeros((10, 10, 28, 28, 1))
        model.eval()
        with torch.no_grad():
            for test_images, test_labels in tqdm_in_epoch(test_loader, desc='Testing'):
                test_outputs = model(test_images)
                _, test_predicted = torch.max(test_outputs.data, 1)

                y_true.extend(test_labels)
                y_pred.extend(test_predicted)
                if return_matrix:
                    for idx, (true, pred) in enumerate(zip(test_labels, test_predicted)):
                        image_matrix[true, pred] = test_images[idx].reshape((28, 28, 1))
        acc_test = accuracy_score(y_true, y_pred)
        print('Test Accuracy: {:.2f} %'.format(acc_test * 100))
        if metered:
            metrics[TEST_ACC].append(acc_test)

        if return_matrix:
            return y_true, y_pred, image_matrix

    for epoch in trange(num_epochs):
        i = -1
        model.train()
        losses = []
        train_true, train_pred = [], []
        for images, labels in tqdm_in_epoch(train_loader):
            i += 1
            # Прямой запуск
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Обратное распространение и оптимизатор
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            _, predicted = torch.max(outputs.data, 1)

            train_true.extend(labels)
            train_pred.extend(predicted)

        train_acc = accuracy_score(train_true, train_pred)
        metrics[TRAIN_ACC].append(train_acc)
        metrics[TRAIN_LOSS_MEAN].append(np.mean(np.array(losses)))
        print(f'[Epoch {epoch + 1}/{num_epochs}]')
        print('Train: accuracy {:.2f}%, mean loss {:.2f}'.format(train_acc * 100, metrics[TRAIN_LOSS_MEAN][-1]))
        test_model(return_matrix=False, metered=True)
        if (epoch + 1) % PLOT_BY_EPOCH_FREQ == 0 and IN_EPOCH_VERBOSE:
            plots(metrics, title=f'Epoch {epoch + 1}')

    # Сохраняем модель и строим график
    torch.save(model.state_dict(), 'conv_net_model.ckpt')
    true_test, pred_test, result_img_mtrx = test_model(return_matrix=True, metered=False)
    plots(metrics, 'After training end')
    conf_matrix(true_test, pred_test)
    show_img_matrix(result_img_mtrx)


# %%
train_test_model(MNIST, criterion=nn.CrossEntropyLoss(), optimizer_cls=torch.optim.Adam, num_epochs=12)
