import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from IPython.display import clear_output
from torch.utils.data import Dataset


def visualize_augmentations(dataset, samples=5, initial_idx=0):
    """Visualises several samples from original dataset and threir augmented counterpars

    Parameters
    ----------
    dataset : utils.NumpyAugDataset
        the dataset with original and augmented images
    samples : int
        number of samples to show
    initial_idx : int
        initial index. The images shown have the following indices:
        initial_idx, initial_idx+ 1, ..., initial_idx + samples
    """

    figure, ax = plt.subplots(nrows=2, ncols=samples, figsize=(12, 6))

    for i in range(samples):
        images = dataset[initial_idx + i]
        ax[0, i].imshow(images["original"].transpose([1, 2, 0]), cmap="gray")
        ax[1, i].imshow(images["aug"].transpose([1, 2, 0]), cmap="gray")

    plt.tight_layout()
    plt.show()


def weight_init(model):
    """Initialises the model weights"""

    if isinstance(model, nn.Conv2d):
        nn.init.xavier_normal_(model.weight, gain=nn.init.calculate_gain("relu"))
        if model.bias is not None:
            nn.init.zeros_(model.bias)

    elif isinstance(model, nn.Linear):
        nn.init.xavier_normal_(model.weight)
        if model.bias is not None:
            nn.init.zeros_(model.bias)


def print_while_trainig(epochs_list, loss_history, loss_history_overclustering):
    """Prints loss and overcluster loss

    Parameters
    ----------
    epochs_list : list of ints
        epochs for wich loss is availiable
    loss_history: list of floats
        losses, the same size as the epochs_list
    loss_history_overclustering : list of floats
        overcluseting losses, the same size as the epochs_list
    """

    clear_output(True)

    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)
    fig.set_figwidth(12)

    ax1.plot(epochs_list, loss_history, label="train_loss")
    ax1.legend()
    ax1.grid()

    ax2.plot(
        epochs_list, loss_history_overclustering, label="train_loss_overclustering"
    )
    ax2.legend()
    ax2.grid()
    plt.show()


def get_cluster_labeling(model, dataloader, device=torch.device("cpu")):
    """Assing the cluster labels for the data from dataloader

    Parameters
    ----------
    model : resnetModel
        the model perfoming the clustering
    dataloader : torch.utils.data.dataloader.DataLoader
        the dataloader with the data to cluster
    device : torch.device
        device to perform the clustering on

    Returns
    ----------
    answers :
        original_labels : list of int
            original labels from the dataset
        cluster_labels :  list of int
            labels obtained from the model
    """
    model.eval()
    original_labels = []
    cluster_labels = []
    for batch in dataloader:
        images = batch["original"].to(device)
        labels = batch["label"].to(device)
        outputs = model(images, False)
        original_labels += labels.tolist()
        cluster_labels += torch.argmax(outputs, dim=1).tolist()
    return original_labels, cluster_labels


def visualise_clusetering_results(original_labels, cluster_labels, figwidth=20):
    """Visualises clusterisation results
    Shows, how the images with certain original labels are distributed fater clusterization

    Parameters:
    ----------
    original_labels : list of int
        list of original labels
    cluster_labels :  list of int
        list of clusterisator answers
    figwidth : int, optional
        width of plt figure
    """

    original_labels = np.array(original_labels)
    cluster_labels = np.array(cluster_labels)
    class_ids = np.unique(original_labels)
    fig, axes = plt.subplots(2, 5, constrained_layout=True)
    fig.set_figwidth(figwidth)

    for idx, ax in enumerate(axes.reshape(-1)):
        labels_distribution = original_labels[cluster_labels == idx]
        counts = np.array([np.sum(labels_distribution == i) for i in range(10)])
        ax.bar(list(range(10)), counts)
        ax.set_xticks(np.arange(10))
        ax.set_xlim([0, 9])
        ax.set_title("Original label: %i" % idx)


class startifiedAgentDataset(Dataset):
    """
    The dataset produced by stratified_split function.
    This datset takes the data directly from the initial dataset instance.
    It can acces only a certain part of original dataset data

    Attributes
    ----------
    original_dataset : torch.utils.data.dataset.Dataset like
        the original dataset ttake the data from
    indices_list : list
        list of indices wich the can be acessed from the orgignal dataset
    """

    def __init__(self, original_dataset, indices_list):
        super().__init__()
        self.original_dataset = original_dataset
        self.indices_list = indices_list

    def __len__(self):
        return len(self.indices_list)

    def __getitem__(self, idx):
        original_idx = self.indices_list[idx]
        return self.original_dataset[original_idx]


def stratified_split(original_dataset, train_size, label_key="label"):
    """Split the generic torch dataset in a stratified manned
    Based on sklearn.model_selection.train_test_split function
    The output datasets contain no data but refer to a certan indices in original dataset.

    Parameters
    ----------
    original_dataset : torch.utils.data.dataset.Dataset like
        original dataset to be splited
    train_size : float
        a fraction of the data train (first returned argument) dataset
    label_key : str or int
        a key to access the label from the original dataset.
        If original dataset returns a dict, it should prabably be "label"
        If original dataset returns a tuble, it should be int and in most cases equal to 1

    Returns
    ----------
    train_dataset : startifiedAgentDataset
        train part of the split. The realtive size is defined by train_size parameter
    test_dataset : startifiedAgentDataset
        test part of the split. The realtive size is len(original_dataset) - train_size


    """

    labels = [x[label_key] for x in original_dataset]
    indices = np.arange(len(labels))
    indices_and_labels = np.vstack([indices, labels]).transpose()
    train_indices_and_labels, test_indices_and_labels = train_test_split(
        indices_and_labels, train_size=train_size, stratify=labels
    )
    train_indices = train_indices_and_labels[:, 0].astype(np.int)
    test_indices = test_indices_and_labels[:, 0].astype(np.int)
    train_dataset = startifiedAgentDataset(original_dataset, train_indices)
    test_dataset = startifiedAgentDataset(original_dataset, test_indices)

    return train_dataset, test_dataset


def create_mapping(original_labels, cluster_labels):
    """Creates mapping from cluster labels to original labels.
    In each cluster the most frequent original label is chosen. This label is assumed to be cluster's true label

    Parameters
    ----------
    original_labels : list of int
        true labels of the data
    cluster_labels : list of int
        labels produced by a clustering algotithm

    Returns
    -------
    mapping : dict
        mapping from the clusters to original classes

    Notes
    -----
    This function should be used only if the clusters are well-defined
    """

    original_labels = np.array(original_labels, dtype=np.int)
    cluster_labels - np.array(cluster_labels, dtype=np.int)
    class_ids = np.unique(original_labels)
    cluster_ids = np.unique(cluster_labels)
    mapping = {}
    for cluster_id in cluster_ids:
        original_labels_in_cluster = original_labels[cluster_labels == cluster_id]
        map_to_id = np.bincount(original_labels_in_cluster).argmax()
        mapping[cluster_id] = map_to_id

    return mapping


def print_mapping(mapping):
    """Visualises the mapping

    Parameters
    ----------
    mapping : dict (int to int)
        a mapping from clusters to classes
    """
    print("Cluster  Class")
    for key, value in mapping.items():
        print("%3i -----> %i" % (key, value))


class LinearClusetrizator(nn.Module):
    """linear clusterisator for IIC. Useful for testing purposes"""

    def __init__(self):
        super(LinearClusetrizator, self).__init__()

        self.linear = nn.Linear(28 * 28, 10)
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, overclustering=False):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x
