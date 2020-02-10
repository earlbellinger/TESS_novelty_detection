import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os, re
import torch.utils.data as utils
from torch.utils.data.sampler import BatchSampler
print(torch.__version__)

from sklearn.model_selection import train_test_split
from torch.utils import data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from triplet_utils import AllTripletSelector,HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector # Strategies for selecting triplets within a minibatch
from sklearn.decomposition import PCA

import warnings

warnings.filterwarnings("ignore")

classes = ['APERIODIC', 'CONSTANT', 'CONTACT_ROT', 'DSCT_BCEP', 'ECLIPSE', 'GDOR_SPB',  'RRLYR_CEPHEID', 'SOLARLIKE']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    if embeddings.shape[1] > 2:
        pca = PCA(n_components=2)
        embeddings = pca.fit_transform(embeddings)
        print('Explained Ratio: ', pca.explained_variance_ratio_)
    plt.figure(figsize=(10,10))
    for i in range(10):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(classes)
    plt.show()
    plt.close()

def extract_embeddings(dataloader, model, embedding_size=2):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), embedding_size))
        labels = np.zeros(len(dataloader.dataset))
        filenames = []
        k = 0
        for images, target, filename in tqdm(dataloader, total=len(dataloader)):
            images = images.cuda().float()
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
            filenames.append(filename)
    return embeddings, labels, np.concatenate(filenames, axis=0)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """
    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)

class NPZ_Dataset(data.Dataset):
    def __init__(self, filenames,labels, outlier_mode='none'):

        self.filenames = filenames
        self.labels=labels
        self.indexes = np.arange(len(self.filenames))
        self.outlier_mode = outlier_mode

        assert len(self.indexes) == len(self.filenames) == len(self.labels)

    def __len__(self):
        'Total number of samples'
        return len(self.indexes)

    def __getitem__(self, index):
        'Generates ONE sample of data'

        batch_filenames = self.filenames[index]
        # Generate data
        X, y = self.data_generation(batch_filenames)

        return (X.copy(), y, batch_filenames)
        # print('y: ', y)
        # print('y_sigma: ', y_sigma)

    def data_generation(self, batch_filenames):
        data = np.load(batch_filenames)
        im = data['im']
        try:
            y = data['label']
        except:
            y = data['det']

        return im, y

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - Samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """
    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == int(label))[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size

class SLOSH_Embedding(nn.Module):
    def __init__(self, embed_size=2):
        super(SLOSH_Embedding, self).__init__()

        self.conv1 = nn.Conv2d(1, 4, kernel_size=7, padding=3)  # same padding 2P = K-1
        self.conv2 = nn.Conv2d(4, 8, kernel_size=5, padding=2)  # same padding 2P = K-1
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # same padding 2P = K-1

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.5)
        self.embed_size = embed_size
        self.linear1 = nn.Linear(int(16 * 16 * 16), self.embed_size)

    def forward(self, input_image):
        conv1 = self.conv1(input_image.unsqueeze(1)) # (N, C, H, W)
        conv1 = F.leaky_relu(conv1)
        conv1 = self.pool1(conv1)

        conv2 = F.leaky_relu(self.conv2(conv1))
        conv2 = self.pool2(conv2)


        conv3 = F.leaky_relu(self.conv3(conv2))
        conv3 = self.pool3(conv3)
        conv3 = self.drop1(conv3)

        linear1 = self.linear1(conv3.view(conv3.size()[0], -1))
        return linear1

    def get_embedding(self, x):
        return self.forward(x)

class SLOSH_Classifier(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(SLOSH_Classifier, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.linear2 = nn.Linear(self.embedding_net.embed_size, n_classes, bias=False)

    def forward(self, input_image):
        embedding = self.embedding_net(input_image)
        linear2 = self.linear2(F.relu(embedding))

        return linear2

    def get_embedding(self, x):

        return self.embedding_net(x)

def initialization(model):
    for name, param in model.named_parameters():  # initializing model weights
        if 'bias' in name:
            nn.init.constant_(param, 0.00)
        elif 'weight' in name:
            try:
                nn.init.xavier_uniform_(param)
            except:
                pass


def online_metric_learning():
    embed_size=2

    embedding_net = SLOSH_Embedding(embed_size=embed_size)
    model = embedding_net
    model.cuda()

    torch.backends.cudnn.benchmark = True
    print('CUDNN Enabled? ', torch.backends.cudnn.enabled)
    print(str(model))
    print('parameters', model.parameters)
    initialization(model)
    margin = 1.

    loss_function = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))

    model_checkpoint=False
    best_loss = 9999

    learning_rate = 0.001
    model_optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = ReduceLROnPlateau(model_optimizer, mode='min', factor=0.5, patience=10, verbose=True,
                                  min_lr=1E-6)

    root_folder = '/starclass_image_StandardScaled'  # folder with npz images here

    folder_filenames = []
    file_kic = []
    labels = []

    print('Parsing files in folder... ')
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for i, filex in enumerate(
                filenames):  # Getting the mags, KICS, and numax sigma for all stars in catalog
            if filex.endswith('.npz'):  # I infer the class label '0' or '1' according to subfolder names
                folder_filenames.append(os.path.join(dirpath, filex))
                kicx = int(re.search(r'\d+', filex).group())
                file_kic.append(kicx)
                labels.append(np.load(os.path.join(dirpath, filex))['label'])

    file_kic = np.array(file_kic)
    folder_filenames = np.array(folder_filenames)
    labels = np.array(labels)

    train_ids, val_ids, train_labels, val_labels = train_test_split(folder_filenames, labels, stratify=labels,
                                                                    test_size=0.15, random_state=137)

    train_labels = labels[np.in1d(folder_filenames, train_ids)]
    val_labels = labels[np.in1d(folder_filenames, val_ids)]
    train_filenames = folder_filenames[np.in1d(folder_filenames, train_ids)]
    val_filenames = folder_filenames[np.in1d(folder_filenames, val_ids)]

    print('Total Files: ', len(file_kic))

    print('Train Unique IDs: ', len(train_ids))
    print('Setting up generators... ')

    train_gen = NPZ_Dataset(filenames=train_filenames, labels=train_labels)
    train_dataloader = utils.DataLoader(train_gen, shuffle=True, batch_size=32, num_workers=4)
    train_batch_sampler = BalancedBatchSampler(train_gen.labels, n_classes=8, n_samples=25)

    train_dataloader_online = utils.DataLoader(train_gen, num_workers=10, batch_sampler=train_batch_sampler)


    val_gen = NPZ_Dataset(filenames=val_filenames, labels=val_labels)
    val_dataloader = utils.DataLoader(val_gen, shuffle=True, batch_size=32, num_workers=4)
    val_batch_sampler = BalancedBatchSampler(val_gen.labels, n_classes=8, n_samples=25)

    val_dataloader_online = utils.DataLoader(val_gen, num_workers=10, batch_sampler=val_batch_sampler)

    train_loader = train_dataloader_online
    val_loader = val_dataloader_online

    n_epochs = 301
    for epoch in range(1, n_epochs + 1):
        print('---------------------')
        print('Epoch: ', epoch)
        total_loss = 0
        train_batches = 0

        model.train()  # set to training mode
        losses = []
        for i, (data, target, _) in tqdm(enumerate(train_loader, 0), total=len(train_loader), unit='batches'):
            train_batches += 1
            data = data.float().cuda()
            target = target.long().cuda()
            data = data[target < 6]
            target = target[target < 6] # train without outliers

            model_optimizer.zero_grad()
            # Combined forward pass
            outputs = model(data)
            if type(outputs) not in (tuple, list):
                outputs = (outputs,)

            # Calculate loss and backpropagate

            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_function(*loss_inputs)

            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs

            losses.append(loss.item())
            total_loss += loss.item()
            loss.backward()
            model_optimizer.step()

        train_loss = total_loss / train_batches

        val_loss = 0
        val_batches = 0
        model.eval()
        with torch.no_grad():
            for i, (data, target, _) in tqdm(enumerate(val_loader, 0), total=len(val_loader), unit='batches'):
                val_batches += 1
                data = data.float().cuda()
                target = target.long().cuda()
                # data = data[target < 6]
                # target = target[target < 6]  # validate without outliers

                outputs = model(data)

                if type(outputs) not in (tuple, list):
                    outputs = (outputs,)
                loss_inputs = outputs
                if target is not None:
                    target = (target,)
                    loss_inputs += target

                loss_outputs = loss_function(*loss_inputs)
                loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
                val_loss += loss.item()
            val_loss = val_loss / val_batches

        print('\n\nTrain Loss: ', train_loss)
        print('Val Loss: ', val_loss)
        scheduler.step(train_loss)  # reduce LR on loss plateau

        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        print('Current Best Metric: ', best_loss)

        if epoch % 50 == 0:
            train_embeddings_baseline, train_labels_baseline, train_embed_filenames = extract_embeddings(
                train_dataloader, model, embedding_size=embed_size)
            np.savez_compressed('MULTISLOSH_OnlineTripletTrain_Outlier(6-7)_Embed2_', embedding=train_embeddings_baseline, label=train_labels_baseline, filename=train_embed_filenames)
            val_embeddings_baseline, val_labels_baseline, val_embed_filenames = extract_embeddings(
                val_dataloader, model, embedding_size=embed_size)
            # np.savez_compressed('MULTISLOSH_OnlineTripletVal_Trial2_Embed2', embedding=val_embeddings_baseline, label=val_labels_baseline, filename=val_embed_filenames)

            plot_embeddings(train_embeddings_baseline, train_labels_baseline)
            plot_embeddings(val_embeddings_baseline, val_labels_baseline)


online_metric_learning()
