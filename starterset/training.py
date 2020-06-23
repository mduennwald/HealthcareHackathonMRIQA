# GPU choice
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import random
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from mriqa_dataset import MRIQADataset
from networks import ClassicCNN


# set random seeds for reproducibility
random.seed(21062020)
np.random.seed(21062020)
torch.manual_seed(21062020)
torch.cuda.manual_seed(21062020)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train():
    num_epochs = 500
    batch_size = 8
    loss_csv = open('losses.csv', 'w')
    loss_csv.write('epoch,training,validation\n')

    # create dataset (automatically downloads IXI at first run)
    dataset = MRIQADataset(
        '.',    # path to save data to
        modalities=('T1', 'T2'),
        download=True,
    )

    # split data into training and validation sets
    train_set, validation_set = torch.utils.data.random_split(dataset, (100, 85))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)

    net = ClassicCNN(num_classes=5)
    net = net.cuda()
    optimizer = optim.Adam(net.parameters())
    ce = CrossEntropyLoss().cuda()

    num_mini_batches = len(train_loader)
    best_val_loss = 999
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        net.train()

        # train loop
        pbar = tqdm(train_loader, total=len(train_loader))
        for sample, label in pbar:
            sample = sample.cuda()
            label = label.cuda()

            prediction = net(sample)
            loss = ce(prediction, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print('[{}] train-loss: {}'.format(epoch, epoch_loss / num_mini_batches))
        loss_csv.write(str(epoch) + ',' + str(epoch_loss / num_mini_batches))
        loss_csv.flush()

        # validation loop
        net.eval()
        mean_validation_loss = 0
        num_validation_mini_batches = len(validation_loader)
        with torch.no_grad():
            pbar = tqdm(validation_loader, total=len(validation_loader))
            for sample, label in pbar:
                sample = sample.cuda()
                label = label.cuda()

                prediction = net(sample)
                validation_loss = ce(prediction, label)

                mean_validation_loss += validation_loss.item()
            print('[{}] validation-loss: {}'.format(epoch, mean_validation_loss / num_validation_mini_batches))
            loss_csv.write(',' + str(mean_validation_loss / num_validation_mini_batches) + '\n')
            loss_csv.flush()
        # save best model
        if mean_validation_loss <= best_val_loss:
            torch.save({'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': validation_loss.item()}, 'checkpoint_best')
            best_val_loss = mean_validation_loss
    print('DONE.')

    
if __name__ == '__main__':
    train()
