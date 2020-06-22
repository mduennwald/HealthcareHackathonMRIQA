import csv
import os
from os.path import join as pjoin

import nibabel as nib
import torch

from networks import CustomResNet

class BenchmarkDataset:
    def __init__(self):
        super().__init__()
        self.all_files = sorted(os.listdir('benchmark'))

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, index):
        sample = nib.load(pjoin('benchmark', self.all_files[index])).get_fdata()
        sample = torch.from_numpy(sample).float()[None, None, ...]
        sample = sample - torch.mean(sample)
        sample = sample / torch.std(sample)
        return sample, self.all_files[index]

def main():
    checkpoint_dict = torch.load('checkpoint_best')
    model = CustomResNet(num_classes=5)
    model.load_state_dict(checkpoint_dict['model_state_dict'])
    model.cuda()
    model.eval()

    dataset = BenchmarkDataset()
    softmax = torch.nn.Softmax()

    with open('benchmark.csv', 'w', newline='') as csvhandle:
        csvfile = csv.writer(csvhandle)
        for num_item, (sample, fname) in enumerate(dataset):
            print(f'{num_item + 1}/{len(dataset)}')
            prediction = model(sample.cuda()).detach().cpu()
            prediction = softmax(prediction)
            _, pred_label = prediction.max(1)
            binary_prediction = torch.zeros(5)
            binary_prediction[pred_label] = 1
            csvfile.writerow([fname] + list(prediction.numpy()[0]) + list(binary_prediction.numpy()))

if __name__ == "__main__":
    main()
