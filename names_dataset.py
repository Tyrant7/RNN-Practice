import torch
import time
import glob
import os

from torch.utils.data import Dataset

import word_util


class NamesDataset(Dataset):

    def __init__(self, data_dir):
        self.data_dir = data_dir  # for provenance of the dataset
        self.load_time = time.localtime()  # for provenance of the dataset
        labels_set = set()  # set of all classes

        self.data = []
        self.data_tensors = []
        self.labels = []
        self.labels_tensors = []

        # Read all the ".txt" files into the specified directory
        text_files = glob.glob(os.path.join(data_dir, "*.txt"))
        for filename in text_files:
            label = os.path.splitext(os.path.basename(filename))[0]
            labels_set.add(label)
            lines = open(filename, encoding="utf-8").read().strip().split("\n")
            for name in lines:
                self.data.append(name)
                self.data_tensors.append(word_util.line_to_tensor(name))
                self.labels.append(label)

        # Cache the tensor representation of the labels
        self.labels_uniq = list(labels_set)
        for idx in range(len(self.labels)):
            temp_tensor = torch.tensor([self.labels_uniq.index(self.labels[idx])], dtype=torch.long)
            self.labels_tensors.append(temp_tensor)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        data_label = self.labels[idx]
        data_tensor = self.data_tensors[idx]
        label_tensor = self.labels_tensors[idx]

        return label_tensor, data_tensor, data_label, data_item
