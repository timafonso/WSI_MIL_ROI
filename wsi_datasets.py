from __future__ import print_function, division

import sys
sys.path.append("..")

from torch.utils.data import Dataset
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
torch.cuda.empty_cache()

import h5py
import numpy as np

class TumorEmbeddingDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.bags, self.coords, self.labels, self.paths, self.tissue_percentages = self.get_bag_embeddings()

    def get_bag_embeddings(self):
        bags = []
        labels = []
        coords = []
        paths = []
        tissue_percentages = []
        count = 0
        with h5py.File(self.path, 'r') as file:
            for group in file.values():
                count += 1
                bags.append(torch.tensor(np.array(group["embeddings"])))
                coords.append(torch.tensor(np.array(group["coords"]).squeeze(0)))
                labels.append(torch.tensor(np.array(group["label"])))
                paths.append(group["path"][()].decode().split("_")[0])
                tissue_percentages.append(torch.tensor(np.array(group["tissue_percentage"])))
        return bags, coords, labels, paths, tissue_percentages
    
    def __len__(self):
        return len(self.bags)
    
    def __getitem__(self, index):
        return self.bags[index], self.coords[index], self.labels[index], self.paths[index], self.tissue_percentages[index]


class GeneExpressionDataset(Dataset):
    def __init__(self, hdf5_file, label):
        self.label = label
        self.bags, self.coords, self.labels, self.slide_ids, self.case_ids = self.get_bag_embeddings(hdf5_file, label)

    def get_bag_embeddings(self, hdf5_file, label):
        bags = []
        coords = []
        labels = []
        slide_ids = []
        case_ids = []
        pos, neg = 0, 0
        with h5py.File(hdf5_file, 'r') as file:
            for group in file.values():
                bags.append(torch.tensor(np.array(group["embeddings"])))
                coords.append(torch.tensor(np.array(group["coords"])))
                labels.append(torch.tensor(np.array(group[label])))
                slide_ids.append(group["slide_id"][()].decode())
                case_ids.append(group["case_id"][()].decode())
        
        print(len(bags)) 
        return bags, coords, labels, slide_ids, case_ids            

    def __len__(self):
        return len(self.bags)
    
    def __getitem__(self, index):
        return self.bags[index], self.coords[index], self.labels[index], self.slide_ids[index], self.case_ids[index]






# ======================================= Collate Functions ===================================================


def tumor_collate(batch):
    imgs, coords, labels, path, percentage = zip(*batch)
    return torch.cat(imgs), coords, labels[0], path, percentage


def tumor_pad_collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    bags, coords, labels, paths, tissue_percentages = zip(*batch)

    separate_bags = []
    separate_coords = []
    separate_labels = []
    separate_paths = []
    separate_perc = []
    is_aug = []

    for bag_index in range(len(bags)):
        for aug in range(bags[bag_index].shape[0]):
            separate_bags.append(bags[bag_index][aug])
            separate_coords.append(coords[bag_index])
            separate_labels.append(labels[bag_index])
            separate_paths.append(paths[bag_index])
            separate_perc.append(tissue_percentages[bag_index])
            is_aug.append(torch.tensor(aug!=0))
                
    padded_bags = torch.nn.utils.rnn.pad_sequence(separate_bags, batch_first=True, padding_value=-np.Inf)
    padded_coords = torch.nn.utils.rnn.pad_sequence(separate_coords, batch_first=True, padding_value=-1)
    padded_labels = torch.stack(separate_labels)
    padded_paths = separate_paths
    padded_tissue_percentages = torch.stack(separate_perc)
    padded_aug = torch.stack(is_aug)

    return padded_bags, padded_coords, padded_labels, paths, padded_tissue_percentages, padded_aug


def gene_collate(batch):
    imgs, coords, labels, slide_id, case_id = zip(*batch)
    return torch.cat(imgs), coords, labels, slide_id, case_id

def gene_pad_collate_fn(batch):
    # Enumerate the batch to keep track of original indices
    batch_with_indices = [(item, index) for index, item in enumerate(batch)]
    batch_with_indices.sort(key=lambda x: len(x[0][0]), reverse=True)

    # Unpack the sorted batch and original indices
    sorted_batch, sorted_indices = zip(*batch_with_indices)

    bags, coords, labels, slide_ids, case_ids = zip(*sorted_batch)

    separate_bags = []
    separate_coords = []
    separate_labels = []
    separate_slide_ids = []
    separate_case_ids = []
    is_aug = []

    for bag_index in range(len(bags)):
        for aug in range(bags[bag_index].shape[0]):
            separate_bags.append(bags[bag_index][aug])
            separate_coords.append(coords[bag_index])
            separate_labels.append(labels[bag_index])
            separate_slide_ids.append(slide_ids[bag_index])
            separate_case_ids.append(case_ids[bag_index])
            is_aug.append(torch.tensor(aug != 0))

    padded_bags = torch.nn.utils.rnn.pad_sequence(separate_bags, batch_first=True, padding_value=-np.Inf)
    padded_coords = torch.nn.utils.rnn.pad_sequence(separate_coords, batch_first=True, padding_value=-1)
    padded_labels = torch.stack(separate_labels)
    padded_slide_ids = separate_slide_ids
    padded_case_ids = separate_case_ids
    padded_aug = torch.stack(is_aug)

    # Use sorted_indices to reorder the data
    original_order = torch.argsort(torch.tensor(sorted_indices))
    padded_bags = padded_bags[original_order]
    padded_coords = padded_coords[original_order]
    padded_labels = padded_labels[original_order]
    padded_slide_ids = [padded_slide_ids[i] for i in original_order]
    padded_case_ids = [padded_case_ids[i] for i in original_order]
    padded_aug = padded_aug[original_order]

    return padded_bags, padded_coords, padded_labels, padded_slide_ids, padded_case_ids, padded_aug