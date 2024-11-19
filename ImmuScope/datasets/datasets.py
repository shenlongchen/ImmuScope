# -*- coding: utf-8 -*-
"""
@Time ： 2023/8/15
@Auth ： shenlongchen
"""
import numpy as np
from ..utils.data_utils import ACIDS
from torch.utils.data.dataset import Dataset
import h5py


class SinInstanceBag(Dataset):
    def __init__(self, data_path, mhc_name_seq, indices=None):
        self.data_path = data_path
        self.indices = indices
        self.mhc_name_seq = mhc_name_seq
        self.mhc_names = None
        self.peptide_seqs = None
        self.peptide_contexts = None
        self.labels = None
        self.mhc_embedding_dict = {}

        if self.indices is None:
            with h5py.File(self.data_path, 'r') as f:
                self.indices = np.arange(len(f['peptide_embedding']))

        for mhc_name in mhc_name_seq:
            mhc_seq = mhc_name_seq[mhc_name]
            mhc_embedding = np.asarray([ACIDS.index(x if x in ACIDS else '-') for x in mhc_seq])
            self.mhc_embedding_dict[mhc_name] = np.expand_dims(mhc_embedding, axis=0)

        with h5py.File(self.data_path, 'r') as f:
            # decode byte to string
            self.mhc_names = f['mhc_names'][()]
            self.mhc_names = np.array([x.decode('utf-8') for x in self.mhc_names])
            self.peptide_embedding = f['peptide_embedding'][()]
            self.peptide_contexts = f['peptide_contexts'][()]
            self.peptide_contexts = np.array([x.decode('utf-8') for x in self.peptide_contexts])
            self.labels = np.asarray(f['labels'][()], dtype=np.float32)

        self.peptide_embedding = np.expand_dims(self.peptide_embedding, axis=1)

    def __getitem__(self, idx):
        index = self.indices[idx]
        mhc_name = self.mhc_names[index]
        return (self.peptide_embedding[index], self.mhc_embedding_dict[mhc_name]), self.labels[index]

    def __len__(self):
        return len(self.indices)


class MABags(Dataset):
    def __init__(self, dataset_path, mhc_name_seq, bag_size=10, onlyPositive=False):

        self.bag_size = bag_size
        self.bags_id = None
        self.mhc_names = None
        self.peptide_embedding = None
        self.peptide_contexts = None
        self.labels = None
        self.mhc_embedding_dict = {}

        for mhc_name in mhc_name_seq:
            mhc_seq = mhc_name_seq[mhc_name]
            self.mhc_embedding_dict[mhc_name] = np.asarray([ACIDS.index(x if x in ACIDS else '-') for x in mhc_seq])

        with h5py.File(dataset_path, 'r') as f:
            # decode byte to string
            self.bags_id = np.asarray(f['bags_id'][()], dtype=np.int32)
            self.mhc_names = f['mhc_names'][()]
            self.mhc_names = np.array([x.decode('utf-8') for x in self.mhc_names])
            self.peptide_embedding = f['peptide_embedding'][()]
            self.peptide_contexts = f['peptide_contexts'][()]
            self.peptide_contexts = np.array([x.decode('utf-8') for x in self.peptide_contexts])
            self.labels = np.asarray(f['labels'][()], dtype=np.float32)

            if onlyPositive:
                positive_idx = []
                for i in range(len(self.bags_id) // self.bag_size):
                    idx_start = i * self.bag_size
                    idx_end = (i + 1) * self.bag_size
                    if max(self.labels[idx_start:idx_end]) == 1:
                        positive_idx.extend(range(idx_start, idx_end))
                self.mhc_names = np.array([self.mhc_names[i] for i in positive_idx])
                self.peptide_embedding = self.peptide_embedding[positive_idx]
                self.peptide_contexts = np.array([self.peptide_contexts[i] for i in positive_idx])
                self.labels = self.labels[positive_idx]

    def __len__(self):
        return len(self.labels) // self.bag_size

    def __getitem__(self, index):
        # peptide, mhc
        idx_start = index * self.bag_size
        idx_end = (index + 1) * self.bag_size

        mhc_embedding = []
        for item in self.mhc_names[idx_start: idx_end]:
            mhc_embedding.append(self.mhc_embedding_dict[item])

        bag = (self.peptide_embedding[idx_start:idx_end], np.asarray(mhc_embedding))
        label = [max(self.labels[idx_start:idx_end]), self.labels[idx_start:idx_end]]

        return bag, label


class SinInstanceForLogo(Dataset):
    def __init__(self, data_path, mhc_name_seq, mhc_name, indices=None):
        self.data_path = data_path
        self.indices = indices
        self.mhc_name_seq = mhc_name_seq
        self.mhc_names = None
        self.peptide_seqs = None

        if self.indices is None:
            with h5py.File(self.data_path, 'r') as f:
                self.indices = np.arange(len(f['peptide_embedding']))

        self.mhc_seq = mhc_name_seq[mhc_name]
        self.mhc_embedding = np.asarray([ACIDS.index(x if x in ACIDS else '-') for x in self.mhc_seq])
        self.mhc_embedding = np.expand_dims(self.mhc_embedding, axis=0)

        with h5py.File(self.data_path, 'r') as f:
            # decode byte to string
            self.peptide_embedding = f['peptide_embedding'][()]
            self.peptide_seqs = f['peptide_seqs'][()]
            self.peptide_seqs = [x.decode('utf-8') for x in self.peptide_seqs]

        self.peptide_embedding = np.expand_dims(self.peptide_embedding, axis=1)

    def __getitem__(self, idx):
        index = self.indices[idx]
        return (self.peptide_embedding[index], self.mhc_embedding), np.float32(0)

    def __len__(self):
        return len(self.indices)
