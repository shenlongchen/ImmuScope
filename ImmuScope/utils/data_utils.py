# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/15
@Auth ： shenlongchen
"""
import os
import h5py
import numpy as np

ACIDS = '0-ACDEFGHIKLMNPQRSTVWY'

def get_mhc_name_seq(mhc_name_seq_file):
    mhc_name_seq = {}
    with open(mhc_name_seq_file) as fp:
        for line in fp:
            mhc_name, mhc_seq = line.split()
            mhc_name_seq[mhc_name] = mhc_seq
    return mhc_name_seq

def create_splits(training_data, split_ratio=0.1, seed=2024):
    np.random.seed(seed)
    with h5py.File(training_data, 'r') as f:
        data_len = len(f['labels'])
    indices = np.arange(data_len)
    np.random.shuffle(indices)

    split = int(np.floor(split_ratio * data_len))
    train_indices, valid_indices = indices[split:], indices[:split]
    return train_indices, valid_indices

def create_splits_train_valid_test(training_data, train_ratio=0.9, valid_ratio=0.05, test_ratio=0.05, seed=2024):
    np.random.seed(seed)
    with h5py.File(training_data, 'r') as f:
        data_len = len(f['labels'])
    indices = np.arange(data_len)
    np.random.shuffle(indices)

    train_end = int(np.floor(train_ratio * data_len))
    valid_end = train_end + int(np.floor(valid_ratio * data_len))

    train_indices = indices[:train_end]
    valid_indices = indices[train_end:valid_end]
    test_indices = indices[valid_end:]

    return train_indices, valid_indices, test_indices

def get_data_id_mhc_name(data_id_mhc_name_file):
    data_id_mhc_name = {}
    with open(data_id_mhc_name_file) as fp:
        for line in fp:
            data_id, mhc_names = line.split()
            data_id = data_id.replace('#', '')
            data_id_mhc_name[data_id] = mhc_names.split(',')
    return data_id_mhc_name

def save_mhc_peptide_h5py(data, save_file, label_dtype=np.float32, peptide_len=21, padding_idx=0, peptide_pad=3):
    mhc_names = [x[0] for x in data]
    peptide_seqs = [x[1] for x in data]
    peptide_contexts = [x[2] for x in data]
    labels = np.array([x[3] for x in data], dtype=label_dtype)

    peptide_embedding_list = get_peptide_embedding(peptide_seqs, peptide_len=peptide_len,
                                                   padding_idx=padding_idx, peptide_pad=peptide_pad)

    # create HDF5 file to save training and testing data
    save_h5py_file(save_file, labels, mhc_names, peptide_contexts, peptide_embedding_list, label_dtype)

def get_peptide_embedding(peptide_seqs, peptide_len=21, padding_idx=0, peptide_pad=3):
    peptide_embedding_list = []
    for peptide_seq in peptide_seqs:
        peptide_seq = [ACIDS.index(x if x in ACIDS else '-') for x in peptide_seq][:peptide_len]
        peptide_embedding_list.append([padding_idx] * peptide_pad +
                                      peptide_seq + [padding_idx] * (peptide_len - len(peptide_seq)) +
                                      [padding_idx] * peptide_pad)
        assert len(peptide_embedding_list[-1]) == peptide_len + peptide_pad * 2
    return peptide_embedding_list

def save_h5py_file(save_file, labels, mhc_names, peptide_contexts, peptide_embedding_list, label_dtype, bags_id=None):
    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    if os.path.exists(save_file):
        os.remove(save_file)
    h5 = h5py.File(save_file, 'w')
    dt = h5py.string_dtype(encoding='utf-8')
    if bags_id is not None:
        h5.create_dataset('bags_id', data=bags_id, dtype=np.int32)
    h5.create_dataset('mhc_names', data=mhc_names, dtype=dt)
    h5.create_dataset('peptide_embedding', data=peptide_embedding_list, dtype=np.int32)
    h5.create_dataset('peptide_contexts', data=peptide_contexts, dtype=dt)
    h5.create_dataset('labels', data=labels, dtype=label_dtype)
    h5.close()

def restore_peptide_sequences(embeddings, peptide_pad=3):
    # Remove padding and convert indexes back to characters
    peptide_seqs = []
    for index in range(embeddings.shape[0]):
        emb = embeddings[index]
        # Slice the padding off both ends
        trimmed_emb = emb[peptide_pad:-peptide_pad]
        # Convert indexes back to amino acids
        seq = ''.join(ACIDS[idx] for idx in trimmed_emb)
        seq = seq.replace('0', '')
        peptide_seqs.append(seq)
    return peptide_seqs