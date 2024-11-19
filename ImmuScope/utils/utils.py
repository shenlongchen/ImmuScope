# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/15
@Auth ： shenlongchen
"""

import csv
import os
from collections import namedtuple
from pathlib import Path
from ruamel.yaml import YAML
from datetime import datetime

import numpy as np
import pandas as pd
import logging
import torch
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve
Metrics = namedtuple('Metrics', ['auc0_1', 'aupr', 'ppv'])


@torch.no_grad()
def truncated_normal_(tensor, mean=0.0, std=1.0):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def load_config_and_setup_logging(data_cnf, model_cnf, logger_name='ImmuScope'):
    # load config
    yaml = YAML(typ='safe')
    data_cnf, model_cnf = yaml.load(Path(data_cnf)), yaml.load(Path(model_cnf))
    # log file setting
    timestamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    logger_path = Path(os.path.join(data_cnf['logs'], f'{model_cnf["name"]}_{timestamp}.log'))
    logger_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname).1s %(asctime)s %(module)s:%(lineno)d] %(message)s',
                                datefmt='%y%m%d %H:%M')

    # create file handler which logs even debug messages
    debug_file_handler = logging.FileHandler(logger_path.with_stem(f'{logger_path.stem}-debug'))
    debug_file_handler.setLevel(logging.DEBUG)
    debug_file_handler.setFormatter(formatter)

    # create file handler which logs only info messages
    info_file_handler = logging.FileHandler(logger_path.with_stem(f'{logger_path.stem}-info'))
    info_file_handler.setLevel(logging.INFO)
    info_file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    logger.addHandler(debug_file_handler)
    logger.addHandler(info_file_handler)
    logger.addHandler(console_handler)

    logger.info(f'Model Name: {model_cnf["name"]}')
    return logger, data_cnf, model_cnf

def calculate_auc(labels, scores):
    return roc_auc_score(labels, scores)

def calculate_auc_0_1(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)

    index_10_percent = np.where(fpr <= 0.1)[0][-1]
    if index_10_percent < 1:
        return -1
    auc_0_1 = auc(fpr[:index_10_percent + 1], tpr[:index_10_percent + 1])
    return auc_0_1

def calculate_ppv_from_probabilities(labels, scores):
    """
    Calculate PPV based on prediction probabilities and true labels.

    Parameters:
    - y_true: numpy array of true binary labels (0 or 1).
    - y_prob: numpy array of prediction probabilities (0 to 1).

    Returns:
    - PPV: Positive Predictive Value as a float.
    """
    # Determine pos_num as the number of actual positives in y_true
    pos_num = np.sum(labels == 1)

    # Sort predictions by probability in descending order
    sorted_indices = np.argsort(scores)[::-1]

    # Select top N predictions
    top_n_predictions = labels[sorted_indices][:pos_num]

    # Calculate the number of true positives in the top pos_num predictions
    true_positives = np.sum(top_n_predictions == 1)

    # Calculate PPV
    ppv = true_positives / pos_num if pos_num > 0 else 0
    return ppv

def calculate_aupr(labels, scores):
    """
    Calculate the Area Under the Precision-Recall Curve (AUPR).

    Parameters:
    - y_true: numpy array or list, true binary labels (0 or 1).
    - y_scores: numpy array or list, predicted probabilities or scores.

    Returns:
    - AUPR: Area Under the Precision-Recall Curve as a float.
    """
    # Calculate Precision and Recall values for different thresholds
    precision, recall, _ = precision_recall_curve(labels, scores)
    # Calculate the area under the Precision-Recall curve
    aupr = auc(recall, precision)
    return aupr

def get_group_metrics(mhc_names, labels, scores, reduce=True):
    mhc_names, labels, scores = np.asarray(mhc_names), np.asarray(labels), np.asarray(scores)
    mhc_groups, metrics = [], Metrics([], [], [])
    for mhc_name_ in sorted(set(mhc_names)):
        t_, s_ = labels[mhc_names == mhc_name_], scores[mhc_names == mhc_name_]
        if len(t_) > 30 and len(t_[t_ == 1]) >= 25 and len(t_[t_ == 0]) > 0:
            t_ = np.where(t_ > 0.5, 1, 0)
            mhc_groups.append(mhc_name_)
            metrics.auc0_1.append(calculate_auc_0_1(t_, s_))
            metrics.aupr.append(calculate_aupr(t_, s_))
            metrics.ppv.append(calculate_ppv_from_probabilities(t_, s_))
    return (np.mean(x) for x in metrics) if reduce else (mhc_groups,) + metrics


def output_res(mhc_names, labels, scores, output_path: Path, logger):
    # save the results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, scores)
    # save the results to csv
    eval_out_path = output_path.with_suffix('.csv')
    mhc_names, labels, scores, metrics = np.asarray(mhc_names), np.asarray(labels), np.asarray(scores), []
    with open(eval_out_path, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['allele', 'total', 'positive', 'AUC0.1', 'AUPR', 'PPV'])
        mhc_groups, auc0_1, aupr, ppv = get_group_metrics(mhc_names, labels, scores, reduce=False)
        for mhc_name_, auc0_1_, aupr_, ppv_ in zip(mhc_groups, auc0_1, aupr, ppv):
            t_ = labels[mhc_names == mhc_name_]
            writer.writerow([mhc_name_, len(t_), len(t_[t_ == 1]), auc0_1_, aupr_, ppv_])
            metrics.append((auc0_1_, aupr_, ppv_))
        metrics = np.mean(np.array(metrics), axis=0)
        writer.writerow([''] * 3 + metrics.tolist())
    logger.info("--- Saved csv and npy, Group-Test: AUC0_1: {:.4f}  AUPR: {:.4f}  PPV: {:.4f} ---".format(*metrics))


def calculate_all_metrics(labels, scores):
    labels = np.where(labels > 0.5, 1, 0)
    auc0_1 = calculate_auc_0_1(labels, scores)
    aupr = calculate_aupr(labels, scores)
    ppv = calculate_ppv_from_probabilities(labels, scores)
    return auc0_1, aupr, ppv

def calculate_group_metrics(labels, scores, mhc_names):
    labels, scores, metrics = np.asarray(labels), np.asarray(scores), []
    mhc_groups, auc0_1, aupr, ppv = get_group_metrics(mhc_names, labels, scores, reduce=False)
    for mhc_name_, auc0_1_, aupr_, ppv_ in zip(mhc_groups, auc0_1, aupr, ppv):
        metrics.append((auc0_1_, aupr_, ppv_))
    metrics = np.mean(np.array(metrics), axis=0)

    auc0_1_group, aupr_group, ppv_group = metrics[0], metrics[1], metrics[2]
    return auc0_1_group, aupr_group, ppv_group


def calculate_group_auc(labels, scores, mhc_names, min_pos_num=25, min_total_num=30):
    labels, scores, metrics = np.asarray(labels), np.asarray(scores), []

    mhc_names, labels, scores = np.asarray(mhc_names), np.asarray(labels), np.asarray(scores)
    mhc_groups, auc_group = [], []
    for mhc_name_ in sorted(set(mhc_names)):
        t_, s_ = labels[mhc_names == mhc_name_], scores[mhc_names == mhc_name_]
        if len(t_) > min_total_num and len(t_[t_ == 1]) >= min_pos_num and len(t_[t_ == 0]) > 0:
            t_ = np.where(t_ > 0.5, 1, 0)
            mhc_groups.append(mhc_name_)
            auc_group.append(calculate_auc(t_, s_))
    avg_auc_group = np.mean(auc_group)
    return avg_auc_group


def calculate_auc_base_protein(protein_list, pred_instances, labels):
    unique_proteins = np.unique(protein_list)
    protein_to_indices = {protein: np.where(protein_list == protein)[0] for protein in unique_proteins}
    auc_list, protein_name_list = [], []
    for protein_, indices in protein_to_indices.items():
        pred_instances_protein = pred_instances[indices]
        labels_protein = labels[indices]
        pred_auc_protein = roc_auc_score(labels_protein, pred_instances_protein)
        auc_list.append(pred_auc_protein)
        protein_name_list.append(protein_)

    pred_auc = roc_auc_score(labels, pred_instances)
    peptide_protein_df = pd.DataFrame({'Protein': protein_name_list, 'AUC': auc_list})
    return np.median(auc_list), np.mean(auc_list), pred_auc, peptide_protein_df
