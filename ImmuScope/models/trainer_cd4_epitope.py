# -*- coding: utf-8 -*-
"""
@Time ： 2024/3/1
@Auth ： shenlongchen
"""
from typing import Tuple

import torch.nn as nn
from torch.utils.data import DataLoader

from .losses import SupConLoss, TripletLoss
from ..utils.utils import *



class Trainer(object):
    """
    Trainer class for training models
    """

    def __init__(self, model, model_path, device, logger, **kwargs):
        self.device = device
        self.logger = logger
        self.model = model(**kwargs).to(self.device)
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.triplet_loss = TripletLoss(device, margin=0.6)
        self.contrastive_loss = SupConLoss(temperature=0.02)
        self.metric_fn = "Triplet"
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.training_state = {}

    def test_epitopes(self, test_loader, **kwargs):
        pred_instances, pred_bags, loss = self.predict(test_loader, valid=True, **kwargs)
        labels = test_loader.dataset.labels[test_loader.dataset.indices]
        peptide_protein = test_loader.dataset.peptide_contexts[test_loader.dataset.indices]
        auc_list = []
        for protein_ in list(set(peptide_protein)):
            # select the data about the protein
            protein_index = [i for i, x in enumerate(peptide_protein) if x == protein_]
            pred_instances_protein = pred_instances[protein_index]
            labels_protein = labels[protein_index]
            pred_auc_protein = roc_auc_score(labels_protein, pred_instances_protein)
            auc_list.append(pred_auc_protein)
        self.logger.info(" ============== Test AUC: {:.4f}".format(np.mean(auc_list)))
        return ""

    def predict(self, data_loader: DataLoader, valid=False, model_prefix="fine-tune-b", **kwargs):
        if not valid:
            if model_prefix is None or model_prefix == "":
                self.load_model(self.model_path.with_stem(f'{self.model_path.stem}'))
            else:
                self.load_model(self.model_path.with_stem(f'{self.model_path.stem}-{model_prefix}'))
        pred_instance = []
        pred_bag = []
        loss_sum = 0
        sample_num = 0
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in data_loader:
                labels = labels.to(self.device)
                inputs = tuple(x.to(self.device) for x in inputs)
                z_rep, instance_prob, bag_prob, bag_hat, att = self.model(inputs, **kwargs)
                pred_instance.append(instance_prob.cpu().numpy())
                pred_bag.append(bag_prob.cpu().numpy())
                if instance_prob.dim() == 0:
                    instance_prob = instance_prob.unsqueeze(0)
                if bag_prob.dim() == 0:
                    bag_prob = bag_prob.unsqueeze(0)
                loss_sa = self.bce_loss(instance_prob, labels)
                loss_sa2 = self.bce_loss(bag_prob, labels)
                loss_sum += loss_sa.item() * len(labels) + loss_sa2.item() * len(labels)
                sample_num += len(labels)

        return np.hstack(pred_instance), np.hstack(pred_bag), loss_sum / sample_num

    def save_model(self, save_path=None):
        path = save_path if save_path is not None else self.model_path
        save_dict = {'model_state_dict': self.model.state_dict()}
        torch.save(save_dict, path)

    def load_model(self, model_path=None):
        path = model_path if model_path is not None else self.model_path
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.logger.info(f'==== Model loaded from {path} ====')

    def valid_and_test(self, valid_loader, test_loader, epoch_pretrain, loss_info=None, prefix="", bag_test=True):
        if prefix == "":
            model_path = self.model_path.with_stem(f'{self.model_path.stem}')
        else:
            model_path = self.model_path.with_stem(f'{self.model_path.stem}-{prefix}')
        auc0_1_valid, aupr_valid, ppv_valid, loss_val = self.valid(valid_loader, epoch_pretrain,
                                                                   save_model_path=model_path, bag_test=bag_test)
        if test_loader is None:
            test_log_info = ""
        else:
            test_log_info, _ = self.test(test_loader, bag_test=bag_test)
        loss_info = f"{loss_info} -VAL {loss_val:.4f}"
        self.logger.info("Epoch-{}: {} - {} - Valid:[AUC0_1:{:.4f} AUPR:{:.4f} PPV:{:.4f}] {}".format(
            prefix, epoch_pretrain, loss_info, auc0_1_valid, aupr_valid, ppv_valid, test_log_info))

    def valid_and_test_without_save_model(self, valid_loader, test_loader):
        auc0_1_valid, aupr_valid, ppv_valid, loss_val = self.valid(valid_loader, save_model=False)
        if test_loader is None:
            test_log_info = ""
            pred_instances = None
        else:
            test_log_info, pred_instances = self.test(test_loader)
        self.logger.info("-- Loss:{:4f} -Valid:[AUC0_1: {:.4f} AUPR: {:.4f} PPV: {:.4f}] {}".format(
            loss_val, auc0_1_valid, aupr_valid, ppv_valid, test_log_info))
        return pred_instances

    def training_record_init(self):
        self.training_state['best'] = np.inf
        self.training_state['best_epoch'] = 0

    def save_model_every_n_epoch(self, epoch_idx, suffix, every_n_epoch=2, start_epoch=0):
        if epoch_idx % every_n_epoch == 0 and epoch_idx >= start_epoch:
            save_file = self.model_path.with_stem(f'{self.model_path.stem}-{suffix}-{epoch_idx}')
            save_dict = {
                'model_state_dict': self.model.state_dict(),
            }
            torch.save(save_dict, save_file)

    def train_epoch_ba(self, train_loader: DataLoader, optimizer, **kwargs) -> Tuple[float, float]:
        self.model.train()
        total_loss_ba = 0.0
        total_samples_ba = 0
        train_loader_ba, train_loader_sa = train_loader

        for inputs, labels in train_loader_ba:
            inputs = tuple(x.to(self.device) for x in inputs)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            z_rep_ba, instance_prob_ba, bag_prob_ba, bag_hat_ba, att_ba = self.model(inputs, **kwargs)

            loss = self.mse_loss(instance_prob_ba, labels)
            loss.backward()
            optimizer.step()

            total_loss_ba += loss.item() * len(labels)
            total_samples_ba += len(labels)

        average_loss_ba = total_loss_ba / total_samples_ba
        return average_loss_ba, 0

    def train_with_ba(self, train_loader: Tuple[DataLoader, DataLoader], valid_loader: DataLoader,
                      test_loader: DataLoader = None, pretrained_model_path: str = None, num_epochs=20, **kwargs):
        params = kwargs['opt_params']
        self.training_state['best'] = 0
        self.training_state['best_epoch'] = 0

        optimizer = torch.optim.Adam(self.model.parameters(), lr=params['lr'], weight_decay=params['wd'])
        self.load_model(model_path=pretrained_model_path)
        for epoch_idx in range(num_epochs):
            loss_ba, loss_sa = self.train_epoch_ba(train_loader, optimizer, **kwargs)
            loss_info = "Loss: BA {:.5f}, SA {:.5f}".format(loss_ba, loss_sa)

            auc0_1_valid, aupr_valid, ppv_valid = self.valid_ba_sa(valid_loader, epoch_idx)
            self.logger.info("Epoch: {} - {} - Valid :[AUC0_1:{:.4f} AUPR:{:.4f} PPV:{:.4f}]".format(
                epoch_idx, loss_info, auc0_1_valid, aupr_valid, ppv_valid))

            self.save_model_every_n_epoch(epoch_idx, suffix="ep")

        self.logger.info(f'Best Epoch: {self.training_state["best_epoch"]} ')

    def valid_ba_sa(self, valid_loader, epoch_idx, **kwargs):
        valid_loader_ba, valid_loader_sa = valid_loader

        scores_ba, pred_bags, _ = self.predict(valid_loader_ba, valid=True, **kwargs)
        labels_ba = valid_loader_ba.dataset.labels[valid_loader_ba.dataset.indices]

        scores_sa, pred_bags, _ = self.predict(valid_loader_sa, valid=True, **kwargs)
        labels_sa = valid_loader_sa.dataset.labels[valid_loader_sa.dataset.indices]

        auc0_1_valid_ba, aupr_valid_ba, ppv_valid_ba = calculate_all_metrics(labels_ba, scores_ba)
        auc0_1_valid_sa, aupr_valid_sa, ppv_valid_sa = calculate_all_metrics(labels_sa, scores_sa)
        ratio = 0.8
        auc0_1_valid = auc0_1_valid_ba * ratio + auc0_1_valid_sa * (1 - ratio)
        aupr_valid = aupr_valid_ba * ratio + aupr_valid_sa * (1 - ratio)
        ppv_valid = ppv_valid_ba * ratio + ppv_valid_sa * (1 - ratio)

        if aupr_valid > self.training_state['best']:
            self.save_model()
            self.training_state['best'] = aupr_valid
            self.training_state['best_epoch'] = epoch_idx
        return auc0_1_valid, aupr_valid, ppv_valid
