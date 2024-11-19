# -*- coding: utf-8 -*-
"""
@Time ： 2024/3/1
@Auth ： shenlongchen
"""

import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ImmuScope.models.losses import SupConLoss, TripletLoss
from ImmuScope.utils.utils import *

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


    def test_imm(self, test_loader, **kwargs):
        pred_instances, scores_bag, loss = self.predict(test_loader, valid=True, **kwargs)
        labels = test_loader.dataset.labels[test_loader.dataset.indices]
        mhc_names = test_loader.dataset.mhc_names[test_loader.dataset.indices]
        auc_group = calculate_group_auc(labels, pred_instances, mhc_names, min_pos_num=1)
        auc_all = calculate_auc(labels, pred_instances)
        log_info = ("-- Test: [AUC-Group: {:.4f} - AUC-All: {:.4f}]".format(auc_group, auc_all))

        return log_info, pred_instances


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


    def save_model_every_n_epoch(self, epoch_idx, suffix, every_n_epoch=2, start_epoch=0):
        if epoch_idx % every_n_epoch == 0 and epoch_idx >= start_epoch:
            save_file = self.model_path.with_stem(f'{self.model_path.stem}-{suffix}-{epoch_idx}')
            save_dict = {
                'model_state_dict': self.model.state_dict(),
            }
            torch.save(save_dict, save_file)


    def train_with_imm(self, train_loader: DataLoader, valid_loader: DataLoader, test_loader: DataLoader = None,
                       pretrained_model_path: str = None, num_epochs=20, **kwargs):
        params = kwargs['opt_params']
        self.training_state['best'] = 0
        self.training_state['best_epoch'] = 0
        optimizer = torch.optim.Adam(self.model.parameters(), lr=params['lr'], weight_decay=params['wd'])
        self.load_model(model_path=pretrained_model_path)
        for epoch_idx in range(num_epochs):
            loss_imm = self.train_epoch_imm(train_loader, optimizer, **kwargs)
            loss_info = "Loss: IMM {:.5f} ".format(loss_imm)

            auc_valid = self.valid_imm(valid_loader, epoch_idx)

            if test_loader is None:
                test_log_info = ""
            else:
                test_log_info, pred_instances = self.test_imm(test_loader)
            self.logger.info("Epoch: {} - {} - Valid-AUC:{:.4f}  {}".format(
                epoch_idx, loss_info, auc_valid, test_log_info))

            self.save_model_every_n_epoch(epoch_idx, suffix="ep")

        self.logger.info(f'Best IMM Epoch: {self.training_state["best_epoch"]} ')

    def train_epoch_imm(self, train_loader: DataLoader, optimizer, **kwargs) -> float:
        self.model.train()
        total_loss_imm = 0.0
        total_samples_imm = 0

        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = tuple(x.to(self.device) for x in inputs)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            z_rep_imm, instance_prob_imm, bag_prob_imm, bag_hat_imm, att_imm = self.model(inputs, **kwargs)
            loss = self.bce_loss(instance_prob_imm, labels)
            loss.backward()
            optimizer.step()

            total_loss_imm += loss.item() * len(labels)
            total_samples_imm += len(labels)
        average_loss_imm = total_loss_imm / total_samples_imm
        return average_loss_imm

    def valid_imm(self, valid_loader, epoch_idx, **kwargs):
        scores_imm, pred_bags, _ = self.predict(valid_loader, valid=True, **kwargs)
        labels_imm = valid_loader.dataset.labels[valid_loader.dataset.indices]

        auc_score = calculate_auc(labels_imm, scores_imm)

        if auc_score > self.training_state['best']:
            self.save_model()
            self.training_state['best'] = auc_score
            self.training_state['best_epoch'] = epoch_idx
        return auc_score
