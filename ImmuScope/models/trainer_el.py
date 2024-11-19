# -*- coding: utf-8 -*-
"""
@Time ： 2024/3/1
@Auth ： shenlongchen
"""
from itertools import cycle
from typing import Tuple

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..datasets.datasets import SinInstanceBag
from .losses import SupConLoss, TripletLoss
from ..utils.utils import *
from ..utils.data_utils import *


class Trainer(object):
    """
    Trainer for ImmuScope-EL
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

    def train_epoch_instance(self, train_loader: DataLoader, optimizer, **kwargs):
        self.model.train()
        train_loss = 0.0
        total_samples = 0

        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = tuple(x.to(self.device) for x in inputs)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            z_rep, instance_prob, bag_prob, bag_hat, att = self.model(inputs, **kwargs)
            loss = self.bce_loss(instance_prob, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            total_samples += len(labels)

        train_loss /= total_samples
        loss_info = "Loss: SA_I {:.4f}".format(train_loss)
        return loss_info

    def process_batch(self, inputs_ma, inputs_sa, metric_fn, total_loss, total_samples, **kwargs):
        inputs_ma, labels_ma = inputs_ma
        inputs_sa, labels_sa = inputs_sa
        inputs_ma = tuple(x.to(self.device) for x in inputs_ma)
        inputs_sa = tuple(x.to(self.device) for x in inputs_sa)

        labels_ma = labels_ma[0].to(self.device)
        labels_sa = labels_sa.to(self.device)

        z_rep_ma, instance_prob_ma, bag_prob_ma, bag_hat_ma, att_ma = self.model(inputs_ma, **kwargs)
        z_rep_sa, instance_prob_sa, bag_prob_sa, bag_hat_sa, att_sa = self.model(inputs_sa, **kwargs)

        loss_ma_bag = self.bce_loss(bag_prob_ma, labels_ma)
        loss_sa_ins = self.bce_loss(instance_prob_sa, labels_sa)
        loss_sa_bag = self.bce_loss(bag_prob_sa, labels_sa)

        z_rep_sa = F.normalize(z_rep_sa, dim=1, p=2)
        if metric_fn == 'Contrastive':
            z_rep_sa = z_rep_sa.unsqueeze(1)
            loss_metric = self.contrastive_loss(z_rep_sa, labels_sa) * 0.1
        elif metric_fn == 'Triplet':
            anchor, positive, negative = self.triplet_loss.select_triplets(z_rep_sa, labels_sa)
            loss_metric = self.triplet_loss(anchor, positive, negative) * 0.1
        else:
            raise ValueError(f"Metric function {metric_fn} is not supported.")

        total_loss['sa_instance'] += loss_sa_ins.item() * len(labels_sa)
        total_loss['sa_bag'] += loss_sa_bag.item() * len(labels_sa)
        total_loss['ma_bag'] += loss_ma_bag.item() * len(labels_ma)
        total_loss['sa_metric'] += loss_metric.item() * len(labels_sa)

        total_samples['sa'] += len(labels_sa)
        total_samples['ma'] += len(labels_ma)

        return loss_ma_bag + loss_sa_bag + loss_sa_ins + loss_metric

    def train_epoch_mil_with_metric_loss(self, train_loader: Tuple[DataLoader, DataLoader], metric_fn, **kwargs) -> str:
        self.model.train()
        total_loss = {'sa_instance': 0.0, 'sa_bag': 0.0, 'ma_bag': 0.0, 'sa_metric': 0.0}
        total_samples = {'sa': 0, 'ma': 0}

        train_loader_sa, train_loader_ma = train_loader
        pbar = tqdm(total=max(len(train_loader_ma), len(train_loader_sa)), desc="Training")

        loader_ma = cycle(train_loader_ma) if len(train_loader_ma) < len(train_loader_sa) else train_loader_ma
        loader_sa = cycle(train_loader_sa) if len(train_loader_sa) < len(train_loader_ma) else train_loader_sa

        for inputs_ma, inputs_sa in zip(loader_ma, loader_sa):
            self.optimizer.zero_grad()
            loss = self.process_batch(inputs_ma, inputs_sa, metric_fn, total_loss, total_samples, **kwargs)
            loss.backward()
            self.optimizer.step()
            pbar.update(1)

        pbar.close()

        avg_loss = {k: total_loss[k] / total_samples[k.split('_')[0]] for k in total_loss.keys()}
        return "Loss[SA_I {:.4f} SA_B {:.4f} MA {:.4f} {} {:.4f}]".format(avg_loss['sa_instance'],
                                                                          avg_loss['sa_bag'], avg_loss['ma_bag'],
                                                                          metric_fn[:3], avg_loss['sa_metric'])

    def train_immuscope_el(self, train_loader: Tuple[DataLoader, DataLoader], valid_loader: DataLoader,
                           test_loader: DataLoader = None, test_loader_ma_pos: DataLoader = None, train_path_ms=None,
                           pretrain_epochs=15, sample_incorporate_epochs=25, fine_tune_epochs=3, batch_size=128,
                           res_path=None,
                           **kwargs):
        params = kwargs['opt_params']

        # -------- Training with SA+MA --------
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params['lr_pre'], weight_decay=params['wd_pre'])
        self.training_record_init()
        for epoch_pretrain in range(pretrain_epochs):
            loss_info = self.train_epoch_mil_with_metric_loss(train_loader, metric_fn=self.metric_fn, **kwargs)
            self.valid_and_test(valid_loader, test_loader, epoch_pretrain, loss_info, prefix="pretrain")
            self.save_model_every_n_epoch(epoch_pretrain, suffix="pretrain")
        self.logger.info(f'Best Pretrain Epoch: {self.training_state["best_epoch"]}\n')

        # -------- Training with pseudo positive data and SA data --------
        self.training_record_init()
        self.load_model(self.model_path.with_stem(f'{self.model_path.stem}-pretrain'))
        self.valid_and_test_without_save_model(valid_loader, test_loader)

        optimizer_instance = torch.optim.Adam(self.model.parameters(), lr=params['lr_si'], weight_decay=params['wd_si'])
        for epoch in range(sample_incorporate_epochs):
            percent = calculate_percentage(epoch, sample_incorporate_epochs, initial_percentage=92, final_percentage=88)

            train_loader_sa, _ = train_loader
            self.build_pseudo_pos_dataset_connect_sa_dataset(train_loader_sa, test_loader_ma_pos, train_path_ms,
                                                             percent=percent)
            # fine-tuning Classifier
            ms_dataloader_for_instance = DataLoader(
                SinInstanceBag(train_path_ms, train_loader[0].dataset.mhc_name_seq), batch_size=batch_size,
                shuffle=True, drop_last=True)

            loss_info = self.train_epoch_instance(ms_dataloader_for_instance, optimizer_instance, **kwargs)
            self.valid_and_test(valid_loader, test_loader, epoch, loss_info, prefix="fine-tune-a", bag_test=False)
            # fine-tuning MIL
            ms_dataloader = DataLoader(
                SinInstanceBag(train_path_ms, train_loader[0].dataset.mhc_name_seq), batch_size=batch_size * 10,
                shuffle=True, drop_last=True)
            train_loader_self_iter = [ms_dataloader, train_loader[1]]

            self.train_epoch_mil_with_metric_loss(train_loader_self_iter, metric_fn=self.metric_fn, **kwargs)
            self.valid_and_test_without_save_model(valid_loader, test_loader)
            self.save_model_every_n_epoch(epoch, suffix="fine-tune-a")

        self.logger.info(f'Best Epoch of fine-tuning: {self.training_state["best_epoch"]}\n')

        # -------- Training with sa data and pseudo label based on pretrained model--------
        self.training_record_init()
        self.load_model(self.model_path.with_stem(f'{self.model_path.stem}-pretrain'))
        optimizer_instance = torch.optim.Adam(self.model.parameters(), lr=params['lr_finetune'],
                                              weight_decay=params['wd_finetune'])
        for epoch_idx in range(fine_tune_epochs):
            ms_dataloader = DataLoader(
                SinInstanceBag(train_path_ms, train_loader[0].dataset.mhc_name_seq), batch_size=batch_size,
                shuffle=True, drop_last=True)

            loss_info = self.train_epoch_instance(ms_dataloader, optimizer_instance, **kwargs)
            self.valid_and_test(valid_loader, test_loader, epoch_idx, loss_info, prefix="fine-tune-b", bag_test=False)
            self.save_model_every_n_epoch(epoch_idx, suffix="fine-tune-b")
        self.logger.info(f'Best Epoch of fine-tuning based on pretrained model: {self.training_state["best_epoch"]}\n')

    def valid(self, valid_loader, epoch_idx=0, save_model_path=None, save_model=True, bag_test=True, **kwargs):
        scores, scores_bag, loss = self.predict(valid_loader, valid=True, **kwargs)
        labels = valid_loader.dataset.labels[valid_loader.dataset.indices]
        if bag_test:
            auc0_1_valid_bag, aupr_valid_bag, ppv_valid_bag = calculate_all_metrics(labels, scores_bag)
            self.logger.debug(" ============== Valid Bag: AUC0_1: {:.4f} AUPR: {:.4f} PPV: {:.4f}".format(
                auc0_1_valid_bag, aupr_valid_bag, ppv_valid_bag))
        auc0_1_valid, aupr_valid, ppv_valid = calculate_all_metrics(labels, scores)

        if loss < self.training_state['best'] and save_model:
            self.save_model(save_model_path)
            self.training_state['best'] = loss
            self.training_state['best_epoch'] = epoch_idx
        return auc0_1_valid, aupr_valid, ppv_valid, loss

    def test(self, test_loader, bag_test=True, **kwargs):
        pred_instances, scores_bag, loss = self.predict(test_loader, valid=True, **kwargs)
        labels = test_loader.dataset.labels[test_loader.dataset.indices]
        auc0_1_all, aupr_all, ppv_all = calculate_all_metrics(labels, pred_instances)
        auc0_1_group, aupr_group, ppv_group = calculate_group_metrics(labels, pred_instances,
                                                                      test_loader.dataset.mhc_names[
                                                                          test_loader.dataset.indices])
        if bag_test:
            auc0_1_group_bag, aupr_group_bag, ppv_group_bag = calculate_group_metrics(labels, scores_bag,
                                                                                      test_loader.dataset.mhc_names[
                                                                                          test_loader.dataset.indices])
            self.logger.debug(" ============== Test Bag: AUC0_1: {:.4f} AUPR: {:.4f} PPV: {:.4f}".format(
                auc0_1_group_bag, aupr_group_bag, ppv_group_bag))

        log_info = ("-- Test: AUPR: {:.4f} -Group [AUC0_1: {:.4f}, AUPR: {:.4f}, PPV: {:.4f}]".format(
            aupr_all, auc0_1_group, aupr_group, ppv_group))

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

    def predict_ma_bag(self, data_loader: DataLoader, valid=False, **kwargs):
        if not valid:
            self.load_model()
        pred_instance = []
        pred_bag = []
        pred_att = []

        total_true_predict_ma = 0
        total_positive_ma = 0

        self.model.eval()
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = tuple(x.to(self.device) for x in inputs)
                labels = labels[0].to(self.device)
                z_rep, instance_prob, bag_prob, bag_hat, att = self.model(inputs, **kwargs)
                pred_instance.append(instance_prob.cpu().numpy())
                pred_bag.append(bag_prob.cpu().numpy())
                pred_att.append(att.squeeze().cpu().numpy())
                # the label the same as the instance label and the label is positive
                total_true_predict_ma += true_predict_positive(bag_prob, labels)
                total_positive_ma += labels.sum().cpu().numpy()

        self.logger.info(f"MA-bag: True Predict Positive: {total_true_predict_ma} Total Positive: {total_positive_ma}, "
                         f"Positive Rate: {total_true_predict_ma / total_positive_ma:.4f}")

        return np.hstack(pred_instance), np.hstack(pred_bag), np.vstack(pred_att)

    def predict_ma_bag_drop(self, data_loader: DataLoader, valid=False, **kwargs):
        if not valid:
            self.load_model()
        pred_instance_mean = []
        pred_bag_mean = []
        pred_att_mean = []
        pred_instance_std = []
        pred_bag_std = []
        pred_att_std = []

        total_true_predict_ma = 0
        total_positive_ma = 0

        self.model.eval()
        # dropout
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
        # set the dropout to train

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = tuple(x.to(self.device) for x in inputs)
                labels = labels[0].to(self.device)

                uncertainty_pred_instance = []
                uncertainty_pred_bag = []
                uncertainty_pred_att = []
                for forward_index in range(10):
                    z_rep, instance_prob, bag_prob, bag_hat, att = self.model(inputs, **kwargs)
                    uncertainty_pred_instance.append(instance_prob)
                    uncertainty_pred_bag.append(bag_prob)
                    uncertainty_pred_att.append(att.squeeze())

                uncertainty_pred_instance = torch.stack(uncertainty_pred_instance, dim=0)
                uncertainty_pred_bag = torch.stack(uncertainty_pred_bag, dim=0)
                uncertainty_pred_att = torch.stack(uncertainty_pred_att, dim=0)

                # mean
                pred_instance_mean.append(torch.mean(uncertainty_pred_instance, dim=0).cpu().numpy())
                pred_bag_mean.append(torch.mean(uncertainty_pred_bag, dim=0).cpu().numpy())
                pred_att_mean.append(torch.mean(uncertainty_pred_att, dim=0).cpu().numpy())
                # std
                pred_instance_std.append(torch.std(uncertainty_pred_instance, dim=0).cpu().numpy())
                pred_bag_std.append(torch.std(uncertainty_pred_bag, dim=0).cpu().numpy())
                pred_att_std.append(torch.std(uncertainty_pred_att, dim=0).cpu().numpy())

                total_true_predict_ma += true_predict_positive(torch.mean(uncertainty_pred_bag, dim=0), labels)
                total_positive_ma += labels.sum().cpu().numpy()

        return np.hstack(pred_instance_mean), np.hstack(pred_bag_mean), np.vstack(pred_att_mean), \
            np.hstack(pred_instance_std), np.hstack(pred_bag_std), np.vstack(pred_att_std)

    def save_model(self, save_path=None):
        path = save_path if save_path is not None else self.model_path
        save_dict = {'model_state_dict': self.model.state_dict()}
        torch.save(save_dict, path)

    def load_model(self, model_path=None):
        path = model_path if model_path is not None else self.model_path
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.logger.info(f'==== Model loaded from {path} ====')

    def build_pseudo_pos_dataset_connect_sa_dataset(self, train_loader_sa, test_loader_ma_pos, train_path_ms,
                                                    percent=90):

        # build pseudo labels from ma data
        mhc_names = test_loader_ma_pos.dataset.mhc_names
        peptide_embedding = test_loader_ma_pos.dataset.peptide_embedding
        peptide_contexts = test_loader_ma_pos.dataset.peptide_contexts
        labels_ma = test_loader_ma_pos.dataset.labels
        # with drop
        self.logger.info(f"-----Current ratio: {percent:.2f}% ------")
        (pred_instances_mean, pred_bags_mean, pred_att_mean,
         pred_instances_std, pred_bags_std, pred_att_std) = self.predict_ma_bag_drop(
            test_loader_ma_pos, valid=True)

        high_confidence_idx = self.get_high_confidence_index(pred_instances_mean, pred_instances_std,
                                                             pred_att_mean, pred_att_std, labels_ma,
                                                             pred_bags_mean, percent_instance=96, percent_att=percent)

        peptide_embedding_hc = peptide_embedding[high_confidence_idx]
        peptide_contexts_hc = peptide_contexts[high_confidence_idx]
        mhc_names_hc = mhc_names[high_confidence_idx]

        pred_labels_hc = pred_instances_mean[high_confidence_idx]
        pred_labels_hc = np.ones_like(pred_labels_hc)

        # save as dataset
        # 1. get dataset from train_loader_sa
        mhc_names_sa = train_loader_sa.dataset.mhc_names
        peptide_embedding_sa = np.squeeze(train_loader_sa.dataset.peptide_embedding, axis=1)

        peptide_contexts_sa = train_loader_sa.dataset.peptide_contexts
        labels_sa = train_loader_sa.dataset.labels

        self.logger.info(
            f"[ sum of labels_sa: {sum(labels_sa)}, pseudo pos samples: {len(mhc_names_hc)}, "
            f"rate of added pos samples: {len(mhc_names_hc) / sum(labels_sa):.4f} ]")

        # 2. combine the two datasets
        mhc_names_comb = np.concatenate((mhc_names_sa, mhc_names_hc), axis=0)
        peptide_embedding_comb = np.concatenate((peptide_embedding_sa, peptide_embedding_hc), axis=0)
        peptide_contexts_comb = np.concatenate((peptide_contexts_sa, peptide_contexts_hc), axis=0)
        labels_comb = np.concatenate((labels_sa, pred_labels_hc), axis=0)

        mhc_names_comb = np.array(mhc_names_comb, dtype=object)
        peptide_contexts_comb = np.array(peptide_contexts_comb, dtype=object)
        save_h5py_file(train_path_ms, labels_comb, mhc_names_comb, peptide_contexts_comb,
                       peptide_embedding_comb, label_dtype=np.float32)

    def get_high_confidence_index(self, pred_instances_mean, pred_instances_std,
                                  pred_att_mean, pred_att_std, labels_ma, pred_bags_mean,
                                  percent_instance=95, percent_att=90):

        # percentile of att
        att_mean_threshold = np.percentile(pred_att_mean.reshape(-1), percent_att)
        att_std_threshold = np.percentile(pred_att_std.reshape(-1), 80)

        # percentile of instance
        instance_mean_threshold = np.percentile(pred_instances_mean, percent_instance)
        instance_std_threshold = np.percentile(pred_instances_std, 40)

        bag_num = pred_att_mean.shape[0]
        bag_size = pred_att_mean.shape[1]

        high_confidence_idx = []
        high_confidence_values = []
        count_positive_label = 0

        count_add = 0
        count_del = 0

        for i in range(bag_num):
            bag_start_idx = i * bag_size
            bag_end_idx = (i + 1) * bag_size

            labels_ma_i = labels_ma[bag_start_idx: bag_end_idx]
            # positive bag
            if max(labels_ma_i) == 1:
                count_positive_label += 1
                # predict true positive
                if pred_bags_mean[i] < 0.5:
                    continue
                pred_instances_mean_i = pred_instances_mean[bag_start_idx: bag_end_idx]
                pred_instances_std_i = pred_instances_std[bag_start_idx: bag_end_idx]
                pred_att_mean_i = pred_att_mean[i]
                pred_att_std_i = pred_att_std[i]

                for idx in range(bag_size):
                    if (pred_att_mean_i[idx] >= att_mean_threshold and pred_att_std_i[idx] <= att_std_threshold
                            and labels_ma_i[idx] == 1):
                        if not (pred_instances_mean_i[idx] >= instance_mean_threshold
                                and pred_instances_std_i[idx] <= instance_std_threshold):
                            high_confidence_idx.append(bag_start_idx + idx)
                            high_confidence_values.append(pred_instances_mean_i[idx])
                            count_add += 1
                        else:
                            count_del += 1
        self.logger.debug(f"Add: {count_add} Del: {count_del}")
        return high_confidence_idx

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


def true_predict_positive(bag_prob, labels):
    bag_prob_array = bag_prob.cpu().detach().numpy()
    labels_array = labels.cpu().detach().numpy()
    return np.sum((bag_prob_array > 0.5) & (labels_array == 1))


def calculate_percentage(current_epoch, total_epoch, initial_percentage=95, final_percentage=80):
    decrement_per_iteration = (initial_percentage - final_percentage) / (total_epoch - 1)
    current_percentage = initial_percentage - current_epoch * decrement_per_iteration
    return current_percentage
