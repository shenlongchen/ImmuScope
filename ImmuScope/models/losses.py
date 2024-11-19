from __future__ import print_function

import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    def __init__(self, device, margin=1.0):
        super(TripletLoss, self).__init__()
        self.device = device
        self.margin = margin
        self.triplet_margin_loss = nn.TripletMarginLoss(margin=self.margin, p=2)

    def forward(self, anchor, positive, negative):
        return self.triplet_margin_loss(anchor, positive, negative)

    def pairwise_distances(self, x):
        """
        Compute the 2D matrix of distances between all the embeddings.

        Args:
        x: torch.Tensor, shape (batch_size, features), Input feature matrix.

        Returns:
        torch.Tensor, shape (batch_size, batch_size), Matrix of distances between embeddings.
        """
        x_norm = (x ** 2).sum(1).view(-1, 1)
        dist = x_norm + x_norm.T - 2.0 * torch.mm(x, x.T)
        dist = torch.clamp(dist, min=0.0)
        return dist

    def select_triplets(self, embeddings, labels):
        dist_matrix = self.pairwise_distances(embeddings).to(self.device)

        original_indices = torch.arange(len(embeddings)).to(self.device)
        pos_mask = labels == 1
        neg_mask = labels == 0

        anchors = embeddings[pos_mask]

        # Avoid selecting itself as the hardest positive
        dist_matrix.fill_diagonal_(-float('inf'))

        pos_original_indices = original_indices[pos_mask]
        pos_distances = dist_matrix[pos_mask][:, pos_mask]
        hardest_positives_indices = torch.argmax(pos_distances, dim=1)

        positives = embeddings[pos_original_indices[hardest_positives_indices]]

        if neg_mask.any():
            neg_original_indices = original_indices[neg_mask]
            neg_distances = dist_matrix[pos_mask][:, neg_mask]
            hardest_negatives_indices = torch.argmin(neg_distances, dim=1)
            negatives = embeddings[neg_original_indices[hardest_negatives_indices]]
        else:
            negatives = torch.empty((0, embeddings.size(1)), device=self.device)

        return anchors, positives, negatives


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, bag_label=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # loss = loss.view(anchor_count, batch_size).mean()

        mean_log_prob_pos_positives = mean_log_prob_pos * (labels.view(-1))
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos_positives
        loss = loss.sum() / (labels.sum()+1e-6)
        return loss
