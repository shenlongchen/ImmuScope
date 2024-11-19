# -*- coding: utf-8 -*-
"""
@Time ： 2024/3/1
@Auth ： shenlongchen
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.data_utils import ACIDS
from ..utils.utils import truncated_normal_


class EmbeddingLayer(nn.Module):
    """
        embedding layer for peptide and mhc
    """

    def __init__(self, *, emb_size, vocab_size=len(ACIDS), padding_idx=0, peptide_pad=3, mhc_len=34, **kwargs):
        super(EmbeddingLayer, self).__init__()
        self.peptide_emb = nn.Embedding(vocab_size, emb_size)
        self.mhc_emb = nn.Embedding(vocab_size, emb_size)
        self.peptide_pad, self.padding_idx, self.mhc_len = peptide_pad, padding_idx, mhc_len

    def forward(self, peptide_x, mhc_x, *args, **kwargs):
        masks = peptide_x[:, self.peptide_pad: peptide_x.shape[1] - self.peptide_pad] != self.padding_idx
        return self.peptide_emb(peptide_x.long()), self.mhc_emb(mhc_x.long()), masks

    def reset_parameters(self):
        nn.init.uniform_(self.peptide_emb.weight, -0.1, 0.1)
        nn.init.uniform_(self.mhc_emb.weight, -0.1, 0.1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.layer_norm1 = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.layer_norm2 = nn.LayerNorm(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.layer_norm1(out.transpose(1, 2)).transpose(1, 2)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.layer_norm2(out.transpose(1, 2)).transpose(1, 2)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class AttnNetGated(nn.Module):
    def __init__(self, L=128, D=64, dropout=False, n_classes=1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(AttnNetGated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x, batch_size=1):
        a = self.attention_a(x)
        b = self.attention_b(x)
        a = a.view(batch_size, -1, a.shape[1])
        b = b.view(batch_size, -1, b.shape[1])
        att = a.mul(b)
        att = self.attention_c(att)  # N x n_classes
        return att  # batch_size, bag_size, n_classes


class ResConvModule(EmbeddingLayer):
    def __init__(self, *, conv_num, conv_size, conv_off, dropout=0.5, **kwargs):
        super(ResConvModule, self).__init__(**kwargs)
        self.conv = nn.ModuleList(IConv(cn, cs, self.mhc_len) for cn, cs in zip(conv_num, conv_size))
        self.conv_ln = nn.ModuleList(nn.LayerNorm(cn) for cn in conv_num)
        self.cross_attn = nn.MultiheadAttention(embed_dim=16, num_heads=4)
        self.cross_conv1d = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=9, stride=1, padding=0)
        self.cross_ln = nn.LayerNorm(32)

        self.conv_off = conv_off
        self.dropout = nn.Dropout(dropout)

        self.relu = nn.ReLU()

        layers = [2, 2, 2]
        block = ResidualBlock
        self.in_channels = 96
        self.layer1 = self._make_layer(block, 96, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.reset_parameters()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                Transpose(),
                nn.LayerNorm(out_channels),
                Transpose(),
            )
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, peptide_x, mhc_x, pooling=True, **kwargs):
        peptide_x, mhc_x, masks = super(ResConvModule, self).forward(peptide_x, mhc_x)
        peptide_ = peptide_x[:, 3: -3, :].transpose(0, 1)
        mhc_ = mhc_x.transpose(0, 1)
        output, attn_weight = self.cross_attn(query=peptide_, key=mhc_, value=mhc_, key_padding_mask=None)
        output = output.transpose(0, 1)
        att_embedding = self.relu(self.cross_conv1d(output.transpose(1, 2)))
        att_embedding = self.cross_ln(att_embedding.transpose(1, 2)).transpose(1, 2)

        conv_out = torch.cat(
            [conv_ln(F.relu(conv(peptide_x[:, off: peptide_x.shape[1] - off], mhc_x).transpose(1, 2))).transpose(1, 2)
             for conv, conv_ln, off in zip(self.conv, self.conv_ln, self.conv_off)], dim=1)

        conv_out = self.dropout(conv_out)
        att_embedding = self.dropout(att_embedding)
        conv_out = torch.cat([conv_out, att_embedding], dim=1)
        masks = masks[:, -conv_out.shape[2]:]

        x = self.layer1(conv_out)
        if pooling:
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.avg_pool(x)
            output_embedding = torch.flatten(x, 1)  # batch_size x bag_size, 128
            return output_embedding
        else:
            feature = torch.sigmoid(torch.mean(conv_out[:, :, :], dim=1)).masked_fill(~masks, -np.inf)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.avg_pool(x)
            output_embedding = torch.flatten(x, 1)  # batch_size x bag_size, 128
            return feature, output_embedding

    def reset_parameters(self):
        super(ResConvModule, self).reset_parameters()
        for conv, conv_ln in zip(self.conv, self.conv_ln):
            conv.reset_parameters()
            conv_ln.reset_parameters()
            nn.init.normal_(conv_ln.weight.data, mean=1.0, std=0.002)


class ImmuScope(nn.Module):
    def __init__(self, *, conv_num, conv_size, conv_off, dropout=0.25, **kwargs):
        super(ImmuScope, self).__init__()
        self.encoder = ResConvModule(conv_num=conv_num, conv_size=conv_size, conv_off=conv_off, dropout=dropout,
                                     **kwargs)

        self.mlp_cl = nn.Sequential(nn.Linear(256, 32),
                                    nn.ReLU(),
                                    nn.Linear(32, 1))
        self.mlp_mil = nn.Sequential(nn.Linear(256, 32),
                                     nn.ReLU(),
                                     nn.Linear(32, 1))

        self.gat = AttnNetGated(L=256, D=128, dropout=True, n_classes=1)

    def forward(self, inputs, **kwargs):
        peptide_x, mhc_x = inputs
        batch_size = peptide_x.shape[0]
        peptide_x = peptide_x.view(-1, peptide_x.shape[2])
        mhc_x = mhc_x.view(-1, mhc_x.shape[2])
        z_rep = self.encoder(peptide_x, mhc_x, **kwargs)

        # calculate single instance-level prediction
        instance_prob = torch.sigmoid(self.mlp_cl(z_rep).squeeze())  # batch_size X bag_size

        # gated attention
        att = self.gat(z_rep, batch_size)
        att = att.view(batch_size, -1, 1)
        att = torch.transpose(att, 1, 2)
        att = F.softmax(att, dim=2)  # batch_size, 1, bag_size

        bag_output = z_rep.view(batch_size, -1, z_rep.shape[1])
        bag_prob = torch.bmm(att, bag_output)
        bag_prob = bag_prob.squeeze(1)
        bag_prob = torch.sigmoid(self.mlp_mil(bag_prob).squeeze())  # batch_size
        bag_hat = torch.ge(bag_prob, 0.5).float()  # batch_size
        return z_rep, instance_prob, bag_prob, bag_hat, att


class Transpose(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)


class IConv(nn.Module):
    def __init__(self, out_channels, kernel_size, mhc_len=34, stride=1):
        super(IConv, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_channels, kernel_size, mhc_len))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.stride, self.kernel_size = stride, kernel_size
        self.reset_parameters()

    def forward(self, peptide_x, mhc_x):
        bs = peptide_x.shape[0]
        kernel = F.relu(torch.einsum('nld,okl->nodk', mhc_x, self.weight))
        outputs = F.conv1d(peptide_x.transpose(1, 2).reshape(1, -1, peptide_x.shape[1]),
                           kernel.contiguous().view(-1, *kernel.shape[-2:]), stride=self.stride, groups=bs)
        return outputs.view(bs, -1, outputs.shape[-1]) + self.bias[:, None]

    def reset_parameters(self):
        truncated_normal_(self.weight, std=0.02)
        nn.init.zeros_(self.bias)
