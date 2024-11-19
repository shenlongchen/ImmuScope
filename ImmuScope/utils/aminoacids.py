# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/15
@Auth ： shenlongchen
"""

import numpy as np

ACIDS = '0-ACDEFGHIKLMNPQRSTVWY'
A_A_LETTER = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
AA_NUM = len(A_A_LETTER)
AAINDEX = dict()
for i in range(len(A_A_LETTER)):
    AAINDEX[A_A_LETTER[i]] = i + 1
INVALID_ACIDS = {'U', 'O', 'B', 'Z', 'J', 'X', '*'}


def to_onehot(seq, max_len=20, start=0):
    onehot = np.zeros((max_len, 21), dtype=np.float32)
    seq_len = min(max_len, len(seq))
    for index in range(start, start + seq_len):
        onehot[index, AAINDEX.get(seq[index - start], 0)] = 1
    return onehot
