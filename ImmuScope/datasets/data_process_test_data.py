# -*- coding: utf-8 -*-
"""
@Time ： 2024/02/20
@Auth ： shenlongchen
@Description: convert test data to h5py file for model evaluation
"""

from pathlib import Path
from ruamel.yaml import YAML
from ..utils.data_utils import *

data_cnf = "configs/data.yaml"
h5py_dir = "data/imm"
yaml = YAML(typ='safe')
data_cnf = yaml.load(Path(data_cnf))
mhc_name_seq = get_mhc_name_seq(data_cnf['mhc_seq'])
data_id_mhc_name = get_data_id_mhc_name(data_cnf['dataset_id_mhc'])

imm_dataset_path = "data/imm/immu_test.txt"
imm_data = []
with open(imm_dataset_path) as fp:
    for line in fp:
        peptide_seq, score, mhc_name, context = line.split()
        imm_data.append((mhc_name, peptide_seq, context, score))
print("len(train_data):", len(imm_data))

h5_file = h5py_dir + f"/imm_test.h5"
save_mhc_peptide_h5py(imm_data, h5_file, label_dtype=np.int32)
