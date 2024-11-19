# -*- coding: utf-8 -*-
"""
@Time ： 2024/2/29
@Auth ： shenlongchen
@Description : Fine-tune the model based on ImmuScope-EL model with immunogenicity data
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import click
from torch.utils.data.dataloader import DataLoader
from ImmuScope.datasets.datasets import SinInstanceBag
from ImmuScope.models.trainer_immunogenicity import Trainer
from ImmuScope.models.ImmuScope import ImmuScope
from ImmuScope.utils.data_utils import *
from ImmuScope.utils.utils import *


def train_immuscope_im(trainer, pretrain_model, mhc_name_seq, data_cnf, model_cnf, logger, random_state=2024):
    logger.info(f'Start training model {trainer.model_path}')
    train_path_imm = data_cnf['train_imm']
    test_path_imm = data_cnf['test_imm']

    train_ba_idx, valid_ba_idx = create_splits(train_path_imm, split_ratio=0.1, seed=random_state)

    train_loader = DataLoader(
        SinInstanceBag(train_path_imm, mhc_name_seq, indices=train_ba_idx), batch_size=model_cnf['train']['batch_size'],
        shuffle=True)

    valid_loader = DataLoader(
        SinInstanceBag(train_path_imm, mhc_name_seq, indices=valid_ba_idx), batch_size=model_cnf['valid']['batch_size'])

    test_loader = DataLoader(
        SinInstanceBag(test_path_imm, mhc_name_seq, indices=None), batch_size=model_cnf['test']['batch_size'])

    trainer.train_with_imm(train_loader, valid_loader, test_loader, pretrained_model_path=pretrain_model,
                           **model_cnf['train'])
    logger.info(f'Finish training model {trainer.model_path}')


@click.command()
@click.option('-d', '--data-cnf', type=click.Path(exists=True), default="configs/data.yaml")
@click.option('-m', '--model-cnf', type=click.Path(exists=True), default="configs/ImmuScope-IM.yaml")
@click.option('-s', '--start-id', default=0)
@click.option('-n', '--num_models', default=10)
def main(data_cnf, model_cnf, start_id, num_models):
    logger, data_cnf, model_cnf = load_config_and_setup_logging(data_cnf=data_cnf, model_cnf=model_cnf)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = model_cnf["name"]
    result_path = Path.joinpath(Path(data_cnf['results']), model_name)

    pretrained_model_path = Path(os.path.join(Path(model_cnf['path']), 'EL', 'ImmuScope-EL.pt'))
    imm_model_path = Path(os.path.join(Path(model_cnf['path']), 'IM', f'{model_name}.pt'))

    mhc_name_seq = get_mhc_name_seq(data_cnf['mhc_seq'])

    pred_instances_models = []
    labels_models = []

    test_path_imm = data_cnf['test_imm']
    test_loader = DataLoader(
        SinInstanceBag(test_path_imm, mhc_name_seq, indices=None), batch_size=model_cnf['test']['batch_size'])

    for model_id in range(start_id, start_id + num_models):
        pretrain_model = pretrained_model_path.with_stem(f'{pretrained_model_path.stem}-{model_id}-fine-tune-b')
        imm_path = imm_model_path.with_stem(f'{imm_model_path.stem}-{model_id}')
        trainer = Trainer(ImmuScope, model_path=imm_path, device=device, logger=logger, **model_cnf['model'])

        train_immuscope_im(trainer, pretrain_model, mhc_name_seq, data_cnf, model_cnf, logger=logger)

        pred_instances, _, _ = trainer.predict(test_loader, model_prefix="")
        labels = test_loader.dataset.labels
        auc_group = calculate_group_auc(labels, pred_instances, test_loader.dataset.mhc_names, min_pos_num=1)
        auc_all = calculate_auc(labels, pred_instances)
        logger.info(f'|**TEST: AUC_GROUP: {auc_group:.4f}**|')
        logger.info(f'|**TEST: AUC_ALL: {auc_all:.4f}**|')

        save_path = Path.joinpath(result_path, f'results_{model_name}_{model_id}.csv')
        peptide_test_data = restore_peptide_sequences(test_loader.dataset.peptide_embedding.reshape(-1, 27))
        res_df = pd.DataFrame({'mhc': test_loader.dataset.mhc_names, 'peptide': peptide_test_data,
                               'label': labels, 'pred': pred_instances})
        res_df.to_csv(save_path, index=False)

        pred_instances_models.append(pred_instances)
        labels_models.append(labels)

    logger.info(f'-----------------Average-----------------')
    save_path = Path.joinpath(result_path, f'results_{model_name}_avg.csv')
    peptide_test_data = restore_peptide_sequences(test_loader.dataset.peptide_embedding.reshape(-1, 27))
    pred_instances_models = np.array(pred_instances_models).mean(axis=0)
    labels_models = np.array(labels_models).mean(axis=0)
    res_df = pd.DataFrame({'mhc': test_loader.dataset.mhc_names, 'peptide': peptide_test_data,
                           'label': labels_models, 'pred': pred_instances_models})
    res_df.to_csv(save_path, index=False)

    auc_group = calculate_group_auc(labels_models, pred_instances_models, test_loader.dataset.mhc_names, min_pos_num=1)
    auc_all = calculate_auc(labels_models, pred_instances_models)
    logger.info('|**========== TEST: AUC_GROUP: {:.4f} =========**|'.format(auc_group))
    logger.info('|**========== TEST: AUC_ALL: {:.4f} =========**|'.format(auc_all))


if __name__ == '__main__':
    main()