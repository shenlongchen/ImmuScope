# -*- coding: utf-8 -*-
"""
@Time ： 2024/2/29
@Auth ： shenlongchen
@Description : ImmuScope TEST
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import click
from torch.utils.data.dataloader import DataLoader
from ImmuScope.datasets.datasets import SinInstanceBag
from ImmuScope.models.trainer_cd4_epitope import Trainer
from ImmuScope.models.ImmuScope import ImmuScope
from ImmuScope.utils.data_utils import *
from ImmuScope.utils.utils import *


@click.command()
@click.option('-d', '--data-cnf', type=click.Path(exists=True), default="configs/data.yaml")
@click.option('-m', '--model-cnf', type=click.Path(exists=True), default="configs/ImmuScope.yaml")
@click.option('-s', '--start-id', default=0)
@click.option('-n', '--num_models', default=10)
def main(data_cnf, model_cnf, start_id, num_models):
    logger, data_cnf, model_cnf = load_config_and_setup_logging(data_cnf=data_cnf, model_cnf=model_cnf)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = model_cnf["name"]

    all_model_path = Path(os.path.join(Path(model_cnf['path']), 'CD4', f'{model_name}.pt'))

    mhc_name_seq = get_mhc_name_seq(data_cnf['mhc_seq'])

    pred_instances_models = []
    labels_models = []
    peptide_protein_models = []
    for model_id in range(start_id, start_id + num_models):
        saved_model_path = all_model_path.with_stem(f'{all_model_path.stem}-{model_id}')
        trainer = Trainer(ImmuScope, model_path=saved_model_path, device=device, logger=logger, **model_cnf['model'])

        test_loader = DataLoader(
            SinInstanceBag(data_cnf['test'], mhc_name_seq, indices=None), batch_size=model_cnf['test']['batch_size'])

        pred_instances, _, _ = trainer.predict(test_loader, model_prefix=f"")
        peptide_protein = test_loader.dataset.peptide_contexts
        labels = test_loader.dataset.labels

        median_auc, mean_auc, avg_auc, res_df = calculate_auc_base_protein(peptide_protein, pred_instances, labels)
        logger.info('|**---------Model {}--- TEST: Median AUC: {:.4f}; Mean AUC: {:.4f}; AVG AUC: {:.4f}---------**|'.
                    format(model_id, median_auc, mean_auc, avg_auc))
        pred_instances_models.append(pred_instances)
        labels_models.append(labels)
        peptide_protein_models.append(peptide_protein)

    logger.info(f'-----------------Average-----------------')
    protein_ids = peptide_protein_models[0]
    pred_mean = np.mean(np.array(pred_instances_models), axis=0)
    label_mean = np.mean(np.array(labels_models), axis=0)
    df_pred = pd.DataFrame({'protein': protein_ids, 'pred': pred_mean, 'label': label_mean})
    df_pred.to_csv(f'results/ImmuScope-CD4/results_pred_protein_avg.csv', index=False)
    median_auc, mean_auc, avg_auc, res_df = calculate_auc_base_protein(protein_ids, pred_mean, label_mean)
    res_df.to_csv(f'results/ImmuScope-CD4/results_auc_protein_avg.csv', index=False)
    logger.info('|**========== TEST: Median AUC: {:.4f}; Mean AUC: {:.4f}; AVG AUC: {:.4f}==========**|'.
                format(median_auc, mean_auc, avg_auc))


if __name__ == '__main__':
    main()
