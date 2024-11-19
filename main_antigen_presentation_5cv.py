# -*- coding: utf-8 -*-
"""
@Time ： 2024/2/29
@Auth ： shenlongchen
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import click
from torch.utils.data.dataloader import DataLoader
from ImmuScope.datasets.datasets import MABags, SinInstanceBag
from ImmuScope.models.trainer_el import Trainer
from ImmuScope.models.ImmuScope import ImmuScope
from ImmuScope.utils.data_utils import *
from ImmuScope.utils.utils import *


def get_data_loader(data_cnf, mhc_name_seq, model_cnf, cv_):
    path_sa = data_cnf['5cv_sa'].replace('_.h5', f'_{cv_}_train.h5')
    path_ma = data_cnf['5cv_ma'].replace('_.h5', f'_{cv_}_train.h5')
    test_path = data_cnf['5cv_sa'].replace('_.h5', f'_{cv_}_test.h5')
    # split train data into train and valid
    train_idx, valid_idx = create_splits(path_sa, split_ratio=0.1, seed=model_cnf['seed'])
    train_loader_sa = DataLoader(
        SinInstanceBag(path_sa, mhc_name_seq, indices=train_idx),
        batch_size=model_cnf['train']['batch_size'] * 10, shuffle=True, drop_last=True)
    train_loader_ma = DataLoader(
        MABags(path_ma, mhc_name_seq, model_cnf['model']['bag_size']),
        batch_size=model_cnf['train']['batch_size'], shuffle=True, drop_last=True)
    train_loader = [train_loader_sa, train_loader_ma]

    valid_loader = DataLoader(
        SinInstanceBag(path_sa, mhc_name_seq, indices=valid_idx), batch_size=model_cnf['valid']['batch_size'])

    test_loader = DataLoader(
        SinInstanceBag(test_path, mhc_name_seq, indices=None), batch_size=model_cnf['test']['batch_size'])
    # select only positive instances
    test_loader_ma_pos = DataLoader(
        MABags(path_ma, mhc_name_seq, model_cnf['model']['bag_size'], onlyPositive=True),
        batch_size=model_cnf['test']['batch_size'])
    return train_loader, valid_loader, test_loader, test_loader_ma_pos


def test_immuscope_el(trainer, model_cnf, test_path, mhc_name_seq):
    test_loader = DataLoader(SinInstanceBag(test_path, mhc_name_seq, indices=None),
                             batch_size=model_cnf['test']['batch_size'])
    pred_instances, pred_bags, _ = trainer.predict(test_loader)
    return pred_instances, pred_bags, test_loader.dataset.labels, test_loader.dataset.mhc_names


def test_immuscope_el_with_loader(trainer, test_loader):
    pred_instances, pred_bags, _ = trainer.predict(test_loader)
    return (pred_instances, pred_bags, test_loader.dataset.labels[test_loader.dataset.indices],
            test_loader.dataset.mhc_names[test_loader.dataset.indices])


@click.command()
@click.option('-d', '--data-cnf', type=click.Path(exists=True), default="configs/data.yaml")
@click.option('-m', '--model-cnf', type=click.Path(exists=True), default="configs/ImmuScope-EL.yaml")
@click.option('-s', '--start-id', default=0)
@click.option('-n', '--num_models', default=10)
def main(data_cnf, model_cnf, start_id, num_models):
    logger, data_cnf, model_cnf = load_config_and_setup_logging(data_cnf=data_cnf, model_cnf=model_cnf)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = model_cnf["name"]
    model_path = Path(os.path.join(Path(model_cnf['path']), 'EL', f'{model_name}.pt'))
    res_path = Path(data_cnf['results']) / f'{model_name}'
    res_path.mkdir(parents=True, exist_ok=True)

    mhc_name_seq = get_mhc_name_seq(data_cnf['mhc_seq'])

    all_models_scores, all_models_labels = [], []

    for model_id in range(start_id, start_id + num_models):
        scores_all, labels_all, mhc_groups_all = [], [], []
        for cv_ in range(5):
            logger.info(f'------------- Start training model_id: {model_id} - cv: {cv_} ------------')

            path_ = model_path.with_stem(f'{model_path.stem}-{model_id}-CV{cv_}')

            loader = get_data_loader(data_cnf, mhc_name_seq, model_cnf, cv_)

            train_path_ms = Path(os.path.join(data_cnf["dataset_ms"], f"{model_name}_{model_id}_cv{cv_}_train.h5"))
            res_path_5cv = Path(res_path, f'{model_name}-5CV-{model_id}-cv{cv_}')

            trainer = Trainer(ImmuScope, model_path=path_, device=device, logger=logger, **model_cnf['model'])

            train_loader, valid_loader, test_loader, test_loader_ma_pos = loader
            trainer.train_immuscope_el(train_loader, valid_loader, test_loader, test_loader_ma_pos, train_path_ms,
                                       res_path=res_path_5cv, **model_cnf['train'])

            pred_instances, pred_bags, labels, mhc_names = test_immuscope_el_with_loader(trainer, test_loader)
            output_res(mhc_names, labels, pred_instances, res_path_5cv, logger=logger)

            scores_all.extend(pred_instances)
            labels_all.extend(labels)
            mhc_groups_all.extend(mhc_names)
            logger.info(f'------------- Finish training model_id: {model_id} - cv: {cv_} ------------\n')

        all_models_scores.append(np.array(scores_all))
        all_models_labels.append(np.array(labels_all))

        scores_test = np.mean(all_models_scores, axis=0)
        data_truth = np.mean(all_models_labels, axis=0)
        auc0_1_test, aupr_test, ppv_test = calculate_all_metrics(data_truth, scores_test)
        logger.info("All Mean Test: AUC0_1: {:.4f} - AUPR: {:.4f} - PPV: {:.4f}".format(
            auc0_1_test, aupr_test, ppv_test))
        logger.info(f'Finish test ---------------------------------- {model_id}')
        res_path_final = Path(res_path, f'{model_name}-5CV-{model_id}')
        output_res(mhc_groups_all, data_truth, scores_test, res_path_final, logger=logger)


if __name__ == '__main__':
    main()
