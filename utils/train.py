import argparse
import ast
import copy
import itertools
import json
import multiprocessing as mp
import os
import re
import sys
import time
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from lightly.utils.scheduler import CosineWarmupScheduler
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss

import metrics
import snuffy
import snuffy_multiclass
from froc import mp_computeFROC_list_no_cache
from utils import (
    WEIGHT_INITS, OPTIMIZERS,
    pretty_print, print_table, replace_key_names, delete_files_for_epoch,
    to_wandb_format, NumpyFloatValuesEncoder,
    load_data, load_mil_data,
    dropout_patches, multi_label_roc, compute_pos_weight
)

print('Imports Finished.')

device = "cuda" if torch.cuda.is_available() else "cpu"

EMBEDDINGS_PATH = 'embeddings/'
CAMELYON16_REFERENCE = 'datasets/camelyon16/reference.csv'
CAMELYON16_MASK_PATH = 'datasets/camelyon16/masks'
SAVE_PATH = 'runs/'
ROC_PATH = 'roc/'
HISTOPATHOLOGY_DATASETS = ['camelyon16', 'tcga']
MIL_DATASETS = ['musk1', 'musk2', 'elephant']




def mp_thresholding(args):
    data, threshold, key = args
    data = list(filter(lambda x: x[0] > threshold, data))
    return key, data


class Trainer:
    def __init__(self, args):
        self.args = args
        self.milnet = self._get_milnet()
        self._load_init_weights()
        self.__is_criterion_set = False
        self.criterion = self._get_criterion()
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        self.froc_path = './froc'

    def _get_milnet(self) -> nn.Module:
        raise NotImplementedError

    def _get_criterion(self) -> Optional[_Loss]:
        # For MIL datasets, For all models (ours and DSMIL) (not ABMIL), criterion should be weighted BCE,
        # where weights are determined by train split labels.
        self.__is_criterion_set = not (
                self.args.dataset in MIL_DATASETS
        )
        return nn.BCEWithLogitsLoss()

    def _get_optimizer(self) -> optim.Optimizer:
        try:
            optimizer_cls = OPTIMIZERS[self.args.optimizer]
        except KeyError:
            raise Exception(f'Optimizer not found. Given: {self.args.optimizer}, Have: {OPTIMIZERS.keys()}')

        print(
            f'Optimizer {self.args.optimizer} with lr={self.args.lr}, betas={(self.args.betas[0], self.args.betas[1])}, wd={self.args.weight_decay}'
        )
        return optimizer_cls(
            params=self.milnet.parameters(),
            lr=self.args.lr,
            betas=(self.args.betas[0], self.args.betas[1]),
            weight_decay=self.args.weight_decay
        )

    def _get_scheduler(self):
        if self.args.scheduler == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self.optimizer,
                T_max=self.args.num_epochs,
                eta_min=self.args.eta_min
            )
        elif self.args.scheduler == 'cosinewarmup':
            return CosineWarmupScheduler(
                optimizer=self.optimizer,
                warmup_epochs=int(self.args.num_epochs / 20),
                max_epochs=self.args.num_epochs
            )
        else:
            print(f'Scheduler set to None')
            return None

    def _load_init_weights(self):
        try:
            # weight_init_i_part = WEIGHT_INITS[self.args.weight_init__weight_init_i__weight_init_b[1]] 
            weight_init_i_part = WEIGHT_INITS[self.args.weight_init__weight_init_i__weight_init_b[1]]
            weight_init_b_part = WEIGHT_INITS[self.args.weight_init__weight_init_i__weight_init_b[2]]
            # print(f'\n\nweight_init_i_part: {weight_init_f_part}') 
            print(f'\n\nweight_init_i_part: {weight_init_i_part}')
            print(f'weight_init_b_part: {weight_init_b_part}\n\n')
            # ------------------------------
            # self.milnet.feature_extractor.apply(weight_init_f_part)
            self.milnet.i_classifier.apply(weight_init_i_part)
            self.milnet.b_classifier.apply(weight_init_b_part)
            # ------------------------------

        except KeyError:
            if self.args.weight_init__weight_init_i__weight_init_b[0] is not None:
                raise Exception(
                    f'Weight init not found. Given: {self.args.weight_init__weight_init_i__weight_init_b[0]}, Have: {WEIGHT_INITS.keys()} ')

    @staticmethod
    def _should_calc_feats_metrics(data):
        """
        TCGA dataset doesn't have patch-level labels. Therefore, we can't calculate feat metrics for it.
        Official DSMIL-WSI features do not have patch-lebel labels either.
        """
        return data[2] is not None

    def train(self, data, cur_epoch):
        self.milnet.train()
        if data[2] is not None:
            data = shuffle(data[0], data[1], data[2], data[3])
        else:
            data = shuffle(data[0], data[1])
            data = data[0], data[1], None, None
            
        all_labels, all_feats, all_feats_labels, all_positions = data
        
        num_bags = len(all_labels)

        if device == 'cpu':
            Tensor = torch.FloatTensor
        else:
            Tensor = torch.cuda.FloatTensor

        total_loss = 0
        labels = all_labels
        predictions = []
        feat_labels = all_feats_labels
        feat_predictions = []

        if not self.__is_criterion_set:
            pos_weight = torch.tensor(compute_pos_weight(labels), device=device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight)
            self.__is_criterion_set = True

        for i in range(num_bags):
            bag_label, bag_feats = labels[i], all_feats[i]
            if self.args.l2normed_embeddings == 1:
                bag_feats = bag_feats / np.linalg.norm(bag_feats, axis=1, keepdims=True)
            bag_feats = dropout_patches(bag_feats, self.args.dropout_patch)
            # ------------------------
            bag_label = Variable(Tensor(np.array([bag_label])).to(device))  # .unsqueeze(dim=0)
            bag_feats = Variable(Tensor(np.array([bag_feats])).to(device))  # .unsqueeze(dim=0)
            # ------------------------
            bag_prediction, loss, attentions = self._run_model(bag_feats, bag_label)
            loss.backward()
            # ----------------------------------------
            step = num_bags * (cur_epoch - 1) + i
            self._after_run_model_in_training_mode(step=step, num_bags=num_bags, batch_idx=i)
            # ----------------------------------------
            total_loss = total_loss + loss.item()
            step_train_metrics = {'step_train_bag_loss': loss.item()}
            wandb.log(step_train_metrics)
            sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (i, num_bags, loss.item()))
            with torch.no_grad():
                predictions.extend([bag_prediction])
                if self._should_calc_feats_metrics(data):
                    feat_predictions.extend(attentions.cpu().numpy().squeeze())

        labels = np.array(labels)
        predictions = np.array(predictions)
        accuracy, auc_scores, _ = self._calc_metrics(labels, predictions)

        feats_accuracy, feats_auc_scores = None, None
        if self._should_calc_feats_metrics(data):
            feat_labels = list(itertools.chain(*feat_labels))  # convert a list of lists to a flat list
            feat_labels = np.array(feat_labels)
            feat_predictions = np.array(feat_predictions)
            feats_accuracy, feats_auc_scores, _ = self._calc_feats_metrics(
                feat_labels, feat_predictions
            )

        res = {
            'epoch_train_loss': total_loss / num_bags,
            'epoch_train_accuracy': accuracy,
            'epoch_train_aucs': auc_scores,
            'epoch_train_feat_accuracy': feats_accuracy,
            'epoch_train_feat_aucs': feats_auc_scores,
        }
        return res

    def valid(self, data, predefined_thresholds_optimal=None, predefined_feats_thresholds_optimal=None,
              plot_prefix=None, metric=None, mode='valid'):
        is_test = (mode != 'valid')
        self.milnet.eval()
        if data[2] is not None:
            data = shuffle(data[0], data[1], data[2], data[3], data[4])
        else:
            data = shuffle(data[0], data[1])
            data = data[0], data[1], None, None, None
        all_labels, all_feats, all_feats_labels, all_positions, all_image_names = data
        num_bags = len(all_labels)
        if device == 'cpu':
            Tensor = torch.FloatTensor
        else:
            Tensor = torch.cuda.FloatTensor

        # converts positions from string to int, might want to optimize later
        if is_test and self.args.dataset == 'camelyon16':
            reg = r'[^\d]*(\d+)[^\d]*(\d+)[^\d]*'
            all_positions_int = [
                [
                    tuple(map(int, re.search(reg, positions).group(1, 2)))
                    for positions in slide_positions
                ]
                for slide_positions in all_positions
            ]

        total_loss = 0
        labels = all_labels
        predictions = []
        feat_labels = all_feats_labels
        feat_predictions = []

        # for froc
        detections = []
        detections_dict = {}

        # for ece
        if (mode == 'test' and self.args.dataset == 'camelyon16'):
            self._calibration_cal(data[:4], metric)
        with torch.no_grad():
            for i in range(num_bags):
                bag_label, bag_feats = labels[i], all_feats[i]
                # ------------------------
                if self.args.l2normed_embeddings == 1:
                    bag_feats = bag_feats / np.linalg.norm(bag_feats, axis=1, keepdims=True)
                # ------------------------
                bag_label = Variable(Tensor(np.array([bag_label])).to(device))
                bag_feats = Variable(Tensor(np.array([bag_feats])).to(device))

                bag_prediction, loss, attentions = self._run_model(bag_feats, bag_label)
                if (is_test and self.args.dataset == 'camelyon16'):
                    slide_detections = [(float(prob), position[0] * 512 + 256, position[1] * 512 + 256)
                                        for position, prob in
                                        zip(all_positions_int[i], attentions.cpu().numpy().squeeze())]
                    detections.append(slide_detections)

                total_loss = total_loss + loss.item()
                step_validation_metrics = {
                    'step_valid_bag_loss': loss.item()
                }
                wandb.log(step_validation_metrics)
                sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, num_bags, loss.item()))
                predictions.extend([bag_prediction])
                if self._should_calc_feats_metrics(data):
                    feat_predictions.extend(attentions.cpu().numpy().squeeze())

        accuracy, auc_scores, thresholds_optimal = self._calc_metrics(
            labels, predictions, predefined_thresholds_optimal
        )
        if self.args.for_roc_curve:
            print(f'\nPredictions: {predictions}')
            print(f'Labels: {labels}')
            roc_base_dir = os.path.join(ROC_PATH, self.args.roc_run_name)
            os.makedirs(roc_base_dir, exist_ok=True)
            labels_predictions_f_path = os.path.join(roc_base_dir, f'{self.args.roc_run_epoch}.npz')
            np.savez(labels_predictions_f_path, labels=labels, predictions=predictions, )
            print(f'\n\nSaved at {labels_predictions_f_path}')

        feats_accuracy, feats_auc_scores, feats_thresholds_optimal = None, None, None
        if self._should_calc_feats_metrics(data):
            feat_labels = list(itertools.chain(*feat_labels))  # convert a list of lists to a flat list
            feat_labels = np.array(feat_labels)
            feat_predictions = np.array(feat_predictions)
            feats_accuracy, feats_auc_scores, feats_thresholds_optimal = \
                self._calc_feats_metrics(feat_labels, feat_predictions, predefined_feats_thresholds_optimal)

        res = {
            'epoch_valid_loss': total_loss / num_bags,
            'epoch_valid_accuracy': accuracy,
            'epoch_valid_aucs': auc_scores,
            'epoch_valid_thresholds_optimal': thresholds_optimal,
            'epoch_valid_feat_accuracy': feats_accuracy,
            'epoch_valid_feat_aucs': feats_auc_scores,
            'epoch_valid_feats_thresholds_optimal': feats_thresholds_optimal,
        }

        if self._should_calc_feats_metrics(data) and is_test and self.args.dataset == 'camelyon16':
            with mp.Pool(self.args.num_processes) as pool:
                res_ = pool.map(mp_thresholding, [(data, feats_thresholds_optimal[0], key) for data, key in
                                                  zip(detections, all_image_names)])
            for key, data in res_:
                detections_dict[key] = data
            challenge_froc_score = mp_computeFROC_list_no_cache(
                CAMELYON16_REFERENCE,
                CAMELYON16_MASK_PATH,
                detections_dict,
                os.path.join(self.froc_path, 'results'),
                False,
                True,
                5,  # mask level
                all_image_names,
                self.froc_path,
                plot_prefix,
                self.args.num_processes
            )
            res['epoch_valid_challenge_froc_score'] = challenge_froc_score

        return res

    def _calibration_cal(self, data, metric):
        self.milnet.eval()
        all_labels, all_feats, all_feats_labels, all_positions = data
        num_bags = len(all_labels)
        if device == 'cpu':
            Tensor = torch.FloatTensor
        else:
            Tensor = torch.cuda.FloatTensor

        total_loss = 0
        labels = all_labels
        predictions = []
        feat_labels = all_feats_labels
        feat_predictions = []
        # ------------------
        softmaxes = np.zeros((num_bags, self.args.num_classes))
        with torch.no_grad():
            for i in range(num_bags):
                bag_label, bag_feats = labels[i], all_feats[i]
                # ------------------------
                if self.args.l2normed_embeddings == 1:
                    bag_feats = bag_feats / np.linalg.norm(bag_feats, axis=1, keepdims=True)
                # ------------------------
                bag_label = Variable(Tensor(np.array([bag_label])).to(device))
                bag_feats = Variable(Tensor(np.array([bag_feats])).to(device))
                bag_prediction, loss, attentions = self._run_model(bag_feats, bag_label)
                if (self.args.num_classes == 1):
                    softmaxes[i] = bag_prediction
                else:
                    for j in range(self.args.num_classes):
                        softmaxes[i, j] = bag_prediction[j]
        labels_np = np.array(labels)
        # --------------------------
        ece_criterion = metrics.ECELoss()
        # --------------------------
        ece_error = ece_criterion.loss(softmaxes, labels_np, 0.5, n_bins=self.args.bins, logits=False)
        wandb.log({f"calibration/ECE/{metric}": ece_error})

    def test(self, data, thresholds_optimal, feats_thresholds_optimal, plot_prefix, metric):
        res = self.valid(
            data,
            thresholds_optimal,
            feats_thresholds_optimal,
            plot_prefix=plot_prefix,
            metric=metric,
            mode='test'
        )  # solved
        res = replace_key_names(d=res, old_term='valid', new_term='test')
        return res

    def _run_model(self, bag_feats, bag_label) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _after_run_model_in_training_mode(self, step, num_bags, batch_idx):
        if self.args.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.milnet.parameters(), max_norm=self.args.clip_grad)

        self.optimizer.step()
        self.optimizer.zero_grad()

    def _calc_metrics(self, labels, predictions, predefined_thresholds_optimal=None):
        assert len(labels) == len(predictions), \
            f"Number of predictions ({len(predictions)}) and labels ({len(labels)}) do not match"

        num_bags = len(labels)
        labels = np.array(labels)
        predictions = np.array(predictions)
        auc_scores, _, thresholds_optimal = multi_label_roc(labels, predictions, self.args.num_classes)

        if predefined_thresholds_optimal is not None:
            thresholds_optimal = predefined_thresholds_optimal

        if self.args.num_classes == 1:
            class_prediction_bag = copy.deepcopy(predictions)
            class_prediction_bag[predictions >= thresholds_optimal[0]] = 1
            class_prediction_bag[predictions < thresholds_optimal[0]] = 0
            predictions = class_prediction_bag
            labels = np.squeeze(labels)
        else:
            for i in range(self.args.num_classes):
                class_prediction_bag = copy.deepcopy(predictions[:, i])
                class_prediction_bag[predictions[:, i] >= thresholds_optimal[i]] = 1
                class_prediction_bag[predictions[:, i] < thresholds_optimal[i]] = 0
                predictions[:, i] = class_prediction_bag

        bag_score = 0
        for i in range(num_bags):
            bag_score += np.array_equal(labels[i], predictions[i])
        accuracy = bag_score / num_bags

        return accuracy, auc_scores, thresholds_optimal

    def _calc_feats_metrics(self, feats_labels, feats_predictions, predefined_thresholds_optimal=None):
        auc_scores, _, thresholds_optimal = multi_label_roc(
            feats_labels, feats_predictions, self.args.num_classes, for_feats=True
        )

        if predefined_thresholds_optimal is not None:
            thresholds_optimal = predefined_thresholds_optimal

        accuracy = accuracy_score(
            feats_labels,
            (feats_predictions >= thresholds_optimal[0]).astype(int)
        )

        return accuracy, auc_scores, thresholds_optimal


class Runner:
    def __init__(self, args, trainer: Trainer):
        self.args = args
        self.trainer = trainer
        self._set_dirs()

        if self.args.dataset in HISTOPATHOLOGY_DATASETS:
            if self.args.embedding == 'official':
                self.train_data, self.valid_data, self.test_data = self._get_official_data()
            else:
                self.train_data, self.valid_data, self.test_data = self._get_data()
        elif self.args.dataset in MIL_DATASETS:
            self.train_data, self.valid_data, self.test_data = load_mil_data(args)

        print(
            f'Num Bags'
            f' (Train: {len(self.train_data[0])})'
            f' (Valid: {len(self.valid_data[0])})'
            f' (Test: {len(self.test_data[0])})'
        )

    def _set_dirs(self):
        self.save_path = os.path.join(SAVE_PATH, self.args.dataset, wandb.run.name)
        self.trainer.froc_path = self.save_path
        os.makedirs(self.save_path, exist_ok=True)

    def _get_data(self):
        """
        bag_df:         [column_0]                  [column_1]
                        path_to_bag_feats_csv       label
        """
        path_prefix = os.path.join('.', EMBEDDINGS_PATH, self.args.dataset, self.args.embedding)

        bags_csv = os.path.join(path_prefix, self.args.dataset + '.csv')
        bags_df = pd.read_csv(bags_csv)
        # --------------------------------
        if self.args.dataset == 'camelyon16':
            train_df, valid_df, test_df = self._get_dataframe_splits_by_folder(bags_df, path_prefix)
        elif self.args.dataset == 'tcga':
            train_df, valid_df, test_df = self._get_dataframe_splits_by_folder(bags_df, path_prefix)

        print(f'Num Bags (Train: {len(train_df)}) (Valid: {len(valid_df)}) (Test: {len(test_df)})')

        train_data = self._load_split_data(train_df, 'train')
        valid_data = self._load_split_data(valid_df, 'valid')
        test_data = self._load_split_data(test_df, 'test')

        return train_data, valid_data, test_data

    def _get_official_data(self):
        bags_csv = os.path.join(EMBEDDINGS_PATH, self.args.dataset, 'official', f'{self.args.dataset.capitalize()}.csv')
        bags_df = pd.read_csv(bags_csv)
        train_df, valid_df, test_df = self._get_dataframe_splits_by_args(bags_df)
        train_df = shuffle(train_df).reset_index(drop=True)
        valid_df = shuffle(valid_df).reset_index(drop=True)
        test_df = shuffle(test_df).reset_index(drop=True)

        train_data = self._load_split_data(train_df, 'train')
        valid_data = self._load_split_data(valid_df, 'valid')
        test_data = self._load_split_data(test_df, 'test')

        return train_data, valid_data, test_data

    def _get_dataframe_splits_by_folder(self, bags_df, path_prefix):
        split_names = ['train', 'valid', 'test']
        dataframe_splits = (
            bags_df[
                bags_df['0'].str.startswith(f'{path_prefix}/{split_name}')
            ] for split_name in split_names
        )
        return dataframe_splits

    def _get_dataframe_splits_by_args(self, bags_df):
        train_df = bags_df.iloc[0:int(len(bags_df) * (1 - self.args.split)), :]
        valid_df = bags_df.iloc[int(len(bags_df) * (1 - self.args.split)):, :]
        valid_df, test_df = (
            valid_df.iloc[0:len(valid_df) // 2, :],
            valid_df.iloc[len(valid_df) // 2:, :]
        )
        return train_df, valid_df, test_df

    def _load_split_data(self, split_path, split_name):
        print(f'Loading {split_name} data... (mp={self.args.use_mp})...')
        start_time = time.time()
        data = load_data(split_path, self.args)
        print(f'DONE (Took {(time.time() - start_time):.1f}s)')
        return data

    def _log_initial_metrics(self):
        initial_metrics = self.trainer.valid(self.valid_data)  # solved
        print(f'\nInitial Metrics')
        print_table(initial_metrics)
        initial_metrics_file_path = os.path.join(self.save_path, f'initial_results.txt')
        with open(initial_metrics_file_path, 'w') as f:
            json.dump(initial_metrics, f, cls=NumpyFloatValuesEncoder)
        wandb.save(initial_metrics_file_path)

    def _load_epoch_model(self, epoch: int):
        model_save_path = os.path.join(self.save_path, f'{epoch}.pth')
        log_save_path = os.path.join(self.save_path, f'thresholds_{epoch}.txt')
        single_weight_parameter_save_path = os.path.join(self.save_path, f'single_weight_parameter_{epoch}')

        self.trainer.milnet.load_state_dict(torch.load(model_save_path), strict=True)

        with open(log_save_path, 'r') as f:
            epoch_valid_metrics = json.load(f)
        thresholds_optimal = np.asarray(a=eval(epoch_valid_metrics['thresholds_optimal']), dtype=np.float32)
        report = f'Using thresholds_optimal: {thresholds_optimal}'

        feats_thresholds_optimal = epoch_valid_metrics['feats_thresholds_optimal']
        if feats_thresholds_optimal is not None:
            feats_thresholds_optimal = np.asarray(a=eval(feats_thresholds_optimal), dtype=np.float32)
            report += f' feats_thresholds_optimal: {feats_thresholds_optimal}'

        if hasattr(self.trainer, 'single_weight_parameter'):
            report += f' single_weight_parameter: {self.trainer.single_weight_parameter}'
            self.trainer.single_weight_parameter = torch.load(single_weight_parameter_save_path)
        print(report)
        return thresholds_optimal, feats_thresholds_optimal

    def _save_epoch_model(
            self,
            thresholds_optimal: list,
            epoch: int,
            auc: float,
            feats_thresholds_optimal=None,
            report_prefix: str = None,
    ):
        model_save_path = os.path.join(self.save_path, f'{epoch}.pth')
        log_save_path = os.path.join(self.save_path, f'thresholds_{epoch}.txt')
        single_weight_parameter_save_path = os.path.join(self.save_path, f'single_weight_parameter_{epoch}')

        model_report = f'model saved at: {model_save_path}'
        torch.save(self.trainer.milnet.state_dict(), model_save_path)

        thresholds_report = f'threshold: {str(thresholds_optimal)}'
        with open(log_save_path, 'w') as f:
            json.dump({
                'auc': auc,
                'thresholds_optimal': str(thresholds_optimal),
                'feats_thresholds_optimal': str(
                    feats_thresholds_optimal
                ) if feats_thresholds_optimal is not None else None
            }, f)

        single_weight_parameter_report = ''
        if hasattr(self.trainer, 'single_weight_parameter'):
            single_weight_parameter_report = f'single_weight_parameter: {self.trainer.single_weight_parameter}'
            torch.save(self.trainer.single_weight_parameter, single_weight_parameter_save_path)

        should_log_report = report_prefix is not None
        if should_log_report:
            print(f'\t[{report_prefix}] {model_report} {thresholds_report} {single_weight_parameter_report}')

    def run(self):
        best_auc_epochs = self.run_train()
        self.run_test(best_auc_epochs)
        self.clean_up(best_auc_epochs)

    def run_train(self):
        best_auc = 0
        best_auc_epochs = []

        self._log_initial_metrics()
        for epoch in range(1, self.args.num_epochs + 1):
            start_train_epoch_time = time.time()
            epoch_train_metrics = self.trainer.train(self.train_data, epoch)
            start_valid_epoch_time = time.time()
            epoch_valid_metrics = self.trainer.valid(self.valid_data)
            end_valid_epoch_time = time.time()

            valid_aucs = epoch_valid_metrics['epoch_valid_aucs']
            thresholds_optimal = epoch_valid_metrics['epoch_valid_thresholds_optimal']
            feats_thresholds_optimal = epoch_valid_metrics['epoch_valid_feats_thresholds_optimal']
            epoch_train_time = int(start_valid_epoch_time - start_train_epoch_time)
            epoch_valid_time = int(end_valid_epoch_time - start_valid_epoch_time)

            wandb.log({
                'epoch': epoch,
                'epoch_train_time': epoch_train_time,
                'epoch_valid_time': epoch_valid_time,
                **to_wandb_format(epoch_train_metrics),
                **to_wandb_format(epoch_valid_metrics),
            })
            print(
                '\rEpoch [%d/%d] time %.1fs train loss: %.4f test loss: %.4f,'
                ' thresholds_optimal: %s, feats_thresholds_optimal: %s, accuracy: %.4f, AUC: ' % (
                    epoch,
                    self.args.num_epochs,
                    epoch_train_time + epoch_valid_time,
                    epoch_train_metrics['epoch_train_loss'],
                    epoch_valid_metrics['epoch_valid_loss'],
                    epoch_valid_metrics['epoch_valid_thresholds_optimal'],
                    epoch_valid_metrics['epoch_valid_feats_thresholds_optimal'],
                    epoch_valid_metrics['epoch_valid_accuracy']
                ) +
                '|'.join('class-{0}>>{1:.4f}'.format(*k) for k in enumerate(valid_aucs))
            )

            if self.trainer.scheduler is not None:
                self.trainer.scheduler.step()

            current_auc = valid_aucs[0]

            report_prefix = ''
            if current_auc >= best_auc:
                report_prefix += '[best auc]'
                if current_auc > best_auc:
                    best_auc_epochs = []
                best_auc = current_auc
                best_auc_epochs.append(epoch)

            self._save_epoch_model(
                thresholds_optimal, epoch, current_auc, feats_thresholds_optimal, report_prefix=report_prefix
            )

        train_metrics = {
            'best_auc': best_auc,
            'best_auc_epochs': best_auc_epochs,
        }
        with open(os.path.join(self.save_path, 'train_metrics.json'), 'w') as f:
            json.dump(train_metrics, f)
        print(f'Train Metrics')
        print(json.dumps(train_metrics) + '\n')

        earliest_best_auc_epoch = min(best_auc_epochs, default=None)

        return [earliest_best_auc_epoch]

    def run_test(self, best_auc_epochs):
        earliest_best_auc_epoch = min(best_auc_epochs, default=None)

        last_epoch = self.args.num_epochs
        special_epochs = [
            (earliest_best_auc_epoch, 'best_auc'),
            (last_epoch, 'last_epoch'),
        ]
        special_epochs = [x for x in special_epochs if x[0] is not None]
        for epoch, plot_prefix in special_epochs:
            start_test_epoch_time = time.time()
            thresholds_optimal, feats_thresholds_optimal = self._load_epoch_model(epoch)
            epoch_test_metrics = self.trainer.test(self.test_data, thresholds_optimal, feats_thresholds_optimal,
                                                   plot_prefix=plot_prefix, metric=plot_prefix)
            res = replace_key_names(d=epoch_test_metrics, old_term='epoch', new_term=plot_prefix)
            epoch_test_time = int(time.time() - start_test_epoch_time)
            wandb.log({
                'epoch': epoch,
                'epoch_test_time': epoch_test_time,
                **to_wandb_format(res),
            })
            print('\r', end='')
            print_table({
                'epoch_test_time': epoch_test_time,
                **epoch_test_metrics
            })
            print()

    def clean_up(self, best_auc_epochs):
        last_epoch = self.args.num_epochs

        special_epochs = list(
            set(best_auc_epochs + [last_epoch])
        )
        special_epochs = [x for x in special_epochs if x is not None]

        wanted_epochs = []
        for epoch in special_epochs:
            wanted_epochs.extend(list(range(epoch - 0, epoch + 1)))

        for epoch in range(1, self.args.num_epochs + 1):
            if epoch not in wanted_epochs:
                delete_files_for_epoch(self.save_path, epoch)


class SmallWeightTrainer(Trainer):
    def __init__(self, args):
        self.args = args
        self.single_weight_parameter = self._get_single_weight_parameter()
        super().__init__(args)

    def _get_single_weight_parameter(self):
        single_weight_parameter = torch.tensor(0.5, requires_grad=self.args.soft_average, device=device)
        print('single_weight_parameter.requires_grad:', single_weight_parameter.requires_grad)
        single_weight_parameter.data.clamp_(0, 1)
        return single_weight_parameter

    def _get_optimizer(self) -> optim.Optimizer:
        try:
            optimizer_cls = OPTIMIZERS[self.args.optimizer]
        except KeyError:
            raise Exception(f'Optimizer not found. Given: {self.args.optimizer}, Have: {OPTIMIZERS.keys()}')

        print(
            f'Optimizer {self.args.optimizer} with lr={self.args.lr}, betas={(self.args.betas[0], self.args.betas[1])}, wd={self.args.weight_decay}'
        )
        return optimizer_cls(
            params=[
                {'params': self.single_weight_parameter, 'lr': self.args.lr * self.args.single_weight__lr_multiplier},
                {'params': self.milnet.parameters()}
            ],
            lr=self.args.lr,
            betas=(self.args.betas[0], self.args.betas[1]),
            weight_decay=self.args.weight_decay
        )

    def _run_model(self, bag_feats, bag_label) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        ins_prediction, bag_prediction, attentions = self.milnet(bag_feats)
        if len(ins_prediction.shape) == 2:
            max_prediction, _ = torch.max(ins_prediction, 0)
        else:
            max_prediction, _ = torch.max(ins_prediction, 1)

        bag_loss = self.criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
        max_loss = self.criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
        loss = self.single_weight_parameter * bag_loss + (1 - self.single_weight_parameter) * max_loss

        with torch.no_grad():
            bag_prediction = (
                    (1 - self.single_weight_parameter) * torch.sigmoid(max_prediction) +
                    self.single_weight_parameter * torch.sigmoid(bag_prediction)
            ).squeeze().cpu().numpy()

        return bag_prediction, loss, ins_prediction

    def train(self, data, cur_epoch):
        res = super().train(data, cur_epoch)
        return res

    def _after_run_model_in_training_mode(self, step, num_bags, batch_idx):
        super()._after_run_model_in_training_mode(step, num_bags, batch_idx)
        self.single_weight_parameter.data.clamp_(0, 1)

    def __str__(self):
        return f'Single_Weight__sa{self.args.soft_average}'

