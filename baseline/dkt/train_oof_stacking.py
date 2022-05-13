from dkt.trainer import *

from sklearn.linear_model import LinearRegression
from dkt.utils import setSeeds
from copy import deepcopy


import os

import torch
import wandb
from args import parse_args
from dkt import trainer
from dkt.dataloader import Preprocess
from dkt.utils import setSeeds

import json


def main(args):
    # wandb.login()
    dump(args, 'args.json')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()

    train_data, valid_data = preprocess.split_data(train_data)
    
    # args 생성.
    args_list = [deepcopy(args) for _ in range(5)]
    
    # seed 설정 
    setSeeds(args.seed)

    # oof stacking ensemble
    # Train
    stacking = Stacking(Trainer())
    meta_model = LinearRegression()
    meta_model, models_list, S_train, target = stacking.train(meta_model, args_list, train_data)

    # Test
    stacking = Stacking(Trainer())
    test_predict, S_test = stacking.test(meta_model, models_list, temp_test_data)
    test_target = trainer.get_target(temp_test_data)

    # 테스트셋 성능
    stack_test_auc, stack_test_acc = get_metric(test_target, test_predict)
    stack_test_auc, stack_test_acc

def dump(args, json_file):
    with open(json_file) as f:
        dict_json = json.load(f)
    args.__dict__.update(dict_json)


if __name__ == "__main__":
    args = parse_args(mode="train")
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
