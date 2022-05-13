import os
from sklearn.model_selection import StratifiedKFold, KFold

import torch
import wandb
from args import parse_args
from dkt import trainer
from dkt.dataloader import Preprocess
from dkt.utils import setSeeds


def main(args):
    wandb.login()

    setSeeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()

    # train_data, valid_data = preprocess.split_data(train_data)
    # skf = StratifiedKFold(n_splits=args.n_kfold)
    kf = KFold(n_splits=args.n_kfold)

    wandb.init(project="DKT", entity="iksadnorth", config=vars(args))
    
    # for train_index, test_index in skf.split(train_data, y):
    for train_index, test_index in kf.split(train_data):
        train_dataset, valid_dataset = train_data[train_index], train_data[test_index]
        trainer.run(args, train_dataset, valid_dataset)


if __name__ == "__main__":
    args = parse_args(mode="train")
    
    args.n_kfold = 5
    args.model = 'saint'
    args.n_epochs = 50
    args.patience = 1000
    args.best_auc = -1
    
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
