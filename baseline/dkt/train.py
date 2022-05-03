import os

import torch
import wandb
from args import parse_args
from dkt import trainer
from dkt.dataloader import Preprocess
from dkt.utils import setSeeds

import json


def main(args):
    wandb.login()

    dump(args, 'args.json')
    
    setSeeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()

    train_data, valid_data = preprocess.split_data(train_data)

    # args.json 파일에서 wandb_mode를 조작해 디버깅 시에는 logging을 일시중지 시킬 수 있다.
    # 활성화는 "online", 비활성화는 "offline"
    wandb.init(project="dkt", config=vars(args), mode=args.wandb_mode)
    trainer.run(args, train_data, valid_data)

def dump(args, json_file):
    with open(json_file) as f:
        dict_json = json.load(f)
    args.__dict__.update(dict_json)


if __name__ == "__main__":
    args = parse_args(mode="train")
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
