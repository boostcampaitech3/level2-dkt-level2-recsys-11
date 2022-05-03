import os

import torch
from args import parse_args
from dkt import trainer
from dkt.dataloader import Preprocess

import json

def main(args):
    dump(args, 'args.json')
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    preprocess = Preprocess(args)
    preprocess.load_test_data(args.test_file_name)
    test_data = preprocess.get_test_data()

    trainer.inference(args, test_data)

def dump(args, json_file):
    with open(json_file) as f:
        dict_json = json.load(f)
    args.__dict__.update(dict_json)

if __name__ == "__main__":
    args = parse_args(mode="train")
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
