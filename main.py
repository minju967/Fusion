import argparse
from cgi import test
import math
import os
import random
import sys
import time
import wandb
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

FILE = Path(__file__).resolve() 
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from Model.models import FusionNet
from Tools.dataset import FusionDataset
from Tools.Trainer import FusionmodelTrainer

parser = argparse.ArgumentParser()

parser.add_argument("-obj_path", type=str, default='/root/MINJU/Fusion/DATA/OBJ')
parser.add_argument("-img_path", type=str, default='/root/MINJU/Fusion/DATA/Image')
parser.add_argument("-bs", type=int, help="Batch size for the second stage", default=8)
parser.add_argument("-cnn_name", "--cnn_name", type=str, help="cnn model name", default="vgg16")
parser.add_argument("-pretrain", "--CNN_pretrain", type=bool, help="pre-train", default=True)
parser.add_argument("-nview", type=int, help="number of views", default=6)
parser.add_argument("-num_class", type=int, help="number of class(positive, negative)", default=2)
parser.add_argument("-save_folder", type=str, help='path of saving result', default='./result')
parser.add_argument('-device', default=3, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('-project', default=ROOT /  'train', help='dir name')
parser.add_argument("-lr", type=float, help="learning rate", default=5e-5)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.0)
parser.add_argument("-MVCNN_output", type=int, help="MVCNN's output dimension", default=256)
parser.add_argument("-MLP_input", type=int, help="MLP's input dimension", default=12)
parser.add_argument("-MLP_output", type=int, help="MLP's output dimension", default=16)
parser.add_argument("-CLF_hnode", type=int, help="Classifer's hidden node dimension", default=64)


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path

def main():
    wandb.init()
    args = parser.parse_args()
    print(torch.cuda.is_available())
    wandb.config.update(args)

    # createing save directory 
    save_dir = increment_path(Path(args.project) / 'exp', exist_ok=False)
    # save_dir.mkdir(parents=True, exist_ok=True)

    #
    MLP_hidnode = [16, 32, 32]
    CLF_input = args.MVCNN_output + args.MLP_output
    # Load modeld
    device = int(args.device)
    MVCNN_params = [True, args.MVCNN_output, args.cnn_name, args.nview]
    MLP_params = [args.MLP_input, args.MLP_output, MLP_hidnode]
    CLF_params = [CLF_input, args.CLF_hnode, args.num_class]
    model = FusionNet(args, MVCNN_args=MVCNN_params, MLP_args=MLP_params, CLF_args=CLF_params)  # MVCNN.args, MLP.args, CLF.args
    wandb.watch(model)

    # Dataloader
    train_dataset = FusionDataset(args.img_path, args.obj_path, args.nview, shuffle=True)   # image_path, obj_path, nviews, shuffle
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=False)

    # test_dataset = FusionDataset()
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=False)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    # Train
    Trainer = FusionmodelTrainer(model, train_loader=train_dataloader, test_loader=None, optimizer=optimizer , loss_fn=nn.CrossEntropyLoss(), save_dir=save_dir, num_views=6, wandb=wandb, device=device)
    Trainer.train(100)
    # Test(validation)


if __name__ == "__main__":
    main()