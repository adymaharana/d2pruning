# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import dweionervnm

import argparse
import os
import sys
import time
from logging import getLogger

import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

logger = getLogger()


parser = argparse.ArgumentParser(description="Evaluate models: Linear classification on ImageNet")

#########################
#### main parameters ####
#########################
parser.add_argument("--dump_path", type=str, default=".",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=31, help="seed")
parser.add_argument("--data_path", type=str, default="/path/to/imagenet",
                    help="path to dataset repository")
parser.add_argument("--workers", default=8, type=int,
                    help="number of data loading workers")

#########################
#### model parameters ###
#########################
parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained weights")

#########################
#### dist parameters ###
#########################
parser.add_argument("--dist_url", default="env://", type=str,
                    help="url used to set up distributed training")
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")


def main():
    global args, best_acc
    args = parser.parse_args()

    if args.arch == 'swav':
        # build model
        model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
        modules = list(model.children())[:-1]
        model = nn.Sequential(*modules)
    elif args.arch == 'resnet34':
        import torchvision.models as models
        model = models.resnet34(pretrained=True)
        modules = list(model.children())[:-1]
        model = nn.Sequential(*modules)
    elif args.arch == 'dinov2':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        # modules = list(model.children())[:-1]
        # model = nn.Sequential(*modules)
        # print(modules)
    else:
        raise ValueError

    # model to gpu
    model = model.cuda()
    model.eval()

    # build data
    train_dataset = datasets.ImageFolder(os.path.join(args.data_path, "train"))

    if args.arch == 'dinov2':
        tr_normalize = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    else:
        tr_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]
        )

    if args.arch == 'dinov2':
        train_dataset.transform = transforms.Compose([
            transforms.ToTensor(),
            tr_normalize,
        ])
    else:
        train_dataset.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            tr_normalize,
        ])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=512,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False
    )
    logger.info("Building data done")

    all_embeddings = []
    counter = 0
    for iter_epoch, (inp, target) in tqdm(enumerate(train_loader)):
        # measure data loading time

        # move to gpu
        inp = inp.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            output = model(inp).detach().cpu().squeeze().numpy()
            all_embeddings.append(output)

        # verbose
        if args.rank == 0 and iter_epoch % 100 == 0 and iter_epoch > 0:
            np.save(args.dump_path.replace('.npy', '_%s.npy' % counter), np.concatenate(all_embeddings, axis=0))
            all_embeddings = []
            counter += 1

    np.save(args.dump_path, np.concatenate(all_embeddings, axis=0))

if __name__ == "__main__":
    main()