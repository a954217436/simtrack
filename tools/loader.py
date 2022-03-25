import os
import numpy as np
import torch
import torch.multiprocessing as mp


from det3d.datasets import build_dataset
from det3d.models import build_detector
from det3d.torchie import Config

import time 
from det3d.torchie.parallel import collate_kitti
from torch.utils.data import DataLoader


import argparse
from tqdm import tqdm
import pickle as pkl
from multiprocessing import Pool


BATCH_SIZE=16
workers = 8

cfg = Config.fromfile("./experiments/pointpillars/configs/waymo_pp_centernet_tracking.py")

print(cfg.data.val)

dataset = build_dataset(cfg.data.val)

data_loader = DataLoader(
    dataset,batch_size=BATCH_SIZE,
    sampler=None,
    shuffle=False,
    num_workers=workers,
    collate_fn=collate_kitti,
    pin_memory=True,
)

for idx, data_batch in tqdm(enumerate(data_loader)):
    print(data_batch.keys())


