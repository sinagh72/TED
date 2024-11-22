import os
import numpy as np
import torch
import pytorch_lightning as pl
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.set_float32_matmul_precision('medium')
