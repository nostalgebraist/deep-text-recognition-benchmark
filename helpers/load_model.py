import os
import time
import string
import argparse
import re
import validators

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from nltk.metrics.distance import edit_distance

from utils import CTCLabelConverter, AttnLabelConverter, Averager, TokenLabelConverter
from dataset import hierarchical_dataset, AlignCollate
from model import Model
from helpers.args_hack import get_args


def load_model(device=None, **kwargs):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    opt = get_args(**kwargs)

    converter = TokenLabelConverter(opt)
    model = Model(opt)

    print('loading pretrained model from %s' % opt.saved_model)

    sd = torch.hub.load_state_dict_from_url(opt.saved_model, progress=True, map_location=device)
    sd2 = {k.partition('module.')[2]: v for k, v in sd.items()}

    model.load_state_dict(sd2)
    model.to(device)
