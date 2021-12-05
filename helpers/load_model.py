import torch
import torch.utils.data

from utils import TokenLabelConverter
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

    return model
