from .data_simmim_pt import build_loader_simmim
from .data_simmim_ft import build_loader_finetune


def build_loader(config, is_pretrain=False):
    if is_pretrain:
        return build_loader_simmim(config)
    else:
        return build_loader_finetune(config)