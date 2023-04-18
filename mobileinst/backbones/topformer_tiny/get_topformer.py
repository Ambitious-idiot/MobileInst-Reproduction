import mmcv
import torch
from .mmseg.models.backbones import Topformer
import os


def get_topformer(device, pretrained=False):
    config=os.path.join(os.path.dirname(__file__), 'local_configs', 'topformer_tiny_448x448_160k_2x8_ade20k.py')
    cfg = mmcv.Config.fromfile(config)
    cfg.model.train_cfg = None
    config = cfg.model['backbone']
    config.pop('type')
    model = Topformer(**config)
    if pretrained:
        pretrained_dict = torch.load(os.path.join(os.path.dirname(__file__), 'checkpoints',
                                                  'TopFormer-T_448x448_2x8_160k-32.5.pth'))
        model_dict = model.state_dict()
        backbone_dict = {k: v for k, v in pretrained_dict.items() if k.startswith('backbone')}
        model_dict.update(backbone_dict)
        model.load_state_dict(model_dict)

    return model.to(device)
