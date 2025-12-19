import torch.nn as nn


def build_norm_layer(cfg, num_features, postfix=''):
    """
    Args:
        cfg (dict): Ví dụ:
            dict(type='BN')
            dict(type='GN', num_groups=32)
            dict(type='LN')
        num_features (int): số channel
        postfix (int | str): để đặt tên layer giống mmcv

    Returns:
        (name, layer)
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg = cfg.copy()

    layer_type = cfg.pop('type')
    name = layer_type + str(postfix)

    if layer_type == 'BN':
        layer = nn.BatchNorm2d(num_features, **cfg)

    elif layer_type == 'SyncBN':
        layer = nn.SyncBatchNorm(num_features, **cfg)

    elif layer_type == 'GN':
        num_groups = cfg.pop('num_groups', 32)
        layer = nn.GroupNorm(num_groups, num_features, **cfg)

    elif layer_type == 'LN':
        # mmcv LN cho CNN → normalize theo channel
        layer = nn.LayerNorm(num_features, **cfg)

    elif layer_type == 'IN':
        layer = nn.InstanceNorm2d(num_features, **cfg)

    else:
        raise KeyError(f'Unsupported norm type "{layer_type}"')

    return name, layer
