# components/framework/models/utils/resize.py
import torch.nn.functional as F

def resize(
    input,
    size=None,
    scale_factor=None,
    mode='bilinear',
    align_corners=False,
    warning=False
):
    """
    Resize tensor (giống mmseg resize)
    """
    if size is not None and scale_factor is not None:
        raise ValueError('Only one of size or scale_factor should be defined.')

    return F.interpolate(
        input,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners
    )

__all__ = ['resize']
