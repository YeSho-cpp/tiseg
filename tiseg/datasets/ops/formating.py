from collections.abc import Sequence

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


def format_img(img):
    if len(img.shape) < 3:
        img = np.expand_dims(img, -1)
    img = np.ascontiguousarray(img.transpose(2, 0, 1))
    if img.dtype is not np.float32:
        img = img.astype(np.float32)

    img = DC(to_tensor(img), stack=True)

    return img


def format_seg(seg):
    # segmentation gt convert to long
    seg = DC(to_tensor(seg[None, ...].astype(np.int64)), stack=True)

    return seg


def format_reg(reg):
    # regression gt convert to float
    reg = DC(to_tensor(reg[None, ...].astype(np.float32)), stack=True)

    return reg


def format_info(info):
    info = DC(info, cpu_only=True)

    return info
