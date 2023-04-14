import typing

import numpy as np


class ImgSize(typing.NamedTuple):
    width: int
    height: int


def image_size(img: np.ndarray) -> ImgSize:
    height, width, *_ = img.shape
    return ImgSize(width=width, height=height)
