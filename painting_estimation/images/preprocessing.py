import io
import typing

import cv2
import numpy as np
from PIL import Image


class ImgSize(typing.NamedTuple):
    width: int
    height: int


def image_size(img: np.ndarray) -> ImgSize:
    height, width, *_ = img.shape
    return ImgSize(width=width, height=height)


def cv2_image_from_byte_io(byte_io: io.BytesIO) -> np.ndarray:
    pil_image: Image = Image.open(byte_io).convert("RGB")
    return np.asarray(pil_image, dtype=np.uint8)


class ImagePreprocessor:
    def __init__(
        self,
        target_size: ImgSize,
        target_dim_order: tuple[int, int, int],
        target_dtype: type[np.int_] | type[np.float_],
        interpolation: int = cv2.INTER_LINEAR,
        to_bgr: bool = False,
        extra_batch_dim: typing.Optional[int] = 0,
        means: typing.Optional[tuple[float, float, float]] = None,
        stds: typing.Optional[tuple[float, float, float]] = None,
    ):
        self.target_size = target_size
        self.target_dim_order = target_dim_order
        self.target_dtype = target_dtype
        self.interpolation = interpolation
        self.to_bgr = to_bgr
        self.extra_batch_dim = extra_batch_dim
        self.means = means
        self.stds = stds

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if self.to_bgr:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        normalized_img = cv2.resize(
            image, dsize=(self.target_size.width, self.target_size.height), interpolation=self.interpolation
        ).astype(self.target_dtype)

        if self.means is not None:
            normalized_img = normalized_img - np.asarray(self.means, dtype=self.target_dtype)
        if self.stds is not None:
            normalized_img = normalized_img / np.asarray(self.stds, dtype=self.target_dtype)

        prepared_img = normalized_img.transpose(self.target_dim_order)

        if self.extra_batch_dim is not None:
            prepared_img = np.expand_dims(prepared_img, axis=self.extra_batch_dim)

        return prepared_img
