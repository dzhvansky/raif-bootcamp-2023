import io
import math
import typing

import cv2
import numpy as np
from PIL import Image

from painting_estimation.images.utils import ImgSize, image_size


def cv2_image_from_byte_io(byte_io: io.BytesIO) -> np.ndarray:
    pil_image: Image = Image.open(byte_io).convert("RGB")
    return np.asarray(pil_image, dtype=np.uint8)


def cv2_image_to_bytes(image: np.ndarray) -> bytes:
    bytes_io = io.BytesIO()
    Image.fromarray(image).save(bytes_io, format='PNG')
    return bytes_io.getvalue()


class ImagePreprocessor:
    def __init__(
        self,
        target_size: ImgSize,
        target_dim_order: tuple[int, int, int],
        target_dtype: type[np.int_] | type[np.float_],
        interpolation: int = cv2.INTER_LINEAR,
        to_bgr: bool = False,
        extra_batch_dim: typing.Optional[int] = 0,
        normalize: bool = False,
        means: typing.Optional[tuple[float, float, float]] = None,
        stds: typing.Optional[tuple[float, float, float]] = None,
        initial_size_before_crop: ImgSize | None = None,
    ):
        self.target_size = target_size
        self.target_dim_order = target_dim_order
        self.target_dtype = target_dtype
        self.interpolation = interpolation
        self.to_bgr = to_bgr
        self.extra_batch_dim = extra_batch_dim
        self.normalize = normalize
        self.means = means
        self.stds = stds
        self.initial_size_before_crop = initial_size_before_crop

    @staticmethod
    def _central_crop(image: np.ndarray, *, target_size: ImgSize) -> np.ndarray:
        img_size: ImgSize = image_size(image)
        cx: int
        cy: int
        cx, cy = img_size.width // 2, img_size.height // 2
        height_start: int = cy - target_size.height // 2
        height_end: int = cy + math.ceil(target_size.height / 2)
        width_start: int = cx - target_size.width // 2
        width_end: int = cx + math.ceil(target_size.width / 2)
        return image[height_start:height_end, width_start:width_end]

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if self.to_bgr:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.initial_size_before_crop is not None:
            size: ImgSize = self.initial_size_before_crop
        else:
            size = self.target_size
        normalized_img = cv2.resize(image, dsize=(size.width, size.height), interpolation=self.interpolation).astype(
            self.target_dtype
        )
        if self.initial_size_before_crop is not None:
            normalized_img = self._central_crop(normalized_img, target_size=self.target_size)

        if self.normalize:
            normalized_img /= 255.0
        if self.means is not None:
            normalized_img = normalized_img - np.asarray(self.means, dtype=self.target_dtype)
        if self.stds is not None:
            normalized_img = normalized_img / np.asarray(self.stds, dtype=self.target_dtype)

        prepared_img = normalized_img.transpose(self.target_dim_order)

        if self.extra_batch_dim is not None:
            prepared_img = np.expand_dims(prepared_img, axis=self.extra_batch_dim)

        return prepared_img
