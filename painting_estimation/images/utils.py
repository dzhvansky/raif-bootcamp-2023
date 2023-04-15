import io
import typing

import cv2
import numpy as np
from PIL import Image


def cv2_image_from_byte_io(byte_io: io.BytesIO | bytes) -> np.ndarray:
    pil_image: Image = Image.open(byte_io).convert("RGB")
    return np.asarray(pil_image, dtype=np.uint8)


def cv2_image_to_bytes(image: np.ndarray) -> bytes:
    bytes_io = io.BytesIO()
    Image.fromarray(image).save(bytes_io, format='PNG')
    return bytes_io.getvalue()


class ImgSize(typing.NamedTuple):
    width: int
    height: int


def image_size(img: np.ndarray) -> ImgSize:
    height, width, *_ = img.shape
    return ImgSize(width=width, height=height)


def build_circle_shape(image: np.ndarray) -> np.ndarray:
    img_size: ImgSize = image_size(image)
    min_dim: int = min(img_size.width, img_size.height)

    black_img = np.zeros_like(image)
    center = (round(img_size.width / 2), round(img_size.height / 2))

    cv2.circle(black_img, center, min_dim // 2, 255)
    circle_contour = np.transpose(np.where(black_img == 255))

    return circle_contour[:, :2][:, [1, 0]].reshape(-1, 1, 2)
